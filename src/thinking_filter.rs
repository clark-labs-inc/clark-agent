//! Streaming filter that splits assistant text into a hidden
//! `<thought>` channel and visible content.
//!
//! A host typically prompts the model to begin every turn with exactly
//! one `<thought>...</thought>` block (private scratch space — the typed
//! record this filter extracts and routes off the visible stream),
//! optionally followed by a short `<narrate>...</narrate>` sentence
//! (user-visible diary text), and then the tool call(s). Emitting
//! `<thought>` first preserves the audit record even if generation is
//! cut short before any user-visible token streams. This filter sits
//! in the OpenRouter SSE path and routes content tokens to the right
//! place as they arrive:
//!
//! - Text outside any thinking tag flows through as visible text.
//! - Text inside a recognized hidden tag (`<thought>`, `<thinking>`,
//!   `<think>`, `<reasoning>`, `<reflection>`) is buffered separately
//!   and surfaced via [`ThinkingTagStreamFilter::take_completed_thought`]
//!   when the closing tag arrives.
//!
//! The filter is delta-aware: a tag may be split across SSE chunk
//! boundaries (`<thi` then `nking>` then `hidden` then `</thinking>`).
//! Ambiguous prefixes (anything starting with `<` that *could* be a
//! thinking tag) are buffered until they can be confirmed or rejected.
//!
//! Why a fresh copy in clark-agent rather than reusing
//! `clark_core::runtime_core::json_extract::ThinkingTagStreamFilter`:
//! clark-agent is the lean loop crate (no redis, no chrono-tz, no
//! sentry); pulling in clark-core would bloat its compile graph by an
//! order of magnitude. The two implementations are kept narrow enough
//! that drift is cheap to spot in review.

/// Tag names whose content is treated as hidden reasoning and
/// stripped from the visible assistant stream. Synonyms kept for
/// compatibility with diverse provider conventions; the canonical
/// Clark prompt asks for `<thought>`.
const THINKING_TAGS: &[&str] = &["think", "thinking", "thought", "reasoning", "reflection"];

/// Streaming-aware filter that suppresses content inside thinking
/// XML tags as deltas arrive token-by-token.
#[derive(Debug, Default, Clone)]
pub struct ThinkingTagStreamFilter {
    inside: bool,
    pending: String,
    thought_buffer: Option<String>,
    completed_thought: Option<String>,
}

impl ThinkingTagStreamFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a delta and return the visible portion (which may be empty).
    pub fn feed(&mut self, delta: &str) -> String {
        let mut out = String::with_capacity(delta.len());
        for ch in delta.chars() {
            self.consume_char(ch, &mut out);
        }
        out
    }

    /// End-of-stream flush. If we're outside a tag and have a pending
    /// prefix (a stray `<` that turned out not to start a tag), emit
    /// it. If we're inside an unclosed tag, drop the trailing content
    /// (mirrors [`strip_thinking_tags`]).
    pub fn flush(&mut self) -> String {
        let out = if self.inside {
            self.thought_buffer.take();
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        };
        self.pending.clear();
        self.inside = false;
        self.completed_thought.take();
        out
    }

    /// Reset between turns / bounces.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.inside = false;
        self.thought_buffer.take();
        self.completed_thought.take();
    }

    /// Take any thought block completed during the most recent
    /// [`Self::feed`] call. Returns `None` if no thought closed since the
    /// last call.
    pub fn take_completed_thought(&mut self) -> Option<String> {
        self.completed_thought.take()
    }

    fn consume_char(&mut self, ch: char, out: &mut String) {
        if self.inside {
            self.consume_inside(ch);
        } else {
            self.consume_outside(ch, out);
        }
    }

    fn consume_outside(&mut self, ch: char, out: &mut String) {
        if self.pending.is_empty() {
            if ch == '<' {
                self.pending.push(ch);
            } else {
                out.push(ch);
            }
            return;
        }
        self.pending.push(ch);
        match classify(&self.pending, /* closing = */ false) {
            TagMatch::Complete => {
                self.inside = true;
                self.pending.clear();
                self.thought_buffer = Some(String::new());
            }
            TagMatch::Possible => {}
            TagMatch::No => {
                out.push_str(&self.pending);
                self.pending.clear();
            }
        }
    }

    fn consume_inside(&mut self, ch: char) {
        if self.pending.is_empty() {
            if ch == '<' {
                self.pending.push(ch);
            } else if let Some(ref mut buf) = self.thought_buffer {
                buf.push(ch);
            }
            return;
        }
        self.pending.push(ch);
        match classify(&self.pending, /* closing = */ true) {
            TagMatch::Complete => {
                self.inside = false;
                self.completed_thought = self.thought_buffer.take();
                self.pending.clear();
            }
            TagMatch::Possible => {}
            TagMatch::No => {
                if let Some(ref mut buf) = self.thought_buffer {
                    buf.push_str(&self.pending);
                } else {
                    self.thought_buffer = Some(std::mem::take(&mut self.pending));
                }
                self.pending.clear();
            }
        }
    }
}

enum TagMatch {
    Complete,
    Possible,
    No,
}

fn classify(buf: &str, closing: bool) -> TagMatch {
    let lower = canonicalize_tag_candidate(buf);
    for tag in THINKING_TAGS {
        let full = if closing {
            format!("</{tag}>")
        } else {
            format!("<{tag}>")
        };
        if lower == full {
            return TagMatch::Complete;
        }
        if full.starts_with(&lower) {
            return TagMatch::Possible;
        }
    }
    TagMatch::No
}

fn canonicalize_tag_candidate(buf: &str) -> String {
    let mut out = String::with_capacity(buf.len());
    let mut chars = buf.chars().peekable();

    let Some(first) = chars.next() else {
        return out;
    };
    out.push(first.to_ascii_lowercase());
    if first != '<' {
        out.extend(chars.map(|ch| ch.to_ascii_lowercase()));
        return out;
    }

    while matches!(chars.peek(), Some(ch) if ch.is_ascii_whitespace()) {
        chars.next();
    }

    if matches!(chars.peek(), Some('/')) {
        out.push('/');
        chars.next();
        while matches!(chars.peek(), Some(ch) if ch.is_ascii_whitespace()) {
            chars.next();
        }
    }

    for ch in chars {
        if ch.is_ascii_whitespace() {
            continue;
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// Remove XML-like thinking blocks from a complete string. Used as a
/// safety net by [`crate::types::AssistantContent::plain_text`]'s
/// callers when serializing assistant text back to the wire on the
/// next request — defends against blocks that slipped past the
/// streaming filter (provider buffering, malformed nesting, etc.).
pub fn strip_thinking_tags(text: &str) -> String {
    let mut filter = ThinkingTagStreamFilter::new();
    let mut result = filter.feed(text);
    result.push_str(&filter.flush());
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(filter: &mut ThinkingTagStreamFilter, deltas: &[&str]) -> String {
        let mut out = String::new();
        for d in deltas {
            out.push_str(&filter.feed(d));
        }
        out.push_str(&filter.flush());
        out
    }

    #[test]
    fn passes_through_when_no_tag() {
        let mut f = ThinkingTagStreamFilter::new();
        assert_eq!(feed_all(&mut f, &["hello", " ", "world"]), "hello world");
        assert!(f.take_completed_thought().is_none());
    }

    #[test]
    fn strips_complete_thought_block_in_one_delta() {
        let mut f = ThinkingTagStreamFilter::new();
        let visible = feed_all(&mut f, &["<thought>hidden</thought>visible"]);
        assert_eq!(visible, "visible");
    }

    #[test]
    fn captures_completed_thought_text() {
        let mut f = ThinkingTagStreamFilter::new();
        let _ = f.feed("<thought>recall and frame</thought>");
        assert_eq!(
            f.take_completed_thought().as_deref(),
            Some("recall and frame")
        );
        // Second take returns None — the buffer is consumed.
        assert!(f.take_completed_thought().is_none());
    }

    #[test]
    fn handles_tag_split_across_deltas() {
        let mut f = ThinkingTagStreamFilter::new();
        let visible = feed_all(
            &mut f,
            &[
                "<thi", "nking", ">", "hidden ", "stuff", "</thi", "nking>", "out",
            ],
        );
        assert_eq!(visible, "out");
    }

    #[test]
    fn tolerant_thought_tags_are_hidden() {
        for raw in [
            "< thought >hidden</ thought >visible",
            "<\tthought>hidden</\tthought>visible",
            "<Thinking >hidden</ Thinking >visible",
        ] {
            let mut f = ThinkingTagStreamFilter::new();
            assert_eq!(feed_all(&mut f, &[raw]), "visible", "raw={raw:?}");
        }
    }

    #[test]
    fn stray_lt_is_emitted_when_not_a_tag() {
        let mut f = ThinkingTagStreamFilter::new();
        // `<` followed by non-tag chars should pass through.
        let visible = feed_all(&mut f, &["a < b > c"]);
        assert_eq!(visible, "a < b > c");
    }

    #[test]
    fn unclosed_tag_drops_trailing_content() {
        let mut f = ThinkingTagStreamFilter::new();
        let visible = feed_all(&mut f, &["before<thought>never closed"]);
        assert_eq!(visible, "before");
    }

    #[test]
    fn synonyms_recognized() {
        for tag in &["think", "thinking", "thought", "reasoning", "reflection"] {
            let mut f = ThinkingTagStreamFilter::new();
            let raw = format!("a<{tag}>x</{tag}>b");
            assert_eq!(feed_all(&mut f, &[raw.as_str()]), "ab", "tag={tag}");
        }
    }

    #[test]
    fn case_insensitive_tag_recognition() {
        let mut f = ThinkingTagStreamFilter::new();
        let visible = feed_all(&mut f, &["<Thought>hidden</Thought>visible"]);
        assert_eq!(visible, "visible");
    }

    #[test]
    fn strip_thinking_tags_removes_blocks() {
        assert_eq!(strip_thinking_tags("a<thought>x</thought>b"), "ab");
        assert_eq!(strip_thinking_tags("<reasoning>r</reasoning>tail"), "tail");
        assert_eq!(strip_thinking_tags("a< thought >x</ thought >b"), "ab");
    }

    #[test]
    fn strip_thinking_tags_handles_unclosed() {
        // Unclosed tag → everything from the open tag onward is dropped.
        assert_eq!(strip_thinking_tags("keep<thought>drop the rest"), "keep");
    }

    #[test]
    fn reset_clears_state() {
        let mut f = ThinkingTagStreamFilter::new();
        let _ = f.feed("<thought>partial");
        f.reset();
        assert_eq!(feed_all(&mut f, &["fresh"]), "fresh");
        assert!(f.take_completed_thought().is_none());
    }
}
