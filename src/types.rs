//! Agent message shapes.
//!
//! `AgentMessage` is the canonical typed conversation transcript. Apps that
//! need richer shapes either extend the `Custom` variant (kind-tagged JSON
//! payload) or wrap the entire enum in their own outer enum. The loop never
//! peeks into `Custom` — it's pass-through context.
//!
//! The discriminator lives on the role tag, content is typed, and the loop
//! avoids `Value` walking via field-name strings.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::SystemTime;

use crate::tool::ToolCall;

/// One message in the conversation transcript.
///
/// Discriminated by `role`. Each variant carries its own payload shape;
/// the loop pattern-matches, never field-walks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum AgentMessage {
    /// System prompt. Typically only one, at the head of the transcript.
    System {
        content: String,
        #[serde(default = "default_timestamp", skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// User input. May be a single text block or rich blocks (text + images).
    User {
        content: UserContent,
        #[serde(default = "default_timestamp", skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Model output. Carries text, thinking blocks, and tool calls.
    Assistant {
        content: AssistantContent,
        stop_reason: StopReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        #[serde(default = "default_timestamp", skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
        /// Provider-reported token accounting for the call that produced
        /// this message. Populated by streaming transports that request
        /// `stream_options.include_usage`; consumed by cost/billing
        /// observers (e.g. eval matrix). `None` when the transport
        /// didn't surface usage data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    /// Output of a tool call. Always paired with a prior assistant message
    /// that contains the corresponding `ToolCall` block.
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: ToolResultContent,
        #[serde(default)]
        is_error: bool,
        /// Tool-side prose summary — the row-caption sentence the UI
        /// renders ("Ran `ls -la`.", "Wrote `index.html` (4 KB).",
        /// "Searched: `rust async` — 8 results."). The loop fills this
        /// from `ToolResult::narration` when the typed result is
        /// appended to history; tools may set it deterministically
        /// from their own structured signals. Optional for
        /// backward-compatibility with persisted histories that
        /// pre-date this field.
        ///
        /// `working_memory_anchor` and other history-aware plugins
        /// consume this in preference to walking the content blocks
        /// for a preview; the model's first-line peek of a densified
        /// shell result is opaque metadata, while narration carries
        /// the actual prose every other surface already shows.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        narration: Option<String>,
        /// Host-side structured payload carried from the tool's
        /// `ToolResult::details`. Stripped from provider wire formats
        /// (the model sees `content` only) but preserved into history
        /// so host-side plugins — delivery gates, artifact dispatchers,
        /// UI projectors — can read structured fields without
        /// text-grepping. Typed producers (`create_slides`,
        /// `create_website`, `publish`) put canonical artifact metadata
        /// here (`html_url`, `artifacts: [...]`, …). `None` when the
        /// tool returned no structured payload, or for messages
        /// deserialized from histories persisted before this field
        /// existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(default = "default_timestamp", skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Escape hatch for app-specific message types (UI notifications, hidden
    /// runtime feedback, replay markers). The loop ignores these for tool
    /// dispatch but apps can route them through plugins or the event sink.
    Custom {
        kind: String,
        #[serde(default)]
        payload: Value,
        #[serde(default = "default_timestamp", skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
}

fn default_timestamp() -> Option<u64> {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// Provider-reported token accounting for one LLM call.
///
/// All counts are in tokens; field names mirror the OpenAI-shape
/// `usage` block (input/output) plus the cache-related fields the
/// OpenRouter and Anthropic streams expose. Cost aggregators
/// (`evals/rust/src/cost.rs`) read `input_tokens`,
/// `output_tokens`, `cache_creation_input_tokens`,
/// `cache_read_input_tokens` directly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: i64,
    #[serde(default)]
    pub output_tokens: i64,
    #[serde(default)]
    pub cache_creation_input_tokens: i64,
    #[serde(default)]
    pub cache_read_input_tokens: i64,
}

/// Why an assistant turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model emitted a natural end-of-turn (no tool calls).
    EndTurn,
    /// Model emitted one or more tool calls; loop will dispatch and continue.
    ToolUse,
    /// Provider hit max output tokens.
    MaxTokens,
    /// Provider raised an error during streaming.
    Error,
    /// Caller cancelled via the abort signal.
    Aborted,
    /// Other / provider-specific stop. Use the model's own value.
    Other,
}

/// User-message content. Plain text is the common case; the block form
/// supports images, attachments, and other multimodal inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<UserBlock>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserBlock {
    Text(TextContent),
    Image(ImageContent),
}

/// Assistant-message content. Carries text, hidden reasoning blocks, and
/// tool call requests in source order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssistantContent {
    pub blocks: Vec<AssistantBlock>,
}

impl AssistantContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![AssistantBlock::Text(TextContent { text: text.into() })],
        }
    }

    pub fn with_tool_calls(text: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        let mut blocks = Vec::new();
        if let Some(t) = text.filter(|s| !s.trim().is_empty()) {
            blocks.push(AssistantBlock::Text(TextContent { text: t }));
        }
        for call in tool_calls {
            blocks.push(AssistantBlock::ToolCall(call));
        }
        Self { blocks }
    }

    /// Concatenate all text blocks into a single string.
    pub fn plain_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Return all tool call blocks in source order.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    pub fn thinking_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Thinking(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn reasoning_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Reasoning(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn reasoning_details_values(&self) -> Vec<Value> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::ReasoningDetails(d) => Some(d.details.as_slice()),
                _ => None,
            })
            .flatten()
            .cloned()
            .collect()
    }
}

/// Blocks an assistant message can carry.
///
/// ## Channel separation contract
///
/// `Thinking` and `Reasoning` are two **independent** channels and the
/// loop must never mix them:
///
/// - [`Thinking`](AssistantBlock::Thinking) is **prompt-elicited
///   tag-text**. The model wraps reasoning inside
///   `<thought>...</thought>` (or one of the synonym tags handled by
///   [`crate::ThinkingTagStreamFilter`]) inside its visible text
///   stream. The bridge parses those tags out of the visible-text
///   channel and stores the captured content here. On the next
///   provider request it is **rewoven into the `content` field as a
///   `<thought>...</thought>` tag** — never as the wire `reasoning`
///   field.
///
/// - [`Reasoning`](AssistantBlock::Reasoning) and
///   [`ReasoningDetails`](AssistantBlock::ReasoningDetails) are
///   **provider-native reasoning**. They arrive on a dedicated
///   sideband (`delta.reasoning` / `delta.reasoning_details` on the
///   OpenRouter wire) and represent the provider's own
///   chain-of-thought tokens. On the next provider request they are
///   replayed verbatim through the typed `reasoning` /
///   `reasoning_details` fields — never wrapped in a
///   `<thought>...</thought>` tag in the `content` field.
///
/// The two processes are not interchangeable: tag-elicited scratch is
/// the model writing into its visible output by convention, and the
/// loop strips it before the user sees anything. Provider-native
/// reasoning is the upstream API delivering structured thinking
/// alongside the message. Conflating them risks (a) shipping the
/// model's hidden scratch as if it were native reasoning (some
/// providers reject unknown content there or refuse to bill it as
/// cached input) and (b) leaking native reasoning into visible text
/// by way of `<thought>` rewrap (would round-trip native tokens
/// through the visible channel and double-count them).
///
/// The invariants are pinned by tests in
/// `openrouter_request_tests.rs::channel_separation_invariants` and
/// `openrouter_stream::tests::stream_chunk_routing_invariants`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantBlock {
    Text(TextContent),
    /// Prompt-elicited hidden scratchpad. Captured from
    /// `<thought>...</thought>` tags the model writes inside its
    /// visible text stream; rewoven into the wire `content` field as
    /// `<thought>...</thought>` on the next request. **Never** flows
    /// into the wire `reasoning` field — see the type-level docs for
    /// the channel-separation contract.
    Thinking(TextContent),
    /// Provider-native reasoning (xAI Grok, OpenAI o-series, Anthropic
    /// native thinking). Arrives on the dedicated `delta.reasoning`
    /// sideband and replayed via the wire `reasoning` field. **Never**
    /// wrapped in a `<thought>...</thought>` tag inside the `content`
    /// field — see the type-level docs.
    Reasoning(TextContent),
    /// Native provider reasoning detail blocks (xAI's
    /// `reasoning.encrypted` envelopes, etc.). Replayed unmodified on
    /// tool-continuation requests for reasoning models that rely on
    /// signed/encrypted thinking continuity. Same channel contract as
    /// [`Reasoning`](AssistantBlock::Reasoning).
    ReasoningDetails(ReasoningDetailsContent),
    /// Tool call request. Loop dispatches via the registry.
    ToolCall(ToolCall),
}

/// Persistent envelope for provider-native reasoning items on an
/// assistant turn. The wire shape is `Vec<Value>` matching
/// OpenRouter's `reasoning_details[]` schema (the broadest typed
/// surface across providers); `as_items` lifts it to typed
/// [`crate::reasoning::ReasoningItem`]s for codec operations and `from_items` projects
/// typed items back. The `details` field stays the source of truth so
/// persisted trajectories round-trip byte-exact, even when a future
/// provider sends shapes the typed enum doesn't yet recognize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningDetailsContent {
    pub details: Vec<Value>,
}

impl ReasoningDetailsContent {
    pub fn new(details: Vec<Value>) -> Self {
        Self { details }
    }

    /// Lift the stored `details` array into typed
    /// [`crate::reasoning::ReasoningItem`]s. Items that don't match a known variant
    /// are preserved in `details` but elided from the typed view —
    /// so consumers iterating typed items never see corrupt data
    /// while replay-via-`details` still ships the original bytes.
    pub fn as_items(&self) -> Vec<crate::reasoning::ReasoningItem> {
        self.details
            .iter()
            .filter_map(crate::reasoning::ReasoningItem::from_openrouter_value)
            .collect()
    }

    /// Build from typed items. Used when a codec produces typed
    /// items from a non-OpenRouter provider response.
    pub fn from_items(items: &[crate::reasoning::ReasoningItem]) -> Self {
        Self {
            details: items
                .iter()
                .map(crate::reasoning::ReasoningItem::to_openrouter_value)
                .collect(),
        }
    }

    /// True iff any item carries a signed/encrypted payload that a
    /// strict-replay provider would reject if missing on next turn.
    pub fn has_signed_payload(&self) -> bool {
        self.as_items()
            .iter()
            .any(crate::reasoning::ReasoningItem::carries_signed_payload)
    }
}

/// Tool result content. Multiple blocks support image-returning tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolResultContent {
    pub blocks: Vec<ToolResultBlock>,
}

impl ToolResultContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ToolResultBlock::Text(TextContent { text: text.into() })],
        }
    }

    pub fn plain_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ToolResultBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text(TextContent),
    Image(ImageContent),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageContent {
    /// Either a data: URL or an external URL the provider can fetch.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alt: Option<String>,
}

/// Identity of one agent run.
///
/// Threaded through child spawns so the loop, its plugins, and any
/// trajectory sink can answer "who am I, who is my parent, how deep am
/// I, what conversation, when do I expire" without consulting a
/// side-channel. The fields are typed at the same level as
/// [`AgentMessage`] — every run has identity, full stop. Today's bridge
/// scatters these across `LoopConfig.conversation_id`,
/// `RunnerJob.depth`, `ChildScope.parent_conversation_id`, and
/// `parent_deadline`; `RunIdentity` is the merge target.
///
/// Identity is serializable so trajectory writers can pin every event
/// to its run without inventing a parallel key store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunIdentity {
    /// Stable identifier for this run. UUIDv4 by default; callers may
    /// supply their own value (e.g. to keep run ids aligned with an
    /// external trace system).
    pub run_id: String,
    /// `Some(parent.run_id)` when this run was spawned by another run;
    /// `None` for top-level runs initiated by a user-facing entry
    /// point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
    /// 0 for top-level runs; +1 per nested spawn.
    #[serde(default)]
    pub depth: usize,
    /// Conversation this run belongs to (when the host runtime has
    /// one). `None` for tests and isolated runs that don't carry
    /// conversation identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    /// Wall-clock deadline as milliseconds since the UNIX epoch. The
    /// loop does not enforce this directly — plugins that care
    /// (wall-clock steering, soft-cancel) read it. `None` means no
    /// parent-imposed deadline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_unix_ms: Option<u64>,
}

impl RunIdentity {
    /// Construct a top-level identity. Generates a fresh UUIDv4 run id.
    pub fn root() -> Self {
        Self {
            run_id: uuid::Uuid::new_v4().to_string(),
            parent_run_id: None,
            depth: 0,
            conversation_id: None,
            deadline_unix_ms: None,
        }
    }

    /// Construct a child identity from a parent. Inherits
    /// `conversation_id` and `deadline_unix_ms`, bumps `depth`, sets
    /// `parent_run_id`, and generates a fresh `run_id`.
    pub fn child_of(parent: &Self) -> Self {
        Self {
            run_id: uuid::Uuid::new_v4().to_string(),
            parent_run_id: Some(parent.run_id.clone()),
            depth: parent.depth + 1,
            conversation_id: parent.conversation_id.clone(),
            deadline_unix_ms: parent.deadline_unix_ms,
        }
    }

    pub fn with_run_id(mut self, id: impl Into<String>) -> Self {
        self.run_id = id.into();
        self
    }

    pub fn with_conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    pub fn with_deadline_unix_ms(mut self, ms: u64) -> Self {
        self.deadline_unix_ms = Some(ms);
        self
    }
}

/// Snapshot of agent state passed into the loop.
///
/// Carries the system prompt, the current transcript, and an optional
/// [`RunIdentity`]. Plain data — the loop builds an internal mutable
/// copy and returns the new tail.
///
/// `identity` is optional for backward compatibility with callers that
/// don't yet thread one through. When `None`, the loop treats the run
/// as an anonymous root; plugins that key on identity see `None` and
/// degrade gracefully.
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub identity: Option<RunIdentity>,
}

impl AgentContext {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            identity: None,
        }
    }

    pub fn with_messages(mut self, messages: Vec<AgentMessage>) -> Self {
        self.messages = messages;
        self
    }

    /// Attach a [`RunIdentity`] to this context. Use
    /// [`RunIdentity::root`] for top-level runs and
    /// [`RunIdentity::child_of`] for spawned children.
    pub fn with_identity(mut self, identity: RunIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    /// Convenience: produce a child `AgentContext` for a spawned run.
    /// Returns a fresh context with the supplied `system_prompt`, no
    /// messages, and a child identity derived from this context's
    /// identity (or a fresh root if this context has none).
    pub fn spawn_child(&self, system_prompt: impl Into<String>) -> Self {
        let parent_identity = self.identity.clone().unwrap_or_else(RunIdentity::root);
        Self {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            identity: Some(RunIdentity::child_of(&parent_identity)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_text_round_trip() {
        let msg = AgentMessage::User {
            content: UserContent::Text("hello".into()),
            timestamp: Some(0),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn assistant_with_tool_call_blocks() {
        let content = AssistantContent::with_tool_calls(
            Some("calling…".into()),
            vec![ToolCall {
                id: "call_1".into(),
                name: "shell".into(),
                arguments: serde_json::json!({"cmd": "ls"}),
            }],
        );
        assert_eq!(content.tool_calls().len(), 1);
        assert_eq!(content.plain_text(), "calling…");
    }

    #[test]
    fn custom_message_passthrough() {
        let msg = AgentMessage::Custom {
            kind: "ui_notification".into(),
            payload: serde_json::json!({"text": "build started"}),
            timestamp: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "custom");
        assert_eq!(json["kind"], "ui_notification");
    }
}
