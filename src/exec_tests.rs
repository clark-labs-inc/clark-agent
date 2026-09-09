use super::*;
use crate::ToolResultBlock;
use std::sync::Arc;

struct LimitTool {
    name: &'static str,
    counts: bool,
    vote_counts: bool,
}

#[async_trait::async_trait]
impl AgentTool for LimitTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "test tool"
    }

    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }

    fn counts_toward_tool_call_limit(&self) -> bool {
        self.counts
    }

    fn counts_toward_termination_vote(&self) -> bool {
        self.vote_counts
    }

    async fn execute(
        &self,
        _call_id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: mpsc::UnboundedSender<ToolResult>,
    ) -> Result<ToolResult, ToolError> {
        unreachable!("split tests do not execute tools")
    }
}

fn registry() -> crate::tool::ToolRegistry {
    // Same registry the call-limit tests use plus the
    // termination-vote opt-out: `message_info` is advisory, the
    // other tools count.
    crate::tool::ToolRegistry::new()
        .with(Arc::new(LimitTool {
            name: "message_info",
            counts: false,
            vote_counts: false,
        }))
        .with(Arc::new(LimitTool {
            name: "browser_navigate",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "browser_capture",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "browser_inspect",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "shell",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "message_result",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "message_ask",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "web_search",
            counts: true,
            vote_counts: true,
        }))
        .with(Arc::new(LimitTool {
            name: "file_read",
            counts: true,
            vote_counts: true,
        }))
}

fn call(name: &str) -> ToolCall {
    ToolCall {
        id: format!("tc-{name}"),
        name: name.to_string(),
        arguments: Value::Null,
    }
}

fn names(calls: &[ToolCall]) -> Vec<&str> {
    calls.iter().map(|call| call.name.as_str()).collect()
}

#[test]
fn progress_only_tools_do_not_starve_first_work_tool() {
    let registry = registry();
    let (executable, unexecuted, max) = split_tool_calls_for_execution(
        vec![call("message_info"), call("browser_navigate")],
        &registry,
        Some(1),
    );

    assert_eq!(max, Some(1));
    assert_eq!(names(&executable), vec!["message_info", "browser_navigate"]);
    assert!(unexecuted.is_empty());
}

#[test]
fn extra_limit_counted_tools_still_get_synthetic_errors() {
    let registry = registry();
    let (executable, unexecuted, max) = split_tool_calls_for_execution(
        vec![call("message_info"), call("shell"), call("message_result")],
        &registry,
        Some(1),
    );

    assert_eq!(max, Some(1));
    assert_eq!(names(&executable), vec!["message_info", "shell"]);
    assert_eq!(names(&unexecuted), vec!["message_result"]);
}

#[test]
fn read_search_and_browser_calls_share_the_work_budget() {
    let registry = registry();
    for name in [
        "file_read",
        "web_search",
        "browser_navigate",
        "browser_capture",
        "browser_inspect",
    ] {
        let (executable, unexecuted, max) = split_tool_calls_for_execution(
            vec![call(name), call(name), call("shell")],
            &registry,
            Some(2),
        );
        assert_eq!(max, Some(2));
        assert_eq!(names(&executable), vec![name, name]);
        assert_eq!(names(&unexecuted), vec!["shell"]);
        assert_eq!(count_limit_counted_tool_calls(&executable, &registry), 2);
    }
}

#[test]
fn malformed_calls_do_not_burn_the_cap_or_preempt_real_work() {
    // A call that resolves to no registered tool (unknown name, or an
    // empty/blank name from a streaming glitch) does no real work — it
    // only yields a synthetic "no such tool" error in `prepare_call`. It
    // must NOT spend the turn's slot, or it bumps a real call into the
    // unexecuted bin. Regression for a production case where, under
    // `max_tool_calls_per_turn = 1`, an empty-name call preempted the
    // model's real next call.
    let registry = registry();

    // Unknown name first, real counting tool second: both run; nothing deferred.
    let (executable, unexecuted, _) = split_tool_calls_for_execution(
        vec![call("missing"), call("shell")],
        &registry,
        Some(1),
    );
    assert_eq!(names(&executable), vec!["missing", "shell"]);
    assert!(
        unexecuted.is_empty(),
        "real work must not be preempted by an unknown name: {:?}",
        names(&unexecuted)
    );

    // Empty name first (the prod glitch shape): the real call still runs.
    let (executable, unexecuted, _) =
        split_tool_calls_for_execution(vec![call(""), call("shell")], &registry, Some(1));
    assert_eq!(names(&executable), vec!["", "shell"]);
    assert!(
        unexecuted.is_empty(),
        "empty-name glitch must not preempt real work: {:?}",
        names(&unexecuted)
    );

    // Two real counting tools: the cap still bites — the second is deferred.
    let (executable, unexecuted, _) =
        split_tool_calls_for_execution(vec![call("shell"), call("shell")], &registry, Some(1));
    assert_eq!(names(&executable), vec!["shell"]);
    assert_eq!(names(&unexecuted), vec!["shell"]);
}

#[test]
fn compute_batch_terminate_passes_when_only_advisory_siblings_dont_vote() {
    // Some models tail a terminating delivery call with a polite,
    // advisory sign-off call that opts out of the termination vote
    // (`counts_toward_termination_vote == false`). Under a strict
    // every-result-must-vote rule the trailing advisory call
    // (terminate=false) would block termination and the run would
    // grind to its iteration cap. With the advisory opt-out the
    // batch terminates on the strength of the delivery call alone.
    let registry = registry();
    let votes = [("message_result", true), ("message_info", false)];
    assert!(compute_batch_terminate(
        &registry,
        votes.iter().map(|(n, t)| (*n, *t))
    ));
}

#[test]
fn compute_batch_terminate_fails_when_any_counted_tool_did_not_vote_terminate() {
    let registry = registry();
    // `message_result` voted yes, but a real work tool (`shell`)
    // is still mid-flight or didn't vote — keep running.
    let votes = [("message_result", true), ("shell", false)];
    assert!(!compute_batch_terminate(
        &registry,
        votes.iter().map(|(n, t)| (*n, *t))
    ));
}

#[test]
fn compute_batch_terminate_returns_false_for_all_advisory_batches() {
    // An all-`message_info` batch must NEVER end the run; progress
    // notes are status, not termination, even when the model
    // emits several in a row.
    let registry = registry();
    let votes = [("message_info", false), ("message_info", false)];
    assert!(!compute_batch_terminate(
        &registry,
        votes.iter().map(|(n, t)| (*n, *t))
    ));
}

#[test]
fn compute_batch_terminate_returns_false_for_empty_batch() {
    let registry = registry();
    let votes: Vec<(&str, bool)> = Vec::new();
    assert!(!compute_batch_terminate(&registry, votes.into_iter()));
}

#[test]
fn compute_batch_terminate_treats_unknown_tools_as_counted() {
    // Unknown / unregistered tool names default to counted so a
    // stray call cannot accidentally terminate the run by being
    // silently classified as advisory.
    let registry = registry();
    // `message_result` voted yes, but an unknown tool emitted
    // `terminate=false`. Unknown counts → must not terminate.
    let votes = [("message_result", true), ("ghost_tool", false)];
    assert!(!compute_batch_terminate(
        &registry,
        votes.iter().map(|(n, t)| (*n, *t))
    ));

    // And the symmetric case: an unknown tool that voted yes,
    // alongside `message_result` voting yes → still counted, so
    // the batch terminates.
    let votes = [("message_result", true), ("ghost_tool", true)];
    assert!(compute_batch_terminate(
        &registry,
        votes.iter().map(|(n, t)| (*n, *t))
    ));
}

#[test]
fn compute_batch_terminate_passes_when_message_ask_is_only_counted_terminator() {
    // Symmetric to the message_result case: message_ask (also a
    // terminating tool) tailed by message_info still terminates.
    let registry = registry();
    let votes = [("message_ask", true), ("message_info", false)];
    assert!(compute_batch_terminate(
        &registry,
        votes.iter().map(|(n, t)| (*n, *t))
    ));
}

#[test]
fn head_tail_for_log_returns_full_text_when_short() {
    // Short payloads (≤ HEAD+TAIL) round-trip in `head` with an
    // empty `tail` so the trace line stays compact and the
    // diagnostic reader doesn't have to reconstruct the full
    // string from two halves when there's nothing to truncate.
    let (head, tail) = head_tail_for_log("hello");
    assert_eq!(head, "hello");
    assert_eq!(tail, "");
}

#[test]
fn head_tail_for_log_truncates_long_text_with_head_and_tail() {
    let payload: String = "abc".repeat(500);
    assert!(payload.len() > TOOL_RESULT_LOG_HEAD + TOOL_RESULT_LOG_TAIL);
    let (head, tail) = head_tail_for_log(&payload);
    assert_eq!(head.len(), TOOL_RESULT_LOG_HEAD);
    assert_eq!(tail.len(), TOOL_RESULT_LOG_TAIL);
    // First/last bytes must come from the original — guards
    // against a regression where the helper accidentally re-orders
    // or drops the boundary characters.
    assert!(payload.starts_with(&head));
    assert!(payload.ends_with(&tail));
}

#[test]
fn head_tail_for_log_respects_utf8_char_boundaries() {
    // Multi-byte chars must not be split mid-codepoint or the
    // tracing macro would panic (and instrumentation would crash
    // the loop). Build a payload long enough to truncate, padded
    // with multi-byte chars at both boundary regions.
    let mid = "πλάκα".repeat(50); // each char is 2 bytes
    let prefix: String = "x".repeat(150);
    let suffix: String = "y".repeat(150);
    let payload = format!("{prefix}{mid}{suffix}");
    let (head, tail) = head_tail_for_log(&payload);
    // Validity assertions: both slices are valid UTF-8 (they
    // already are since they came from `&str`), and the boundary
    // is on a char boundary in the original. Round-trip check:
    // the head must be a prefix of payload and tail a suffix.
    assert!(payload.starts_with(&head));
    assert!(payload.ends_with(&tail));
    // Head capped at HEAD bytes (last char-boundary at or before).
    assert!(head.len() <= TOOL_RESULT_LOG_HEAD);
    assert!(tail.len() <= TOOL_RESULT_LOG_TAIL + 1); // +1 for boundary slack
}

#[test]
fn unexecuted_message_mentions_progress_only_calls_when_present() {
    let result = unexecuted_tool_call_result(3, 2, 1);
    let text = match result.content.first() {
        Some(ToolResultBlock::Text(text)) => text.text.as_str(),
        _ => panic!("expected text result"),
    };

    assert!(text.contains("2 limit-counted tool calls"));
    assert!(text.contains("3 tool calls total, including progress-only calls"));
    assert_eq!(
        result
            .details
            .get("limit_counted_tool_calls")
            .and_then(Value::as_u64),
        Some(2)
    );
}
