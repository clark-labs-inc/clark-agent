//! End-to-end tests for the graceful turn-limit plugin.
//!
//! These tests exercise the auto-installed `GracefulTurnLimit` through
//! the full loop: a scripted stream returns canned assistant turns, the
//! loop emits steering messages between them, and we assert on the
//! typed `LoopOutcome` returned by `clark_agent::run`.

use async_trait::async_trait;
use clark_agent::{
    AgentBuilder, AgentContext, AgentMessage, AssistantBlock, AssistantContent, LoopOutcome,
    StopReason, StreamEvent, StreamFn, StreamRequest, ToolCall, UserContent,
};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::json;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Scripted stream that walks through a canned list of assistant turns
/// in order, looping on the last one if the loop drains the list. The
/// "loop on last" behavior matters for the over-budget tests where the
/// model is expected to keep tool-calling forever.
struct ScriptedStream {
    queue: Mutex<Vec<AgentMessage>>,
}

impl ScriptedStream {
    fn new(turns: Vec<AgentMessage>) -> Self {
        Self {
            queue: Mutex::new(turns),
        }
    }
}

#[async_trait]
impl StreamFn for ScriptedStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let mut q = self.queue.lock().unwrap();
        // If only one entry remains, repeat it forever — tests that drive
        // the loop until the cap fires need a steady supply of turns.
        let next = if q.len() > 1 {
            q.remove(0)
        } else {
            q.first().cloned().expect("ScriptedStream ran dry")
        };
        let events = vec![
            StreamEvent::Start {
                partial: next.clone(),
            },
            StreamEvent::Done { message: next },
        ];
        stream::iter(events).boxed()
    }
}

fn assistant_with_tool_call(id: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ToolCall(ToolCall {
                id: id.into(),
                name: "noop".into(),
                arguments: json!({}),
            })],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn assistant_end_turn(text: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::text(text),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

/// Minimal tool that always succeeds. The loop needs *some* registered
/// tool so the scripted tool-call has somewhere to land.
struct NoopTool;

#[async_trait]
impl clark_agent::AgentTool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "no-op"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: serde_json::Value,
        _signal: CancellationToken,
        _update: clark_agent::ToolUpdateSink,
    ) -> Result<clark_agent::ToolResult, clark_agent::ToolError> {
        Ok(clark_agent::ToolResult::text("ok"))
    }
}

fn user_prompt() -> Vec<AgentMessage> {
    vec![AgentMessage::User {
        content: UserContent::Text("go".into()),
        timestamp: None,
    }]
}

/// Ergonomic builder: cap + grace + scripted stream + noop tool.
fn build_config(
    max_iterations: usize,
    grace: usize,
    turns: Vec<AgentMessage>,
) -> clark_agent::LoopConfig {
    let stream = std::sync::Arc::new(ScriptedStream::new(turns));
    let registry = clark_agent::ToolRegistry::new().with(std::sync::Arc::new(NoopTool));
    AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .max_iterations(max_iterations)
        .grace_iterations(grace)
        .build()
        .expect("build")
}

// ─── Tests ──────────────────────────────────────────────────────────

/// With `grace_iterations = 0`, the soft-warning plugin is not
/// installed. A model that stops naturally well before the cap
/// produces `LoopOutcome::Done`, identical to the pre-grace behavior.
#[tokio::test]
async fn grace_zero_natural_stop_is_done() {
    let config = build_config(10, 0, vec![assistant_end_turn("done")]);

    let result = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(result.outcome, LoopOutcome::Done);
}

/// Soft limit fires; the next assistant turn the script returns has no
/// tool calls, so the loop exits naturally before the hard cap. The
/// outcome is `WrappedUp` — the model "responded" to the wrap-up
/// steer in time.
#[tokio::test]
async fn soft_limit_fires_and_model_wraps_up() {
    // Cap 8, grace 3 → soft limit at iteration 5.
    // Script: 5 tool-calling turns, then an EndTurn after the wrap-up
    // steer lands. The plugin emits its message between turn 5 and the
    // next stream call; the next assistant ends naturally.
    let mut turns: Vec<_> = (0..5)
        .map(|i| assistant_with_tool_call(&format!("c{i}")))
        .collect();
    turns.push(assistant_end_turn("wrapped up cleanly"));
    let config = build_config(8, 3, turns);

    let result = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(
        result.outcome,
        LoopOutcome::WrappedUp,
        "expected WrappedUp; got {:?}",
        result.outcome
    );

    // The wrap-up text should be present in the transcript as a system
    // message — that's how the model learned to stop. We don't pin
    // the exact string here (the message is an implementation detail
    // of the plugin) but we assert a system message landed after the
    // first assistant tool call.
    let system_msgs_after_initial = result
        .messages
        .iter()
        .skip(1) // skip the initial prompt
        .filter(|m| matches!(m, AgentMessage::System { .. }))
        .count();
    assert!(
        system_msgs_after_initial >= 1,
        "wrap-up steering message should appear in the transcript",
    );
}

/// Soft limit fires, but the model keeps tool-calling. The hard cap
/// fires; outcome is `HitMaxIterations`. The wrap-up message having
/// been emitted does not promote this to `WrappedUp`.
#[tokio::test]
async fn soft_limit_ignored_hits_max_iterations() {
    // Cap 6, grace 2 → soft at 4. Script returns tool-calls forever
    // (single-entry queue repeats per ScriptedStream's loop semantics).
    let config = build_config(6, 2, vec![assistant_with_tool_call("c0")]);

    let result = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(
        result.outcome,
        LoopOutcome::HitMaxIterations,
        "expected HitMaxIterations after grace exhausted; got {:?}",
        result.outcome
    );
    assert!(!result.outcome.is_complete());
    assert_eq!(result.outcome.label(), "hit_max_iterations");
}

/// When `grace >= max_iterations` the auto-installer skips the plugin
/// (no useful soft window). Even though the model stops naturally,
/// the outcome is `Done` — there is no plugin to flip the
/// wrap-up signal.
#[tokio::test]
async fn grace_at_or_above_cap_is_no_op() {
    // grace == max → plugin not installed.
    let config = build_config(5, 5, vec![assistant_end_turn("done")]);

    let result = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(result.outcome, LoopOutcome::Done);
}

/// Without `max_iterations` configured, no plugin is installed even
/// with a non-zero grace setting — there's no cap to be graceful
/// before. Smoke check that this combination doesn't panic and
/// returns `Done` on natural stop.
#[tokio::test]
async fn no_max_iterations_means_no_plugin() {
    let stream = std::sync::Arc::new(ScriptedStream::new(vec![assistant_end_turn("done")]));
    let registry = clark_agent::ToolRegistry::new().with(std::sync::Arc::new(NoopTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .grace_iterations(5)
        .build()
        .expect("build");

    let result = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(result.outcome, LoopOutcome::Done);
}
