use super::*;
use crate::config::AgentBuilder;
use crate::plugin::{
    FollowUpSource, Plugin, PluginCapabilities, ToolGate, ToolGateClass, ToolGateContext,
};
use crate::stream::{ReasoningEffort, StreamFn};
use crate::types::{AssistantBlock, UserContent};
use futures::stream::{self, BoxStream};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};


fn empty_assistant_message() -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent { blocks: Vec::new() },
        stop_reason: StopReason::Other,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn text_assistant_message(text: impl Into<String>) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::text(text),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn tool_call_assistant_message(name: impl Into<String>, id: impl Into<String>) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::with_tool_calls(
            None,
            vec![crate::tool::ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: serde_json::json!({}),
            }],
        ),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

struct EmptyThenTextStream {
    calls: AtomicUsize,
    failures_before_success: usize,
}

struct ProviderRateLimitedStream {
    calls: AtomicUsize,
    stall_after_first: bool,
}

impl ProviderRateLimitedStream {
    fn immediate() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            stall_after_first: false,
        }
    }

    fn then_stall() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            stall_after_first: true,
        }
    }
}

impl EmptyThenTextStream {
    fn new(failures_before_success: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            failures_before_success,
        }
    }
}

#[derive(Default)]
struct ZeroOutputThenTextStream {
    calls: AtomicUsize,
    requests: Mutex<Vec<StreamRequest>>,
}

impl ZeroOutputThenTextStream {
    fn requests(&self) -> Vec<StreamRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[derive(Default)]
struct RepeatedTextStream {
    calls: AtomicUsize,
}

struct PlainTextThenTerminatorStream {
    calls: AtomicUsize,
    plain_text_turns: usize,
}

struct TerminalOnlyGate;
struct TerminalWithStatusGate;

/// A product protocol policy that declares several delivery/status
/// tools (beyond the configured fallback tool) as terminal, so an
/// allowlist narrowed to `{message_info, message_result}` still
/// classifies as terminal-only. The core ships none of these names;
/// they live behind the policy.
struct TestTerminalPolicy;
impl crate::protocol::ProtocolPolicy for TestTerminalPolicy {
    fn terminal_tool_names(&self) -> std::collections::HashSet<String> {
        [
            "message_info",
            "message_ask",
            "message_result",
            "terminator",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}
struct StaticAllowGate {
    name: &'static str,
    tools: &'static [&'static str],
    priority: i32,
    class: ToolGateClass,
    suppresses_advisory: bool,
}

impl Plugin for TerminalOnlyGate {
    fn name(&self) -> &'static str {
        "terminal_only_gate"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::tool_gate()
    }
}

#[async_trait::async_trait]
impl ToolGate for TerminalOnlyGate {
    async fn next_turn_tool_allowlist(
        &self,
        _ctx: ToolGateContext<'_>,
    ) -> Option<std::collections::HashSet<String>> {
        Some(["message_result".to_string()].into_iter().collect())
    }
}

impl Plugin for TerminalWithStatusGate {
    fn name(&self) -> &'static str {
        "terminal_with_status_gate"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::tool_gate()
    }
}

#[async_trait::async_trait]
impl ToolGate for TerminalWithStatusGate {
    async fn next_turn_tool_allowlist(
        &self,
        _ctx: ToolGateContext<'_>,
    ) -> Option<std::collections::HashSet<String>> {
        Some(
            ["message_info".to_string(), "message_result".to_string()]
                .into_iter()
                .collect(),
        )
    }
}

impl Plugin for StaticAllowGate {
    fn name(&self) -> &'static str {
        self.name
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::tool_gate()
    }
}

#[async_trait::async_trait]
impl ToolGate for StaticAllowGate {
    fn conflict_priority(&self) -> i32 {
        self.priority
    }

    fn tool_gate_class(&self) -> ToolGateClass {
        self.class
    }

    fn suppresses_advisory_gates(&self, _ctx: ToolGateContext<'_>) -> bool {
        self.suppresses_advisory
    }

    async fn next_turn_tool_allowlist(
        &self,
        _ctx: ToolGateContext<'_>,
    ) -> Option<std::collections::HashSet<String>> {
        Some(self.tools.iter().map(|name| (*name).to_string()).collect())
    }
}

impl PlainTextThenTerminatorStream {
    fn new(plain_text_turns: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            plain_text_turns,
        }
    }
}

#[async_trait::async_trait]
impl StreamFn for EmptyThenTextStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = empty_assistant_message();
        if call < self.failures_before_success {
            return Box::pin(stream::iter(vec![
                StreamEvent::Start {
                    partial: partial.clone(),
                },
                StreamEvent::Error {
                    partial,
                    kind: StreamErrorKind::Empty,
                    message: "empty provider response".to_string(),
                },
            ]));
        }
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done {
                message: text_assistant_message("recovered"),
            },
        ]))
    }
}

#[async_trait::async_trait]
impl StreamFn for ProviderRateLimitedStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.stall_after_first && call > 0 {
            return Box::pin(stream::pending());
        }
        let partial = empty_assistant_message();
        Box::pin(stream::iter(vec![
            StreamEvent::Start {
                partial: partial.clone(),
            },
            StreamEvent::Error {
                partial,
                kind: StreamErrorKind::ProviderRateLimited,
                message: "upstream provider shared pool is rate limited".to_string(),
            },
        ]))
    }
}

#[async_trait::async_trait]
impl StreamFn for RepeatedTextStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = empty_assistant_message();
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done {
                message: text_assistant_message(format!("plain stop {call}")),
            },
        ]))
    }
}

#[async_trait::async_trait]
impl StreamFn for PlainTextThenTerminatorStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = empty_assistant_message();
        let message = if call < self.plain_text_turns {
            text_assistant_message(format!("plain stop {call}"))
        } else if call == self.plain_text_turns {
            tool_call_assistant_message("terminator", "tc-terminator")
        } else {
            panic!("unexpected stream call after terminal turn: {call}")
        };
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done { message },
        ]))
    }
}

#[async_trait::async_trait]
impl StreamFn for ZeroOutputThenTextStream {
    async fn stream(
        &self,
        request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        self.requests.lock().unwrap().push(request);
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = empty_assistant_message();
        if call == 0 {
            return Box::pin(stream::iter(vec![
                StreamEvent::Start {
                    partial: partial.clone(),
                },
                StreamEvent::Error {
                    partial,
                    kind: StreamErrorKind::ZeroOutputTransport,
                    message: "response body decode failed before output".to_string(),
                },
            ]));
        }
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done {
                message: text_assistant_message("recovered from transport"),
            },
        ]))
    }
}

/// Overflows on the first call, then (once the history is shrunk)
/// returns text. Records each request so a test can assert what the
/// retried call actually sent.
struct OverflowThenTextStream {
    calls: AtomicUsize,
    requests: Mutex<Vec<StreamRequest>>,
    overflows_before_success: usize,
}

impl OverflowThenTextStream {
    fn new(overflows_before_success: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
            overflows_before_success,
        }
    }
}

impl Default for OverflowThenTextStream {
    fn default() -> Self {
        Self::new(1)
    }
}

#[async_trait::async_trait]
impl StreamFn for OverflowThenTextStream {
    async fn stream(
        &self,
        request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        self.requests.lock().unwrap().push(request);
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = empty_assistant_message();
        if call < self.overflows_before_success {
            return Box::pin(stream::iter(vec![
                StreamEvent::Start {
                    partial: partial.clone(),
                },
                StreamEvent::Error {
                    partial,
                    kind: StreamErrorKind::ContextOverflow,
                    message: "maximum context length exceeded".to_string(),
                },
            ]));
        }
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done {
                message: text_assistant_message("recovered after shrink"),
            },
        ]))
    }
}

/// Recovery that drops the oldest message — enough to prove the loop
/// persists each material shrink and can retry repeatedly.
struct KeepLastRecovery {
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl crate::plugin::ContextOverflowRecovery for KeepLastRecovery {
    async fn recover(
        &self,
        mut messages: Vec<AgentMessage>,
        _cx: &TransformContext<'_>,
    ) -> Vec<AgentMessage> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !messages.is_empty() {
            messages.remove(0);
        }
        messages
    }
    fn name(&self) -> &'static str {
        "keep_last_recovery"
    }
}









/// `StreamFn` that emits one assistant turn with a single
/// `terminator` tool call, then panics on subsequent invocations
/// — the test asserts the loop never re-enters the LLM.
struct TerminatorOnlyStream {
    calls: AtomicUsize,
}

struct RepeatingToolStream {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl StreamFn for RepeatingToolStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = empty_assistant_message();
        let assistant = tool_call_assistant_message("continue", format!("tc-{call}"));
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done { message: assistant },
        ]))
    }
}

struct NonTerminatingTool;

#[async_trait::async_trait]
impl crate::tool::AgentTool for NonTerminatingTool {
    fn name(&self) -> &str {
        "continue"
    }

    fn description(&self) -> &str {
        "keep working"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn execute(
        &self,
        _call_id: &str,
        _args: serde_json::Value,
        _signal: CancellationToken,
        _update: tokio::sync::mpsc::UnboundedSender<crate::tool::ToolResult>,
    ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
        Ok(crate::tool::ToolResult::text("continue"))
    }
}


impl Default for TerminatorOnlyStream {
    fn default() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl StreamFn for TerminatorOnlyStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            call, 0,
            "terminate-on-turn-1 test must NOT re-enter the LLM after a successful terminator"
        );
        let partial = empty_assistant_message();
        let assistant = AgentMessage::Assistant {
            content: AssistantContent {
                blocks: vec![AssistantBlock::ToolCall(crate::tool::ToolCall {
                    id: "tc-terminator-1".into(),
                    name: "terminator".into(),
                    arguments: serde_json::json!({}),
                })],
            },
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: None,
            usage: None,
        };
        Box::pin(stream::iter(vec![
            StreamEvent::Start { partial },
            StreamEvent::Done { message: assistant },
        ]))
    }
}

/// Tool that always votes `terminate=true`. Mirrors the contract a
/// downstream terminal/delivery tool upholds.
struct TerminatorTool;

#[async_trait::async_trait]
impl crate::tool::AgentTool for TerminatorTool {
    fn name(&self) -> &str {
        "terminator"
    }

    fn description(&self) -> &str {
        "test terminator"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn execute(
        &self,
        _call_id: &str,
        _args: serde_json::Value,
        _signal: CancellationToken,
        _update: tokio::sync::mpsc::UnboundedSender<crate::tool::ToolResult>,
    ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
        Ok(crate::tool::ToolResult {
            content: vec![crate::types::ToolResultBlock::Text(
                crate::types::TextContent {
                    text: "delivered".into(),
                },
            )],
            is_error: false,
            details: serde_json::Value::Null,
            terminate: true,
            narration: None,
        })
    }
}

/// `SteeringSource` that always returns one wrap-up message. Used
/// to prove the loop does NOT poll steering after a terminator
/// vote (otherwise this would re-enter the LLM and trip the
/// `assert_eq!(call, 0)` in `TerminatorOnlyStream`).
struct AlwaysSteer {
    polls: Arc<AtomicUsize>,
}

impl Plugin for AlwaysSteer {
    fn name(&self) -> &'static str {
        "always_steer"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            steering: true,
            ..PluginCapabilities::default()
        }
    }
}

#[async_trait::async_trait]
impl crate::plugin::SteeringSource for AlwaysSteer {
    async fn next_steering_messages(&self) -> Vec<AgentMessage> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        vec![AgentMessage::System {
            content: "wrap up now".into(),
            timestamp: None,
        }]
    }
}


/// `FollowUpSource` that always emits one nudge. Counts polls so
/// the test can prove `collect_follow_up` is NOT invoked after a
/// terminator batch.
struct AlwaysFollowUp {
    polls: Arc<AtomicUsize>,
}

impl Plugin for AlwaysFollowUp {
    fn name(&self) -> &'static str {
        "always_follow_up"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::follow_up()
    }
}

#[async_trait::async_trait]
impl FollowUpSource for AlwaysFollowUp {
    async fn next_follow_up_messages(&self) -> Vec<AgentMessage> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        vec![AgentMessage::System {
            content: "deliver something".into(),
            timestamp: None,
        }]
    }
}









struct TerminalNamedTool(&'static str);

#[async_trait::async_trait]
impl crate::tool::AgentTool for TerminalNamedTool {
    fn name(&self) -> &str {
        self.0
    }

    fn description(&self) -> &str {
        "test terminal tool"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn execute(
        &self,
        _call_id: &str,
        _args: serde_json::Value,
        _signal: CancellationToken,
        _update: tokio::sync::mpsc::UnboundedSender<crate::tool::ToolResult>,
    ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
        Ok(crate::tool::ToolResult {
            content: vec![crate::types::ToolResultBlock::Text(
                crate::types::TextContent {
                    text: "not used".into(),
                },
            )],
            is_error: false,
            details: serde_json::Value::Null,
            terminate: true,
            narration: None,
        })
    }
}

mod provider_and_loop;

mod terminal_protocol;
