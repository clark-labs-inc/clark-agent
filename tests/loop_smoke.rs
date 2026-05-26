//! End-to-end smoke test: scripted stream + echo tool + before/after
//! plugins + steering. Exercises the full hook surface.

use async_trait::async_trait;
use clark_agent::stream::StreamErrorKind;
use clark_agent::{
    AfterToolCall, AfterToolDecision, AgentBuilder, AgentContext, AgentEvent, AgentMessage,
    AgentTool, AssistantBlock, AssistantContent, BeforeToolCall, BeforeToolDecision, ChannelSink,
    EventObserver, LoopError, Plugin, PluginCapabilities, StopReason, StreamError, StreamEvent,
    StreamFn, StreamRequest, ToolCall, ToolError, ToolRegistry, ToolResult, ToolResultBlock,
    ToolUpdateSink, UserContent,
};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

// ─── A scripted stream that returns canned responses by call index ────

struct ScriptedStream {
    responses: Mutex<Vec<AgentMessage>>,
}

impl ScriptedStream {
    fn new(responses: Vec<AgentMessage>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }
}

struct EventScriptedStream {
    events: Vec<StreamEvent>,
}

impl EventScriptedStream {
    fn new(events: Vec<StreamEvent>) -> Self {
        Self { events }
    }
}

#[async_trait]
impl StreamFn for EventScriptedStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        stream::iter(self.events.clone()).boxed()
    }
}

#[async_trait]
impl StreamFn for ScriptedStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let next = self.responses.lock().unwrap().remove(0);
        let events = vec![
            StreamEvent::Start {
                partial: next.clone(),
            },
            StreamEvent::Done { message: next },
        ];
        stream::iter(events).boxed()
    }
}

// ─── A tool ─────────────────────────────────────────────────────────

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echo input back"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        let text = args
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok(ToolResult::text(text))
    }
}

struct SoftFailTool;

#[async_trait]
impl AgentTool for SoftFailTool {
    fn name(&self) -> &str {
        "soft_fail"
    }
    fn description(&self) -> &str {
        "Return a recoverable tool error"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::error("recoverable failure"))
    }
}

struct ExecutionFailTool;

#[async_trait]
impl AgentTool for ExecutionFailTool {
    fn name(&self) -> &str {
        "execution_fail"
    }
    fn description(&self) -> &str {
        "Return a recoverable ToolError::Execution"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        Err(ToolError::Execution("recoverable execution failure".into()))
    }
}

struct RetainUpdateSinkTool {
    retained: Arc<Mutex<Option<ToolUpdateSink>>>,
}

#[async_trait]
impl AgentTool for RetainUpdateSinkTool {
    fn name(&self) -> &str {
        "retain_update_sink"
    }
    fn description(&self) -> &str {
        "Retain a cloned update sink after returning"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        if update.send(ToolResult::text("partial update")).is_err() {
            // The test keeps a clone below; an early receiver drop is harmless here.
        }
        *self.retained.lock().unwrap() = Some(update.clone());
        Ok(ToolResult::text("done"))
    }
}

struct FatalTool;

#[async_trait]
impl AgentTool for FatalTool {
    fn name(&self) -> &str {
        "fatal_fail"
    }
    fn description(&self) -> &str {
        "Return a fatal tool error"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        Err(ToolError::Fatal("unrecoverable failure".into()))
    }
}

struct AbortTool;

#[async_trait]
impl AgentTool for AbortTool {
    fn name(&self) -> &str {
        "abort_fail"
    }
    fn description(&self) -> &str {
        "Return an aborted tool error"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        Err(ToolError::Aborted)
    }
}

// ─── A blocking before-hook ─────────────────────────────────────────

struct BlockBananas;
impl Plugin for BlockBananas {
    fn name(&self) -> &'static str {
        "block_bananas"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::before_tool_call()
    }
}
#[async_trait]
impl BeforeToolCall for BlockBananas {
    async fn on_before_tool_call(
        &self,
        ctx: clark_agent::plugin::BeforeToolCallContext<'_>,
    ) -> BeforeToolDecision {
        let s = ctx.args.get("text").and_then(Value::as_str).unwrap_or("");
        if s.contains("banana") {
            BeforeToolDecision::block("no bananas allowed")
        } else {
            BeforeToolDecision::allow()
        }
    }
}

// ─── An after-hook that votes terminate on a marker ─────────────────

struct TerminateOnMarker;
impl Plugin for TerminateOnMarker {
    fn name(&self) -> &'static str {
        "terminate_on_marker"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::after_tool_call()
    }
}
#[async_trait]
impl AfterToolCall for TerminateOnMarker {
    async fn on_after_tool_call(
        &self,
        ctx: clark_agent::plugin::AfterToolCallContext<'_>,
    ) -> AfterToolDecision {
        let plain = ctx
            .result
            .content
            .iter()
            .filter_map(|b| match b {
                ToolResultBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<String>();
        if plain.contains("END") {
            AfterToolDecision {
                terminate: Some(true),
                ..AfterToolDecision::default()
            }
        } else {
            AfterToolDecision::passthrough()
        }
    }
}

// ─── An event observer that counts events ───────────────────────────

#[derive(Default)]
struct CountObserver(AtomicUsize);
impl Plugin for CountObserver {
    fn name(&self) -> &'static str {
        "count_observer"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::event_observer()
    }
}
#[async_trait]
impl EventObserver for CountObserver {
    async fn on_event(&self, _event: &AgentEvent) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn assistant_calling(tool_name: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ToolCall(ToolCall {
                id: "c1".into(),
                name: tool_name.into(),
                arguments: serde_json::json!({}),
            })],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn empty_assistant(stop_reason: StopReason, error_message: Option<&str>) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent { blocks: Vec::new() },
        stop_reason,
        error_message: error_message.map(str::to_string),
        timestamp: None,
        usage: None,
    }
}

// ─── Test ───────────────────────────────────────────────────────────

#[tokio::test]
async fn loop_runs_one_tool_call_then_finishes() {
    // Turn 1: model calls `echo` with "END".
    // Turn 2: model would finish (but `TerminateOnMarker` votes terminate
    // on the tool result, so the loop ends after turn 1).
    let turn1 = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ToolCall(ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"text": "END"}),
            })],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let stream = Arc::new(ScriptedStream::new(vec![turn1]));

    let (sink, mut rx) = ChannelSink::new();
    let counter = Arc::new(CountObserver::default());

    let registry = ToolRegistry::new().with(Arc::new(EchoTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .event_sink(Arc::new(sink))
        .before_tool_call(BlockBananas)
        .after_tool_call(TerminateOnMarker)
        .event_observer_arc(counter.clone())
        .max_iterations(10)
        .build()
        .unwrap();

    let messages = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .messages;

    // Drain events.
    drop(config);
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }

    // Sanity: messages contains user, assistant, tool result.
    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[0], AgentMessage::User { .. }));
    assert!(matches!(messages[1], AgentMessage::Assistant { .. }));
    assert!(matches!(messages[2], AgentMessage::ToolResult { .. }));

    // ToolResult content is "END".
    let AgentMessage::ToolResult { content, .. } = &messages[2] else {
        panic!()
    };
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert_eq!(t.text, "END");

    // Events include AgentStart, TurnStart, TurnEnd, ToolExecutionEnd, AgentEnd.
    assert!(events.iter().any(|e| matches!(e, AgentEvent::AgentStart)));
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::AgentEnd { .. })));

    // Observer was called.
    assert!(counter.0.load(Ordering::Relaxed) > 0);
}

#[tokio::test]
async fn before_hook_blocks_tool_call() {
    let turn1 = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ToolCall(ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"text": "banana split"}),
            })],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    // Turn 2: model "responds" to the blocked tool result with end-turn.
    let turn2 = AgentMessage::Assistant {
        content: AssistantContent::text("ok"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let stream = Arc::new(ScriptedStream::new(vec![turn1, turn2]));

    let registry = ToolRegistry::new().with(Arc::new(EchoTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .before_tool_call(BlockBananas)
        .max_iterations(10)
        .build()
        .unwrap();

    let messages = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .messages;

    // Tool result should be the blocked-error, not the echo output.
    let AgentMessage::ToolResult {
        is_error, content, ..
    } = &messages[2]
    else {
        panic!("expected tool result");
    };
    assert!(*is_error);
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert!(t.text.contains("no bananas allowed"));
}

#[tokio::test]
async fn max_tool_calls_per_turn_preserves_extra_calls_with_error_results() {
    let turn1 = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"text": "first"}),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "c2".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"text": "second"}),
                }),
            ],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let turn2 = AgentMessage::Assistant {
        content: AssistantContent::text("done"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let stream = Arc::new(ScriptedStream::new(vec![turn1, turn2]));

    let (sink, mut rx) = ChannelSink::new();
    let registry = ToolRegistry::new().with(Arc::new(EchoTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .event_sink(Arc::new(sink))
        .max_tool_calls_per_turn(1)
        .max_iterations(10)
        .build()
        .unwrap();

    let messages = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .messages;

    drop(config);
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }

    let AgentMessage::Assistant { content, .. } = &messages[1] else {
        panic!("expected assistant tool call");
    };
    assert_eq!(content.tool_calls().len(), 2);
    assert_eq!(content.tool_calls()[0].id, "c1");
    assert_eq!(content.tool_calls()[1].id, "c2");
    assert_eq!(
        messages
            .iter()
            .filter(|message| matches!(
                message,
                AgentMessage::ToolResult { tool_call_id, .. } if tool_call_id == "c2"
            ))
            .count(),
        1,
    );
    let AgentMessage::ToolResult {
        tool_call_id,
        content,
        ..
    } = &messages[2]
    else {
        panic!("expected first tool result");
    };
    assert_eq!(tool_call_id, "c1");
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert_eq!(t.text, "first");

    let AgentMessage::ToolResult {
        tool_call_id,
        content,
        is_error,
        ..
    } = &messages[3]
    else {
        panic!("expected synthetic error result");
    };
    assert_eq!(tool_call_id, "c2");
    assert!(*is_error);
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert!(t.text.contains("not executed"));
    assert!(t.text.contains("only the first 1 call"));
    assert_ne!(t.text, "second");

    let c2_end = events
        .iter()
        .find(|event| {
            matches!(
                event,
                AgentEvent::ToolExecutionEnd {
                    tool_call_id,
                    is_error: true,
                    ..
                } if tool_call_id == "c2"
            )
        })
        .expect("synthetic result should emit a tool end event");
    let AgentEvent::ToolExecutionEnd { result, .. } = c2_end else {
        unreachable!()
    };
    assert!(result.is_error);
}

#[tokio::test]
async fn ok_tool_result_can_still_mark_context_error() {
    let turn1 = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ToolCall(ToolCall {
                id: "c1".into(),
                name: "soft_fail".into(),
                arguments: serde_json::json!({}),
            })],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let turn2 = AgentMessage::Assistant {
        content: AssistantContent::text("recovered"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let stream = Arc::new(ScriptedStream::new(vec![turn1, turn2]));

    let registry = ToolRegistry::new().with(Arc::new(SoftFailTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .max_iterations(10)
        .build()
        .unwrap();

    let messages = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .messages;

    let AgentMessage::ToolResult {
        is_error, content, ..
    } = &messages[2]
    else {
        panic!("expected tool result");
    };
    assert!(*is_error);
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert_eq!(t.text, "recoverable failure");
}

#[tokio::test]
async fn execution_tool_error_remains_context_event() {
    let stream = Arc::new(ScriptedStream::new(vec![
        assistant_calling("execution_fail"),
        AgentMessage::Assistant {
            content: AssistantContent::text("recovered"),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: None,
            usage: None,
        },
    ]));

    let registry = ToolRegistry::new().with(Arc::new(ExecutionFailTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .max_iterations(10)
        .build()
        .unwrap();

    let messages = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .messages;

    let AgentMessage::ToolResult {
        is_error, content, ..
    } = &messages[2]
    else {
        panic!("expected tool result");
    };
    assert!(*is_error);
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert!(t.text.contains("recoverable execution failure"));
}

#[tokio::test]
async fn transient_stream_error_propagates_out_of_loop() {
    let stream = Arc::new(EventScriptedStream::new(vec![StreamEvent::Error {
        partial: empty_assistant(StopReason::Other, None),
        kind: StreamErrorKind::Transient,
        message: "rate limited".into(),
    }]));
    let config = AgentBuilder::new()
        .stream(stream)
        .max_iterations(10)
        .build()
        .unwrap();

    let err = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect_err("transient stream errors should abort the run");

    assert!(matches!(
        err,
        LoopError::Stream(StreamError::Transient(message)) if message == "rate limited"
    ));
}

#[tokio::test]
async fn aborted_stream_error_emits_aborted_message_end() {
    let stream = Arc::new(EventScriptedStream::new(vec![StreamEvent::Error {
        partial: empty_assistant(StopReason::Other, None),
        kind: StreamErrorKind::Aborted,
        message: "aborted before send".into(),
    }]));
    let (sink, mut rx) = ChannelSink::new();
    let config = AgentBuilder::new()
        .stream(stream)
        .event_sink(Arc::new(sink))
        .max_iterations(10)
        .build()
        .unwrap();

    let err = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect_err("aborted stream errors should abort the run");

    assert!(matches!(err, LoopError::Aborted));

    drop(config);
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }

    let message_end = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::MessageEnd {
                message: message @ AgentMessage::Assistant { .. },
            } => Some(message),
            _ => None,
        })
        .expect("aborted stream should emit a message end event");

    let AgentMessage::Assistant {
        stop_reason,
        error_message,
        ..
    } = message_end
    else {
        panic!("expected assistant message end");
    };
    assert_eq!(*stop_reason, StopReason::Aborted);
    assert_eq!(error_message.as_deref(), Some("aborted before send"));
}

#[tokio::test]
async fn empty_stream_error_propagates_out_of_loop() {
    let stream = Arc::new(EventScriptedStream::new(Vec::new()));
    let config = AgentBuilder::new()
        .stream(stream)
        .max_iterations(10)
        .build()
        .unwrap();

    let err = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect_err("stream without a terminal event should abort the run");

    assert!(matches!(err, LoopError::Stream(StreamError::Empty)));
}

#[tokio::test]
async fn assistant_error_stop_reason_propagates_out_of_loop() {
    let stream = Arc::new(ScriptedStream::new(vec![empty_assistant(
        StopReason::Error,
        Some("provider stopped with error"),
    )]));
    let config = AgentBuilder::new()
        .stream(stream)
        .max_iterations(10)
        .build()
        .unwrap();

    let err = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect_err("assistant error stop reason should abort the run");

    assert!(matches!(
        err,
        LoopError::Stream(StreamError::Transient(message))
            if message == "provider stopped with error"
    ));
}

#[tokio::test]
async fn retained_tool_update_sender_does_not_hang_after_tool_returns() {
    let retained = Arc::new(Mutex::new(None));
    let stream = Arc::new(ScriptedStream::new(vec![
        assistant_calling("retain_update_sink"),
        AgentMessage::Assistant {
            content: AssistantContent::text("finished"),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: None,
            usage: None,
        },
    ]));

    let registry = ToolRegistry::new().with(Arc::new(RetainUpdateSinkTool {
        retained: retained.clone(),
    }));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .max_iterations(10)
        .build()
        .unwrap();

    let result = tokio::time::timeout(
        Duration::from_secs(1),
        clark_agent::run(
            vec![AgentMessage::User {
                content: UserContent::Text("go".into()),
                timestamp: None,
            }],
            AgentContext::new("test"),
            &config,
            CancellationToken::new(),
        ),
    )
    .await
    .expect("retained update sink should not hang the loop")
    .expect("run should succeed");

    assert!(retained.lock().unwrap().is_some());
    let AgentMessage::ToolResult { content, .. } = &result.messages[2] else {
        panic!("expected tool result");
    };
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert_eq!(t.text, "done");
}

#[tokio::test]
async fn fatal_tool_error_propagates_out_of_loop() {
    let stream = Arc::new(ScriptedStream::new(vec![assistant_calling("fatal_fail")]));
    let registry = ToolRegistry::new().with(Arc::new(FatalTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .max_iterations(10)
        .build()
        .unwrap();

    let err = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect_err("fatal tool error should abort the loop");

    assert!(matches!(
        err,
        LoopError::ToolFatal { tool, reason }
            if tool == "fatal_fail" && reason == "unrecoverable failure"
    ));
}

#[tokio::test]
async fn aborted_tool_error_propagates_out_of_loop() {
    let stream = Arc::new(ScriptedStream::new(vec![assistant_calling("abort_fail")]));
    let registry = ToolRegistry::new().with(Arc::new(AbortTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .max_iterations(10)
        .build()
        .unwrap();

    let err = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect_err("aborted tool error should abort the loop");

    assert!(matches!(err, LoopError::Aborted));
}
