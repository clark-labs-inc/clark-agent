//! Batch-downgrade test for `requires_exclusive_sandbox`.
//!
//! Proves that a tool which declares `requires_exclusive_sandbox = true`
//! forces its batch to run sequentially. This is the only per-tool
//! signal for sequential dispatch; loop-wide sequential mode is
//! configured separately via `AgentBuilder::default_execution_mode`.

use async_trait::async_trait;
use clark_agent::{
    AgentBuilder, AgentContext, AgentMessage, AgentTool, AssistantBlock, AssistantContent,
    StopReason, StreamEvent, StreamFn, StreamRequest, ToolCall, ToolError, ToolRegistry,
    ToolResult, ToolUpdateSink, UserContent,
};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Scripted stream: turn 1 emits two parallel tool calls, turn 2
/// emits a natural EndTurn so the loop exits cleanly.
struct ScriptedStream {
    queue: Mutex<Vec<AgentMessage>>,
}

#[async_trait]
impl StreamFn for ScriptedStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let next = self.queue.lock().unwrap().remove(0);
        let events = vec![
            StreamEvent::Start {
                partial: next.clone(),
            },
            StreamEvent::Done { message: next },
        ];
        stream::iter(events).boxed()
    }
}

/// Tool whose `execute` records overlap by tracking how many calls are
/// in flight at once. The constructor takes the `exclusive` flag so we
/// can flip it on/off in different test cases without two tool types.
struct OverlapProbe {
    name: &'static str,
    exclusive: bool,
    in_flight: Arc<AtomicUsize>,
    max_observed: Arc<AtomicUsize>,
}

impl OverlapProbe {
    fn new(name: &'static str, exclusive: bool) -> (Self, Arc<AtomicUsize>) {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_observed = Arc::new(AtomicUsize::new(0));
        let probe = Self {
            name,
            exclusive,
            in_flight,
            max_observed: max_observed.clone(),
        };
        (probe, max_observed)
    }
}

#[async_trait]
impl AgentTool for OverlapProbe {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "overlap probe"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn requires_exclusive_sandbox(&self) -> bool {
        self.exclusive
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        let mut prev = self.max_observed.load(Ordering::SeqCst);
        while now > prev {
            match self
                .max_observed
                .compare_exchange(prev, now, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }
        // Hold the slot long enough that any concurrent call WILL be
        // observed if the dispatcher ran them in parallel. Sequential
        // dispatch will never see in_flight > 1.
        tokio::time::sleep(Duration::from_millis(80)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::text("ok"))
    }
}

fn two_call_turn(name: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: name.into(),
                    arguments: json!({}),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "c2".into(),
                    name: name.into(),
                    arguments: json!({}),
                }),
            ],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn end_turn() -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::text("done"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn user_prompt() -> Vec<AgentMessage> {
    vec![AgentMessage::User {
        content: UserContent::Text("go".into()),
        timestamp: None,
    }]
}

/// Baseline: parallel-default tool (no exclusive flag) → batch runs
/// concurrently → max_observed >= 2.
#[tokio::test]
async fn parallel_default_overlaps() {
    let (probe, max_observed) = OverlapProbe::new("probe", false);
    let stream = Arc::new(ScriptedStream {
        queue: Mutex::new(vec![two_call_turn("probe"), end_turn()]),
    });
    let registry = ToolRegistry::new().with(Arc::new(probe));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .build()
        .expect("build");

    let _ = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(
        max_observed.load(Ordering::SeqCst),
        2,
        "parallel-default tool should overlap with itself in a batch",
    );
}

/// `requires_exclusive_sandbox = true` → batch downgrades to
/// Sequential → max_observed == 1.
#[tokio::test]
async fn exclusive_sandbox_serializes_batch() {
    let (probe, max_observed) = OverlapProbe::new("probe", true);
    let stream = Arc::new(ScriptedStream {
        queue: Mutex::new(vec![two_call_turn("probe"), end_turn()]),
    });
    let registry = ToolRegistry::new().with(Arc::new(probe));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .build()
        .expect("build");

    let _ = clark_agent::run(
        user_prompt(),
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("run");

    assert_eq!(
        max_observed.load(Ordering::SeqCst),
        1,
        "requires_exclusive_sandbox should serialize the entire batch",
    );
}
