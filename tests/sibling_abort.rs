//! Sibling-abort behavior in parallel tool batches.
//!
//! Proves three things:
//!
//! 1. When a tool with `aborts_siblings_on_error = true` errors, the
//!    still-running siblings see their cancellation token fire and
//!    exit early with a recoverable `is_error: true` ToolResult — not
//!    a `LoopError`.
//! 2. The unanimous-vote termination rule is preserved: aborted
//!    siblings vote `terminate: false`, so the run continues.
//! 3. When the failing tool is opt-out (default), siblings run to
//!    completion as before.

use async_trait::async_trait;
use clark_agent::{
    AgentBuilder, AgentContext, AgentMessage, AgentTool, AssistantBlock, AssistantContent,
    StopReason, StreamEvent, StreamFn, StreamRequest, ToolCall, ToolError, ToolRegistry,
    ToolResult, ToolUpdateSink, UserContent,
};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

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

/// Errors instantly. Optionally opts in to sibling-abort.
struct FailingTool {
    aborts_siblings: bool,
}

#[async_trait]
impl AgentTool for FailingTool {
    fn name(&self) -> &str {
        "failing"
    }
    fn description(&self) -> &str {
        "always errors"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn aborts_siblings_on_error(&self) -> bool {
        self.aborts_siblings
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::error("simulated failure"))
    }
}

/// Sleeps for `slow_ms` while polling the cancellation signal. Records
/// (a) whether it observed cancellation, (b) whether it ran to natural
/// completion. Returns success when not cancelled.
struct SlowSibling {
    name: &'static str,
    slow_ms: u64,
    saw_cancel: Arc<AtomicBool>,
    completed_naturally: Arc<AtomicBool>,
}

impl SlowSibling {
    fn new(name: &'static str, slow_ms: u64) -> (Self, Arc<AtomicBool>, Arc<AtomicBool>) {
        let saw_cancel = Arc::new(AtomicBool::new(false));
        let completed_naturally = Arc::new(AtomicBool::new(false));
        (
            Self {
                name,
                slow_ms,
                saw_cancel: saw_cancel.clone(),
                completed_naturally: completed_naturally.clone(),
            },
            saw_cancel,
            completed_naturally,
        )
    }
}

#[async_trait]
impl AgentTool for SlowSibling {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "slow well-behaved sibling"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        let sleep = tokio::time::sleep(Duration::from_millis(self.slow_ms));
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {
                self.completed_naturally.store(true, Ordering::SeqCst);
                Ok(ToolResult::text("ok"))
            }
            _ = signal.cancelled() => {
                self.saw_cancel.store(true, Ordering::SeqCst);
                Err(ToolError::Aborted)
            }
        }
    }
}

fn two_call_turn(first: &str, second: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: first.into(),
                    arguments: json!({}),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "c2".into(),
                    name: second.into(),
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

fn natural_end() -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::text("done"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

#[tokio::test]
async fn sibling_aborts_when_failing_tool_opts_in() {
    let (slow, saw_cancel, completed_naturally) = SlowSibling::new("slow", 1_000);
    let saw_cancel = saw_cancel.clone();
    let completed_naturally = completed_naturally.clone();

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(FailingTool {
        aborts_siblings: true,
    }));
    registry.register(Arc::new(slow));

    let stream = ScriptedStream {
        queue: Mutex::new(vec![two_call_turn("failing", "slow"), natural_end()]),
    };
    let config = AgentBuilder::new()
        .stream(Arc::new(stream))
        .tools(registry)
        .max_iterations(5)
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("go".into()),
        timestamp: None,
    };

    let started = std::time::Instant::now();
    let result = clark_agent::run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");
    let elapsed = started.elapsed();

    // The slow sibling would naturally take 1s; sibling-abort should
    // exit it well before then.
    assert!(
        elapsed < Duration::from_millis(800),
        "expected fast abort, took {elapsed:?}"
    );
    assert!(saw_cancel.load(Ordering::SeqCst), "slow tool saw cancel");
    assert!(
        !completed_naturally.load(Ordering::SeqCst),
        "slow tool should not have completed naturally"
    );

    // Both tool results land in the transcript.
    let tool_results: Vec<&AgentMessage> = result
        .messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::ToolResult { .. }))
        .collect();
    assert_eq!(tool_results.len(), 2);

    // Both have is_error=true: the original failure plus the sibling
    // abort marker.
    for tr in &tool_results {
        let AgentMessage::ToolResult {
            is_error,
            content,
            tool_name,
            ..
        } = tr
        else {
            unreachable!()
        };
        assert!(is_error, "tool {tool_name} should be marked error");
        let text = content
            .blocks
            .iter()
            .filter_map(|b| match b {
                clark_agent::ToolResultBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<String>();
        if tool_name == "slow" {
            assert!(
                text.contains("sibling tool"),
                "slow's result should carry the sibling-abort marker, got: {text}"
            );
        }
    }
}

#[tokio::test]
async fn opt_out_failures_do_not_abort_siblings() {
    let (slow, saw_cancel, completed_naturally) = SlowSibling::new("slow", 200);

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(FailingTool {
        aborts_siblings: false,
    }));
    registry.register(Arc::new(slow));

    let stream = ScriptedStream {
        queue: Mutex::new(vec![two_call_turn("failing", "slow"), natural_end()]),
    };
    let config = AgentBuilder::new()
        .stream(Arc::new(stream))
        .tools(registry)
        .max_iterations(5)
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("go".into()),
        timestamp: None,
    };

    let _ = clark_agent::run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");

    assert!(!saw_cancel.load(Ordering::SeqCst), "no cancel expected");
    assert!(
        completed_naturally.load(Ordering::SeqCst),
        "slow tool should have completed normally"
    );
}

/// Counts how many concurrent executions overlap. Records the max
/// in-flight at any point so the test can verify that, before the
/// trigger fires, parallelism happened.
struct ConcurrentProbe {
    in_flight: Arc<AtomicUsize>,
    max_observed: Arc<AtomicUsize>,
}

impl ConcurrentProbe {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_observed = Arc::new(AtomicUsize::new(0));
        (
            Self {
                in_flight,
                max_observed: max_observed.clone(),
            },
            max_observed,
        )
    }
}

#[async_trait]
impl AgentTool for ConcurrentProbe {
    fn name(&self) -> &str {
        "probe"
    }
    fn description(&self) -> &str {
        "concurrent probe"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
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
        tokio::time::sleep(Duration::from_millis(50)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::text("ok"))
    }
}

#[tokio::test]
async fn baseline_parallelism_preserved_with_sibling_abort_path() {
    // Same machinery (FuturesUnordered + batch_token) must still run
    // tools in parallel when nothing aborts.
    let (probe, max_observed) = ConcurrentProbe::new();
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(probe));

    let two_probes = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: "probe".into(),
                    arguments: json!({}),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "c2".into(),
                    name: "probe".into(),
                    arguments: json!({}),
                }),
            ],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let stream = ScriptedStream {
        queue: Mutex::new(vec![two_probes, natural_end()]),
    };
    let config = AgentBuilder::new()
        .stream(Arc::new(stream))
        .tools(registry)
        .max_iterations(5)
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("go".into()),
        timestamp: None,
    };
    let _ = clark_agent::run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");

    assert_eq!(
        max_observed.load(Ordering::SeqCst),
        2,
        "two probes should have run concurrently"
    );
}
