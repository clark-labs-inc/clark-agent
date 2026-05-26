//! Integration test for the max-output-tokens recovery ladder.
//!
//! Wires a `ScriptedStream` that returns truncated turns until enough
//! retries land, then a clean turn. Verifies the loop discards the
//! truncated turns, emits one `OutputTokensEscalation` per retry, and
//! eventually accepts the clean turn into the transcript.

use async_trait::async_trait;
use clark_agent::{
    run, AgentBuilder, AgentContext, AgentEvent, AgentMessage, AssistantContent, ChannelSink,
    EventObserver, MaxTokensRecovery, Plugin, PluginCapabilities, StopReason, StreamEvent,
    StreamFn, StreamRequest, TokenScaling, UserContent,
};
use futures::stream::{self, BoxStream, StreamExt};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
use tokio_util::sync::CancellationToken;

/// Stream that hands back the next scripted message and records the
/// `max_output_tokens` cap that was on the request — so tests can
/// assert the recovery ladder threaded the right values.
struct ScriptedStream {
    responses: Mutex<Vec<AgentMessage>>,
    seen_caps: Mutex<Vec<Option<u32>>>,
}

impl ScriptedStream {
    fn new(responses: Vec<AgentMessage>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses),
            seen_caps: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl StreamFn for ScriptedStream {
    async fn stream(
        &self,
        request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        self.seen_caps
            .lock()
            .unwrap()
            .push(request.max_output_tokens);
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

fn truncated_assistant() -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::text(""),
        stop_reason: StopReason::MaxTokens,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

fn complete_assistant(text: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: AssistantContent::text(text),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    }
}

/// Counts `OutputTokensEscalation` events surfaced through
/// `EventObserver`. Use this rather than draining the channel sink so
/// the assertion targets the agent-level event, not its serialized
/// shape.
struct EscalationCounter {
    count: Arc<AtomicU32>,
}

impl Plugin for EscalationCounter {
    fn name(&self) -> &'static str {
        "escalation_counter"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::event_observer()
    }
}

#[async_trait]
impl EventObserver for EscalationCounter {
    async fn on_event(&self, event: &AgentEvent) {
        if matches!(event, AgentEvent::OutputTokensEscalation { .. }) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[tokio::test]
async fn recovery_doubles_cap_until_clean_turn() {
    // Two truncated turns then a clean one — recovery should walk
    // 4096 -> 8192 -> 16384, accept the third turn.
    let scripted = ScriptedStream::new(vec![
        truncated_assistant(),
        truncated_assistant(),
        complete_assistant("hello"),
    ]);
    let counter = Arc::new(AtomicU32::new(0));
    let observer = EscalationCounter {
        count: counter.clone(),
    };
    let (sink, mut rx) = ChannelSink::new();

    let config = AgentBuilder::new()
        .stream(scripted.clone() as Arc<dyn StreamFn>)
        .event_sink(Arc::new(sink))
        .event_observer(observer)
        .max_output_tokens(4096)
        .max_output_tokens_recovery(MaxTokensRecovery::doubling())
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("hi".into()),
        timestamp: None,
    };

    let result = run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");

    // Recovery fired twice (the third call returned a clean turn).
    assert_eq!(counter.load(Ordering::Relaxed), 2);

    // Caps observed: starting cap, then 8192, then 16384.
    let caps = scripted.seen_caps.lock().unwrap().clone();
    assert_eq!(caps, vec![Some(4096), Some(8192), Some(16384)]);

    // The accepted turn is the clean one.
    let final_assistant = result
        .messages
        .iter()
        .rev()
        .find(|m| matches!(m, AgentMessage::Assistant { .. }))
        .expect("assistant message");
    let AgentMessage::Assistant {
        stop_reason,
        content,
        ..
    } = final_assistant
    else {
        unreachable!()
    };
    assert_eq!(*stop_reason, StopReason::EndTurn);
    assert_eq!(content.plain_text(), "hello");

    // The discarded truncated turns must not be in the run's emitted
    // tail — only the accepted assistant message is appended.
    let truncated_count = result
        .messages
        .iter()
        .filter(|m| {
            matches!(
                m,
                AgentMessage::Assistant {
                    stop_reason: StopReason::MaxTokens,
                    ..
                }
            )
        })
        .count();
    assert_eq!(truncated_count, 0);

    // Drain the channel sink so the receiver doesn't deadlock the test
    // runner if more producers exist.
    while rx.try_recv().is_ok() {}
}

#[tokio::test]
async fn recovery_off_by_default_accepts_truncated_turn() {
    let scripted = ScriptedStream::new(vec![truncated_assistant()]);
    let counter = Arc::new(AtomicU32::new(0));
    let observer = EscalationCounter {
        count: counter.clone(),
    };

    let config = AgentBuilder::new()
        .stream(scripted.clone() as Arc<dyn StreamFn>)
        .event_observer(observer)
        .max_output_tokens(4096)
        // No .max_output_tokens_recovery() call.
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("hi".into()),
        timestamp: None,
    };

    let result = run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");

    assert_eq!(counter.load(Ordering::Relaxed), 0);
    // Truncated turn lands in the transcript when recovery is off.
    let assistants: Vec<_> = result
        .messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::Assistant { .. }))
        .collect();
    assert_eq!(assistants.len(), 1);
    let AgentMessage::Assistant { stop_reason, .. } = assistants[0] else {
        unreachable!()
    };
    assert_eq!(*stop_reason, StopReason::MaxTokens);
}

#[tokio::test]
async fn recovery_gives_up_after_max_attempts_and_accepts_truncated() {
    // 4 truncated turns scripted; max_attempts=2 means the loop tries
    // the original cap + 2 retries (3 total streamings) then accepts
    // the third truncated turn.
    let scripted = ScriptedStream::new(vec![
        truncated_assistant(),
        truncated_assistant(),
        truncated_assistant(),
    ]);
    let counter = Arc::new(AtomicU32::new(0));
    let observer = EscalationCounter {
        count: counter.clone(),
    };

    let recovery = MaxTokensRecovery {
        max_attempts: 2,
        scaling: TokenScaling::Double,
        ceiling: None,
    };
    let config = AgentBuilder::new()
        .stream(scripted.clone() as Arc<dyn StreamFn>)
        .event_observer(observer)
        .max_output_tokens(4096)
        .max_output_tokens_recovery(recovery)
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("hi".into()),
        timestamp: None,
    };

    let result = run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");

    assert_eq!(counter.load(Ordering::Relaxed), 2);
    assert_eq!(scripted.seen_caps.lock().unwrap().len(), 3);
    // Final turn (truncated) is accepted into the transcript.
    let assistants: Vec<_> = result
        .messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::Assistant { .. }))
        .collect();
    assert_eq!(assistants.len(), 1);
    let AgentMessage::Assistant { stop_reason, .. } = assistants[0] else {
        unreachable!()
    };
    assert_eq!(*stop_reason, StopReason::MaxTokens);
}

#[tokio::test]
async fn recovery_skipped_when_no_starting_cap() {
    // Without an initial `max_output_tokens`, the recovery has no
    // number to scale from. The truncated turn is accepted as-is.
    let scripted = ScriptedStream::new(vec![truncated_assistant()]);
    let counter = Arc::new(AtomicU32::new(0));
    let observer = EscalationCounter {
        count: counter.clone(),
    };

    let config = AgentBuilder::new()
        .stream(scripted.clone() as Arc<dyn StreamFn>)
        .event_observer(observer)
        .max_output_tokens_recovery(MaxTokensRecovery::doubling())
        // No .max_output_tokens(...) call.
        .build()
        .expect("builder");

    let context = AgentContext::new("system");
    let prompt = AgentMessage::User {
        content: UserContent::Text("hi".into()),
        timestamp: None,
    };

    let _ = run(vec![prompt], context, &config, CancellationToken::new())
        .await
        .expect("run");

    assert_eq!(counter.load(Ordering::Relaxed), 0);
}
