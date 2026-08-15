//! Integration test for the max-output-tokens recovery ladder.
//!
//! Wires a `ScriptedStream` that returns truncated turns until enough
//! retries land, then a clean turn. Verifies the loop discards the
//! truncated turns, emits one `OutputTokensEscalation` per retry, and
//! eventually accepts the clean turn into the transcript.

use async_trait::async_trait;
use clark_agent::{
    run, AgentBuilder, AgentContext, AgentEvent, AgentMessage, AssistantContent, ChannelSink,
    EventObserver, Plugin, PluginCapabilities, StopReason, StreamEvent, StreamFn, StreamRequest,
    UserContent,
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
        // No .max_output_tokens(...) call, so there is no cap to grow.
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
