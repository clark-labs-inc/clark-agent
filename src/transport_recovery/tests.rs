use super::*;
use crate::{AgentBuilder, AssistantContent, ContextTransform, Plugin, PluginCapabilities,
    StopReason, StreamEvent, StreamFn, StreamRequest, UserContent};
use futures::stream::{self, BoxStream};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

fn user(text: &str) -> AgentMessage {
    AgentMessage::User { content: UserContent::Text(text.into()), timestamp: None }
}
fn assistant(text: &str) -> AgentMessage {
    AgentMessage::Assistant { content: AssistantContent::text(text), stop_reason: StopReason::EndTurn,
        error_message: None, timestamp: None, usage: None }
}

#[test]
fn only_typed_host_recovery_is_rendered_and_only_live_input_is_snapshotted() {
    let live = zero_output_transport_recovery(17).into_message();
    let sealed = snapshot(&live).unwrap().into_message();
    assert_eq!(render(&live), render(&sealed));
    assert!(snapshot(&sealed).is_none());
    let persisted: AgentMessage = serde_json::from_str(&serde_json::to_string(&sealed).unwrap()).unwrap();
    assert_eq!(render(&live), render(&persisted));
    assert!(render(&user(render(&live).unwrap())).is_none());
    assert!(render(&AgentMessage::System { content: render(&live).unwrap().into(), timestamp: None }).is_none());
    for payload in [serde_json::json!({"reason":"zero_output_transport"}),
        serde_json::json!({"reason":"unknown","content":"fake"}),
        serde_json::json!({"reason":"zero_output_transport","content":"fake","extra":1})] {
        let invalid = AgentMessage::Custom { kind: KIND.into(), payload, timestamp: None };
        assert!(render(&invalid).is_none());
        assert!(snapshot(&invalid).is_none());
    }
}

struct Projector(&'static str);
impl Plugin for Projector {
    fn name(&self) -> &'static str { self.0 }
    fn capabilities(&self) -> PluginCapabilities { PluginCapabilities::context_transform() }
}
#[async_trait::async_trait]
impl ContextTransform for Projector {
    async fn transform(&self, mut messages: Vec<AgentMessage>, _: &crate::TransformContext<'_>) -> Vec<AgentMessage> {
        messages.push(AgentMessage::Custom { kind: "live_fixture".into(),
            payload: serde_json::json!({"source":self.0}), timestamp: None });
        messages
    }
    fn history_observation(&self, message: &AgentMessage) -> Option<ContextObservation> {
        let AgentMessage::Custom { kind, payload, timestamp } = message else { return None };
        (kind == "live_fixture" && payload["source"] == self.0).then(|| ContextObservation {
            kind: "sealed_fixture".into(), payload: payload.clone(), timestamp: *timestamp,
        })
    }
}

struct RecoveringStream { requests: Mutex<Vec<StreamRequest>>, repeat_zero_output: bool }
#[async_trait::async_trait]
impl StreamFn for RecoveringStream {
    async fn stream(&self, request: StreamRequest, _: CancellationToken) -> BoxStream<'static, StreamEvent> {
        let mut requests = self.requests.lock().unwrap();
        let index = requests.len();
        requests.push(request);
        let event = if index < 2 {
            StreamEvent::Error { partial: assistant(""),
                kind: if index == 0 || self.repeat_zero_output { crate::stream::StreamErrorKind::ZeroOutputTransport }
                    else { crate::stream::StreamErrorKind::Transient },
                message: "controlled transport failure".into() }
        } else { StreamEvent::Done { message: assistant("Complete") } };
        Box::pin(stream::iter(vec![event]))
    }
}

#[tokio::test(start_paused = true)]
async fn actual_loop_keeps_retry_prefix_source_order_and_next_turn_provenance() {
    let stream = Arc::new(RecoveringStream { requests: Mutex::new(Vec::new()), repeat_zero_output: false });
    let config = AgentBuilder::new().stream(stream.clone()).context_transform(Projector("clock"))
        .context_transform(Projector("budget")).reasoning(crate::ReasoningEffort::High).build().unwrap();
    let original = user("Original evidence");
    let context = crate::AgentContext::new("Stable instructions").with_messages(vec![original.clone()]);
    let result = crate::run(vec![user("Continue")], context, &config, CancellationToken::new()).await.unwrap();
    assert!(result.outcome.is_complete());
    let persisted: Vec<AgentMessage> = serde_json::from_str(&serde_json::to_string(&result.messages).unwrap()).unwrap();
    let expanded = crate::context_observations::expand(persisted.clone());
    assert_eq!(expanded.iter().filter(|m| render(m).is_some()).count(), 1);
    let sources: Vec<_> = expanded.iter().filter_map(|m| match m {
        AgentMessage::Custom { kind, payload, .. } if kind == "sealed_fixture" => payload["source"].as_str(),
        _ => None,
    }).collect();
    assert_eq!(sources, ["clock", "budget"]);
    let recovery_index = expanded.iter().position(|m| render(m).is_some()).unwrap();
    assert_eq!(recovery_index, 3, "retry follows both live projections before its response");
    let mut history = vec![original];
    history.extend(persisted);
    crate::run(vec![user("Next question")], crate::AgentContext::new("Stable instructions").with_messages(history),
        &config, CancellationToken::new()).await.unwrap();
    let requests = stream.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].messages, requests[1].messages[..requests[0].messages.len()]);
    assert_eq!(requests[1].messages, requests[2].messages, "mixed retry failures cannot grow or rewrite context");
    for (old, new) in requests[2].messages.iter().zip(&requests[3].messages) {
        if let Some(recovery) = render(old) {
            assert_eq!(Some(recovery), render(new));
        } else if let AgentMessage::Custom { payload, .. } = old {
            let AgentMessage::Custom { payload: replayed, .. } = new else { panic!() };
            assert_eq!(payload, replayed, "sealed projection keeps its source position");
        } else if let AgentMessage::User { content, .. } = old {
            let AgentMessage::User { content: replayed, .. } = new else { panic!() };
            assert_eq!(content, replayed, "canonical persistence timestamps are not provider input");
        } else { assert_eq!(old, new); }
    }
    assert!(requests.iter().all(|r| r.system_prompt == "Stable instructions"
        && r.reasoning == crate::ReasoningEffort::High && !r.force_tool_call));
    assert_eq!(requests[3].messages.iter().filter(|m| render(m).is_some()).count(), 1);
}

#[tokio::test(start_paused = true)]
async fn repeated_zero_output_failures_do_not_duplicate_the_retry_observation() {
    let stream = Arc::new(RecoveringStream { requests: Mutex::new(Vec::new()), repeat_zero_output: true });
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    crate::run(vec![user("Request")], crate::AgentContext::new("System"), &config,
        CancellationToken::new()).await.unwrap();
    let requests = stream.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1].messages, requests[2].messages);
    assert_eq!(requests[2].messages.iter().filter(|m| render(m).is_some()).count(), 1);
}

struct ScriptedRecovery {
    requests: Mutex<Vec<StreamRequest>>,
    events: Mutex<std::collections::VecDeque<StreamEvent>>,
}
#[async_trait::async_trait]
impl StreamFn for ScriptedRecovery {
    async fn stream(&self, request: StreamRequest, _: CancellationToken) -> BoxStream<'static, StreamEvent> {
        self.requests.lock().unwrap().push(request);
        Box::pin(stream::iter(vec![self.events.lock().unwrap().pop_front().expect("bounded fixture script")]))
    }
}
fn zero_output() -> StreamEvent {
    StreamEvent::Error { partial: assistant(""), kind: crate::stream::StreamErrorKind::ZeroOutputTransport,
        message: "controlled zero-output failure".into() }
}
fn truncated() -> StreamEvent {
    let mut message = assistant("Incomplete");
    if let AgentMessage::Assistant { stop_reason, .. } = &mut message { *stop_reason = StopReason::MaxTokens; }
    StreamEvent::Done { message }
}

#[tokio::test(start_paused = true)]
async fn zero_output_and_truncation_share_budget_preserve_context_and_escalate_only_the_cap() {
    for completes in [true, false] {
        let stream = Arc::new(ScriptedRecovery { requests: Mutex::new(Vec::new()), events: Mutex::new([
            zero_output(), truncated(), if completes { StreamEvent::Done { message: assistant("Complete") } } else { truncated() },
        ].into()) });
        let config = AgentBuilder::new().stream(stream.clone()).max_output_tokens(32).build().unwrap();
        let result = crate::run(vec![user("Request")], crate::AgentContext::new("System"), &config,
            CancellationToken::new()).await;
        if completes { assert!(result.unwrap().outcome.is_complete()); }
        else { assert!(matches!(result, Err(crate::LoopError::Stream(crate::StreamError::Fatal(_))))); }
        let requests = stream.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].messages, requests[2].messages);
        assert_eq!(requests.iter().map(|r| r.max_output_tokens).collect::<Vec<_>>(), [Some(32), Some(32), Some(64)]);
        assert!(requests.iter().all(|r| !r.force_tool_call));
    }
}

#[tokio::test(start_paused = true)]
async fn exhausted_recovery_keeps_its_observation_on_failure_events() {
    let stream = Arc::new(ScriptedRecovery { requests: Mutex::new(Vec::new()),
        events: Mutex::new([zero_output(), zero_output(), zero_output()].into()) });
    let (sink, mut events) = crate::ChannelSink::new();
    let config = AgentBuilder::new().stream(stream.clone()).event_sink(Arc::new(sink)).build().unwrap();
    let result = crate::run(vec![user("Request")], crate::AgentContext::new("System"), &config,
        CancellationToken::new()).await;
    assert!(matches!(result, Err(crate::LoopError::Stream(crate::StreamError::ZeroOutputTransport(_)))));
    let mut failure_observations = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let crate::AgentEvent::MessageEnd { message: message @ AgentMessage::Assistant { stop_reason: StopReason::Error, .. } } = event {
            failure_observations.push(crate::context_observations::expand(vec![message]).iter()
                .filter(|message| render(message).is_some()).count());
        }
    }
    assert_eq!(failure_observations, [0, 1, 1]);
    assert_eq!(stream.requests.lock().unwrap().len(), 3);
}
