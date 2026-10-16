use super::*;
use crate::context_observations::expand;
use crate::ContextObservation;
use serde_json::json;

fn assistant() -> AgentMessage {
    AgentMessage::Assistant { content: AssistantContent::text("answer"),
        stop_reason: StopReason::EndTurn, error_message: None,
        timestamp: Some(42), usage: None }
}

fn observation() -> ContextObservation {
    ContextObservation { kind: "host_snapshot".into(),
        payload: json!({"clock":"2026-10-09T16:00:00Z"}), timestamp: Some(40) }
}

struct ObservationProjector;
impl crate::Plugin for ObservationProjector {
    fn name(&self) -> &'static str { "fixture_observation" }
    fn capabilities(&self) -> crate::PluginCapabilities {
        crate::PluginCapabilities::context_transform()
    }
}
#[async_trait::async_trait]
impl crate::ContextTransform for ObservationProjector {
    async fn transform(&self, mut messages: Vec<AgentMessage>, _: &crate::TransformContext<'_>) -> Vec<AgentMessage> {
        messages.push(observation().into_message());
        messages
    }
    fn history_observations(&self, messages: &[AgentMessage]) -> Vec<ContextObservation> {
        assert_eq!(messages.last(), Some(&observation().into_message()));
        vec![observation()]
    }
}

struct OneResponse;
#[async_trait::async_trait]
impl crate::StreamFn for OneResponse {
    async fn stream(&self, request: crate::StreamRequest,
        _: tokio_util::sync::CancellationToken) -> futures::stream::BoxStream<'static, crate::StreamEvent> {
        assert_eq!(request.messages.last(), Some(&observation().into_message()));
        use futures::StreamExt;
        futures::stream::iter(vec![crate::StreamEvent::Done { message: assistant() }]).boxed()
    }
}

#[tokio::test]
async fn canonical_loop_retains_the_final_prepared_observations_on_the_response() {
    let (sink, mut events) = crate::event::ChannelSink::new();
    let config = crate::AgentBuilder::new()
        .stream(std::sync::Arc::new(OneResponse))
        .event_sink(std::sync::Arc::new(sink))
        .context_transform(ObservationProjector)
        .build().unwrap();
    let mut context = crate::AgentContext::new("system");
    let signal = tokio_util::sync::CancellationToken::new();
    let (response, _) = stream_assistant_response(
        &context, &config, &signal, 0, None, crate::ReasoningEffort::High,
    ).await.unwrap();
    assert_eq!(context.messages.len(), 0, "no parallel history was mutated");
    assert_eq!(expand(vec![response.clone()]), vec![observation().into_message(), assistant()]);
    context.messages.push(response);
    context.messages.push(AgentMessage::User {
        content: crate::UserContent::Text("next question".into()), timestamp: None,
    });
    stream_assistant_response(&context, &config, &signal, 1, None, crate::ReasoningEffort::High)
        .await.unwrap();
    let mut requests = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let crate::AgentEvent::ProviderRequestPrepared { messages, .. } = event {
            requests.push(messages);
        }
    }
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1][0], requests[0][0], "the complete prior input is an exact prefix");
    assert_eq!(requests[1][1], assistant());
    assert_eq!(requests[1].len(), 4, "one historical and one fresh observation");
}

struct FailedResponse;
#[async_trait::async_trait]
impl crate::StreamFn for FailedResponse {
    async fn stream(&self, _: crate::StreamRequest,
        _: CancellationToken) -> BoxStream<'static, crate::StreamEvent> {
        use futures::StreamExt;
        stream::iter(vec![crate::StreamEvent::Error {
            partial: assistant(), kind: crate::stream::StreamErrorKind::Transient,
            message: "fixture failure".into(),
        }]).boxed()
    }
}

#[tokio::test]
async fn failed_response_event_retains_request_provenance() {
    let (sink, mut events) = crate::event::ChannelSink::new();
    let config = crate::AgentBuilder::new().stream(Arc::new(FailedResponse))
        .event_sink(Arc::new(sink)).context_transform(ObservationProjector).build().unwrap();
    let result = stream_assistant_response(&crate::AgentContext::new("system"), &config,
        &CancellationToken::new(), 0, None, crate::ReasoningEffort::High).await;
    assert!(result.is_err());
    let response = std::iter::from_fn(|| events.try_recv().ok()).find_map(|event| {
        if let crate::AgentEvent::MessageEnd { message } = event { Some(message) } else { None }
    }).expect("failure emits the durable response event");
    let expanded = expand(vec![response]);
    assert_eq!(expanded[0], observation().into_message());
    assert!(matches!(&expanded[1], AgentMessage::Assistant { error_message: Some(error), .. }
        if error == "fixture failure"));
}
