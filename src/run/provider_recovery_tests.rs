use super::*;
use crate::{AgentBuilder, StreamFn};
use futures::stream::{self, BoxStream};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct FailingStream {
    calls: AtomicUsize,
    kinds: Vec<StreamErrorKind>,
    error_stop: bool,
}

#[async_trait::async_trait]
impl StreamFn for FailingStream {
    async fn stream(
        &self,
        _: StreamRequest,
        _: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let partial = AgentMessage::Assistant {
            content: AssistantContent::text(""),
            stop_reason: if self.error_stop {
                StopReason::Error
            } else {
                StopReason::Other
            },
            error_message: Some("original provider failure".into()),
            timestamp: None,
            usage: None,
        };
        let event = if self.error_stop {
            StreamEvent::Done { message: partial }
        } else {
            StreamEvent::Error {
                partial,
                kind: self.kinds[call % self.kinds.len()],
                message: "original provider failure".into(),
            }
        };
        Box::pin(stream::iter(vec![event]))
    }
}

#[tokio::test(start_paused = true)]
async fn every_retryable_transport_failure_has_the_same_ceiling() {
    for kind in [
        StreamErrorKind::Empty,
        StreamErrorKind::ZeroOutputTransport,
        StreamErrorKind::Transient,
        StreamErrorKind::ProviderRateLimited,
    ] {
        let stream = Arc::new(FailingStream {
            calls: AtomicUsize::new(0),
            kinds: vec![kind],
            error_stop: false,
        });
        let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
        let error = stream_with_max_tokens_recovery(
            &AgentContext::new("system"),
            &config,
            &CancellationToken::new(),
            0,
        )
        .await
        .unwrap_err();
        assert_eq!(stream.calls.load(Ordering::SeqCst), 3, "{kind:?}");
        match (kind, error) {
            (StreamErrorKind::Empty, LoopError::Stream(StreamError::Empty)) => {}
            (
                StreamErrorKind::ZeroOutputTransport,
                LoopError::Stream(StreamError::ZeroOutputTransport(message)),
            )
            | (StreamErrorKind::Transient, LoopError::Stream(StreamError::Transient(message)))
            | (
                StreamErrorKind::ProviderRateLimited,
                LoopError::Stream(StreamError::ProviderRateLimited(message)),
            ) => assert_eq!(message, "original provider failure"),
            pair => panic!("lost error classification: {pair:?}"),
        }
    }
}

#[tokio::test(start_paused = true)]
async fn changing_error_kinds_does_not_reset_the_budget() {
    let stream = Arc::new(FailingStream {
        calls: AtomicUsize::new(0),
        kinds: vec![
            StreamErrorKind::Empty,
            StreamErrorKind::ZeroOutputTransport,
            StreamErrorKind::Transient,
        ],
        error_stop: false,
    });
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    let result = stream_with_max_tokens_recovery(
        &AgentContext::new("system"),
        &config,
        &CancellationToken::new(),
        0,
    )
    .await;
    assert!(matches!(
        result,
        Err(LoopError::Stream(StreamError::Transient(_)))
    ));
    assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test(start_paused = true)]
async fn error_stop_cannot_bypass_the_transport_budget() {
    let stream = Arc::new(FailingStream {
        calls: AtomicUsize::new(0),
        kinds: vec![],
        error_stop: true,
    });
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    let result = stream_with_max_tokens_recovery(
        &AgentContext::new("system"),
        &config,
        &CancellationToken::new(),
        0,
    )
    .await;
    assert!(
        matches!(result, Err(LoopError::Stream(StreamError::Transient(message)))
        if message == "original provider failure")
    );
    assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
}
