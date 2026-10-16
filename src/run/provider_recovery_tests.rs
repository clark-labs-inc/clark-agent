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

struct IncidentTimedStream {
    calls: AtomicUsize,
    seconds: Vec<u64>,
    signals: std::sync::Mutex<Vec<CancellationToken>>,
}

#[async_trait::async_trait]
impl StreamFn for IncidentTimedStream {
    fn attempt_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(180)
    }

    async fn stream(
        &self,
        _: StreamRequest,
        signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.signals.lock().unwrap().push(signal);
        let seconds = self.seconds[call];
        tokio::time::sleep(std::time::Duration::from_secs(seconds)).await;
        let message = AgentMessage::Assistant {
            content: AssistantContent::text(if call < 2 { "partial" } else { "complete" }),
            stop_reason: if call < 2 { StopReason::Error } else { StopReason::EndTurn },
            error_message: None,
            timestamp: None,
            usage: None,
        };
        Box::pin(stream::iter(vec![StreamEvent::Done { message }]))
    }
}

struct StalledAttempt(AtomicUsize);

#[async_trait::async_trait]
impl StreamFn for StalledAttempt {
    fn attempt_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(1)
    }

    async fn stream(&self, _: StreamRequest, _: CancellationToken) -> BoxStream<'static, StreamEvent> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(stream::pending())
    }
}

#[tokio::test(start_paused = true)]
async fn configured_attempt_timeout_bounds_original_and_retries() {
    let stream = Arc::new(StalledAttempt(AtomicUsize::new(0)));
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    let parent = CancellationToken::new();
    let started = tokio::time::Instant::now();
    let result = stream_with_max_tokens_recovery(
        &AgentContext::new("synthetic attempt deadline"), &config, &parent, 0,
    ).await;
    assert!(matches!(result, Err(LoopError::Stream(StreamError::RequestTimeout { timeout_ms: 1_000 }))));
    assert_eq!(stream.0.load(Ordering::SeqCst), 3);
    assert_eq!(started.elapsed(), std::time::Duration::from_millis(4_500));
    assert!(!parent.is_cancelled());
}

#[tokio::test(start_paused = true)]
async fn slow_third_attempt_can_complete_after_retry_admission_window() {
    let stream = Arc::new(IncidentTimedStream {
        calls: AtomicUsize::new(0),
        seconds: vec![69, 70, 61],
        signals: std::sync::Mutex::new(Vec::new()),
    });
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    let started = tokio::time::Instant::now();
    let result = stream_with_max_tokens_recovery(
        &AgentContext::new("synthetic incident replay"),
        &config,
        &CancellationToken::new(),
        10,
    ).await;
    assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
    assert!(result.is_ok(), "admitted request was interrupted: {result:?}");
    assert_eq!(started.elapsed(), std::time::Duration::from_millis(201_500));
    assert!(stream.signals.lock().unwrap().iter().all(CancellationToken::is_cancelled));
}

#[tokio::test(start_paused = true)]
async fn elapsed_retry_window_prevents_another_request_after_slow_failure() {
    let stream = Arc::new(IncidentTimedStream {
        calls: AtomicUsize::new(0),
        seconds: vec![69, 121],
        signals: std::sync::Mutex::new(Vec::new()),
    });
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    let result = stream_with_max_tokens_recovery(
        &AgentContext::new("synthetic retry admission"), &config,
        &CancellationToken::new(), 0,
    ).await;
    assert!(matches!(result, Err(LoopError::Stream(StreamError::Transient(_)))));
    assert_eq!(stream.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn caller_cancellation_interrupts_an_admitted_request_and_cancels_its_child() {
    let stream = Arc::new(IncidentTimedStream {
        calls: AtomicUsize::new(0),
        seconds: vec![69, 70, 61],
        signals: std::sync::Mutex::new(Vec::new()),
    });
    let config = AgentBuilder::new().stream(stream.clone()).build().unwrap();
    let signal = CancellationToken::new();
    let context = AgentContext::new("synthetic cancellation");
    let run = stream_with_max_tokens_recovery(&context, &config, &signal, 0);
    let cancel = async {
        tokio::time::sleep(std::time::Duration::from_secs(150)).await;
        signal.cancel();
    };
    let (result, ()) = tokio::join!(run, cancel);
    assert!(matches!(result, Err(LoopError::Aborted)));
    assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
    assert!(stream.signals.lock().unwrap().iter().all(CancellationToken::is_cancelled));
}
