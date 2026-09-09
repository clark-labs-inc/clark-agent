use super::*;

/// One shared recovery budget, including mixed transport failures and output truncation.
pub(super) const PROVIDER_RECOVERY_MAX_ATTEMPTS: u32 = 3;
pub(super) const PROVIDER_RECOVERY_MAX_ELAPSED: std::time::Duration =
    std::time::Duration::from_secs(120);

pub(super) async fn stream_with_max_tokens_recovery(
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    iteration: usize,
) -> Result<(AgentMessage, Option<std::collections::HashSet<String>>), LoopError> {
    let mut current_cap = config.max_output_tokens;
    let mut max_tokens_attempt: u32 = 0;
    let mut attempts: u32 = 0;
    let mut recovery_started_at: Option<tokio::time::Instant> = None;
    let mut last_error: Option<StreamError> = None;
    let mut zero_output_recovery_context: Option<AgentContext> = None;

    loop {
        attempts += 1;
        let attempt_context = zero_output_recovery_context.as_ref().unwrap_or(context);
        let attempt = stream_assistant_response(
            attempt_context,
            config,
            signal,
            iteration,
            current_cap,
            config.reasoning,
        );
        let response = if let Some(started_at) = recovery_started_at {
            let remaining = PROVIDER_RECOVERY_MAX_ELAPSED.saturating_sub(started_at.elapsed());
            tokio::select! {
                _ = signal.cancelled() => return Err(LoopError::Aborted),
                result = tokio::time::timeout(remaining, attempt) => match result {
                    Ok(result) => result,
                    Err(_) => return Err(LoopError::Stream(last_error.take().expect("recovery error"))),
                }
            }
        } else {
            attempt.await
        };
        let response = match response {
            Ok((assistant, allowlist)) => match &assistant {
                AgentMessage::Assistant {
                    stop_reason: StopReason::Aborted,
                    ..
                } => return Err(LoopError::Aborted),
                AgentMessage::Assistant {
                    stop_reason: StopReason::Error,
                    error_message,
                    ..
                } => Err(LoopError::Stream(StreamError::Transient(
                    error_message
                        .clone()
                        .unwrap_or_else(|| "Provider ended the stream with an error".into()),
                ))),
                _ => Ok((assistant, allowlist)),
            },
            error => error,
        };
        let (assistant, allowlist) = match response {
            Ok(pair) => pair,
            Err(LoopError::Stream(
                error @ (StreamError::Empty
                | StreamError::ZeroOutputTransport(_)
                | StreamError::ProviderRateLimited(_)
                | StreamError::Transient(_)),
            )) => {
                let started_at = *recovery_started_at.get_or_insert_with(tokio::time::Instant::now);
                if attempts >= PROVIDER_RECOVERY_MAX_ATTEMPTS {
                    return Err(LoopError::Stream(error));
                }
                if matches!(&error, StreamError::ZeroOutputTransport(_)) {
                    zero_output_recovery_context =
                        Some(context_with_zero_output_transport_recovery(context));
                }
                let delay =
                    std::time::Duration::from_millis(if matches!(&error, StreamError::Empty) {
                        250
                    } else {
                        500
                    })
                    .saturating_mul(attempts);
                let remaining = PROVIDER_RECOVERY_MAX_ELAPSED.saturating_sub(started_at.elapsed());
                if delay >= remaining {
                    return Err(LoopError::Stream(error));
                }
                last_error = Some(error);
                tokio::select! {
                    _ = signal.cancelled() => return Err(LoopError::Aborted),
                    _ = tokio::time::sleep(delay) => {}
                }
                continue;
            }
            Err(error) => return Err(error),
        };
        let stop_reason = match &assistant {
            AgentMessage::Assistant { stop_reason, .. } => *stop_reason,
            _ => StopReason::Other,
        };
        if stop_reason != StopReason::MaxTokens {
            return Ok((assistant, allowlist));
        }
        // No starting cap means there's no number to scale from. Refuse
        // recovery rather than guess — the deployment hadn't pinned a
        // cap, so the truncation came from a provider-side limit we
        // don't know how to raise.
        let Some(prev_cap) = current_cap else {
            return Ok((assistant, allowlist));
        };
        let new_cap = prev_cap.saturating_mul(2);
        if new_cap <= prev_cap {
            return Ok((assistant, allowlist));
        }

        let error = StreamError::Fatal(
            "Provider repeatedly exhausted the output token limit before completing a response"
                .into(),
        );
        if attempts >= PROVIDER_RECOVERY_MAX_ATTEMPTS {
            return Err(LoopError::Stream(error));
        }
        recovery_started_at.get_or_insert_with(tokio::time::Instant::now);
        last_error = Some(error);
        max_tokens_attempt = max_tokens_attempt.saturating_add(1);
        emit(
            config,
            AgentEvent::OutputTokensEscalation {
                attempt: max_tokens_attempt,
                prev_cap,
                new_cap,
            },
        )
        .await;
        current_cap = Some(new_cap);
        // Discard the truncated `assistant` by simply not pushing it
        // into the caller's transcript. The MessageStart/MessageEnd
        // events for it already fired from the inner streamer; the
        // OutputTokensEscalation event above is the listener's signal
        // to roll the previous pair back from any projection.
    }
}

#[cfg(test)]
#[path = "provider_recovery_tests.rs"]
mod tests;
