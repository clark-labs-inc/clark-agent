//! Provider streaming and recovery in the canonical loop.
use super::*;

/// Wrap [`stream_assistant_response`] with the configured max-output-
/// tokens recovery ladder. When recovery is disabled (the default), this
/// reduces to a single call. When enabled, a `StopReason::MaxTokens`
/// turn is discarded and the next attempt re-streams with a larger
/// cap until the ladder runs out or the model produces a non-truncated
/// turn.
///
/// Discarded turns *do* fire `MessageStart`/`MessageEnd` from the
/// inner streamer — listeners that care must correlate via the
/// `OutputTokensEscalation` event that this wrapper emits before each
/// retry. Persistence layers should treat the message that immediately
/// precedes an `OutputTokensEscalation` as overridden by the next
/// `MessageEnd`.
/// Wraps [`stream_with_max_tokens_recovery`] with context-overflow
/// recovery: when the provider rejects the request for exceeding its
/// window ([`StreamError::ContextOverflow`]) and an overflow-recovery
/// hook is installed, shrink `current.messages` in place (persisting it
/// so later turns don't re-expand), emit the diff event, and retry the
/// same LLM call. With no hook, the overflow propagates unchanged. A
/// recovery that fails to shrink also stops the loop rather than spinning.
pub(super) async fn stream_with_overflow_recovery(
    current: &mut AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    iteration: usize,
) -> Result<(AgentMessage, Option<std::collections::HashSet<String>>), LoopError> {
    loop {
        match stream_with_max_tokens_recovery(current, config, signal, iteration).await {
            Err(LoopError::Stream(StreamError::ContextOverflow(message))) => {
                let Some(recovery) = config.overflow_recovery.clone() else {
                    return Err(LoopError::Stream(StreamError::ContextOverflow(message)));
                };
                if signal.is_cancelled() {
                    return Err(LoopError::Stream(StreamError::ContextOverflow(message)));
                }

                // Compute the observables before taking the history, so the
                // borrow doesn't collide with the `mem::take` below.
                let usage = last_provider_usage(&current.messages);
                let cx = TransformContext {
                    signal,
                    model_id: config.model_id.as_deref().unwrap_or(""),
                    iteration,
                    last_provider_usage: usage.as_ref(),
                    estimator: &*config.token_estimator,
                };
                let before = std::mem::take(&mut current.messages);
                let before_size = cx.estimator.estimate_messages(&before);
                let after = recovery.recover(before.clone(), &cx).await;

                // No-progress guard: a recovery that didn't actually SHRINK the
                // history (measured in estimated tokens, not message count —
                // compaction can trade many messages for a summary + tail
                // without reducing the count) would just overflow again.
                // Surface the overflow instead of retrying forever.
                if cx.estimator.estimate_messages(&after) >= before_size {
                    current.messages = before;
                    return Err(LoopError::Stream(StreamError::ContextOverflow(message)));
                }
                emit(
                    config,
                    AgentEvent::ContextTransformApplied {
                        iteration,
                        plugin: recovery.name(),
                        before,
                        after: after.clone(),
                    },
                )
                .await;
                current.messages = after;
                // Retry the same LLM call against the shrunk history.
            }
            other => return other,
        }
    }
}

pub(super) async fn stream_assistant_response(
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    iteration: usize,
    max_output_tokens: Option<u32>,
    reasoning: ReasoningEffort,
) -> Result<(AgentMessage, Option<std::collections::HashSet<String>>), LoopError> {
    // Apply context transforms in registration order. The
    // `TransformContext` carries the cancellation signal plus a few
    // cheap observables (model id, iteration, last-turn provider
    // usage, token estimator) so each transform can decide locally
    // without the loop widening the trait per-knob.
    let last_provider_usage = last_provider_usage(&context.messages);
    let cx = TransformContext {
        signal,
        model_id: config.model_id.as_deref().unwrap_or(""),
        iteration,
        last_provider_usage: last_provider_usage.as_ref(),
        estimator: &*config.token_estimator,
    };
    let mut messages = crate::context_observations::expand(context.messages.clone());
    // Each transform's diff is observable so post-mortems can attribute
    // a specific compaction (shrinker, microcompactor, history-repair,
    // …) to the missing slice the model went on to misuse. Cloning is
    // cheap relative to the actual transform work, and the eval-side
    // observer is the one consumer that wants this much detail; other
    // sinks ignore the variant.
    for transform in &config.plugins.context_transform {
        // Cheap pre-check: plugins that can locally decide they have
        // nothing to do (no browser snapshots, history under budget, …)
        // skip the clone + diff-event entirely. Default impl returns
        // `true`, so plugins that haven't opted in still run on every
        // round.
        if !transform.should_run(&messages, &cx) {
            continue;
        }
        let before = messages.clone();
        messages = transform.transform(messages, &cx).await;
        emit(
            config,
            AgentEvent::ContextTransformApplied {
                iteration,
                plugin: transform.name(),
                before,
                after: messages.clone(),
            },
        )
        .await;
    }

    // Consult any registered ToolGate plugins for a per-turn allowlist.
    // Each plugin returns `Some(set)` to narrow the advertised tools for
    // exactly this LLM call. Multiple plugins compose by intersection;
    // `None` plugins do not constrain. See `ToolGate` docs for rationale.
    let allowlist = collect_tool_allowlist_with_events(config, iteration, &messages).await;

    let tools = build_tool_schemas(config, allowlist.as_ref());
    // Final snapshot of what the loop is about to send, captured after
    // every transform/gate. Observers (eval per-turn dump, debugger,
    // replay) take this as the source of truth for "what did the
    // model see this turn?".
    emit(
        config,
        AgentEvent::ProviderRequestPrepared {
            iteration,
            model_id: config.model_id.clone(),
            system_prompt: context.system_prompt.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
            temperature: config.temperature,
            max_output_tokens,
        },
    )
    .await;
    let observations = config.plugins.context_transform.iter()
        .flat_map(|transform| transform.history_observations(&messages))
        .collect::<Vec<_>>();
    let request = StreamRequest {
        system_prompt: context.system_prompt.clone(),
        messages,
        tools,
        temperature: config.temperature,
        max_output_tokens,
        reasoning,
        provider_extras: config
            .provider_extras
            .clone()
            .unwrap_or(serde_json::Value::Null),
        // `tool_choice: "required"` on every turn. The LLM-in-charge
        // contract is "context → LLM → tool call → append result →
        // repeat" — the model's job is to pick a tool, not emit
        // narration. This assumes the catalog includes a terminal
        // text-delivery tool, so required-on-every-turn doesn't trap the
        // model: when the work is done it calls that delivery tool to
        // return the answer. If the model loops on verification instead,
        // the bug is in the catalog or prompt — not in the requirement.
        force_tool_call: true,
    };

    let mut stream = config.stream.stream(request, signal.clone()).await;

    let mut last_partial: Option<AgentMessage> = None;

    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Start { partial } => {
                emit(
                    config,
                    AgentEvent::MessageStart {
                        message: partial.clone(),
                    },
                )
                .await;
                last_partial = Some(partial);
            }
            StreamEvent::Chunk(chunk) => {
                if let Some(ref partial) = last_partial {
                    emit(
                        config,
                        AgentEvent::MessageUpdate {
                            partial: partial.clone(),
                            chunk,
                        },
                    )
                    .await;
                }
            }
            StreamEvent::Done { mut message } => {
                crate::context_observations::attach(&mut message, observations);
                emit(
                    config,
                    AgentEvent::MessageEnd {
                        message: message.clone(),
                    },
                )
                .await;
                return Ok((message, allowlist));
            }
            StreamEvent::Error {
                partial,
                kind,
                message,
            } => {
                let stop_reason = match kind {
                    StreamErrorKind::Aborted => StopReason::Aborted,
                    _ => StopReason::Error,
                };
                let mut error_message = AgentMessage::Assistant {
                    content: match &partial {
                        AgentMessage::Assistant { content, .. } => content.clone(),
                        _ => AssistantContent { blocks: Vec::new() },
                    },
                    stop_reason,
                    error_message: Some(message.clone()),
                    timestamp: Some(now_ms()),
                    usage: None,
                };
                crate::context_observations::attach(&mut error_message, observations);
                emit(
                    config,
                    AgentEvent::MessageEnd {
                        message: error_message.clone(),
                    },
                )
                .await;
                return Err(loop_error_from_stream_kind(kind, message));
            }
        }
    }

    // Stream ended without `Done` or `Error`. Synthesize an empty
    // assistant message so the loop can recover.
    let mut empty = AgentMessage::Assistant {
        content: AssistantContent { blocks: Vec::new() },
        stop_reason: StopReason::Error,
        error_message: Some("stream ended without terminal event".into()),
        timestamp: Some(now_ms()),
        usage: None,
    };
    crate::context_observations::attach(&mut empty, observations);
    emit(
        config,
        AgentEvent::MessageEnd {
            message: empty.clone(),
        },
    )
    .await;
    Err(LoopError::Stream(StreamError::Empty))
}

pub(super) fn loop_error_from_stream_kind(kind: StreamErrorKind, message: String) -> LoopError {
    // StreamFn implementations own transport retries. Once an error
    // reaches the loop, it is the terminal outcome of that provider
    // attempt and must not be reclassified as a successful assistant
    // turn.
    match kind {
        StreamErrorKind::Transient => LoopError::Stream(StreamError::Transient(message)),
        StreamErrorKind::ProviderRateLimited => {
            LoopError::Stream(StreamError::ProviderRateLimited(message))
        }
        StreamErrorKind::ZeroOutputTransport => {
            LoopError::Stream(StreamError::ZeroOutputTransport(message))
        }
        StreamErrorKind::Fatal => LoopError::Stream(StreamError::Fatal(message)),
        StreamErrorKind::InconsistentToolHistory => {
            LoopError::Stream(StreamError::InconsistentToolHistory(message))
        }
        StreamErrorKind::Empty => LoopError::Stream(StreamError::Empty),
        StreamErrorKind::Aborted => LoopError::Aborted,
        StreamErrorKind::ContextOverflow => {
            LoopError::Stream(StreamError::ContextOverflow(message))
        }
    }
}

