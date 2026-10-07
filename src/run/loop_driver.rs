use super::*;

pub(super) async fn inner_run(
    current: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &LoopConfig,
    signal: &CancellationToken,
) -> Result<LoopOutcome, LoopError> {
    let mut first_turn = true;
    let mut iterations: usize = 0;
    let mut hit_max_iterations = false;

    // Steering messages may already be queued (caller produced them
    // before calling `run`).
    let mut pending = collect_steering(config).await;

    'outer: loop {
        let mut has_more_tool_calls = true;
        // Did the most recent tool batch vote terminate? Reset per
        // outer iteration so a follow-up-driven re-entry starts clean.
        //
        // When the batch produces a unanimous terminator (every
        // finalized result votes `terminate = true`), the run is over —
        // `SteeringSource` and `FollowUpSource` plugins must NOT
        // re-prompt the model with another LLM call. Without this
        // guard a steering source whose firing condition lined up
        // with the same turn (e.g. `graceful_turn_limit` reaching
        // its soft limit on the same turn the model delivered)
        // would inject a wrap-up message and the loop would burn
        // another turn after a clean delivery — observed in production,
        // where a model drifted into hallucinated content on the
        // wrap-up re-entry after the prior batch had already produced
        // the correct terminal delivery.
        let mut last_batch_terminated = false;

        while has_more_tool_calls || !pending.is_empty() {
            if signal.is_cancelled() {
                return Err(LoopError::Aborted);
            }
            if config.max_iterations.is_some_and(|max| iterations >= max) {
                hit_max_iterations = true;
                break;
            }
            iterations += 1;

            if !first_turn {
                emit(config, AgentEvent::TurnStart).await;
            } else {
                first_turn = false;
            }

            // Inject any pending steering messages before the next LLM call.
            if !pending.is_empty() {
                for msg in pending.drain(..) {
                    emit(
                        config,
                        AgentEvent::MessageStart {
                            message: msg.clone(),
                        },
                    )
                    .await;
                    emit(
                        config,
                        AgentEvent::MessageEnd {
                            message: msg.clone(),
                        },
                    )
                    .await;
                    current.messages.push(msg.clone());
                    new_messages.push(msg);
                }
            }

            for guard in &config.plugins.run_guard {
                guard.check(crate::plugin::RunGuardContext {
                    messages: &current.messages,
                    iteration: iterations - 1,
                })?;
            }

            // Stream one assistant response, applying the configured
            // max-tokens recovery ladder if a turn comes back truncated,
            // and the context-overflow recovery hook if the request is
            // rejected for exceeding the model's window. `iteration` is
            // 0-indexed and counts LLM calls within this run — `iterations`
            // was already incremented above for cap-checking, so the
            // 0-indexed turn number is `iterations - 1`.
            let (assistant, turn_allowlist) =
                stream_with_overflow_recovery(current, config, signal, iterations - 1).await?;
            // The assistant message must land in *both* the live conversation
            // (so the next turn's request body includes it — providers reject
            // tool messages that don't follow a matching assistant tool_call)
            // and the run's emitted-messages tail.
            current.messages.push(assistant.clone());
            new_messages.push(assistant.clone());

            // Extract tool calls.
            let tool_calls: Vec<_> = match &assistant {
                AgentMessage::Assistant { content, .. } => {
                    content.tool_calls().into_iter().cloned().collect()
                }
                _ => Vec::new(),
            };

            let mut tool_result_messages = Vec::new();
            has_more_tool_calls = false;

            if tool_calls.is_empty() {
                if let Some(tool_name) = config.plain_text_terminal_fallback_tool.as_deref() {
                    let eager = config.plain_text_terminal_fallback_eager;
                    let terminal_tool_names = config.protocol.terminal_tool_names();
                    let narrowed_to_terminators = is_terminal_only_allowlist(
                        turn_allowlist.as_ref(),
                        tool_name,
                        &terminal_tool_names,
                    );
                    let nudge_mode = config.plain_text_terminal_fallback_eager_nudge
                        && eager
                        && !narrowed_to_terminators;
                    if nudge_mode {
                        // Catalog still contains real work tools (e.g. `plan`)
                        // but the model emitted prose. Inject an explicit
                        // protocol-recovery system message and force the
                        // inner loop to re-stream rather than laundering
                        // the prose into a synthetic `message_result`.
                        // Push directly into `current.messages` (mirrors the
                        // synthesize path) rather than `pending`, which is
                        // overwritten by `collect_steering` at end-of-iter.
                        // Set `has_more_tool_calls = true` to satisfy the
                        // inner while-loop's continuation predicate.
                        //
                        // The recovery prose comes from the active
                        // `ProtocolPolicy` (which may name the product's
                        // delivery / ask tools); the core falls back to a
                        // generic, vocabulary-free nudge.
                        let available_tool_names: Vec<&str> =
                            config.tools.iter().map(|t| t.name()).collect();
                        let nudge_text = config
                            .protocol
                            .plain_text_recovery_prompt(crate::protocol::PlainTextRecoveryContext {
                                messages: &current.messages,
                                iteration: iterations - 1,
                                available_tool_names: &available_tool_names,
                                terminal_fallback_tool: Some(tool_name),
                            })
                            .unwrap_or_else(|| {
                                crate::protocol::DEFAULT_PLAIN_TEXT_RECOVERY_PROMPT.to_string()
                            });
                        let nudge = AgentMessage::System {
                            content: nudge_text,
                            timestamp: Some(now_ms()),
                        };
                        current.messages.push(nudge.clone());
                        new_messages.push(nudge);
                        has_more_tool_calls = true;
                    } else if let Some(result_msg) = synthesize_plain_text_terminal_result(
                        &assistant,
                        turn_allowlist.as_ref(),
                        tool_name,
                        eager,
                        &terminal_tool_names,
                    ) {
                        last_batch_terminated = true;
                        current.messages.push(result_msg.clone());
                        new_messages.push(result_msg.clone());
                        tool_result_messages.push(result_msg);
                    }
                }
            } else {
                let ExecutedBatch {
                    messages,
                    terminate,
                } = execute_tool_batch(
                    &assistant,
                    tool_calls,
                    current,
                    config,
                    signal,
                    turn_allowlist.as_ref(),
                )
                .await?;

                tool_result_messages = messages;
                has_more_tool_calls = !terminate;
                last_batch_terminated = terminate;

                for result_msg in &tool_result_messages {
                    current.messages.push(result_msg.clone());
                    new_messages.push(result_msg.clone());
                }
            }

            emit(
                config,
                AgentEvent::TurnEnd {
                    message: assistant,
                    tool_results: tool_result_messages,
                },
            )
            .await;

            // Drain any new steering messages that arrived during the
            // turn — except when the batch just emitted a unanimous
            // terminator. A clean terminator vote is the model's
            // "we're done" signal; further steering would re-prompt
            // past the delivery and let the model drift.
            pending = if last_batch_terminated {
                Vec::new()
            } else {
                collect_steering(config).await
            };
        }

        // The model produced no tool calls and no steering is queued. Give
        // follow-up sources one last chance to inject another turn.
        // Skip the follow-up source pass when the last batch
        // terminated for the same reason steering is skipped above:
        // a clean terminator vote means the run is done; follow-up
        // sources exist to nudge the model toward a terminator when
        // it failed to emit one, not to overrule one it already cast.
        let follow_up = if last_batch_terminated || hit_max_iterations {
            Vec::new()
        } else {
            collect_follow_up(config).await
        };
        if !follow_up.is_empty() {
            pending = follow_up;
            continue 'outer;
        }

        if hit_max_iterations {
            break 'outer;
        }

        break;
    }

    emit(
        config,
        AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        },
    )
    .await;

    Ok(if hit_max_iterations {
        LoopOutcome::HitMaxIterations
    } else {
        LoopOutcome::Done
    })
}

