//! The canonical agent loop.
//!
//! One free function each for run/start and continue — no god-class.
//!
//! Shape:
//!
//! ```text
//! agent_start
//!  └ loop:                          ← outer (follow-up) loop
//!     turn_start
//!     [pending steering messages]   ← injected before LLM call
//!     stream assistant response     ← StreamFn → AssistantMessage
//!     execute tool batch (if any)   ← parallel/sequential dispatch
//!     turn_end
//!     ↻ until no more tool calls AND no steering ready
//!     check follow-up               ← post-stop injection
//!  agent_end
//! ```
//!
//! Termination is unanimous-tool-vote: a batch ends the run only when
//! every finalized tool result sets `terminate = true`. One tool wanting
//! to stop does not stop the batch.

use futures::stream::StreamExt;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_util::sync::CancellationToken;

use crate::config::LoopConfig;
use crate::error::{LoopError, StreamError};
use crate::event::AgentEvent;
use crate::exec::{execute_tool_batch, ExecutedBatch};
use crate::plugin::TransformContext;
use crate::stream::{ReasoningEffort, StreamErrorKind, StreamEvent, StreamRequest, ToolSchema};
use crate::types::{
    AgentContext, AgentMessage, AssistantContent, StopReason, ToolResultContent, Usage,
};

const EMPTY_STREAM_RETRY_INITIAL_DELAY: std::time::Duration = std::time::Duration::from_millis(250);
const ZERO_OUTPUT_TRANSPORT_RETRY_INITIAL_DELAY: std::time::Duration =
    std::time::Duration::from_millis(500);
const PROVIDER_RETRY_MAX_DELAY: std::time::Duration = std::time::Duration::from_secs(30);
const ZERO_OUTPUT_TRANSPORT_RECOVERY_CONTEXT: &str = "\
[runtime context — transport recovery, not user instruction]\n\
The previous provider attempt produced no actionable output: no visible assistant text and no usable tool call reached the runtime. \
It may have produced private-only reasoning or an unusable burst of partial tool calls. \
Do not continue with private reasoning only. Re-read the latest observation and immediately choose exactly one next structured tool call; \
if the answer is ready, use the final response tool.";

/// Outcome label for a completed run.
///
/// A hard error becomes [`LoopError`]; caller cancellation is the only
/// external lifetime boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOutcome {
    /// Model emitted a final assistant turn with no tool calls and no
    /// pending steering. The natural happy path.
    Done,
    /// The configured iteration ceiling was reached while more model work was
    /// still pending. Earlier typed transcript events remain available, but
    /// this is not a complete answer.
    HitMaxIterations,
}

impl LoopOutcome {
    /// Whether this outcome implies a clean, non-partial final answer.
    pub fn is_complete(self) -> bool {
        matches!(self, Self::Done)
    }

    /// Short stable label suitable for logs and tool-result prefixes.
    pub fn label(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::HitMaxIterations => "hit_max_iterations",
        }
    }
}

/// Result of a completed run: emitted messages plus a typed outcome label.
///
/// Returned by [`run`] and [`run_continue`]. `messages` is the slice of
/// messages produced **during this run** (not the full transcript).
/// `outcome` records the natural close without requiring message inspection.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub messages: Vec<AgentMessage>,
    pub outcome: LoopOutcome,
}

/// Run the loop with one or more starting prompts.
///
/// The prompts are appended to the context's existing message list, then
/// the loop runs until natural stop (no more tool calls, no follow-up).
/// Returns the messages produced **during this run** plus a typed outcome
/// label — not the full transcript. Callers that want the full transcript
/// should fold prior messages into their own state, or read from the
/// event sink.
pub async fn run(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: &LoopConfig,
    signal: CancellationToken,
) -> Result<RunResult, LoopError> {
    let mut current = context;
    let mut new_messages = prompts.clone();

    current.messages.extend(prompts.iter().cloned());

    emit(config, AgentEvent::AgentStart).await;
    if let Some(identity) = current.identity.clone() {
        emit(config, AgentEvent::RunIdentified { identity }).await;
    }
    emit(config, AgentEvent::TurnStart).await;
    for prompt in &prompts {
        emit(
            config,
            AgentEvent::MessageStart {
                message: prompt.clone(),
            },
        )
        .await;
        emit(
            config,
            AgentEvent::MessageEnd {
                message: prompt.clone(),
            },
        )
        .await;
    }

    let outcome = inner_run(&mut current, &mut new_messages, config, &signal).await?;

    Ok(RunResult {
        messages: new_messages,
        outcome,
    })
}

/// Continue an existing context without adding a new prompt.
///
/// Used when the trailing message is already a `User` (e.g., steering
/// queued externally) or `ToolResult` (e.g., an out-of-band tool result
/// was injected). Errors if the trailing message is `Assistant` — the
/// model would not respond to its own message.
pub async fn run_continue(
    context: AgentContext,
    config: &LoopConfig,
    signal: CancellationToken,
) -> Result<RunResult, LoopError> {
    let last = context
        .messages
        .last()
        .ok_or_else(|| LoopError::InvalidContinuation("no messages in context".into()))?;
    if matches!(last, AgentMessage::Assistant { .. }) {
        return Err(LoopError::InvalidContinuation(
            "trailing message is assistant".into(),
        ));
    }

    let mut current = context;
    let mut new_messages = Vec::new();

    emit(config, AgentEvent::AgentStart).await;
    if let Some(identity) = current.identity.clone() {
        emit(config, AgentEvent::RunIdentified { identity }).await;
    }
    emit(config, AgentEvent::TurnStart).await;

    let outcome = inner_run(&mut current, &mut new_messages, config, &signal).await?;

    Ok(RunResult {
        messages: new_messages,
        outcome,
    })
}

// ─── Internals ─────────────────────────────────────────────────────

async fn emit(config: &LoopConfig, event: AgentEvent) {
    config.event_sink.emit(event.clone()).await;
    for observer in &config.plugins.event_observer {
        observer.on_event(&event).await;
    }
}

async fn inner_run(
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
            if config
                .max_iterations
                .is_some_and(|max| iterations >= max)
            {
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

async fn collect_steering(config: &LoopConfig) -> Vec<AgentMessage> {
    let mut out = Vec::new();
    for source in &config.plugins.steering {
        out.extend(source.next_steering_messages().await);
    }
    out
}

async fn collect_follow_up(config: &LoopConfig) -> Vec<AgentMessage> {
    let mut out = Vec::new();
    for source in &config.plugins.follow_up {
        out.extend(source.next_follow_up_messages().await);
    }
    out
}

fn synthesize_plain_text_terminal_result(
    assistant: &AgentMessage,
    turn_allowlist: Option<&std::collections::HashSet<String>>,
    tool_name: &str,
    eager: bool,
    terminal_tool_names: &std::collections::HashSet<String>,
) -> Option<AgentMessage> {
    // The default contract is "only convert plain text once the runtime
    // has narrowed the catalog to terminators" — preserves strict
    // delivery shape for everyone else. When `eager` is set the gate is
    // lifted: the host has signalled this provider can never honor
    // forced tool choice, so prose IS the failure mode and the nudge
    // cycle that normally narrows the allowlist would just burn turns.
    if !eager && !is_terminal_only_allowlist(turn_allowlist, tool_name, terminal_tool_names) {
        return None;
    }
    let text = plain_assistant_text(assistant)?;
    Some(AgentMessage::ToolResult {
        tool_call_id: format!("plain_text_terminal_fallback_{}", now_ms()),
        tool_name: tool_name.to_string(),
        content: ToolResultContent::text(text),
        is_error: false,
        narration: Some(
            "Converted plain assistant text into terminal delivery for an auto-tool-choice provider."
                .to_string(),
        ),
        details: None,
        timestamp: Some(now_ms()),
    })
}

fn plain_assistant_text(assistant: &AgentMessage) -> Option<String> {
    let AgentMessage::Assistant { content, .. } = assistant else {
        return None;
    };
    let text = crate::strip_thinking_tags(&content.plain_text())
        .trim()
        .to_string();
    (!text.is_empty()).then_some(text)
}

/// Whether a turn's allowlist has narrowed to "terminal only" — it
/// contains the configured fallback terminal tool and nothing but
/// terminal/delivery tools. The set of *other* names that count as
/// terminal comes from the active [`crate::protocol::ProtocolPolicy`]
/// ([`crate::protocol::ProtocolPolicy::terminal_tool_names`]); the core
/// hardcodes no product tool names. With the default policy (empty extra
/// set) an allowlist is terminal-only exactly when it contains only the
/// fallback tool itself.
fn is_terminal_only_allowlist(
    turn_allowlist: Option<&std::collections::HashSet<String>>,
    terminal_tool: &str,
    terminal_tool_names: &std::collections::HashSet<String>,
) -> bool {
    let Some(allowlist) = turn_allowlist else {
        return false;
    };
    !allowlist.is_empty()
        && allowlist.contains(terminal_tool)
        && allowlist
            .iter()
            .all(|tool| tool == terminal_tool || terminal_tool_names.contains(tool))
}

// ─── Stream one assistant response ─────────────────────────────────

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
async fn stream_with_overflow_recovery(
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

async fn stream_with_max_tokens_recovery(
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    iteration: usize,
) -> Result<(AgentMessage, Option<std::collections::HashSet<String>>), LoopError> {
    let mut current_cap = config.max_output_tokens;
    let mut max_tokens_attempt: u32 = 0;
    let mut empty_stream_attempts: u32 = 0;
    let mut zero_output_transport_attempts: u32 = 0;
    let mut transient_stream_attempts: u32 = 0;
    let mut zero_output_recovery_context: Option<AgentContext> = None;
    let mut reasoning = config.reasoning;

    loop {
        let attempt_context = zero_output_recovery_context.as_ref().unwrap_or(context);
        let (assistant, allowlist) = match stream_assistant_response(
            attempt_context,
            config,
            signal,
            iteration,
            current_cap,
            reasoning,
        )
        .await
        {
            Ok(pair) => pair,
            Err(LoopError::Stream(StreamError::Empty)) => {
                empty_stream_attempts = empty_stream_attempts.saturating_add(1);
                let delay =
                    provider_retry_delay(EMPTY_STREAM_RETRY_INITIAL_DELAY, empty_stream_attempts);
                tokio::select! {
                    _ = signal.cancelled() => return Err(LoopError::Aborted),
                    _ = tokio::time::sleep(delay) => {}
                }
                continue;
            }
            Err(LoopError::Stream(StreamError::ZeroOutputTransport(_))) => {
                zero_output_transport_attempts = zero_output_transport_attempts.saturating_add(1);
                zero_output_recovery_context =
                    Some(context_with_zero_output_transport_recovery(context));
                reasoning = zero_output_transport_retry_reasoning(config.reasoning);
                let delay = provider_retry_delay(
                    ZERO_OUTPUT_TRANSPORT_RETRY_INITIAL_DELAY,
                    zero_output_transport_attempts,
                );
                tokio::select! {
                    _ = signal.cancelled() => return Err(LoopError::Aborted),
                    _ = tokio::time::sleep(delay) => {}
                }
                continue;
            }
            Err(LoopError::Stream(
                StreamError::Transient(_) | StreamError::ProviderRateLimited(_),
            )) => {
                transient_stream_attempts = transient_stream_attempts.saturating_add(1);
                let delay = provider_retry_delay(
                    ZERO_OUTPUT_TRANSPORT_RETRY_INITIAL_DELAY,
                    transient_stream_attempts,
                );
                tokio::select! {
                    _ = signal.cancelled() => return Err(LoopError::Aborted),
                    _ = tokio::time::sleep(delay) => {}
                }
                continue;
            }
            Err(err) => return Err(err),
        };

        let stop_reason = match &assistant {
            AgentMessage::Assistant { stop_reason, .. } => *stop_reason,
            _ => StopReason::Other,
        };
        if stop_reason == StopReason::Aborted {
            return Err(LoopError::Aborted);
        }
        if stop_reason == StopReason::Error {
            transient_stream_attempts = transient_stream_attempts.saturating_add(1);
            let delay = provider_retry_delay(
                ZERO_OUTPUT_TRANSPORT_RETRY_INITIAL_DELAY,
                transient_stream_attempts,
            );
            tokio::select! {
                _ = signal.cancelled() => return Err(LoopError::Aborted),
                _ = tokio::time::sleep(delay) => {}
            }
            continue;
        }
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

fn provider_retry_delay(base: std::time::Duration, attempt: u32) -> std::time::Duration {
    base.saturating_mul(attempt.min(64))
        .min(PROVIDER_RETRY_MAX_DELAY)
}

async fn stream_assistant_response(
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
    let mut messages = context.messages.clone();
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
            StreamEvent::Done { message } => {
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
                let error_message = AgentMessage::Assistant {
                    content: match &partial {
                        AgentMessage::Assistant { content, .. } => content.clone(),
                        _ => AssistantContent { blocks: Vec::new() },
                    },
                    stop_reason,
                    error_message: Some(message.clone()),
                    timestamp: Some(now_ms()),
                    usage: None,
                };
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
    let empty = AgentMessage::Assistant {
        content: AssistantContent { blocks: Vec::new() },
        stop_reason: StopReason::Error,
        error_message: Some("stream ended without terminal event".into()),
        timestamp: Some(now_ms()),
        usage: None,
    };
    emit(
        config,
        AgentEvent::MessageEnd {
            message: empty.clone(),
        },
    )
    .await;
    Err(LoopError::Stream(StreamError::Empty))
}

fn context_with_zero_output_transport_recovery(context: &AgentContext) -> AgentContext {
    let mut recovered = context.clone();
    recovered.messages.push(AgentMessage::System {
        content: ZERO_OUTPUT_TRANSPORT_RECOVERY_CONTEXT.to_string(),
        timestamp: Some(now_ms()),
    });
    recovered
}

fn zero_output_transport_retry_reasoning(reasoning: ReasoningEffort) -> ReasoningEffort {
    match reasoning {
        ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::XHigh => {
            ReasoningEffort::Minimal
        }
        ReasoningEffort::None | ReasoningEffort::Minimal | ReasoningEffort::Low => reasoning,
    }
}

fn loop_error_from_stream_kind(kind: StreamErrorKind, message: String) -> LoopError {
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

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Walk back through `messages` and return the most recent provider
/// usage block reported on an assistant turn, if any. `None` on the
/// very first turn or when the active provider doesn't surface usage.
fn last_provider_usage(messages: &[AgentMessage]) -> Option<Usage> {
    messages.iter().rev().find_map(|message| match message {
        AgentMessage::Assistant {
            usage: Some(usage), ..
        } => Some(usage.clone()),
        _ => None,
    })
}

fn build_tool_schemas(
    config: &LoopConfig,
    allowlist: Option<&std::collections::HashSet<String>>,
) -> Vec<ToolSchema> {
    config
        .tools
        .iter()
        .filter(|tool| match allowlist {
            Some(set) => set.contains(tool.name()),
            None => true,
        })
        .map(|tool| ToolSchema {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.parameters_schema(),
        })
        .collect()
}

/// Poll every registered `ToolGate` plugin and intersect their
/// allowlists. Returns `None` when no plugin returned an allowlist
/// (the common case — no narrowing). Returns `Some(set)` when at
/// least one plugin is gating; multiple gates compose by intersection
/// unless their non-empty allowlists conflict to an empty set, in which
/// case the highest-priority gate wins and a typed conflict event is
/// emitted.
/// Resolve the per-turn tool allowlist by composing every registered
/// `ToolGate` plugin (intersection) and emit one
/// [`AgentEvent::ToolGateApplied`] per gate so observers can attribute
/// the final allowlist to specific plugins.
async fn collect_tool_allowlist_with_events(
    config: &LoopConfig,
    iteration: usize,
    messages: &[AgentMessage],
) -> Option<std::collections::HashSet<String>> {
    if config.plugins.tool_gate.is_empty() {
        return None;
    }
    let conversation_id = config.conversation_id.as_deref();
    let available_tool_names: Vec<&str> = config.tools.iter().map(|t| t.name()).collect();
    let mut decisions: Vec<GateAllowDecision> = Vec::new();
    for gate in &config.plugins.tool_gate {
        let ctx = crate::plugin::ToolGateContext {
            iteration,
            messages,
            conversation_id,
            available_tool_names: &available_tool_names,
        };
        let decision = gate.next_turn_tool_allowlist(ctx).await;
        emit(
            config,
            AgentEvent::ToolGateApplied {
                iteration,
                plugin: gate.name(),
                allow: decision.as_ref().map(|set| {
                    let mut sorted: Vec<String> = set.iter().cloned().collect();
                    sorted.sort();
                    sorted
                }),
            },
        )
        .await;
        if let Some(set) = decision {
            let suppresses_advisory =
                gate.suppresses_advisory_gates(crate::plugin::ToolGateContext {
                    iteration,
                    messages,
                    conversation_id,
                    available_tool_names: &available_tool_names,
                });
            decisions.push(GateAllowDecision {
                plugin: gate.name(),
                priority: gate.conflict_priority(),
                class: gate.tool_gate_class(),
                suppresses_advisory,
                allow: set,
            });
        }
    }
    let suppression_priority = decisions
        .iter()
        .filter(|decision| decision.suppresses_advisory)
        .map(|decision| decision.priority)
        .max();
    let active_decisions = decisions
        .iter()
        .filter(|decision| {
            !matches!(
                suppression_priority,
                Some(priority)
                    if decision.class == crate::plugin::ToolGateClass::Advisory
                        && decision.priority < priority
            )
        })
        .collect::<Vec<_>>();
    let mut combined: Option<std::collections::HashSet<String>> = None;
    for decision in &active_decisions {
        combined = Some(match combined {
            Some(prev) => prev.intersection(&decision.allow).cloned().collect(),
            None => decision.allow.clone(),
        });
    }
    if combined.as_ref().is_some_and(|allow| allow.is_empty()) {
        let non_empty_decisions = active_decisions
            .iter()
            .filter(|decision| !decision.allow.is_empty())
            .map(|decision| (decision.plugin, decision.priority, decision.allow.clone()))
            .collect::<Vec<_>>();
        let resolved = resolve_empty_tool_gate_intersection(&non_empty_decisions);
        let (chosen_plugin, allow, reason) = match resolved {
            Some((plugin, allow, reason)) => (Some(plugin.to_string()), allow, reason),
            None => (
                None,
                std::collections::HashSet::new(),
                "all gating plugins returned empty allowlists".to_string(),
            ),
        };
        let sorted_allow = sorted_tool_names(&allow);
        emit(
            config,
            AgentEvent::ToolGateConflictResolved {
                iteration,
                plugins: active_decisions
                    .iter()
                    .map(|decision| decision.plugin.to_string())
                    .collect(),
                chosen_plugin,
                allow: sorted_allow,
                reason,
            },
        )
        .await;
        return if allow.is_empty() { None } else { Some(allow) };
    }
    combined
}

struct GateAllowDecision {
    plugin: &'static str,
    priority: i32,
    class: crate::plugin::ToolGateClass,
    suppresses_advisory: bool,
    allow: std::collections::HashSet<String>,
}

fn resolve_empty_tool_gate_intersection(
    decisions: &[(&'static str, i32, std::collections::HashSet<String>)],
) -> Option<(&'static str, std::collections::HashSet<String>, String)> {
    decisions
        .iter()
        .max_by(|(left_plugin, left_priority, left), (right_plugin, right_priority, right)| {
            left_priority
                .cmp(right_priority)
                .then_with(|| right.len().cmp(&left.len()))
                .then_with(|| right_plugin.cmp(left_plugin))
        })
        .map(|(plugin, priority, allow)| {
            (
                *plugin,
                allow.clone(),
                format!(
                    "empty intersection repaired by highest-priority owner `{plugin}` (priority {priority})"
                ),
            )
        })
}

fn sorted_tool_names(set: &std::collections::HashSet<String>) -> Vec<String> {
    let mut sorted: Vec<String> = set.iter().cloned().collect();
    sorted.sort();
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentBuilder;
    use crate::plugin::{
        FollowUpSource, Plugin, PluginCapabilities, ToolGate, ToolGateClass, ToolGateContext,
    };
    use crate::stream::{ReasoningEffort, StreamFn};
    use crate::types::{AssistantBlock, UserContent};
    use futures::stream::{self, BoxStream};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    #[test]
    fn inconsistent_tool_history_stream_kind_stays_typed_at_loop_boundary() {
        let error = loop_error_from_stream_kind(
            StreamErrorKind::InconsistentToolHistory,
            "interleaved tool result batch".into(),
        );

        assert!(matches!(
            error,
            LoopError::Stream(StreamError::InconsistentToolHistory(message))
                if message == "interleaved tool result batch"
        ));
    }

    fn empty_assistant_message() -> AgentMessage {
        AgentMessage::Assistant {
            content: AssistantContent { blocks: Vec::new() },
            stop_reason: StopReason::Other,
            error_message: None,
            timestamp: None,
            usage: None,
        }
    }

    fn text_assistant_message(text: impl Into<String>) -> AgentMessage {
        AgentMessage::Assistant {
            content: AssistantContent::text(text),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: None,
            usage: None,
        }
    }

    fn tool_call_assistant_message(name: impl Into<String>, id: impl Into<String>) -> AgentMessage {
        AgentMessage::Assistant {
            content: AssistantContent::with_tool_calls(
                None,
                vec![crate::tool::ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: serde_json::json!({}),
                }],
            ),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: None,
            usage: None,
        }
    }

    struct EmptyThenTextStream {
        calls: AtomicUsize,
        failures_before_success: usize,
    }

    impl EmptyThenTextStream {
        fn new(failures_before_success: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                failures_before_success,
            }
        }
    }

    #[derive(Default)]
    struct ZeroOutputThenTextStream {
        calls: AtomicUsize,
        requests: Mutex<Vec<StreamRequest>>,
    }

    impl ZeroOutputThenTextStream {
        fn requests(&self) -> Vec<StreamRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[derive(Default)]
    struct RepeatedTextStream {
        calls: AtomicUsize,
    }

    struct PlainTextThenTerminatorStream {
        calls: AtomicUsize,
        plain_text_turns: usize,
    }

    struct TerminalOnlyGate;
    struct TerminalWithStatusGate;

    /// A product protocol policy that declares several delivery/status
    /// tools (beyond the configured fallback tool) as terminal, so an
    /// allowlist narrowed to `{message_info, message_result}` still
    /// classifies as terminal-only. The core ships none of these names;
    /// they live behind the policy.
    struct TestTerminalPolicy;
    impl crate::protocol::ProtocolPolicy for TestTerminalPolicy {
        fn terminal_tool_names(&self) -> std::collections::HashSet<String> {
            [
                "message_info",
                "message_ask",
                "message_result",
                "terminator",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect()
        }
    }
    struct StaticAllowGate {
        name: &'static str,
        tools: &'static [&'static str],
        priority: i32,
        class: ToolGateClass,
        suppresses_advisory: bool,
    }

    impl Plugin for TerminalOnlyGate {
        fn name(&self) -> &'static str {
            "terminal_only_gate"
        }

        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::tool_gate()
        }
    }

    #[async_trait::async_trait]
    impl ToolGate for TerminalOnlyGate {
        async fn next_turn_tool_allowlist(
            &self,
            _ctx: ToolGateContext<'_>,
        ) -> Option<std::collections::HashSet<String>> {
            Some(["message_result".to_string()].into_iter().collect())
        }
    }

    impl Plugin for TerminalWithStatusGate {
        fn name(&self) -> &'static str {
            "terminal_with_status_gate"
        }

        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::tool_gate()
        }
    }

    #[async_trait::async_trait]
    impl ToolGate for TerminalWithStatusGate {
        async fn next_turn_tool_allowlist(
            &self,
            _ctx: ToolGateContext<'_>,
        ) -> Option<std::collections::HashSet<String>> {
            Some(
                ["message_info".to_string(), "message_result".to_string()]
                    .into_iter()
                    .collect(),
            )
        }
    }

    impl Plugin for StaticAllowGate {
        fn name(&self) -> &'static str {
            self.name
        }

        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::tool_gate()
        }
    }

    #[async_trait::async_trait]
    impl ToolGate for StaticAllowGate {
        fn conflict_priority(&self) -> i32 {
            self.priority
        }

        fn tool_gate_class(&self) -> ToolGateClass {
            self.class
        }

        fn suppresses_advisory_gates(&self, _ctx: ToolGateContext<'_>) -> bool {
            self.suppresses_advisory
        }

        async fn next_turn_tool_allowlist(
            &self,
            _ctx: ToolGateContext<'_>,
        ) -> Option<std::collections::HashSet<String>> {
            Some(self.tools.iter().map(|name| (*name).to_string()).collect())
        }
    }

    impl PlainTextThenTerminatorStream {
        fn new(plain_text_turns: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                plain_text_turns,
            }
        }
    }

    #[async_trait::async_trait]
    impl StreamFn for EmptyThenTextStream {
        async fn stream(
            &self,
            _request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            if call < self.failures_before_success {
                return Box::pin(stream::iter(vec![
                    StreamEvent::Start {
                        partial: partial.clone(),
                    },
                    StreamEvent::Error {
                        partial,
                        kind: StreamErrorKind::Empty,
                        message: "empty provider response".to_string(),
                    },
                ]));
            }
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done {
                    message: text_assistant_message("recovered"),
                },
            ]))
        }
    }

    #[async_trait::async_trait]
    impl StreamFn for RepeatedTextStream {
        async fn stream(
            &self,
            _request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done {
                    message: text_assistant_message(format!("plain stop {call}")),
                },
            ]))
        }
    }

    #[async_trait::async_trait]
    impl StreamFn for PlainTextThenTerminatorStream {
        async fn stream(
            &self,
            _request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            let message = if call < self.plain_text_turns {
                text_assistant_message(format!("plain stop {call}"))
            } else if call == self.plain_text_turns {
                tool_call_assistant_message("terminator", "tc-terminator")
            } else {
                panic!("unexpected stream call after terminal turn: {call}")
            };
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done { message },
            ]))
        }
    }

    #[async_trait::async_trait]
    impl StreamFn for ZeroOutputThenTextStream {
        async fn stream(
            &self,
            request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            self.requests.lock().unwrap().push(request);
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            if call == 0 {
                return Box::pin(stream::iter(vec![
                    StreamEvent::Start {
                        partial: partial.clone(),
                    },
                    StreamEvent::Error {
                        partial,
                        kind: StreamErrorKind::ZeroOutputTransport,
                        message: "response body decode failed before output".to_string(),
                    },
                ]));
            }
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done {
                    message: text_assistant_message("recovered from transport"),
                },
            ]))
        }
    }

    /// Overflows on the first call, then (once the history is shrunk)
    /// returns text. Records each request so a test can assert what the
    /// retried call actually sent.
    struct OverflowThenTextStream {
        calls: AtomicUsize,
        requests: Mutex<Vec<StreamRequest>>,
        overflows_before_success: usize,
    }

    impl OverflowThenTextStream {
        fn new(overflows_before_success: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                requests: Mutex::new(Vec::new()),
                overflows_before_success,
            }
        }
    }

    impl Default for OverflowThenTextStream {
        fn default() -> Self {
            Self::new(1)
        }
    }

    #[async_trait::async_trait]
    impl StreamFn for OverflowThenTextStream {
        async fn stream(
            &self,
            request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            self.requests.lock().unwrap().push(request);
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            if call < self.overflows_before_success {
                return Box::pin(stream::iter(vec![
                    StreamEvent::Start {
                        partial: partial.clone(),
                    },
                    StreamEvent::Error {
                        partial,
                        kind: StreamErrorKind::ContextOverflow,
                        message: "maximum context length exceeded".to_string(),
                    },
                ]));
            }
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done {
                    message: text_assistant_message("recovered after shrink"),
                },
            ]))
        }
    }

    /// Recovery that drops the oldest message — enough to prove the loop
    /// persists each material shrink and can retry repeatedly.
    struct KeepLastRecovery {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::plugin::ContextOverflowRecovery for KeepLastRecovery {
        async fn recover(
            &self,
            mut messages: Vec<AgentMessage>,
            _cx: &TransformContext<'_>,
        ) -> Vec<AgentMessage> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !messages.is_empty() {
                messages.remove(0);
            }
            messages
        }
        fn name(&self) -> &'static str {
            "keep_last_recovery"
        }
    }

    #[tokio::test]
    async fn context_overflow_is_recovered_by_shrinking_and_retrying() {
        let stream = Arc::new(OverflowThenTextStream::default());
        let recovery_calls = Arc::new(AtomicUsize::new(0));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .overflow_recovery(KeepLastRecovery {
                calls: recovery_calls.clone(),
            })
            .build()
            .expect("config builds");
        let mut context = AgentContext::new("system").with_messages(vec![
            AgentMessage::User {
                content: UserContent::Text("first".to_string()),
                timestamp: None,
            },
            AgentMessage::User {
                content: UserContent::Text("keep me".to_string()),
                timestamp: None,
            },
        ]);

        let (assistant, _allowlist) =
            stream_with_overflow_recovery(&mut context, &config, &CancellationToken::new(), 0)
                .await
                .expect("overflow recovery should retry");

        let AgentMessage::Assistant { content, .. } = assistant else {
            panic!("expected assistant response");
        };
        assert_eq!(content.plain_text(), "recovered after shrink");
        assert_eq!(stream.calls.load(Ordering::SeqCst), 2, "one retry");
        assert_eq!(recovery_calls.load(Ordering::SeqCst), 1);
        // The shrink is persisted into the live transcript…
        assert_eq!(context.messages.len(), 1);
        // …and the retried request sent only the shrunk history.
        let retried = &stream.requests.lock().unwrap()[1];
        assert_eq!(retried.messages.len(), 1);
    }

    #[tokio::test]
    async fn context_overflow_recovery_outlives_the_legacy_two_attempt_cap() {
        let stream = Arc::new(OverflowThenTextStream::new(3));
        let recovery_calls = Arc::new(AtomicUsize::new(0));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .overflow_recovery(KeepLastRecovery {
                calls: recovery_calls.clone(),
            })
            .build()
            .expect("config builds");
        let mut context = AgentContext::new("system").with_messages(
            ["first", "second", "third", "keep me"]
                .into_iter()
                .map(|text| AgentMessage::User {
                    content: UserContent::Text(text.to_string()),
                    timestamp: None,
                })
                .collect(),
        );

        let (assistant, _allowlist) =
            stream_with_overflow_recovery(&mut context, &config, &CancellationToken::new(), 0)
                .await
                .expect("fourth stream attempt should recover");

        let AgentMessage::Assistant { content, .. } = assistant else {
            panic!("expected assistant response");
        };
        assert_eq!(content.plain_text(), "recovered after shrink");
        assert_eq!(stream.calls.load(Ordering::SeqCst), 4);
        assert_eq!(recovery_calls.load(Ordering::SeqCst), 3);
        assert_eq!(context.messages.len(), 1);
    }

    #[tokio::test]
    async fn context_overflow_without_recovery_propagates() {
        let stream = Arc::new(OverflowThenTextStream::default());
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .build()
            .expect("config builds");
        let mut context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
            content: UserContent::Text("hi".to_string()),
            timestamp: None,
        }]);

        let result =
            stream_with_overflow_recovery(&mut context, &config, &CancellationToken::new(), 0)
                .await;
        assert!(matches!(
            result,
            Err(LoopError::Stream(StreamError::ContextOverflow(_)))
        ));
        assert_eq!(stream.calls.load(Ordering::SeqCst), 1, "no retry");
    }

    #[test]
    fn done_is_complete() {
        assert!(LoopOutcome::Done.is_complete());
    }

    #[tokio::test(start_paused = true)]
    async fn empty_stream_recovery_outlives_the_legacy_three_attempt_cap() {
        let stream = Arc::new(EmptyThenTextStream::new(4));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .build()
            .expect("config builds");
        let context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
            content: UserContent::Text("continue".to_string()),
            timestamp: None,
        }]);

        let (assistant, _allowlist) =
            stream_with_max_tokens_recovery(&context, &config, &CancellationToken::new(), 0)
                .await
                .expect("fifth stream attempt should recover");

        let AgentMessage::Assistant { content, .. } = assistant else {
            panic!("expected assistant response");
        };
        assert_eq!(content.plain_text(), "recovered");
        assert_eq!(stream.calls.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn zero_output_transport_error_is_retried_before_returning() {
        let stream = Arc::new(ZeroOutputThenTextStream::default());
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .reasoning(ReasoningEffort::High)
            .build()
            .expect("config builds");
        let context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
            content: UserContent::Text("continue".to_string()),
            timestamp: None,
        }]);

        let (assistant, _allowlist) =
            stream_with_max_tokens_recovery(&context, &config, &CancellationToken::new(), 0)
                .await
                .expect("second zero-output transport attempt should recover");

        let AgentMessage::Assistant { content, .. } = assistant else {
            panic!("expected assistant response");
        };
        assert_eq!(content.plain_text(), "recovered from transport");
        assert_eq!(stream.calls.load(Ordering::SeqCst), 2);

        let requests = stream.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].reasoning, ReasoningEffort::High);
        assert_eq!(
            requests[1].reasoning,
            ReasoningEffort::Minimal,
            "zero-output replay should lower high reasoning so reasoning-heavy private-only spins can produce a tool call"
        );
        assert!(
            requests[1].messages.iter().any(|message| matches!(
                message,
                AgentMessage::System { content, .. }
                    if content.contains("transport recovery")
                        && content.contains("no visible assistant text")
                        && content.contains("no usable tool call")
                        && content.contains("unusable burst of partial tool calls")
                        && content.contains("exactly one next structured tool call")
                        && content.contains("next structured tool call")
            )),
            "zero-output replay must carry explicit recovery context"
        );
    }

    /// `StreamFn` that emits one assistant turn with a single
    /// `terminator` tool call, then panics on subsequent invocations
    /// — the test asserts the loop never re-enters the LLM.
    struct TerminatorOnlyStream {
        calls: AtomicUsize,
    }

    struct RepeatingToolStream {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl StreamFn for RepeatingToolStream {
        async fn stream(
            &self,
            _request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            let assistant = tool_call_assistant_message("continue", format!("tc-{call}"));
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done { message: assistant },
            ]))
        }
    }

    struct NonTerminatingTool;

    #[async_trait::async_trait]
    impl crate::tool::AgentTool for NonTerminatingTool {
        fn name(&self) -> &str {
            "continue"
        }

        fn description(&self) -> &str {
            "keep working"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(
            &self,
            _call_id: &str,
            _args: serde_json::Value,
            _signal: CancellationToken,
            _update: tokio::sync::mpsc::UnboundedSender<crate::tool::ToolResult>,
        ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
            Ok(crate::tool::ToolResult::text("continue"))
        }
    }

    #[tokio::test]
    async fn max_iterations_stops_a_nonterminating_tool_loop() {
        let stream = Arc::new(RepeatingToolStream {
            calls: AtomicUsize::new(0),
        });
        let tools = crate::tool::ToolRegistry::new().with(Arc::new(NonTerminatingTool));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .tools(tools)
            .max_iterations(3)
            .build()
            .expect("config builds");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("work".to_string()),
            timestamp: None,
        }];

        let result = run(
            prompts,
            AgentContext::new("system"),
            &config,
            CancellationToken::new(),
        )
        .await
        .expect("iteration cap is a typed outcome");

        assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
        assert_eq!(result.outcome, LoopOutcome::HitMaxIterations);
        assert!(!result.outcome.is_complete());
    }

    impl Default for TerminatorOnlyStream {
        fn default() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl StreamFn for TerminatorOnlyStream {
        async fn stream(
            &self,
            _request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(
                call, 0,
                "terminate-on-turn-1 test must NOT re-enter the LLM after a successful terminator"
            );
            let partial = empty_assistant_message();
            let assistant = AgentMessage::Assistant {
                content: AssistantContent {
                    blocks: vec![AssistantBlock::ToolCall(crate::tool::ToolCall {
                        id: "tc-terminator-1".into(),
                        name: "terminator".into(),
                        arguments: serde_json::json!({}),
                    })],
                },
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: None,
                usage: None,
            };
            Box::pin(stream::iter(vec![
                StreamEvent::Start { partial },
                StreamEvent::Done { message: assistant },
            ]))
        }
    }

    /// Tool that always votes `terminate=true`. Mirrors the contract a
    /// downstream terminal/delivery tool upholds.
    struct TerminatorTool;

    #[async_trait::async_trait]
    impl crate::tool::AgentTool for TerminatorTool {
        fn name(&self) -> &str {
            "terminator"
        }

        fn description(&self) -> &str {
            "test terminator"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(
            &self,
            _call_id: &str,
            _args: serde_json::Value,
            _signal: CancellationToken,
            _update: tokio::sync::mpsc::UnboundedSender<crate::tool::ToolResult>,
        ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
            Ok(crate::tool::ToolResult {
                content: vec![crate::types::ToolResultBlock::Text(
                    crate::types::TextContent {
                        text: "delivered".into(),
                    },
                )],
                is_error: false,
                details: serde_json::Value::Null,
                terminate: true,
                narration: None,
            })
        }
    }

    /// `SteeringSource` that always returns one wrap-up message. Used
    /// to prove the loop does NOT poll steering after a terminator
    /// vote (otherwise this would re-enter the LLM and trip the
    /// `assert_eq!(call, 0)` in `TerminatorOnlyStream`).
    struct AlwaysSteer {
        polls: Arc<AtomicUsize>,
    }

    impl Plugin for AlwaysSteer {
        fn name(&self) -> &'static str {
            "always_steer"
        }

        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities {
                steering: true,
                ..PluginCapabilities::default()
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::plugin::SteeringSource for AlwaysSteer {
        async fn next_steering_messages(&self) -> Vec<AgentMessage> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            vec![AgentMessage::System {
                content: "wrap up now".into(),
                timestamp: None,
            }]
        }
    }

    #[tokio::test]
    async fn terminator_vote_skips_post_batch_steering_collection() {
        // Regression: a `SteeringSource` whose firing condition lines
        // up with the same turn the model delivers (e.g.
        // `graceful_turn_limit` reaching its soft limit on the delivery
        // turn) used to re-enter the loop and prompt the model for
        // ANOTHER turn after a clean terminator. The model's drift on
        // that extra turn corrupted the user-visible answer in
        // production. With the fix, a unanimous terminator vote is a
        // hard exit — steering sources are not polled once the run has
        // decided it's done.
        let stream = Arc::new(TerminatorOnlyStream::default());
        let polls = Arc::new(AtomicUsize::new(0));
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminatorTool));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .tools(tool_registry)
            .steering(AlwaysSteer {
                polls: polls.clone(),
            })
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("deliver".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("run completes after one terminator turn");

        // Exactly one LLM call — the terminator turn.
        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        // Outcome is a clean natural completion.
        assert_eq!(result.outcome, LoopOutcome::Done);
        // Steering source is consulted exactly once — the pre-loop
        // priming poll at the top of `inner_run`. After the terminator
        // batch, `collect_steering` MUST NOT fire again.
        assert_eq!(
            polls.load(Ordering::SeqCst),
            1,
            "steering source polled more than once — terminator vote did not gate post-batch re-entry"
        );
    }

    /// `FollowUpSource` that always emits one nudge. Counts polls so
    /// the test can prove `collect_follow_up` is NOT invoked after a
    /// terminator batch.
    struct AlwaysFollowUp {
        polls: Arc<AtomicUsize>,
    }

    impl Plugin for AlwaysFollowUp {
        fn name(&self) -> &'static str {
            "always_follow_up"
        }

        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::follow_up()
        }
    }

    #[async_trait::async_trait]
    impl FollowUpSource for AlwaysFollowUp {
        async fn next_follow_up_messages(&self) -> Vec<AgentMessage> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            vec![AgentMessage::System {
                content: "deliver something".into(),
                timestamp: None,
            }]
        }
    }

    #[tokio::test]
    async fn terminator_vote_skips_post_batch_follow_up_collection() {
        // Mirror of the steering test for the follow-up source path.
        // `FollowUpSource` exists to nudge the model toward a
        // terminator when it failed to emit one — not to overrule a
        // terminator the model already cast. After a clean delivery,
        // follow-up must be silent.
        let stream = Arc::new(TerminatorOnlyStream::default());
        let polls = Arc::new(AtomicUsize::new(0));
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminatorTool));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .tools(tool_registry)
            .follow_up(AlwaysFollowUp {
                polls: polls.clone(),
            })
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("deliver".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("run completes after one terminator turn");

        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.outcome, LoopOutcome::Done);
        assert_eq!(
            polls.load(Ordering::SeqCst),
            0,
            "follow-up source polled after a terminator vote — terminator did not gate post-batch re-entry"
        );
    }

    #[tokio::test]
    async fn empty_tool_gate_intersection_prefers_delivery_repair_owner() {
        let (sink, mut rx) = crate::event::ChannelSink::new();
        let config = AgentBuilder::new()
            .stream(Arc::new(RepeatedTextStream::default()))
            .event_sink(Arc::new(sink))
            .tool_gate_arc(Arc::new(StaticAllowGate {
                name: "delivery_repair_gate",
                tools: &["browser_interact"],
                priority: 100,
                class: ToolGateClass::Required,
                suppresses_advisory: false,
            }))
            .tool_gate_arc(Arc::new(StaticAllowGate {
                name: "terminal_message_guard",
                tools: &["message_result"],
                priority: 10,
                class: ToolGateClass::Required,
                suppresses_advisory: false,
            }))
            .build()
            .expect("config builds");

        let allow = collect_tool_allowlist_with_events(&config, 3, &[])
            .await
            .expect("conflict repair should keep a non-empty allowlist");

        assert_eq!(
            allow,
            ["browser_interact".to_string()].into_iter().collect()
        );

        let mut saw_conflict = false;
        while let Ok(event) = rx.try_recv() {
            if let AgentEvent::ToolGateConflictResolved {
                chosen_plugin,
                allow,
                ..
            } = event
            {
                saw_conflict = true;
                assert_eq!(chosen_plugin.as_deref(), Some("delivery_repair_gate"));
                assert_eq!(allow, vec!["browser_interact".to_string()]);
            }
        }
        assert!(saw_conflict, "tool-gate deadlock should be diagnosable");
    }

    #[tokio::test]
    async fn repair_owner_suppresses_advisory_gate_before_plan_only_intersection() {
        let config = AgentBuilder::new()
            .stream(Arc::new(RepeatedTextStream::default()))
            .tool_gate_arc(Arc::new(StaticAllowGate {
                name: "delivery_repair_gate",
                tools: &["plan", "file_write"],
                priority: 100,
                class: ToolGateClass::Required,
                suppresses_advisory: true,
            }))
            .tool_gate_arc(Arc::new(StaticAllowGate {
                name: "wrap_up_gate",
                tools: &["plan", "message_result", "message_ask"],
                priority: 0,
                class: ToolGateClass::Advisory,
                suppresses_advisory: false,
            }))
            .build()
            .expect("config builds");

        let allow = collect_tool_allowlist_with_events(&config, 3, &[])
            .await
            .expect("repair owner should keep its own allowlist");

        assert_eq!(
            allow,
            ["plan".to_string(), "file_write".to_string()]
                .into_iter()
                .collect()
        );
    }

    #[tokio::test]
    async fn terminal_only_plain_text_fallback_synthesizes_terminal_result() {
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("auto-tool-provider")
            .tools(tool_registry)
            .tool_gate_arc(Arc::new(TerminalOnlyGate))
            .plain_text_terminal_fallback_tool("message_result")
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("answer directly".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("plain text should be converted on terminal-only turn");

        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.outcome, LoopOutcome::Done);
        assert!(result.messages.iter().any(|message| matches!(
            message,
            AgentMessage::ToolResult {
                tool_name,
                content,
                is_error: false,
                ..
            } if tool_name == "message_result"
                && content.plain_text() == "plain stop 0"
        )));
    }

    #[tokio::test]
    async fn eager_plain_text_fallback_fires_without_terminal_only_allowlist() {
        // Providers in the "auto-when-forced" class can never be
        // wire-forced into a tool call, so prose IS their failure mode.
        // The eager flag lifts the "allowlist must already be narrowed"
        // precondition so the fallback fires on the FIRST plain-text
        // stop instead of after a narrowing gate has burned 2-3 nudge
        // turns.
        //
        // No `tool_gate_arc` is installed in this test, so the catalog
        // stays at the full registry — exactly the situation where the
        // non-eager path would refuse to convert.
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("auto-tool-provider-eager")
            .tools(tool_registry)
            .plain_text_terminal_fallback_tool("message_result")
            .plain_text_terminal_fallback_eager(true)
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("answer directly".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("eager fallback should convert plain text on first stop");

        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.outcome, LoopOutcome::Done);
        assert!(result.messages.iter().any(|message| matches!(
            message,
            AgentMessage::ToolResult {
                tool_name,
                content,
                is_error: false,
                ..
            } if tool_name == "message_result"
                && content.plain_text() == "plain stop 0"
        )));
    }

    #[tokio::test]
    async fn eager_nudge_mode_has_no_implicit_retry_ceiling() {
        // With `plain_text_terminal_fallback_eager_nudge(true)` the eager
        // path nudges the model with a protocol-recovery system message
        // on every consecutive plain-text stop. Four failures exceed the
        // deleted two-nudge cap; the fifth turn supplies the real terminal
        // tool call rather than relying on synthesized completion.
        let stream = Arc::new(PlainTextThenTerminatorStream::new(4));
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry
            .with(Arc::new(TerminalNamedTool("message_result")))
            .with(Arc::new(TerminatorTool));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("auto-tool-provider-eager-nudge")
            .tools(tool_registry)
            .plain_text_terminal_fallback_tool("message_result")
            .plain_text_terminal_fallback_eager(true)
            .plain_text_terminal_fallback_eager_nudge(true)
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("answer directly".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("nudge mode should wait for a real terminal tool call");

        assert_eq!(stream.calls.load(Ordering::SeqCst), 5);
        assert_eq!(result.outcome, LoopOutcome::Done);

        let nudge_count = result
            .messages
            .iter()
            .filter(|m| matches!(m, AgentMessage::System { content, .. } if content == crate::protocol::DEFAULT_PLAIN_TEXT_RECOVERY_PROMPT))
            .count();
        assert_eq!(
            nudge_count, 4,
            "expected one protocol-recovery message per plain-text turn, got {nudge_count}",
        );
    }

    #[tokio::test]
    async fn non_eager_plain_text_fallback_still_requires_narrowed_allowlist() {
        // Default behaviour preserved: when eager is NOT set and the
        // turn allowlist is the full catalog, plain text is NOT
        // converted. It remains the run's natural assistant completion.
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("non-eager-provider")
            .tools(tool_registry)
            .plain_text_terminal_fallback_tool("message_result")
            // Eager NOT set → defaults to false.
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("answer directly".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("plain assistant completion should remain valid");

        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        assert!(!result.messages.iter().any(|message| matches!(
            message,
            AgentMessage::ToolResult { tool_name, .. } if tool_name == "message_result"
        )));
    }

    #[tokio::test]
    async fn terminal_plain_text_fallback_allows_status_delivery_gate() {
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("auto-tool-provider")
            .tools(tool_registry)
            .protocol_policy(Arc::new(TestTerminalPolicy))
            .tool_gate_arc(Arc::new(TerminalWithStatusGate))
            .plain_text_terminal_fallback_tool("message_result")
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("answer directly".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect(
                "plain text should be converted when only status and terminal tools are allowed",
            );

        assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.outcome, LoopOutcome::Done);
        assert!(result.messages.iter().any(|message| matches!(
            message,
            AgentMessage::ToolResult {
                tool_name,
                content,
                is_error: false,
                ..
            } if tool_name == "message_result"
                && content.plain_text() == "plain stop 0"
        )));
    }

    struct TerminalNamedTool(&'static str);

    #[async_trait::async_trait]
    impl crate::tool::AgentTool for TerminalNamedTool {
        fn name(&self) -> &str {
            self.0
        }

        fn description(&self) -> &str {
            "test terminal tool"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(
            &self,
            _call_id: &str,
            _args: serde_json::Value,
            _signal: CancellationToken,
            _update: tokio::sync::mpsc::UnboundedSender<crate::tool::ToolResult>,
        ) -> Result<crate::tool::ToolResult, crate::error::ToolError> {
            Ok(crate::tool::ToolResult {
                content: vec![crate::types::ToolResultBlock::Text(
                    crate::types::TextContent {
                        text: "not used".into(),
                    },
                )],
                is_error: false,
                details: serde_json::Value::Null,
                terminate: true,
                narration: None,
            })
        }
    }
}
