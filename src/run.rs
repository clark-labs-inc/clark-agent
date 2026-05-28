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

const EMPTY_STREAM_MAX_ATTEMPTS: u8 = 3;
const EMPTY_STREAM_RETRY_INITIAL_DELAY: std::time::Duration = std::time::Duration::from_millis(250);
const ZERO_OUTPUT_TRANSPORT_MAX_ATTEMPTS: u8 = 2;
const ZERO_OUTPUT_TRANSPORT_RETRY_INITIAL_DELAY: std::time::Duration =
    std::time::Duration::from_millis(500);
const ZERO_OUTPUT_TRANSPORT_RECOVERY_CONTEXT: &str = "\
[runtime context — transport recovery, not user instruction]\n\
The previous provider attempt produced no actionable output: no visible assistant text and no usable tool call reached the runtime. \
It may have produced private-only reasoning or an unusable burst of partial tool calls. \
Do not continue with private reasoning only. Re-read the latest observation and immediately choose exactly one next structured tool call; \
if the answer is ready, use the final response tool.";

/// Hard cap on consecutive plain-text-fallback nudges before the loop
/// falls back to synthesizing a terminal tool result as a last resort.
/// Two nudges plus one synthesize keeps the recovery window bounded
/// without leaning on a caller-configured `empty_outcome_retry_budget`.
const MAX_PLAIN_TEXT_NUDGE_RETRIES: usize = 2;

const PLAIN_TEXT_NUDGE_CONTEXT: &str = "\
[runtime context — protocol recovery, not user instruction]\n\
Your previous response was plain text with no tool call. The runtime cannot deliver, ask, or work on prose — every turn MUST select exactly one structured tool call.\n\
\n\
The user's intent is clear from their message. Do not ask \"Would you like me to proceed?\", \"Shall I continue?\", or \"Do you need credentials?\" — those questions are friction. Make the obvious decision and execute.\n\
\n\
Pick exactly one tool now and call it:\n\
- `message_result` — final delivery (full answer text, or a partial result naming a concrete blocker).\n\
- `message_ask` — ONLY for user-owned input the user did not provide (real credential, personal preference, destination for their data).\n\
- `plan` — set phases for multi-step work (required on turn 1 for non-trivial tasks).\n\
- Any other catalog tool when real work needs to happen first.\n\
\n\
Re-read the user's request and call a tool. Do not type a clarifying question.";

/// Outcome label for a completed run.
///
/// Distinguishes natural termination from budget-pressure terminations so
/// callers (notably parent agents reading a subagent's tool result) can
/// reason about whether the answer is complete or partial. All variants
/// are non-error — a hard error becomes [`LoopError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOutcome {
    /// Model emitted a final assistant turn with no tool calls and no
    /// pending steering. The natural happy path.
    Done,
    /// The graceful turn-limit plugin injected a wrap-up steering message
    /// and the model produced a clean final turn within the grace window.
    /// Result text reflects the model's deliberate close-out, not a
    /// truncated transcript.
    WrappedUp,
    /// `max_iterations` was reached before the model wrapped up. The run
    /// stopped at the cap, but earlier turns are still in the transcript.
    /// The most recent assistant turn may have had pending tool calls.
    HitMaxIterations,
}

impl LoopOutcome {
    /// Whether this outcome implies a clean, non-partial final answer.
    pub fn is_complete(self) -> bool {
        matches!(self, LoopOutcome::Done | LoopOutcome::WrappedUp)
    }

    /// Short stable label suitable for logs and tool-result prefixes.
    pub fn label(self) -> &'static str {
        match self {
            LoopOutcome::Done => "done",
            LoopOutcome::WrappedUp => "wrapped_up",
            LoopOutcome::HitMaxIterations => "hit_max_iterations",
        }
    }
}

/// Result of a completed run: emitted messages plus a typed outcome label.
///
/// Returned by [`run`] and [`run_continue`]. `messages` is the slice of
/// messages produced **during this run** (not the full transcript).
/// `outcome` lets callers distinguish a natural close from a budget-driven
/// one without inspecting message content.
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
    let mut empty_outcomes_seen: usize = 0;
    let mut last_turn_stopped_without_tool = false;
    let mut plain_text_terminal_fallback_candidate: Option<AgentMessage> = None;

    // Steering messages may already be queued (caller produced them
    // before calling `run`).
    let mut pending = collect_steering(config).await;

    'outer: loop {
        let mut has_more_tool_calls = true;
        // Did the most recent tool batch vote terminate? Reset per
        // outer iteration so a follow-up-driven re-entry starts clean.
        //
        // When the model produces a unanimous terminator (e.g.
        // `message_result.terminate=true`), the run is over —
        // `SteeringSource` and `FollowUpSource` plugins must NOT
        // re-prompt the model with another LLM call. Without this
        // guard a steering source whose firing condition lined up
        // with the same turn (e.g. `graceful_turn_limit` reaching
        // its soft limit on the same turn the model delivered)
        // would inject a wrap-up message and the loop would burn
        // another turn after a clean delivery — observed on
        // `working-checkpoint-store-and-recall` in matrix run
        // 20260507_100311, where `gemini-3-flash-preview` drifted
        // into hallucinated content on the wrap-up re-entry after
        // the prior batch had already produced the correct
        // `message_result`.
        let mut last_batch_terminated = false;

        while has_more_tool_calls || !pending.is_empty() {
            if signal.is_cancelled() {
                return Err(LoopError::Aborted);
            }
            if let Some(max) = config.max_iterations {
                if iterations >= max {
                    // Hit the iteration cap. Break out of the inner
                    // loop so the follow-up sources get one last
                    // chance to inject a terminator nudge before the
                    // run ends. The outer loop's own cap-check (added
                    // below) ensures we don't loop forever.
                    break;
                }
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
            // max-tokens recovery ladder if a turn comes back
            // truncated. `iteration` is 0-indexed and counts LLM calls
            // within this run — `iterations` was already incremented
            // above for cap-checking, so the 0-indexed turn number is
            // `iterations - 1`.
            let (assistant, turn_allowlist) =
                stream_with_max_tokens_recovery(current, config, signal, iterations - 1).await?;
            // The assistant message must land in *both* the live conversation
            // (so the next turn's request body includes it — providers reject
            // tool messages that don't follow a matching assistant tool_call)
            // and the run's emitted-messages tail.
            current.messages.push(assistant.clone());
            new_messages.push(assistant.clone());

            // Stop on stream-level error/abort. Well-behaved
            // transports surface these as `StreamEvent::Error`, which
            // `stream_assistant_response` converts to `LoopError`
            // before returning. Keep this branch as a guard for
            // transports that incorrectly finalize a `Done` message
            // with an error stop reason.
            let stop_reason = match &assistant {
                AgentMessage::Assistant { stop_reason, .. } => *stop_reason,
                _ => StopReason::Other,
            };
            if matches!(stop_reason, StopReason::Error | StopReason::Aborted) {
                let loop_error = match &assistant {
                    AgentMessage::Assistant {
                        stop_reason: StopReason::Aborted,
                        ..
                    } => LoopError::Aborted,
                    AgentMessage::Assistant { error_message, .. } => LoopError::Stream(
                        StreamError::Transient(error_message.clone().unwrap_or_else(|| {
                            "assistant stream ended with error stop reason".into()
                        })),
                    ),
                    _ => LoopError::Stream(StreamError::Transient(
                        "assistant stream ended with error stop reason".into(),
                    )),
                };
                emit(
                    config,
                    AgentEvent::TurnEnd {
                        message: assistant,
                        tool_results: Vec::new(),
                    },
                )
                .await;
                emit(
                    config,
                    AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    },
                )
                .await;
                return Err(loop_error);
            }

            // Extract tool calls.
            let tool_calls: Vec<_> = match &assistant {
                AgentMessage::Assistant { content, .. } => {
                    content.tool_calls().into_iter().cloned().collect()
                }
                _ => Vec::new(),
            };
            last_turn_stopped_without_tool = tool_calls.is_empty();
            if last_turn_stopped_without_tool {
                empty_outcomes_seen = empty_outcomes_seen.saturating_add(1);
            }

            let mut tool_result_messages = Vec::new();
            has_more_tool_calls = false;

            if tool_calls.is_empty() {
                if let Some(tool_name) = config.plain_text_terminal_fallback_tool.as_deref() {
                    let eager = config.plain_text_terminal_fallback_eager;
                    let narrowed_to_terminators =
                        is_terminal_only_allowlist(turn_allowlist.as_ref(), tool_name);
                    let preserve_plain_text_candidate = plain_assistant_text(&assistant)
                        .is_some_and(|text| should_preserve_plain_text_terminal_candidate(&text));
                    if plain_text_terminal_fallback_candidate.is_none()
                        && preserve_plain_text_candidate
                    {
                        plain_text_terminal_fallback_candidate = Some(assistant.clone());
                    }
                    let nudge_mode = config.plain_text_terminal_fallback_eager_nudge
                        && eager
                        && !narrowed_to_terminators
                        && empty_outcomes_seen <= MAX_PLAIN_TEXT_NUDGE_RETRIES;
                    if nudge_mode {
                        // Catalog still contains real work tools (e.g. `plan`)
                        // but the model emitted prose. Inject an explicit
                        // protocol-recovery system message and force the
                        // inner loop to re-stream rather than laundering
                        // the prose into a synthetic `message_result`.
                        // After MAX_PLAIN_TEXT_NUDGE_RETRIES the synthesizer
                        // below fires as a last resort, preferring the first
                        // preserved non-clarifying answer so retry drift does
                        // not replace a good response with recovery chatter.
                        //
                        // Push directly into `current.messages` (mirrors the
                        // synthesize path) rather than `pending`, which is
                        // overwritten by `collect_steering` at end-of-iter.
                        // Set `has_more_tool_calls = true` to satisfy the
                        // inner while-loop's continuation predicate.
                        let nudge = AgentMessage::System {
                            content: PLAIN_TEXT_NUDGE_CONTEXT.to_string(),
                            timestamp: Some(now_ms()),
                        };
                        current.messages.push(nudge.clone());
                        new_messages.push(nudge);
                        has_more_tool_calls = true;
                    } else if let Some(result_msg) = synthesize_plain_text_terminal_result(
                        plain_text_terminal_fallback_candidate
                            .as_ref()
                            .unwrap_or(&assistant),
                        turn_allowlist.as_ref(),
                        tool_name,
                        eager,
                    ) {
                        plain_text_terminal_fallback_candidate = None;
                        last_turn_stopped_without_tool = false;
                        empty_outcomes_seen = 0;
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

                // A real tool batch is forward progress; the empty-outcome
                // budget tracks being stuck, not lifetime empty stops.
                empty_outcomes_seen = 0;
                plain_text_terminal_fallback_candidate = None;
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

        // Inner loop exhausted: either (a) the model produced no tool
        // calls AND no steering is queued, or (b) we hit the iteration
        // cap. In either case, give the follow-up sources one last
        // chance to inject a terminator nudge before declaring the
        // run done. To prevent infinite looping when a follow-up
        // re-arms but we're already past the cap, exit unconditionally
        // if the cap was hit.
        let cap_hit = config.max_iterations.is_some_and(|max| iterations >= max);
        // Skip the follow-up source pass when the last batch
        // terminated for the same reason steering is skipped above:
        // a clean terminator vote means the run is done; follow-up
        // sources exist to nudge the model toward a terminator when
        // it failed to emit one, not to overrule one it already cast.
        let follow_up = if last_batch_terminated {
            Vec::new()
        } else {
            collect_follow_up(config).await
        };
        if last_turn_stopped_without_tool {
            if let Some(budget) = config.empty_outcome_retry_budget {
                if empty_outcomes_seen > budget {
                    emit(
                        config,
                        AgentEvent::AgentEnd {
                            messages: new_messages.clone(),
                        },
                    )
                    .await;
                    return Err(LoopError::EmptyOutcomeBudgetExhausted {
                        budget,
                        observed: empty_outcomes_seen,
                    });
                }
            }
        }
        if !follow_up.is_empty() && !cap_hit {
            pending = follow_up;
            continue 'outer;
        }
        // If the cap was hit but a follow-up was produced, append it
        // to the transcript so listeners see the final nudge — but do
        // NOT re-enter the LLM loop. The user-facing run still ends
        // with this message as the last appended turn.
        if cap_hit {
            for msg in follow_up {
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

        break;
    }

    emit(
        config,
        AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        },
    )
    .await;

    // Classify outcome.
    // - HitMaxIterations: hard cap was reached before the model stopped
    //   tool-calling. The transcript may end on a turn that wanted to do
    //   more.
    // - WrappedUp: the graceful-turn-limit plugin fired its one-shot
    //   wrap-up steer AND we exited naturally (cap not hit). The model
    //   responded to the warning and produced a clean close.
    // - Done: natural termination with no budget pressure.
    let cap_hit_final = config.max_iterations.is_some_and(|max| iterations >= max);
    let wrap_up_fired = config
        .grace_signal
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed));
    let outcome = if cap_hit_final {
        LoopOutcome::HitMaxIterations
    } else if wrap_up_fired {
        LoopOutcome::WrappedUp
    } else {
        LoopOutcome::Done
    };
    Ok(outcome)
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
) -> Option<AgentMessage> {
    // The default contract is "only convert plain text once the runtime
    // has narrowed the catalog to terminators" — preserves strict
    // delivery shape for everyone else. When `eager` is set the gate is
    // lifted: the bridge has signalled this provider can never honor
    // forced tool choice, so prose IS the failure mode and the nudge
    // cycle that normally narrows the allowlist would just burn turns.
    if !eager && !is_terminal_only_allowlist(turn_allowlist, tool_name) {
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

fn should_preserve_plain_text_terminal_candidate(text: &str) -> bool {
    !looks_like_permission_or_clarification_question(text)
}

fn looks_like_permission_or_clarification_question(text: &str) -> bool {
    let trimmed = text.trim();
    if !trimmed.contains('?') {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let starts_with_prompt = [
        "would you like",
        "shall i",
        "should i",
        "do you want",
        "what would you like",
        "what do you need",
        "what's your next move",
        "what is your next move",
        "continue what",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix));
    starts_with_prompt
        || (trimmed.len() <= 500
            && lower.contains("what")
            && (lower.contains("next") || lower.contains("continue")))
}

fn is_terminal_only_allowlist(
    turn_allowlist: Option<&std::collections::HashSet<String>>,
    terminal_tool: &str,
) -> bool {
    let Some(allowlist) = turn_allowlist else {
        return false;
    };
    !allowlist.is_empty()
        && allowlist.contains(terminal_tool)
        && allowlist.iter().all(|tool| {
            tool == terminal_tool
                || tool == "message_ask"
                || tool == "message_info"
                || tool == "message_result"
                || tool == "terminator"
        })
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
async fn stream_with_max_tokens_recovery(
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    iteration: usize,
) -> Result<(AgentMessage, Option<std::collections::HashSet<String>>), LoopError> {
    let mut current_cap = config.max_output_tokens;
    let mut max_tokens_attempt: u8 = 0;
    let mut empty_stream_attempts: u8 = 0;
    let mut zero_output_transport_attempts: u8 = 0;
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
            Err(LoopError::Stream(StreamError::Empty))
                if empty_stream_attempts + 1 < EMPTY_STREAM_MAX_ATTEMPTS =>
            {
                empty_stream_attempts = empty_stream_attempts.saturating_add(1);
                let delay = EMPTY_STREAM_RETRY_INITIAL_DELAY * u32::from(empty_stream_attempts);
                tokio::select! {
                    _ = signal.cancelled() => return Err(LoopError::Aborted),
                    _ = tokio::time::sleep(delay) => {}
                }
                continue;
            }
            Err(LoopError::Stream(StreamError::ZeroOutputTransport(_)))
                if zero_output_transport_attempts + 1 < ZERO_OUTPUT_TRANSPORT_MAX_ATTEMPTS =>
            {
                zero_output_transport_attempts = zero_output_transport_attempts.saturating_add(1);
                zero_output_recovery_context =
                    Some(context_with_zero_output_transport_recovery(context));
                reasoning = zero_output_transport_retry_reasoning(config.reasoning);
                let delay = ZERO_OUTPUT_TRANSPORT_RETRY_INITIAL_DELAY
                    * u32::from(zero_output_transport_attempts);
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
        if stop_reason != StopReason::MaxTokens {
            return Ok((assistant, allowlist));
        }
        let Some(recovery) = config.max_output_tokens_recovery.as_ref() else {
            return Ok((assistant, allowlist));
        };
        if max_tokens_attempt >= recovery.max_attempts {
            return Ok((assistant, allowlist));
        }
        // No starting cap means there's no number to scale from. Refuse
        // recovery rather than guess — the deployment hadn't pinned a
        // cap, so the truncation came from a provider-side limit we
        // don't know how to raise.
        let Some(prev_cap) = current_cap else {
            return Ok((assistant, allowlist));
        };
        let Some(new_cap) = recovery.next_cap(prev_cap, max_tokens_attempt) else {
            return Ok((assistant, allowlist));
        };

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
        provider_extras: config.provider_extras.clone().unwrap_or(serde_json::Value::Null),
        // `tool_choice: "required"` on every turn. The LLM-in-charge
        // contract is "context → LLM → tool call → append result →
        // repeat" — the model's job is to pick a tool, not emit
        // narration. `message_result` IS the terminal text-delivery
        // tool, so required-on-every-turn doesn't trap the model:
        // when the work is done it calls `message_result` to deliver
        // the answer. If the model loops on verification instead, the
        // bug is in the catalog or prompt — not in the requirement.
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

    #[derive(Default)]
    struct EmptyThenTextStream {
        calls: AtomicUsize,
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

    #[derive(Default)]
    struct EmptyStopsAroundProgressStream {
        calls: AtomicUsize,
    }

    struct CountingFollowUp {
        remaining: AtomicUsize,
    }

    struct TerminalOnlyGate;
    struct TerminalWithStatusGate;
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

    impl CountingFollowUp {
        fn new(remaining: usize) -> Self {
            Self {
                remaining: AtomicUsize::new(remaining),
            }
        }
    }

    impl Plugin for CountingFollowUp {
        fn name(&self) -> &'static str {
            "counting_follow_up"
        }

        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::follow_up()
        }
    }

    #[async_trait::async_trait]
    impl FollowUpSource for CountingFollowUp {
        async fn next_follow_up_messages(&self) -> Vec<AgentMessage> {
            let used = self
                .remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .unwrap_or(0);
            if used == 0 {
                return Vec::new();
            }
            vec![AgentMessage::System {
                content: "retry after no-tool stop".into(),
                timestamp: None,
            }]
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
            if call == 0 {
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
    impl StreamFn for EmptyStopsAroundProgressStream {
        async fn stream(
            &self,
            _request: StreamRequest,
            _signal: CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let partial = empty_assistant_message();
            let message = match call {
                0 | 2 | 4 => text_assistant_message(format!("plain stop {call}")),
                1 | 3 => tool_call_assistant_message("progress", format!("tc-progress-{call}")),
                5 => tool_call_assistant_message("terminator", "tc-terminator"),
                other => panic!("unexpected stream call after terminal turn: {other}"),
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

    #[test]
    fn wrapped_up_is_complete() {
        assert!(LoopOutcome::Done.is_complete());
        assert!(LoopOutcome::WrappedUp.is_complete());
        assert!(!LoopOutcome::HitMaxIterations.is_complete());
    }

    #[tokio::test]
    async fn empty_stream_response_is_retried_before_returning() {
        let stream = Arc::new(EmptyThenTextStream::default());
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
                .expect("second stream attempt should recover");

        let AgentMessage::Assistant { content, .. } = assistant else {
            panic!("expected assistant response");
        };
        assert_eq!(content.plain_text(), "recovered");
        assert_eq!(stream.calls.load(Ordering::SeqCst), 2);
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
            "zero-output replay should lower high reasoning so Gemini-style private-only spins can produce a tool call"
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

    /// Tool that always votes `terminate=true`. Mirrors the contract
    /// the bridge's `MessageResultTool` upholds.
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

    struct ProgressTool;

    #[async_trait::async_trait]
    impl crate::tool::AgentTool for ProgressTool {
        fn name(&self) -> &str {
            "progress"
        }

        fn description(&self) -> &str {
            "test progress tool"
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
            Ok(crate::tool::ToolResult::text("made progress"))
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
        // Pattern C-3-a regression: a `SteeringSource` whose firing
        // condition lines up with the same turn the model delivers
        // (e.g. `graceful_turn_limit` reaching its soft limit on the
        // delivery turn) used to re-enter the loop and prompt the
        // model for ANOTHER turn after a clean terminator. The
        // model's drift on that extra turn corrupted the user-visible
        // answer in matrix run 20260507_100311. With the fix, a
        // unanimous terminator vote is a hard exit — steering sources
        // are not polled once the run has decided it's done.
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
        // Outcome is a clean Done, not WrappedUp (no graceful flag) and
        // not HitMaxIterations.
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
    async fn exhausted_empty_outcome_budget_returns_typed_loop_error() {
        let stream = Arc::new(RepeatedTextStream::default());
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .empty_outcome_retry_budget(1)
            .follow_up(CountingFollowUp::new(1))
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("continue".to_string()),
            timestamp: None,
        }];

        let err = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect_err("second no-tool stop should exhaust the budget");

        assert!(
            matches!(
                err,
                LoopError::EmptyOutcomeBudgetExhausted {
                    budget: 1,
                    observed: 2,
                }
            ),
            "unexpected error: {err:?}"
        );
        assert_eq!(stream.calls.load(Ordering::SeqCst), 2);
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
    async fn productive_tool_batch_resets_empty_outcome_budget() {
        let stream = Arc::new(EmptyStopsAroundProgressStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry
            .with(Arc::new(ProgressTool))
            .with(Arc::new(TerminatorTool));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("test-model")
            .tools(tool_registry)
            .empty_outcome_retry_budget(1)
            .follow_up(CountingFollowUp::new(3))
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("continue".to_string()),
            timestamp: None,
        }];

        let result = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect("productive tool batches should reset the empty-outcome budget");

        assert_eq!(result.outcome, LoopOutcome::Done);
        assert_eq!(stream.calls.load(Ordering::SeqCst), 6);
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
            .empty_outcome_retry_budget(0)
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
        // Models in the auto-when-forced class (Qwen 3.5 Flash etc.)
        // can never be wire-forced into a tool call, so prose IS their
        // failure mode. The eager flag lifts the
        // "allowlist must already be narrowed" precondition so the
        // fallback fires on the FIRST plain-text stop instead of after
        // `TerminalMessageGuard` has burned 2-3 nudge turns.
        //
        // No `tool_gate_arc` is installed in this test, so the catalog
        // stays at the full registry — exactly the situation where the
        // non-eager path would refuse to convert and the run would die
        // on the empty-outcome budget.
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("auto-tool-provider-eager")
            .tools(tool_registry)
            .plain_text_terminal_fallback_tool("message_result")
            .plain_text_terminal_fallback_eager(true)
            .empty_outcome_retry_budget(0)
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
    async fn eager_nudge_mode_injects_protocol_recovery_before_synthesizing() {
        // With `plain_text_terminal_fallback_eager_nudge(true)` the eager
        // path nudges the model with a protocol-recovery system message
        // on each consecutive plain-text stop, up to
        // `MAX_PLAIN_TEXT_NUDGE_RETRIES`. After the cap a synthesizer
        // fires as a last resort so the run still terminates with the
        // model's prose as the delivered text — never silently, never
        // forever. Verifies the recovery path is observable in the
        // emitted message stream (the model sees the nudges in context)
        // and that the synthesizer ultimately delivers the first
        // substantive plain-text answer, not later recovery drift.
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
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
            .expect("nudge mode should eventually synthesize after retries");

        // MAX_PLAIN_TEXT_NUDGE_RETRIES = 2 → two nudges fire, then on the
        // third empty stop the synthesizer takes over. Total LLM calls = 3.
        assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
        assert_eq!(result.outcome, LoopOutcome::Done);

        let nudge_count = result
            .messages
            .iter()
            .filter(|m| matches!(m, AgentMessage::System { content, .. } if content == PLAIN_TEXT_NUDGE_CONTEXT))
            .count();
        assert_eq!(
            nudge_count, 2,
            "expected two protocol-recovery system messages in the run output, got {nudge_count}",
        );

        let synthesized_text = result
            .messages
            .iter()
            .find_map(|message| match message {
                AgentMessage::ToolResult {
                    tool_name,
                    content,
                    is_error: false,
                    ..
                } if tool_name == "message_result" => Some(content.plain_text()),
                _ => None,
            })
            .expect("a terminal tool result should be synthesized as last resort");
        assert_eq!(
            synthesized_text, "plain stop 0",
            "synthesizer should deliver the first preserved plain text, not later recovery drift",
        );
    }

    #[test]
    fn plain_text_fallback_candidate_skips_obvious_clarifying_questions() {
        assert!(!should_preserve_plain_text_terminal_candidate(
            "Continue what, exactly? What's your next move?"
        ));
        assert!(!should_preserve_plain_text_terminal_candidate(
            "Would you like me to proceed?"
        ));
        assert!(should_preserve_plain_text_terminal_candidate(
            "# Machine Learning\n\nMachine learning is the branch of artificial intelligence that studies systems which improve from data."
        ));
    }

    #[tokio::test]
    async fn non_eager_plain_text_fallback_still_requires_narrowed_allowlist() {
        // Default behaviour preserved: when eager is NOT set and the
        // turn allowlist is the full catalog, plain text is NOT
        // converted — the run dies on the empty-outcome budget,
        // matching the pre-eager contract for every other model.
        let stream = Arc::new(RepeatedTextStream::default());
        let mut tool_registry = crate::tool::ToolRegistry::new();
        tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
        let config = AgentBuilder::new()
            .stream(stream.clone())
            .model_id("non-eager-provider")
            .tools(tool_registry)
            .plain_text_terminal_fallback_tool("message_result")
            // Eager NOT set → defaults to false.
            .empty_outcome_retry_budget(0)
            .build()
            .expect("config builds");
        let context = AgentContext::new("system");
        let prompts = vec![AgentMessage::User {
            content: UserContent::Text("answer directly".to_string()),
            timestamp: None,
        }];

        let err = run(prompts, context, &config, CancellationToken::new())
            .await
            .expect_err("non-eager fallback must not convert without narrowed allowlist");

        assert!(
            matches!(err, LoopError::EmptyOutcomeBudgetExhausted { .. }),
            "unexpected error: {err:?}"
        );
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
            .tool_gate_arc(Arc::new(TerminalWithStatusGate))
            .plain_text_terminal_fallback_tool("message_result")
            .empty_outcome_retry_budget(0)
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
