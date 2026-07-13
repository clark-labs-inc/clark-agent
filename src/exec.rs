//! Tool batch execution.
//!
//! Canonical prepare / execute / finalize chain for model-emitted tool
//! calls.
//!
//! Two modes:
//!
//! - **Parallel** (default): all tools in the batch prep sequentially,
//!   then run concurrently, then finalize sequentially in source order.
//! - **Sequential**: each tool is prepped, executed, and finalized
//!   before the next starts. Triggered by either:
//!     - any tool in the batch setting `requires_exclusive_sandbox = true`, or
//!     - `LoopConfig.default_execution_mode = Sequential` (loop-wide pin).
//!
//! Hook plumbing:
//! - `BeforeToolCall::on_before_tool_call` runs after argument validation,
//!   before `tool.execute`. May `block` to short-circuit with an error
//!   tool result.
//! - `AfterToolCall::on_after_tool_call` runs after `tool.execute`. May
//!   `override_result`, `mark_error`, or vote `terminate`.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::config::LoopConfig;
use crate::error::{LoopError, ToolError};
use crate::event::{AgentEvent, EventSink};
use crate::plugin::{AfterToolCallContext, BeforeToolCallContext, EventObserver};
use crate::tool::{detect_arg_parse_error, AgentTool, ExecutionMode, ToolCall, ToolResult};
use crate::types::{AgentContext, AgentMessage, AssistantContent, ToolResultContent};

const TOOL_UPDATE_DRAIN_GRACE: Duration = Duration::from_millis(50);
const TOOL_UPDATE_EVENT_QUEUE_CAPACITY: usize = 256;

fn spawn_tool_update_dispatcher(
    event_sink: Arc<dyn EventSink>,
    observers: Vec<Arc<dyn EventObserver>>,
) -> mpsc::Sender<AgentEvent> {
    let (tx, mut rx) = mpsc::channel::<AgentEvent>(TOOL_UPDATE_EVENT_QUEUE_CAPACITY);
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            event_sink.emit(event.clone()).await;
            for observer in observers.iter() {
                observer.on_event(&event).await;
            }
        }
    });
    tx
}

fn enqueue_tool_update_event(tx: &mpsc::Sender<AgentEvent>, event: AgentEvent) {
    match tx.try_send(event) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            tracing::warn!("tool update event queue full; dropping partial update");
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {}
    }
}

/// Result of executing one batch.
pub(crate) struct ExecutedBatch {
    /// Tool result messages in source order, ready to push to history.
    pub messages: Vec<AgentMessage>,
    /// Unanimous-vote terminate signal: true when every finalized result
    /// in the batch had `terminate = true`. Empty batches return false.
    pub terminate: bool,
}

pub(crate) async fn execute_tool_batch(
    assistant: &AgentMessage,
    tool_calls: Vec<ToolCall>,
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    turn_allowlist: Option<&std::collections::HashSet<String>>,
) -> Result<ExecutedBatch, LoopError> {
    if tool_calls.is_empty() {
        return Ok(ExecutedBatch {
            messages: Vec::new(),
            terminate: false,
        });
    }

    // Let the active protocol policy normalize the batch before registry
    // lookup — e.g. fold a product's known alias names into a canonical
    // tool. The default policy is a no-op; the core performs no alias
    // repair of its own, so no product tool vocabulary lives here.
    let mut tool_calls = tool_calls;
    config
        .protocol
        .normalize_tool_calls(&mut tool_calls, &config.tools);

    let total_tool_calls = tool_calls.len();
    let limit_counted_tool_calls = count_limit_counted_tool_calls(&tool_calls, &config.tools);
    let (tool_calls, unexecuted_tool_calls, max_executed) =
        split_tool_calls_for_execution(tool_calls, &config.tools, config.max_tool_calls_per_turn);

    let assistant_content = match assistant {
        AgentMessage::Assistant { content, .. } => content.clone(),
        _ => AssistantContent { blocks: Vec::new() },
    };

    if tool_calls.is_empty() {
        let messages = synthesize_unexecuted_tool_results(
            assistant,
            &assistant_content,
            unexecuted_tool_calls,
            total_tool_calls,
            limit_counted_tool_calls,
            max_executed.unwrap_or(0),
            context,
            config,
        )
        .await;
        return Ok(ExecutedBatch {
            messages,
            terminate: false,
        });
    }

    // A batch downgrades to Sequential when either (a) the loop is
    // pinned to Sequential mode, or (b) any participating tool needs
    // exclusive sandbox access.
    let any_exclusive = tool_calls.iter().any(|call| {
        config
            .tools
            .get(&call.name)
            .map(|t| t.requires_exclusive_sandbox())
            .unwrap_or(false)
    });

    let effective_mode =
        if any_exclusive || config.default_execution_mode == ExecutionMode::Sequential {
            ExecutionMode::Sequential
        } else {
            ExecutionMode::Parallel
        };

    let mut batch = match effective_mode {
        ExecutionMode::Sequential => {
            execute_sequential(
                assistant,
                &assistant_content,
                tool_calls,
                context,
                config,
                signal,
                turn_allowlist,
            )
            .await
        }
        ExecutionMode::Parallel => {
            execute_parallel(
                assistant,
                &assistant_content,
                tool_calls,
                context,
                config,
                signal,
                turn_allowlist,
            )
            .await
        }
    }?;

    if !unexecuted_tool_calls.is_empty() {
        batch.messages.extend(
            synthesize_unexecuted_tool_results(
                assistant,
                &assistant_content,
                unexecuted_tool_calls,
                total_tool_calls,
                limit_counted_tool_calls,
                max_executed.unwrap_or(0),
                context,
                config,
            )
            .await,
        );
        batch.terminate = false;
    }

    Ok(batch)
}

fn split_tool_calls_for_execution(
    tool_calls: Vec<ToolCall>,
    tools: &crate::tool::ToolRegistry,
    max_tool_calls: Option<usize>,
) -> (Vec<ToolCall>, Vec<ToolCall>, Option<usize>) {
    let Some(max_tool_calls) = max_tool_calls else {
        return (tool_calls, Vec::new(), None);
    };
    let max_tool_calls = max_tool_calls.max(1);
    if count_limit_counted_tool_calls(&tool_calls, tools) <= max_tool_calls {
        return (tool_calls, Vec::new(), Some(max_tool_calls));
    }

    let mut executable = Vec::with_capacity(tool_calls.len());
    let mut unexecuted = Vec::new();
    let mut executed_counted = 0usize;
    for call in tool_calls {
        if !tool_counts_toward_call_limit(tools, &call.name) {
            // Progress-only tools, parallel-safe reads, and malformed calls
            // that resolve to no registered tool never burn the per-turn cap;
            // let them all through (see `tool_counts_toward_call_limit`). A
            // malformed call still lands as a synthetic "no such tool" error
            // in `prepare_call`; it just cannot preempt a real call's slot.
            executable.push(call);
        } else if executed_counted < max_tool_calls {
            executed_counted += 1;
            executable.push(call);
        } else {
            unexecuted.push(call);
        }
    }
    (executable, unexecuted, Some(max_tool_calls))
}

fn count_limit_counted_tool_calls(
    tool_calls: &[ToolCall],
    tools: &crate::tool::ToolRegistry,
) -> usize {
    tool_calls
        .iter()
        .filter(|call| tool_counts_toward_call_limit(tools, &call.name))
        .count()
}

/// Whether a tool consumes a slot from the per-turn cap.
///
/// A call is exempt from the cap when ANY of these hold:
/// - it opts out of the cap (progress-only signals that report status
///   without doing work — see [`AgentTool::counts_toward_tool_call_limit`]);
/// - it is marked parallel-safe (idempotent reads like `web_search`,
///   `file_read`, `grep`, `glob`);
/// - it does not resolve to a registered tool at all — an empty/blank name,
///   or a name no tool exists for.
///
/// The last case is the load-bearing one. A call that resolves to no tool
/// does NO real work: `prepare_call` short-circuits it to a synchronous
/// "no such tool" error result without ever invoking a tool. Counting it
/// would let a glitch — e.g. a streamed tool call that arrived with an empty
/// `name` — spend the turn's only slot and bump a *real* call into the
/// unexecuted bin. That was observed in production: under
/// `max_tool_calls_per_turn = 1`, an empty-name call preempted the model's
/// real next call. A malformed call must never preempt real work; it still
/// surfaces its error so the model can react next turn.
///
/// Note this is the opposite of the *termination*-vote rule: there, an
/// unresolved name DOES count (see `tool_counts_toward_termination_vote`),
/// so a stray call cannot accidentally end the run. The asymmetry is
/// intentional — a malformed call does no work (so it must not spend the
/// work budget) and is not a satisfied terminator (so it must not vote to
/// stop).
fn tool_counts_toward_call_limit(tools: &crate::tool::ToolRegistry, name: &str) -> bool {
    tools
        .get(name)
        .map(|tool| tool.counts_toward_tool_call_limit() && !tool.parallel_safe_per_turn())
        .unwrap_or(false)
}

/// Whether a tool's terminate vote is counted in the unanimous-vote
/// tally. Unknown / unregistered names default to `true` so a stray
/// tool call cannot accidentally end the run by being treated as
/// advisory. See `AgentTool::counts_toward_termination_vote`.
fn tool_counts_toward_termination_vote(tools: &crate::tool::ToolRegistry, name: &str) -> bool {
    tools
        .get(name)
        .map(|tool| tool.counts_toward_termination_vote())
        .unwrap_or(true)
}

/// Compute the batch-level terminate signal, ignoring tools that opt
/// out via `counts_toward_termination_vote() == false`.
///
/// The batch terminates iff:
/// - at least one *counted* tool is present, AND
/// - every counted tool voted `terminate: true`.
///
/// An all-advisory batch (e.g. only progress-note calls) returns
/// `false` because no counted tool voted yes — progress notes never
/// end the run on their own.
///
/// When the batch terminates AND advisory siblings were skipped from
/// the tally, emits a structured `tracing::info` line so operators can
/// measure how often this fallback actually fires in production — a
/// non-zero rate names which model still needs the safety net.
fn compute_batch_terminate<'a, I>(tools: &crate::tool::ToolRegistry, votes: I) -> bool
where
    I: IntoIterator<Item = (&'a str, bool)>,
{
    let mut counted_total = 0usize;
    let mut counted_terminate = 0usize;
    let mut terminating: Vec<&'a str> = Vec::new();
    let mut advisory_skipped: Vec<&'a str> = Vec::new();
    for (name, terminate) in votes {
        if !tool_counts_toward_termination_vote(tools, name) {
            advisory_skipped.push(name);
            continue;
        }
        counted_total += 1;
        if terminate {
            counted_terminate += 1;
            terminating.push(name);
        }
    }
    let terminated = counted_total > 0 && counted_terminate == counted_total;
    if terminated && !advisory_skipped.is_empty() {
        tracing::info!(
            terminating_tools = ?terminating,
            advisory_tools = ?advisory_skipped,
            counted_total,
            "advisory siblings excluded from unanimous termination vote"
        );
    }
    terminated
}

// The execution helpers share the same loop context tuple. Keeping the
// signatures explicit is clearer than introducing a one-off bag of references.
#[allow(clippy::too_many_arguments)]
async fn synthesize_unexecuted_tool_results(
    assistant: &AgentMessage,
    assistant_content: &AssistantContent,
    tool_calls: Vec<ToolCall>,
    total_tool_calls: usize,
    limit_counted_tool_calls: usize,
    max_executed: usize,
    context: &AgentContext,
    config: &LoopConfig,
) -> Vec<AgentMessage> {
    let mut messages = Vec::with_capacity(tool_calls.len());
    for call in tool_calls {
        let outcome = finalize(
            assistant,
            assistant_content,
            &call,
            &call.arguments,
            ExecutedOutcome {
                result: unexecuted_tool_call_result(
                    total_tool_calls,
                    limit_counted_tool_calls,
                    max_executed,
                ),
                is_error: true,
            },
            &context.messages,
            &config.plugins.after_tool_call,
        )
        .await;
        emit_tool_end(config, &call, &outcome).await;
        messages.push(outcome_to_message(&call, outcome));
    }
    messages
}

fn unexecuted_tool_call_message(
    total_tool_calls: usize,
    limit_counted_tool_calls: usize,
    max_executed: usize,
) -> String {
    let call_word = if total_tool_calls == 1 {
        "tool call"
    } else {
        "tool calls"
    };
    let limited_call_word = if limit_counted_tool_calls == 1 {
        "limit-counted tool call"
    } else {
        "limit-counted tool calls"
    };
    let executed_word = if max_executed == 1 { "call" } else { "calls" };
    if limit_counted_tool_calls != total_tool_calls {
        return format!(
            "This tool call was not executed because the assistant turn emitted \
             {limit_counted_tool_calls} {limited_call_word} ({total_tool_calls} \
             {call_word} total, including progress-only calls), but only the \
             first {max_executed} limit-counted {executed_word} can run in one \
             turn. The earlier allowed calls already ran. Reissue this call in \
             a later turn, one tool call at a time."
        );
    }
    format!(
        "This tool call was not executed because the assistant turn emitted \
         {total_tool_calls} {call_word}, but only the first {max_executed} \
         {executed_word} can run in one turn. The earlier {max_executed} \
         {executed_word} already ran. Reissue this call in a later turn, \
         one tool call at a time."
    )
}

fn unexecuted_tool_call_result(
    total_tool_calls: usize,
    limit_counted_tool_calls: usize,
    max_executed: usize,
) -> ToolResult {
    let mut result = ToolResult::error(unexecuted_tool_call_message(
        total_tool_calls,
        limit_counted_tool_calls,
        max_executed,
    ));
    result.details = json!({
        "kind": "tool_call_not_executed",
        "reason": "max_tool_calls_per_turn",
        "total_tool_calls": total_tool_calls,
        "limit_counted_tool_calls": limit_counted_tool_calls,
        "max_executed": max_executed,
    });
    result
}

#[allow(clippy::too_many_arguments)]
async fn execute_sequential(
    assistant: &AgentMessage,
    assistant_content: &AssistantContent,
    tool_calls: Vec<ToolCall>,
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    turn_allowlist: Option<&std::collections::HashSet<String>>,
) -> Result<ExecutedBatch, LoopError> {
    let mut messages = Vec::with_capacity(tool_calls.len());
    let mut votes: Vec<(String, bool)> = Vec::with_capacity(tool_calls.len());

    for call in tool_calls {
        let outcome = run_one(
            assistant,
            assistant_content,
            &call,
            context,
            config,
            signal,
            turn_allowlist,
        )
        .await?;
        votes.push((call.name.clone(), outcome.terminate));
        messages.push(outcome_to_message(&call, outcome));
    }

    let terminate =
        compute_batch_terminate(&config.tools, votes.iter().map(|(n, t)| (n.as_str(), *t)));

    Ok(ExecutedBatch {
        messages,
        terminate,
    })
}

#[allow(clippy::too_many_arguments)]
async fn execute_parallel(
    assistant: &AgentMessage,
    assistant_content: &AssistantContent,
    tool_calls: Vec<ToolCall>,
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    turn_allowlist: Option<&std::collections::HashSet<String>>,
) -> Result<ExecutedBatch, LoopError> {
    use futures::stream::{FuturesUnordered, StreamExt};

    // Per-batch cancellation lever. As a child of `signal` it auto-
    // cancels when the run-wide signal cancels (so tools react to the
    // user's abort). It can also be cancelled independently on
    // sibling-error opt-in (`AgentTool::aborts_siblings_on_error`),
    // propagating only to siblings in *this* batch — neither sibling
    // failures nor sibling-triggered cancels affect the run-wide
    // signal.
    let batch_token = signal.child_token();

    // Prep + emit starts sequentially so validation and event ordering are
    // deterministic. A start means the real tool implementation is about to
    // run; parse, registry, validation, and before-hook failures never start.
    let mut prepared: Vec<(ToolCall, PreparedCall)> = Vec::with_capacity(tool_calls.len());
    for call in tool_calls {
        let prep = prepare_call(
            assistant,
            assistant_content,
            &call,
            context,
            config,
            turn_allowlist,
        )
        .await;
        if matches!(prep, PreparedCall::Prepared { .. }) {
            emit_tool_start(config, &call).await;
        }
        prepared.push((call, prep));
    }

    let mut futures = Vec::with_capacity(prepared.len());
    let mut immediate: Vec<(usize, ToolCall, FinalizedOutcome)> = Vec::new();

    for (idx, (call, prep)) in prepared.into_iter().enumerate() {
        match prep {
            PreparedCall::Immediate(executed) => {
                // Route Immediate outcomes through finalize so
                // AfterToolCall hooks observe every tool result —
                // including arg-parse / validation / before-block
                // errors. The `args` we hand to hooks is the original
                // (potentially sentinel-bearing) call arguments since
                // we never built prepared args for short-circuited
                // calls.
                let finalized = finalize(
                    assistant,
                    assistant_content,
                    &call,
                    &call.arguments,
                    executed,
                    &context.messages,
                    &config.plugins.after_tool_call,
                )
                .await;
                immediate.push((idx, call, finalized));
            }
            PreparedCall::Prepared { tool, args } => {
                let tool_signal = batch_token.child_token();
                let run_signal = signal.clone();
                let batch_token_clone = batch_token.clone();
                let assistant_clone = assistant.clone();
                let assistant_content_clone = assistant_content.clone();
                let context_messages = context.messages.clone();
                let after_hooks = config.plugins.after_tool_call.clone();
                let event_sink = config.event_sink.clone();
                let event_observers = config.plugins.event_observer.clone();
                let call_clone = call.clone();
                let fut = async move {
                    let id = call_clone.id.clone();
                    let name = call_clone.name.clone();
                    let name_for_message = name.clone();
                    let update_events = spawn_tool_update_dispatcher(event_sink, event_observers);
                    let executed_result = execute_prepared(
                        &tool,
                        &call_clone,
                        args.clone(),
                        tool_signal,
                        Box::new(move |update| {
                            let event = AgentEvent::ToolExecutionUpdate {
                                tool_call_id: id.clone(),
                                tool_name: name.clone(),
                                partial: update,
                            };
                            enqueue_tool_update_event(&update_events, event);
                        }),
                    )
                    .await;
                    let executed = match executed_result {
                        Ok(executed) => executed,
                        Err(LoopError::Aborted)
                            if batch_token_clone.is_cancelled() && !run_signal.is_cancelled() =>
                        {
                            // Sibling abort, not user abort. Convert
                            // to a recoverable tool result so the
                            // model sees what happened next turn and
                            // the unanimous-vote termination rule
                            // stays intact.
                            ExecutedOutcome {
                                result: ToolResult::error(format!(
                                    "aborted because a sibling tool in the \
                                     parallel batch errored — re-run this \
                                     {name_for_message} call after addressing the \
                                     sibling failure"
                                )),
                                is_error: true,
                            }
                        }
                        Err(other) => return Err(other),
                    };
                    let finalized = finalize(
                        &assistant_clone,
                        &assistant_content_clone,
                        &call_clone,
                        &args,
                        executed,
                        &context_messages,
                        &after_hooks,
                    )
                    .await;
                    Ok::<_, LoopError>((idx, call_clone, finalized))
                };
                futures.push(fut);
            }
        }
    }

    // Drain futures as they complete. When an opted-in tool returns
    // an error, cancel `batch_token` so still-running siblings exit
    // promptly (cooperatively — they must check the signal). The
    // futures already in flight that complete *before* the trigger
    // produce their natural result. Cancelled siblings produce a
    // typed `is_error: true` ToolResult via the match arm above.
    let mut unordered: FuturesUnordered<_> = futures.into_iter().collect();
    let mut completed: Vec<(usize, ToolCall, FinalizedOutcome)> =
        Vec::with_capacity(unordered.len() + immediate.len());
    while let Some(result) = unordered.next().await {
        let entry = result?;
        if entry.2.is_error {
            let aborts = config
                .tools
                .get(&entry.1.name)
                .map(|t| t.aborts_siblings_on_error())
                .unwrap_or(false);
            if aborts && !batch_token.is_cancelled() {
                batch_token.cancel();
            }
        }
        completed.push(entry);
    }
    completed.extend(immediate);
    completed.sort_by_key(|(idx, _, _)| *idx);

    let mut messages = Vec::with_capacity(completed.len());
    let mut votes: Vec<(String, bool)> = Vec::with_capacity(completed.len());
    for (_idx, call, outcome) in completed {
        emit_tool_end(config, &call, &outcome).await;
        votes.push((call.name.clone(), outcome.terminate));
        messages.push(outcome_to_message(&call, outcome));
    }

    let terminate =
        compute_batch_terminate(&config.tools, votes.iter().map(|(n, t)| (n.as_str(), *t)));

    Ok(ExecutedBatch {
        messages,
        terminate,
    })
}

/// Execute one tool call synchronously: prep → execute → finalize.
/// Used by the sequential path.
#[allow(clippy::too_many_arguments)]
async fn run_one(
    assistant: &AgentMessage,
    assistant_content: &AssistantContent,
    call: &ToolCall,
    context: &AgentContext,
    config: &LoopConfig,
    signal: &CancellationToken,
    turn_allowlist: Option<&std::collections::HashSet<String>>,
) -> Result<FinalizedOutcome, LoopError> {
    let prep = prepare_call(
        assistant,
        assistant_content,
        call,
        context,
        config,
        turn_allowlist,
    )
    .await;
    let outcome = match prep {
        PreparedCall::Immediate(executed) => {
            finalize(
                assistant,
                assistant_content,
                call,
                &call.arguments,
                executed,
                &context.messages,
                &config.plugins.after_tool_call,
            )
            .await
        }
        PreparedCall::Prepared { tool, args } => {
            emit_tool_start(config, call).await;
            let event_sink = config.event_sink.clone();
            let event_observers = config.plugins.event_observer.clone();
            let id = call.id.clone();
            let name = call.name.clone();
            let update_events = spawn_tool_update_dispatcher(event_sink, event_observers);
            let executed = execute_prepared(
                &tool,
                call,
                args.clone(),
                signal.clone(),
                Box::new(move |update| {
                    let event = AgentEvent::ToolExecutionUpdate {
                        tool_call_id: id.clone(),
                        tool_name: name.clone(),
                        partial: update,
                    };
                    enqueue_tool_update_event(&update_events, event);
                }),
            )
            .await?;
            finalize(
                assistant,
                assistant_content,
                call,
                &args,
                executed,
                &context.messages,
                &config.plugins.after_tool_call,
            )
            .await
        }
    };

    emit_tool_end(config, call, &outcome).await;
    Ok(outcome)
}

// ─── Internal pipeline ────────────────────────────────────────────

enum PreparedCall {
    /// Argument validation, parse-error detection, or `BeforeToolCall`
    /// short-circuited the call. The loop emits the error tool result
    /// without invoking `tool.execute`, but still runs `AfterToolCall`
    /// hooks so observers (terminal-message guard, system-reminder hook,
    /// etc.) see every tool result — successes and failures alike.
    Immediate(ExecutedOutcome),
    /// Ready to execute.
    Prepared {
        tool: Arc<dyn AgentTool>,
        args: Value,
    },
}

struct ExecutedOutcome {
    result: ToolResult,
    is_error: bool,
}

pub(crate) struct FinalizedOutcome {
    pub result: ToolResult,
    pub is_error: bool,
    pub terminate: bool,
}

/// Walk every registered `ToolGate` and ask each for a specific reason
/// it denies `tool_name`. Returns the first specific reason; `None` if
/// no gate claims responsibility (caller falls back to the active
/// [`crate::protocol::ProtocolPolicy`], then to the core's generic
/// hidden-tool message).
struct GateDenial {
    reason: String,
    gate: &'static str,
}

async fn gate_attributed_denial(
    tool_name: &str,
    config: &LoopConfig,
    messages: &[AgentMessage],
) -> Option<GateDenial> {
    let available_tool_names: Vec<&str> = config.tools.iter().map(|t| t.name()).collect();
    let iteration = messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::Assistant { .. }))
        .count();
    for gate in &config.plugins.tool_gate {
        let ctx = crate::plugin::ToolGateContext {
            iteration,
            messages,
            conversation_id: config.conversation_id.as_deref(),
            available_tool_names: &available_tool_names,
        };
        if let Some(reason) = gate.denial_reason(tool_name, ctx).await {
            return Some(GateDenial {
                reason,
                gate: gate.name(),
            });
        }
    }
    None
}

async fn prepare_call(
    assistant: &AgentMessage,
    assistant_content: &AssistantContent,
    call: &ToolCall,
    context: &AgentContext,
    config: &LoopConfig,
    turn_allowlist: Option<&std::collections::HashSet<String>>,
) -> PreparedCall {
    let Some(tool) = config.tools.get(&call.name) else {
        return PreparedCall::Immediate(ExecutedOutcome {
            result: ToolResult::error(format!("Tool `{}` not found", call.name)),
            is_error: true,
        });
    };

    // Hard-enforce per-turn `ToolGate` narrowing. The allowlist filters
    // what schemas the model SEES; without this check, the model can
    // hallucinate a tool name that wasn't advertised this turn and the
    // dispatcher runs it anyway because the registry is global. That was
    // observed in production: a model called a terminal delivery tool
    // after a no-work narrowing had dropped it from the catalog, claimed
    // success without doing any work, and the file it claimed to create
    // didn't exist. Refuse here so the model sees a typed tool error and
    // either picks an allowed tool or surfaces an unrecoverable state.
    //
    // Message + details are sourced in priority order:
    //   1. a `ToolGate` that attributes the denial via `denial_reason`;
    //   2. the active `ProtocolPolicy` (product vocabulary, if any);
    //   3. the core's generic, vocabulary-free fallback.
    if let Some(allowlist) = turn_allowlist {
        if !allowlist.contains(call.name.as_str()) {
            let attributed = gate_attributed_denial(&call.name, config, &context.messages).await;
            let (message, details) =
                match attributed {
                    Some(denial) => {
                        let details = crate::protocol::generic_hidden_tool_details(
                            &call.name,
                            allowlist,
                            Some(denial.gate),
                        );
                        (denial.reason, details)
                    }
                    None => match config.protocol.hidden_tool_error(
                        crate::protocol::HiddenToolContext {
                            requested_tool: &call.name,
                            allowlist,
                            messages: &context.messages,
                        },
                    ) {
                        Some(err) => (err.message, err.details),
                        None => (
                            crate::protocol::generic_hidden_tool_message(&call.name, allowlist),
                            crate::protocol::generic_hidden_tool_details(
                                &call.name, allowlist, None,
                            ),
                        ),
                    },
                };
            let mut result = ToolResult::error(message);
            result.details = details;
            return PreparedCall::Immediate(ExecutedOutcome {
                result,
                is_error: true,
            });
        }
    }

    // Provider stream layers wrap a malformed-JSON tool-args buffer in
    // a sentinel object so we can surface a clean, model-recoverable
    // error here instead of the cryptic `invalid type: string, expected
    // struct …` that comes from each tool's `serde_json::from_value`
    // running over a `Value::String` fallback. Detect the sentinel
    // before validation/dispatch.
    if let Some((parse_err, raw)) = detect_arg_parse_error(&call.arguments) {
        return PreparedCall::Immediate(ExecutedOutcome {
            result: ToolResult::error(format_arg_parse_error(&call.name, parse_err, raw)),
            is_error: true,
        });
    }

    let prepared_args = tool.prepare_arguments(call.arguments.clone());

    if let Err(err) = tool.validate(&prepared_args) {
        return PreparedCall::Immediate(ExecutedOutcome {
            result: ToolResult::error(err.to_string()),
            is_error: true,
        });
    }

    let ctx = BeforeToolCallContext {
        assistant_message: assistant,
        assistant_content,
        tool_call: call,
        args: &prepared_args,
        messages: &context.messages,
    };
    for hook in &config.plugins.before_tool_call {
        let decision = hook
            .on_before_tool_call(BeforeToolCallContext {
                assistant_message: ctx.assistant_message,
                assistant_content: ctx.assistant_content,
                tool_call: ctx.tool_call,
                args: ctx.args,
                messages: ctx.messages,
            })
            .await;
        if decision.block {
            let reason = decision
                .reason
                .unwrap_or_else(|| format!("blocked by {}", hook.name()));
            let mut result = ToolResult::error(reason);
            if let Some(details) = decision.details {
                result.details = details;
            }
            return PreparedCall::Immediate(ExecutedOutcome {
                result,
                is_error: true,
            });
        }
    }

    PreparedCall::Prepared {
        tool,
        args: prepared_args,
    }
}

async fn execute_prepared(
    tool: &Arc<dyn AgentTool>,
    call: &ToolCall,
    args: Value,
    signal: CancellationToken,
    on_update: Box<dyn Fn(ToolResult) + Send + Sync + 'static>,
) -> Result<ExecutedOutcome, LoopError> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ToolResult>();

    // Drain partial updates concurrently so they don't backpressure the tool.
    let mut drain_handle = tokio::spawn(async move {
        while let Some(partial) = rx.recv().await {
            on_update(partial);
        }
    });

    let result = match tool.execute(&call.id, args, signal, tx).await {
        Ok(result) => {
            let is_error = result.is_error;
            Ok(ExecutedOutcome { result, is_error })
        }
        Err(ToolError::Execution(reason)) => Ok(ExecutedOutcome {
            result: ToolResult::error(ToolError::Execution(reason).to_string()),
            is_error: true,
        }),
        Err(ToolError::Aborted) => Err(LoopError::Aborted),
        Err(ToolError::Fatal(reason)) => Err(LoopError::ToolFatal {
            tool: call.name.clone(),
            reason,
        }),
    };

    match timeout(TOOL_UPDATE_DRAIN_GRACE, &mut drain_handle).await {
        Ok(joined) => {
            if let Err(error) = joined {
                tracing::debug!(?error, "tool update dispatcher join failed");
            }
        }
        Err(_) => {
            drain_handle.abort();
            if let Err(error) = drain_handle.await {
                tracing::debug!(?error, "aborted tool update dispatcher");
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
async fn finalize(
    assistant: &AgentMessage,
    _assistant_content: &AssistantContent,
    call: &ToolCall,
    args: &Value,
    mut executed: ExecutedOutcome,
    messages: &[AgentMessage],
    after_hooks: &[Arc<dyn crate::plugin::AfterToolCall>],
) -> FinalizedOutcome {
    for hook in after_hooks {
        let ctx = AfterToolCallContext {
            assistant_message: assistant,
            tool_call: call,
            args,
            result: &executed.result,
            is_error: executed.is_error,
            messages,
        };
        let decision = hook.on_after_tool_call(ctx).await;
        if let Some(new_result) = decision.result {
            executed.is_error = new_result.is_error;
            executed.result = new_result;
        }
        if let Some(mark_error) = decision.mark_error {
            executed.is_error = mark_error;
            executed.result.is_error = mark_error;
        }
        if let Some(terminate) = decision.terminate {
            executed.result.terminate = terminate;
        }
    }

    FinalizedOutcome {
        result: executed.result,
        is_error: executed.is_error,
        terminate: false,
    }
    // Carry forward the result's own `terminate` field as the outcome's
    // vote. (Done after the after-hooks have had a chance to override.)
    .with_vote()
}

impl FinalizedOutcome {
    fn with_vote(mut self) -> Self {
        self.terminate = self.result.terminate;
        self
    }
}

fn outcome_to_message(call: &ToolCall, outcome: FinalizedOutcome) -> AgentMessage {
    let details = match outcome.result.details {
        serde_json::Value::Null => None,
        other => Some(other),
    };
    let message = AgentMessage::ToolResult {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: ToolResultContent {
            blocks: outcome.result.content,
        },
        is_error: outcome.is_error,
        // Carry the row-caption prose ("Ran `ls`.", "Wrote
        // `index.html` (4 KB).") into the persisted history so
        // history-aware plugins (working_memory_anchor, smart_context,
        // history_repair) see the same prose the UI renders without
        // having to walk content blocks past densification headers.
        narration: outcome.result.narration,
        // Carry the host-side structured payload so downstream plugins
        // (delivery gates, artifact dispatchers, …) can read canonical
        // fields without text-grepping the prose body. Stripped from
        // provider wire formats — the model still sees `content` only.
        details,
        timestamp: Some(now_ms()),
    };
    // Instrumentation: the post-`AfterToolCall` boundary is where any
    // plugin-driven `override_result` has already landed. Logging the
    // final content text head/tail at this point lets
    // `RUST_LOG=clark_agent::exec::tool_result_built=debug` captures show
    // what actually enters `messages` per turn — useful for triangulating
    // any divergence between a tool's emitted args and the user-visible
    // result a downstream terminal walker later selects.
    if let AgentMessage::ToolResult {
        content,
        is_error,
        tool_call_id,
        tool_name,
        ..
    } = &message
    {
        let plain = content.plain_text();
        let (head, tail) = head_tail_for_log(&plain);
        tracing::debug!(
            target: "clark_agent::exec::tool_result_built",
            tool_call_id = %tool_call_id,
            tool_name = %tool_name,
            is_error = *is_error,
            content_len = plain.len(),
            content_head = %head,
            content_tail = %tail,
            "outcome_to_message wrote ToolResult into messages"
        );
    }
    message
}

const TOOL_RESULT_LOG_HEAD: usize = 200;
const TOOL_RESULT_LOG_TAIL: usize = 200;

/// Head/tail snippets of a tool-result text for diagnostic logging.
/// Avoids dumping multi-KB tool outputs into the trace stream while
/// still making divergence between two snapshots of the "same" text
/// visible at a glance.
fn head_tail_for_log(text: &str) -> (String, String) {
    if text.len() <= TOOL_RESULT_LOG_HEAD + TOOL_RESULT_LOG_TAIL {
        return (text.to_string(), String::new());
    }
    let head_end = char_boundary_at_or_before(text, TOOL_RESULT_LOG_HEAD);
    let tail_start = char_boundary_at_or_after(text, text.len() - TOOL_RESULT_LOG_TAIL);
    (text[..head_end].to_string(), text[tail_start..].to_string())
}

fn char_boundary_at_or_before(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn char_boundary_at_or_after(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while idx < text.len() && !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

async fn emit_tool_start(config: &LoopConfig, call: &ToolCall) {
    let event = AgentEvent::ToolExecutionStart {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        args: call.arguments.clone(),
    };
    config.event_sink.emit(event.clone()).await;
    for o in &config.plugins.event_observer {
        o.on_event(&event).await;
    }
}

/// Build a human-readable, model-recoverable error for an argument
/// payload that failed JSON parsing in the provider stream layer. Shape
/// the message so the model knows (1) it was a syntax problem, not a
/// schema problem, (2) what raw text it produced, and (3) what to do
/// next. Truncate the raw payload to keep error contexts bounded.
fn format_arg_parse_error(tool_name: &str, parse_err: &str, raw: &str) -> String {
    const RAW_MAX: usize = 1024;
    let raw_snippet = if raw.len() > RAW_MAX {
        format!(
            "{}…<{} bytes truncated>",
            &raw[..RAW_MAX],
            raw.len() - RAW_MAX
        )
    } else {
        raw.to_string()
    };
    format!(
        "Tool `{tool_name}` arguments were not valid JSON: {parse_err}. \
         You sent (raw): {raw_snippet}. \
         Re-emit the call with a JSON object matching the tool's schema; \
         this is a syntax error in your tool-call arguments, not a problem \
         with the file or the runtime."
    )
}

async fn emit_tool_end(config: &LoopConfig, call: &ToolCall, outcome: &FinalizedOutcome) {
    let event = AgentEvent::ToolExecutionEnd {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        result: outcome.result.clone(),
        is_error: outcome.is_error,
    };
    config.event_sink.emit(event.clone()).await;
    for o in &config.plugins.event_observer {
        o.on_event(&event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolResultBlock;
    use std::sync::Arc;

    struct LimitTool {
        name: &'static str,
        counts: bool,
        vote_counts: bool,
        parallel_safe: bool,
    }

    #[async_trait::async_trait]
    impl AgentTool for LimitTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn parameters_schema(&self) -> Value {
            json!({"type": "object"})
        }

        fn counts_toward_tool_call_limit(&self) -> bool {
            self.counts
        }

        fn parallel_safe_per_turn(&self) -> bool {
            self.parallel_safe
        }

        fn counts_toward_termination_vote(&self) -> bool {
            self.vote_counts
        }

        async fn execute(
            &self,
            _call_id: &str,
            _args: Value,
            _signal: CancellationToken,
            _update: mpsc::UnboundedSender<ToolResult>,
        ) -> Result<ToolResult, ToolError> {
            unreachable!("split tests do not execute tools")
        }
    }

    fn registry() -> crate::tool::ToolRegistry {
        // Same registry the call-limit tests use plus the
        // termination-vote opt-out: `message_info` is advisory, the
        // other tools count.
        crate::tool::ToolRegistry::new()
            .with(Arc::new(LimitTool {
                name: "message_info",
                counts: false,
                vote_counts: false,
                parallel_safe: false,
            }))
            .with(Arc::new(LimitTool {
                name: "browser_navigate",
                counts: true,
                vote_counts: true,
                parallel_safe: true,
            }))
            .with(Arc::new(LimitTool {
                name: "browser_capture",
                counts: true,
                vote_counts: true,
                parallel_safe: true,
            }))
            .with(Arc::new(LimitTool {
                name: "browser_inspect",
                counts: true,
                vote_counts: true,
                parallel_safe: true,
            }))
            .with(Arc::new(LimitTool {
                name: "shell",
                counts: true,
                vote_counts: true,
                parallel_safe: false,
            }))
            .with(Arc::new(LimitTool {
                name: "message_result",
                counts: true,
                vote_counts: true,
                parallel_safe: false,
            }))
            .with(Arc::new(LimitTool {
                name: "message_ask",
                counts: true,
                vote_counts: true,
                parallel_safe: false,
            }))
            .with(Arc::new(LimitTool {
                name: "web_search",
                counts: true,
                vote_counts: true,
                parallel_safe: true,
            }))
            .with(Arc::new(LimitTool {
                name: "file_read",
                counts: true,
                vote_counts: true,
                parallel_safe: true,
            }))
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: format!("tc-{name}"),
            name: name.to_string(),
            arguments: Value::Null,
        }
    }

    fn names(calls: &[ToolCall]) -> Vec<&str> {
        calls.iter().map(|call| call.name.as_str()).collect()
    }

    #[test]
    fn progress_only_tools_do_not_starve_first_work_tool() {
        let registry = registry();
        let (executable, unexecuted, max) = split_tool_calls_for_execution(
            vec![call("message_info"), call("browser_navigate")],
            &registry,
            Some(1),
        );

        assert_eq!(max, Some(1));
        assert_eq!(names(&executable), vec!["message_info", "browser_navigate"]);
        assert!(unexecuted.is_empty());
    }

    #[test]
    fn extra_limit_counted_tools_still_get_synthetic_errors() {
        let registry = registry();
        let (executable, unexecuted, max) = split_tool_calls_for_execution(
            vec![call("message_info"), call("shell"), call("message_result")],
            &registry,
            Some(1),
        );

        assert_eq!(max, Some(1));
        assert_eq!(names(&executable), vec!["message_info", "shell"]);
        assert_eq!(names(&unexecuted), vec!["message_result"]);
    }

    #[test]
    fn parallel_safe_reads_do_not_burn_the_per_turn_cap() {
        // Two web_searches + one browser_navigate in a single turn:
        // before this change the second web_search would be dropped with
        // "only the first 1 call can run". After, the parallel-safe
        // reads execute alongside the one counted work tool.
        let registry = registry();
        let (executable, unexecuted, max) = split_tool_calls_for_execution(
            vec![
                call("web_search"),
                call("web_search"),
                call("browser_navigate"),
            ],
            &registry,
            Some(1),
        );

        assert_eq!(max, Some(1));
        assert_eq!(
            names(&executable),
            vec!["web_search", "web_search", "browser_navigate"]
        );
        assert!(
            unexecuted.is_empty(),
            "unexecuted: {:?}",
            names(&unexecuted)
        );
    }

    #[test]
    fn parallel_safe_reads_do_not_compete_with_a_write_for_the_cap() {
        // shell (write) still gets its single slot; the parallel-safe
        // reads pass through. A second shell would still be dropped.
        let registry = registry();
        let (executable, unexecuted, max) = split_tool_calls_for_execution(
            vec![
                call("file_read"),
                call("file_read"),
                call("shell"),
                call("shell"),
            ],
            &registry,
            Some(1),
        );

        assert_eq!(max, Some(1));
        assert_eq!(names(&executable), vec!["file_read", "file_read", "shell"]);
        assert_eq!(names(&unexecuted), vec!["shell"]);
    }

    #[test]
    fn browser_tools_do_not_burn_the_per_turn_cap() {
        // Browser tools require exclusive sandbox access, so the
        // executor still runs this batch sequentially. They are
        // nevertheless safe to admit together in one assistant turn:
        // a model often opens two related URLs, captures one page, and
        // inspects another before yielding. The per-turn cap should
        // not drop the later browser calls.
        let registry = registry();
        let (executable, unexecuted, max) = split_tool_calls_for_execution(
            vec![
                call("browser_navigate"),
                call("browser_navigate"),
                call("browser_capture"),
                call("browser_inspect"),
                call("shell"),
            ],
            &registry,
            Some(1),
        );

        assert_eq!(max, Some(1));
        assert_eq!(
            names(&executable),
            vec![
                "browser_navigate",
                "browser_navigate",
                "browser_capture",
                "browser_inspect",
                "shell",
            ]
        );
        assert!(
            unexecuted.is_empty(),
            "unexecuted: {:?}",
            names(&unexecuted)
        );
    }

    #[test]
    fn malformed_calls_do_not_burn_the_cap_or_preempt_real_work() {
        // A call that resolves to no registered tool (unknown name, or an
        // empty/blank name from a streaming glitch) does no real work — it
        // only yields a synthetic "no such tool" error in `prepare_call`. It
        // must NOT spend the turn's slot, or it bumps a real call into the
        // unexecuted bin. Regression for a production case where, under
        // `max_tool_calls_per_turn = 1`, an empty-name call preempted the
        // model's real next call.
        let registry = registry();

        // Unknown name first, real counting tool second: both run; nothing deferred.
        let (executable, unexecuted, _) = split_tool_calls_for_execution(
            vec![call("missing"), call("shell")],
            &registry,
            Some(1),
        );
        assert_eq!(names(&executable), vec!["missing", "shell"]);
        assert!(
            unexecuted.is_empty(),
            "real work must not be preempted by an unknown name: {:?}",
            names(&unexecuted)
        );

        // Empty name first (the prod glitch shape): the real call still runs.
        let (executable, unexecuted, _) =
            split_tool_calls_for_execution(vec![call(""), call("shell")], &registry, Some(1));
        assert_eq!(names(&executable), vec!["", "shell"]);
        assert!(
            unexecuted.is_empty(),
            "empty-name glitch must not preempt real work: {:?}",
            names(&unexecuted)
        );

        // Two real counting tools: the cap still bites — the second is deferred.
        let (executable, unexecuted, _) =
            split_tool_calls_for_execution(vec![call("shell"), call("shell")], &registry, Some(1));
        assert_eq!(names(&executable), vec!["shell"]);
        assert_eq!(names(&unexecuted), vec!["shell"]);
    }

    #[test]
    fn compute_batch_terminate_passes_when_only_advisory_siblings_dont_vote() {
        // Some models tail a terminating delivery call with a polite,
        // advisory sign-off call that opts out of the termination vote
        // (`counts_toward_termination_vote == false`). Under a strict
        // every-result-must-vote rule the trailing advisory call
        // (terminate=false) would block termination and the run would
        // grind to its iteration cap. With the advisory opt-out the
        // batch terminates on the strength of the delivery call alone.
        let registry = registry();
        let votes = [("message_result", true), ("message_info", false)];
        assert!(compute_batch_terminate(
            &registry,
            votes.iter().map(|(n, t)| (*n, *t))
        ));
    }

    #[test]
    fn compute_batch_terminate_fails_when_any_counted_tool_did_not_vote_terminate() {
        let registry = registry();
        // `message_result` voted yes, but a real work tool (`shell`)
        // is still mid-flight or didn't vote — keep running.
        let votes = [("message_result", true), ("shell", false)];
        assert!(!compute_batch_terminate(
            &registry,
            votes.iter().map(|(n, t)| (*n, *t))
        ));
    }

    #[test]
    fn compute_batch_terminate_returns_false_for_all_advisory_batches() {
        // An all-`message_info` batch must NEVER end the run; progress
        // notes are status, not termination, even when the model
        // emits several in a row.
        let registry = registry();
        let votes = [("message_info", false), ("message_info", false)];
        assert!(!compute_batch_terminate(
            &registry,
            votes.iter().map(|(n, t)| (*n, *t))
        ));
    }

    #[test]
    fn compute_batch_terminate_returns_false_for_empty_batch() {
        let registry = registry();
        let votes: Vec<(&str, bool)> = Vec::new();
        assert!(!compute_batch_terminate(&registry, votes.into_iter()));
    }

    #[test]
    fn compute_batch_terminate_treats_unknown_tools_as_counted() {
        // Unknown / unregistered tool names default to counted so a
        // stray call cannot accidentally terminate the run by being
        // silently classified as advisory.
        let registry = registry();
        // `message_result` voted yes, but an unknown tool emitted
        // `terminate=false`. Unknown counts → must not terminate.
        let votes = [("message_result", true), ("ghost_tool", false)];
        assert!(!compute_batch_terminate(
            &registry,
            votes.iter().map(|(n, t)| (*n, *t))
        ));

        // And the symmetric case: an unknown tool that voted yes,
        // alongside `message_result` voting yes → still counted, so
        // the batch terminates.
        let votes = [("message_result", true), ("ghost_tool", true)];
        assert!(compute_batch_terminate(
            &registry,
            votes.iter().map(|(n, t)| (*n, *t))
        ));
    }

    #[test]
    fn compute_batch_terminate_passes_when_message_ask_is_only_counted_terminator() {
        // Symmetric to the message_result case: message_ask (also a
        // terminating tool) tailed by message_info still terminates.
        let registry = registry();
        let votes = [("message_ask", true), ("message_info", false)];
        assert!(compute_batch_terminate(
            &registry,
            votes.iter().map(|(n, t)| (*n, *t))
        ));
    }

    #[test]
    fn head_tail_for_log_returns_full_text_when_short() {
        // Short payloads (≤ HEAD+TAIL) round-trip in `head` with an
        // empty `tail` so the trace line stays compact and the
        // diagnostic reader doesn't have to reconstruct the full
        // string from two halves when there's nothing to truncate.
        let (head, tail) = head_tail_for_log("hello");
        assert_eq!(head, "hello");
        assert_eq!(tail, "");
    }

    #[test]
    fn head_tail_for_log_truncates_long_text_with_head_and_tail() {
        let payload: String = "abc".repeat(500);
        assert!(payload.len() > TOOL_RESULT_LOG_HEAD + TOOL_RESULT_LOG_TAIL);
        let (head, tail) = head_tail_for_log(&payload);
        assert_eq!(head.len(), TOOL_RESULT_LOG_HEAD);
        assert_eq!(tail.len(), TOOL_RESULT_LOG_TAIL);
        // First/last bytes must come from the original — guards
        // against a regression where the helper accidentally re-orders
        // or drops the boundary characters.
        assert!(payload.starts_with(&head));
        assert!(payload.ends_with(&tail));
    }

    #[test]
    fn head_tail_for_log_respects_utf8_char_boundaries() {
        // Multi-byte chars must not be split mid-codepoint or the
        // tracing macro would panic (and instrumentation would crash
        // the loop). Build a payload long enough to truncate, padded
        // with multi-byte chars at both boundary regions.
        let mid = "πλάκα".repeat(50); // each char is 2 bytes
        let prefix: String = "x".repeat(150);
        let suffix: String = "y".repeat(150);
        let payload = format!("{prefix}{mid}{suffix}");
        let (head, tail) = head_tail_for_log(&payload);
        // Validity assertions: both slices are valid UTF-8 (they
        // already are since they came from `&str`), and the boundary
        // is on a char boundary in the original. Round-trip check:
        // the head must be a prefix of payload and tail a suffix.
        assert!(payload.starts_with(&head));
        assert!(payload.ends_with(&tail));
        // Head capped at HEAD bytes (last char-boundary at or before).
        assert!(head.len() <= TOOL_RESULT_LOG_HEAD);
        assert!(tail.len() <= TOOL_RESULT_LOG_TAIL + 1); // +1 for boundary slack
    }

    #[test]
    fn unexecuted_message_mentions_progress_only_calls_when_present() {
        let result = unexecuted_tool_call_result(3, 2, 1);
        let text = match result.content.first() {
            Some(ToolResultBlock::Text(text)) => text.text.as_str(),
            _ => panic!("expected text result"),
        };

        assert!(text.contains("2 limit-counted tool calls"));
        assert!(text.contains("3 tool calls total, including progress-only calls"));
        assert_eq!(
            result
                .details
                .get("limit_counted_tool_calls")
                .and_then(Value::as_u64),
            Some(2)
        );
    }
}
