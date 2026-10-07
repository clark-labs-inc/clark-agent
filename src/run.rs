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

mod provider_recovery;
mod assistant_stream;
use assistant_stream::{stream_with_overflow_recovery, stream_assistant_response};
#[cfg(test)]
use assistant_stream::loop_error_from_stream_kind;
mod loop_driver;
mod tool_gates;
use tool_gates::collect_tool_allowlist_with_events;
use loop_driver::inner_run;
mod transport_recovery;
use provider_recovery::stream_with_max_tokens_recovery;
use transport_recovery::context_with_zero_output_transport_recovery;

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

#[cfg(test)]
mod tests;
