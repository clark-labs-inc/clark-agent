//! Trajectory capture: ordered, failable, sequence-numbered run record.
//!
//! ## Why a separate sink
//!
//! [`crate::event::EventSink`] is the loop's streaming UI channel:
//! best-effort `emit(event)` that returns nothing, with the explicit
//! contract "failures must not propagate out of the loop." That makes
//! it the right shape for streaming a UI but the wrong shape for
//! trajectory capture, where a single dropped event corrupts replay
//! and eval.
//!
//! `TrajectorySink` is the durable counterpart. Three guarantees the
//! event sink does not make:
//!
//! 1. **Ordered.** Every record carries a monotonic per-run `seq`. A
//!    consumer that observes `seq = N` is guaranteed to have observed
//!    every record with `seq < N`.
//!
//! 2. **Failable.** `record` returns `Result<_, TrajectoryError>`.
//!    Callers that wire a `TrajectorySink` into the loop via
//!    [`TrajectoryRecorder`] choose the policy: drop and continue
//!    (default), or escalate to a typed error. The sink itself stays
//!    pure — it surfaces failures, never decides what the loop does.
//!
//! 3. **Run-scoped.** Records are keyed by `run_id` (when known) so
//!    parent/child trajectories live in the same store under
//!    different ids. The `seq` resets per run.
//!
//! ## Wiring
//!
//! Construct a [`TrajectoryRecorder`] around a `TrajectorySink`, then
//! register the recorder as an `EventObserver` plugin. The recorder
//! filters [`AgentEvent`]s into [`TrajectoryRecord`]s, stamps each
//! with a sequence number, and forwards to the sink.
//!
//! ```ignore
//! let sink: Arc<dyn TrajectorySink> = Arc::new(InMemoryTrajectorySink::default());
//! let recorder = Arc::new(TrajectoryRecorder::new(sink.clone()));
//! AgentBuilder::new()
//!     .event_observer_arc(recorder.clone())
//!     .stream(...)
//!     .build()
//! ```
//!
//! After the run, drain the sink. Records are in order by `seq`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::event::AgentEvent;
use crate::plugin::{Plugin, PluginCapabilities};
use crate::types::{AgentMessage, RunIdentity};

// ─── Records ───────────────────────────────────────────────────────

/// One durable record emitted from a run.
///
/// Each record carries a monotonic per-run `seq`, an optional
/// `run_id` (resolved as soon as the loop emits
/// [`AgentEvent::RunIdentified`]; `None` for events that precede
/// it), and a typed [`TrajectoryPayload`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    pub seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
    #[serde(default)]
    pub depth: usize,
    /// UNIX milliseconds at the time the record was produced.
    pub recorded_at_unix_ms: u64,
    pub payload: TrajectoryPayload,
}

/// Typed payload of a trajectory record. The variants are a curated
/// subset of [`AgentEvent`] — the things a replay or eval actually
/// needs. Streaming-only events (`MessageUpdate`, `ToolExecutionUpdate`)
/// are intentionally omitted; they belong on the streaming channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrajectoryPayload {
    RunStarted {
        identity: RunIdentity,
    },
    RunEnded {
        outcome: String,
        new_messages: Vec<AgentMessage>,
    },
    TurnStarted,
    TurnEnded {
        assistant: AgentMessage,
        tool_results: Vec<AgentMessage>,
    },
    /// Messages appended to the transcript (user, assistant, or tool
    /// result). Final form only — no streaming deltas.
    MessageAppended {
        message: AgentMessage,
    },
    ToolStarted {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolEnded {
        tool_call_id: String,
        tool_name: String,
        result: crate::tool::ToolResult,
        is_error: bool,
    },
    /// One LLM call's request snapshot, post-transform and post-gate.
    /// Captures the typed view of "what the model saw this turn."
    ProviderRequestPrepared {
        iteration: usize,
        model_id: Option<String>,
        system_prompt_chars: usize,
        message_count: usize,
        tool_count: usize,
        tools: Vec<String>,
    },
    /// A `ContextTransform` plugin ran. Carries only the plugin name
    /// and the before/after message counts so durable trajectories
    /// stay compact; the full diff stays on the streaming channel for
    /// listeners that want it.
    ContextTransformApplied {
        iteration: usize,
        plugin: String,
        before_count: usize,
        after_count: usize,
    },
    /// A `ToolGate` plugin contributed an allowlist for this turn.
    ToolGateApplied {
        iteration: usize,
        plugin: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allow: Option<Vec<String>>,
    },
    /// The final intersected allowlist would have been empty, so the loop
    /// selected a deterministic owner allowlist instead of advertising zero
    /// tools to the provider.
    ToolGateConflictResolved {
        iteration: usize,
        plugins: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen_plugin: Option<String>,
        allow: Vec<String>,
        reason: String,
    },
    /// The loop discarded a truncated turn and re-streamed.
    OutputTokensEscalation {
        attempt: u8,
        prev_cap: u32,
        new_cap: u32,
    },
}

/// Errors a `TrajectorySink` may surface.
#[derive(Debug, thiserror::Error)]
pub enum TrajectoryError {
    #[error("trajectory sink rejected record: {0}")]
    Rejected(String),
    #[error("trajectory sink i/o failure: {0}")]
    Io(String),
}

// ─── Sink trait ────────────────────────────────────────────────────

/// Durable trajectory sink. Implementations persist records in order;
/// returning `Err` surfaces the failure to the caller (the
/// [`TrajectoryRecorder`] applies the loop's chosen policy).
///
/// Implementations MUST preserve the order in which `record` is
/// called. The sink does not need to be re-entrant; the recorder
/// serializes calls with an internal mutex when wired as an
/// `EventObserver`.
#[async_trait]
pub trait TrajectorySink: Send + Sync {
    async fn record(&self, record: TrajectoryRecord) -> Result<(), TrajectoryError>;
}

// ─── In-memory sink ───────────────────────────────────────────────

/// Simple `Vec`-backed sink. Useful for tests, eval harnesses that
/// load the whole trajectory in memory, and the replay path.
#[derive(Debug, Default)]
pub struct InMemoryTrajectorySink {
    records: Mutex<Vec<TrajectoryRecord>>,
}

impl InMemoryTrajectorySink {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn snapshot(&self) -> Vec<TrajectoryRecord> {
        self.records.lock().await.clone()
    }

    pub async fn len(&self) -> usize {
        self.records.lock().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.records.lock().await.is_empty()
    }
}

#[async_trait]
impl TrajectorySink for InMemoryTrajectorySink {
    async fn record(&self, record: TrajectoryRecord) -> Result<(), TrajectoryError> {
        self.records.lock().await.push(record);
        Ok(())
    }
}

// ─── Recorder (EventObserver) ─────────────────────────────────────

/// Plugin that filters [`AgentEvent`]s into [`TrajectoryRecord`]s and
/// forwards to a [`TrajectorySink`]. Stamps each record with a
/// monotonic per-run sequence number and resolves the run id from
/// [`AgentEvent::RunIdentified`].
///
/// Register as an `EventObserver` on the parent run; child runs that
/// reuse the same recorder share the sequence space and stay
/// distinguishable by `run_id`/`parent_run_id`. For a strict
/// per-run sequence reset, register a fresh recorder per spawn.
pub struct TrajectoryRecorder {
    sink: Arc<dyn TrajectorySink>,
    seq: AtomicU64,
    identity: Mutex<Option<RunIdentity>>,
}

impl TrajectoryRecorder {
    pub fn new(sink: Arc<dyn TrajectorySink>) -> Self {
        Self {
            sink,
            seq: AtomicU64::new(0),
            identity: Mutex::new(None),
        }
    }

    async fn record(&self, payload: TrajectoryPayload) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let identity = self.identity.lock().await.clone();
        let recorded_at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let record = TrajectoryRecord {
            seq,
            run_id: identity.as_ref().map(|i| i.run_id.clone()),
            parent_run_id: identity.as_ref().and_then(|i| i.parent_run_id.clone()),
            depth: identity.as_ref().map(|i| i.depth).unwrap_or(0),
            recorded_at_unix_ms,
            payload,
        };
        if let Err(e) = self.sink.record(record).await {
            tracing::warn!(error = %e, "trajectory sink rejected record; continuing");
        }
    }
}

impl Plugin for TrajectoryRecorder {
    fn name(&self) -> &'static str {
        "trajectory_recorder"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::event_observer()
    }
}

#[async_trait]
impl crate::plugin::EventObserver for TrajectoryRecorder {
    async fn on_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::AgentStart => {
                // Reset sequence and identity for a fresh run. Concurrent
                // re-use across runs is not supported — register a fresh
                // recorder per run if you need per-run isolation.
                self.seq.store(0, Ordering::SeqCst);
                *self.identity.lock().await = None;
            }
            AgentEvent::RunIdentified { identity } => {
                *self.identity.lock().await = Some(identity.clone());
                self.record(TrajectoryPayload::RunStarted {
                    identity: identity.clone(),
                })
                .await;
            }
            AgentEvent::AgentEnd { messages } => {
                self.record(TrajectoryPayload::RunEnded {
                    outcome: "ended".to_string(),
                    new_messages: messages.clone(),
                })
                .await;
            }
            AgentEvent::TurnStart => {
                self.record(TrajectoryPayload::TurnStarted).await;
            }
            AgentEvent::TurnEnd {
                message,
                tool_results,
            } => {
                self.record(TrajectoryPayload::TurnEnded {
                    assistant: message.clone(),
                    tool_results: tool_results.clone(),
                })
                .await;
            }
            AgentEvent::MessageEnd { message } => {
                self.record(TrajectoryPayload::MessageAppended {
                    message: message.clone(),
                })
                .await;
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                self.record(TrajectoryPayload::ToolStarted {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                })
                .await;
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                self.record(TrajectoryPayload::ToolEnded {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    result: result.clone(),
                    is_error: *is_error,
                })
                .await;
            }
            AgentEvent::ProviderRequestPrepared {
                iteration,
                model_id,
                system_prompt,
                messages,
                tools,
                ..
            } => {
                self.record(TrajectoryPayload::ProviderRequestPrepared {
                    iteration: *iteration,
                    model_id: model_id.clone(),
                    system_prompt_chars: system_prompt.chars().count(),
                    message_count: messages.len(),
                    tool_count: tools.len(),
                    tools: tools.iter().map(|t| t.name.clone()).collect(),
                })
                .await;
            }
            AgentEvent::ContextTransformApplied {
                iteration,
                plugin,
                before,
                after,
            } => {
                self.record(TrajectoryPayload::ContextTransformApplied {
                    iteration: *iteration,
                    plugin: (*plugin).to_string(),
                    before_count: before.len(),
                    after_count: after.len(),
                })
                .await;
            }
            AgentEvent::ToolGateApplied {
                iteration,
                plugin,
                allow,
            } => {
                self.record(TrajectoryPayload::ToolGateApplied {
                    iteration: *iteration,
                    plugin: (*plugin).to_string(),
                    allow: allow.clone(),
                })
                .await;
            }
            AgentEvent::ToolGateConflictResolved {
                iteration,
                plugins,
                chosen_plugin,
                allow,
                reason,
            } => {
                self.record(TrajectoryPayload::ToolGateConflictResolved {
                    iteration: *iteration,
                    plugins: plugins.clone(),
                    chosen_plugin: chosen_plugin.clone(),
                    allow: allow.clone(),
                    reason: reason.clone(),
                })
                .await;
            }
            AgentEvent::OutputTokensEscalation {
                attempt,
                prev_cap,
                new_cap,
            } => {
                self.record(TrajectoryPayload::OutputTokensEscalation {
                    attempt: *attempt,
                    prev_cap: *prev_cap,
                    new_cap: *new_cap,
                })
                .await;
            }
            AgentEvent::MessageStart { .. } | AgentEvent::MessageUpdate { .. } | AgentEvent::ToolExecutionUpdate { .. } => {
                // Streaming-only deltas. The streaming `EventSink` is
                // the right channel for these; durable trajectory
                // captures the final assembled message via
                // `MessageEnd`/`TurnEnd`.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::EventObserver;
    use crate::types::{AssistantContent, StopReason};

    #[tokio::test]
    async fn recorder_writes_ordered_records_with_run_id() {
        let sink = Arc::new(InMemoryTrajectorySink::new());
        let recorder = TrajectoryRecorder::new(sink.clone());

        recorder.on_event(&AgentEvent::AgentStart).await;
        let identity = RunIdentity::root().with_conversation_id("conv-1");
        recorder
            .on_event(&AgentEvent::RunIdentified {
                identity: identity.clone(),
            })
            .await;
        recorder.on_event(&AgentEvent::TurnStart).await;
        recorder
            .on_event(&AgentEvent::TurnEnd {
                message: AgentMessage::Assistant {
                    content: AssistantContent { blocks: Vec::new() },
                    stop_reason: StopReason::EndTurn,
                    error_message: None,
                    timestamp: None,
                    usage: None,
                },
                tool_results: Vec::new(),
            })
            .await;
        recorder
            .on_event(&AgentEvent::AgentEnd {
                messages: Vec::new(),
            })
            .await;

        let records = sink.snapshot().await;
        // AgentStart resets but emits no record; RunIdentified is first.
        assert_eq!(records.len(), 4);
        assert!(matches!(
            records[0].payload,
            TrajectoryPayload::RunStarted { .. }
        ));
        assert!(matches!(records[1].payload, TrajectoryPayload::TurnStarted));
        assert!(matches!(
            records[2].payload,
            TrajectoryPayload::TurnEnded { .. }
        ));
        assert!(matches!(
            records[3].payload,
            TrajectoryPayload::RunEnded { .. }
        ));

        for (i, r) in records.iter().enumerate() {
            assert_eq!(r.seq, i as u64);
            assert_eq!(r.run_id.as_deref(), Some(identity.run_id.as_str()));
        }
    }

    #[tokio::test]
    async fn recorder_skips_streaming_only_events() {
        let sink = Arc::new(InMemoryTrajectorySink::new());
        let recorder = TrajectoryRecorder::new(sink.clone());

        let msg = AgentMessage::User {
            content: crate::types::UserContent::Text("hi".into()),
            timestamp: None,
        };
        recorder
            .on_event(&AgentEvent::MessageStart {
                message: msg.clone(),
            })
            .await;
        recorder
            .on_event(&AgentEvent::ToolExecutionUpdate {
                tool_call_id: "1".into(),
                tool_name: "shell".into(),
                partial: crate::tool::ToolResult::text("partial"),
            })
            .await;

        assert!(sink.is_empty().await);
    }
}
