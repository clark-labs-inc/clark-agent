//! Plugin extension points.
//!
//! All cross-cutting concerns plug into the loop through these traits.
//! No inline `if special_case_X` branches inside the loop; keep hook
//! discipline in explicit extension points.
//!
//! Two families:
//!
//! 1. **Capability traits** (this module) — `BeforeToolCall`,
//!    `AfterToolCall`, `ContextTransform`, `EventObserver`,
//!    `SteeringSource`, `FollowUpSource`. Each is narrow: a hook that
//!    needs the assistant message gets the assistant message, never a
//!    fat `&mut LoopState`. New capabilities add a new trait; they do
//!    not widen an existing one.
//!
//! 2. **`Plugin` marker** — a single registry entry that may implement
//!    one or more capability traits. `AgentBuilder` holds plugins as
//!    `Arc<dyn Plugin>` and dispatches to whichever capabilities the
//!    plugin declares via [`Plugin::capabilities`].

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::event::AgentEvent;
use crate::tokens::{TokenEstimator, CHAR_HEURISTIC};
use crate::tool::{ToolCall, ToolResult};
use crate::types::{AgentMessage, AssistantContent, ContextObservation, Usage};

// ─── Plugin marker ─────────────────────────────────────────────────

/// A registered extension. Each plugin declares which capability traits
/// it implements via [`PluginCapabilities`].
///
/// A plugin can implement any subset of: `BeforeToolCall`, `AfterToolCall`,
/// `ContextTransform`, `EventObserver`, `SteeringSource`, `FollowUpSource`.
/// The loop's plugin dispatcher iterates registered plugins for each
/// extension point.
pub trait Plugin: Send + Sync + 'static {
    /// Stable identifier for logs and telemetry.
    fn name(&self) -> &'static str;

    /// Which capabilities this plugin implements. Default: none — meaning
    /// pure observation by inheriting from `EventObserver`. Override and
    /// return the relevant set when adding behavior.
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::default()
    }
}

/// Bitset of which extension points a plugin participates in.
///
/// The dispatcher reads this to skip plugins that don't implement a
/// given hook, avoiding wasteful trait-object cast attempts.
///
/// `inheritable_to_child` is the spawn-time signal: when a parent run
/// calls [`crate::LoopConfig::child_builder`], every parent plugin
/// whose capabilities have `inheritable_to_child = true` is carried
/// into the child's plugin registry as-is. Default `false` — plugins
/// that hold conversation-scoped state, mutate parent-only stores, or
/// know about the parent's UI/persistence must opt in explicitly so a
/// child run cannot silently inherit parent identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PluginCapabilities {
    pub before_tool_call: bool,
    pub after_tool_call: bool,
    pub context_transform: bool,
    pub event_observer: bool,
    pub steering: bool,
    pub follow_up: bool,
    pub tool_gate: bool,
    /// When `true`, [`crate::LoopConfig::child_builder`] carries this
    /// plugin into every spawned child run. When `false` (default),
    /// the plugin is parent-only and the caller assembling the child
    /// must register the child-specific equivalent.
    pub inheritable_to_child: bool,
}

impl PluginCapabilities {
    pub fn before_tool_call() -> Self {
        Self {
            before_tool_call: true,
            ..Self::default()
        }
    }
    pub fn after_tool_call() -> Self {
        Self {
            after_tool_call: true,
            ..Self::default()
        }
    }
    pub fn context_transform() -> Self {
        Self {
            context_transform: true,
            ..Self::default()
        }
    }
    pub fn event_observer() -> Self {
        Self {
            event_observer: true,
            ..Self::default()
        }
    }
    pub fn steering() -> Self {
        Self {
            steering: true,
            ..Self::default()
        }
    }
    pub fn follow_up() -> Self {
        Self {
            follow_up: true,
            ..Self::default()
        }
    }
    pub fn tool_gate() -> Self {
        Self {
            tool_gate: true,
            ..Self::default()
        }
    }

    pub fn with_follow_up(mut self) -> Self {
        self.follow_up = true;
        self
    }
    pub fn with_tool_gate(mut self) -> Self {
        self.tool_gate = true;
        self
    }
    /// Mark this plugin as inheritable to child runs spawned via
    /// [`crate::LoopConfig::child_builder`].
    pub fn with_inheritable_to_child(mut self) -> Self {
        self.inheritable_to_child = true;
        self
    }
}

// ─── BeforeToolCall ────────────────────────────────────────────────

/// Read-only context handed to a `BeforeToolCall` hook.
///
/// Narrow on purpose: the hook gets the assistant message that requested
/// the call, the call itself, and the validated arguments. It does not
/// get a fat `&mut LoopState`.
pub struct BeforeToolCallContext<'a> {
    pub assistant_message: &'a AgentMessage,
    pub assistant_content: &'a AssistantContent,
    pub tool_call: &'a ToolCall,
    pub args: &'a Value,
    pub messages: &'a [AgentMessage],
}

/// Decision returned by a `BeforeToolCall` hook.
///
/// `block: true` short-circuits execution; the loop synthesizes an error
/// tool result with `reason` (or a default message) and emits a
/// `ToolExecutionEnd` with `is_error = true`.
///
/// `replacement_args` is a narrow normalization boundary for facts already
/// proven by typed history. It lets a plugin carry transcript-owned metadata
/// (for example, an evidenced delivery receipt) into execution without asking
/// the model to copy the same value into another JSON shape. The loop validates
/// replacement arguments before execution and passes them to later hooks.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolDecision {
    pub block: bool,
    pub reason: Option<String>,
    pub details: Option<Value>,
    pub replacement_args: Option<Value>,
}

impl BeforeToolDecision {
    pub fn allow() -> Self {
        Self::default()
    }
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            block: true,
            reason: Some(reason.into()),
            details: None,
            replacement_args: None,
        }
    }

    pub fn block_with_details(reason: impl Into<String>, details: Value) -> Self {
        Self {
            block: true,
            reason: Some(reason.into()),
            details: Some(details),
            replacement_args: None,
        }
    }

    /// Continue with normalized arguments derived from typed runtime context.
    /// The replacement is revalidated before execution.
    pub fn allow_with_args(args: Value) -> Self {
        Self {
            replacement_args: Some(args),
            ..Self::default()
        }
    }
}

/// Hook that runs after argument validation, before tool execution.
///
/// Cheap and side-effect-free: no I/O, no LLM calls, no spawning, no
/// state mutation. Pure transform of context → decision.
#[async_trait]
pub trait BeforeToolCall: Plugin {
    async fn on_before_tool_call(&self, ctx: BeforeToolCallContext<'_>) -> BeforeToolDecision;
}

// ─── AfterToolCall ─────────────────────────────────────────────────

/// Read-only context handed to an `AfterToolCall` hook.
///
/// Includes the executed result so the hook can override it. The hook
/// cannot re-execute the tool; it can only transform the result the
/// model will see.
pub struct AfterToolCallContext<'a> {
    pub assistant_message: &'a AgentMessage,
    pub tool_call: &'a ToolCall,
    pub args: &'a Value,
    pub result: &'a ToolResult,
    pub is_error: bool,
    pub messages: &'a [AgentMessage],
}

/// Override returned by an `AfterToolCall` hook. Each field is opt-in:
/// omitted fields keep the original tool result. No deep merge.
#[derive(Debug, Clone, Default)]
pub struct AfterToolDecision {
    pub result: Option<ToolResult>,
    pub mark_error: Option<bool>,
    pub terminate: Option<bool>,
}

impl AfterToolDecision {
    pub fn passthrough() -> Self {
        Self::default()
    }

    pub fn override_result(result: ToolResult) -> Self {
        Self {
            result: Some(result),
            ..Self::default()
        }
    }
}

/// Hook that runs after tool execution, before the result is appended to
/// history. May override the result, flip the error flag, or vote to
/// terminate.
///
/// Termination semantics are unanimous across the batch: the
/// run only ends when *every* finalized tool result in the batch has
/// `terminate = true`.
#[async_trait]
pub trait AfterToolCall: Plugin {
    async fn on_after_tool_call(&self, ctx: AfterToolCallContext<'_>) -> AfterToolDecision;
}

// ─── ContextTransform ──────────────────────────────────────────────

/// Read-only context handed to a `ContextTransform` hook.
///
/// Carries the cancellation signal plus a few cheap observables that
/// transforms key on (model identity, iteration index, last-turn token
/// usage, the loop's configured token estimator). Gathering these on
/// the hook context — rather than widening the trait one parameter at
/// a time — keeps the trait stable as later compaction layers
/// (per-tool-result cap, cache-aware microcompact, auto-compact) come
/// online.
///
/// New fields are additive: transforms that don't care can ignore them.
pub struct TransformContext<'a> {
    /// Cancellation signal for the current run.
    pub signal: &'a CancellationToken,
    /// Model identifier the run is targeting (e.g. provider/model). May
    /// be empty when the host runtime doesn't surface one — tests,
    /// fixture-replay transports, etc. Plugins that key per-model
    /// behavior should treat empty as "unknown".
    pub model_id: &'a str,
    /// Zero-indexed iteration within the current run. Same semantics as
    /// [`ToolGateContext::iteration`]: the very first LLM call of the
    /// run is `0`.
    pub iteration: usize,
    /// Token usage reported by the provider on the most recent assistant
    /// turn that surfaced a `Usage` block. `None` on the very first turn
    /// or when the provider didn't surface usage. Useful for
    /// cache-aware decisions (read `cache_read_input_tokens` to see if
    /// the prompt prefix actually hit cache last turn).
    pub last_provider_usage: Option<&'a Usage>,
    /// Estimator the loop is configured with. Plugins use this to count
    /// tokens for budgeting and compaction without duplicating the
    /// loop's tokenizer choice.
    pub estimator: &'a dyn TokenEstimator,
}

impl<'a> TransformContext<'a> {
    /// Convenience constructor for tests and ad-hoc callers that don't
    /// have a model id, iteration counter, or usage data. Picks the
    /// default char-heuristic estimator.
    pub fn for_test(signal: &'a CancellationToken) -> Self {
        Self {
            signal,
            model_id: "",
            iteration: 0,
            last_provider_usage: None,
            estimator: &CHAR_HEURISTIC,
        }
    }
}

/// Hook that transforms the message slice before it's converted to the
/// LLM provider format.
///
/// Common use: token-budget pruning. See [`crate::budget`] for the
/// default implementation.
///
/// Contract: must not throw; on failure return the input unchanged.
/// Multiple plugins compose left-to-right.
#[async_trait]
pub trait ContextTransform: Plugin {
    /// Cheap predicate the loop consults before invoking `transform`.
    /// Default returns `true` — preserves existing behavior. Plugins that
    /// can decide locally that they have nothing to do (no browser
    /// snapshots in history, history under budget, idle timer not
    /// elapsed, no queued recovery notice, …) should override to return
    /// `false` in those states.
    ///
    /// When `false`, the loop skips the full message-vec clone + the
    /// `ContextTransformApplied` diff event — eliminating the
    /// per-transform cost on rounds where the plugin is a no-op. This
    /// shows up most clearly in long-running scenarios: with several
    /// transforms installed, each firing hundreds of times as a no-op,
    /// the full before-clone + event emit otherwise happens every time.
    ///
    /// Predicates MUST be O(1) or O(small-constant); a predicate that
    /// itself walks the entire history defeats the optimization.
    fn should_run(&self, _messages: &[AgentMessage], _cx: &TransformContext<'_>) -> bool {
        true
    }

    /// Seal one exact host observation from the final prepared provider view.
    /// The loop visits messages in source order and retains observations on the
    /// response, never in a parallel cache. Return none for sealed history,
    /// compaction, repair and unrelated messages.
    fn history_observation(&self, _message: &AgentMessage) -> Option<ContextObservation> {
        None
    }

    async fn transform(
        &self,
        messages: Vec<AgentMessage>,
        cx: &TransformContext<'_>,
    ) -> Vec<AgentMessage>;
}

// ─── ContextOverflowRecovery ───────────────────────────────────────

/// Recovers from a provider context-window rejection MID-RUN.
///
/// The token-estimate heuristics that drive `ContextTransform` are
/// approximate, so a request can still exceed the model's real window
/// and come back as [`crate::StreamError::ContextOverflow`]. When one is
/// installed, the loop hands this hook the current history, the impl
/// returns a smaller one (typically an aggressive compaction), and the
/// loop retries the SAME LLM call, with the shrunk history persisted into
/// the live transcript so later turns don't immediately re-expand.
///
/// Distinct from [`ContextTransform`] on purpose: this fires ONLY on an
/// overflow (never on every round), its result is written back to the
/// caller's transcript (not just the request clone), and returning an
/// unshrunk history ends recovery rather than spinning.
///
#[async_trait]
pub trait ContextOverflowRecovery: Send + Sync {
    /// Produce a smaller history for the retried request. `cx` carries the
    /// same observables as a [`ContextTransform`] (cancellation signal,
    /// model id, iteration, last provider usage, estimator). Returning a
    /// history no shorter than the input signals "cannot shrink further"
    /// and the loop stops retrying.
    async fn recover(
        &self,
        messages: Vec<AgentMessage>,
        cx: &TransformContext<'_>,
    ) -> Vec<AgentMessage>;

    /// Label for the [`AgentEvent::ContextTransformApplied`] diff event
    /// emitted after a successful shrink, so observers can attribute the
    /// history change.
    ///
    /// [`AgentEvent::ContextTransformApplied`]: crate::AgentEvent::ContextTransformApplied
    fn name(&self) -> &'static str {
        "context_overflow_recovery"
    }
}

// ─── EventObserver ─────────────────────────────────────────────────

/// Pure observation hook. Logs, telemetry, replay writers. Cannot change
/// loop state — the event sink (`crate::event::EventSink`) is the formal
/// channel; this trait exists so plugins can subscribe declaratively
/// alongside their other hooks instead of wiring a separate sink.
#[async_trait]
pub trait EventObserver: Plugin {
    async fn on_event(&self, event: &AgentEvent);
}

// ─── SteeringSource (steer()) ──────────────────────────────────────

/// Source of "steering messages" — extra messages the user / harness
/// wants to inject mid-run.
///
/// The loop calls `next_steering_messages` after the current assistant
/// turn finishes executing its tool calls and before the next LLM call.
/// Returned messages are appended verbatim to the transcript, then the
/// loop continues. Use cases: user typed something while the agent was
/// thinking, harness wants to inject a hint, watchdog wants to force a
/// checkpoint.
///
/// Tool calls already in flight are not interrupted — steering messages
/// land between batches.
#[async_trait]
pub trait SteeringSource: Plugin {
    async fn next_steering_messages(&self) -> Vec<AgentMessage>;
}

// ─── FollowUpSource ────────────────────────────────────────────────

/// Source of "follow-up messages" — extra messages the loop should
/// process after the agent would otherwise stop.
///
/// Distinct from steering: steering is consulted *between batches* and
/// keeps the agent running; follow-up is consulted *after natural stop*
/// and re-starts the agent if there's more to do. Use case: queued user
/// turns that arrived while the previous turn was still running.
#[async_trait]
pub trait FollowUpSource: Plugin {
    async fn next_follow_up_messages(&self) -> Vec<AgentMessage>;
}

mod tool_gate;
pub use tool_gate::{ToolGate, ToolGateContext, ToolGateClass};
mod steering_channel;
pub use steering_channel::{ChannelSteering, SteeringHandle};
mod run_guard;
pub use run_guard::{RunGuard, RunGuardContext, RunGuardViolation};
