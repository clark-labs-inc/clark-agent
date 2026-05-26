//! Graceful turn limit: warn the model before the hard `max_iterations`
//! cap fires.
//!
//! ## Why
//!
//! The hard cap (`LoopConfig::max_iterations`) is a circuit breaker. When
//! it trips today the loop just stops — whatever the model was doing is
//! abandoned. For long subagent runs that means a half-finished
//! investigation, a dangling tool call, or a missing summary. The parent
//! agent gets no signal that the answer is partial.
//!
//! This plugin adds a single steering message a few iterations before the
//! cap: a host-provided wrap-up instruction, defaulting to *"You have used
//! your turn budget. Stop calling work tools and call `message_result`
//! now."* If the model complies, the run ends naturally
//! with a clean close-out and the loop reports
//! [`LoopOutcome::WrappedUp`](crate::run::LoopOutcome::WrappedUp). If the
//! model ignores the warning, the existing hard cap still fires.
//!
//! ## Capabilities
//!
//! - [`EventObserver`] — increments a completed-turn counter on every
//!   `AgentEvent::TurnEnd`. The plugin owns its own counter rather
//!   than reading the loop's because the trait surface intentionally
//!   doesn't expose iteration state to plugins.
//! - [`SteeringSource`] — drains a one-shot wrap-up message once the
//!   counter crosses the soft threshold. Subsequent polls return empty.
//!
//! ## Lifecycle
//!
//! Auto-installed by [`AgentBuilder::build`](crate::config::AgentBuilder::build)
//! when both `max_iterations` and `grace_iterations > 0` are set. Callers
//! never need to register it manually.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::event::AgentEvent;
use crate::plugin::{EventObserver, Plugin, PluginCapabilities, SteeringSource};
use crate::types::AgentMessage;

type MessageProvider = Arc<dyn Fn() -> String + Send + Sync>;
/// Host-supplied callback that returns the desired grace window at
/// fire-check time. Lets hosts scale the wrap-up window to the size of
/// the work in flight (e.g. "leave at least 2 × open_phase_count + 2
/// iterations for plan close-out") without leaking host types into
/// `clark-agent`. The callback is invoked on every steering poll, so
/// keep it cheap: a single `Mutex::lock` + count, no I/O.
type GraceProvider = Arc<dyn Fn() -> usize + Send + Sync>;

/// Wrap-up steering text injected when the soft limit fires.
///
/// Phrased as a directive, not a suggestion — pi-subagents found that
/// hedged language ("you might want to wrap up") is reliably ignored,
/// while a clear stop-and-summarize instruction lands. The closing
/// bullets prompt the model to surface what it accomplished and what
/// remains so the parent agent can act on partial results.
///
/// The wrap-up still has to go through Clark's terminal-message contract:
/// plain assistant text is invisible, so the model must call `message_result`
/// (or `message_ask` if a user answer is genuinely required).
const DEFAULT_WRAP_UP_MESSAGE: &str = "\
You have used your turn budget. Stop calling work tools and deliver now with \
`message_result`. Summarize:\n\
- What you accomplished.\n\
- What remains unfinished, if anything.\n\
- Any partial findings the caller should know about.\n\
\n\
Then stop. Do not call any more work tools. If a user answer is genuinely \
required before you can continue, call `message_ask` with one concrete \
question instead.";

/// Soft pre-cap warning. See module docs.
pub struct GracefulTurnLimit {
    /// Hard cap from `LoopConfig::max_iterations`. The fire-time soft
    /// limit is `max_iterations - effective_grace()` (saturating), where
    /// `effective_grace()` is either a host-supplied dynamic value
    /// (`grace_provider`) or the static `default_grace` set at
    /// construction.
    max_iterations: usize,

    /// Static fallback grace used when no `grace_provider` is set, and
    /// also as the floor when validating the constructor inputs (a
    /// zero or out-of-range static grace yields `None` so the plugin
    /// is never installed).
    default_grace: usize,

    /// Optional host-supplied dynamic grace. When `Some`, it is called
    /// on every steering poll and the returned value is used as the
    /// effective grace window. Clamped to `[1, max_iterations - 1]` so
    /// the soft trigger is always at least one turn before the hard cap.
    grace_provider: Option<GraceProvider>,

    /// Number of completed assistant turns observed. Used to decide
    /// when the soft limit has been reached. Atomic because plugins are
    /// accessed from `&self` across await points.
    turns_completed: AtomicUsize,

    /// One-shot guard so the wrap-up message is emitted at most once
    /// even if `next_steering_messages` is polled multiple times after
    /// the threshold.
    fired: Arc<AtomicBool>,

    /// Host-specific wrap-up wording. Clark sessions use this to make
    /// the warning plan-aware without teaching the agent core about
    /// Clark's `PlanState` type.
    message_provider: MessageProvider,
}

impl GracefulTurnLimit {
    /// Build a plugin from a hard cap and a grace window. Returns
    /// `None` when no useful soft threshold can be derived — caller
    /// should skip installation in that case.
    ///
    /// Soft limit is `max - grace`, clamped so a soft trigger remains
    /// observable on at least one turn before the cap. Combinations
    /// where the soft and hard limits would coincide (`grace == 0` or
    /// `grace >= max`) yield `None`: at that point the warning would
    /// either fire after the cap or fire at the same moment, neither
    /// of which gives the model a chance to recover.
    pub fn from_hard_cap(max_iterations: usize, grace_iterations: usize) -> Option<Self> {
        Self::from_hard_cap_with_message_provider(
            max_iterations,
            grace_iterations,
            Arc::new(|| DEFAULT_WRAP_UP_MESSAGE.to_string()),
        )
    }

    /// Build with host-provided wrap-up wording. The provider is called
    /// only when the one-shot warning fires.
    pub fn from_hard_cap_with_message_provider(
        max_iterations: usize,
        grace_iterations: usize,
        message_provider: MessageProvider,
    ) -> Option<Self> {
        Self::from_hard_cap_with_providers(
            max_iterations,
            grace_iterations,
            message_provider,
            None,
        )
    }

    /// Build with host-provided wrap-up wording AND a dynamic grace
    /// provider. The grace provider is consulted on every steering
    /// poll; its return value is clamped to `[1, max_iterations - 1]`
    /// before being used as the effective wrap-up window. When the
    /// provider returns the default value (or is unset), behavior
    /// matches [`Self::from_hard_cap`].
    pub fn from_hard_cap_with_providers(
        max_iterations: usize,
        default_grace: usize,
        message_provider: MessageProvider,
        grace_provider: Option<GraceProvider>,
    ) -> Option<Self> {
        if default_grace == 0 || default_grace >= max_iterations {
            return None;
        }
        Some(Self {
            max_iterations,
            default_grace,
            grace_provider,
            turns_completed: AtomicUsize::new(0),
            fired: Arc::new(AtomicBool::new(false)),
            message_provider,
        })
    }

    pub fn default_wrap_up_message() -> &'static str {
        DEFAULT_WRAP_UP_MESSAGE
    }

    /// Shared one-shot flag the loop reads to distinguish a clean
    /// natural close from one prompted by the wrap-up steer. Set to
    /// `true` exactly when (and if) the plugin emits its message.
    pub fn signal(&self) -> Arc<AtomicBool> {
        self.fired.clone()
    }

    /// Effective grace window for this poll. Reads the dynamic
    /// provider when set, else the static `default_grace`. Always
    /// clamped to `[1, max_iterations - 1]` so the soft trigger
    /// fires at least one turn before the hard cap.
    fn effective_grace(&self) -> usize {
        let raw = self
            .grace_provider
            .as_ref()
            .map(|p| p())
            .unwrap_or(self.default_grace);
        raw.clamp(1, self.max_iterations.saturating_sub(1).max(1))
    }

    /// Inspection helper for tests and diagnostics. Returns the
    /// soft threshold (turns count at which the wrap-up fires) computed
    /// against the current dynamic grace, if any.
    pub fn soft_limit(&self) -> usize {
        self.max_iterations.saturating_sub(self.effective_grace())
    }
}

impl Plugin for GracefulTurnLimit {
    fn name(&self) -> &'static str {
        "graceful_turn_limit"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            event_observer: true,
            steering: true,
            ..PluginCapabilities::default()
        }
    }
}

#[async_trait]
impl EventObserver for GracefulTurnLimit {
    async fn on_event(&self, event: &AgentEvent) {
        if matches!(event, AgentEvent::TurnEnd { .. }) {
            self.turns_completed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[async_trait]
impl SteeringSource for GracefulTurnLimit {
    async fn next_steering_messages(&self) -> Vec<AgentMessage> {
        // Read counter first so a completed turn that happens between
        // this check and the swap can't sneak past — at worst we fire
        // one turn late, never twice. `soft_limit()` recomputes from
        // the dynamic grace provider on each poll, so a host can grow
        // the wrap-up window as the work in flight grows (e.g. more
        // open plan phases mean more iterations needed to deliver a
        // partial answer cleanly).
        if self.turns_completed.load(Ordering::Relaxed) < self.soft_limit() {
            return Vec::new();
        }
        // One-shot: swap is the cheapest way to guarantee a single
        // emission even if the loop polls steering more than once.
        if self.fired.swap(true, Ordering::Relaxed) {
            return Vec::new();
        }
        let content = (self.message_provider)();
        vec![AgentMessage::System {
            content,
            timestamp: None,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_hard_cap_rejects_zero_grace() {
        assert!(GracefulTurnLimit::from_hard_cap(50, 0).is_none());
    }

    #[test]
    fn from_hard_cap_rejects_grace_at_or_above_cap() {
        assert!(GracefulTurnLimit::from_hard_cap(10, 10).is_none());
        assert!(GracefulTurnLimit::from_hard_cap(10, 99).is_none());
    }

    #[test]
    fn from_hard_cap_computes_soft_limit() {
        let plugin = GracefulTurnLimit::from_hard_cap(50, 5).unwrap();
        assert_eq!(plugin.soft_limit(), 45);
    }

    #[tokio::test]
    async fn does_not_fire_before_soft_limit() {
        let plugin = GracefulTurnLimit::from_hard_cap(10, 3).unwrap();
        // soft_limit = 7. Complete 6 turns and assert no message.
        for _ in 0..6 {
            plugin.on_event(&turn_end()).await;
        }
        let msgs = plugin.next_steering_messages().await;
        assert!(msgs.is_empty());
        assert!(!plugin.fired.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn fires_once_at_soft_limit() {
        let plugin = GracefulTurnLimit::from_hard_cap(10, 3).unwrap();
        // soft_limit = 7. Complete 7 turns; expect one message.
        for _ in 0..7 {
            plugin.on_event(&turn_end()).await;
        }
        let first = plugin.next_steering_messages().await;
        assert_eq!(first.len(), 1, "should emit one wrap-up message");
        match &first[0] {
            AgentMessage::System { content, .. } => {
                assert!(content.starts_with("You have used your turn budget."))
            }
            other => panic!("expected system wrap-up message, got {other:?}"),
        }
        assert!(plugin.fired.load(Ordering::Relaxed));

        // Second poll: empty. One-shot.
        let second = plugin.next_steering_messages().await;
        assert!(second.is_empty(), "wrap-up must be one-shot");
    }

    #[tokio::test]
    async fn ignores_non_turn_start_events() {
        let plugin = GracefulTurnLimit::from_hard_cap(10, 3).unwrap();
        // Pump a flood of unrelated events. Counter stays at 0.
        for _ in 0..20 {
            plugin.on_event(&AgentEvent::AgentStart).await;
        }
        let msgs = plugin.next_steering_messages().await;
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn wrap_up_points_to_terminal_message_tool() {
        let plugin = GracefulTurnLimit::from_hard_cap(10, 3).unwrap();
        for _ in 0..7 {
            plugin.on_event(&turn_end()).await;
        }
        let msgs = plugin.next_steering_messages().await;
        let AgentMessage::System { content: text, .. } = &msgs[0] else {
            panic!("expected system wrap-up message");
        };

        assert!(text.contains("`message_result`"), "{text}");
        assert!(
            !text.contains("write your final answer in this message"),
            "{text}"
        );
        assert!(!text.contains("Do not call any more tools."), "{text}");
    }

    #[tokio::test]
    async fn wrap_up_uses_custom_message_provider_when_supplied() {
        let plugin = GracefulTurnLimit::from_hard_cap_with_message_provider(
            10,
            3,
            Arc::new(|| "custom wrap-up".to_string()),
        )
        .unwrap();
        for _ in 0..7 {
            plugin.on_event(&turn_end()).await;
        }

        let msgs = plugin.next_steering_messages().await;
        let AgentMessage::System { content, .. } = &msgs[0] else {
            panic!("expected system wrap-up message");
        };
        assert_eq!(content, "custom wrap-up");
    }

    #[tokio::test]
    async fn does_not_fire_before_first_completed_turn() {
        let plugin = GracefulTurnLimit::from_hard_cap(6, 5).unwrap();

        plugin.on_event(&AgentEvent::TurnStart).await;
        let msgs = plugin.next_steering_messages().await;

        assert!(msgs.is_empty());
        assert!(!plugin.fired.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn dynamic_grace_provider_widens_wrap_up_window_for_bigger_jobs() {
        // The host-supplied callback returns the grace window the
        // plugin should use at fire-check time. This lets Clark
        // scale wrap-up budget with plan size (see
        // `clark-agent-bridge/src/session.rs::build_loop_config`).
        let grace = Arc::new(std::sync::Mutex::new(3usize));
        let grace_for_provider = grace.clone();
        let plugin = GracefulTurnLimit::from_hard_cap_with_providers(
            20,
            3,
            Arc::new(|| "wrap".to_string()),
            Some(Arc::new(move || *grace_for_provider.lock().unwrap())),
        )
        .unwrap();

        // grace=3 → soft_limit=17. 16 turns is below the threshold.
        for _ in 0..16 {
            plugin.on_event(&turn_end()).await;
        }
        let early = plugin.next_steering_messages().await;
        assert!(early.is_empty(), "should not fire at 16 turns with grace=3");
        assert_eq!(plugin.soft_limit(), 17);

        // Host widens grace BEFORE the plugin would have fired. Now
        // grace=8 → soft_limit=12. Already past 16 completed turns
        // (>12), so the next poll fires.
        *grace.lock().unwrap() = 8;
        assert_eq!(plugin.soft_limit(), 12);
        let fired = plugin.next_steering_messages().await;
        assert_eq!(
            fired.len(),
            1,
            "widened grace must let the plugin fire on the next poll"
        );
        assert!(plugin.fired.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn dynamic_grace_provider_clamps_out_of_range_returns() {
        // A buggy host that returns 0 or `>= max_iterations` must not
        // be allowed to disable the soft trigger; the plugin clamps
        // into `[1, max-1]` so the wrap-up always lands at least one
        // turn before the hard cap.
        let plugin = GracefulTurnLimit::from_hard_cap_with_providers(
            10,
            3,
            Arc::new(|| "wrap".to_string()),
            Some(Arc::new(|| 0)),
        )
        .unwrap();
        // grace clamps to 1 → soft_limit = 9.
        assert_eq!(plugin.soft_limit(), 9);

        let plugin = GracefulTurnLimit::from_hard_cap_with_providers(
            10,
            3,
            Arc::new(|| "wrap".to_string()),
            Some(Arc::new(|| 999)),
        )
        .unwrap();
        // grace clamps to max-1 = 9 → soft_limit = 1.
        assert_eq!(plugin.soft_limit(), 1);
    }

    fn turn_end() -> AgentEvent {
        AgentEvent::TurnEnd {
            message: AgentMessage::Assistant {
                content: crate::types::AssistantContent::text(""),
                stop_reason: crate::types::StopReason::ToolUse,
                error_message: None,
                timestamp: None,
                usage: None,
            },
            tool_results: Vec::new(),
        }
    }
}
