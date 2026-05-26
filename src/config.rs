//! Loop configuration + builder.
//!
//! `LoopConfig` is the assembled, immutable configuration the loop
//! reads. `AgentBuilder` is the ergonomic constructor: chain method
//! calls to add stream transport, tools, plugins, then `.build()` to
//! freeze.
//!
//! Plugins are stored as `Arc<dyn Plugin>` and queried by capability via
//! the dispatcher (see `crate::run::PluginDispatch`). This avoids
//! repeated trait-object downcast attempts at every hook point.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::event::{EventSink, NoopSink};
use crate::plugin::{
    AfterToolCall, BeforeToolCall, ContextTransform, EventObserver, FollowUpSource, Plugin,
    SteeringSource, ToolGate,
};
use crate::plugins::graceful_turn_limit::GracefulTurnLimit;
use crate::stream::{ReasoningEffort, StreamFn};
use crate::tokens::{CharHeuristicEstimator, TokenEstimator};
use crate::tool::{ExecutionMode, ToolRegistry};

/// Default number of grace iterations the soft-limit warning leaves before
/// the hard `max_iterations` cap. Picked to match pi-subagents' default and
/// to give a wrap-up turn plus a couple of recovery turns when the model
/// needs them.
pub const DEFAULT_GRACE_ITERATIONS: usize = 5;

/// Assembled loop configuration. Construct via [`AgentBuilder`].
///
/// The system prompt is run state, not builder configuration: callers
/// provide it through [`crate::types::AgentContext`].
pub struct LoopConfig {
    pub stream: Arc<dyn StreamFn>,
    pub tools: Arc<ToolRegistry>,
    pub event_sink: Arc<dyn EventSink>,

    /// Optional conversation identifier, surfaced to plugins via
    /// `ToolGateContext::conversation_id`. The agent core itself does
    /// not use this — it's metadata for diagnostics and
    /// conversation-scoped policy. `None` when the loop is invoked
    /// outside a conversation context (tests, isolated subagent runs).
    pub conversation_id: Option<String>,

    /// Optional model identifier surfaced to plugins via
    /// [`crate::plugin::TransformContext::model_id`]. The loop does not
    /// use this directly — the active `StreamFn` already knows its
    /// model. Plugins that key per-model behavior (cache-aware
    /// compaction, model-specific token estimators, model-specific
    /// system reminders) read it from here. `None` when the host
    /// runtime doesn't surface one.
    pub model_id: Option<String>,

    /// Token estimator the loop hands to context transforms. Defaults
    /// to [`CharHeuristicEstimator`]; apps with a real tokenizer
    /// implement [`TokenEstimator`] and supply their own via
    /// [`AgentBuilder::token_estimator`].
    pub token_estimator: Arc<dyn TokenEstimator>,

    /// Default tool execution mode. A batch downgrades to `Sequential`
    /// if any tool in it sets `requires_exclusive_sandbox = true`.
    /// Set this to `Sequential` to pin the entire loop to sequential
    /// dispatch regardless of per-tool flags (deterministic eval,
    /// debugging, ordered replay).
    pub default_execution_mode: ExecutionMode,

    /// Optional hard cap on limit-counted tool calls executed from a
    /// single assistant turn. When set to `1`, the loop preserves every
    /// emitted tool call in the assistant message, executes the first
    /// limit-counted call plus any zero-weight progress signals, appends
    /// synthetic error results for the rest, then asks the model to choose
    /// the next action.
    pub max_tool_calls_per_turn: Option<usize>,

    /// Optional sampling controls forwarded to the stream transport.
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,

    /// Reasoning-effort knob forwarded to the stream transport on every
    /// turn. Single source of truth: the bridge no longer hardcodes a
    /// per-request default and `provider_extras` no longer carries
    /// `reasoning_effort`. Default is [`ReasoningEffort::Minimal`].
    pub reasoning: ReasoningEffort,

    /// Recovery strategy for `StopReason::MaxTokens` truncations. When
    /// `Some`, the loop discards a truncated assistant turn and
    /// re-streams with a larger cap up to `max_attempts` times before
    /// accepting the truncated turn. Default `None` — today's
    /// behavior. Opt-in because the cost can be large (worst case
    /// 8× output tokens with `Double` × 3 attempts).
    pub max_output_tokens_recovery: Option<MaxTokensRecovery>,

    /// Hard ceiling on iterations within a single `run`. Prevents
    /// runaway loops if neither the model nor tools ever vote to
    /// terminate. `None` = unbounded.
    pub max_iterations: Option<usize>,

    /// Number of no-tool assistant stops the loop may recover from
    /// before treating another no-tool stop as a typed failure. `None`
    /// preserves the generic core's historical natural-stop behavior.
    pub empty_outcome_retry_budget: Option<usize>,

    /// Optional terminal-tool compatibility shim for providers that cannot
    /// honor forced tool choice. When set, a non-empty plain assistant text
    /// stop may be converted into this terminal tool result, but only on a
    /// turn whose advertised tool allowlist has already been narrowed to
    /// terminal delivery tools. Default `None` preserves the strict
    /// "terminal text must arrive through a tool call" contract.
    pub plain_text_terminal_fallback_tool: Option<String>,

    /// When true, [`Self::plain_text_terminal_fallback_tool`] fires on the
    /// FIRST plain-text stop instead of waiting for the turn allowlist to
    /// narrow to terminators. Intended for providers in the
    /// auto-when-forced class (e.g. Qwen 3.5 Flash) where wire-level
    /// `tool_choice: "required"` is rejected and so plain text is the
    /// model's default failure mode — there's no benefit to running the
    /// `TerminalMessageGuard` nudge cycle first because the model will
    /// emit prose every time. Default `false` preserves the post-narrowing
    /// gate for everyone else.
    pub plain_text_terminal_fallback_eager: bool,

    /// When true, the eager plain-text fallback path nudges the model with
    /// an explicit protocol-recovery system message BEFORE synthesizing a
    /// terminal tool result, giving the model a bounded number of retries
    /// to follow the protocol. Only synthesizes as a last-resort after the
    /// nudges are exhausted. Default `false` preserves the original
    /// silent-synthesize behavior. Has no effect unless both
    /// [`Self::plain_text_terminal_fallback_tool`] and
    /// [`Self::plain_text_terminal_fallback_eager`] are set.
    pub plain_text_terminal_fallback_eager_nudge: bool,

    /// Number of iterations before `max_iterations` at which the
    /// graceful-turn-limit plugin injects a one-shot wrap-up steering
    /// message. `0` disables the soft warning (behavior identical to
    /// pre-grace versions). Has no effect when `max_iterations` is `None`.
    pub grace_iterations: usize,

    /// One-shot flag flipped by the graceful-turn-limit plugin when it
    /// emits its wrap-up steering message. The loop reads this at end of
    /// run to choose between `LoopOutcome::WrappedUp` and
    /// `LoopOutcome::Done`. `None` when no plugin is installed (no soft
    /// warning configured).
    pub(crate) grace_signal: Option<Arc<AtomicBool>>,

    pub(crate) plugins: PluginRegistry,
}

#[derive(Default)]
pub(crate) struct PluginRegistry {
    pub before_tool_call: Vec<Arc<dyn BeforeToolCall>>,
    pub after_tool_call: Vec<Arc<dyn AfterToolCall>>,
    pub context_transform: Vec<Arc<dyn ContextTransform>>,
    pub event_observer: Vec<Arc<dyn EventObserver>>,
    pub steering: Vec<Arc<dyn SteeringSource>>,
    pub follow_up: Vec<Arc<dyn FollowUpSource>>,
    pub tool_gate: Vec<Arc<dyn ToolGate>>,
}

/// Recovery strategy when the provider returns `StopReason::MaxTokens`.
///
/// On hit, the loop discards the truncated assistant turn (the
/// `MessageStart`/`MessageEnd` events for it still fired — listeners
/// correlate via the new `AgentEvent::OutputTokensEscalation`) and
/// re-streams with a higher cap.
///
/// Bounded by `max_attempts` per turn. Hits the `ceiling` if set.
/// `Fixed` ladders run out by definition once `attempts >=
/// caps.len()`.
#[derive(Debug, Clone)]
pub struct MaxTokensRecovery {
    /// Hard upper bound on retries within a single turn. The loop
    /// emits at most `max_attempts` escalation events per turn; the
    /// `attempts + 1`th call simply uses the ladder's last cap and
    /// the result is accepted regardless.
    pub max_attempts: u8,
    /// How to derive the next cap from the previous one.
    pub scaling: TokenScaling,
    /// Hard upper bound on the cap itself. `None` means no ceiling
    /// (relies on `max_attempts` to bound the spend). Recovery stops
    /// short when the next computed cap would equal or fall below
    /// the previous one (no progress).
    pub ceiling: Option<u32>,
}

/// How successive recovery attempts grow `max_output_tokens`.
#[derive(Debug, Clone)]
pub enum TokenScaling {
    /// Double per attempt: 4096 → 8192 → 16384 → ... Worst case
    /// `2^max_attempts` × the starting cap. Default for callers
    /// that prefer a small ladder with big steps.
    Double,
    /// Add a fixed step per attempt: 4096 → 4096+step → 4096+2·step.
    /// Predictable cost ladder; better when the model usually only
    /// needs a little more room.
    Linear { step: u32 },
    /// Explicit progression: `caps[0]` for the first retry, `caps[1]`
    /// for the second, etc. Lets callers express "try 8k then 16k
    /// then give up" without computing scales.
    Fixed(Vec<u32>),
}

impl MaxTokensRecovery {
    /// Default: 3 retries with doubling, no ceiling. Meant as the
    /// "least-config option" for callers who just want the recovery
    /// without tuning. Real deployments usually pin a `ceiling`
    /// matching their model's hard max.
    pub fn doubling() -> Self {
        Self {
            max_attempts: 3,
            scaling: TokenScaling::Double,
            ceiling: None,
        }
    }

    /// Compute the cap for retry attempt `attempt_zero_indexed`
    /// (0 = the first retry, after the original turn). Returns
    /// `None` when the ladder cannot make further progress (Fixed
    /// exhausted, ceiling reached at the previous step).
    pub fn next_cap(&self, prev_cap: u32, attempt_zero_indexed: u8) -> Option<u32> {
        let raw = match &self.scaling {
            TokenScaling::Double => prev_cap.saturating_mul(2),
            TokenScaling::Linear { step } => prev_cap.saturating_add(*step),
            TokenScaling::Fixed(caps) => {
                let idx = attempt_zero_indexed as usize;
                *caps.get(idx)?
            }
        };
        let bounded = match self.ceiling {
            Some(c) => raw.min(c),
            None => raw,
        };
        if bounded > prev_cap {
            Some(bounded)
        } else {
            None
        }
    }
}

/// Fluent builder for [`LoopConfig`].
///
/// ```ignore
/// let config = AgentBuilder::new()
///     .stream(provider)
///     .tools(registry)
///     .event_sink(channel_sink)
///     .before_tool_call(retired_path_gate)
///     .after_tool_call(repeat_detector)
///     .context_transform(token_budget_pruner)
///     .steering(steering_source)
///     .max_iterations(50)
///     .build();
/// ```
pub struct AgentBuilder {
    stream: Option<Arc<dyn StreamFn>>,
    tools: Arc<ToolRegistry>,
    event_sink: Arc<dyn EventSink>,
    default_execution_mode: ExecutionMode,
    max_tool_calls_per_turn: Option<usize>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    reasoning: ReasoningEffort,
    max_output_tokens_recovery: Option<MaxTokensRecovery>,
    max_iterations: Option<usize>,
    empty_outcome_retry_budget: Option<usize>,
    plain_text_terminal_fallback_tool: Option<String>,
    plain_text_terminal_fallback_eager: bool,
    plain_text_terminal_fallback_eager_nudge: bool,
    grace_iterations: usize,
    graceful_turn_limit_message_provider: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    graceful_turn_limit_grace_provider: Option<Arc<dyn Fn() -> usize + Send + Sync>>,
    conversation_id: Option<String>,
    model_id: Option<String>,
    token_estimator: Arc<dyn TokenEstimator>,
    plugins: PluginRegistry,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            stream: None,
            tools: Arc::new(ToolRegistry::new()),
            event_sink: Arc::new(NoopSink),
            default_execution_mode: ExecutionMode::Parallel,
            max_tool_calls_per_turn: None,
            temperature: None,
            max_output_tokens: None,
            reasoning: ReasoningEffort::default(),
            max_output_tokens_recovery: None,
            max_iterations: None,
            empty_outcome_retry_budget: None,
            plain_text_terminal_fallback_tool: None,
            plain_text_terminal_fallback_eager: false,
            plain_text_terminal_fallback_eager_nudge: false,
            grace_iterations: DEFAULT_GRACE_ITERATIONS,
            graceful_turn_limit_message_provider: None,
            graceful_turn_limit_grace_provider: None,
            conversation_id: None,
            model_id: None,
            token_estimator: Arc::new(CharHeuristicEstimator),
            plugins: PluginRegistry::default(),
        }
    }

    pub fn stream(mut self, stream: Arc<dyn StreamFn>) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Arc::new(tools);
        self
    }

    /// Variant for callers that already share a registry by `Arc`.
    pub fn tools_arc(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }

    pub fn event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.event_sink = sink;
        self
    }

    pub fn default_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.default_execution_mode = mode;
        self
    }

    pub fn max_tool_calls_per_turn(mut self, max: usize) -> Self {
        self.max_tool_calls_per_turn = Some(max.max(1));
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn max_output_tokens(mut self, t: u32) -> Self {
        self.max_output_tokens = Some(t);
        self
    }

    /// Set the reasoning-effort knob forwarded to the stream transport
    /// on every turn. Replaces the legacy stringly-typed
    /// `provider_config["reasoning_effort"]` knob; per-job overrides
    /// flow through this typed surface.
    pub fn reasoning(mut self, level: ReasoningEffort) -> Self {
        self.reasoning = level;
        self
    }

    /// Enable max-output-tokens recovery. When the provider returns
    /// `StopReason::MaxTokens`, the loop discards the truncated turn
    /// and re-streams with a larger cap up to `recovery.max_attempts`
    /// times. Off by default — opt in by passing a configured
    /// `MaxTokensRecovery`. See the type for cost discussion.
    pub fn max_output_tokens_recovery(mut self, recovery: MaxTokensRecovery) -> Self {
        self.max_output_tokens_recovery = Some(recovery);
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Enable the no-tool outcome watchdog. `n` is the number of
    /// no-tool assistant stops that recovery plugins may handle; the
    /// next no-tool stop ends the run with a typed
    /// [`crate::error::LoopError`].
    pub fn empty_outcome_retry_budget(mut self, n: usize) -> Self {
        self.empty_outcome_retry_budget = Some(n);
        self
    }

    /// Convert plain assistant text into a terminal tool result on
    /// terminal-only compatibility turns. Intended for providers that reject
    /// `tool_choice: "required"` and therefore can leak final prose even
    /// while Clark advertises only delivery tools.
    pub fn plain_text_terminal_fallback_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.plain_text_terminal_fallback_tool = Some(tool_name.into());
        self
    }

    /// Make [`Self::plain_text_terminal_fallback_tool`] fire on the FIRST
    /// plain-text stop instead of waiting for the turn allowlist to be
    /// narrowed to terminators by `TerminalMessageGuard`. Use this for
    /// providers in the auto-when-forced class (Qwen 3.5 Flash today)
    /// where wire-level forcing isn't available, so prose is the
    /// model's default failure mode and the nudge cycle just burns
    /// turns. Has no effect unless
    /// [`Self::plain_text_terminal_fallback_tool`] is also set.
    pub fn plain_text_terminal_fallback_eager(mut self, eager: bool) -> Self {
        self.plain_text_terminal_fallback_eager = eager;
        self
    }

    /// Make the eager plain-text fallback path nudge the model with an
    /// explicit protocol-recovery system message before synthesizing a
    /// terminal tool result, giving up to a bounded number of retries to
    /// follow the protocol. Off by default — opt in for evals and runs
    /// where silently laundering prose into delivery is a worse outcome
    /// than a small number of extra streaming turns. Has no effect unless
    /// both [`Self::plain_text_terminal_fallback_tool`] and
    /// [`Self::plain_text_terminal_fallback_eager`] are set.
    pub fn plain_text_terminal_fallback_eager_nudge(mut self, on: bool) -> Self {
        self.plain_text_terminal_fallback_eager_nudge = on;
        self
    }

    /// Override the grace window used by the auto-installed
    /// graceful-turn-limit plugin. Pass `0` to disable the soft warning
    /// entirely (the loop will hit `max_iterations` with no advance
    /// notice). Default is [`DEFAULT_GRACE_ITERATIONS`].
    pub fn grace_iterations(mut self, n: usize) -> Self {
        self.grace_iterations = n;
        self
    }

    /// Override the one-shot wrap-up message emitted by the
    /// auto-installed graceful-turn-limit plugin. Hosts can use this to
    /// make the warning aware of product state while keeping the core
    /// loop independent of product-specific types.
    pub fn graceful_turn_limit_message_provider<F>(mut self, provider: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.graceful_turn_limit_message_provider = Some(Arc::new(provider));
        self
    }

    /// Supply a dynamic grace-iterations provider for the
    /// auto-installed graceful-turn-limit plugin. The callback is
    /// invoked on every steering poll, and its return value is
    /// clamped into `[1, max_iterations - 1]`. Use this to scale the
    /// wrap-up window with the size of the work in flight (e.g. more
    /// open plan phases ⇒ a longer wrap-up window so a partial
    /// delivery can still land). When unset, the static
    /// [`grace_iterations`](Self::grace_iterations) value is used.
    pub fn graceful_turn_limit_grace_provider<F>(mut self, provider: F) -> Self
    where
        F: Fn() -> usize + Send + Sync + 'static,
    {
        self.graceful_turn_limit_grace_provider = Some(Arc::new(provider));
        self
    }

    /// Attach a conversation identifier so plugins can include
    /// conversation-scoped diagnostics or policy. The agent core itself
    /// does not consume this — it's just metadata threaded through
    /// `ToolGateContext`. Optional; absent for tests and isolated
    /// subagent runs.
    pub fn conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Attach a model identifier so context transforms can read it via
    /// [`crate::plugin::TransformContext::model_id`]. The loop itself
    /// does not consume this; the active `StreamFn` already knows its
    /// model. Optional — defaults to `None` (transforms see the empty
    /// string).
    pub fn model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Plug in a token estimator for budgeting and compaction. Defaults
    /// to the char-heuristic estimator when not set. Pass an `Arc` if
    /// the estimator is shared across multiple builders.
    pub fn token_estimator<E: TokenEstimator>(mut self, est: E) -> Self {
        self.token_estimator = Arc::new(est);
        self
    }

    /// Variant for callers that already share an estimator by `Arc`.
    pub fn token_estimator_arc(mut self, est: Arc<dyn TokenEstimator>) -> Self {
        self.token_estimator = est;
        self
    }

    // ─── Plugin registration (one method per capability) ────────────

    pub fn before_tool_call<P: BeforeToolCall + 'static>(mut self, plugin: P) -> Self {
        self.plugins.before_tool_call.push(Arc::new(plugin));
        self
    }

    pub fn after_tool_call<P: AfterToolCall + 'static>(mut self, plugin: P) -> Self {
        self.plugins.after_tool_call.push(Arc::new(plugin));
        self
    }

    pub fn context_transform<P: ContextTransform + 'static>(mut self, plugin: P) -> Self {
        self.plugins.context_transform.push(Arc::new(plugin));
        self
    }

    pub fn event_observer<P: EventObserver + 'static>(mut self, plugin: P) -> Self {
        self.plugins.event_observer.push(Arc::new(plugin));
        self
    }

    pub fn steering<P: SteeringSource + 'static>(mut self, plugin: P) -> Self {
        self.plugins.steering.push(Arc::new(plugin));
        self
    }

    pub fn follow_up<P: FollowUpSource + 'static>(mut self, plugin: P) -> Self {
        self.plugins.follow_up.push(Arc::new(plugin));
        self
    }

    /// Variant that takes pre-`Arc`'d trait objects, useful when the
    /// caller already has shared plugin instances.
    pub fn before_tool_call_arc(mut self, plugin: Arc<dyn BeforeToolCall>) -> Self {
        self.plugins.before_tool_call.push(plugin);
        self
    }
    pub fn after_tool_call_arc(mut self, plugin: Arc<dyn AfterToolCall>) -> Self {
        self.plugins.after_tool_call.push(plugin);
        self
    }
    pub fn context_transform_arc(mut self, plugin: Arc<dyn ContextTransform>) -> Self {
        self.plugins.context_transform.push(plugin);
        self
    }
    pub fn event_observer_arc(mut self, plugin: Arc<dyn EventObserver>) -> Self {
        self.plugins.event_observer.push(plugin);
        self
    }
    pub fn follow_up_arc(mut self, plugin: Arc<dyn FollowUpSource>) -> Self {
        self.plugins.follow_up.push(plugin);
        self
    }
    pub fn steering_arc(mut self, plugin: Arc<dyn SteeringSource>) -> Self {
        self.plugins.steering.push(plugin);
        self
    }
    pub fn tool_gate_arc(mut self, plugin: Arc<dyn ToolGate>) -> Self {
        self.plugins.tool_gate.push(plugin);
        self
    }

    /// Generic plugin registration. Inspects [`Plugin::capabilities`] to
    /// decide which dispatch lists to add the plugin to. Same `Arc` is
    /// shared across all enabled capabilities so a single plugin
    /// instance can implement multiple traits.
    pub fn plugin<P>(mut self, plugin: Arc<P>) -> Self
    where
        P: Plugin
            + BeforeToolCall
            + AfterToolCall
            + ContextTransform
            + EventObserver
            + SteeringSource
            + FollowUpSource
            + ToolGate
            + 'static,
    {
        let caps = plugin.capabilities();
        if caps.before_tool_call {
            self.plugins
                .before_tool_call
                .push(plugin.clone() as Arc<dyn BeforeToolCall>);
        }
        if caps.after_tool_call {
            self.plugins
                .after_tool_call
                .push(plugin.clone() as Arc<dyn AfterToolCall>);
        }
        if caps.context_transform {
            self.plugins
                .context_transform
                .push(plugin.clone() as Arc<dyn ContextTransform>);
        }
        if caps.event_observer {
            self.plugins
                .event_observer
                .push(plugin.clone() as Arc<dyn EventObserver>);
        }
        if caps.steering {
            self.plugins
                .steering
                .push(plugin.clone() as Arc<dyn SteeringSource>);
        }
        if caps.follow_up {
            self.plugins
                .follow_up
                .push(plugin.clone() as Arc<dyn FollowUpSource>);
        }
        if caps.tool_gate {
            self.plugins.tool_gate.push(plugin as Arc<dyn ToolGate>);
        }
        self
    }

    pub fn build(mut self) -> Result<LoopConfig, BuilderError> {
        let stream = self.stream.ok_or(BuilderError::MissingStream)?;

        // Auto-install the graceful-turn-limit plugin when both a hard
        // cap and a grace window are configured. Mirrors how
        // `ThinkingTagStreamFilter` is auto-wired by the bridge: callers
        // shouldn't have to remember to register the standard
        // safety-net plugins.
        let grace_signal = match (self.max_iterations, self.grace_iterations) {
            (Some(max), grace) if grace > 0 => {
                let grace_provider = self.graceful_turn_limit_grace_provider.take();
                let message_provider = self
                    .graceful_turn_limit_message_provider
                    .take()
                    .unwrap_or_else(|| {
                        Arc::new(|| GracefulTurnLimit::default_wrap_up_message().to_string())
                    });
                let plugin = GracefulTurnLimit::from_hard_cap_with_providers(
                    max,
                    grace,
                    message_provider,
                    grace_provider,
                );
                if let Some(plugin) = plugin {
                    let signal = plugin.signal();
                    let arc: Arc<GracefulTurnLimit> = Arc::new(plugin);
                    self.plugins
                        .event_observer
                        .push(arc.clone() as Arc<dyn EventObserver>);
                    self.plugins.steering.push(arc as Arc<dyn SteeringSource>);
                    Some(signal)
                } else {
                    // grace >= max: no useful soft window — skip install.
                    None
                }
            }
            _ => None,
        };

        Ok(LoopConfig {
            stream,
            tools: self.tools,
            event_sink: self.event_sink,
            default_execution_mode: self.default_execution_mode,
            max_tool_calls_per_turn: self.max_tool_calls_per_turn,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            reasoning: self.reasoning,
            max_output_tokens_recovery: self.max_output_tokens_recovery,
            max_iterations: self.max_iterations,
            empty_outcome_retry_budget: self.empty_outcome_retry_budget,
            plain_text_terminal_fallback_tool: self.plain_text_terminal_fallback_tool,
            plain_text_terminal_fallback_eager: self.plain_text_terminal_fallback_eager,
            plain_text_terminal_fallback_eager_nudge: self
                .plain_text_terminal_fallback_eager_nudge,
            grace_iterations: self.grace_iterations,
            conversation_id: self.conversation_id,
            model_id: self.model_id,
            token_estimator: self.token_estimator,
            grace_signal,
            plugins: self.plugins,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("missing stream transport: call AgentBuilder::stream() before build()")]
    MissingStream,
}

/// Snapshot of registered plugin names per category, in registration order.
///
/// Returned by [`LoopConfig::plugin_names`] for inspection / regression
/// tests. Order matches the order the loop will invoke each plugin
/// (left-to-right composition for `ContextTransform`, etc.). Pure read
/// — does not clone the plugins themselves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginNames {
    pub before_tool_call: Vec<&'static str>,
    pub after_tool_call: Vec<&'static str>,
    pub context_transform: Vec<&'static str>,
    pub event_observer: Vec<&'static str>,
    pub steering: Vec<&'static str>,
    pub follow_up: Vec<&'static str>,
    pub tool_gate: Vec<&'static str>,
}

impl LoopConfig {
    /// Build an [`AgentBuilder`] pre-populated for a child run spawned
    /// from this config.
    ///
    /// Inherits, by value or `Arc`:
    /// - stream transport, tool registry, token estimator
    /// - sampling controls (`temperature`, `max_output_tokens`,
    ///   `reasoning`)
    /// - max-output-tokens recovery ladder
    /// - default execution mode, `max_tool_calls_per_turn`
    /// - model id, grace iterations
    /// - plain-text-terminal fallback knobs
    /// - every plugin whose
    ///   [`crate::plugin::PluginCapabilities::inheritable_to_child`]
    ///   bit is set
    ///
    /// Does **not** inherit:
    /// - `event_sink` — callers install a child-scoped sink (typically
    ///   `ChildRunSink` in the bridge) before `build`.
    /// - `max_iterations` — children get their own budget; defaults
    ///   to unbounded until the caller sets one.
    /// - `empty_outcome_retry_budget` — child runs make independent
    ///   recovery decisions.
    /// - `conversation_id` — the child should carry its own identity
    ///   via [`crate::AgentContext::identity`].
    /// - plugins that did **not** opt in to inheritance — they remain
    ///   parent-only.
    ///
    /// This is the single primitive for "spawn a fresh Clark with the
    /// same execution shape as me." Replaces the bridge's hand-rolled
    /// re-assembly in `InProcessRunner::run`; bridges still register
    /// child-specific guards (delivery gate, terminal guard, profile
    /// guards, etc.) on top of the returned builder.
    pub fn child_builder(&self) -> AgentBuilder {
        let mut builder = AgentBuilder::new()
            .stream(self.stream.clone())
            .tools_arc(self.tools.clone())
            .default_execution_mode(self.default_execution_mode)
            .reasoning(self.reasoning)
            .grace_iterations(self.grace_iterations)
            .token_estimator_arc(self.token_estimator.clone());
        if let Some(t) = self.temperature {
            builder = builder.temperature(t);
        }
        if let Some(m) = self.max_output_tokens {
            builder = builder.max_output_tokens(m);
        }
        if let Some(n) = self.max_tool_calls_per_turn {
            builder = builder.max_tool_calls_per_turn(n);
        }
        if let Some(r) = self.max_output_tokens_recovery.clone() {
            builder = builder.max_output_tokens_recovery(r);
        }
        if let Some(id) = &self.model_id {
            builder = builder.model_id(id.clone());
        }
        if let Some(tool) = &self.plain_text_terminal_fallback_tool {
            builder = builder
                .plain_text_terminal_fallback_tool(tool.clone())
                .plain_text_terminal_fallback_eager(self.plain_text_terminal_fallback_eager)
                .plain_text_terminal_fallback_eager_nudge(
                    self.plain_text_terminal_fallback_eager_nudge,
                );
        }

        for p in &self.plugins.before_tool_call {
            if p.capabilities().inheritable_to_child {
                builder = builder.before_tool_call_arc(p.clone());
            }
        }
        for p in &self.plugins.after_tool_call {
            if p.capabilities().inheritable_to_child {
                builder = builder.after_tool_call_arc(p.clone());
            }
        }
        for p in &self.plugins.context_transform {
            if p.capabilities().inheritable_to_child {
                builder = builder.context_transform_arc(p.clone());
            }
        }
        for p in &self.plugins.event_observer {
            if p.capabilities().inheritable_to_child {
                builder = builder.event_observer_arc(p.clone());
            }
        }
        for p in &self.plugins.steering {
            if p.capabilities().inheritable_to_child {
                builder = builder.steering_arc(p.clone());
            }
        }
        for p in &self.plugins.follow_up {
            if p.capabilities().inheritable_to_child {
                builder = builder.follow_up_arc(p.clone());
            }
        }
        for p in &self.plugins.tool_gate {
            if p.capabilities().inheritable_to_child {
                builder = builder.tool_gate_arc(p.clone());
            }
        }

        builder
    }

    /// Plugin names per category, in registration order. The composition
    /// order is part of the loop's external contract — bridges and host
    /// runtimes assemble plugins in a specific order so transforms run
    /// before token-budget pruning, gates fire before terminator
    /// validation, etc. Tests use this to pin the assembled order so
    /// silent reorderings during refactors surface as a diff instead of
    /// a runtime regression.
    pub fn plugin_names(&self) -> PluginNames {
        PluginNames {
            before_tool_call: self
                .plugins
                .before_tool_call
                .iter()
                .map(|p| p.name())
                .collect(),
            after_tool_call: self
                .plugins
                .after_tool_call
                .iter()
                .map(|p| p.name())
                .collect(),
            context_transform: self
                .plugins
                .context_transform
                .iter()
                .map(|p| p.name())
                .collect(),
            event_observer: self
                .plugins
                .event_observer
                .iter()
                .map(|p| p.name())
                .collect(),
            steering: self.plugins.steering.iter().map(|p| p.name()).collect(),
            follow_up: self.plugins.follow_up.iter().map(|p| p.name()).collect(),
            tool_gate: self.plugins.tool_gate.iter().map(|p| p.name()).collect(),
        }
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    #[test]
    fn doubling_walks_powers_of_two() {
        let recovery = MaxTokensRecovery::doubling();
        assert_eq!(recovery.next_cap(4096, 0), Some(8192));
        assert_eq!(recovery.next_cap(8192, 1), Some(16384));
        assert_eq!(recovery.next_cap(16384, 2), Some(32768));
    }

    #[test]
    fn ceiling_clamps_growth() {
        let recovery = MaxTokensRecovery {
            max_attempts: 3,
            scaling: TokenScaling::Double,
            ceiling: Some(10_000),
        };
        assert_eq!(recovery.next_cap(4096, 0), Some(8192));
        // Doubling 8192 -> 16384, clamped to 10_000 (still > prev).
        assert_eq!(recovery.next_cap(8192, 1), Some(10_000));
        // Already at ceiling: no progress possible.
        assert_eq!(recovery.next_cap(10_000, 2), None);
    }

    #[test]
    fn linear_step_adds_per_attempt() {
        let recovery = MaxTokensRecovery {
            max_attempts: 3,
            scaling: TokenScaling::Linear { step: 2_000 },
            ceiling: None,
        };
        assert_eq!(recovery.next_cap(4096, 0), Some(6_096));
        assert_eq!(recovery.next_cap(6_096, 1), Some(8_096));
    }

    #[test]
    fn fixed_progression_runs_out() {
        let recovery = MaxTokensRecovery {
            max_attempts: 4,
            scaling: TokenScaling::Fixed(vec![8_000, 16_000]),
            ceiling: None,
        };
        assert_eq!(recovery.next_cap(4_096, 0), Some(8_000));
        assert_eq!(recovery.next_cap(8_000, 1), Some(16_000));
        // Ladder exhausted even though attempts remain.
        assert_eq!(recovery.next_cap(16_000, 2), None);
    }

    #[test]
    fn no_progress_returns_none() {
        // Linear with step=0 cannot make progress.
        let recovery = MaxTokensRecovery {
            max_attempts: 3,
            scaling: TokenScaling::Linear { step: 0 },
            ceiling: None,
        };
        assert_eq!(recovery.next_cap(4_096, 0), None);
    }
}

#[cfg(test)]
mod child_builder_tests {
    use super::*;
    use crate::plugin::{Plugin, PluginCapabilities};
    use crate::stream::{StreamEvent, StreamFn, StreamRequest};
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use futures::StreamExt;

    struct EmptyStream;
    #[async_trait]
    impl StreamFn for EmptyStream {
        async fn stream(
            &self,
            _r: StreamRequest,
            _s: tokio_util::sync::CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            futures::stream::empty().boxed()
        }
    }

    struct ParentOnlyPlugin;
    impl Plugin for ParentOnlyPlugin {
        fn name(&self) -> &'static str {
            "parent_only"
        }
        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::event_observer()
        }
    }
    #[async_trait]
    impl crate::EventObserver for ParentOnlyPlugin {
        async fn on_event(&self, _event: &crate::AgentEvent) {}
    }

    struct InheritablePlugin;
    impl Plugin for InheritablePlugin {
        fn name(&self) -> &'static str {
            "inheritable"
        }
        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::event_observer().with_inheritable_to_child()
        }
    }
    #[async_trait]
    impl crate::EventObserver for InheritablePlugin {
        async fn on_event(&self, _event: &crate::AgentEvent) {}
    }

    #[test]
    fn child_builder_inherits_only_opted_in_plugins() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .event_observer(ParentOnlyPlugin)
            .event_observer(InheritablePlugin)
            .max_iterations(10)
            .build()
            .expect("parent builds");

        let child = parent
            .child_builder()
            .build()
            .expect("child builds");

        let names = child.plugin_names();
        assert_eq!(
            names.event_observer,
            vec!["inheritable"],
            "child must drop parent-only plugins"
        );
    }

    #[test]
    fn child_builder_carries_sampling_and_recovery_knobs() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .temperature(0.3)
            .max_output_tokens(8192)
            .max_tool_calls_per_turn(3)
            .max_output_tokens_recovery(MaxTokensRecovery::doubling())
            .model_id("test-model")
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        assert_eq!(child.temperature, Some(0.3));
        assert_eq!(child.max_output_tokens, Some(8192));
        assert_eq!(child.max_tool_calls_per_turn, Some(3));
        assert!(child.max_output_tokens_recovery.is_some());
        assert_eq!(child.model_id.as_deref(), Some("test-model"));
    }

    #[test]
    fn child_builder_does_not_inherit_max_iterations() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .max_iterations(50)
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");
        assert_eq!(
            child.max_iterations, None,
            "child gets its own iteration budget, not the parent's"
        );
    }
}
