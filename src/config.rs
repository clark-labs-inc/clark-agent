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

use std::sync::Arc;

use serde_json::Value;

use crate::event::{EventSink, NoopSink};
use crate::plugin::{
    AfterToolCall, BeforeToolCall, ContextOverflowRecovery, ContextTransform, EventObserver,
    FollowUpSource, Plugin, SteeringSource, ToolGate,
};
use crate::protocol::{default_policy, ProtocolPolicy};
use crate::stream::{ReasoningEffort, StreamFn};
use crate::tokens::{CharHeuristicEstimator, TokenEstimator};
use crate::tool::{ExecutionMode, ToolRegistry};

/// Assembled loop configuration. Construct via [`AgentBuilder`].
///
/// The system prompt is run state, not builder configuration: callers
/// provide it through [`crate::types::AgentContext`].
pub struct LoopConfig {
    pub stream: Arc<dyn StreamFn>,
    pub tools: Arc<ToolRegistry>,
    pub event_sink: Arc<dyn EventSink>,

    /// Conversation-protocol policy: the seam that supplies any
    /// product-specific tool vocabulary (plain-text recovery prose,
    /// tool-call alias repair, hidden-tool errors, terminal-tool
    /// classification). Defaults to [`crate::DefaultProtocolPolicy`],
    /// whose behavior is generic and names no specific tools. Downstream
    /// product crates install their own via
    /// [`AgentBuilder::protocol_policy`]. See [`crate::protocol`].
    pub protocol: Arc<dyn ProtocolPolicy>,

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
    /// turn. The single source of truth for per-request reasoning effort:
    /// the transport reads it here rather than from per-provider extras.
    /// Default is [`ReasoningEffort::Minimal`].
    pub reasoning: ReasoningEffort,

    /// Provider-specific extras forwarded to the stream transport on
    /// every turn (e.g., `response_format` for structured output
    /// enforcement, custom routing pins). Passed as-is into
    /// [`crate::StreamRequest::provider_extras`]; `None` sends
    /// `Value::Null`.
    pub provider_extras: Option<Value>,

    /// Hard ceiling on model iterations within a single `run`. Prevents a
    /// malformed tool-call/recovery cycle from consuming unbounded provider
    /// requests. `None` leaves lifetime control to the caller.
    pub max_iterations: Option<usize>,

    /// Recovery for a provider context-window rejection mid-run. When
    /// `Some`, a [`crate::StreamError::ContextOverflow`] triggers the
    /// hook (typically an aggressive compaction), the shrunk history is
    /// persisted into the live transcript, and the loop retries the same
    /// LLM call. Default `None` — today's behavior (the overflow ends
    /// the run). See [`crate::ContextOverflowRecovery`].
    pub(crate) overflow_recovery: Option<Arc<dyn ContextOverflowRecovery>>,

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
    /// "auto-when-forced" class where wire-level `tool_choice: "required"`
    /// is rejected and so plain text is the model's default failure mode —
    /// there's no benefit to running the narrowing-gate nudge cycle first
    /// because the model will emit prose every time. Default `false`
    /// preserves the post-narrowing gate for everyone else.
    pub plain_text_terminal_fallback_eager: bool,

    /// When true, the eager plain-text fallback path nudges the model with
    /// an explicit protocol-recovery system message BEFORE synthesizing a
    /// terminal tool result. Recovery continues until the model follows
    /// the protocol or the caller cancels. Default `false` preserves the
    /// original silent-synthesize behavior. Has no effect unless both
    /// [`Self::plain_text_terminal_fallback_tool`] and
    /// [`Self::plain_text_terminal_fallback_eager`] are set.
    pub plain_text_terminal_fallback_eager_nudge: bool,

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
    provider_extras: Option<Value>,
    max_iterations: Option<usize>,
    overflow_recovery: Option<Arc<dyn ContextOverflowRecovery>>,
    plain_text_terminal_fallback_tool: Option<String>,
    plain_text_terminal_fallback_eager: bool,
    plain_text_terminal_fallback_eager_nudge: bool,
    conversation_id: Option<String>,
    model_id: Option<String>,
    token_estimator: Arc<dyn TokenEstimator>,
    protocol: Arc<dyn ProtocolPolicy>,
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
            provider_extras: None,
            max_iterations: None,
            overflow_recovery: None,
            plain_text_terminal_fallback_tool: None,
            plain_text_terminal_fallback_eager: false,
            plain_text_terminal_fallback_eager_nudge: false,
            conversation_id: None,
            model_id: None,
            token_estimator: Arc::new(CharHeuristicEstimator),
            protocol: default_policy(),
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

    /// Set a hard ceiling on model iterations for one run.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n.max(1));
        self
    }

    /// Set the reasoning-effort knob forwarded to the stream transport
    /// on every turn. Per-run overrides flow through this typed surface
    /// rather than through stringly-typed provider extras.
    pub fn reasoning(mut self, level: ReasoningEffort) -> Self {
        self.reasoning = level;
        self
    }

    /// Set provider-specific extras forwarded to the stream transport
    /// on every turn (e.g., `response_format` for structured output
    /// enforcement).
    pub fn provider_extras(mut self, extras: Value) -> Self {
        self.provider_extras = Some(extras);
        self
    }

    /// Enable context-overflow recovery. When a request is rejected for
    /// exceeding the model's context window
    /// ([`crate::StreamError::ContextOverflow`]), the loop asks `recovery`
    /// for a smaller history, persists it, and retries the same LLM call.
    /// Off by default. See [`ContextOverflowRecovery`] for the contract.
    pub fn overflow_recovery<R: ContextOverflowRecovery + 'static>(mut self, recovery: R) -> Self {
        self.overflow_recovery = Some(Arc::new(recovery));
        self
    }

    /// [`Self::overflow_recovery`] for a pre-wrapped `Arc` (share one
    /// recovery across multiple builders).
    pub fn overflow_recovery_arc(mut self, recovery: Arc<dyn ContextOverflowRecovery>) -> Self {
        self.overflow_recovery = Some(recovery);
        self
    }

    /// Convert plain assistant text into a terminal tool result on
    /// terminal-only compatibility turns. Intended for providers that reject
    /// `tool_choice: "required"` and therefore can leak final prose even
    /// while the host advertises only delivery tools.
    pub fn plain_text_terminal_fallback_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.plain_text_terminal_fallback_tool = Some(tool_name.into());
        self
    }

    /// Make [`Self::plain_text_terminal_fallback_tool`] fire on the FIRST
    /// plain-text stop instead of waiting for the turn allowlist to be
    /// narrowed to terminators by a downstream tool gate. Use this for
    /// providers in the "auto-when-forced" class where wire-level forcing
    /// isn't available, so prose is the model's default failure mode and
    /// the nudge cycle just burns turns. Has no effect unless
    /// [`Self::plain_text_terminal_fallback_tool`] is also set.
    pub fn plain_text_terminal_fallback_eager(mut self, eager: bool) -> Self {
        self.plain_text_terminal_fallback_eager = eager;
        self
    }

    /// Make the eager plain-text fallback path nudge the model with an
    /// explicit protocol-recovery system message before synthesizing a
    /// terminal tool result. It keeps nudging until the model follows the
    /// protocol or the caller cancels. Has no effect unless
    /// both [`Self::plain_text_terminal_fallback_tool`] and
    /// [`Self::plain_text_terminal_fallback_eager`] are set.
    pub fn plain_text_terminal_fallback_eager_nudge(mut self, on: bool) -> Self {
        self.plain_text_terminal_fallback_eager_nudge = on;
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

    /// Install a [`ProtocolPolicy`] — the seam through which a downstream
    /// product supplies its tool vocabulary (plain-text recovery prose,
    /// tool-call alias repair, hidden-tool errors, terminal-tool
    /// classification). Defaults to [`crate::DefaultProtocolPolicy`] when
    /// not set, which keeps the core free of any product tool names. See
    /// [`crate::protocol`].
    pub fn protocol_policy(mut self, policy: Arc<dyn ProtocolPolicy>) -> Self {
        self.protocol = policy;
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

    pub fn build(self) -> Result<LoopConfig, BuilderError> {
        let stream = self.stream.ok_or(BuilderError::MissingStream)?;

        Ok(LoopConfig {
            stream,
            tools: self.tools,
            event_sink: self.event_sink,
            default_execution_mode: self.default_execution_mode,
            max_tool_calls_per_turn: self.max_tool_calls_per_turn,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            reasoning: self.reasoning,
            provider_extras: self.provider_extras,
            max_iterations: self.max_iterations,
            overflow_recovery: self.overflow_recovery,
            plain_text_terminal_fallback_tool: self.plain_text_terminal_fallback_tool,
            plain_text_terminal_fallback_eager: self.plain_text_terminal_fallback_eager,
            plain_text_terminal_fallback_eager_nudge: self.plain_text_terminal_fallback_eager_nudge,
            conversation_id: self.conversation_id,
            model_id: self.model_id,
            token_estimator: self.token_estimator,
            protocol: self.protocol,
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
    /// - default execution mode, `max_tool_calls_per_turn`
    /// - model id
    /// - protocol policy ([`ProtocolPolicy`])
    /// - plain-text-terminal fallback knobs
    /// - every plugin whose
    ///   [`crate::plugin::PluginCapabilities::inheritable_to_child`]
    ///   bit is set
    ///
    /// Does **not** inherit:
    /// - `event_sink` — callers install a child-scoped sink before
    ///   `build`.
    /// - `conversation_id` — the child should carry its own identity
    ///   via [`crate::AgentContext::identity`].
    /// - plugins that did **not** opt in to inheritance — they remain
    ///   parent-only.
    ///
    /// This is the single primitive for "spawn a fresh child agent with
    /// the same execution shape as me." A host runtime still registers
    /// any child-specific guards (delivery gates, terminal guards, etc.)
    /// on top of the returned builder.
    pub fn child_builder(&self) -> AgentBuilder {
        let mut builder = AgentBuilder::new()
            .stream(self.stream.clone())
            .tools_arc(self.tools.clone())
            .default_execution_mode(self.default_execution_mode)
            .reasoning(self.reasoning)
            .token_estimator_arc(self.token_estimator.clone())
            .protocol_policy(self.protocol.clone());
        if let Some(t) = self.temperature {
            builder = builder.temperature(t);
        }
        if let Some(m) = self.max_output_tokens {
            builder = builder.max_output_tokens(m);
        }
        if let Some(n) = self.max_tool_calls_per_turn {
            builder = builder.max_tool_calls_per_turn(n);
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
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        let names = child.plugin_names();
        assert_eq!(
            names.event_observer,
            vec!["inheritable"],
            "child must drop parent-only plugins"
        );
    }

    #[test]
    fn child_builder_carries_sampling_knobs() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .temperature(0.3)
            .max_output_tokens(8192)
            .max_tool_calls_per_turn(3)
            .model_id("test-model")
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        assert_eq!(child.temperature, Some(0.3));
        assert_eq!(child.max_output_tokens, Some(8192));
        assert_eq!(child.max_tool_calls_per_turn, Some(3));
        assert_eq!(child.model_id.as_deref(), Some("test-model"));
    }

    #[test]
    fn child_builder_gets_an_independent_iteration_budget() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .max_iterations(30)
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        assert_eq!(parent.max_iterations, Some(30));
        assert_eq!(child.max_iterations, None);
    }
}
