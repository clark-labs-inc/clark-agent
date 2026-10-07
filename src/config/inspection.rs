use super::*;

/// Snapshot of registered plugin names per category, in registration order.
///
/// Returned by [`LoopConfig::plugin_names`] for inspection / regression
/// tests. Order matches the order the loop will invoke each plugin
/// (left-to-right composition for `ContextTransform`, etc.). Pure read
/// — does not clone the plugins themselves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginNames {
    pub run_guard: Vec<&'static str>,
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

        for guard in &self.plugins.run_guard {
            if guard.capabilities().inheritable_to_child {
                builder = builder.run_guard_arc(guard.clone());
            }
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
            run_guard: self.plugins.run_guard.iter().map(|guard| guard.name()).collect(),
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

