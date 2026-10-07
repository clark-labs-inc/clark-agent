use super::*;

impl AgentBuilder {
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

}
