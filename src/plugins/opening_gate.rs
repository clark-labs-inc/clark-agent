//! `OpeningGate` — narrow the very first LLM call's tool list to a
//! framing/delivery/clarification subset
//! (`[message_result, message_info, plan, message_ask]`).
//!
//! Pairs with the `<agent_loop>` prompt sentence telling the model
//! that its first response must call one of those tools. Two contracts
//! together: the gate enforces it on the wire, the prompt explains it
//! in prose. The model behaves predictably even when the system prompt
//! is partially ignored, because the wire-level constraint is harder
//! to ignore than text instructions.
//!
//! ## Why restored
//!
//! An earlier version of this gate also offered a `PerConversation`
//! scope backed by a `ConversationGateRegistry`. The 2026-05-02
//! migration pulled the gate AND the registry without a replacement,
//! which in turn caused models to skip planning entirely on every
//! eval matrix run (no plan call → no capability tags → downstream
//! `CapabilityGate` and `WorkProgressHook` machinery never engages).
//!
//! This restoration ships the simpler per-run-only flavor. The
//! per-conversation scope can come back when the registry trait is
//! reintroduced; in the meantime, per-run is the default that was
//! enabled in production for months and the one matrix scenarios
//! depend on.

use async_trait::async_trait;
use std::collections::HashSet;

use crate::plugin::{Plugin, PluginCapabilities, ToolGate, ToolGateContext};

/// Default opener allowlist: direct delivery, framing tools, and genuine
/// clarification.
/// Excludes every other tool — no browsing, no file work, no shell, no
/// sandbox work — until after the opening turn.
fn default_opener_allowlist() -> HashSet<String> {
    ["message_result", "message_info", "plan", "message_ask"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Narrows iteration 0's tool advertisement to a framing subset.
/// Composes via allowlist intersection with other `ToolGate` plugins.
pub struct OpeningGate {
    allowlist: HashSet<String>,
}

impl OpeningGate {
    /// Construct a gate using the default acknowledgement allowlist
    /// (`[message_result, message_info, plan, message_ask]`).
    pub fn for_acknowledgement() -> Self {
        Self {
            allowlist: default_opener_allowlist(),
        }
    }

    /// Construct a gate with a custom allowlist. Use only when an
    /// experiment needs a different opening contract.
    pub fn with_allowlist(allowlist: HashSet<String>) -> Self {
        Self { allowlist }
    }
}

impl Default for OpeningGate {
    fn default() -> Self {
        Self::for_acknowledgement()
    }
}

impl Plugin for OpeningGate {
    fn name(&self) -> &'static str {
        "opening_gate"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::tool_gate()
    }
}

#[async_trait]
impl ToolGate for OpeningGate {
    async fn next_turn_tool_allowlist(&self, ctx: ToolGateContext<'_>) -> Option<HashSet<String>> {
        if ctx.iteration != 0 {
            return None;
        }
        // Intersect the configured allowlist with what the registry
        // actually has. If a tool we want to advertise isn't in the
        // registry (test fixture, surface that disabled it, etc.),
        // don't synthesize the name on the wire.
        let mut allowed = HashSet::new();
        for tool in ctx.available_tool_names {
            if self.allowlist.contains(*tool) {
                allowed.insert((*tool).to_string());
            }
        }
        Some(allowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(iteration: usize, tools: &'static [&'static str]) -> ToolGateContext<'static> {
        ToolGateContext {
            iteration,
            messages: &[],
            conversation_id: None,
            available_tool_names: tools,
        }
    }

    #[tokio::test]
    async fn gate_fires_on_iteration_zero_with_intersection() {
        let gate = OpeningGate::for_acknowledgement();
        let allowed = gate
            .next_turn_tool_allowlist(ctx(
                0,
                &[
                    "plan",
                    "message_result",
                    "message_info",
                    "message_ask",
                    "skill",
                    "shell",
                    "file_write",
                    "update_working_checkpoint",
                ],
            ))
            .await
            .expect("opening turn should narrow");
        assert!(allowed.contains("plan"));
        assert!(allowed.contains("message_result"));
        assert!(allowed.contains("message_info"));
        assert!(allowed.contains("message_ask"));
        assert!(
            !allowed.contains("skill"),
            "opening turn must hide skill loading until after the initial plan"
        );
        assert!(
            !allowed.contains("shell"),
            "opening turn must hide work tools so the model frames first"
        );
        assert!(
            !allowed.contains("file_write"),
            "opening turn must hide work tools"
        );
        assert!(
            !allowed.contains("update_working_checkpoint"),
            "checkpoint tool is hidden in the opening turn — model can call it after the initial plan"
        );
    }

    #[tokio::test]
    async fn gate_returns_none_after_first_iteration() {
        let gate = OpeningGate::for_acknowledgement();
        let result = gate
            .next_turn_tool_allowlist(ctx(1, &["plan", "shell"]))
            .await;
        assert!(
            result.is_none(),
            "iteration > 0 must NOT narrow — let the model use the full catalog"
        );
    }

    #[tokio::test]
    async fn allowlist_does_not_synthesize_tools_missing_from_registry() {
        // Test fixture only has message_ask, message_result, and shell.
        // Even though the gate would normally allow plan/message_info,
        // they aren't in the registry — don't fabricate them on the wire.
        let gate = OpeningGate::for_acknowledgement();
        let allowed = gate
            .next_turn_tool_allowlist(ctx(0, &["message_ask", "message_result", "shell"]))
            .await
            .unwrap();
        assert_eq!(allowed.len(), 2);
        assert!(allowed.contains("message_ask"));
        assert!(allowed.contains("message_result"));
        assert!(!allowed.contains("plan"));
        assert!(!allowed.contains("message_info"));
        assert!(!allowed.contains("shell"));
    }

    #[tokio::test]
    async fn custom_allowlist_overrides_default() {
        let mut custom = HashSet::new();
        custom.insert("plan".to_string());
        let gate = OpeningGate::with_allowlist(custom);
        let allowed = gate
            .next_turn_tool_allowlist(ctx(0, &["plan", "message_info", "message_ask"]))
            .await
            .unwrap();
        assert_eq!(allowed.len(), 1);
        assert!(allowed.contains("plan"));
        assert!(!allowed.contains("message_info"));
    }
}
