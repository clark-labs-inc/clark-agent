//! `OpeningGate` — narrow the very first LLM call's tool list to a
//! caller-supplied subset.
//!
//! A common product pattern is "the model must frame / plan / clarify on
//! its opening turn before any work tools are available." This gate
//! enforces that on the wire: on iteration 0 it advertises only the tools
//! whose names are in the configured allowlist (intersected with the tools
//! the registry actually has); from iteration 1 on it imposes no
//! narrowing. Pair it with a system-prompt sentence that explains the
//! same contract in prose — the wire-level constraint is harder for a
//! model to ignore than text instructions.
//!
//! The gate ships **no** default allowlist: it is vocabulary-free, so the
//! core stays free of any product's tool names. Callers supply the
//! opening subset via [`OpeningGate::with_allowlist`]. Composes with other
//! [`ToolGate`] plugins through allowlist intersection.

use async_trait::async_trait;
use std::collections::HashSet;

use crate::plugin::{Plugin, PluginCapabilities, ToolGate, ToolGateContext};

/// Narrows iteration 0's tool advertisement to a caller-supplied subset.
/// Composes via allowlist intersection with other `ToolGate` plugins.
pub struct OpeningGate {
    allowlist: HashSet<String>,
}

impl OpeningGate {
    /// Construct a gate that narrows the opening turn to `allowlist`.
    /// The names are product-specific and supplied by the caller — the
    /// gate itself knows no tool vocabulary.
    pub fn with_allowlist(allowlist: HashSet<String>) -> Self {
        Self { allowlist }
    }

    /// Convenience constructor from an iterator of tool names.
    pub fn new<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowlist: tools.into_iter().map(Into::into).collect(),
        }
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
        // A product allowlist of framing tools; work tools must be hidden
        // on the opening turn.
        let gate = OpeningGate::new(["frame", "deliver", "ask"]);
        let allowed = gate
            .next_turn_tool_allowlist(ctx(
                0,
                &[
                    "frame",
                    "deliver",
                    "ask",
                    "load_skill",
                    "shell",
                    "file_write",
                ],
            ))
            .await
            .expect("opening turn should narrow");
        assert!(allowed.contains("frame"));
        assert!(allowed.contains("deliver"));
        assert!(allowed.contains("ask"));
        assert!(
            !allowed.contains("load_skill"),
            "opening turn must hide non-framing tools"
        );
        assert!(
            !allowed.contains("shell"),
            "opening turn must hide work tools so the model frames first"
        );
        assert!(
            !allowed.contains("file_write"),
            "opening turn must hide work tools"
        );
    }

    #[tokio::test]
    async fn gate_returns_none_after_first_iteration() {
        let gate = OpeningGate::new(["frame", "shell"]);
        let result = gate
            .next_turn_tool_allowlist(ctx(1, &["frame", "shell"]))
            .await;
        assert!(
            result.is_none(),
            "iteration > 0 must NOT narrow — let the model use the full catalog"
        );
    }

    #[tokio::test]
    async fn allowlist_does_not_synthesize_tools_missing_from_registry() {
        // The gate would allow `frame`/`deliver`, but only `ask`/`deliver`
        // and `shell` are in the registry — don't fabricate names on the
        // wire.
        let gate = OpeningGate::new(["frame", "deliver", "ask"]);
        let allowed = gate
            .next_turn_tool_allowlist(ctx(0, &["ask", "deliver", "shell"]))
            .await
            .unwrap();
        assert_eq!(allowed.len(), 2);
        assert!(allowed.contains("ask"));
        assert!(allowed.contains("deliver"));
        assert!(!allowed.contains("frame"));
        assert!(!allowed.contains("shell"));
    }

    #[tokio::test]
    async fn with_allowlist_takes_an_explicit_set() {
        let mut custom = HashSet::new();
        custom.insert("frame".to_string());
        let gate = OpeningGate::with_allowlist(custom);
        let allowed = gate
            .next_turn_tool_allowlist(ctx(0, &["frame", "deliver", "ask"]))
            .await
            .unwrap();
        assert_eq!(allowed.len(), 1);
        assert!(allowed.contains("frame"));
        assert!(!allowed.contains("deliver"));
    }
}
