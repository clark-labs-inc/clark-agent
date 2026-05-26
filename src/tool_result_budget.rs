//! Per-tool-result content cap.
//!
//! `TokenBudget` (the global trim) only fires when the *total* context
//! crosses the budget — by then, a single oversized tool output has
//! already polluted multiple turns of cache and crowded out other
//! observations. `ToolResultBudget` runs immediately after structural
//! history repair in the `ContextTransform` chain and clips
//! per-tool-result before global pressure ever builds up.
//!
//! Cheapest-first ordering follows the loop's lazy-degradation rule: the
//! least-disruptive compression layer fires first; later layers only
//! see what survived. After structural validity is restored, per-tool
//! clipping is cheaper than recompacting and cheaper than summarizing,
//! so it earns the first compression slot.
//!
//! The full result stays in the persisted event log
//! (`AgentEvent::ToolExecutionEnd` carries the original) and in the
//! in-memory context. Only the projection sent to the provider is
//! clipped, so resume reconstructs the original messages and re-applies
//! this transform — no destructive edits, no new persistence shape.

use std::sync::Arc;

use async_trait::async_trait;

use crate::plugin::{ContextTransform, Plugin, PluginCapabilities, TransformContext};
use crate::tool::ToolRegistry;
use crate::types::{AgentMessage, TextContent, ToolResultBlock, ToolResultContent};

/// Default per-tool cap when neither the tool nor the deployment
/// declares one. 32 kchars ≈ 8k tokens by the char heuristic — large
/// enough that ordinary tool output stays verbatim, small enough that
/// a single runaway result can't pin a whole turn.
pub const DEFAULT_PER_TOOL_CHARS: usize = 32_000;

/// Maximum size of the marker substring that replaces clipped content.
/// The marker carries the original size so the model can decide
/// whether to re-run the tool, but it must not itself be a budget
/// problem on transcripts with many clipped results.
const MARKER_BUDGET_CHARS: usize = 256;

/// `ContextTransform` that caps the size of individual `ToolResult`
/// content blocks per turn, ahead of any global budget pass.
///
/// Looks up `AgentTool::max_result_chars()` for each tool name to get
/// the per-tool cap; falls back to `default_max_chars` when the tool
/// doesn't declare one. `Some(usize::MAX)` from a tool means "leave
/// verbatim" — no clip happens for that tool.
pub struct ToolResultBudget {
    /// Cap applied to tools whose `max_result_chars()` returns `None`.
    pub default_max_chars: usize,
    /// Used to resolve per-tool overrides via `AgentTool::max_result_chars()`.
    /// Shared with `LoopConfig.tools` (same `Arc`) so the plugin sees
    /// whatever registry the rest of the loop sees.
    registry: Arc<ToolRegistry>,
}

impl ToolResultBudget {
    /// Construct with the default per-tool cap. The registry should
    /// be the same `Arc` handed to `AgentBuilder::tools_arc`.
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            default_max_chars: DEFAULT_PER_TOOL_CHARS,
            registry,
        }
    }

    /// Override the global per-tool cap. Tools that declare their own
    /// `max_result_chars()` are unaffected.
    pub fn with_default_max_chars(mut self, chars: usize) -> Self {
        self.default_max_chars = chars;
        self
    }

    /// Effective cap for a given tool name. Looks up the tool in the
    /// registry; if the tool declares an explicit override, use it,
    /// otherwise fall back to the default. Tools not in the registry
    /// (synthetic / aliased / removed-since) get the default.
    fn cap_for(&self, tool_name: &str) -> usize {
        self.registry
            .get(tool_name)
            .and_then(|tool| tool.max_result_chars())
            .unwrap_or(self.default_max_chars)
    }
}

impl Plugin for ToolResultBudget {
    fn name(&self) -> &'static str {
        "tool_result_budget"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::context_transform()
    }
}

#[async_trait]
impl ContextTransform for ToolResultBudget {
    async fn transform(
        &self,
        mut messages: Vec<AgentMessage>,
        _cx: &TransformContext<'_>,
    ) -> Vec<AgentMessage> {
        // Skip the very last tool result so the most recent observation
        // stays verbatim — the model needs full fidelity on the freshest
        // result it's reasoning about. Anything older that overflows
        // gets clipped.
        let last_tool_idx = messages
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, m)| matches!(m, AgentMessage::ToolResult { .. }).then_some(idx));

        for (idx, message) in messages.iter_mut().enumerate() {
            let AgentMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } = message
            else {
                continue;
            };
            if Some(idx) == last_tool_idx {
                continue;
            }
            let cap = self.cap_for(tool_name);
            if cap == usize::MAX {
                continue;
            }
            let original = content_chars(content);
            if original <= cap {
                continue;
            }
            if is_already_marker(content) {
                continue;
            }
            let marker = render_marker(tool_call_id, tool_name, original, cap);
            *content = ToolResultContent {
                blocks: vec![ToolResultBlock::Text(TextContent { text: marker })],
            };
        }

        messages
    }
}

fn content_chars(content: &ToolResultContent) -> usize {
    content
        .blocks
        .iter()
        .map(|b| match b {
            ToolResultBlock::Text(t) => t.text.len(),
            // Image blocks have no usable char-size signal and are
            // rare; leave them untouched. A future audio/binary block
            // would land here.
            ToolResultBlock::Image(_) => 0,
        })
        .sum()
}

/// Marker prefix used both to render new markers and to detect prior
/// truncations so the transform stays idempotent across re-applies.
const MARKER_PREFIX: &str = "[tool_result_budget: clipped";

fn render_marker(tool_call_id: &str, tool_name: &str, original_chars: usize, cap: usize) -> String {
    let body = format!(
        "{MARKER_PREFIX} {tool_name} result of {original_chars} chars to {cap} cap; \
         tool_call_id={tool_call_id}; rerun the tool to refetch the original output]"
    );
    if body.len() <= MARKER_BUDGET_CHARS {
        body
    } else {
        // Defensive: truncate the marker itself if a pathological
        // tool_call_id ever pushes it past the marker budget.
        let mut t = body;
        t.truncate(MARKER_BUDGET_CHARS);
        t
    }
}

fn is_already_marker(content: &ToolResultContent) -> bool {
    content.blocks.len() == 1
        && matches!(
            &content.blocks[0],
            ToolResultBlock::Text(t) if t.text.starts_with(MARKER_PREFIX)
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ToolError;
    use crate::tool::{AgentTool, ToolResult, ToolUpdateSink};
    use async_trait::async_trait;
    use serde_json::Value;
    use tokio_util::sync::CancellationToken;

    struct FakeTool {
        name: String,
        cap: Option<usize>,
    }

    #[async_trait]
    impl AgentTool for FakeTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            ""
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        fn max_result_chars(&self) -> Option<usize> {
            self.cap
        }
        async fn execute(
            &self,
            _call_id: &str,
            _args: Value,
            _signal: CancellationToken,
            _update: ToolUpdateSink,
        ) -> Result<ToolResult, ToolError> {
            unreachable!("not invoked in budget tests")
        }
    }

    fn registry_with(tools: Vec<(&str, Option<usize>)>) -> Arc<ToolRegistry> {
        let mut r = ToolRegistry::new();
        for (name, cap) in tools {
            r.register(Arc::new(FakeTool {
                name: name.into(),
                cap,
            }));
        }
        Arc::new(r)
    }

    fn tool_result(id: &str, name: &str, body: String) -> AgentMessage {
        AgentMessage::ToolResult {
            tool_call_id: id.into(),
            tool_name: name.into(),
            content: ToolResultContent::text(body),
            is_error: false,
            narration: None,
            details: None,
            timestamp: None,
        }
    }

    fn user(text: &str) -> AgentMessage {
        AgentMessage::User {
            content: crate::types::UserContent::Text(text.into()),
            timestamp: None,
        }
    }

    fn block_text(message: &AgentMessage) -> &str {
        let AgentMessage::ToolResult { content, .. } = message else {
            panic!("expected tool result");
        };
        let ToolResultBlock::Text(t) = &content.blocks[0] else {
            panic!("expected text block");
        };
        &t.text
    }

    #[tokio::test]
    async fn clips_oversize_tool_results_above_default_cap() {
        let registry = registry_with(vec![("shell", None)]);
        let budget = ToolResultBudget::new(registry).with_default_max_chars(100);
        let big = "x".repeat(500);
        let messages = vec![
            user("hi"),
            tool_result("a", "shell", big.clone()),
            user("again"),
            tool_result("b", "shell", big),
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let out = budget.transform(messages, &cx).await;
        // First (older) tool result clipped.
        assert!(block_text(&out[1]).starts_with(MARKER_PREFIX));
        // Last (newer) tool result preserved verbatim.
        assert_eq!(block_text(&out[3]).len(), 500);
    }

    #[tokio::test]
    async fn preserves_tool_results_within_cap() {
        let registry = registry_with(vec![("shell", None)]);
        let budget = ToolResultBudget::new(registry).with_default_max_chars(100);
        let small = "x".repeat(50);
        let messages = vec![
            user("hi"),
            tool_result("a", "shell", small.clone()),
            user("again"),
            tool_result("b", "shell", small),
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let out = budget.transform(messages.clone(), &cx).await;
        assert_eq!(out, messages);
    }

    #[tokio::test]
    async fn per_tool_override_unlimited_keeps_verbatim() {
        let registry = registry_with(vec![("publish", Some(usize::MAX))]);
        let budget = ToolResultBudget::new(registry).with_default_max_chars(100);
        let big = "x".repeat(500);
        let messages = vec![
            user("hi"),
            tool_result("a", "publish", big.clone()),
            user("more"),
            user("again"),
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let out = budget.transform(messages, &cx).await;
        // Even though it's an old result, the unlimited cap keeps it.
        assert_eq!(block_text(&out[1]).len(), 500);
    }

    #[tokio::test]
    async fn per_tool_override_smaller_clips_below_default() {
        let registry = registry_with(vec![("verbose", Some(50))]);
        let budget = ToolResultBudget::new(registry).with_default_max_chars(1_000_000);
        let body = "x".repeat(200);
        let messages = vec![
            user("hi"),
            tool_result("a", "verbose", body.clone()),
            user("more"),
            tool_result("b", "verbose", body),
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let out = budget.transform(messages, &cx).await;
        assert!(block_text(&out[1]).starts_with(MARKER_PREFIX));
    }

    #[tokio::test]
    async fn idempotent_across_repeated_apply() {
        let registry = registry_with(vec![("shell", None)]);
        let budget = ToolResultBudget::new(registry).with_default_max_chars(100);
        let big = "x".repeat(500);
        let messages = vec![
            user("hi"),
            tool_result("a", "shell", big.clone()),
            user("again"),
            tool_result("b", "shell", big),
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let once = budget.transform(messages, &cx).await;
        let twice = budget.transform(once.clone(), &cx).await;
        assert_eq!(once, twice);
    }

    #[tokio::test]
    async fn unknown_tool_falls_back_to_default_cap() {
        let registry = registry_with(vec![]);
        let budget = ToolResultBudget::new(registry).with_default_max_chars(100);
        let big = "x".repeat(500);
        let messages = vec![
            user("hi"),
            tool_result("a", "synthetic", big.clone()),
            user("again"),
            tool_result("b", "synthetic", big),
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let out = budget.transform(messages, &cx).await;
        assert!(block_text(&out[1]).starts_with(MARKER_PREFIX));
    }
}
