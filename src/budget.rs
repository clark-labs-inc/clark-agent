//! Default token-budget context transform.
//!
//! Full provider history is the source of truth. When the conversation
//! grows past the budget, we drop the oldest tool results' content
//! (keeping the call shape) so the model still sees
//! what it tried, but doesn't pay token cost for stale, re-fetchable
//! payloads.
//!
//! Token counts come from the loop's configured
//! [`crate::tokens::TokenEstimator`] (default: char heuristic; apps may
//! plug a real tokenizer). Apps that want an entirely different policy
//! implement their own `ContextTransform` plugin and skip this one.

use async_trait::async_trait;

use crate::plugin::{ContextTransform, Plugin, PluginCapabilities, TransformContext};
use crate::tokens::TokenEstimator;
use crate::types::{AgentMessage, TextContent, ToolResultBlock, ToolResultContent};

/// Configurable token budget. Default `60_000` tokens with a 70% trim
/// trigger.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub max_tokens: usize,
    /// When total estimate exceeds `trim_trigger * max_tokens`, start
    /// truncating tool results.
    pub trim_trigger: f32,
    /// Replacement text inserted in place of truncated tool result content.
    pub truncation_marker: String,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_tokens: 60_000,
            trim_trigger: 0.7,
            truncation_marker: "[truncated for context budget — re-run tool to refetch]".into(),
        }
    }
}

impl Plugin for TokenBudget {
    fn name(&self) -> &'static str {
        "token_budget"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::context_transform()
    }
}

#[async_trait]
impl ContextTransform for TokenBudget {
    async fn transform(
        &self,
        mut messages: Vec<AgentMessage>,
        cx: &TransformContext<'_>,
    ) -> Vec<AgentMessage> {
        let trigger = (self.max_tokens as f32 * self.trim_trigger).round() as usize;
        let total = cx.estimator.estimate_messages(&messages);
        if total <= trigger {
            return messages;
        }

        // Walk oldest-first (skip the very last user/tool exchange so the
        // current turn keeps full fidelity). Replace tool result content
        // with the marker until under budget or nothing left to truncate.
        let last_idx = messages.len().saturating_sub(2);
        let mut idx = 0;
        while idx < last_idx {
            let truncated = if let AgentMessage::ToolResult { content, .. } = &mut messages[idx] {
                if !content_already_marker(content, &self.truncation_marker) {
                    *content = ToolResultContent {
                        blocks: vec![ToolResultBlock::Text(TextContent {
                            text: self.truncation_marker.clone(),
                        })],
                    };
                    true
                } else {
                    false
                }
            } else {
                false
            };
            idx += 1;
            if truncated {
                let total = cx.estimator.estimate_messages(&messages);
                if total <= trigger {
                    break;
                }
            }
        }

        messages
    }
}

/// Char-heuristic estimate kept as a free function for callers that
/// want a one-off count without holding a `TokenEstimator`. Prefer
/// [`crate::plugin::TransformContext::estimator`] inside transforms.
pub fn estimate_tokens(message: &AgentMessage) -> usize {
    crate::tokens::CHAR_HEURISTIC.estimate_message(message)
}

fn content_already_marker(content: &ToolResultContent, marker: &str) -> bool {
    content.blocks.len() == 1
        && matches!(
            &content.blocks[0],
            ToolResultBlock::Text(t) if t.text == marker
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolResultBlock;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn budget_truncates_old_tool_results() {
        let budget = TokenBudget {
            max_tokens: 200,
            trim_trigger: 0.5,
            truncation_marker: "[trunc]".into(),
        };
        let big = "x".repeat(2000);
        let messages = vec![
            AgentMessage::User {
                content: crate::types::UserContent::Text("start".into()),
                timestamp: None,
            },
            AgentMessage::ToolResult {
                tool_call_id: "1".into(),
                tool_name: "shell".into(),
                content: ToolResultContent::text(big.clone()),
                is_error: false,
                narration: None,
                details: None,
                timestamp: None,
            },
            AgentMessage::User {
                content: crate::types::UserContent::Text("more".into()),
                timestamp: None,
            },
            AgentMessage::ToolResult {
                tool_call_id: "2".into(),
                tool_name: "shell".into(),
                content: ToolResultContent::text(big),
                is_error: false,
                narration: None,
                details: None,
                timestamp: None,
            },
        ];
        let token = CancellationToken::new();
        let cx = TransformContext::for_test(&token);
        let out = budget.transform(messages, &cx).await;
        // Oldest tool result should be truncated.
        let AgentMessage::ToolResult { content, .. } = &out[1] else {
            panic!("expected tool result");
        };
        assert!(matches!(&content.blocks[0], ToolResultBlock::Text(t) if t.text == "[trunc]"));
        // Last tool result preserved.
        let AgentMessage::ToolResult { content, .. } = &out[3] else {
            panic!("expected tool result");
        };
        assert!(matches!(&content.blocks[0], ToolResultBlock::Text(t) if t.text != "[trunc]"));
    }
}
