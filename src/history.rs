//! Provider-facing history invariants.
//!
//! Durable transcripts may contain tool-call identifiers chosen by different
//! provider turns. Some OpenAI-compatible APIs require every identifier in a
//! request to be globally unique, even when each call/result pair is otherwise
//! valid. This module repairs that wire-facing invariant without rewriting the
//! persisted transcript.

use std::collections::{HashMap, HashSet, VecDeque};

use async_trait::async_trait;

use crate::plugin::{ContextTransform, Plugin, PluginCapabilities, TransformContext};
use crate::types::{AgentMessage, AssistantBlock};

/// Result of normalizing one provider-visible history slice.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallIdNormalization {
    pub messages: Vec<AgentMessage>,
    pub renamed_count: usize,
}

/// Make tool-call IDs unique across a provider-visible history.
///
/// The first occurrence keeps its original identifier. Later occurrences are
/// renamed deterministically, together with positionally corresponding tool
/// results immediately following that assistant turn. Existing identifiers are
/// reserved so generated replacements cannot collide with transcript data.
pub fn normalize_tool_call_ids(messages: Vec<AgentMessage>) -> ToolCallIdNormalization {
    let mut reserved = HashSet::new();
    for message in &messages {
        match message {
            AgentMessage::Assistant { content, .. } => {
                for block in &content.blocks {
                    if let AssistantBlock::ToolCall(call) = block {
                        reserved.insert(call.id.clone());
                    }
                }
            }
            AgentMessage::ToolResult { tool_call_id, .. } => {
                reserved.insert(tool_call_id.clone());
            }
            _ => {}
        }
    }

    let mut used = HashSet::new();
    let mut next_suffix = 1usize;
    let mut renamed_count = 0usize;
    let mut normalized = Vec::with_capacity(messages.len());
    let mut input = messages.into_iter().peekable();

    while let Some(mut message) = input.next() {
        let AgentMessage::Assistant { content, .. } = &mut message else {
            normalized.push(message);
            continue;
        };

        let mut result_ids: HashMap<String, VecDeque<String>> = HashMap::new();
        for block in &mut content.blocks {
            let AssistantBlock::ToolCall(call) = block else {
                continue;
            };
            let original = call.id.clone();
            let wire_id = if used.insert(original.clone()) {
                original.clone()
            } else {
                let replacement = loop {
                    let candidate = format!("clark_agent_call_{next_suffix}");
                    next_suffix += 1;
                    if reserved.insert(candidate.clone()) {
                        break candidate;
                    }
                };
                used.insert(replacement.clone());
                call.id = replacement.clone();
                renamed_count += 1;
                replacement
            };
            result_ids.entry(original).or_default().push_back(wire_id);
        }
        normalized.push(message);

        while matches!(input.peek(), Some(AgentMessage::ToolResult { .. })) {
            let mut result = input.next().expect("peeked tool result");
            if let AgentMessage::ToolResult { tool_call_id, .. } = &mut result {
                if let Some(wire_id) = result_ids
                    .get_mut(tool_call_id.as_str())
                    .and_then(VecDeque::pop_front)
                {
                    *tool_call_id = wire_id;
                }
            }
            normalized.push(result);
        }
    }

    ToolCallIdNormalization {
        messages: normalized,
        renamed_count,
    }
}

/// Stateless context transform that applies [`normalize_tool_call_ids`] before
/// each provider request.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniqueToolCallIds;

impl Plugin for UniqueToolCallIds {
    fn name(&self) -> &'static str {
        "unique_tool_call_ids"
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::context_transform()
    }
}

#[async_trait]
impl ContextTransform for UniqueToolCallIds {
    async fn transform(
        &self,
        messages: Vec<AgentMessage>,
        _cx: &TransformContext<'_>,
    ) -> Vec<AgentMessage> {
        let normalized = normalize_tool_call_ids(messages);
        if normalized.renamed_count > 0 {
            tracing::warn!(
                renamed_count = normalized.renamed_count,
                "normalized duplicate tool-call IDs in provider history"
            );
        }
        normalized.messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolCall;
    use crate::types::{AssistantContent, StopReason, ToolResultContent, UserContent};
    use serde_json::json;

    fn user(text: &str) -> AgentMessage {
        AgentMessage::User {
            content: UserContent::Text(text.to_string()),
            timestamp: None,
        }
    }

    fn assistant(ids: &[&str]) -> AgentMessage {
        AgentMessage::Assistant {
            content: AssistantContent {
                blocks: ids
                    .iter()
                    .map(|id| {
                        AssistantBlock::ToolCall(ToolCall {
                            id: (*id).to_string(),
                            name: "shell".to_string(),
                            arguments: json!({}),
                        })
                    })
                    .collect(),
            },
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: None,
            usage: None,
        }
    }

    fn result(id: &str) -> AgentMessage {
        AgentMessage::ToolResult {
            tool_call_id: id.to_string(),
            tool_name: "shell".to_string(),
            content: ToolResultContent::text("ok"),
            is_error: false,
            narration: None,
            details: None,
            timestamp: None,
        }
    }

    fn call_ids(messages: &[AgentMessage]) -> Vec<&str> {
        messages
            .iter()
            .flat_map(|message| match message {
                AgentMessage::Assistant { content, .. } => content
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        AssistantBlock::ToolCall(call) => Some(call.id.as_str()),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    fn result_ids(messages: &[AgentMessage]) -> Vec<&str> {
        messages
            .iter()
            .filter_map(|message| match message {
                AgentMessage::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn reused_shell_id_is_renamed_with_its_result() {
        let normalized = normalize_tool_call_ids(vec![
            user("first"),
            assistant(&["shell:89"]),
            result("shell:89"),
            user("second"),
            assistant(&["shell:89"]),
            result("shell:89"),
        ]);

        assert_eq!(normalized.renamed_count, 1);
        assert_eq!(
            call_ids(&normalized.messages),
            vec!["shell:89", "clark_agent_call_1"]
        );
        assert_eq!(
            result_ids(&normalized.messages),
            vec!["shell:89", "clark_agent_call_1"]
        );
    }

    #[test]
    fn duplicate_ids_in_one_parallel_batch_pair_by_position() {
        let normalized = normalize_tool_call_ids(vec![
            user("parallel"),
            assistant(&["shell:89", "shell:89"]),
            result("shell:89"),
            result("shell:89"),
        ]);

        assert_eq!(
            call_ids(&normalized.messages),
            vec!["shell:89", "clark_agent_call_1"]
        );
        assert_eq!(
            result_ids(&normalized.messages),
            vec!["shell:89", "clark_agent_call_1"]
        );
    }

    #[test]
    fn normalization_is_idempotent_and_avoids_reserved_replacements() {
        let once = normalize_tool_call_ids(vec![
            assistant(&["shell:89"]),
            result("shell:89"),
            assistant(&["shell:89", "clark_agent_call_1"]),
            result("shell:89"),
            result("clark_agent_call_1"),
        ]);
        let twice = normalize_tool_call_ids(once.messages.clone());

        assert_eq!(once.renamed_count, 1);
        assert_eq!(twice.renamed_count, 0);
        assert_eq!(once.messages, twice.messages);
        assert_eq!(
            call_ids(&twice.messages),
            vec!["shell:89", "clark_agent_call_2", "clark_agent_call_1"]
        );
    }
}
