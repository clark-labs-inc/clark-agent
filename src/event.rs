//! Typed events emitted by the loop.
//!
//! Single sink, single enum. Streaming consumers pattern-match on the
//! event kind. Events are observation-only — they cannot change loop
//! state. Plugins that need to mutate state use the dedicated capability
//! traits in [`crate::plugin`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use crate::stream::{AssistantStreamChunk, ToolSchema};
use crate::tool::ToolResult;
use crate::types::{
    AgentMessage, AssistantBlock, RunIdentity, ToolResultBlock, UserBlock, UserContent,
};

/// All events the loop emits.
///
/// Lifecycle events (`AgentStart`, `AgentEnd`, `TurnStart`, `TurnEnd`)
/// bracket the run. Message events (`MessageStart`, `MessageUpdate`,
/// `MessageEnd`) bracket each individual message. Tool events
/// (`ToolExecutionStart`, `ToolExecutionUpdate`, `ToolExecutionEnd`)
/// bracket each tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// First event in a run. Emitted once.
    AgentStart,

    /// Run identity, emitted immediately after [`AgentEvent::AgentStart`]
    /// when the context carries a [`RunIdentity`]. Trajectory sinks key
    /// every subsequent event of the same run on `identity.run_id`;
    /// child runs surface their `parent_run_id` so the spawn tree
    /// rebuilds without external bookkeeping.
    ///
    /// Existing observers that don't care about identity ignore this
    /// variant (every match arm in the tree already has a wildcard
    /// fallback). Plugins and sinks that want identity pattern-match
    /// directly.
    RunIdentified { identity: RunIdentity },

    /// Last event in a run. Carries the messages produced *during this run*
    /// (not the full transcript). Listeners that want the full transcript
    /// should fold prior messages into a state of their own.
    AgentEnd { messages: Vec<AgentMessage> },

    /// Bracket: a new turn begins. A turn is one assistant response plus
    /// any tool calls/results it spawned.
    TurnStart,

    /// Bracket: a turn ends. Carries the assistant message and the tool
    /// results for that turn (empty if the model didn't call any tools).
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<AgentMessage>,
    },

    /// A message has been added to the transcript (user, assistant, or
    /// tool result). For assistant messages, this fires before streaming
    /// begins; subsequent `MessageUpdate` events carry deltas.
    MessageStart { message: AgentMessage },

    /// Streaming delta for the in-progress assistant message.
    MessageUpdate {
        partial: AgentMessage,
        chunk: AssistantStreamChunk,
    },

    /// The message has been fully assembled (final content, stop reason).
    MessageEnd { message: AgentMessage },

    /// A tool execution has begun.
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },

    /// Partial progress from a long-running tool. The tool calls
    /// `update.send(...)` to surface intermediate state without ending.
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        partial: ToolResult,
    },

    /// A tool execution has finished.
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: ToolResult,
        is_error: bool,
    },

    /// The loop discarded a truncated assistant turn and re-streamed
    /// with a higher `max_output_tokens` cap. Emitted once per
    /// retry attempt; multiple events for the same turn signal the
    /// recovery walked the configured ladder. See
    /// [`crate::config::MaxTokensRecovery`].
    OutputTokensEscalation {
        /// 1-indexed retry counter within the current turn.
        attempt: u8,
        /// Cap that produced the truncated turn we're discarding.
        prev_cap: u32,
        /// Cap we're re-streaming with.
        new_cap: u32,
    },

    /// A `ContextTransform` plugin ran on this turn's transcript.
    /// Emitted once per active transform per turn, in registration
    /// order. Carries the full before/after message slices so observers
    /// can reconstruct exactly which messages each transform removed,
    /// added, or rewrote — the canonical answer to "which compaction
    /// stripped that tool result we expected the model to still see?".
    ContextTransformApplied {
        /// Zero-indexed turn within the current run. Same semantics as
        /// [`crate::plugin::TransformContext::iteration`].
        iteration: usize,
        /// `Plugin::name` of the transform that just ran.
        plugin: &'static str,
        /// Transcript handed to the transform.
        before: Vec<AgentMessage>,
        /// Transcript the transform returned.
        after: Vec<AgentMessage>,
    },

    /// A `ToolGate` plugin contributed to this turn's allowlist.
    /// Emitted once per gate per turn. Multiple gates compose by
    /// intersection downstream; this event records the gate's own
    /// decision before composition so observers can attribute the
    /// final allowlist to specific plugins.
    ToolGateApplied {
        /// Zero-indexed turn within the current run.
        iteration: usize,
        /// `Plugin::name` of the gate.
        plugin: &'static str,
        /// `None` when the gate declined to constrain;
        /// `Some(names)` when it returned an allowlist (sorted for
        /// stable diffing).
        allow: Option<Vec<String>>,
    },

    /// Multiple `ToolGate` plugins narrowed the same turn to disjoint
    /// non-empty allowlists. The loop repaired the composition to avoid
    /// advertising an empty tool catalog to the model.
    ToolGateConflictResolved {
        /// Zero-indexed turn within the current run.
        iteration: usize,
        /// Gate names that returned a non-empty allowlist.
        plugins: Vec<String>,
        /// Gate whose allowlist won the deterministic repair policy.
        chosen_plugin: Option<String>,
        /// Final repaired allowlist, sorted for stable diffing.
        allow: Vec<String>,
        /// Human-readable policy reason for trajectory/debug inspection.
        reason: String,
    },

    /// Snapshot of the request the loop is about to send to the
    /// provider on this turn, taken after every `ContextTransform`
    /// has run and every `ToolGate` has filtered. This is the typed
    /// view of "what the model sees" — wire-format conversion
    /// (provider-specific shapes) happens downstream inside the
    /// `StreamFn`. Emitted once per turn, just before the stream call.
    ProviderRequestPrepared {
        /// Zero-indexed turn within the current run.
        iteration: usize,
        /// Model identifier the host associated with this loop, when
        /// known. Provider transports still own their wire conversion,
        /// so this is observability metadata only.
        model_id: Option<String>,
        /// System prompt for this turn. May include ephemeral system
        /// reminders injected by `ContextTransform` plugins.
        system_prompt: String,
        /// Full message history the loop is about to send.
        messages: Vec<AgentMessage>,
        /// Tool schemas advertised this turn, post-`ToolGate` filtering.
        tools: Vec<ToolSchema>,
        /// Sampling temperature forwarded to the provider stream, when
        /// configured.
        temperature: Option<f32>,
        /// Resolved per-turn output cap.
        max_output_tokens: Option<u32>,
    },
}

/// Redacted, durable metadata for one provider request.
///
/// This deliberately excludes free-form prompt, message, image URL,
/// tool-description, and schema content. It keeps the dimensions needed
/// to debug "what shape did we send?" without leaking user text or
/// hidden/private reasoning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderRequestSummary {
    pub iteration: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    pub system_prompt_bytes: usize,
    pub system_prompt_chars: usize,
    pub message_count: usize,
    pub message_counts: ProviderMessageCounts,
    pub content_counts: ProviderContentCounts,
    pub tool_count: usize,
    pub tool_names: Vec<String>,
    pub tool_schema_bytes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_message_role: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderMessageCounts {
    pub system: usize,
    pub user: usize,
    pub assistant: usize,
    pub tool_result: usize,
    pub custom: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderContentCounts {
    pub system_message_bytes: usize,
    pub user_text_blocks: usize,
    pub user_text_bytes: usize,
    pub user_image_blocks: usize,
    pub user_image_with_media_type: usize,
    pub assistant_text_blocks: usize,
    pub assistant_text_bytes: usize,
    pub assistant_thinking_blocks: usize,
    pub assistant_thinking_bytes: usize,
    pub assistant_reasoning_blocks: usize,
    pub assistant_reasoning_bytes: usize,
    pub assistant_reasoning_detail_blocks: usize,
    pub assistant_reasoning_detail_bytes: usize,
    pub assistant_tool_call_blocks: usize,
    pub assistant_error_messages: usize,
    pub tool_result_text_blocks: usize,
    pub tool_result_text_bytes: usize,
    pub tool_result_image_blocks: usize,
    pub tool_result_error_messages: usize,
    pub custom_payload_bytes: usize,
}

impl ProviderRequestSummary {
    // Mirrors the provider-request fields directly; grouping would only hide
    // the shape this summary is meant to expose.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        iteration: usize,
        model_id: Option<&str>,
        temperature: Option<f32>,
        max_output_tokens: Option<u32>,
        system_prompt: &str,
        messages: &[AgentMessage],
        tools: &[ToolSchema],
    ) -> Self {
        let mut message_counts = ProviderMessageCounts::default();
        let mut content_counts = ProviderContentCounts::default();

        for message in messages {
            match message {
                AgentMessage::System { content, .. } => {
                    message_counts.system += 1;
                    content_counts.system_message_bytes += content.len();
                }
                AgentMessage::User { content, .. } => {
                    message_counts.user += 1;
                    count_user_content(content, &mut content_counts);
                }
                AgentMessage::Assistant {
                    content,
                    error_message,
                    ..
                } => {
                    message_counts.assistant += 1;
                    if error_message.is_some() {
                        content_counts.assistant_error_messages += 1;
                    }
                    count_assistant_content(content, &mut content_counts);
                }
                AgentMessage::ToolResult {
                    content, is_error, ..
                } => {
                    message_counts.tool_result += 1;
                    if *is_error {
                        content_counts.tool_result_error_messages += 1;
                    }
                    count_tool_result_content(content, &mut content_counts);
                }
                AgentMessage::Custom { payload, .. } => {
                    message_counts.custom += 1;
                    content_counts.custom_payload_bytes += json_size(payload);
                }
            }
        }

        let tool_names = tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        let tool_schema_bytes = tools.iter().map(tool_schema_size).sum();

        Self {
            iteration,
            model_id: model_id
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_string),
            temperature,
            max_output_tokens,
            system_prompt_bytes: system_prompt.len(),
            system_prompt_chars: system_prompt.chars().count(),
            message_count: messages.len(),
            message_counts,
            content_counts,
            tool_count: tools.len(),
            tool_names,
            tool_schema_bytes,
            last_message_role: messages.last().map(message_role).map(str::to_string),
        }
    }
}

fn count_user_content(content: &UserContent, counts: &mut ProviderContentCounts) {
    match content {
        UserContent::Text(text) => {
            counts.user_text_blocks += 1;
            counts.user_text_bytes += text.len();
        }
        UserContent::Blocks(blocks) => {
            for block in blocks {
                match block {
                    UserBlock::Text(text) => {
                        counts.user_text_blocks += 1;
                        counts.user_text_bytes += text.text.len();
                    }
                    UserBlock::Image(image) => {
                        counts.user_image_blocks += 1;
                        if image.media_type.is_some() {
                            counts.user_image_with_media_type += 1;
                        }
                    }
                }
            }
        }
    }
}

fn count_assistant_content(
    content: &crate::types::AssistantContent,
    counts: &mut ProviderContentCounts,
) {
    for block in &content.blocks {
        match block {
            AssistantBlock::Text(text) => {
                counts.assistant_text_blocks += 1;
                counts.assistant_text_bytes += text.text.len();
            }
            AssistantBlock::Thinking(text) => {
                counts.assistant_thinking_blocks += 1;
                counts.assistant_thinking_bytes += text.text.len();
            }
            AssistantBlock::Reasoning(text) => {
                counts.assistant_reasoning_blocks += 1;
                counts.assistant_reasoning_bytes += text.text.len();
            }
            AssistantBlock::ReasoningDetails(details) => {
                counts.assistant_reasoning_detail_blocks += 1;
                counts.assistant_reasoning_detail_bytes += json_size(&details.details);
            }
            AssistantBlock::ToolCall(_) => {
                counts.assistant_tool_call_blocks += 1;
            }
        }
    }
}

fn count_tool_result_content(
    content: &crate::types::ToolResultContent,
    counts: &mut ProviderContentCounts,
) {
    for block in &content.blocks {
        match block {
            ToolResultBlock::Text(text) => {
                counts.tool_result_text_blocks += 1;
                counts.tool_result_text_bytes += text.text.len();
            }
            ToolResultBlock::Image(_) => {
                counts.tool_result_image_blocks += 1;
            }
        }
    }
}

fn message_role(message: &AgentMessage) -> &'static str {
    match message {
        AgentMessage::System { .. } => "system",
        AgentMessage::User { .. } => "user",
        AgentMessage::Assistant { .. } => "assistant",
        AgentMessage::ToolResult { .. } => "tool_result",
        AgentMessage::Custom { .. } => "custom",
    }
}

fn tool_schema_size(tool: &ToolSchema) -> usize {
    tool.name.len()
        + tool.description.len()
        + serde_json::to_vec(&tool.parameters)
            .map(|bytes| bytes.len())
            .unwrap_or(0)
}

fn json_size(value: &impl Serialize) -> usize {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}

/// Sink the loop publishes events to.
///
/// Implementations buffer, log, forward, persist, etc. The loop awaits
/// `emit` so backpressure flows naturally. Failures inside the sink must
/// not propagate out of the loop — observers that fail are logged and
/// skipped.
#[async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: AgentEvent);
}

/// Trivial discard sink, useful when the caller only cares about the
/// final result of `run`.
pub struct NoopSink;

#[async_trait]
impl EventSink for NoopSink {
    async fn emit(&self, _event: AgentEvent) {}
}

/// Sink that forwards events into a `tokio::sync::mpsc::UnboundedSender`.
///
/// The loop is the producer; the consumer drains the channel and renders /
/// persists / forwards each event. Drops events silently when the receiver
/// is gone.
pub struct ChannelSink {
    tx: tokio::sync::mpsc::UnboundedSender<AgentEvent>,
}

impl ChannelSink {
    pub fn new() -> (Self, tokio::sync::mpsc::UnboundedReceiver<AgentEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Self { tx }, rx)
    }
}

#[async_trait]
impl EventSink for ChannelSink {
    async fn emit(&self, event: AgentEvent) {
        if self.tx.send(event).is_err() {
            // Receiver shutdown is an accepted best-effort sink outcome.
        }
    }
}

/// Composite sink that fans events out to multiple downstream sinks in
/// declaration order. Useful for mixing a logger, a UI forwarder, and a
/// persistence layer.
pub struct FanOutSink {
    sinks: Vec<Arc<dyn EventSink>>,
}

impl FanOutSink {
    pub fn new(sinks: Vec<Arc<dyn EventSink>>) -> Self {
        Self { sinks }
    }
}

#[async_trait]
impl EventSink for FanOutSink {
    async fn emit(&self, event: AgentEvent) {
        for sink in &self.sinks {
            sink.emit(event.clone()).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn channel_sink_forwards_events() {
        let (sink, mut rx) = ChannelSink::new();
        sink.emit(AgentEvent::AgentStart).await;
        sink.emit(AgentEvent::TurnStart).await;
        drop(sink);

        let mut received = Vec::new();
        while let Some(e) = rx.recv().await {
            received.push(e);
        }
        assert_eq!(received.len(), 2);
        assert!(matches!(received[0], AgentEvent::AgentStart));
        assert!(matches!(received[1], AgentEvent::TurnStart));
    }

    #[tokio::test]
    async fn fan_out_sink_replicates() {
        let (a, mut a_rx) = ChannelSink::new();
        let (b, mut b_rx) = ChannelSink::new();
        let fanout = FanOutSink::new(vec![Arc::new(a), Arc::new(b)]);
        fanout.emit(AgentEvent::AgentStart).await;
        drop(fanout);

        assert!(matches!(a_rx.recv().await, Some(AgentEvent::AgentStart)));
        assert!(matches!(b_rx.recv().await, Some(AgentEvent::AgentStart)));
    }

    #[test]
    fn provider_request_summary_counts_shape_without_text() {
        let messages = vec![
            AgentMessage::User {
                content: UserContent::Blocks(vec![
                    UserBlock::Text(crate::types::TextContent {
                        text: "secret user request".into(),
                    }),
                    UserBlock::Image(crate::types::ImageContent {
                        source: "data:image/png;base64,secret".into(),
                        media_type: Some("image/png".into()),
                        alt: Some("screenshot".into()),
                    }),
                ]),
                timestamp: None,
            },
            AgentMessage::Assistant {
                content: crate::types::AssistantContent {
                    blocks: vec![
                        AssistantBlock::Thinking(crate::types::TextContent {
                            text: "private scratch".into(),
                        }),
                        AssistantBlock::ToolCall(crate::tool::ToolCall {
                            id: "call-1".into(),
                            name: "web_search".into(),
                            arguments: serde_json::json!({"q": "secret"}),
                        }),
                    ],
                },
                stop_reason: crate::types::StopReason::ToolUse,
                error_message: None,
                timestamp: None,
                usage: None,
            },
            AgentMessage::ToolResult {
                tool_call_id: "call-1".into(),
                tool_name: "web_search".into(),
                content: crate::types::ToolResultContent::text("secret result"),
                is_error: false,
                narration: None,
                details: None,
                timestamp: None,
            },
        ];
        let tools = vec![ToolSchema {
            name: "web_search".into(),
            description: "Search the web".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }];

        let summary = ProviderRequestSummary::from_parts(
            2,
            Some("google/gemini-3.1-flash-lite-preview"),
            Some(0.2),
            Some(4096),
            "system prompt secret",
            &messages,
            &tools,
        );

        assert_eq!(summary.iteration, 2);
        assert_eq!(
            summary.model_id.as_deref(),
            Some("google/gemini-3.1-flash-lite-preview")
        );
        assert_eq!(summary.message_counts.user, 1);
        assert_eq!(summary.message_counts.assistant, 1);
        assert_eq!(summary.message_counts.tool_result, 1);
        assert_eq!(summary.content_counts.user_text_blocks, 1);
        assert_eq!(
            summary.content_counts.user_text_bytes,
            "secret user request".len()
        );
        assert_eq!(summary.content_counts.user_image_blocks, 1);
        assert_eq!(summary.content_counts.assistant_thinking_blocks, 1);
        assert_eq!(summary.content_counts.assistant_tool_call_blocks, 1);
        assert_eq!(summary.content_counts.tool_result_text_blocks, 1);
        assert_eq!(summary.tool_names, vec!["web_search"]);
        assert_eq!(summary.last_message_role.as_deref(), Some("tool_result"));

        let serialized = serde_json::to_string(&summary).unwrap();
        assert!(!serialized.contains("secret user request"));
        assert!(!serialized.contains("private scratch"));
        assert!(!serialized.contains("secret result"));
        assert!(!serialized.contains("data:image"));
    }
}
