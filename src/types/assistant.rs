//! Generated assistant content and private request provenance.
use super::{AgentMessage, TextContent};
use crate::tool::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Assistant-message content. Carries text, hidden reasoning blocks, and
/// tool call requests in source order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssistantContent {
    pub blocks: Vec<AssistantBlock>,
}

impl AssistantContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![AssistantBlock::Text(TextContent { text: text.into() })],
        }
    }

    pub fn with_tool_calls(text: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        let mut blocks = Vec::new();
        if let Some(t) = text.filter(|s| !s.trim().is_empty()) {
            blocks.push(AssistantBlock::Text(TextContent { text: t }));
        }
        for call in tool_calls {
            blocks.push(AssistantBlock::ToolCall(call));
        }
        Self { blocks }
    }

    /// Concatenate all text blocks into a single string.
    pub fn plain_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Return all tool call blocks in source order.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    pub fn thinking_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Thinking(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn reasoning_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Reasoning(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn reasoning_details_values(&self) -> Vec<Value> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::ReasoningDetails(d) => Some(d.details.as_slice()),
                _ => None,
            })
            .flatten()
            .cloned()
            .collect()
    }
}

/// Blocks an assistant message can carry.
///
/// ## Channel separation contract
///
/// `Thinking` and `Reasoning` are two **independent** channels and the
/// loop must never mix them:
///
/// - [`Thinking`](AssistantBlock::Thinking) is **prompt-elicited
///   tag-text**. The model wraps reasoning inside
///   `<thought>...</thought>` (or one of the synonym tags handled by
///   [`crate::ThinkingTagStreamFilter`]) inside its visible text
///   stream. The bridge parses those tags out of the visible-text
///   channel and stores the captured content here. On the next
///   provider request it is **rewoven into the `content` field as a
///   `<thought>...</thought>` tag** — never as the wire `reasoning`
///   field.
///
/// - [`Reasoning`](AssistantBlock::Reasoning) and
///   [`ReasoningDetails`](AssistantBlock::ReasoningDetails) are
///   **provider-native reasoning**. They arrive on a dedicated
///   sideband (`delta.reasoning` / `delta.reasoning_details` on the
///   OpenRouter wire) and represent the provider's own
///   chain-of-thought tokens. On the next provider request they are
///   replayed verbatim through the typed `reasoning` /
///   `reasoning_details` fields — never wrapped in a
///   `<thought>...</thought>` tag in the `content` field.
///
/// The two processes are not interchangeable: tag-elicited scratch is
/// the model writing into its visible output by convention, and the
/// loop strips it before the user sees anything. Provider-native
/// reasoning is the upstream API delivering structured thinking
/// alongside the message. Conflating them risks (a) shipping the
/// model's hidden scratch as if it were native reasoning (some
/// providers reject unknown content there or refuse to bill it as
/// cached input) and (b) leaking native reasoning into visible text
/// by way of `<thought>` rewrap (would round-trip native tokens
/// through the visible channel and double-count them).
///
/// The invariants are pinned by tests in
/// `openrouter_request_tests.rs::channel_separation_invariants` and
/// `openrouter_stream::tests::stream_chunk_routing_invariants`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantBlock {
    Text(TextContent),
    /// Host observations supplied on the request that produced this response.
    /// Private input history, never generated prose or native reasoning.
    ContextObservations(ContextObservationsContent),
    /// Prompt-elicited hidden scratchpad. Captured from
    /// `<thought>...</thought>` tags the model writes inside its
    /// visible text stream; rewoven into the wire `content` field as
    /// `<thought>...</thought>` on the next request. **Never** flows
    /// into the wire `reasoning` field — see the type-level docs for
    /// the channel-separation contract.
    Thinking(TextContent),
    /// Provider-native reasoning (xAI Grok, OpenAI o-series, Anthropic
    /// native thinking). Arrives on the dedicated `delta.reasoning`
    /// sideband and replayed via the wire `reasoning` field. **Never**
    /// wrapped in a `<thought>...</thought>` tag inside the `content`
    /// field — see the type-level docs.
    Reasoning(TextContent),
    /// Native provider reasoning detail blocks (xAI's
    /// `reasoning.encrypted` envelopes, etc.). Replayed unmodified on
    /// tool-continuation requests for reasoning models that rely on
    /// signed/encrypted thinking continuity. Same channel contract as
    /// [`Reasoning`](AssistantBlock::Reasoning).
    ReasoningDetails(ReasoningDetailsContent),
    /// Tool call request. Loop dispatches via the registry.
    ToolCall(ToolCall),
}

/// Exact host-owned request observations retained with their response.
/// Applications seal provenance before returning these from a context plugin.
/// The canonical projection expands them before the associated assistant turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextObservationsContent {
    pub messages: Vec<ContextObservation>,
}

/// A host-owned observation. The open payload belongs to its kind's typed
/// producer; this cannot contain nested assistant turns or tool transactions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextObservation {
    pub kind: String,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

impl ContextObservation {
    pub fn into_message(self) -> AgentMessage {
        AgentMessage::Custom { kind: self.kind, payload: self.payload, timestamp: self.timestamp }
    }
}

/// Persistent envelope for provider-native reasoning items on an
/// assistant turn. The wire shape is `Vec<Value>` matching
/// OpenRouter's `reasoning_details[]` schema (the broadest typed
/// surface across providers); `as_items` lifts it to typed
/// [`crate::reasoning::ReasoningItem`]s for codec operations and `from_items` projects
/// typed items back. The `details` field stays the source of truth so
/// persisted trajectories round-trip byte-exact, even when a future
/// provider sends shapes the typed enum doesn't yet recognize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningDetailsContent {
    pub details: Vec<Value>,
}

impl ReasoningDetailsContent {
    pub fn new(details: Vec<Value>) -> Self {
        Self { details }
    }

    /// Lift the stored `details` array into typed
    /// [`crate::reasoning::ReasoningItem`]s. Items that don't match a known variant
    /// are preserved in `details` but elided from the typed view —
    /// so consumers iterating typed items never see corrupt data
    /// while replay-via-`details` still ships the original bytes.
    pub fn as_items(&self) -> Vec<crate::reasoning::ReasoningItem> {
        self.details
            .iter()
            .filter_map(crate::reasoning::ReasoningItem::from_openrouter_value)
            .collect()
    }

    /// Build from typed items. Used when a codec produces typed
    /// items from a non-OpenRouter provider response.
    pub fn from_items(items: &[crate::reasoning::ReasoningItem]) -> Self {
        Self {
            details: items
                .iter()
                .map(crate::reasoning::ReasoningItem::to_openrouter_value)
                .collect(),
        }
    }

    /// True iff any item carries a signed/encrypted payload that a
    /// strict-replay provider would reject if missing on next turn.
    pub fn has_signed_payload(&self) -> bool {
        self.as_items()
            .iter()
            .any(crate::reasoning::ReasoningItem::carries_signed_payload)
    }
}

