//! LLM transport extension point.
//!
//! `StreamFn` is the swappable boundary between the loop and any
//! provider: real LLM API call (production), recorded-fixture replay
//! (eval), scripted scenario (tests), remote proxy (gateway). One trait
//! method, typed request/response.
//!
//! The loop never imports a specific provider; the caller assembles a
//! `LoopConfig` with a `StreamFn` implementation of their choice.

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::tool::ToolCall;
use crate::types::{AgentMessage, AssistantContent, StopReason};

/// Inputs for one LLM call.
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// System prompt prepended by the provider.
    pub system_prompt: String,
    /// Conversation transcript. The transport converts to provider-native
    /// format before sending.
    pub messages: Vec<AgentMessage>,
    /// Tool schemas the provider should expose to the model.
    pub tools: Vec<ToolSchema>,
    /// Optional sampling controls. Implementations may ignore.
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    /// Reasoning-effort knob shipped to the provider. Typed contract;
    /// each `StreamFn` impl owns the wire mapping (OpenRouter's
    /// `reasoning: {effort}` block, Fireworks' top-level
    /// `reasoning_effort`, etc.). Single source of truth — replaces the
    /// stringly-typed `provider_config["reasoning_effort"]` knob and the
    /// hardcoded per-bridge defaults.
    pub reasoning: ReasoningEffort,
    /// Provider-specific extras (e.g., response format, custom routing
    /// pins). Open-by-design leaf — typed at the leaf is overkill given
    /// provider fragmentation. Reasoning effort is NOT carried here; use
    /// the typed `reasoning` field above.
    #[allow(clippy::struct_field_names)]
    pub provider_extras: Value,
    /// When `true`, the wire request sets `tool_choice: "required"` —
    /// the provider MUST emit a tool call (no plain-text or empty
    /// completion). Coercive narrowing for the framing turn: paired
    /// with `OpeningGate`'s compact catalog, this forces the model to
    /// pick a message or planning tool instead of emitting
    /// reasoning-only text and skipping the structured turn.
    /// Default `false` leaves tool choice to the provider for every
    /// other turn.
    pub force_tool_call: bool,
}

/// Reasoning-effort levels accepted by every supported provider.
///
/// Wire mapping is owned by each `StreamFn` impl: OpenRouter sends
/// `reasoning: {effort}`; Fireworks rejects `"minimal"` and gets
/// `Minimal → "low"` remapped at the bridge. The enum stays
/// provider-agnostic so the loop and the configuration surface speak
/// one language.
///
/// Default is [`ReasoningEffort::Minimal`] — keeps the reasoning
/// channel open (some providers reject requests that omit it on
/// reasoning-capable models) without burning the visible-completion
/// budget on hidden-thought tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Reasoning channel suppressed entirely. Use only when the model
    /// is known to tolerate `effort: "none"` (Anthropic, xAI Grok,
    /// most non-OpenAI families).
    None,
    /// Lowest non-empty reasoning budget. Universal default — works
    /// on every provider and keeps tool-call latency low.
    #[default]
    Minimal,
    Low,
    Medium,
    High,
    /// OpenRouter-specific extra-high reasoning tier (~95% of max_tokens
    /// on OpenAI/Grok/Anthropic; mapped down to `"high"` for Gemini 3).
    XHigh,
}

impl ReasoningEffort {
    /// Wire form sent to OpenRouter / OpenAI-compatible endpoints.
    pub fn as_wire(self) -> &'static str {
        match self {
            ReasoningEffort::None => "none",
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
        }
    }

    /// Parse the canonical wire string back into the enum. Unknown or
    /// empty inputs return `None` so callers can decide whether to
    /// fall back to the default or surface a configuration error.
    pub fn from_wire(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(ReasoningEffort::None),
            "minimal" => Some(ReasoningEffort::Minimal),
            "low" => Some(ReasoningEffort::Low),
            "medium" => Some(ReasoningEffort::Medium),
            "high" => Some(ReasoningEffort::High),
            "xhigh" => Some(ReasoningEffort::XHigh),
            _ => None,
        }
    }
}

/// Tool schema as the provider sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Streamed event from the provider.
///
/// Streaming surface is rich because UIs need to render token-by-token.
/// The loop folds these into a single `StreamResponse` at the end.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// First event. Provider has accepted the request and started
    /// generating. Carries the (so-far-empty) partial message so listeners
    /// can render an empty assistant placeholder.
    Start { partial: AgentMessage },

    /// A streaming chunk. UI listeners render this; the loop folds it
    /// into the assembled message.
    Chunk(AssistantStreamChunk),

    /// Final event. Provider finished generating. Carries the assembled
    /// final message.
    Done { message: AgentMessage },

    /// Final event. Provider raised an error during streaming. Carries
    /// the partial message produced so far and a human-readable error
    /// description. Stream implementations encode their typed errors
    /// here as strings; callers that need structured detail can parse
    /// `kind`.
    Error {
        partial: AgentMessage,
        kind: StreamErrorKind,
        message: String,
    },
}

/// One token-level chunk during streaming.
///
/// Apps render these as deltas. The loop ignores them (it only looks at
/// `Start` for the placeholder and `Done` / `Error` for the assembled
/// final message).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssistantStreamChunk {
    /// Text being appended to the visible assistant content.
    Text { delta: String },
    /// Hidden reasoning being appended (think-then-act block).
    Thinking { delta: String },
    /// Native provider reasoning being appended.
    Reasoning { delta: String },
    /// Native provider reasoning detail blocks being appended.
    ReasoningDetails { delta: Vec<Value> },
    /// Tool call accumulating: arguments JSON streaming in piece by piece.
    ToolCallDelta {
        index: usize,
        id_delta: Option<String>,
        name_delta: Option<String>,
        arguments_delta: Option<String>,
    },
}

/// Coarse classification of a stream error.
///
/// `ContextOverflow` is split out from `Fatal` so the loop can apply
/// recovery (re-run with a smaller context) instead of terminating.
/// Without this split, providers that surface a `prompt_too_long` /
/// `context_length_exceeded` error look identical to permanent
/// failures (auth, schema, model id) and the loop has no signal to
/// distinguish "trim and retry" from "give up".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamErrorKind {
    Transient,
    /// The selected model/provider is temporarily rate-limited. This
    /// remains retryable at the transport boundary, but surfaces as a
    /// distinct terminal kind if every retry fails.
    ProviderRateLimited,
    /// Transport failed before the provider produced any actionable
    /// assistant turn. Hidden reasoning/details, usage, or an unusable
    /// burst of partial tool-call deltas may have arrived, but Clark
    /// still has no runnable next step, so this is safer to replay than
    /// a generic transient stream error.
    ZeroOutputTransport,
    Fatal,
    Empty,
    Aborted,
    /// Provider rejected the request because the assembled context
    /// exceeds the model's window. Distinct from `Fatal` so a future
    /// recovery layer (Phase 2 in the compaction roadmap) can compact
    /// more aggressively and retry rather than ending the run.
    ContextOverflow,
}

/// Final assembled response from one LLM call. Producers that don't need
/// fine-grained streaming can return a `BoxStream` that yields just one
/// `Done` event.
#[derive(Debug, Clone)]
pub struct StreamResponse {
    pub content: AssistantContent,
    pub stop_reason: StopReason,
    pub error_message: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// The transport trait. One method, typed in/out.
///
/// Contract:
/// - Must not panic on request/model/runtime failures. Encode failures in
///   the returned stream as `StreamEvent::Error` with a partial message,
///   not by returning `Err`.
/// - The returned stream must yield exactly one terminal event
///   (`Done` or `Error`). Apps that prefer non-streaming can yield a
///   single `Start` + `Done` with the full final message.
/// - Honor `signal` for cancellation. On cancel, yield `Error` with
///   `StreamError::Aborted`-equivalent or simply drop the stream.
#[async_trait]
pub trait StreamFn: Send + Sync + 'static {
    async fn stream(
        &self,
        request: StreamRequest,
        signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent>;
}
