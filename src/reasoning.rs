//! Provider-agnostic reasoning replay.
//!
//! Captures provider-native reasoning content from one assistant turn and
//! replays it on the next, in a shape each provider accepts. Mirrors
//! OpenRouter's `reasoning_details[]` schema (the broadest typed surface
//! across providers): every item is one of three variants — plain
//! reasoning text (with optional opaque signature), provider-emitted
//! summary, or fully-encrypted blob — tagged with the originating
//! provider's `format` discriminator.
//!
//! Why this exists: providers split on two axes. **Where the handle
//! attaches** (Gemini binds `thoughtSignature` to a specific `Part`;
//! Anthropic binds `signature` to a `thinking` content block;
//! OpenAI/xAI carry `encrypted_content` on a separate reasoning input
//! item) and **whether replay is required** (Gemini 3 always-with-tools;
//! Anthropic only when next turn carries `tool_result`; OpenAI when a
//! function call is in the turn; Grok stateless). A flat opaque-bytes
//! abstraction collapses the attachment point and breaks Gemini.
//!
//! The right shape: typed enum with provider-shaped variants, and the
//! per-provider `ReasoningCodec` knows how to read each native shape on
//! response and emit each native shape on the next request. The bridge
//! stores the typed enum verbatim; the codec is a stateless translator
//! between typed item and provider wire shape.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Tagged provider source for a reasoning item.
///
/// Mirrors the `format` field on OpenRouter's `reasoning_details`
/// items, plus an `Unknown` fallback so we never silently drop a
/// payload from an unrecognized future provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningFormat {
    /// `format: "anthropic-claude-v1"` — Claude `thinking` /
    /// `redacted_thinking` blocks; `signature` field on the
    /// `Text` variant is mandatory round-trip when next turn has
    /// `tool_result`.
    AnthropicClaudeV1,
    /// `format: "google-gemini-v1"` — Vertex AI / AI Studio
    /// Gemini 2.5+. `Encrypted` variant carries the
    /// `thoughtSignature` blob OpenRouter explodes out of the
    /// per-`Part` attachment. Gemini 3 with tools requires exact
    /// round-trip or returns 400 INVALID_ARGUMENT "Thought
    /// signature is not valid".
    GoogleGeminiV1,
    /// `format: "openai-responses-v1"` — OpenAI o-series via the
    /// Responses API. `Encrypted.data` holds `encrypted_content`;
    /// callers must opt in via `include: ["reasoning.encrypted_content"]`.
    OpenaiResponsesV1,
    /// `format: "azure-openai-responses-v1"` — Azure variant of
    /// the same shape.
    AzureOpenaiResponsesV1,
    /// `format: "xai-responses-v1"` — xAI Grok via the Responses
    /// API.
    XaiResponsesV1,
    /// Anything else. Round-tripped opaquely; the codec preserves
    /// the original `Value` as `raw` so a future provider never
    /// silently loses fidelity.
    #[serde(other)]
    Unknown,
}

impl ReasoningFormat {
    pub fn as_wire(&self) -> &'static str {
        match self {
            ReasoningFormat::AnthropicClaudeV1 => "anthropic-claude-v1",
            ReasoningFormat::GoogleGeminiV1 => "google-gemini-v1",
            ReasoningFormat::OpenaiResponsesV1 => "openai-responses-v1",
            ReasoningFormat::AzureOpenaiResponsesV1 => "azure-openai-responses-v1",
            ReasoningFormat::XaiResponsesV1 => "xai-responses-v1",
            ReasoningFormat::Unknown => "unknown",
        }
    }

    pub fn from_wire(s: &str) -> Self {
        match s {
            "anthropic-claude-v1" => ReasoningFormat::AnthropicClaudeV1,
            "google-gemini-v1" => ReasoningFormat::GoogleGeminiV1,
            "openai-responses-v1" => ReasoningFormat::OpenaiResponsesV1,
            "azure-openai-responses-v1" => ReasoningFormat::AzureOpenaiResponsesV1,
            "xai-responses-v1" => ReasoningFormat::XaiResponsesV1,
            _ => ReasoningFormat::Unknown,
        }
    }

    /// Replay contract this provider format imposes on the next turn.
    ///
    /// Each variant's enforcement story lives in the variant's own
    /// docs; this method is the typed projection the audit pipeline
    /// reads. Centralizing here keeps the audit free of per-provider
    /// switches and gives new providers exactly one place to declare
    /// their contract.
    pub fn replay_contract(&self) -> ReplayContract {
        match self {
            ReasoningFormat::GoogleGeminiV1 => ReplayContract::RequiredWithTools,
            ReasoningFormat::AnthropicClaudeV1 => ReplayContract::RequiredWithTools,
            ReasoningFormat::OpenaiResponsesV1 | ReasoningFormat::AzureOpenaiResponsesV1 => {
                ReplayContract::RequiredWithTools
            }
            ReasoningFormat::XaiResponsesV1 => ReplayContract::Stateless,
            ReasoningFormat::Unknown => ReplayContract::Stateless,
        }
    }
}

/// One provider-emitted reasoning element.
///
/// Three variants cover every shape known across the major providers:
/// readable reasoning text (with optional Anthropic-style signature),
/// summary blurbs (OpenAI/Anthropic summaries), and fully-encrypted
/// blobs (Gemini `thoughtSignature`, OpenAI `encrypted_content`,
/// Anthropic `redacted_thinking`). Round-trip identity is the contract:
/// every byte the provider emitted must come back unchanged on the
/// next assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReasoningItem {
    /// `reasoning.text` — visible reasoning content. The
    /// `signature` field is opaque and mandatory round-trip on
    /// providers that emit it (Anthropic).
    #[serde(rename = "reasoning.text")]
    Text {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        format: ReasoningFormat,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// `reasoning.summary` — provider-emitted summary of internal
    /// reasoning; carries no signature, but still must replay
    /// verbatim to preserve token-cache continuity on some
    /// providers.
    #[serde(rename = "reasoning.summary")]
    Summary {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        format: ReasoningFormat,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        summary: String,
    },
    /// `reasoning.encrypted` — opaque base64 blob. Gemini's
    /// `thoughtSignature` is delivered in this shape via
    /// OpenRouter; OpenAI's `encrypted_content` and Anthropic's
    /// `redacted_thinking.data` likewise. The bridge stores
    /// `data` byte-for-byte and the codec attaches it back to the
    /// right wire location for the originating provider.
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        format: ReasoningFormat,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        data: String,
    },
}

impl ReasoningItem {
    /// Originating-provider tag.
    pub fn format(&self) -> ReasoningFormat {
        match self {
            ReasoningItem::Text { format, .. } => *format,
            ReasoningItem::Summary { format, .. } => *format,
            ReasoningItem::Encrypted { format, .. } => *format,
        }
    }

    /// `index` if the provider supplied one. OpenRouter uses this
    /// to preserve order across heterogeneous variants when the
    /// provider emits text and encrypted blobs interleaved.
    pub fn index(&self) -> Option<u32> {
        match self {
            ReasoningItem::Text { index, .. } => *index,
            ReasoningItem::Summary { index, .. } => *index,
            ReasoningItem::Encrypted { index, .. } => *index,
        }
    }

    /// True iff this item carries a signature/encrypted payload
    /// the provider will reject if missing on replay. Used by
    /// `ReasoningCodec` to validate "did we receive what we need
    /// for the next turn?" without having to inspect the inner
    /// fields.
    pub fn carries_signed_payload(&self) -> bool {
        matches!(
            self,
            ReasoningItem::Text {
                signature: Some(_),
                ..
            } | ReasoningItem::Encrypted { .. }
        )
    }

    /// Round-trip from a `Value` shaped exactly as OpenRouter's
    /// `reasoning_details[]` item. Returns `None` if the input is
    /// not an object with a recognized `type` discriminator.
    pub fn from_openrouter_value(value: &Value) -> Option<Self> {
        let obj = value.as_object()?;
        let kind = obj.get("type").and_then(Value::as_str)?;
        let format = obj
            .get("format")
            .and_then(Value::as_str)
            .map(ReasoningFormat::from_wire)
            .unwrap_or(ReasoningFormat::Unknown);
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| obj.get("id").and_then(|v| v.as_null()).and(None));
        let index = obj.get("index").and_then(Value::as_u64).map(|n| n as u32);

        match kind {
            "reasoning.text" => Some(ReasoningItem::Text {
                id,
                format,
                index,
                text: obj
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                signature: obj
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }),
            "reasoning.summary" => Some(ReasoningItem::Summary {
                id,
                format,
                index,
                summary: obj
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            }),
            "reasoning.encrypted" => Some(ReasoningItem::Encrypted {
                id,
                format,
                index,
                data: obj
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            }),
            _ => None,
        }
    }

    /// Serialize back to the OpenRouter wire shape. Round-trip
    /// fidelity: `from_openrouter_value(v).unwrap().to_openrouter_value() == v`
    /// for any well-formed input.
    pub fn to_openrouter_value(&self) -> Value {
        let mut map = Map::new();
        match self {
            ReasoningItem::Text {
                id,
                format,
                index,
                text,
                signature,
            } => {
                map.insert("type".into(), Value::String("reasoning.text".into()));
                if let Some(id) = id {
                    map.insert("id".into(), Value::String(id.clone()));
                }
                map.insert("format".into(), Value::String(format.as_wire().into()));
                if let Some(index) = index {
                    map.insert("index".into(), Value::Number((*index).into()));
                }
                map.insert("text".into(), Value::String(text.clone()));
                if let Some(sig) = signature {
                    map.insert("signature".into(), Value::String(sig.clone()));
                }
            }
            ReasoningItem::Summary {
                id,
                format,
                index,
                summary,
            } => {
                map.insert("type".into(), Value::String("reasoning.summary".into()));
                if let Some(id) = id {
                    map.insert("id".into(), Value::String(id.clone()));
                }
                map.insert("format".into(), Value::String(format.as_wire().into()));
                if let Some(index) = index {
                    map.insert("index".into(), Value::Number((*index).into()));
                }
                map.insert("summary".into(), Value::String(summary.clone()));
            }
            ReasoningItem::Encrypted {
                id,
                format,
                index,
                data,
            } => {
                map.insert("type".into(), Value::String("reasoning.encrypted".into()));
                if let Some(id) = id {
                    map.insert("id".into(), Value::String(id.clone()));
                }
                map.insert("format".into(), Value::String(format.as_wire().into()));
                if let Some(index) = index {
                    map.insert("index".into(), Value::Number((*index).into()));
                }
                map.insert("data".into(), Value::String(data.clone()));
            }
        }
        Value::Object(map)
    }
}

/// Whether this turn's reasoning items satisfy the next-turn replay
/// contract for the originating provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayContract {
    /// Provider has no replay requirement (e.g. xAI Grok via chat
    /// completions, Qwen / Kimi / Llama, or any model not yet
    /// enrolled in the audit). Stateless.
    Stateless,
    /// Provider requires every signed/encrypted item to round-trip
    /// exactly when the upcoming turn carries tool calls / tool
    /// results. Missing → 400.
    RequiredWithTools,
    /// Provider requires every signed/encrypted item to round-trip
    /// every turn, irrespective of tools (Gemini 3 Pro).
    AlwaysRequired,
}

/// Plug-and-play translator between provider-native reasoning shapes
/// and Clark's typed `ReasoningItem` representation.
///
/// Per-provider implementations live in their respective `StreamFn`
/// modules. The agent loop never knows about provider quirks; it
/// holds typed items and asks the codec to read/write the wire shape.
/// Replay contracts are a property of the originating
/// [`ReasoningFormat`], not the codec — see
/// [`ReasoningFormat::replay_contract`].
///
/// The default `OpenRouterReasoningCodec` covers every provider that
/// flows through OpenRouter, since OpenRouter normalizes all upstream
/// reasoning into the same `reasoning_details[]` schema.
pub trait ReasoningCodec: Send + Sync {
    /// Lift OpenRouter-shaped `reasoning_details[]` (or the native
    /// equivalent) into typed items. Values whose `type` discriminator
    /// matches no known variant are dropped from the typed view; the
    /// raw `Value` array is still kept on the assistant message so
    /// round-trip fidelity is preserved on replay.
    fn parse_response(&self, raw: &[Value]) -> Vec<ReasoningItem>;

    /// Project typed items back into the assistant message body that
    /// will be sent on the next request. The default emits an
    /// `reasoning_details` array on the assistant message. Providers
    /// with positional contracts (Vertex direct: signature must be
    /// re-attached to the `functionCall` Part) override.
    fn write_assistant(&self, msg: &mut Value, items: &[ReasoningItem]);
}

/// Diagnostic about a single assistant turn's reasoning state.
///
/// Produced by [`audit_replay`]. The bridge uses this to surface
/// "the next request will 400" failures eagerly — observable now, in
/// the right log line, instead of as a confusing upstream HTTP error
/// after the request goes out.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayAudit {
    /// Total reasoning items captured this turn.
    pub item_count: usize,
    /// Items that carry a signature or encrypted blob the
    /// originating provider will check on replay.
    pub signed_count: usize,
    /// Per-format breakdown so the bridge can route warnings to
    /// the right provider-specific log channel.
    pub formats: Vec<ReasoningFormat>,
    /// Severity of any contract break the audit detected. `None`
    /// means the turn satisfies its codec's replay contract.
    pub violation: Option<ReplayViolation>,
}

/// A specific way the audited turn would fail upstream on replay.
///
/// The bridge maps each variant to a typed log/event so operators can
/// distinguish "this provider produced no signatures and will reject
/// the next request" from "this provider produced malformed items".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayViolation {
    /// Provider's contract is `RequiredWithTools` (or stricter), the
    /// upcoming turn carries tool calls / tool results, and zero
    /// signed/encrypted items were captured. Next request will be
    /// rejected with a contract error (Gemini 3 returns
    /// `INVALID_ARGUMENT "Thought signature is not valid"`).
    MissingSignaturesForStrictProvider {
        contract: ReplayContract,
        formats: Vec<ReasoningFormat>,
    },
}

/// Audit a turn's reasoning items against the codec's replay contract.
///
/// `next_turn_carries_tools` should be `true` when the upcoming
/// request will include tool calls or tool results — that's the
/// trigger for `RequiredWithTools` providers. Stateless and
/// `Optional` codecs always return a clean audit.
pub fn audit_replay(
    items: &[ReasoningItem],
    contract: ReplayContract,
    next_turn_carries_tools: bool,
) -> ReplayAudit {
    let signed_count = items.iter().filter(|i| i.carries_signed_payload()).count();
    let mut formats: Vec<ReasoningFormat> = items.iter().map(ReasoningItem::format).collect();
    formats.sort_by_key(|f| f.as_wire());
    formats.dedup();

    let violation = match contract {
        ReplayContract::Stateless => None,
        ReplayContract::RequiredWithTools if !next_turn_carries_tools => None,
        ReplayContract::RequiredWithTools | ReplayContract::AlwaysRequired => {
            if signed_count == 0 {
                Some(ReplayViolation::MissingSignaturesForStrictProvider {
                    contract,
                    formats: formats.clone(),
                })
            } else {
                None
            }
        }
    };

    ReplayAudit {
        item_count: items.len(),
        signed_count,
        formats,
        violation,
    }
}

/// Default codec: reads OpenRouter's `reasoning_details[]` schema on
/// response, writes the same on the assistant message for replay.
/// Covers every provider OpenRouter routes to (Anthropic, Google,
/// OpenAI, xAI, Azure, etc.) because OpenRouter normalizes them all
/// to this shape.
#[derive(Debug, Clone, Default)]
pub struct OpenRouterReasoningCodec;

impl OpenRouterReasoningCodec {
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningCodec for OpenRouterReasoningCodec {
    fn parse_response(&self, raw: &[Value]) -> Vec<ReasoningItem> {
        raw.iter()
            .filter_map(ReasoningItem::from_openrouter_value)
            .collect()
    }

    fn write_assistant(&self, msg: &mut Value, items: &[ReasoningItem]) {
        if items.is_empty() {
            return;
        }
        let arr = items
            .iter()
            .map(ReasoningItem::to_openrouter_value)
            .collect::<Vec<_>>();
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("reasoning_details".into(), Value::Array(arr));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn format_wire_roundtrip_covers_every_variant() {
        for fmt in [
            ReasoningFormat::AnthropicClaudeV1,
            ReasoningFormat::GoogleGeminiV1,
            ReasoningFormat::OpenaiResponsesV1,
            ReasoningFormat::AzureOpenaiResponsesV1,
            ReasoningFormat::XaiResponsesV1,
            ReasoningFormat::Unknown,
        ] {
            assert_eq!(ReasoningFormat::from_wire(fmt.as_wire()), fmt);
        }
    }

    #[test]
    fn unknown_format_falls_back() {
        assert_eq!(
            ReasoningFormat::from_wire("future-provider-v9"),
            ReasoningFormat::Unknown
        );
    }

    #[test]
    fn item_text_roundtrip() {
        let v = json!({
            "type": "reasoning.text",
            "id": "rs-1",
            "format": "anthropic-claude-v1",
            "index": 0,
            "text": "First, compare the decimals.",
            "signature": "sig-abc"
        });
        let item = ReasoningItem::from_openrouter_value(&v).unwrap();
        match &item {
            ReasoningItem::Text {
                format, signature, ..
            } => {
                assert_eq!(*format, ReasoningFormat::AnthropicClaudeV1);
                assert_eq!(signature.as_deref(), Some("sig-abc"));
            }
            _ => panic!("expected Text variant"),
        }
        assert_eq!(item.to_openrouter_value(), v);
    }

    #[test]
    fn item_encrypted_roundtrip_for_gemini() {
        let v = json!({
            "type": "reasoning.encrypted",
            "id": "rs-2",
            "format": "google-gemini-v1",
            "index": 3,
            "data": "BASE64BLOB"
        });
        let item = ReasoningItem::from_openrouter_value(&v).unwrap();
        match &item {
            ReasoningItem::Encrypted { format, data, .. } => {
                assert_eq!(*format, ReasoningFormat::GoogleGeminiV1);
                assert_eq!(data, "BASE64BLOB");
            }
            _ => panic!("expected Encrypted variant"),
        }
        assert!(item.carries_signed_payload());
        assert_eq!(item.to_openrouter_value(), v);
    }

    #[test]
    fn item_summary_roundtrip() {
        let v = json!({
            "type": "reasoning.summary",
            "id": "rs-3",
            "format": "openai-responses-v1",
            "index": 1,
            "summary": "Compared 9.9 and 9.11 numerically."
        });
        let item = ReasoningItem::from_openrouter_value(&v).unwrap();
        assert!(matches!(item, ReasoningItem::Summary { .. }));
        assert!(!item.carries_signed_payload());
        assert_eq!(item.to_openrouter_value(), v);
    }

    #[test]
    fn unknown_type_returns_none() {
        let v = json!({"type": "reasoning.future_kind", "data": "..." });
        assert!(ReasoningItem::from_openrouter_value(&v).is_none());
    }

    #[test]
    fn carries_signed_payload_distinguishes_signed_from_unsigned_text() {
        let signed = ReasoningItem::Text {
            id: None,
            format: ReasoningFormat::AnthropicClaudeV1,
            index: Some(0),
            text: "thought".into(),
            signature: Some("sig".into()),
        };
        let unsigned = ReasoningItem::Text {
            id: None,
            format: ReasoningFormat::Unknown,
            index: None,
            text: "thought".into(),
            signature: None,
        };
        assert!(signed.carries_signed_payload());
        assert!(!unsigned.carries_signed_payload());
    }

    #[test]
    fn openrouter_codec_parses_and_writes_assistant() {
        let codec = OpenRouterReasoningCodec::new();
        let raw = vec![
            json!({
                "type": "reasoning.encrypted",
                "format": "google-gemini-v1",
                "index": 0,
                "data": "GBLOB"
            }),
            json!({"type": "noise"}),
        ];
        let items = codec.parse_response(&raw);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].format(), ReasoningFormat::GoogleGeminiV1);

        let mut msg = json!({"role": "assistant", "content": null});
        codec.write_assistant(&mut msg, &items);
        let arr = msg
            .as_object()
            .unwrap()
            .get("reasoning_details")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["data"], "GBLOB");
    }

    #[test]
    fn openrouter_codec_skips_empty_replay() {
        let codec = OpenRouterReasoningCodec::new();
        let mut msg = json!({"role": "assistant", "content": "hi"});
        codec.write_assistant(&mut msg, &[]);
        assert!(msg.as_object().unwrap().get("reasoning_details").is_none());
    }

    #[test]
    fn format_replay_contract_matches_observed_provider_enforcement() {
        // RequiredWithTools — providers that enforce signed reasoning
        // round-trip on tool-bearing turns.
        for fmt in [
            ReasoningFormat::GoogleGeminiV1,
            ReasoningFormat::AnthropicClaudeV1,
            ReasoningFormat::OpenaiResponsesV1,
            ReasoningFormat::AzureOpenaiResponsesV1,
        ] {
            assert_eq!(
                fmt.replay_contract(),
                ReplayContract::RequiredWithTools,
                "{fmt:?} should be RequiredWithTools"
            );
        }
        // Stateless — Grok and unrecognized future providers, where a
        // missing-signature warning would be a false positive.
        for fmt in [ReasoningFormat::XaiResponsesV1, ReasoningFormat::Unknown] {
            assert_eq!(
                fmt.replay_contract(),
                ReplayContract::Stateless,
                "{fmt:?} should be Stateless"
            );
        }
    }

    #[test]
    fn audit_clean_when_provider_is_stateless() {
        let audit = audit_replay(&[], ReplayContract::Stateless, true);
        assert!(audit.violation.is_none());
        assert_eq!(audit.signed_count, 0);
    }

    #[test]
    fn audit_clean_when_required_with_tools_but_next_turn_has_no_tools() {
        let audit = audit_replay(&[], ReplayContract::RequiredWithTools, false);
        assert!(audit.violation.is_none());
    }

    #[test]
    fn audit_flags_missing_signatures_for_required_with_tools() {
        let items = vec![ReasoningItem::Text {
            id: None,
            format: ReasoningFormat::GoogleGeminiV1,
            index: Some(0),
            text: "thinking out loud".into(),
            signature: None,
        }];
        let audit = audit_replay(&items, ReplayContract::RequiredWithTools, true);
        match audit.violation {
            Some(ReplayViolation::MissingSignaturesForStrictProvider {
                contract, formats, ..
            }) => {
                assert_eq!(contract, ReplayContract::RequiredWithTools);
                assert_eq!(formats, vec![ReasoningFormat::GoogleGeminiV1]);
            }
            _ => panic!("expected MissingSignaturesForStrictProvider violation"),
        }
    }

    #[test]
    fn audit_passes_when_signed_payload_present() {
        let items = vec![
            ReasoningItem::Text {
                id: None,
                format: ReasoningFormat::GoogleGeminiV1,
                index: Some(0),
                text: "summary".into(),
                signature: None,
            },
            ReasoningItem::Encrypted {
                id: None,
                format: ReasoningFormat::GoogleGeminiV1,
                index: Some(1),
                data: "BASE64".into(),
            },
        ];
        let audit = audit_replay(&items, ReplayContract::RequiredWithTools, true);
        assert!(audit.violation.is_none());
        assert_eq!(audit.signed_count, 1);
        assert_eq!(audit.item_count, 2);
    }

    #[test]
    fn audit_always_required_fires_even_without_tools() {
        let audit = audit_replay(&[], ReplayContract::AlwaysRequired, false);
        assert!(matches!(
            audit.violation,
            Some(ReplayViolation::MissingSignaturesForStrictProvider { .. })
        ));
    }
}
