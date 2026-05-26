//! Tool surface.
//!
//! `AgentTool` is the only contract the loop knows about. Tools own their
//! parameter schema, validation, and execution. The loop dispatches and
//! emits events.
//!
//! Termination is a tool decision: a tool result with `terminate: true`
//! ends the run if every tool in the batch agrees (unanimous). One tool
//! wanting to stop does not stop the batch.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{ToolError, ToolValidationError};
pub use crate::types::ToolResultBlock;

/// Loop-wide tool dispatch mode. Per-tool sequential dispatch is
/// requested via [`AgentTool::requires_exclusive_sandbox`]; this enum
/// is for pinning the whole loop (e.g. deterministic eval harness).
///
/// When a batch contains any tool with `requires_exclusive_sandbox =
/// true`, the entire batch runs sequentially regardless of this
/// setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Parallel,
    Sequential,
}

/// A tool call request emitted by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Reserved object key used to mark an argument value that the provider
/// stream layer could not parse as JSON. Tool args are always meant to be
/// JSON objects; when the model emits malformed JSON (e.g. trailing
/// comma, missing value) the provider wraps the failure in a sentinel
/// object carrying this key plus the raw payload, so the loop can emit
/// a structured "your JSON was malformed" error instead of the cryptic
/// "invalid type: string, expected struct …" that comes from
/// `serde_json::from_value` running over a `Value::String` fallback.
pub const ARG_PARSE_ERROR_MARKER: &str = "__clark_arg_parse_error";

/// Companion to [`ARG_PARSE_ERROR_MARKER`]: holds the raw JSON-ish
/// payload the model sent, so the model can see exactly what it
/// produced and fix the syntax in its next turn.
pub const ARG_PARSE_RAW_MARKER: &str = "__clark_arg_raw";

/// Build a [`Value`] that carries an argument-parse error for the loop
/// to surface. Use from any provider stream layer that decoded a tool
/// call whose `arguments` string was not valid JSON.
pub fn arg_parse_error_value(error: impl Into<String>, raw: impl Into<String>) -> Value {
    serde_json::json!({
        ARG_PARSE_ERROR_MARKER: error.into(),
        ARG_PARSE_RAW_MARKER: raw.into(),
    })
}

/// If `args` was produced by [`arg_parse_error_value`], return
/// `(error, raw)`. Otherwise return `None`.
pub fn detect_arg_parse_error(args: &Value) -> Option<(&str, &str)> {
    let obj = args.as_object()?;
    let err = obj.get(ARG_PARSE_ERROR_MARKER)?.as_str()?;
    let raw = obj.get(ARG_PARSE_RAW_MARKER)?.as_str()?;
    Some((err, raw))
}

/// Result of a tool execution.
///
/// Always contains content blocks visible to the model. `details` is
/// arbitrary structured metadata for logs / UI / replay; the model never
/// sees it directly. `terminate` is the unanimous-vote signal.
///
/// `narration` is an optional row-caption sentence shown to the user.
/// It is owned by the tool (or a product-level after-hook) and should
/// be derived from typed tool state such as path, query, exit code, or
/// byte count. The generic loop does not infer narration from private
/// model deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolResultBlock>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub details: Value,
    #[serde(default, skip_serializing_if = "is_false")]
    pub terminate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narration: Option<String>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl ToolResult {
    /// Convenience: a plain-text successful result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultBlock::Text(crate::types::TextContent {
                text: text.into(),
            })],
            is_error: false,
            details: Value::Null,
            terminate: false,
            narration: None,
        }
    }

    /// Convenience: a plain-text terminal result (vote to end the run).
    pub fn terminal(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultBlock::Text(crate::types::TextContent {
                text: text.into(),
            })],
            is_error: false,
            details: Value::Null,
            terminate: true,
            narration: None,
        }
    }

    /// Convenience: an error result. The loop treats this as a context
    /// event, not a fatal — the model can recover.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultBlock::Text(crate::types::TextContent {
                text: text.into(),
            })],
            is_error: true,
            details: Value::Null,
            terminate: false,
            narration: None,
        }
    }

    /// Attach a one-sentence diary entry in the user's voice. Whitespace-only
    /// input is dropped to keep the diary clean. Trims surrounding whitespace
    /// so call sites can hand in templated multi-line strings.
    pub fn with_narration(mut self, narration: impl Into<String>) -> Self {
        let raw: String = narration.into();
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            self.narration = Some(trimmed.to_string());
        }
        self
    }
}

/// Sink the tool can use to publish partial progress while running.
///
/// The loop forwards each partial as `AgentEvent::ToolExecutionUpdate`.
/// Tools call `update.send(...)` zero or more times before returning the
/// final result.
pub type ToolUpdateSink = mpsc::UnboundedSender<ToolResult>;

/// Tool-authored context-retention hints for history transforms.
///
/// The core loop does not interpret these policies directly. They are
/// narrow metadata for `ContextTransform` plugins that need to summarize
/// or trim history without maintaining a parallel list of tool names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolHistoryPolicy {
    /// Argument whose string value identifies duplicate calls of the
    /// same tool. Older successful results for the same value may be
    /// replaced by a marker that points at the latest result.
    pub dedup_arg: Option<&'static str>,
    /// Argument to render in compact one-line summaries.
    pub summary_arg: Option<&'static str>,
    /// Whether old successful results are re-fetchable enough to clear
    /// during time-based microcompaction.
    pub compactable_result: bool,
    /// Whether the latest successful result should be pinned near the
    /// newest user turn as the active plan.
    pub pins_active_plan: bool,
}

impl ToolHistoryPolicy {
    pub const fn new() -> Self {
        Self {
            dedup_arg: None,
            summary_arg: None,
            compactable_result: false,
            pins_active_plan: false,
        }
    }

    pub const fn dedup_arg(mut self, arg: &'static str) -> Self {
        self.dedup_arg = Some(arg);
        self
    }

    pub const fn summary_arg(mut self, arg: &'static str) -> Self {
        self.summary_arg = Some(arg);
        self
    }

    pub const fn compactable_result(mut self) -> Self {
        self.compactable_result = true;
        self
    }

    pub const fn pins_active_plan(mut self) -> Self {
        self.pins_active_plan = true;
        self
    }
}

impl Default for ToolHistoryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// A tool the agent can call.
///
/// Implementations supply: name, description, JSON schema for arguments,
/// optional argument prep + validation, and an async `execute`.
#[async_trait]
pub trait AgentTool: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    /// JSON Schema for the tool's arguments. The loop hands this verbatim
    /// to the LLM provider.
    fn parameters_schema(&self) -> Value;

    /// Whether this tool needs exclusive access to the shared sandbox
    /// state — a single browser/desktop session, a persistent terminal,
    /// the workspace cwd, etc. When ANY tool in a batch declares this,
    /// the entire batch runs sequentially.
    ///
    /// The canonical (and currently only) per-tool knob for forcing
    /// sequential dispatch. If a future use case needs sequential for a
    /// non-sandbox reason (rate-limited external API, host-process
    /// state, etc.), introduce a more specific signal then — keep this
    /// trait surface narrow until the case actually appears.
    ///
    /// For loop-wide sequential mode (e.g. deterministic eval), use
    /// [`crate::config::AgentBuilder::default_execution_mode`] instead.
    ///
    /// Default: `false` (stateless / read-only tools).
    fn requires_exclusive_sandbox(&self) -> bool {
        false
    }

    /// Maximum size of this tool's result content (in chars) that
    /// `ToolResultBudget` allows through to the model on subsequent
    /// turns. `None` means "use the global default"; `Some(usize::MAX)`
    /// means "this tool's output is too important to clip — keep
    /// verbatim". `Some(n)` declares a tool-specific cap that overrides
    /// the global default.
    ///
    /// Tools that produce large structured output the model needs to
    /// inspect in full (publish results, full-page snapshots) should
    /// return `Some(usize::MAX)`. Tools that produce voluminous and
    /// re-fetchable content (shell, file_read, browser body) should
    /// usually leave this at the default.
    ///
    /// Has no effect when `ToolResultBudget` isn't installed in the
    /// loop's `ContextTransform` chain.
    fn max_result_chars(&self) -> Option<usize> {
        None
    }

    /// Tool-owned hints for history transforms. Defaults to no special
    /// handling; tools that emit re-fetchable or summary-worthy results
    /// opt in where their argument contract is defined.
    fn history_policy(&self) -> ToolHistoryPolicy {
        ToolHistoryPolicy::default()
    }

    /// Tool-owned identity declaration for loop-detection plugins. A
    /// tool that dispatches on `action` / `mode` / similar declares
    /// the discriminator here so the runtime never has to re-encode
    /// the same fact in a separate allowlist. See
    /// `clark_agent::tool_identity` for the contract; defaults to
    /// "single opaque operation" which preserves the historical
    /// fall-through behavior for tools that opt out.
    fn identity_policy(&self) -> crate::tool_identity::ToolIdentityPolicy {
        crate::tool_identity::ToolIdentityPolicy::default()
    }

    /// Whether a non-fatal failure of this tool in a parallel batch
    /// should cancel its still-running sibling tools. Default `false`
    /// — failures are isolated and siblings run to completion. Tools
    /// where one failure makes parallel work meaningless (a `shell`
    /// step that gates `npm test`, a delegated build whose result the
    /// siblings depend on) opt in by overriding this to `true`.
    ///
    /// Cancelled siblings produce a `ToolResult` with
    /// `is_error: true, content: "aborted because sibling 'X' failed"`
    /// — they remain context events the next turn can react to, never
    /// `LoopError`s. Sibling-abort therefore never ends the run on its
    /// own; the unanimous-vote termination rule is preserved.
    ///
    /// Cancellation is cooperative: tools must check
    /// `signal.is_cancelled()` (or wrap blocking work in `select!`) to
    /// honor the cancel promptly. Subprocess-based tools should rely
    /// on `Drop` killing the child when the future is dropped.
    fn aborts_siblings_on_error(&self) -> bool {
        false
    }

    /// Whether this tool consumes a slot from
    /// `LoopConfig::max_tool_calls_per_turn`.
    ///
    /// Default `true`: tools do work, ask/answer, mutate state, or otherwise
    /// participate in the loop's bounded execution budget. Lightweight
    /// progress-only signals can opt out so they do not starve the next real
    /// action when a provider emits a status note and a work tool in the same
    /// assistant turn.
    fn counts_toward_tool_call_limit(&self) -> bool {
        true
    }

    /// Whether this tool is safe to invoke multiple times in a single
    /// assistant turn alongside other tool calls.
    ///
    /// Default `false`: tools serialize at the configured cap so writes
    /// and stateful operations stay sequenced. Read-only / idempotent
    /// tools (web search, file read, grep, glob, snapshots) override to
    /// `true` so a provider that batches several independent lookups in
    /// one turn does not get N-1 of them rejected with a "only the first
    /// call can run" error. Parallel-safe tools still execute one at a
    /// time on the runtime side; they just do not contend for the
    /// per-turn cap.
    fn parallel_safe_per_turn(&self) -> bool {
        false
    }

    /// Whether this tool's `terminate` vote is included in the
    /// unanimous-vote tally that decides whether the batch ends the
    /// run.
    ///
    /// Default `true`: every tool's vote counts. The batch terminates
    /// only when *every* tool that opts in voted `terminate: true`.
    ///
    /// Lightweight status-only tools (progress notes, hidden journals)
    /// override to `false`. The runtime then ignores their vote
    /// entirely — they are neither a "yes" nor a "no" — so a model
    /// that emits `[message_result, message_info]` in the same batch
    /// can still terminate. An all-advisory batch (no tool with this
    /// flag set to `true` voted yes) does NOT terminate, preserving
    /// the contract that progress notes never end a run on their own.
    fn counts_toward_termination_vote(&self) -> bool {
        true
    }

    /// Optional argument normalization before validation. Pure function.
    /// Default: identity.
    fn prepare_arguments(&self, args: Value) -> Value {
        args
    }

    /// Validate prepared arguments. Default: succeed.
    /// Implement for tools that have action-specific required fields not
    /// expressible in pure JSON Schema.
    fn validate(&self, _args: &Value) -> Result<(), ToolValidationError> {
        Ok(())
    }

    /// Execute the tool. Returns the final result.
    ///
    /// `update` may be used to publish partial progress while running.
    /// Honor `signal` for cancellation.
    async fn execute(
        &self,
        call_id: &str,
        args: Value,
        signal: CancellationToken,
        update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError>;
}

// ---------------------------------------------------------------------------
// TypedAgentTool — the canonical authoring surface for tools whose argument
// shape is a typed Rust struct or enum.
//
// One source of truth (`Args`) drives both the wire schema (generated
// via schemars) and the runtime parse — no hand-written JSON Schema,
// no opportunity for drift. New tool authors implement `TypedAgentTool` and
// get the `AgentTool` impl for free via the blanket below.
//
// Tag-dispatched tools (e.g. `plan(action="set"|"update"|"advance")`,
// `publish(target="website"|"artifact")`) use a `#[serde(tag = "...")]`
// enum as `Args`; serde routes the discriminator natively, so the
// "unknown field `action`" failure mode that motivated this trait can
// no longer happen.
// ---------------------------------------------------------------------------

/// Implement this for tools whose argument shape is a typed Rust
/// struct/enum. The blanket `AgentTool` impl below derives
/// `parameters_schema` from `Args` via schemars and centralizes the
/// `Value → Args` parse path. New tools should implement `TypedAgentTool`,
/// not `AgentTool` directly; existing tools are migrated incrementally.
#[async_trait]
pub trait TypedAgentTool: Send + Sync + 'static {
    /// The argument shape. The wire schema is generated from this
    /// type; the dispatcher parses incoming `Value` into `Args` once
    /// and hands the typed value to `run`.
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static;

    fn name(&self) -> &str;
    fn description(&self) -> &str;

    /// Whether this tool needs exclusive sandbox access. Default false.
    fn requires_exclusive_sandbox(&self) -> bool {
        false
    }

    /// Per-tool max-result-chars override for `ToolResultBudget`.
    /// Default `None` (use the global default).
    fn max_result_chars(&self) -> Option<usize> {
        None
    }

    /// Tool-owned hints for history transforms. Defaults to no special
    /// handling.
    fn history_policy(&self) -> ToolHistoryPolicy {
        ToolHistoryPolicy::default()
    }

    /// Tool-owned identity declaration for loop-detection plugins.
    /// Mirrors `AgentTool::identity_policy`; defaults to "single
    /// opaque operation". See `clark_agent::tool_identity`.
    fn identity_policy(&self) -> crate::tool_identity::ToolIdentityPolicy {
        crate::tool_identity::ToolIdentityPolicy::default()
    }

    /// Whether a non-fatal failure of this tool in a parallel batch
    /// cancels still-running siblings. Default false.
    fn aborts_siblings_on_error(&self) -> bool {
        false
    }

    /// Whether this tool consumes a slot from
    /// `LoopConfig::max_tool_calls_per_turn`. Default true.
    fn counts_toward_tool_call_limit(&self) -> bool {
        true
    }

    /// Whether this tool is safe to invoke multiple times in a single
    /// assistant turn alongside other tool calls. Default `false`. See
    /// the corresponding `AgentTool::parallel_safe_per_turn` docstring.
    fn parallel_safe_per_turn(&self) -> bool {
        false
    }

    /// Whether this tool's `terminate` vote counts in the
    /// unanimous-vote tally. Default `true`. Status-only progress
    /// tools opt out by returning `false`; see the corresponding
    /// `AgentTool::counts_toward_termination_vote` docstring.
    fn counts_toward_termination_vote(&self) -> bool {
        true
    }

    /// Optional pre-deserialization normalization of raw args. Pure
    /// function. Runs before `strip_top_level_nulls` and
    /// `coerce_string_scalars_at_top_level` so tool-specific
    /// canonicalization (e.g. inferring a tagged-enum's `action`
    /// discriminator from variant-unique fields) lands first.
    ///
    /// Default: identity. See [`AgentTool::prepare_arguments`].
    fn prepare_arguments(&self, args: Value) -> Value {
        args
    }

    /// Execute the tool with already-parsed typed args.
    async fn run(
        &self,
        call_id: &str,
        args: Self::Args,
        signal: CancellationToken,
        update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError>;
}

/// Blanket impl: every `TypedAgentTool` is automatically an `AgentTool`.
///
/// The schema is built with `inline_subschemas = true` because some
/// strict tool-schema validators (Azure, certain OpenAI-compatible
/// proxies) reject `$ref` chains in tool schemas. Inlining keeps the
/// generated JSON Schema flat and provider-portable.
#[async_trait]
impl<T: TypedAgentTool> AgentTool for T {
    fn name(&self) -> &str {
        TypedAgentTool::name(self)
    }

    fn description(&self) -> &str {
        TypedAgentTool::description(self)
    }

    fn parameters_schema(&self) -> Value {
        let settings = schemars::gen::SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
        });
        let generator = settings.into_generator();
        let schema = generator.into_root_schema_for::<T::Args>();
        let value = serde_json::to_value(schema).expect("typed-tool schema serializes");
        let mut value = flatten_tagged_oneof_schema(value);
        normalize_strict_validator_quirks(&mut value);
        value
    }

    fn requires_exclusive_sandbox(&self) -> bool {
        TypedAgentTool::requires_exclusive_sandbox(self)
    }

    fn max_result_chars(&self) -> Option<usize> {
        TypedAgentTool::max_result_chars(self)
    }

    fn history_policy(&self) -> ToolHistoryPolicy {
        TypedAgentTool::history_policy(self)
    }

    fn identity_policy(&self) -> crate::tool_identity::ToolIdentityPolicy {
        TypedAgentTool::identity_policy(self)
    }

    fn aborts_siblings_on_error(&self) -> bool {
        TypedAgentTool::aborts_siblings_on_error(self)
    }

    fn counts_toward_tool_call_limit(&self) -> bool {
        TypedAgentTool::counts_toward_tool_call_limit(self)
    }

    fn parallel_safe_per_turn(&self) -> bool {
        TypedAgentTool::parallel_safe_per_turn(self)
    }

    fn counts_toward_termination_vote(&self) -> bool {
        TypedAgentTool::counts_toward_termination_vote(self)
    }

    fn prepare_arguments(&self, args: Value) -> Value {
        TypedAgentTool::prepare_arguments(self, args)
    }

    async fn execute(
        &self,
        call_id: &str,
        args: Value,
        signal: CancellationToken,
        update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        // Strip top-level `null` fields before deserializing.
        // Tagged-enum tools (shell, plan, ...) have variants with
        // `deny_unknown_fields`, but `flatten_tagged_oneof_schema`
        // exposes EVERY variant's fields as a union. Models like
        // gemini-3-flash-preview populate non-applicable fields with
        // `null` ("being helpful" — submit all schema-known fields).
        // Without this strip, the chosen variant rejects with
        // `unknown field` and a turn is wasted (matrix run
        // 20260502_111307: every gemini scenario hit this on shell).
        // Nulls carry no semantic value at this boundary — they are
        // either "field not set" or "inapplicable to the chosen
        // variant"; both collapse to "drop the field and let the
        // variant's `serde(default)` apply".
        //
        // Run tool-specific `prepare_arguments` FIRST so per-tool
        // canonicalization (e.g. inferring a tagged-enum's `action`
        // discriminator from variant-unique fields like `url`) lands
        // before the generic null-strip and string-scalar coercion.
        let prepared = AgentTool::prepare_arguments(self, args);
        let stripped = strip_top_level_nulls(prepared);
        // Coerce string-encoded scalars (integers, numbers, booleans)
        // to their declared types BEFORE serde validation runs. Models
        // in the auto-when-forced class (Qwen 3.5 Flash today) emit
        // tool-call arguments where every value is a JSON string —
        // `{"max_iterations": "50"}` instead of `{"max_iterations": 50}`.
        // The strict serde path rejects every such call, wasting a turn
        // per field; coercion converts the obvious case in-place using
        // the tool's own schema as the source of truth.
        let schema = AgentTool::parameters_schema(self);
        let coerced = coerce_string_scalars_at_top_level(stripped, &schema);
        let parsed: T::Args = match serde_json::from_value(coerced) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "{}: invalid arguments: {}",
                    TypedAgentTool::name(self),
                    enrich_arg_parse_error_message(&e),
                )));
            }
        };
        TypedAgentTool::run(self, call_id, parsed, signal, update).await
    }
}

/// Convert top-level string-encoded scalars to their JSON-Schema-declared
/// types when the conversion is unambiguous. Walks `value` (which must be
/// an object) and, for each property whose schema declares a single scalar
/// type (`integer`, `number`, `boolean`), parses the corresponding string
/// value in place. Leaves arrays, objects, nested oneOf branches, and
/// fields with a non-string current value untouched — those go through
/// the strict serde path unchanged. Conservative by design: any
/// ambiguity (multi-type schemas, untyped properties, unparseable
/// strings) preserves the original value so the strict validator still
/// catches genuinely-malformed args.
fn coerce_string_scalars_at_top_level(value: Value, schema: &Value) -> Value {
    let Value::Object(mut map) = value else {
        return value;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Value::Object(map);
    };
    for (key, val) in map.iter_mut() {
        let Some(prop_schema) = properties.get(key) else {
            continue;
        };
        coerce_one_scalar_in_place(val, prop_schema);
    }
    Value::Object(map)
}

fn coerce_one_scalar_in_place(value: &mut Value, prop_schema: &Value) {
    let Some(text) = value.as_str() else {
        return;
    };
    let Some(target) = scalar_target_from_schema(prop_schema) else {
        return;
    };
    match target {
        ScalarTarget::Integer => {
            let trimmed = text.trim();
            if let Ok(n) = trimmed.parse::<i64>() {
                *value = Value::Number(serde_json::Number::from(n));
            } else if let Ok(n) = trimmed.parse::<u64>() {
                *value = Value::Number(serde_json::Number::from(n));
            }
        }
        ScalarTarget::Number => {
            let trimmed = text.trim();
            if let Ok(n) = trimmed.parse::<f64>() {
                if let Some(num) = serde_json::Number::from_f64(n) {
                    *value = Value::Number(num);
                }
            }
        }
        ScalarTarget::Boolean => match text.trim() {
            "true" | "True" | "TRUE" => *value = Value::Bool(true),
            "false" | "False" | "FALSE" => *value = Value::Bool(false),
            _ => {}
        },
    }
}

#[derive(Debug, Clone, Copy)]
enum ScalarTarget {
    Integer,
    Number,
    Boolean,
}

fn scalar_target_from_schema(prop_schema: &Value) -> Option<ScalarTarget> {
    let type_field = prop_schema.get("type")?;
    let single = match type_field {
        Value::String(s) => Some(s.as_str()),
        // Optional-shaped schemas often render as ["T", "null"]; pick the
        // non-null entry. Anything wider (e.g. ["string", "integer"]) is
        // genuinely ambiguous — skip and let the strict validator decide.
        Value::Array(arr) => {
            let non_null: Vec<&str> = arr
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|s| *s != "null")
                .collect();
            if non_null.len() == 1 {
                Some(non_null[0])
            } else {
                None
            }
        }
        _ => None,
    }?;
    match single {
        "integer" => Some(ScalarTarget::Integer),
        "number" => Some(ScalarTarget::Number),
        "boolean" => Some(ScalarTarget::Boolean),
        _ => None,
    }
}

/// Append a self-correcting hint to a serde-deserialize error message
/// when the failure pattern is something a model can fix on the next
/// turn (e.g. "string \"50\", expected usize" → "Did you mean the
/// integer 50?"). The base error text is preserved verbatim so the
/// existing format stays diffable; the hint is suffixed after a period.
fn enrich_arg_parse_error_message(err: &serde_json::Error) -> String {
    let raw = err.to_string();
    match arg_parse_hint(&raw) {
        Some(hint) => format!("{raw}. {hint}"),
        None => raw,
    }
}

fn arg_parse_hint(raw: &str) -> Option<String> {
    let value = extract_invalid_string_value(raw)?;
    if expects_integer(raw) {
        let parsed: i128 = value.trim().parse().ok()?;
        return Some(format!(
            "Did you mean the integer {parsed}? Resend without quotes."
        ));
    }
    if expects_number(raw) {
        let parsed: f64 = value.trim().parse().ok()?;
        return Some(format!(
            "Did you mean the number {parsed}? Resend without quotes."
        ));
    }
    if expects_boolean(raw) {
        return match value.trim() {
            "true" | "True" | "TRUE" => Some(
                "Did you mean true? Resend as a boolean literal (lowercase, no quotes)."
                    .to_string(),
            ),
            "false" | "False" | "FALSE" => Some(
                "Did you mean false? Resend as a boolean literal (lowercase, no quotes)."
                    .to_string(),
            ),
            _ => None,
        };
    }
    if expects_sequence(raw) {
        return Some(
            "Expected a JSON array (e.g. `[{...}, {...}]`); the field cannot be a string. \
             Resend the value as an array of structured items, not a string of XML-like markup."
                .to_string(),
        );
    }
    None
}

fn extract_invalid_string_value(raw: &str) -> Option<&str> {
    // Serde's `invalid type` errors quote the offending value as
    // `string "X"`. Locate the inner content without pulling in a regex
    // dependency; bail on the first malformed shape.
    let start = raw.find("string \"")? + "string \"".len();
    let rest = &raw[start..];
    let end = rest.find('\"')?;
    Some(&rest[..end])
}

fn expects_integer(raw: &str) -> bool {
    raw.contains("expected usize")
        || raw.contains("expected isize")
        || raw.contains("expected u8")
        || raw.contains("expected u16")
        || raw.contains("expected u32")
        || raw.contains("expected u64")
        || raw.contains("expected i8")
        || raw.contains("expected i16")
        || raw.contains("expected i32")
        || raw.contains("expected i64")
        || raw.contains("expected integer")
}

fn expects_number(raw: &str) -> bool {
    raw.contains("expected f32")
        || raw.contains("expected f64")
        || raw.contains("expected floating point")
}

fn expects_boolean(raw: &str) -> bool {
    raw.contains("expected a boolean") || raw.contains("expected bool")
}

fn expects_sequence(raw: &str) -> bool {
    raw.contains("expected a sequence") || raw.contains("expected an array")
}

fn strip_top_level_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.into_iter().filter(|(_, v)| !v.is_null()).collect())
        }
        other => other,
    }
}

/// Flatten a top-level `oneOf` of tag-discriminated objects into a
/// single object schema with the discriminator promoted to a top-level
/// `enum` field. Schemars naturally emits `oneOf` for
/// `#[serde(tag = "kind")]` enums, but several strict tool-schema
/// validators (Azure's, certain OpenAI-compatible proxies, observed
/// xAI/Grok behaviour where the model silently refuses to call the
/// tool) reject schemas that have `oneOf` at the top level. The
/// runtime contract is unchanged — serde still routes by the
/// discriminator on the input side and `deny_unknown_fields` still
/// catches per-variant typos on the parse side. The wire schema just
/// presents a flatter union to the model.
///
/// Inputs that aren't a tag-dispatched `oneOf` (single struct, true
/// untagged unions) pass through unchanged.
fn flatten_tagged_oneof_schema(schema: Value) -> Value {
    let Value::Object(mut root) = schema else {
        return schema;
    };
    let Some(Value::Array(variants)) = root.remove("oneOf") else {
        // No oneOf → already a flat schema (single-struct tool).
        if !root.is_empty() {
            return Value::Object(root);
        }
        return Value::Null;
    };

    // Per-variant info captured during the walk so we can annotate
    // each merged property with which variants own it. Strict
    // validators (Azure) reject top-level `allOf` / `oneOf` / `anyOf`
    // / `enum` / `not` even when paired with `type: "object"`, so we
    // can't carry per-variant constraints structurally at the root.
    // Instead, encode the per-variant applicability into each
    // property's `description` ("applies when kind in: [document]"),
    // derived from the same tagged-enum walk. The model reads it; the
    // validator doesn't care about description text.
    struct VariantSpec {
        tag_value_str: Option<String>,
        own_field_names: Vec<String>,
    }

    let mut discriminator: Option<String> = None;
    let mut variant_specs: Vec<VariantSpec> = Vec::with_capacity(variants.len());
    let mut merged_props = serde_json::Map::new();
    let mut required_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut tag_in_required = true;
    // tag-values for the discriminator's enum; preserved as Value to
    // support non-string tags even though only strings are common.
    let mut tag_values: Vec<Value> = Vec::with_capacity(variants.len());

    for variant in &variants {
        let Some(obj) = variant.as_object() else {
            return reassemble_oneof(root, variants);
        };
        let Some(Value::Object(props)) = obj.get("properties").cloned() else {
            return reassemble_oneof(root, variants);
        };
        // Find the variant's tag property: a property whose schema is
        // a single-element `enum` of strings.
        let mut variant_tag: Option<(String, Value)> = None;
        for (name, prop) in props.iter() {
            let Some(prop_obj) = prop.as_object() else {
                continue;
            };
            let Some(Value::Array(enum_values)) = prop_obj.get("enum").cloned() else {
                continue;
            };
            if enum_values.len() == 1 {
                variant_tag = Some((name.clone(), enum_values.into_iter().next().unwrap()));
                break;
            }
        }
        let Some((tag_name, tag_value)) = variant_tag else {
            return reassemble_oneof(root, variants);
        };
        match &discriminator {
            None => discriminator = Some(tag_name.clone()),
            Some(existing) if existing == &tag_name => {}
            Some(_) => return reassemble_oneof(root, variants),
        }
        tag_values.push(tag_value.clone());

        // Merge non-tag properties (union) and record this variant's
        // own field names for the description annotation pass below.
        let mut own_field_names = Vec::new();
        for (name, prop_schema) in props.iter() {
            if name == &tag_name {
                continue;
            }
            merged_props
                .entry(name.clone())
                .or_insert_with(|| prop_schema.clone());
            own_field_names.push(name.clone());
        }

        // Tag is required at the outer level only if every variant
        // requires it. Per-variant non-tag required keys can't be
        // hoisted to the outer schema — they'd break sibling variants.
        // Serde's `deny_unknown_fields` still enforces them per-variant
        // at parse time.
        let mut tag_required_here = false;
        if let Some(Value::Array(req)) = obj.get("required") {
            for r in req {
                if let Some(s) = r.as_str() {
                    if s == tag_name {
                        tag_required_here = true;
                    }
                }
            }
        }
        if !tag_required_here {
            tag_in_required = false;
        }

        variant_specs.push(VariantSpec {
            tag_value_str: tag_value.as_str().map(str::to_string),
            own_field_names,
        });
    }

    let Some(discriminator) = discriminator else {
        return reassemble_oneof(root, variants);
    };

    // Annotate each merged property with the variants that own it.
    // Skip when the property is owned by every variant (no narrowing
    // information to add) and when any variant tag isn't a plain
    // string (annotation requires a stable label).
    let total_variants = variant_specs.len();
    let all_tags_are_strings = variant_specs.iter().all(|s| s.tag_value_str.is_some());
    if all_tags_are_strings && total_variants > 1 {
        let mut owners: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        for spec in &variant_specs {
            let tag_label = spec.tag_value_str.clone().unwrap_or_default();
            for field in &spec.own_field_names {
                owners
                    .entry(field.clone())
                    .or_default()
                    .push(tag_label.clone());
            }
        }
        for (field, mut variant_tags) in owners {
            if variant_tags.len() == total_variants {
                continue;
            }
            variant_tags.sort();
            variant_tags.dedup();
            let suffix = format!(
                " (applies when {discriminator} in: [{}])",
                variant_tags.join(", ")
            );
            if let Some(Value::Object(prop_map)) = merged_props.get_mut(&field) {
                let new_desc = match prop_map.get("description") {
                    Some(Value::String(existing)) if !existing.is_empty() => {
                        format!("{existing}{suffix}")
                    }
                    _ => suffix.trim_start().to_string(),
                };
                prop_map.insert("description".to_string(), Value::String(new_desc));
            }
        }
    }

    // Build the discriminator property with the union of tag values.
    // It must be first in insertion order: models emit JSON
    // autoregressively, so variant-specific fields need to be
    // conditioned on the already-emitted discriminator instead of the
    // other way around.
    let mut tag_prop = serde_json::Map::new();
    tag_prop.insert("type".to_string(), Value::String("string".to_string()));
    tag_prop.insert("enum".to_string(), Value::Array(tag_values));
    let mut ordered_props = serde_json::Map::new();
    ordered_props.insert(discriminator.clone(), Value::Object(tag_prop));
    for (name, schema) in merged_props {
        ordered_props.insert(name, schema);
    }
    if tag_in_required {
        required_set.insert(discriminator);
    }

    let mut out = serde_json::Map::new();
    if let Some(desc) = root.remove("description") {
        out.insert("description".to_string(), desc);
    }
    if let Some(schema) = root.remove("$schema") {
        out.insert("$schema".to_string(), schema);
    }
    out.insert("type".to_string(), Value::String("object".to_string()));
    out.insert("properties".to_string(), Value::Object(ordered_props));
    if !required_set.is_empty() {
        out.insert(
            "required".to_string(),
            Value::Array(required_set.into_iter().map(Value::String).collect()),
        );
    }
    Value::Object(out)
}

fn reassemble_oneof(mut root: serde_json::Map<String, Value>, variants: Vec<Value>) -> Value {
    root.insert("oneOf".to_string(), Value::Array(variants));
    Value::Object(root)
}

/// Coerce schemars output into shapes that strict tool-schema
/// validators (Azure's, OpenAI's via Azure proxy, several
/// OpenAI-compatible upstreams) accept. The current quirks list:
///
/// 1. `items: true` (boolean schema, valid in JSON Schema 2020-12 and
///    schemars's default for `Vec<Value>` cells) → rewrite to
///    `items: {}` (empty-object schema, draft-07 compatible). Azure
///    rejects boolean schemas with
///    `array schema items is not an object`.
///
/// Walks the tree once, mutating in place. Idempotent.
fn normalize_strict_validator_quirks(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // Coerce `items: true` to `items: {}`.
            if let Some(items) = map.get_mut("items") {
                if matches!(items, Value::Bool(true)) {
                    *items = Value::Object(serde_json::Map::new());
                }
            }
            for v in map.values_mut() {
                normalize_strict_validator_quirks(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                normalize_strict_validator_quirks(v);
            }
        }
        _ => {}
    }
}

/// Registry of available tools, keyed by name.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn AgentTool>>,
    order: Vec<String>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, tool: Arc<dyn AgentTool>) -> Self {
        self.register(tool);
        self
    }

    pub fn register(&mut self, tool: Arc<dyn AgentTool>) {
        let name = tool.name().to_string();
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tools.get(name).cloned()
    }

    pub fn history_policy(&self, name: &str) -> ToolHistoryPolicy {
        self.tools
            .get(name)
            .map(|tool| tool.history_policy())
            .unwrap_or_default()
    }

    /// Identity declaration for one tool — used by the semantic-loop
    /// detector and other plugins that need to recognize repeats.
    /// Returns the default ("single opaque operation") for unknown
    /// names; the detector treats that as the historical
    /// fall-through and falls back to canonical-JSON identity.
    pub fn identity_policy(&self, name: &str) -> crate::tool_identity::ToolIdentityPolicy {
        self.tools
            .get(name)
            .map(|tool| tool.identity_policy())
            .unwrap_or_default()
    }

    /// Snapshot of identity policies for every registered tool. The
    /// `SemanticLoopDetector` (and any future plugin that needs the
    /// same identity contract) takes one of these at construction so
    /// it does not have to hold an `Arc<ToolRegistry>`.
    pub fn identity_policies(&self) -> std::collections::HashMap<String, crate::tool_identity::ToolIdentityPolicy> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.identity_policy()))
            .collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn AgentTool>> {
        self.order.iter().filter_map(|name| self.tools.get(name))
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.order)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TextContent;
    use schemars::JsonSchema;
    use serde::Deserialize;

    // ---- flatten_tagged_oneof_schema ----------------------------------

    #[derive(Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct DocVariantArgs {
        filename: String,
        #[serde(default)]
        title: Option<String>,
    }

    #[derive(Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct ExcelVariantArgs {
        filename: String,
        #[serde(default)]
        rows: Vec<Vec<serde_json::Value>>,
    }

    #[derive(Deserialize, JsonSchema)]
    #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
    #[allow(dead_code)]
    enum ExampleArgs {
        Document(DocVariantArgs),
        Excel(ExcelVariantArgs),
    }

    fn build_example_schema() -> Value {
        let settings = schemars::gen::SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
        });
        let g = settings.into_generator();
        let s = g.into_root_schema_for::<ExampleArgs>();
        let raw = serde_json::to_value(s).unwrap();
        flatten_tagged_oneof_schema(raw)
    }

    #[derive(Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct NonAlphabeticOrderCanaryArgs {
        zeta_selector: String,
        alpha_payload: String,
        middle_payload: String,
    }

    #[test]
    fn schema_runtime_preserves_insertion_order_for_tool_objects() {
        // This is the low-level wire invariant behind tool-schema field
        // geometry. The model emits arguments autoregressively in schema
        // order, so serde_json objects must serialize in insertion order,
        // not alphabetical map order.
        let mut object = serde_json::Map::new();
        object.insert("zeta_selector".to_string(), Value::String("z".to_string()));
        object.insert("alpha_payload".to_string(), Value::String("a".to_string()));
        object.insert("middle_payload".to_string(), Value::String("m".to_string()));

        let keys = object.keys().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(
            keys,
            ["zeta_selector", "alpha_payload", "middle_payload"],
            "serde_json::Map must keep insertion order; losing this breaks \
             model-facing tool-schema property order"
        );

        let serialized = serde_json::to_string(&Value::Object(object)).unwrap();
        assert_eq!(
            serialized, r#"{"zeta_selector":"z","alpha_payload":"a","middle_payload":"m"}"#,
            "schema JSON serialization must preserve object insertion order"
        );
    }

    #[test]
    fn schemars_preserves_declared_struct_order_for_tool_args() {
        // Workspace schemars must keep Rust declaration order in
        // JSON-Schema `properties`; otherwise any Args type with a
        // discriminator, planning field, or thinking field silently
        // changes the order in which the model writes arguments.
        let settings = schemars::gen::SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
        });
        let schema = serde_json::to_value(
            settings
                .into_generator()
                .into_root_schema_for::<NonAlphabeticOrderCanaryArgs>(),
        )
        .expect("schema serializes");
        let props = schema
            .get("properties")
            .and_then(Value::as_object)
            .expect("schema must expose properties");
        let order = props.keys().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(
            order,
            ["zeta_selector", "alpha_payload", "middle_payload"],
            "schemars must emit Args fields in declaration order for \
             autoregressive tool-call conditioning"
        );
    }

    #[test]
    fn flatten_tagged_oneof_produces_flat_object_schema() {
        let s = build_example_schema();
        assert_eq!(s.get("type").and_then(Value::as_str), Some("object"));
        // Top level no longer has oneOf — that's the whole point.
        assert!(s.get("oneOf").is_none());
        // Discriminator is at the top level with the union of variant
        // tag values.
        let kind_prop = s.pointer("/properties/kind").expect("kind property");
        assert_eq!(
            kind_prop.get("type").and_then(Value::as_str),
            Some("string")
        );
        let kind_enum = kind_prop
            .get("enum")
            .and_then(Value::as_array)
            .expect("enum");
        let mut kinds: Vec<&str> = kind_enum.iter().filter_map(Value::as_str).collect();
        kinds.sort();
        assert_eq!(kinds, vec!["document", "excel"]);
        let props = s
            .get("properties")
            .and_then(Value::as_object)
            .expect("properties");
        let order: Vec<&str> = props.keys().map(String::as_str).collect();
        assert_eq!(
            order.first().copied(),
            Some("kind"),
            "discriminator must be emitted before payload fields so \
             variant-specific keys are conditioned on the selected kind"
        );
        // Per-variant fields are merged into one properties object.
        assert!(s.pointer("/properties/filename").is_some());
        assert!(s.pointer("/properties/title").is_some());
        assert!(s.pointer("/properties/rows").is_some());
        // `kind` is required at the top level.
        let req = s
            .get("required")
            .and_then(Value::as_array)
            .expect("required");
        assert!(req.iter().any(|v| v.as_str() == Some("kind")));
    }

    #[test]
    fn flatten_tagged_oneof_annotates_variant_specific_property_descriptions() {
        // Strict tool-schema validators (Azure et al.) reject top-level
        // `allOf` / `oneOf` / `anyOf` / `enum` / `not` even with
        // `type: "object"` present (see openai_basic.rs). So we can't
        // express per-variant constraints structurally at the root.
        //
        // Fallback: annotate each merged property's `description` with
        // the variant tags that own it ("applies when <discriminator>
        // in: [...]"). The model reads descriptions; the validator
        // ignores them. Derived purely from the same `JsonSchema`-
        // derived enum walk — single source of truth.
        //
        // Regression target: weaker models mixed sibling-variant
        // fields until the schema told them which fields belong to
        // which discriminator value.
        let s = build_example_schema();

        // Properties present in BOTH variants (`filename`) get no
        // narrowing suffix — they apply universally.
        let filename_desc = s
            .pointer("/properties/filename/description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            !filename_desc.contains("applies when kind in"),
            "shared property `filename` must NOT carry a narrowing \
             suffix; got: {filename_desc:?}"
        );

        // Properties owned by only one variant get a narrowing suffix.
        let title_desc = s
            .pointer("/properties/title/description")
            .and_then(Value::as_str)
            .expect("title description present");
        assert!(
            title_desc.contains("applies when kind in: [document]"),
            "Document-only `title` must declare its variant scope; \
             got: {title_desc:?}"
        );
        let rows_desc = s
            .pointer("/properties/rows/description")
            .and_then(Value::as_str)
            .expect("rows description present");
        assert!(
            rows_desc.contains("applies when kind in: [excel]"),
            "Excel-only `rows` must declare its variant scope; \
             got: {rows_desc:?}"
        );

        // No top-level `allOf` / `oneOf` / `anyOf` — the validator
        // rejects them.
        assert!(
            s.get("allOf").is_none(),
            "top-level allOf would be rejected by Azure's tool validator"
        );
        assert!(s.get("oneOf").is_none());
        assert!(s.get("anyOf").is_none());
    }

    #[test]
    fn normalize_strict_quirks_rewrites_items_true_to_empty_object() {
        // Regression: schemars emits `items: true` for `Vec<Value>`
        // cells. Azure's tool-schema validator rejects boolean
        // schemas with `array schema items is not an object`,
        // failing every nano (Azure-routed) call. The normalizer
        // walks the tree and coerces `items: true` to `items: {}`.
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": true
                    }
                }
            }
        });
        normalize_strict_validator_quirks(&mut schema);
        assert_eq!(
            schema.pointer("/properties/rows/items/items"),
            Some(&serde_json::json!({})),
        );
    }

    #[test]
    fn strip_top_level_nulls_removes_inapplicable_variant_fields() {
        // Regression: matrix run 20260502_111307 surfaced weaker
        // models submitting EVERY field from EVERY tagged-enum
        // variant, with `null` for the non-applicable ones,
        // alongside the chosen discriminator. The chosen variant
        // has `deny_unknown_fields` and rejected with unknown
        // sibling fields. Stripping top-level nulls before
        // deserializing collapses these to missing fields and lets
        // `serde(default)` apply.
        let model_payload = serde_json::json!({
            "action": "run",
            "command": "echo hi",
            "workdir": "/home/user/workspace",
            // Sibling-variant fields the model populated with null:
            "code": null,
            "interpreter": null,
            "ext": null,
            "exec_dir": null,
            "max_token": null,
            "truncate_from": null,
            "run_id": null,
            "after_seq": null,
            "max_events": null,
            "timeout_s": null,
            "timeout_ms": null,
            "terminal": null,
            "force": null,
            // Plus a real value to confirm only nulls are dropped.
            "timeout_secs": 60,
        });
        let stripped = strip_top_level_nulls(model_payload);
        let obj = stripped.as_object().expect("object");
        // Nulls gone.
        assert!(!obj.contains_key("code"));
        assert!(!obj.contains_key("ext"));
        assert!(!obj.contains_key("max_token"));
        assert!(!obj.contains_key("force"));
        // Real values preserved.
        assert_eq!(obj.get("action").and_then(Value::as_str), Some("run"));
        assert_eq!(obj.get("command").and_then(Value::as_str), Some("echo hi"));
        assert_eq!(obj.get("timeout_secs").and_then(Value::as_i64), Some(60));
    }

    #[test]
    fn strip_top_level_nulls_passes_through_non_object_values() {
        // Defensive: tool args at the top level should always be
        // objects, but the helper must not panic on the off-chance
        // a transport hands us a primitive.
        assert_eq!(
            strip_top_level_nulls(serde_json::json!("text")),
            serde_json::json!("text")
        );
        assert_eq!(strip_top_level_nulls(Value::Null), Value::Null);
    }

    // ---- string-encoded-scalar coercion -------------------------------
    //
    // qwen 3.5 Flash (and other auto-when-forced providers) emit tool
    // arguments as JSON strings for fields the schema declares as
    // integers, booleans, or numbers. The 20260517 external-agent-parity
    // run showed `max_iterations: "50"`, `num_results: "10"`,
    // `full_page: "True"`, `full_page: "true"` — each a wasted turn
    // under strict serde. The coercion helpers normalize the dominant
    // cases against the tool's own JSON Schema; ambiguous cases are
    // left to the strict path.

    fn make_schema(properties: Value) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": properties,
        })
    }

    #[test]
    fn coerce_string_to_integer_when_schema_says_integer() {
        let schema = make_schema(serde_json::json!({
            "max_iterations": {"type": "integer"},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"max_iterations": "50"}),
            &schema,
        );
        assert_eq!(coerced, serde_json::json!({"max_iterations": 50}));
    }

    #[test]
    fn coerce_string_to_integer_handles_negative_and_whitespace() {
        let schema = make_schema(serde_json::json!({
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"offset": "-7", "limit": "  42  "}),
            &schema,
        );
        assert_eq!(coerced, serde_json::json!({"offset": -7, "limit": 42}));
    }

    #[test]
    fn coerce_string_to_boolean_for_each_case_variant() {
        let schema = make_schema(serde_json::json!({
            "full_page": {"type": "boolean"},
            "headless": {"type": "boolean"},
            "verbose": {"type": "boolean"},
            "untouched": {"type": "boolean"},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({
                "full_page": "true",
                "headless": "True",
                "verbose": "FALSE",
                "untouched": "maybe",
            }),
            &schema,
        );
        // Recognised forms become bools; gibberish stays a string so the
        // strict validator still rejects with a useful error.
        assert_eq!(coerced["full_page"], serde_json::json!(true));
        assert_eq!(coerced["headless"], serde_json::json!(true));
        assert_eq!(coerced["verbose"], serde_json::json!(false));
        assert_eq!(coerced["untouched"], serde_json::json!("maybe"));
    }

    #[test]
    fn coerce_string_to_number_for_float_schema() {
        let schema = make_schema(serde_json::json!({
            "temperature": {"type": "number"},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"temperature": "0.7"}),
            &schema,
        );
        // f64 → Number round-trips through serde_json::Number::from_f64.
        let n = coerced["temperature"].as_f64().expect("number");
        assert!((n - 0.7).abs() < 1e-9);
    }

    #[test]
    fn coerce_leaves_string_fields_alone() {
        let schema = make_schema(serde_json::json!({
            "query": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"query": "50", "count": "50"}),
            &schema,
        );
        // The string-typed field must NOT be turned into a number even
        // though "50" parses cleanly — schema is the source of truth.
        assert_eq!(coerced["query"], serde_json::json!("50"));
        assert_eq!(coerced["count"], serde_json::json!(50));
    }

    #[test]
    fn coerce_leaves_unparseable_strings_alone() {
        let schema = make_schema(serde_json::json!({
            "max_iterations": {"type": "integer"},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"max_iterations": "fifty"}),
            &schema,
        );
        // Unparseable values pass through so the strict serde path
        // produces the canonical "invalid type" error rather than us
        // silently dropping the value.
        assert_eq!(coerced, serde_json::json!({"max_iterations": "fifty"}));
    }

    #[test]
    fn coerce_treats_nullable_integer_as_integer() {
        // `Option<usize>` renders as `{"type": ["integer", "null"]}`.
        // The non-null branch is unambiguous, so coercion still applies.
        let schema = make_schema(serde_json::json!({
            "max_iterations": {"type": ["integer", "null"]},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"max_iterations": "20"}),
            &schema,
        );
        assert_eq!(coerced, serde_json::json!({"max_iterations": 20}));
    }

    #[test]
    fn coerce_skips_ambiguous_multi_type_schemas() {
        // If the schema genuinely accepts both string and integer, leave
        // the value alone — coercion would discard the model's chosen
        // representation. Multi-type schemas wider than `[T, null]` are
        // ambiguous.
        let schema = make_schema(serde_json::json!({
            "value": {"type": ["integer", "string"]},
        }));
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"value": "42"}),
            &schema,
        );
        assert_eq!(coerced, serde_json::json!({"value": "42"}));
    }

    #[test]
    fn coerce_passes_through_object_without_properties() {
        // No schema info → no coercion. Mirrors the safe path for tools
        // that ship a schema without explicit `properties` (e.g. when
        // the args type is `serde_json::Value`).
        let schema = serde_json::json!({"type": "object"});
        let coerced = coerce_string_scalars_at_top_level(
            serde_json::json!({"x": "50"}),
            &schema,
        );
        assert_eq!(coerced, serde_json::json!({"x": "50"}));
    }

    // ---- arg-parse-error enrichment -----------------------------------

    fn hint_for(json: Value, expected_target: &str) -> Option<String> {
        // Drive serde with a real schema mismatch so the helper sees a
        // genuine `serde_json::Error`, not a hand-written string. Skip
        // the coercion pass on purpose — we want the strict-path error.
        #[derive(Debug, Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct UsizeField {
            n: usize,
        }
        #[derive(Debug, Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct BoolField {
            b: bool,
        }
        #[derive(Debug, Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct VecField {
            items: Vec<serde_json::Value>,
        }
        let raw = match expected_target {
            "usize" => serde_json::from_value::<UsizeField>(json).unwrap_err(),
            "bool" => serde_json::from_value::<BoolField>(json).unwrap_err(),
            "sequence" => serde_json::from_value::<VecField>(json).unwrap_err(),
            _ => panic!("unknown target {expected_target}"),
        };
        Some(enrich_arg_parse_error_message(&raw))
    }

    #[test]
    fn enrich_appends_integer_hint_for_string_encoded_int() {
        let msg = hint_for(serde_json::json!({"n": "50"}), "usize").unwrap();
        assert!(
            msg.contains("Did you mean the integer 50"),
            "expected integer hint, got: {msg}"
        );
        assert!(msg.contains("Resend without quotes"));
    }

    #[test]
    fn enrich_appends_boolean_hint_for_string_encoded_bool() {
        let msg = hint_for(serde_json::json!({"b": "True"}), "bool").unwrap();
        assert!(
            msg.contains("Did you mean true"),
            "expected boolean hint, got: {msg}"
        );
    }

    #[test]
    fn enrich_appends_sequence_hint_for_string_in_array_slot() {
        let xml_soup =
            "\n<ref>{\"kind\":\"file\",\"path\":\"x.md\"}</ref></artifact></file_write>";
        let msg = hint_for(
            serde_json::json!({"items": xml_soup}),
            "sequence",
        )
        .unwrap();
        assert!(
            msg.contains("Expected a JSON array"),
            "expected sequence hint, got: {msg}"
        );
    }

    #[test]
    fn enrich_passes_through_unrecognised_errors_unchanged() {
        // Errors that don't match a known pattern (e.g. missing field)
        // must surface verbatim; making up a hint would mislead.
        #[derive(Debug, Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct R {
            n: usize,
        }
        let err = serde_json::from_value::<R>(serde_json::json!({})).unwrap_err();
        let raw = err.to_string();
        let enriched = enrich_arg_parse_error_message(&err);
        assert_eq!(enriched, raw);
    }

    #[test]
    fn flatten_tagged_oneof_passes_through_single_struct_schemas() {
        // A non-enum schema (the EchoTool shape) has no oneOf and
        // should pass through verbatim.
        let raw = serde_json::json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        });
        let out = flatten_tagged_oneof_schema(raw.clone());
        assert_eq!(out, raw);
    }

    struct EchoTool;

    #[async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo arguments back as text"
        }

        fn parameters_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            })
        }

        async fn execute(
            &self,
            _call_id: &str,
            args: Value,
            _signal: CancellationToken,
            _update: ToolUpdateSink,
        ) -> Result<ToolResult, ToolError> {
            let text = args
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ToolResult {
                content: vec![ToolResultBlock::Text(TextContent { text })],
                is_error: false,
                details: Value::Null,
                terminate: false,
                narration: None,
            })
        }
    }

    #[test]
    fn registry_lookup() {
        let registry = ToolRegistry::new().with(Arc::new(EchoTool));
        assert!(registry.get("echo").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.len(), 1);
    }

    struct NamedTool(&'static str);

    #[async_trait]
    impl AgentTool for NamedTool {
        fn name(&self) -> &str {
            self.0
        }

        fn description(&self) -> &str {
            "named"
        }

        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            _call_id: &str,
            _args: Value,
            _signal: CancellationToken,
            _update: ToolUpdateSink,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("ok"))
        }
    }

    #[test]
    fn registry_preserves_registration_order() {
        let mut registry = ToolRegistry::new()
            .with(Arc::new(NamedTool("message_result")))
            .with(Arc::new(NamedTool("message_ask")))
            .with(Arc::new(NamedTool("plan")));

        registry.register(Arc::new(NamedTool("message_result")));

        assert_eq!(
            registry.names(),
            vec!["message_result", "message_ask", "plan"]
        );
        assert_eq!(
            registry.iter().map(|tool| tool.name()).collect::<Vec<_>>(),
            vec!["message_result", "message_ask", "plan"]
        );
    }

    #[tokio::test]
    async fn echo_tool_executes() {
        let tool = EchoTool;
        let (tx, _rx) = mpsc::unbounded_channel();
        let result = tool
            .execute(
                "call_1",
                serde_json::json!({"text": "hi"}),
                CancellationToken::new(),
                tx,
            )
            .await
            .unwrap();
        let ToolResultBlock::Text(t) = &result.content[0] else {
            panic!("expected text")
        };
        assert_eq!(t.text, "hi");
    }

    // ---- end-to-end execute path ----------------------------------------
    //
    // The blanket impl `AgentTool::execute` for `TypedAgentTool` invokes
    // (1) strip_top_level_nulls, (2) coerce_string_scalars_at_top_level,
    // (3) serde_json::from_value, (4) enrich_arg_parse_error_message.
    // Exercise the full path with a tool whose args mix the scalar types
    // qwen 3.5 Flash routinely encodes as strings.

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct CoercibleArgs {
        max_iterations: usize,
        full_page: bool,
        temperature: f32,
        label: String,
    }

    struct CoercibleTool;

    #[async_trait]
    impl TypedAgentTool for CoercibleTool {
        type Args = CoercibleArgs;
        fn name(&self) -> &str {
            "coercible"
        }
        fn description(&self) -> &str {
            "fixture"
        }
        async fn run(
            &self,
            _call_id: &str,
            args: Self::Args,
            _signal: CancellationToken,
            _update: ToolUpdateSink,
        ) -> Result<ToolResult, ToolError> {
            // Echo the parsed values so the test can assert coercion happened.
            Ok(ToolResult::text(format!(
                "max_iterations={} full_page={} temperature={} label={}",
                args.max_iterations, args.full_page, args.temperature, args.label
            )))
        }
    }

    #[tokio::test]
    async fn execute_coerces_string_encoded_scalars_end_to_end() {
        // The four shapes qwen emits in practice — strings where the
        // schema declares integers, booleans, or floats. Each must pass
        // the validator after coercion and reach the tool's `run` with
        // the typed value.
        let tool = CoercibleTool;
        let (tx, _rx) = mpsc::unbounded_channel();
        let result = AgentTool::execute(
            &tool,
            "call_1",
            serde_json::json!({
                "max_iterations": "50",
                "full_page": "True",
                "temperature": "0.7",
                "label": "actual string",
            }),
            CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();
        let ToolResultBlock::Text(t) = &result.content[0] else {
            panic!("expected text result");
        };
        assert!(
            t.text.contains("max_iterations=50"),
            "integer coercion missing: {}",
            t.text
        );
        assert!(
            t.text.contains("full_page=true"),
            "boolean coercion missing: {}",
            t.text
        );
        assert!(
            t.text.contains("temperature=0.7"),
            "float coercion missing: {}",
            t.text
        );
        assert!(
            t.text.contains("label=actual string"),
            "string field must NOT be coerced: {}",
            t.text
        );
        assert!(!result.is_error, "execute must succeed after coercion");
    }

    #[tokio::test]
    async fn execute_appends_self_correcting_hint_on_unrecoverable_string_int() {
        // The string "fifty" cannot be coerced to an integer; the
        // validator rejects, and the runtime appends a hint only when
        // it's accurate. Here the hint must NOT claim "Did you mean
        // the integer fifty" — there is no such number — so the
        // enrichment should pass through.
        let tool = CoercibleTool;
        let (tx, _rx) = mpsc::unbounded_channel();
        let result = AgentTool::execute(
            &tool,
            "call_2",
            serde_json::json!({
                "max_iterations": "fifty",
                "full_page": true,
                "temperature": 0.1,
                "label": "x",
            }),
            CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();
        assert!(result.is_error, "expected validator rejection");
        let ToolResultBlock::Text(t) = &result.content[0] else {
            panic!("expected text result");
        };
        assert!(
            t.text.starts_with("coercible: invalid arguments:"),
            "preserve canonical error prefix: {}",
            t.text
        );
        assert!(
            !t.text.contains("Did you mean the integer fifty"),
            "must not invent a hint when the value cannot parse: {}",
            t.text
        );
    }

    // Fixture for prepare_arguments wiring — mimics a tagged-enum
    // tool like `browser_navigate` where the discriminator field
    // must be present but can be inferred from a variant-unique
    // field.
    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(tag = "action", rename_all = "snake_case")]
    enum TaggedArgs {
        Open { url: String },
        Reload {},
    }

    struct TaggedTool;

    #[async_trait]
    impl TypedAgentTool for TaggedTool {
        type Args = TaggedArgs;
        fn name(&self) -> &str {
            "tagged_fixture"
        }
        fn description(&self) -> &str {
            "fixture"
        }
        fn prepare_arguments(&self, args: Value) -> Value {
            // Same inference shape as BrowserNavigateTool's real
            // override: if `action` is missing and `url` is present,
            // assume `open`.
            let Value::Object(mut obj) = args else {
                return args;
            };
            if !obj.contains_key("action") && obj.contains_key("url") {
                obj.insert("action".to_string(), Value::String("open".to_string()));
            }
            Value::Object(obj)
        }
        async fn run(
            &self,
            _call_id: &str,
            args: Self::Args,
            _signal: CancellationToken,
            _update: ToolUpdateSink,
        ) -> Result<ToolResult, ToolError> {
            let label = match args {
                TaggedArgs::Open { url } => format!("open:{url}"),
                TaggedArgs::Reload {} => "reload".to_string(),
            };
            Ok(ToolResult::text(label))
        }
    }

    #[tokio::test]
    async fn execute_runs_prepare_arguments_before_typed_deser() {
        // Reproduces the dominant `browser_navigate` failure: the
        // model emits a tagged-enum call without the discriminator.
        // With `prepare_arguments` wired into the blanket execute,
        // the missing `action` is inferred from `url` and the call
        // reaches `run` as the `Open` variant.
        let tool = TaggedTool;
        let (tx, _rx) = mpsc::unbounded_channel();
        let result = AgentTool::execute(
            &tool,
            "call_1",
            serde_json::json!({"url": "https://example.com"}),
            CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();
        let ToolResultBlock::Text(t) = &result.content[0] else {
            panic!("expected text result");
        };
        assert!(!result.is_error, "execute must succeed after action inference");
        assert_eq!(t.text, "open:https://example.com");
    }

    #[tokio::test]
    async fn execute_prepare_arguments_does_not_override_explicit_action() {
        let tool = TaggedTool;
        let (tx, _rx) = mpsc::unbounded_channel();
        let result = AgentTool::execute(
            &tool,
            "call_2",
            serde_json::json!({"action": "reload"}),
            CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();
        let ToolResultBlock::Text(t) = &result.content[0] else {
            panic!("expected text result");
        };
        assert!(!result.is_error);
        assert_eq!(t.text, "reload");
    }
}
