//! Tool surface.
//!
//! `AgentTool` is the only contract the loop knows about. Tools own their
//! parameter schema, validation, and execution. The loop dispatches and
//! emits events.
//!
//! Termination is a tool decision: a tool result with `terminate: true`
//! ends the run if every tool in the batch agrees (unanimous). One tool
//! wanting to stop does not stop the batch.

mod argument_normalization;
mod schema;
use argument_normalization::{
    coerce_string_scalars_at_top_level, enrich_arg_parse_error_message, strip_top_level_nulls,
};

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

    /// A recoverable rejection at the typed tool-argument boundary.
    ///
    /// The text remains part of model-visible history so the model can
    /// repair its next call. The structured details let observers and
    /// guardrails distinguish that expected correction from an operational
    /// tool failure without parsing provider- or serde-authored prose.
    pub fn argument_validation_error(tool: &str, text: impl Into<String>) -> Self {
        let mut result = Self::error(text);
        result.details = serde_json::json!({
            "kind": "tool_argument_validation",
            "recoverable": true,
            "display_hidden": true,
            "tool": tool,
        });
        result
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

    /// Whether a non-fatal failure of this tool in a batch should stop
    /// dependent sibling work. Default `false` — failures are isolated
    /// and siblings run to completion. Tools where one failure makes
    /// sibling work meaningless (a prerequisite state update that gates
    /// later work, a shell step that gates `npm test`) opt in by
    /// overriding this to `true`.
    ///
    /// Cancelled siblings produce a `ToolResult` with
    /// `is_error: true, content: "aborted because sibling 'X' failed"`
    /// — they remain context events the next turn can react to, never
    /// `LoopError`s. Sibling-abort therefore never ends the run on its
    /// own; the unanimous-vote termination rule is preserved.
    ///
    /// In parallel mode, cancellation is cooperative: tools must check
    /// `signal.is_cancelled()` (or wrap blocking work in `select!`) to
    /// honor the cancel promptly. In sequential mode, later siblings
    /// never start and receive typed not-executed results instead.
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

    /// Whether this tool's `terminate` vote is included in the
    /// unanimous-vote tally that decides whether the batch ends the
    /// run.
    ///
    /// Default `true`: every tool's vote counts. The batch terminates
    /// only when *every* tool that opts in voted `terminate: true`.
    ///
    /// Lightweight status-only tools (progress notes, hidden journals)
    /// override to `false`. The runtime then ignores their vote
    /// entirely — they are neither a "yes" nor a "no" — so a model that
    /// emits a terminating delivery call alongside an advisory status
    /// call in the same batch can still terminate. An all-advisory batch
    /// (no tool with this flag set to `true` voted yes) does NOT
    /// terminate, preserving the contract that progress notes never end
    /// a run on their own.
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
// Tag-dispatched tools (a single tool with several modes selected by a
// discriminator field, e.g. `edit(op="insert"|"replace"|"delete")`) use a
// `#[serde(tag = "...")]` enum as `Args`; serde routes the discriminator
// natively, so the "unknown field `op`" failure mode that motivated this
// trait can
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

    /// Whether a non-fatal failure of this tool in a batch stops
    /// dependent siblings. Default false.
    fn aborts_siblings_on_error(&self) -> bool {
        false
    }

    /// Whether this tool consumes a slot from
    /// `LoopConfig::max_tool_calls_per_turn`. Default true.
    fn counts_toward_tool_call_limit(&self) -> bool {
        true
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
        schema::parameters_schema::<T::Args>()
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
        // Provider adapters may flatten action schemas for restricted dialects.
        // Drop unset fields before serde checks the selected action. Tool-owned
        // argument preparation still runs before this generic normalization.
        let prepared = AgentTool::prepare_arguments(self, args);
        let stripped = strip_top_level_nulls(prepared);
        // Coerce string-encoded scalars (integers, numbers, booleans)
        // to their declared types BEFORE serde validation runs. Some
        // providers (notably the "auto-when-forced" class) emit
        // tool-call arguments where every value is a JSON string —
        // `{"item_count": "50"}` instead of `{"item_count": 50}`.
        // The strict serde path rejects every such call, wasting a turn
        // per field; coercion converts the obvious case in-place using
        // the tool's own schema as the source of truth.
        let schema = AgentTool::parameters_schema(self);
        let coerced = coerce_string_scalars_at_top_level(stripped, &schema);
        let parsed: T::Args = match serde_json::from_value(coerced) {
            Ok(v) => v,
            Err(e) => {
                let tool_name = TypedAgentTool::name(self);
                return Ok(ToolResult::argument_validation_error(
                    tool_name,
                    format!(
                        "{}: invalid arguments: {}",
                        tool_name,
                        enrich_arg_parse_error_message(&e),
                    ),
                ));
            }
        };
        TypedAgentTool::run(self, call_id, parsed, signal, update).await
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
    pub fn identity_policies(
        &self,
    ) -> std::collections::HashMap<String, crate::tool_identity::ToolIdentityPolicy> {
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
mod tests;

#[cfg(test)]
mod execution_tests;
