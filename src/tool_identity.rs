//! `ToolIdentityPolicy` — typed declaration each tool gives the runtime
//! about how to recognize "two calls of this tool that are the same
//! operation on the same target".
//!
//! ## Why this lives next to tool definitions
//!
//! The `SemanticLoopDetector` and any future plugin that needs to ask
//! "is this call a repeat of the last one?" must agree with the tool's
//! own dispatch contract. The historical pattern — a hand-curated
//! `match tool_name { ... }` inside the detector — drifts every time a
//! new action-dispatched tool is registered: the detector's allowlist
//! and the tool's `#[serde(tag = "action")]` enum live in different
//! files with no compile-time link. The `office` tool shipped without
//! the matching allowlist update; every `office.pdf_fields` repeat
//! collapsed to `(office, default)` instead of `(office, pdf_fields)`,
//! so the recovery message went generic.
//!
//! `ToolIdentityPolicy` is the single source of truth for that
//! identity contract. Each tool declares it on the `AgentTool` trait;
//! the detector reads the declaration from the `ToolRegistry`. Adding
//! a tool can no longer drift the detector because the detector reads
//! the tool's own declaration.
//!
//! ## Three independent declarations
//!
//! - `operation_arg`: top-level arg whose value names the operation
//!   for action-dispatched tools. Two distinct operations of the same
//!   tool get distinct identities.
//! - `target`: how to extract a "persistent target" string (file
//!   path, URL, shell command prefix). Used both as a display token in
//!   recovery messages and as the key for per-target error counters
//!   that survive intervening successes.
//! - `args_key_fn`: tool-specific override for the full identity
//!   string used to recognize repeats. Defaults to
//!   `target={tool_name}:{target}` when only `target` is declared,
//!   and to a canonical-JSON hash of the normalized args when nothing
//!   else applies.

use serde_json::Value;

/// Tool-specific target extractor. Returns the "persistent target" of
/// a call (file path, URL, shell-command prefix) — the thing whose
/// repeat would mean "the same work is being attempted again." `None`
/// means the call has no persistent target (e.g. `office.pdf_fields`
/// — the action itself is what repeats).
pub type TargetFn = fn(&Value) -> Option<String>;

/// Tool-specific full-identity extractor. Returns the opaque identity
/// string the detector compares for equality, or `None` to fall back
/// to the default `target={tool}:{target}` / canonical-JSON paths.
pub type ArgsKeyFn = fn(&Value) -> Option<String>;

/// How to extract the persistent target of a call.
#[derive(Debug, Clone, Copy)]
pub enum TargetExtractor {
    /// The target is the trimmed string value at this top-level arg
    /// (e.g. `"path"` for `file_*`, `"url"` for `browser_navigate`).
    StringArg(&'static str),
    /// Tool-specific extractor for composite targets (e.g. `shell`'s
    /// `action:command-prefix`).
    Custom(TargetFn),
}

/// Typed identity contract for one tool. All fields default to `None`
/// — a tool that does not override `identity_policy` is treated as
/// "single operation, no persistent target, opaque args" (matches the
/// historical fall-through behavior for unknown tools).
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolIdentityPolicy {
    /// Top-level argument whose string value names the operation when
    /// the tool dispatches on `action` / `mode` / similar. The
    /// detector pairs this with `tool_name` so two distinct operations
    /// of the same tool do not collide into one repeat signature.
    pub operation_arg: Option<&'static str>,
    /// How to extract the persistent target string. `None` means the
    /// tool has no persistent target — every call is identified by
    /// its full args.
    pub target: Option<TargetExtractor>,
    /// Tool-specific full-identity extractor. Overrides the default
    /// `target={tool}:{target}` composition when set.
    pub args_key_fn: Option<ArgsKeyFn>,
}

impl ToolIdentityPolicy {
    pub const fn new() -> Self {
        Self {
            operation_arg: None,
            target: None,
            args_key_fn: None,
        }
    }

    /// Declare the top-level argument that discriminates operations
    /// for this tool (e.g. `"action"` for `office`, `shell`, `plan`).
    pub const fn with_operation_arg(mut self, arg: &'static str) -> Self {
        self.operation_arg = Some(arg);
        self
    }

    /// Declare a top-level string argument as the persistent target
    /// (e.g. `"path"` for `file_*`, `"url"` for `browser_navigate`).
    pub const fn with_target_arg(mut self, arg: &'static str) -> Self {
        self.target = Some(TargetExtractor::StringArg(arg));
        self
    }

    /// Provide a tool-specific target extractor for composite targets
    /// (e.g. `shell`'s `action:command-prefix`).
    pub const fn with_target_fn(mut self, f: TargetFn) -> Self {
        self.target = Some(TargetExtractor::Custom(f));
        self
    }

    /// Provide a tool-specific full-identity extractor. Use this only
    /// when the identity is more than `target={tool}:{target}` — e.g.
    /// `plan`'s `action+status+next_phase_id` composite.
    pub const fn with_args_key_fn(mut self, f: ArgsKeyFn) -> Self {
        self.args_key_fn = Some(f);
        self
    }
}

/// Operation key for a call. Returns `"default"` when the tool has
/// not declared an `operation_arg` — same value the old hardcoded
/// fall-through used, so downstream messaging is unchanged for
/// non-action tools.
pub fn extract_operation_key(policy: &ToolIdentityPolicy, args: &Value) -> String {
    match policy.operation_arg {
        Some(arg) => args
            .get(arg)
            .and_then(Value::as_str)
            .unwrap_or("default")
            .to_string(),
        None => "default".to_string(),
    }
}

/// Persistent target string for a call (file path, URL, shell command
/// prefix). `None` means the tool has no persistent target — the
/// detector then keys per-target counters on the full args identity
/// instead.
pub fn extract_target(policy: &ToolIdentityPolicy, args: &Value) -> Option<String> {
    match &policy.target {
        Some(TargetExtractor::StringArg(name)) => args
            .get(name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        Some(TargetExtractor::Custom(f)) => f(args),
        None => None,
    }
}

/// Full args/identity key for two calls of a tool. `None` means the
/// tool declares neither a custom args-key fn nor a persistent target
/// — the caller should fall back to its own canonical-JSON identity
/// (see `SemanticLoopDetector::semantic_args_key`).
///
/// `tool_name` is threaded through so the `target`-shaped default
/// emits the historical `target={tool_name}:{target}` format without
/// duplicating the tool name on every declaration.
pub fn extract_args_key(
    policy: &ToolIdentityPolicy,
    tool_name: &str,
    args: &Value,
) -> Option<String> {
    if let Some(f) = policy.args_key_fn {
        return f(args);
    }
    extract_target(policy, args).map(|target| format!("target={tool_name}:{target}"))
}
