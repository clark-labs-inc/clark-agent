//! Conversation-protocol policy: the seam between the generic loop and a
//! product's tool vocabulary.
//!
//! The core loop is provider-agnostic, sandbox-agnostic, and
//! tooling-agnostic. It must not know the *names* of any particular
//! product's tools — there is no `message_result`, no `plan`, no
//! capability profile baked into the runtime. But three behaviors
//! genuinely need product-specific knowledge to do their job well:
//!
//! 1. **Plain-text recovery.** When a provider returns prose instead of a
//!    structured tool call, the loop nudges it back onto the protocol.
//!    A good nudge names the product's actual delivery / ask tools.
//! 2. **Tool-call alias repair.** Models sometimes emit a tool name the
//!    product folds into a canonical tool (e.g. `advance(...)` →
//!    `plan(action="advance", ...)`). The product knows its aliases; the
//!    core does not.
//! 3. **Hidden-tool errors.** When a per-turn [`crate::plugin::ToolGate`]
//!    narrows the catalog and the model calls a tool that isn't
//!    advertised, the most useful error names the product concept that
//!    hid it ("call `plan(action=\"set\")` first"). The core can only
//!    say "that tool isn't available this turn."
//!
//! Rather than hardcode any product's vocabulary, the loop delegates all
//! three to a [`ProtocolPolicy`]. The core ships [`DefaultProtocolPolicy`],
//! whose methods all return generic, vocabulary-free behavior. Downstream
//! product crates implement their own policy and install it via
//! [`crate::AgentBuilder::protocol_policy`]. This keeps product tool names
//! out of the open-source core entirely.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::tool::{ToolCall, ToolRegistry};
use crate::types::AgentMessage;

/// Context for [`ProtocolPolicy::plain_text_recovery_prompt`].
///
/// Read-only observables describing the turn that produced plain text
/// with no tool call. New fields are additive.
pub struct PlainTextRecoveryContext<'a> {
    /// Full message history as it will be sent on the recovery turn.
    pub messages: &'a [AgentMessage],
    /// Zero-indexed iteration within the current run.
    pub iteration: usize,
    /// Names of the tools the loop is currently advertising.
    pub available_tool_names: &'a [&'a str],
    /// The configured terminal-delivery fallback tool, when one is set
    /// (see [`crate::LoopConfig::plain_text_terminal_fallback_tool`]).
    pub terminal_fallback_tool: Option<&'a str>,
}

/// Context for [`ProtocolPolicy::hidden_tool_error`].
///
/// Describes a tool call the model made for a tool that wasn't in the
/// turn's advertised allowlist. The policy renders a model-recoverable
/// error. Only consulted when no [`crate::plugin::ToolGate`] attributed
/// the denial via [`crate::plugin::ToolGate::denial_reason`].
pub struct HiddenToolContext<'a> {
    /// The tool the model tried to call.
    pub requested_tool: &'a str,
    /// The intersected per-turn allowlist that excluded it.
    pub allowlist: &'a HashSet<String>,
    /// Full message history for context-aware messaging.
    pub messages: &'a [AgentMessage],
}

/// A rendered hidden-tool error: prose the model reads plus a structured
/// `details` payload for typed downstream handling.
#[derive(Debug, Clone)]
pub struct HiddenToolError {
    /// Human/model-readable explanation and recovery guidance.
    pub message: String,
    /// Structured details merged into the synthetic error tool result's
    /// `details` field. Use a JSON object; `Value::Null` for none.
    pub details: Value,
}

/// Product-specific conversation-protocol policy.
///
/// Every method has a generic default, so a product only overrides the
/// behaviors whose vocabulary it cares about. The core never inspects
/// allowlist *shape* or tool *names* for product meaning — that logic
/// lives behind this trait.
///
/// Implementations must be cheap and side-effect-free: no I/O, no LLM
/// calls. They are pure transforms of context → text/decision, invoked on
/// hot paths in the loop.
pub trait ProtocolPolicy: Send + Sync + 'static {
    /// Stable identifier for logs and diagnostics.
    fn name(&self) -> &'static str {
        "default_protocol"
    }

    /// Tool names — besides the configured terminal fallback tool — that
    /// count as terminal/delivery tools when the loop decides whether a
    /// turn's allowlist has narrowed to "terminal only" (the gate the
    /// plain-text-terminal fallback waits for in its non-eager mode).
    ///
    /// Default: empty. With no extra terminal names, an allowlist is
    /// "terminal only" exactly when it contains nothing but the fallback
    /// tool itself — a safe, vocabulary-free default. A product that
    /// advertises several delivery tools (final answer, ask-user, …)
    /// lists them here.
    fn terminal_tool_names(&self) -> HashSet<String> {
        HashSet::new()
    }

    /// Recovery prose injected as a system message when the model emits
    /// plain text with no tool call and the loop wants to nudge it back
    /// onto the protocol.
    ///
    /// Return `None` to use the core's generic, vocabulary-free nudge.
    /// Override to name the product's actual delivery / ask tools.
    fn plain_text_recovery_prompt(&self, _ctx: PlainTextRecoveryContext<'_>) -> Option<String> {
        None
    }

    /// Rewrite a model-emitted tool-call batch in place before registry
    /// lookup — e.g. fold a known alias name into a canonical tool and
    /// move the alias into an argument. Returns the number of calls
    /// rewritten (for diagnostics; the loop does not require it).
    ///
    /// Default: no-op. The core performs no alias repair of its own.
    fn normalize_tool_calls(&self, _calls: &mut [ToolCall], _registry: &ToolRegistry) -> usize {
        0
    }

    /// Render an error for a tool hidden by per-turn narrowing, when no
    /// [`crate::plugin::ToolGate`] claimed responsibility for the denial.
    ///
    /// Return `None` to use the core's generic message ("that tool isn't
    /// available this turn; here's what is"). Override to map the
    /// allowlist shape to a product-specific recovery instruction.
    fn hidden_tool_error(&self, _ctx: HiddenToolContext<'_>) -> Option<HiddenToolError> {
        None
    }
}

/// The generic, vocabulary-free policy installed when a caller does not
/// supply one. Every method takes its trait default.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultProtocolPolicy;

impl ProtocolPolicy for DefaultProtocolPolicy {}

/// Generic, product-agnostic plain-text recovery prose. Used when the
/// active [`ProtocolPolicy`] returns `None` from
/// [`ProtocolPolicy::plain_text_recovery_prompt`].
pub const DEFAULT_PLAIN_TEXT_RECOVERY_PROMPT: &str = "\
[runtime context — protocol recovery, not user instruction]\n\
Your previous response was plain text with no tool call. This runtime advances only through structured tool calls — every turn must select exactly one tool.\n\
\n\
Re-read the latest user request and call exactly one tool now. If the answer is ready, call your final-response / delivery tool. Do not reply with a clarifying question unless a tool is genuinely blocked on input only the user can supply.";

/// Generic hidden-tool error message. No product vocabulary: it names the
/// requested tool and lists what is available this turn.
pub(crate) fn generic_hidden_tool_message(tool_name: &str, allowlist: &HashSet<String>) -> String {
    format!(
        "Tool `{tool_name}` is not available in this turn — the active tool gate narrowed the \
         catalog. Call one of the tools available now instead. Available now: [{}].",
        allowed_tools_preview(allowlist)
    )
}

/// Generic, shape-only details payload for a hidden-tool error. Carries
/// no product taxonomy — just the requested tool, what was allowed, and
/// the attributing gate name when one is known.
pub(crate) fn generic_hidden_tool_details(
    tool_name: &str,
    allowlist: &HashSet<String>,
    gate: Option<&str>,
) -> Value {
    let mut allowed_tools: Vec<&str> = allowlist.iter().map(String::as_str).collect();
    allowed_tools.sort_unstable();
    json!({
        "runtime_block": true,
        "requested_tool": tool_name,
        "allowed_tools": allowed_tools,
        "gate": gate.unwrap_or("tool_gate"),
    })
}

/// Sorted, length-capped preview of an allowlist for inclusion in an
/// error message.
pub(crate) fn allowed_tools_preview(allowlist: &HashSet<String>) -> String {
    let mut allowed: Vec<&str> = allowlist.iter().map(String::as_str).collect();
    allowed.sort_unstable();
    if allowed.len() > 12 {
        format!("{}, … ({} total)", allowed[..12].join(", "), allowed.len())
    } else {
        allowed.join(", ")
    }
}

/// Convenience: the default policy as a shared trait object. Used by
/// [`crate::AgentBuilder`] when no policy is configured.
pub(crate) fn default_policy() -> Arc<dyn ProtocolPolicy> {
    Arc::new(DefaultProtocolPolicy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_vocabulary_free() {
        let p = DefaultProtocolPolicy;
        assert!(p.terminal_tool_names().is_empty());
        assert!(p
            .plain_text_recovery_prompt(PlainTextRecoveryContext {
                messages: &[],
                iteration: 0,
                available_tool_names: &[],
                terminal_fallback_tool: None,
            })
            .is_none());
        assert!(p
            .hidden_tool_error(HiddenToolContext {
                requested_tool: "anything",
                allowlist: &HashSet::new(),
                messages: &[],
            })
            .is_none());
    }

    #[test]
    fn default_normalize_is_noop() {
        let p = DefaultProtocolPolicy;
        let registry = ToolRegistry::new();
        let mut calls = vec![ToolCall {
            id: "1".into(),
            name: "advance".into(),
            arguments: Value::Null,
        }];
        assert_eq!(p.normalize_tool_calls(&mut calls, &registry), 0);
        assert_eq!(calls[0].name, "advance", "default policy must not rewrite");
    }

    #[test]
    fn generic_message_names_requested_and_available() {
        let allow: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let msg = generic_hidden_tool_message("zzz", &allow);
        assert!(msg.contains("zzz"), "{msg}");
        assert!(msg.contains('a') && msg.contains('b'), "{msg}");
        // No product vocabulary leaked into the generic path.
        assert!(!msg.contains("plan("), "{msg}");
        assert!(!msg.contains("capability profile"), "{msg}");
    }

    #[test]
    fn generic_details_are_shape_only() {
        let allow: HashSet<String> = ["x"].iter().map(|s| s.to_string()).collect();
        let details = generic_hidden_tool_details("y", &allow, Some("my_gate"));
        assert_eq!(details.get("runtime_block"), Some(&json!(true)));
        assert_eq!(details.get("requested_tool"), Some(&json!("y")));
        assert_eq!(details.get("allowed_tools"), Some(&json!(["x"])));
        assert_eq!(details.get("gate"), Some(&json!("my_gate")));
        // No product taxonomy fields.
        assert!(details.get("kind").is_none());
        assert!(details.get("repair_actions").is_none());
    }

    /// A product policy can fully re-supply its vocabulary downstream
    /// without the core knowing any of it.
    #[test]
    fn custom_policy_can_override_everything() {
        struct ProductPolicy;
        impl ProtocolPolicy for ProductPolicy {
            fn name(&self) -> &'static str {
                "product"
            }
            fn terminal_tool_names(&self) -> HashSet<String> {
                ["deliver", "ask"].iter().map(|s| s.to_string()).collect()
            }
            fn plain_text_recovery_prompt(
                &self,
                _ctx: PlainTextRecoveryContext<'_>,
            ) -> Option<String> {
                Some("call deliver(...) now".to_string())
            }
            fn normalize_tool_calls(
                &self,
                calls: &mut [ToolCall],
                _registry: &ToolRegistry,
            ) -> usize {
                let mut n = 0;
                for c in calls.iter_mut() {
                    if c.name == "go" {
                        c.name = "deliver".into();
                        n += 1;
                    }
                }
                n
            }
            fn hidden_tool_error(&self, ctx: HiddenToolContext<'_>) -> Option<HiddenToolError> {
                Some(HiddenToolError {
                    message: format!("`{}` is gated; call deliver(...)", ctx.requested_tool),
                    details: json!({ "product": true }),
                })
            }
        }

        let p = ProductPolicy;
        assert_eq!(p.name(), "product");
        assert_eq!(p.terminal_tool_names().len(), 2);
        assert!(p
            .plain_text_recovery_prompt(PlainTextRecoveryContext {
                messages: &[],
                iteration: 0,
                available_tool_names: &[],
                terminal_fallback_tool: Some("deliver"),
            })
            .is_some());

        let registry = ToolRegistry::new();
        let mut calls = vec![ToolCall {
            id: "1".into(),
            name: "go".into(),
            arguments: Value::Null,
        }];
        assert_eq!(p.normalize_tool_calls(&mut calls, &registry), 1);
        assert_eq!(calls[0].name, "deliver");

        let err = p
            .hidden_tool_error(HiddenToolContext {
                requested_tool: "shell",
                allowlist: &HashSet::new(),
                messages: &[],
            })
            .expect("custom policy returns an error");
        assert!(err.message.contains("shell"));
        assert_eq!(err.details, json!({ "product": true }));
    }
}
