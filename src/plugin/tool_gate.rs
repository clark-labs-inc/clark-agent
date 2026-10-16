use super::*;

// ─── ToolGate ──────────────────────────────────────────────────────

/// Read-only loop state handed to a `ToolGate` so its decision is a
/// pure function of observables, not of internal flag bookkeeping.
/// New fields are additive — gates that don't care can ignore them.
pub struct ToolGateContext<'a> {
    /// Zero-indexed iteration within the current run. The very first
    /// LLM call after the user message has `iteration == 0`. Increments
    /// once per `stream_assistant_response`.
    pub iteration: usize,
    /// Full message history that will be sent on the next request,
    /// after any `ContextTransform` reshaping. Use this to derive
    /// signals like "have we seen a terminator yet" or "how many tool
    /// results in a row didn't make progress".
    pub messages: &'a [AgentMessage],
    /// Conversation identifier when the host runtime knows one (a
    /// session runner threads it through). `None` for embeddings of the
    /// loop that don't carry conversation identity (tests, isolated
    /// subagent runs). Gates can use this for diagnostics or
    /// conversation-scoped policy.
    pub conversation_id: Option<&'a str>,
    /// Names of every tool the loop is about to advertise on the next
    /// request, in registration order. Lets gates compute denylist-style
    /// allowlists ("everything except these terminators") without
    /// hardcoding the catalog or extending the trait. Empty in tests
    /// that don't care about the universe.
    pub available_tool_names: &'a [&'a str],
}

/// How a tool gate should compose with explicit recovery owners.
///
/// Required gates encode typed boundaries: phase capability, workflow
/// ownership, delivery repair, scenario contracts, and similar constraints.
/// Advisory gates encode pressure or nudges: budget wrap-up and terminal
/// recovery. When a required recovery owner says it has live repair work,
/// advisory gates may be ignored for that turn so they cannot erase the
/// tools needed to perform the repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolGateClass {
    Required,
    Advisory,
}

/// Per-turn allowlist of tool names the model may invoke.
///
/// Returning `Some(set)` means: for the *very next* LLM call, narrow
/// the advertised tools to those whose names appear in `set`. Every
/// other tool the agent has access to is omitted from that one
/// request. `None` means no narrowing — the loop sends all tools.
///
/// Composition across multiple gates: the loop intersects every
/// `Some` allowlist; absent (`None`) gates do not constrain. If multiple
/// non-empty gate allowlists conflict to the empty set, the loop repairs
/// the composition by choosing the highest-priority gate and emits a
/// typed conflict event. Gates that own urgent recovery states should
/// override [`ToolGate::conflict_priority`].
///
/// Single-shot semantics emerge from the trigger condition, not from
/// internal mutability: a gate that fires only on `iteration == 0`
/// is naturally single-shot per run. Conversation-scoped gates should
/// keep their cross-run state in an external store, not in the plugin
/// instance.
#[async_trait]
pub trait ToolGate: Plugin {
    async fn next_turn_tool_allowlist(
        &self,
        ctx: ToolGateContext<'_>,
    ) -> Option<std::collections::HashSet<String>>;

    /// This gate's specific reason for denying `tool_name` in the given
    /// context. The runtime queries every gate after a hidden-tool call
    /// so the error message names the actual narrower instead of guessing
    /// from the intersected allowlist's shape — that guess sent the model
    /// to repair the wrong gate (e.g. a `delivery_repair_gate` strip read
    /// as a `capability_gate` phase mismatch and triggered futile
    /// plan-updates until wall-clock timeout).
    ///
    /// Default: `None` — the runtime falls back to its shape-based
    /// heuristic. Return `Some(reason)` only when this gate is actively
    /// narrowing in a way that excludes `tool_name` in this context.
    async fn denial_reason(&self, _tool_name: &str, _ctx: ToolGateContext<'_>) -> Option<String> {
        None
    }

    fn conflict_priority(&self) -> i32 {
        0
    }

    fn tool_gate_class(&self) -> ToolGateClass {
        ToolGateClass::Required
    }

    fn suppresses_advisory_gates(&self, _ctx: ToolGateContext<'_>) -> bool {
        false
    }
}

