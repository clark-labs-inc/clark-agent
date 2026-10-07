use super::*;

/// Poll every registered `ToolGate` plugin and intersect their
/// allowlists. Returns `None` when no plugin returned an allowlist
/// (the common case — no narrowing). Returns `Some(set)` when at
/// least one plugin is gating; multiple gates compose by intersection
/// unless their non-empty allowlists conflict to an empty set, in which
/// case the highest-priority gate wins and a typed conflict event is
/// emitted.
/// Resolve the per-turn tool allowlist by composing every registered
/// `ToolGate` plugin (intersection) and emit one
/// [`AgentEvent::ToolGateApplied`] per gate so observers can attribute
/// the final allowlist to specific plugins.
pub(super) async fn collect_tool_allowlist_with_events(
    config: &LoopConfig,
    iteration: usize,
    messages: &[AgentMessage],
) -> Option<std::collections::HashSet<String>> {
    if config.plugins.tool_gate.is_empty() {
        return None;
    }
    let conversation_id = config.conversation_id.as_deref();
    let available_tool_names: Vec<&str> = config.tools.iter().map(|t| t.name()).collect();
    let mut decisions: Vec<GateAllowDecision> = Vec::new();
    for gate in &config.plugins.tool_gate {
        let ctx = crate::plugin::ToolGateContext {
            iteration,
            messages,
            conversation_id,
            available_tool_names: &available_tool_names,
        };
        let decision = gate.next_turn_tool_allowlist(ctx).await;
        emit(
            config,
            AgentEvent::ToolGateApplied {
                iteration,
                plugin: gate.name(),
                allow: decision.as_ref().map(|set| {
                    let mut sorted: Vec<String> = set.iter().cloned().collect();
                    sorted.sort();
                    sorted
                }),
            },
        )
        .await;
        if let Some(set) = decision {
            let suppresses_advisory =
                gate.suppresses_advisory_gates(crate::plugin::ToolGateContext {
                    iteration,
                    messages,
                    conversation_id,
                    available_tool_names: &available_tool_names,
                });
            decisions.push(GateAllowDecision {
                plugin: gate.name(),
                priority: gate.conflict_priority(),
                class: gate.tool_gate_class(),
                suppresses_advisory,
                allow: set,
            });
        }
    }
    let suppression_priority = decisions
        .iter()
        .filter(|decision| decision.suppresses_advisory)
        .map(|decision| decision.priority)
        .max();
    let active_decisions = decisions
        .iter()
        .filter(|decision| {
            !matches!(
                suppression_priority,
                Some(priority)
                    if decision.class == crate::plugin::ToolGateClass::Advisory
                        && decision.priority < priority
            )
        })
        .collect::<Vec<_>>();
    let mut combined: Option<std::collections::HashSet<String>> = None;
    for decision in &active_decisions {
        combined = Some(match combined {
            Some(prev) => prev.intersection(&decision.allow).cloned().collect(),
            None => decision.allow.clone(),
        });
    }
    if combined.as_ref().is_some_and(|allow| allow.is_empty()) {
        let non_empty_decisions = active_decisions
            .iter()
            .filter(|decision| !decision.allow.is_empty())
            .map(|decision| (decision.plugin, decision.priority, decision.allow.clone()))
            .collect::<Vec<_>>();
        let resolved = resolve_empty_tool_gate_intersection(&non_empty_decisions);
        let (chosen_plugin, allow, reason) = match resolved {
            Some((plugin, allow, reason)) => (Some(plugin.to_string()), allow, reason),
            None => (
                None,
                std::collections::HashSet::new(),
                "all gating plugins returned empty allowlists".to_string(),
            ),
        };
        let sorted_allow = sorted_tool_names(&allow);
        emit(
            config,
            AgentEvent::ToolGateConflictResolved {
                iteration,
                plugins: active_decisions
                    .iter()
                    .map(|decision| decision.plugin.to_string())
                    .collect(),
                chosen_plugin,
                allow: sorted_allow,
                reason,
            },
        )
        .await;
        return if allow.is_empty() { None } else { Some(allow) };
    }
    combined
}

struct GateAllowDecision {
    plugin: &'static str,
    priority: i32,
    class: crate::plugin::ToolGateClass,
    suppresses_advisory: bool,
    allow: std::collections::HashSet<String>,
}

fn resolve_empty_tool_gate_intersection(
    decisions: &[(&'static str, i32, std::collections::HashSet<String>)],
) -> Option<(&'static str, std::collections::HashSet<String>, String)> {
    decisions
        .iter()
        .max_by(|(left_plugin, left_priority, left), (right_plugin, right_priority, right)| {
            left_priority
                .cmp(right_priority)
                .then_with(|| right.len().cmp(&left.len()))
                .then_with(|| right_plugin.cmp(left_plugin))
        })
        .map(|(plugin, priority, allow)| {
            (
                *plugin,
                allow.clone(),
                format!(
                    "empty intersection repaired by highest-priority owner `{plugin}` (priority {priority})"
                ),
            )
        })
}

fn sorted_tool_names(set: &std::collections::HashSet<String>) -> Vec<String> {
    let mut sorted: Vec<String> = set.iter().cloned().collect();
    sorted.sort();
    sorted
}

