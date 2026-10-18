//! Typed retry observations stay chronological and survive canonical replay.
use crate::{AgentMessage, ContextObservation};
use serde::{Deserialize, Serialize};

const KIND: &str = "clark_agent_transport_recovery";
const SNAPSHOT_KIND: &str = "clark_agent_transport_recovery_snapshot";
const ZERO_OUTPUT_CONTEXT: &str = "\
[runtime context — transport recovery, not user instruction]\n\
The previous provider attempt produced no actionable output: no visible assistant text and no usable tool call reached the runtime. \
Re-read the latest user request and available observations. Choose an advertised tool if work remains; otherwise follow the current task's completion contract, including an ordinary final assistant response when appropriate.";

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Reason { ZeroOutputTransport }

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation<'a> { reason: Reason, content: &'a str }

fn observation(message: &AgentMessage) -> Option<Observation<'_>> {
    let AgentMessage::Custom { kind, payload, .. } = message else { return None };
    (kind == KIND || kind == SNAPSHOT_KIND)
        .then(|| Observation::deserialize(payload).ok()).flatten()
}

/// Render only typed host observations, never matching user text or system prose.
pub fn render(message: &AgentMessage) -> Option<&str> {
    observation(message).map(|value| value.content)
}

pub(crate) fn snapshot(message: &AgentMessage) -> Option<ContextObservation> {
    let AgentMessage::Custom { kind, payload, timestamp } = message else { return None };
    if kind != KIND || observation(message).is_none() { return None; }
    Some(ContextObservation {
        kind: SNAPSHOT_KIND.into(), payload: payload.clone(), timestamp: *timestamp,
    })
}

pub(crate) fn zero_output_transport_recovery(timestamp: u64) -> ContextObservation {
    ContextObservation {
        kind: KIND.into(),
        payload: serde_json::to_value(Observation { reason: Reason::ZeroOutputTransport, content: ZERO_OUTPUT_CONTEXT })
            .expect("typed transport observation serializes"),
        timestamp: Some(timestamp),
    }
}

#[cfg(test)]
mod tests;
