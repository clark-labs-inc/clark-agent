//! Request provenance belongs to canonical history, beside its response.
//! Expansion is a pure render, with no provider cache or parallel transcript.
use crate::{AgentMessage, AssistantBlock, ContextObservation, ContextObservationsContent};

pub fn expand(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let mut expanded = Vec::with_capacity(messages.len());
    for mut message in messages {
        if let AgentMessage::Assistant { content, .. } = &mut message {
            let mut generated = Vec::with_capacity(content.blocks.len());
            for block in std::mem::take(&mut content.blocks) {
                match block {
                    AssistantBlock::ContextObservations(observations) => {
                        // Observations are host facts, never nested responses.
                        // Keep source order without recursively creating history.
                        expanded.extend(observations.messages.into_iter().map(ContextObservation::into_message));
                    }
                    other => generated.push(other),
                }
            }
            content.blocks = generated;
        }
        expanded.push(message);
    }
    expanded
}

pub fn attach(message: &mut AgentMessage, observations: Vec<ContextObservation>) {
    if observations.is_empty() { return; }
    let AgentMessage::Assistant { content, .. } = message else { return };
    content.blocks.insert(0, AssistantBlock::ContextObservations(
        ContextObservationsContent { messages: observations },
    ));
}

#[cfg(test)]
mod tests;
