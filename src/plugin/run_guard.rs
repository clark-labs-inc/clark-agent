//! Admission of another model turn from canonical history, before provider I/O.
use super::{AgentMessage, Plugin};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub struct RunGuardContext<'a> {
    pub messages: &'a [AgentMessage],
    pub iteration: usize,
}

/// Structured budget/safety failure. This is neither user cancellation nor a
/// provider response, and cannot manufacture task completion.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{code}: {message}")]
pub struct RunGuardViolation {
    pub code: String,
    pub message: String,
    pub details: Value,
}

/// Pure, bounded inspection only: no I/O, action selection, or history mutation.
/// Register through `AgentBuilder::run_guard_arc`; guards do not alter schemas.
pub trait RunGuard: Plugin {
    fn check(&self, context: RunGuardContext<'_>) -> Result<(), RunGuardViolation>;
}

#[cfg(test)]
mod tests;
