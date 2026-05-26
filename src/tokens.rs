//! Token estimation for budgeting and compaction.
//!
//! `TokenEstimator` is the swappable boundary between the loop and any
//! tokenizer: char heuristic (default), tiktoken, sentencepiece, or a
//! provider-specific count-tokens API. Plugins that need a token count
//! for a single message or a slice (`TokenBudget`, future per-tool-result
//! cap, future auto-compact) read it through this trait so the loop has
//! one source of truth and apps with a real tokenizer can swap once.
//!
//! Threading: the estimator reaches plugins via [`crate::plugin::TransformContext`].
//! The loop holds it as `Arc<dyn TokenEstimator>` on `LoopConfig` and
//! lends a `&dyn TokenEstimator` per turn.

use crate::types::AgentMessage;

/// Estimates the token cost of agent messages for the binding model.
///
/// Implementations should be cheap to call repeatedly per turn — the
/// loop currently estimates total context length on every model call.
pub trait TokenEstimator: Send + Sync + 'static {
    /// Cost of a single message, in tokens.
    fn estimate_message(&self, message: &AgentMessage) -> usize;

    /// Cost of a slice of messages. Default sums per-message costs;
    /// implementations that share work across messages (vocab caches,
    /// streaming tokenizers) may override.
    fn estimate_messages(&self, messages: &[AgentMessage]) -> usize {
        messages.iter().map(|m| self.estimate_message(m)).sum()
    }
}

/// Coarse char-based estimator. Serializes the message to JSON and
/// divides by 4 (the ~English average chars-per-token). Cheap, no
/// external dependencies, no model awareness.
///
/// Apps with a real tokenizer should plug in their own implementation
/// via [`crate::config::AgentBuilder::token_estimator`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CharHeuristicEstimator;

impl TokenEstimator for CharHeuristicEstimator {
    fn estimate_message(&self, message: &AgentMessage) -> usize {
        serde_json::to_string(message)
            .map(|s| s.len() / 4)
            .unwrap_or(0)
    }
}

/// Static instance suitable for `&dyn TokenEstimator` borrows in tests
/// and `TransformContext::for_test`.
pub static CHAR_HEURISTIC: CharHeuristicEstimator = CharHeuristicEstimator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::UserContent;

    #[test]
    fn char_heuristic_estimates_in_token_units() {
        let msg = AgentMessage::User {
            content: UserContent::Text("x".repeat(400)),
            timestamp: None,
        };
        // JSON encoding adds ~30 chars of envelope; divided by 4 is ~107.
        // Ballpark: > 80, < 200.
        let est = CHAR_HEURISTIC.estimate_message(&msg);
        assert!(est > 80 && est < 200, "estimator out of range: {est}");
    }

    #[test]
    fn estimate_messages_sums() {
        let one = AgentMessage::User {
            content: UserContent::Text("a".into()),
            timestamp: None,
        };
        let two = AgentMessage::User {
            content: UserContent::Text("b".into()),
            timestamp: None,
        };
        let messages = vec![one.clone(), two.clone()];
        let summed = CHAR_HEURISTIC.estimate_messages(&messages);
        assert_eq!(
            summed,
            CHAR_HEURISTIC.estimate_message(&one) + CHAR_HEURISTIC.estimate_message(&two)
        );
    }
}
