use super::*;
use crate::{AssistantContent, StopReason};
use serde_json::json;

fn assistant() -> AgentMessage {
    AgentMessage::Assistant { content: AssistantContent::text("answer"),
        stop_reason: StopReason::EndTurn, error_message: None,
        timestamp: Some(42), usage: None }
}

fn observation() -> ContextObservation {
    ContextObservation { kind: "host_snapshot".into(),
        payload: json!({"clock":"2026-10-09T16:00:00Z"}), timestamp: Some(40) }
}

#[test]
fn exact_observations_precede_generated_response_without_changing_output() {
    let original = assistant();
    let mut response = original.clone();
    attach(&mut response, vec![observation()]);
    let AgentMessage::Assistant { content, .. } = &response else { unreachable!() };
    assert_eq!(content.plain_text(), "answer");
    assert!(content.reasoning_text().is_empty());
    assert!(content.tool_calls().is_empty());
    let expanded = expand(vec![response.clone()]);
    assert_eq!(expanded, vec![observation().into_message(), original]);
    assert_eq!(expand(expanded.clone()), expanded, "rendering is idempotent");
    let serialized = serde_json::to_vec(&response).unwrap();
    let replay: AgentMessage = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(response, replay, "canonical metadata survives persistence");
    assert_eq!(expand(vec![replay]), expanded);
}

#[test]
fn no_observations_does_not_change_canonical_message() {
    let mut response = assistant();
    let original = response.clone();
    attach(&mut response, vec![]);
    assert_eq!(response, original);
    assert_eq!(expand(vec![response]), vec![original]);
}


#[test]
fn tool_transactions_and_signed_reasoning_remain_exact_after_expansion() {
    let mut response = assistant();
    let AgentMessage::Assistant { content, .. } = &mut response else { unreachable!() };
    content.blocks.push(AssistantBlock::ReasoningDetails(crate::ReasoningDetailsContent::new(
        vec![json!({"type":"reasoning.encrypted", "data":"opaque-signature", "index":0})],
    )));
    content.blocks.push(AssistantBlock::ToolCall(crate::ToolCall {
        id: "call-1".into(), name: "read".into(), arguments: json!({"path":"file"}),
    }));
    let original = response.clone();
    attach(&mut response, vec![observation()]);
    let result = AgentMessage::ToolResult {
        tool_call_id: "call-1".into(), tool_name: "read".into(),
        content: crate::ToolResultContent::text("ordinary tool failure"), is_error: true,
        narration: None, details: None, timestamp: None,
    };
    assert_eq!(expand(vec![response, result.clone()]),
        vec![observation().into_message(), original, result]);
}
