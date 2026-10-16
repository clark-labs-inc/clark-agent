use super::*;
use crate::*;
use crate::plugin::{ContextTransform, TransformContext, PluginCapabilities};
use async_trait::async_trait;
use futures::{stream, stream::BoxStream, StreamExt};
use serde_json::json;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use tokio_util::sync::CancellationToken;

struct Budget;
impl Plugin for Budget { fn name(&self) -> &'static str { "fixture_failure_budget" } }
impl RunGuard for Budget {
    fn check(&self, ctx: RunGuardContext<'_>) -> Result<(), RunGuardViolation> {
        let count = ctx.messages.iter().filter(|message| matches!(message,
            AgentMessage::ToolResult { is_error: true, .. })).count();
        if count >= 2 {
            Err(RunGuardViolation { code: "fixture_limit".into(), message: "Recovery exhausted".into(),
                details: json!({"attempts":count, "iteration":ctx.iteration}) })
        } else { Ok(()) }
    }
}

struct HideResults;
impl Plugin for HideResults { fn name(&self) -> &'static str { "fixture_projection" } }
#[async_trait]
impl ContextTransform for HideResults {
    async fn transform(&self, messages: Vec<AgentMessage>, _: &TransformContext<'_>) -> Vec<AgentMessage> {
        messages.into_iter().filter(|message| !matches!(message, AgentMessage::ToolResult { .. })).collect()
    }
}

struct Work;
#[async_trait]
impl AgentTool for Work {
    fn name(&self) -> &str { "work" }
    fn description(&self) -> &str { "Fixture" }
    fn parameters_schema(&self) -> serde_json::Value { json!({"type":"object"}) }
    async fn execute(&self, _: &str, _: serde_json::Value, _: CancellationToken, _: ToolUpdateSink)
        -> Result<ToolResult, ToolError> { Ok(ToolResult::error("fixture failure")) }
}

struct Calls(Arc<AtomicUsize>);
#[async_trait]
impl StreamFn for Calls {
    async fn stream(&self, request: StreamRequest, _: CancellationToken) -> BoxStream<'static, StreamEvent> {
        assert_eq!(request.tools.len(), 1, "guard never chooses or hides tools");
        assert!(!request.messages.iter().any(|m| matches!(m, AgentMessage::ToolResult { .. })));
        let i = self.0.fetch_add(1, Ordering::SeqCst);
        stream::iter(vec![StreamEvent::Done { message: AgentMessage::Assistant {
            content: AssistantContent::with_tool_calls(None, vec![ToolCall {
                id: format!("call-{i}"), name: "work".into(), arguments: json!({}),
            }]), stop_reason: StopReason::ToolUse, error_message: None, timestamp: None, usage: None,
        }}]).boxed()
    }
}

struct Prepared(Arc<AtomicUsize>);
#[async_trait]
impl EventSink for Prepared {
    async fn emit(&self, event: AgentEvent) {
        if matches!(event, AgentEvent::ProviderRequestPrepared { .. }) { self.0.fetch_add(1, Ordering::SeqCst); }
    }
}

#[tokio::test]
async fn guard_reads_canonical_history_and_stops_before_provider_accounting() {
    let calls = Arc::new(AtomicUsize::new(0));
    let prepared = Arc::new(AtomicUsize::new(0));
    let config = AgentBuilder::new().stream(Arc::new(Calls(calls.clone())))
        .tools(ToolRegistry::new().with(Arc::new(Work)))
        .run_guard_arc(Arc::new(Budget)).context_transform(HideResults)
        .event_sink(Arc::new(Prepared(prepared.clone()))).max_iterations(5).build().unwrap();
    let signal = CancellationToken::new();
    let result = run(vec![AgentMessage::User { content: UserContent::Text("work".into()), timestamp: None }],
        AgentContext::new("fixture"), &config, signal.clone()).await;
    let Err(LoopError::Guardrail(violation)) = result else { panic!("expected typed guard failure") };
    assert_eq!(violation.code, "fixture_limit");
    assert_eq!(violation.details["attempts"], 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(prepared.load(Ordering::SeqCst), 2);
    assert!(!signal.is_cancelled(), "budget exhaustion is not user cancellation");
    assert_eq!(config.plugin_names().run_guard, vec!["fixture_failure_budget"]);
    assert!(config.child_builder().build().unwrap().plugin_names().run_guard.is_empty());
}

struct InheritedBudget;
impl Plugin for InheritedBudget {
    fn name(&self) -> &'static str { "inherited_budget" }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities { inheritable_to_child: true, ..PluginCapabilities::default() }
    }
}
impl RunGuard for InheritedBudget {
    fn check(&self, _: RunGuardContext<'_>) -> Result<(), RunGuardViolation> { Ok(()) }
}
#[test]
fn guard_inheritance_is_explicit() {
    let config = AgentBuilder::new().stream(Arc::new(Calls(Arc::new(AtomicUsize::new(0)))))
        .run_guard_arc(Arc::new(InheritedBudget)).build().unwrap();
    assert_eq!(config.child_builder().build().unwrap().plugin_names().run_guard, vec!["inherited_budget"]);
}
