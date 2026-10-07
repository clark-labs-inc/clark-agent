use super::*;

#[tokio::test]
async fn repair_owner_suppresses_advisory_gate_before_plan_only_intersection() {
    let config = AgentBuilder::new()
        .stream(Arc::new(RepeatedTextStream::default()))
        .tool_gate_arc(Arc::new(StaticAllowGate {
            name: "delivery_repair_gate",
            tools: &["plan", "file_write"],
            priority: 100,
            class: ToolGateClass::Required,
            suppresses_advisory: true,
        }))
        .tool_gate_arc(Arc::new(StaticAllowGate {
            name: "wrap_up_gate",
            tools: &["plan", "message_result", "message_ask"],
            priority: 0,
            class: ToolGateClass::Advisory,
            suppresses_advisory: false,
        }))
        .build()
        .expect("config builds");

    let allow = collect_tool_allowlist_with_events(&config, 3, &[])
        .await
        .expect("repair owner should keep its own allowlist");

    assert_eq!(
        allow,
        ["plan".to_string(), "file_write".to_string()]
            .into_iter()
            .collect()
    );
}
#[tokio::test]
async fn terminal_only_plain_text_fallback_synthesizes_terminal_result() {
    let stream = Arc::new(RepeatedTextStream::default());
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("auto-tool-provider")
        .tools(tool_registry)
        .tool_gate_arc(Arc::new(TerminalOnlyGate))
        .plain_text_terminal_fallback_tool("message_result")
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("answer directly".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect("plain text should be converted on terminal-only turn");

    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.outcome, LoopOutcome::Done);
    assert!(result.messages.iter().any(|message| matches!(
        message,
        AgentMessage::ToolResult {
            tool_name,
            content,
            is_error: false,
            ..
        } if tool_name == "message_result"
            && content.plain_text() == "plain stop 0"
    )));
}
#[tokio::test]
async fn eager_plain_text_fallback_fires_without_terminal_only_allowlist() {
    // Providers in the "auto-when-forced" class can never be
    // wire-forced into a tool call, so prose IS their failure mode.
    // The eager flag lifts the "allowlist must already be narrowed"
    // precondition so the fallback fires on the FIRST plain-text
    // stop instead of after a narrowing gate has burned 2-3 nudge
    // turns.
    //
    // No `tool_gate_arc` is installed in this test, so the catalog
    // stays at the full registry — exactly the situation where the
    // non-eager path would refuse to convert.
    let stream = Arc::new(RepeatedTextStream::default());
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("auto-tool-provider-eager")
        .tools(tool_registry)
        .plain_text_terminal_fallback_tool("message_result")
        .plain_text_terminal_fallback_eager(true)
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("answer directly".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect("eager fallback should convert plain text on first stop");

    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.outcome, LoopOutcome::Done);
    assert!(result.messages.iter().any(|message| matches!(
        message,
        AgentMessage::ToolResult {
            tool_name,
            content,
            is_error: false,
            ..
        } if tool_name == "message_result"
            && content.plain_text() == "plain stop 0"
    )));
}
#[tokio::test]
async fn eager_nudge_mode_has_no_implicit_retry_ceiling() {
    // With `plain_text_terminal_fallback_eager_nudge(true)` the eager
    // path nudges the model with a protocol-recovery system message
    // on every consecutive plain-text stop. Four failures exceed the
    // deleted two-nudge cap; the fifth turn supplies the real terminal
    // tool call rather than relying on synthesized completion.
    let stream = Arc::new(PlainTextThenTerminatorStream::new(4));
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry
        .with(Arc::new(TerminalNamedTool("message_result")))
        .with(Arc::new(TerminatorTool));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("auto-tool-provider-eager-nudge")
        .tools(tool_registry)
        .plain_text_terminal_fallback_tool("message_result")
        .plain_text_terminal_fallback_eager(true)
        .plain_text_terminal_fallback_eager_nudge(true)
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("answer directly".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect("nudge mode should wait for a real terminal tool call");

    assert_eq!(stream.calls.load(Ordering::SeqCst), 5);
    assert_eq!(result.outcome, LoopOutcome::Done);

    let nudge_count = result
        .messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::System { content, .. } if content == crate::protocol::DEFAULT_PLAIN_TEXT_RECOVERY_PROMPT))
        .count();
    assert_eq!(
        nudge_count, 4,
        "expected one protocol-recovery message per plain-text turn, got {nudge_count}",
    );
}
#[tokio::test]
async fn non_eager_plain_text_fallback_still_requires_narrowed_allowlist() {
    // Default behaviour preserved: when eager is NOT set and the
    // turn allowlist is the full catalog, plain text is NOT
    // converted. It remains the run's natural assistant completion.
    let stream = Arc::new(RepeatedTextStream::default());
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("non-eager-provider")
        .tools(tool_registry)
        .plain_text_terminal_fallback_tool("message_result")
        // Eager NOT set → defaults to false.
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("answer directly".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect("plain assistant completion should remain valid");

    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    assert!(!result.messages.iter().any(|message| matches!(
        message,
        AgentMessage::ToolResult { tool_name, .. } if tool_name == "message_result"
    )));
}
#[tokio::test]
async fn terminal_plain_text_fallback_allows_status_delivery_gate() {
    let stream = Arc::new(RepeatedTextStream::default());
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry.with(Arc::new(TerminalNamedTool("message_result")));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("auto-tool-provider")
        .tools(tool_registry)
        .protocol_policy(Arc::new(TestTerminalPolicy))
        .tool_gate_arc(Arc::new(TerminalWithStatusGate))
        .plain_text_terminal_fallback_tool("message_result")
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("answer directly".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect(
            "plain text should be converted when only status and terminal tools are allowed",
        );

    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.outcome, LoopOutcome::Done);
    assert!(result.messages.iter().any(|message| matches!(
        message,
        AgentMessage::ToolResult {
            tool_name,
            content,
            is_error: false,
            ..
        } if tool_name == "message_result"
            && content.plain_text() == "plain stop 0"
    )));
}
