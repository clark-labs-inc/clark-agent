use super::*;

#[tokio::test]
async fn max_tool_calls_per_turn_preserves_extra_calls_with_error_results() {
    let turn1 = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"text": "first"}),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "c2".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"text": "second"}),
                }),
            ],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let turn2 = AgentMessage::Assistant {
        content: AssistantContent::text("done"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let stream = Arc::new(ScriptedStream::new(vec![turn1, turn2]));

    let (sink, mut rx) = ChannelSink::new();
    let registry = ToolRegistry::new().with(Arc::new(EchoTool));
    let config = AgentBuilder::new()
        .stream(stream)
        .tools(registry)
        .event_sink(Arc::new(sink))
        .max_tool_calls_per_turn(1)
        .build()
        .unwrap();

    let messages = clark_agent::run(
        vec![AgentMessage::User {
            content: UserContent::Text("go".into()),
            timestamp: None,
        }],
        AgentContext::new("test"),
        &config,
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .messages;

    drop(config);
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }

    let AgentMessage::Assistant { content, .. } = &messages[1] else {
        panic!("expected assistant tool call");
    };
    assert_eq!(content.tool_calls().len(), 2);
    assert_eq!(content.tool_calls()[0].id, "c1");
    assert_eq!(content.tool_calls()[1].id, "c2");
    assert_eq!(
        messages
            .iter()
            .filter(|message| matches!(
                message,
                AgentMessage::ToolResult { tool_call_id, .. } if tool_call_id == "c2"
            ))
            .count(),
        1,
    );
    let AgentMessage::ToolResult {
        tool_call_id,
        content,
        ..
    } = &messages[2]
    else {
        panic!("expected first tool result");
    };
    assert_eq!(tool_call_id, "c1");
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert_eq!(t.text, "first");

    let AgentMessage::ToolResult {
        tool_call_id,
        content,
        is_error,
        ..
    } = &messages[3]
    else {
        panic!("expected synthetic error result");
    };
    assert_eq!(tool_call_id, "c2");
    assert!(*is_error);
    let ToolResultBlock::Text(t) = &content.blocks[0] else {
        panic!()
    };
    assert!(t.text.contains("not executed"));
    assert!(t.text.contains("only the first 1 call"));
    assert_ne!(t.text, "second");

    let c2_end = events
        .iter()
        .find(|event| {
            matches!(
                event,
                AgentEvent::ToolExecutionEnd {
                    tool_call_id,
                    is_error: true,
                    ..
                } if tool_call_id == "c2"
            )
        })
        .expect("synthetic result should emit a tool end event");
    let AgentEvent::ToolExecutionEnd { result, .. } = c2_end else {
        unreachable!()
    };
    assert!(result.is_error);
}

struct ReadTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str { "file_read" }
    fn description(&self) -> &str { "Read a file" }
    fn parameters_schema(&self) -> Value { serde_json::json!({"type": "object"}) }
    async fn execute(
        &self,
        _id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::text("file contents"))
    }
}

#[tokio::test]
async fn oversized_read_batch_preserves_every_call_and_returns_excess_as_errors() {
    let count = 904;
    let limit = 8;
    for mode in [clark_agent::ExecutionMode::Sequential, clark_agent::ExecutionMode::Parallel] {
        let assistant = AgentMessage::Assistant {
            content: AssistantContent {
                blocks: (0..count).map(|i| AssistantBlock::ToolCall(ToolCall {
                    id: format!("read-{i}"),
                    name: "file_read".into(),
                    arguments: serde_json::json!({}),
                })).collect(),
            },
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: None,
            usage: None,
        };
        let done = AgentMessage::Assistant {
            content: AssistantContent::text("done"),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: None,
            usage: None,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let config = AgentBuilder::new()
            .stream(Arc::new(ScriptedStream::new(vec![assistant, done])))
            .tools(ToolRegistry::new().with(Arc::new(ReadTool { calls: calls.clone() })))
            .default_execution_mode(mode)
            .max_tool_calls_per_turn(limit)
            .build().unwrap();
        let messages = clark_agent::run(
            vec![AgentMessage::User { content: UserContent::Text("read".into()), timestamp: None }],
            AgentContext::new("batch-budget"),
            &config,
            CancellationToken::new(),
        ).await.unwrap().messages;
        assert_eq!(calls.load(Ordering::SeqCst), limit);
        let AgentMessage::Assistant { content, .. } = &messages[1] else { panic!("assistant missing") };
        assert_eq!(content.tool_calls().len(), count);
        let results: Vec<_> = messages.iter().filter_map(|message| match message {
            AgentMessage::ToolResult { tool_call_id, is_error, content, .. } => Some((tool_call_id, is_error, content)),
            _ => None,
        }).collect();
        assert_eq!(results.len(), count);
        for i in 0..count {
            let matching: Vec<_> = results.iter().filter(|(id, _, _)| **id == format!("read-{i}")).collect();
            assert_eq!(matching.len(), 1);
            let (_, is_error, content) = matching[0];
            assert_eq!(**is_error, i >= limit);
            if i >= limit {
                let ToolResultBlock::Text(text) = &content.blocks[0] else { panic!("error text missing") };
                assert!(text.text.contains("not executed"));
                assert!(text.text.contains("first 8"));
            }
        }
    }
}
