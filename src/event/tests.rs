    use super::*;

    #[tokio::test]
    async fn channel_sink_forwards_events() {
        let (sink, mut rx) = ChannelSink::new();
        sink.emit(AgentEvent::AgentStart).await;
        sink.emit(AgentEvent::TurnStart).await;
        drop(sink);

        let mut received = Vec::new();
        while let Some(e) = rx.recv().await {
            received.push(e);
        }
        assert_eq!(received.len(), 2);
        assert!(matches!(received[0], AgentEvent::AgentStart));
        assert!(matches!(received[1], AgentEvent::TurnStart));
    }

    #[tokio::test]
    async fn fan_out_sink_replicates() {
        let (a, mut a_rx) = ChannelSink::new();
        let (b, mut b_rx) = ChannelSink::new();
        let fanout = FanOutSink::new(vec![Arc::new(a), Arc::new(b)]);
        fanout.emit(AgentEvent::AgentStart).await;
        drop(fanout);

        assert!(matches!(a_rx.recv().await, Some(AgentEvent::AgentStart)));
        assert!(matches!(b_rx.recv().await, Some(AgentEvent::AgentStart)));
    }

    #[test]
    fn provider_request_summary_counts_shape_without_text() {
        let messages = vec![
            AgentMessage::User {
                content: UserContent::Blocks(vec![
                    UserBlock::Text(crate::types::TextContent {
                        text: "secret user request".into(),
                    }),
                    UserBlock::Image(crate::types::ImageContent {
                        source: "data:image/png;base64,secret".into(),
                        media_type: Some("image/png".into()),
                        alt: Some("screenshot".into()),
                    }),
                ]),
                timestamp: None,
            },
            AgentMessage::Assistant {
                content: crate::types::AssistantContent {
                    blocks: vec![
                        AssistantBlock::Thinking(crate::types::TextContent {
                            text: "private scratch".into(),
                        }),
                        AssistantBlock::ToolCall(crate::tool::ToolCall {
                            id: "call-1".into(),
                            name: "web_search".into(),
                            arguments: serde_json::json!({"q": "secret"}),
                        }),
                    ],
                },
                stop_reason: crate::types::StopReason::ToolUse,
                error_message: None,
                timestamp: None,
                usage: None,
            },
            AgentMessage::ToolResult {
                tool_call_id: "call-1".into(),
                tool_name: "web_search".into(),
                content: crate::types::ToolResultContent::text("secret result"),
                is_error: false,
                narration: None,
                details: None,
                timestamp: None,
            },
        ];
        let tools = vec![ToolSchema {
            name: "web_search".into(),
            description: "Search the web".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }];

        let summary = ProviderRequestSummary::from_parts(
            2,
            Some("google/gemini-3.1-flash-lite-preview"),
            Some(0.2),
            Some(4096),
            "system prompt secret",
            &messages,
            &tools,
        );

        assert_eq!(summary.iteration, 2);
        assert_eq!(
            summary.model_id.as_deref(),
            Some("google/gemini-3.1-flash-lite-preview")
        );
        assert_eq!(summary.message_counts.user, 1);
        assert_eq!(summary.message_counts.assistant, 1);
        assert_eq!(summary.message_counts.tool_result, 1);
        assert_eq!(summary.content_counts.user_text_blocks, 1);
        assert_eq!(
            summary.content_counts.user_text_bytes,
            "secret user request".len()
        );
        assert_eq!(summary.content_counts.user_image_blocks, 1);
        assert_eq!(summary.content_counts.assistant_thinking_blocks, 1);
        assert_eq!(summary.content_counts.assistant_tool_call_blocks, 1);
        assert_eq!(summary.content_counts.tool_result_text_blocks, 1);
        assert_eq!(summary.tool_names, vec!["web_search"]);
        assert_eq!(summary.last_message_role.as_deref(), Some("tool_result"));

        let serialized = serde_json::to_string(&summary).unwrap();
        assert!(!serialized.contains("secret user request"));
        assert!(!serialized.contains("private scratch"));
        assert!(!serialized.contains("secret result"));
        assert!(!serialized.contains("data:image"));
    }
