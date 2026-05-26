use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use clark_agent::{
    run, AgentBuilder, AgentContext, AgentMessage, AgentTool, AssistantBlock, AssistantContent,
    StopReason, StreamEvent, StreamFn, StreamRequest, ToolCall, ToolRegistry, ToolResult,
    ToolUpdateSink, UserContent,
};
use futures::stream::{self, BoxStream};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

struct ScriptedStream {
    responses: Mutex<Vec<AgentMessage>>,
}

impl ScriptedStream {
    fn new(responses: Vec<AgentMessage>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }
}

#[async_trait]
impl StreamFn for ScriptedStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let message = self.responses.lock().unwrap().remove(0);
        Box::pin(stream::iter([StreamEvent::Done { message }]))
    }
}

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the provided text."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" }
            },
            "required": ["text"]
        })
    }

    async fn execute(
        &self,
        _id: &str,
        args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, clark_agent::ToolError> {
        let text = args
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Ok(ToolResult::text(format!("echo: {text}")))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let call_echo = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: json!({ "text": "hello tool" }),
            })],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };

    let final_answer = AgentMessage::Assistant {
        content: AssistantContent::text("The echo tool ran successfully."),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: None,
        usage: None,
    };

    let registry = ToolRegistry::new().with(Arc::new(EchoTool));
    let config = AgentBuilder::new()
        .stream(Arc::new(ScriptedStream::new(vec![call_echo, final_answer])))
        .tools(registry)
        .max_iterations(4)
        .build()?;

    let outcome = run(
        vec![AgentMessage::User {
            content: UserContent::Text("Use the echo tool".into()),
            timestamp: None,
        }],
        AgentContext::new("Use tools when useful."),
        &config,
        CancellationToken::new(),
    )
    .await?;

    for message in outcome.messages {
        match message {
            AgentMessage::ToolResult { content, .. } => {
                println!("tool result: {}", content.plain_text());
            }
            AgentMessage::Assistant { content, .. } => {
                let text = content.plain_text();
                if !text.is_empty() {
                    println!("assistant: {text}");
                }
            }
            _ => {}
        }
    }

    Ok(())
}
