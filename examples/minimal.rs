use std::sync::Arc;

use async_trait::async_trait;
use clark_agent::{
    run, AgentBuilder, AgentContext, AgentMessage, AssistantContent, StopReason, StreamEvent,
    StreamFn, StreamRequest, UserContent,
};
use futures::stream::{self, BoxStream};
use tokio_util::sync::CancellationToken;

struct StaticStream;

#[async_trait]
impl StreamFn for StaticStream {
    async fn stream(
        &self,
        _request: StreamRequest,
        _signal: CancellationToken,
    ) -> BoxStream<'static, StreamEvent> {
        let message = AgentMessage::Assistant {
            content: AssistantContent::text("Hello from a scripted transport."),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: None,
            usage: None,
        };

        Box::pin(stream::iter([StreamEvent::Done { message }]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AgentBuilder::new()
        .stream(Arc::new(StaticStream))
        .max_iterations(4)
        .build()?;

    let outcome = run(
        vec![AgentMessage::User {
            content: UserContent::Text("Say hello".into()),
            timestamp: None,
        }],
        AgentContext::new("You are concise."),
        &config,
        CancellationToken::new(),
    )
    .await?;

    for message in outcome.messages {
        if let AgentMessage::Assistant { content, .. } = message {
            let text = content.plain_text();
            if !text.is_empty() {
                println!("{text}");
            }
        }
    }

    Ok(())
}
