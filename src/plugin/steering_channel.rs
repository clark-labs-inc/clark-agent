use super::*;

// ─── Helper: stand-alone steering channel ──────────────────────────

/// `tokio::sync::mpsc`-backed steering source. Producer side
/// (`SteeringHandle`) lets external code call `.steer(message)` from
/// anywhere; consumer side implements `SteeringSource` and drains the
/// channel each batch.
pub struct ChannelSteering {
    rx: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<AgentMessage>>,
}

#[derive(Clone)]
pub struct SteeringHandle {
    tx: tokio::sync::mpsc::UnboundedSender<AgentMessage>,
}

impl SteeringHandle {
    /// Inject a steering message. Returns `Ok` if the loop is still
    /// running, `Err` if it has already shut down.
    // Preserve the standard mpsc error so callers can recover the unsent
    // message; boxing it would make this small helper harder to use.
    #[allow(clippy::result_large_err)]
    pub fn steer(
        &self,
        message: AgentMessage,
    ) -> Result<(), tokio::sync::mpsc::error::SendError<AgentMessage>> {
        self.tx.send(message)
    }
}

impl ChannelSteering {
    pub fn new() -> (Arc<Self>, SteeringHandle) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (
            Arc::new(Self {
                rx: tokio::sync::Mutex::new(rx),
            }),
            SteeringHandle { tx },
        )
    }
}

impl Plugin for ChannelSteering {
    fn name(&self) -> &'static str {
        "channel_steering"
    }
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::steering()
    }
}

#[async_trait]
impl SteeringSource for ChannelSteering {
    async fn next_steering_messages(&self) -> Vec<AgentMessage> {
        let mut rx = self.rx.lock().await;
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            out.push(msg);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::UserContent;

    #[tokio::test]
    async fn channel_steering_drains() {
        let (source, handle) = ChannelSteering::new();
        handle
            .steer(AgentMessage::User {
                content: UserContent::Text("hi".into()),
                timestamp: None,
            })
            .unwrap();
        handle
            .steer(AgentMessage::User {
                content: UserContent::Text("again".into()),
                timestamp: None,
            })
            .unwrap();

        let drained = source.next_steering_messages().await;
        assert_eq!(drained.len(), 2);

        // Second call returns empty.
        let drained2 = source.next_steering_messages().await;
        assert!(drained2.is_empty());
    }
}
