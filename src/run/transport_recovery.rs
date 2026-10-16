use crate::types::{AgentContext, AgentMessage};

const ZERO_OUTPUT_TRANSPORT_RECOVERY_CONTEXT: &str = "\
[runtime context — transport recovery, not user instruction]\n\
The previous provider attempt produced no actionable output: no visible assistant text and no usable tool call reached the runtime. \
Re-read the latest user request and available observations. Choose an advertised tool if work remains; otherwise follow the current task's completion contract, including an ordinary final assistant response when appropriate.";

pub(super) fn context_with_zero_output_transport_recovery(context: &AgentContext) -> AgentContext {
    let mut recovered = context.clone();
    recovered.messages.push(AgentMessage::System {
        content: ZERO_OUTPUT_TRANSPORT_RECOVERY_CONTEXT.to_string(),
        timestamp: Some(super::now_ms()),
    });
    recovered
}
