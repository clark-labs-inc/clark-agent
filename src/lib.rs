//! # clark-agent
//!
//! A small, typed, hookable agent loop. Provider-agnostic, sandbox-agnostic,
//! tooling-agnostic.
//!
//! ## Shape
//!
//! ```text
//! context → LLM (StreamFn) → tool batch → results appended → repeat
//! ```
//!
//! Termination is a tool decision (`ToolResult::terminate = true`, unanimous
//! across the batch). The runtime owns execution and event emission; tools
//! own semantics; plugins own cross-cutting extension.
//!
//! ## Layers
//!
//! - [`types`] — `AgentMessage`, content blocks, `StopReason`. The conversation
//!   transcript is `Vec<AgentMessage>`. Apps extend via `AgentMessage::Custom`
//!   or by wrapping in their own enum.
//! - [`event`] — `AgentEvent` enum + `EventSink` trait. Single sink, typed
//!   events. Streamed and final delivery use the same enum.
//! - [`tool`] — `AgentTool` trait + `ToolRegistry`. Tools own their own state
//!   and validation; the loop only dispatches.
//! - [`stream`] — `StreamFn` trait. Swappable LLM transport: real provider,
//!   fixture replay, scripted scenario, remote proxy.
//! - [`plugin`] — `Plugin` + capability traits (`BeforeToolCall`,
//!   `AfterToolCall`, `ContextTransform`, `EventObserver`, `SteeringSource`).
//!   Cross-cutting concerns register here, not inline in the loop.
//! - [`config`] — `LoopConfig` + `AgentBuilder` for assembling everything.
//! - [`mod@run`] — [`run()`] / [`run_continue()`] — the canonical loop. Pure functions.
//! - [`exec`] — tool execution: parallel + sequential dispatch, hook plumbing.
//! - [`budget`] — default token-budget context transform.
//! - [`error`] — typed error enums.

pub mod budget;
pub mod config;
pub mod error;
pub mod event;
pub mod exec;
pub mod plugin;
pub mod plugins;
pub mod reasoning;
pub mod run;
pub mod stream;
pub mod thinking_filter;
pub mod tokens;
pub mod tool;
pub mod tool_identity;
pub mod tool_result_budget;
pub mod trajectory;
pub mod types;

pub use config::{
    AgentBuilder, LoopConfig, MaxTokensRecovery, PluginNames, TokenScaling,
    DEFAULT_GRACE_ITERATIONS,
};
pub use error::{LoopError, StreamError, ToolError, ToolValidationError};
pub use event::ChannelSink;
pub use event::{AgentEvent, EventSink, ProviderRequestSummary};
pub use plugin::PluginCapabilities;
pub use plugin::{
    AfterToolCall, AfterToolDecision, BeforeToolCall, BeforeToolDecision, ContextTransform,
    EventObserver, FollowUpSource, Plugin, SteeringSource, TransformContext,
};
pub use plugins::GracefulTurnLimit;
pub use reasoning::{
    audit_replay, OpenRouterReasoningCodec, ReasoningCodec, ReasoningFormat, ReasoningItem,
    ReplayAudit, ReplayContract, ReplayViolation,
};
pub use run::{run, run_continue, LoopOutcome, RunResult};
pub use stream::{
    AssistantStreamChunk, ReasoningEffort, StreamEvent, StreamFn, StreamRequest, StreamResponse,
};
pub use thinking_filter::{strip_thinking_tags, ThinkingTagStreamFilter};
pub use tokens::{CharHeuristicEstimator, TokenEstimator, CHAR_HEURISTIC};
pub use tool::{
    arg_parse_error_value, detect_arg_parse_error, AgentTool, ExecutionMode, ToolCall,
    ToolHistoryPolicy, ToolRegistry, ToolResult, ToolUpdateSink, TypedAgentTool,
    ARG_PARSE_ERROR_MARKER, ARG_PARSE_RAW_MARKER,
};
pub use tool_identity::{
    extract_args_key, extract_operation_key, extract_target, ArgsKeyFn, TargetExtractor, TargetFn,
    ToolIdentityPolicy,
};
pub use budget::TokenBudget;
pub use tool_result_budget::{ToolResultBudget, DEFAULT_PER_TOOL_CHARS};
pub use trajectory::{
    InMemoryTrajectorySink, TrajectoryError, TrajectoryPayload, TrajectoryRecord,
    TrajectoryRecorder, TrajectorySink,
};
pub use types::{
    AgentContext, AgentMessage, AssistantBlock, AssistantContent, ImageContent,
    ReasoningDetailsContent, RunIdentity, StopReason, TextContent, ToolResultBlock,
    ToolResultContent, UserBlock, UserContent,
};
