//! Typed error enums.
//!
//! `LoopError` is fatal-only: stream transport unrecoverable failure or
//! caller cancellation. Recoverable tool errors are not loop errors —
//! they're context events: the tool returns `ToolResult` with the error encoded as text,
//! the loop appends it to history, and the model decides what to do.
//! Only explicit tool aborts and fatal tool errors bubble out.

use thiserror::Error;

/// Why the loop terminated abnormally.
///
/// A successful run returns `Ok(messages)` with no error. The loop's
/// natural stop condition (no more tool calls + no follow-up) does not
/// produce an error.
#[derive(Debug, Error)]
pub enum LoopError {
    /// Stream transport raised an unrecoverable error. The provider
    /// implementation decides what's recoverable; everything that bubbles
    /// up through `StreamFn::stream` ends the run.
    #[error("stream transport error: {0}")]
    Stream(#[from] StreamError),

    /// Caller cancelled via the abort signal.
    #[error("aborted")]
    Aborted,

    /// A tool encountered an unrecoverable failure and requested that
    /// the loop stop immediately rather than append a recoverable
    /// context event.
    #[error("fatal tool `{tool}` error: {reason}")]
    ToolFatal { tool: String, reason: String },

    /// Cannot continue without a starting message: `run_continue` was
    /// called on an empty context, or the trailing message is `assistant`
    /// (which the model would not respond to).
    #[error("cannot continue: {0}")]
    InvalidContinuation(String),

    /// The model repeatedly stopped without any tool call after the
    /// configured no-tool recovery budget had already been spent.
    #[error(
        "empty assistant outcome retry budget exhausted: observed {observed} no-tool assistant stop(s), budget {budget}"
    )]
    EmptyOutcomeBudgetExhausted { budget: usize, observed: usize },
}

#[derive(Debug, Error)]
pub enum StreamError {
    /// Transient failure: rate limit, network blip, retryable provider
    /// error. The transport implementation decides whether to retry
    /// internally or surface this.
    #[error("transient stream error: {0}")]
    Transient(String),

    /// The selected model/provider is temporarily rate-limited. The
    /// transport exhausted its own retry budget before surfacing this.
    #[error("provider rate-limited request: {0}")]
    ProviderRateLimited(String),

    /// Transport failed before the provider produced an actionable
    /// assistant turn. The request can be replayed as a clean provider
    /// attempt because there is no runnable assistant turn to preserve.
    #[error("zero-output transport error: {0}")]
    ZeroOutputTransport(String),

    /// Permanent failure: invalid request, auth, unsupported model.
    #[error("fatal stream error: {0}")]
    Fatal(String),

    /// Provider returned an empty response after streaming completed.
    /// The model produced nothing.
    #[error("empty stream response")]
    Empty,

    /// Provider rejected the request because the input context exceeds
    /// the model's window. Distinct from `Fatal` so the loop can apply
    /// recovery (compact + retry) instead of terminating. Today the
    /// run still ends — the recovery path lands with the Phase 2
    /// `OverflowRecovery` plugin chain.
    #[error("context overflow: {0}")]
    ContextOverflow(String),
}

#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool execution failed but the agent should keep running. Maps to
    /// a tool result with the error text and `is_error = true`.
    #[error("tool execution failed: {0}")]
    Execution(String),

    /// Tool was cancelled mid-run via the abort signal.
    #[error("tool aborted")]
    Aborted,

    /// Tool encountered a fatal error that should end the run. Use
    /// sparingly — most failures should be `Execution`.
    #[error("fatal tool error: {0}")]
    Fatal(String),
}

#[derive(Debug, Error)]
pub enum ToolValidationError {
    /// JSON Schema validation failed for the named field.
    #[error("invalid arguments for `{tool}`: {reason}")]
    InvalidArguments { tool: String, reason: String },

    /// Required field is missing for the requested action variant.
    #[error("missing required field `{field}` for `{tool}.{action}`")]
    MissingField {
        tool: String,
        action: String,
        field: String,
    },

    /// Some other validation failure not covered above.
    #[error("{0}")]
    Other(String),
}
