# clark-agent

A small, typed, hookable agent loop. Provider-agnostic, sandbox-agnostic,
tooling-agnostic.

## Shape

```
context → LLM (StreamFn) → tool batch → results appended → repeat
```

Termination is a tool decision (`ToolResult::terminate = true`, unanimous
across the batch). The runtime owns execution and event emission; tools
own semantics; plugins own cross-cutting extension.

## Layers

- **`types`** — `AgentMessage`, content blocks, `StopReason`. Conversation
  is `Vec<AgentMessage>`. Apps extend via `AgentMessage::Custom` or by
  wrapping in their own enum.
- **`event`** — `AgentEvent` enum + `EventSink` trait. Single sink, typed
  events. Streamed and final delivery use the same enum. `ChannelSink`,
  `FanOutSink`, `NoopSink` provided.
- **`tool`** — `AgentTool` trait + `ToolRegistry`. Tools own their schema,
  validation, and execution. The loop only dispatches.
- **`stream`** — `StreamFn` trait. Swappable LLM transport: real provider,
  fixture replay, scripted scenario, remote proxy.
- **`plugin`** — `Plugin` + capability traits (`BeforeToolCall`,
  `AfterToolCall`, `ContextTransform`, `EventObserver`, `SteeringSource`,
  `FollowUpSource`). Cross-cutting concerns register here, not inline in
  the loop.
- **`config`** — `LoopConfig` + `AgentBuilder` for assembling everything.
- **`run`** — `run` / `run_continue` — the canonical loop. Pure functions.
- **`exec`** — tool execution: parallel + sequential dispatch, hook plumbing.
- **`budget`** — default token-budget context transform.
- **`error`** — typed error enums.

## Plugin extension points

| Trait              | When it runs                                                             |
| ------------------ | ------------------------------------------------------------------------ |
| `BeforeToolCall`   | After argument validation, before `tool.execute`. May block with reason. |
| `AfterToolCall`    | After `tool.execute`. May override result, mark error, vote terminate.   |
| `ContextTransform` | Before each LLM call. Window management, redaction.                      |
| `EventObserver`    | On every `AgentEvent`. Logging, telemetry, persistence.                  |
| `SteeringSource`   | Between batches. Inject extra messages mid-run.                          |
| `FollowUpSource`   | After natural stop. Re-start the agent if more is queued.                |

A single struct can implement multiple capability traits — declare the
set via `Plugin::capabilities()` and register once with
`AgentBuilder::plugin()`.

## Quick start

```rust
use std::sync::Arc;
use clark_agent::{AgentBuilder, AgentContext, AgentMessage, ToolRegistry, UserContent};
use tokio_util::sync::CancellationToken;

let registry = ToolRegistry::new()
    .with(Arc::new(my_shell_tool()))
    .with(Arc::new(my_file_tool()));

let config = AgentBuilder::new()
    .stream(Arc::new(my_provider()))
    .tools(registry)
    .before_tool_call(my_security_gate())
    .after_tool_call(my_repeat_detector())
    .context_transform(clark_agent::budget::TokenBudget::default())
    .max_iterations(50)
    .build()?;

let outcome = clark_agent::run(
    vec![AgentMessage::User {
        content: UserContent::Text("List files in /tmp".into()),
        timestamp: None,
    }],
    AgentContext::new("You are a helpful assistant."),
    &config,
    CancellationToken::new(),
).await?;
```

## Examples

Run the smallest possible loop with a scripted transport:

```sh
cargo run --example minimal
```

Run a two-turn loop where the model calls a typed `echo` tool:

```sh
cargo run --example tool_call
```

Real integrations provide their own `StreamFn` implementation for an LLM
provider and register application tools through `AgentTool` or
`TypedAgentTool`.

## Mid-run steering (`steer()`)

```rust
let (steering, handle) = clark_agent::plugin::ChannelSteering::new();
let config = AgentBuilder::new()
    .stream(provider)
    .tools(registry)
    .steering_arc(steering)
    .build()?;

// In another task: inject a message between batches.
handle.steer(AgentMessage::User {
    content: UserContent::Text("actually, focus on /etc instead".into()),
    timestamp: None,
})?;
```

## Design rules

- **One canonical core.** `run` / `run_continue` are pure functions, not
  methods on a god-class.
- **Hooks are typed, narrow, side-effect-free.** No I/O in `BeforeToolCall`
  or `AfterToolCall` — those belong to the tool's own `execute`.
- **Failure is a context event.** Tool errors become tool result content
  with `is_error: true`. The loop appends and continues. Only `LoopError`
  (stream transport unrecoverable / aborted) ends the run.
- **Termination requires unanimity.** A batch ends the run only when
  every finalized tool result votes `terminate: true`. One tool wanting
  to stop does not stop the batch.
- **Strongly typed contracts.** Discriminators are enums; payloads are
  typed structs; field-name string lookups (`obj["role"]`) are forbidden
  in primary contracts. `serde_json::Value` only at open-by-design leaves
  (provider extras, custom message payloads, tool arguments).

## Open-source boundary

`clark-agent` is the reusable loop crate: typed history, tool dispatch,
provider transport traits, events, and extension hooks. Clark product wiring
belongs in downstream crates such as `clark-agent-bridge`.

The current 0.1 line still includes a small compatibility layer for Clark's
legacy delivery/planning tool names (`message_result`, `message_ask`, `plan`)
so existing Clark integrations keep working while the public API stabilizes.
New product-specific behavior should be implemented as bridge plugins or
tool definitions rather than added to this core crate.

## Release checks

```sh
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo publish --dry-run
```

## License

Apache-2.0.
