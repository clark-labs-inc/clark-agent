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
  `FollowUpSource`, `ToolGate`). Cross-cutting concerns register here, not
  inline in the loop.
- **`protocol`** — `ProtocolPolicy`. The seam for product-specific tool
  vocabulary (recovery prose, tool-call alias repair, hidden-tool errors,
  terminal-tool classification). Default is generic and names no tools.
- **`config`** — `LoopConfig` + `AgentBuilder` for assembling everything.
- **`run`** — `run` / `run_continue` — the canonical loop. Pure functions.
- **`exec`** — tool execution: parallel + sequential dispatch, hook plumbing.
- **`history`** — provider-facing transcript invariants, including deterministic
  duplicate tool-call ID normalization without mutating durable storage.
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

Durable applications should normalize provider-chosen tool-call IDs before
wire conversion. Register `UniqueToolCallIds` as a `ContextTransform`, or use
the pure function when an existing structural-history transform owns ordering:

```rust
let normalized = clark_agent::normalize_tool_call_ids(messages);
let provider_messages = normalized.messages;
```

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
provider transport traits, events, and extension hooks. Product wiring
belongs in downstream crates.

The core knows **no product tool names**. The three places that once needed
product vocabulary — plain-text recovery prose, model tool-call alias repair,
and hidden-tool error messages — now go through a single seam, the
[`ProtocolPolicy`] trait:

```rust
pub trait ProtocolPolicy: Send + Sync + 'static {
    fn terminal_tool_names(&self) -> HashSet<String> { ... }
    fn plain_text_recovery_prompt(&self, ctx: PlainTextRecoveryContext<'_>) -> Option<String> { ... }
    fn normalize_tool_calls(&self, calls: &mut [ToolCall], registry: &ToolRegistry) -> usize { ... }
    fn hidden_tool_error(&self, ctx: HiddenToolContext<'_>) -> Option<HiddenToolError> { ... }
}
```

The core ships `DefaultProtocolPolicy` (generic, names no tools). A downstream
product installs its own via `AgentBuilder::protocol_policy(...)` to inject its
delivery/ask/plan vocabulary, tool-call aliases, and recovery prose — none of
which lives in this crate. New product-specific behavior should be implemented
as a `ProtocolPolicy`, a plugin (`ToolGate`, `ContextTransform`, …), or a tool
definition rather than added to the core loop.

[`ProtocolPolicy`]: https://docs.rs/clark-agent/latest/clark_agent/protocol/trait.ProtocolPolicy.html

## Release checks

```sh
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo publish --dry-run
```

## Citation

Citation authorship: Stanislav Kirdey, Clark Labs Inc. See
[`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

Apache-2.0 © Stanislav Kirdey, [Clark Labs Inc.](https://github.com/clark-labs-inc)

---

Built by **Stanislav Kirdey, Clark Labs Inc.** — the team behind
[Clark](https://www.clarkchat.com), AI-powered web automation and research.
If clark-agent is useful to you, a ⭐ on
[GitHub](https://github.com/clark-labs-inc/clark-agent) helps others discover it.
