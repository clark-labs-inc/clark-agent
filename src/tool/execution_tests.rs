use super::*;
use crate::types::TextContent;
use schemars::JsonSchema;
use serde::Deserialize;

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echo arguments back as text"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        })
    }

    async fn execute(
        &self,
        _call_id: &str,
        args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        let text = args
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok(ToolResult {
            content: vec![ToolResultBlock::Text(TextContent { text })],
            is_error: false,
            details: Value::Null,
            terminate: false,
            narration: None,
        })
    }
}

#[test]
fn registry_lookup() {
    let registry = ToolRegistry::new().with(Arc::new(EchoTool));
    assert!(registry.get("echo").is_some());
    assert!(registry.get("missing").is_none());
    assert_eq!(registry.len(), 1);
}

struct NamedTool(&'static str);

#[async_trait]
impl AgentTool for NamedTool {
    fn name(&self) -> &str {
        self.0
    }

    fn description(&self) -> &str {
        "named"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn execute(
        &self,
        _call_id: &str,
        _args: Value,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::text("ok"))
    }
}

#[test]
fn registry_preserves_registration_order() {
    let mut registry = ToolRegistry::new()
        .with(Arc::new(NamedTool("message_result")))
        .with(Arc::new(NamedTool("message_ask")))
        .with(Arc::new(NamedTool("plan")));

    registry.register(Arc::new(NamedTool("message_result")));

    assert_eq!(
        registry.names(),
        vec!["message_result", "message_ask", "plan"]
    );
    assert_eq!(
        registry.iter().map(|tool| tool.name()).collect::<Vec<_>>(),
        vec!["message_result", "message_ask", "plan"]
    );
}

#[tokio::test]
async fn echo_tool_executes() {
    let tool = EchoTool;
    let (tx, _rx) = mpsc::unbounded_channel();
    let result = tool
        .execute(
            "call_1",
            serde_json::json!({"text": "hi"}),
            CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();
    let ToolResultBlock::Text(t) = &result.content[0] else {
        panic!("expected text")
    };
    assert_eq!(t.text, "hi");
}

// ---- end-to-end execute path ----------------------------------------
//
// The blanket impl `AgentTool::execute` for `TypedAgentTool` invokes
// (1) strip_top_level_nulls, (2) coerce_string_scalars_at_top_level,
// (3) serde_json::from_value, (4) enrich_arg_parse_error_message.
// Exercise the full path with a tool whose args mix the scalar types
// some providers routinely encode as strings.

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct CoercibleArgs {
    item_count: usize,
    full_page: bool,
    temperature: f32,
    label: String,
}

struct CoercibleTool;

#[async_trait]
impl TypedAgentTool for CoercibleTool {
    type Args = CoercibleArgs;
    fn name(&self) -> &str {
        "coercible"
    }
    fn description(&self) -> &str {
        "fixture"
    }
    async fn run(
        &self,
        _call_id: &str,
        args: Self::Args,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        // Echo the parsed values so the test can assert coercion happened.
        Ok(ToolResult::text(format!(
            "item_count={} full_page={} temperature={} label={}",
            args.item_count, args.full_page, args.temperature, args.label
        )))
    }
}

#[tokio::test]
async fn execute_coerces_string_encoded_scalars_end_to_end() {
    // The four shapes seen in practice — strings where the schema
    // declares integers, booleans, or floats. Each must pass the
    // validator after coercion and reach the tool's `run` with the
    // typed value.
    let tool = CoercibleTool;
    let (tx, _rx) = mpsc::unbounded_channel();
    let result = AgentTool::execute(
        &tool,
        "call_1",
        serde_json::json!({
            "item_count": "50",
            "full_page": "True",
            "temperature": "0.7",
            "label": "actual string",
        }),
        CancellationToken::new(),
        tx,
    )
    .await
    .unwrap();
    let ToolResultBlock::Text(t) = &result.content[0] else {
        panic!("expected text result");
    };
    assert!(
        t.text.contains("item_count=50"),
        "integer coercion missing: {}",
        t.text
    );
    assert!(
        t.text.contains("full_page=true"),
        "boolean coercion missing: {}",
        t.text
    );
    assert!(
        t.text.contains("temperature=0.7"),
        "float coercion missing: {}",
        t.text
    );
    assert!(
        t.text.contains("label=actual string"),
        "string field must NOT be coerced: {}",
        t.text
    );
    assert!(!result.is_error, "execute must succeed after coercion");
}

#[tokio::test]
async fn execute_appends_self_correcting_hint_on_unrecoverable_string_int() {
    // The string "fifty" cannot be coerced to an integer; the
    // validator rejects, and the runtime appends a hint only when
    // it's accurate. Here the hint must NOT claim "Did you mean
    // the integer fifty" — there is no such number — so the
    // enrichment should pass through.
    let tool = CoercibleTool;
    let (tx, _rx) = mpsc::unbounded_channel();
    let result = AgentTool::execute(
        &tool,
        "call_2",
        serde_json::json!({
            "item_count": "fifty",
            "full_page": true,
            "temperature": 0.1,
            "label": "x",
        }),
        CancellationToken::new(),
        tx,
    )
    .await
    .unwrap();
    assert!(result.is_error, "expected validator rejection");
    assert_eq!(
        result.details,
        serde_json::json!({
            "kind": "tool_argument_validation",
            "recoverable": true,
            "display_hidden": true,
            "tool": "coercible",
        })
    );
    let ToolResultBlock::Text(t) = &result.content[0] else {
        panic!("expected text result");
    };
    assert!(
        t.text.starts_with("coercible: invalid arguments:"),
        "preserve canonical error prefix: {}",
        t.text
    );
    assert!(
        !t.text.contains("Did you mean the integer fifty"),
        "must not invent a hint when the value cannot parse: {}",
        t.text
    );
}

// Fixture for prepare_arguments wiring — mimics a tagged-enum
// tool like `browser_navigate` where the discriminator field
// must be present but can be inferred from a variant-unique
// field.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case")]
enum TaggedArgs {
    Open { url: String },
    Reload {},
}

struct TaggedTool;

#[async_trait]
impl TypedAgentTool for TaggedTool {
    type Args = TaggedArgs;
    fn name(&self) -> &str {
        "tagged_fixture"
    }
    fn description(&self) -> &str {
        "fixture"
    }
    fn prepare_arguments(&self, args: Value) -> Value {
        // Same inference shape as BrowserNavigateTool's real
        // override: if `action` is missing and `url` is present,
        // assume `open`.
        let Value::Object(mut obj) = args else {
            return args;
        };
        if !obj.contains_key("action") && obj.contains_key("url") {
            obj.insert("action".to_string(), Value::String("open".to_string()));
        }
        Value::Object(obj)
    }
    async fn run(
        &self,
        _call_id: &str,
        args: Self::Args,
        _signal: CancellationToken,
        _update: ToolUpdateSink,
    ) -> Result<ToolResult, ToolError> {
        let label = match args {
            TaggedArgs::Open { url } => format!("open:{url}"),
            TaggedArgs::Reload {} => "reload".to_string(),
        };
        Ok(ToolResult::text(label))
    }
}

#[tokio::test]
async fn execute_runs_prepare_arguments_before_typed_deser() {
    // Reproduces the dominant `browser_navigate` failure: the
    // model emits a tagged-enum call without the discriminator.
    // With `prepare_arguments` wired into the blanket execute,
    // the missing `action` is inferred from `url` and the call
    // reaches `run` as the `Open` variant.
    let tool = TaggedTool;
    let (tx, _rx) = mpsc::unbounded_channel();
    let result = AgentTool::execute(
        &tool,
        "call_1",
        serde_json::json!({"url": "https://example.com"}),
        CancellationToken::new(),
        tx,
    )
    .await
    .unwrap();
    let ToolResultBlock::Text(t) = &result.content[0] else {
        panic!("expected text result");
    };
    assert!(
        !result.is_error,
        "execute must succeed after action inference"
    );
    assert_eq!(t.text, "open:https://example.com");
}

#[tokio::test]
async fn execute_prepare_arguments_does_not_override_explicit_action() {
    let tool = TaggedTool;
    let (tx, _rx) = mpsc::unbounded_channel();
    let result = AgentTool::execute(
        &tool,
        "call_2",
        serde_json::json!({"action": "reload"}),
        CancellationToken::new(),
        tx,
    )
    .await
    .unwrap();
    let ToolResultBlock::Text(t) = &result.content[0] else {
        panic!("expected text result");
    };
    assert!(!result.is_error);
    assert_eq!(t.text, "reload");
}
