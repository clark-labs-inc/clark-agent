use super::argument_normalization::*;
use super::schema::normalize_strict_validator_quirks;
use super::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct DocVariantArgs {
    filename: String,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct ExcelVariantArgs {
    filename: String,
    #[serde(default)]
    rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[allow(dead_code)]
enum ExampleArgs {
    Document(DocVariantArgs),
    Excel(ExcelVariantArgs),
}

fn build_example_schema() -> Value {
    let settings = schemars::gen::SchemaSettings::draft07().with(|s| {
        s.inline_subschemas = true;
    });
    let g = settings.into_generator();
    let s = g.into_root_schema_for::<ExampleArgs>();
    serde_json::to_value(s).unwrap()
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct NonAlphabeticOrderCanaryArgs {
    zeta_selector: String,
    alpha_payload: String,
    middle_payload: String,
}

#[test]
fn schema_runtime_preserves_insertion_order_for_tool_objects() {
    // This is the low-level wire invariant behind tool-schema field
    // geometry. The model emits arguments autoregressively in schema
    // order, so serde_json objects must serialize in insertion order,
    // not alphabetical map order.
    let mut object = serde_json::Map::new();
    object.insert("zeta_selector".to_string(), Value::String("z".to_string()));
    object.insert("alpha_payload".to_string(), Value::String("a".to_string()));
    object.insert("middle_payload".to_string(), Value::String("m".to_string()));

    let keys = object.keys().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(
        keys,
        ["zeta_selector", "alpha_payload", "middle_payload"],
        "serde_json::Map must keep insertion order; losing this breaks \
         model-facing tool-schema property order"
    );

    let serialized = serde_json::to_string(&Value::Object(object)).unwrap();
    assert_eq!(
        serialized, r#"{"zeta_selector":"z","alpha_payload":"a","middle_payload":"m"}"#,
        "schema JSON serialization must preserve object insertion order"
    );
}

#[test]
fn schemars_preserves_declared_struct_order_for_tool_args() {
    // Workspace schemars must keep Rust declaration order in
    // JSON-Schema `properties`; otherwise any Args type with a
    // discriminator, planning field, or thinking field silently
    // changes the order in which the model writes arguments.
    let settings = schemars::gen::SchemaSettings::draft07().with(|s| {
        s.inline_subschemas = true;
    });
    let schema = serde_json::to_value(
        settings
            .into_generator()
            .into_root_schema_for::<NonAlphabeticOrderCanaryArgs>(),
    )
    .expect("schema serializes");
    let props = schema
        .get("properties")
        .and_then(Value::as_object)
        .expect("schema must expose properties");
    let order = props.keys().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(
        order,
        ["zeta_selector", "alpha_payload", "middle_payload"],
        "schemars must emit Args fields in declaration order for \
         autoregressive tool-call conditioning"
    );
}

#[test]
fn typed_schema_preserves_action_fields_and_required_payloads() {
    let schema = schema::parameters_schema::<ExampleArgs>();
    let mut raw = build_example_schema();
    raw["oneOf"][1]["properties"]["rows"]["items"]["items"] = serde_json::json!({});
    assert_eq!(schema["oneOf"], raw["oneOf"]);
    assert!(schema.get("properties").is_none());
    let branches = schema["oneOf"].as_array().expect("action branches");
    assert_eq!(branches.len(), 2);
    for branch in branches {
        assert_eq!(branch["additionalProperties"], false);
        let required = branch["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("kind")));
        assert!(required.contains(&serde_json::json!("filename")));
    }
    assert!(branches[0]["properties"].get("rows").is_none());
    assert!(branches[1]["properties"].get("title").is_none());
}

#[test]
fn normalize_strict_quirks_rewrites_items_true_to_empty_object() {
    // Regression: schemars emits `items: true` for `Vec<Value>`
    // cells. Azure's tool-schema validator rejects boolean
    // schemas with `array schema items is not an object`,
    // failing every nano (Azure-routed) call. The normalizer
    // walks the tree and coerces `items: true` to `items: {}`.
    let mut schema = serde_json::json!({
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": true
                }
            }
        }
    });
    normalize_strict_validator_quirks(&mut schema);
    assert_eq!(
        schema.pointer("/properties/rows/items/items"),
        Some(&serde_json::json!({})),
    );
}

#[test]
fn strip_top_level_nulls_removes_inapplicable_variant_fields() {
    // Regression: weaker models submit EVERY field from EVERY
    // tagged-enum variant, with `null` for the non-applicable
    // ones, alongside the chosen discriminator. The chosen variant
    // has `deny_unknown_fields` and rejected with unknown
    // sibling fields. Stripping top-level nulls before
    // deserializing collapses these to missing fields and lets
    // `serde(default)` apply.
    let model_payload = serde_json::json!({
        "action": "run",
        "command": "echo hi",
        "workdir": "/home/user/workspace",
        // Sibling-variant fields the model populated with null:
        "code": null,
        "interpreter": null,
        "ext": null,
        "exec_dir": null,
        "max_token": null,
        "truncate_from": null,
        "run_id": null,
        "after_seq": null,
        "max_events": null,
        "timeout_s": null,
        "timeout_ms": null,
        "terminal": null,
        "force": null,
        // Plus a real value to confirm only nulls are dropped.
        "timeout_secs": 60,
    });
    let stripped = strip_top_level_nulls(model_payload);
    let obj = stripped.as_object().expect("object");
    // Nulls gone.
    assert!(!obj.contains_key("code"));
    assert!(!obj.contains_key("ext"));
    assert!(!obj.contains_key("max_token"));
    assert!(!obj.contains_key("force"));
    // Real values preserved.
    assert_eq!(obj.get("action").and_then(Value::as_str), Some("run"));
    assert_eq!(obj.get("command").and_then(Value::as_str), Some("echo hi"));
    assert_eq!(obj.get("timeout_secs").and_then(Value::as_i64), Some(60));
}

#[test]
fn strip_top_level_nulls_passes_through_non_object_values() {
    // Defensive: tool args at the top level should always be
    // objects, but the helper must not panic on the off-chance
    // a transport hands us a primitive.
    assert_eq!(
        strip_top_level_nulls(serde_json::json!("text")),
        serde_json::json!("text")
    );
    assert_eq!(strip_top_level_nulls(Value::Null), Value::Null);
}

// ---- string-encoded-scalar coercion -------------------------------
//
// Some providers (notably the "auto-when-forced" class) emit tool
// arguments as JSON strings for fields the schema declares as
// integers, booleans, or numbers — e.g. `item_count: "50"`,
// `num_results: "10"`, `full_page: "True"`, `full_page: "true"` —
// each a wasted turn under strict serde. The coercion helpers
// normalize the dominant cases against the tool's own JSON Schema;
// ambiguous cases are left to the strict path.

fn make_schema(properties: Value) -> Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
    })
}

#[test]
fn coerce_string_to_integer_when_schema_says_integer() {
    let schema = make_schema(serde_json::json!({
        "item_count": {"type": "integer"},
    }));
    let coerced =
        coerce_string_scalars_at_top_level(serde_json::json!({"item_count": "50"}), &schema);
    assert_eq!(coerced, serde_json::json!({"item_count": 50}));
}

#[test]
fn coerce_string_to_integer_handles_negative_and_whitespace() {
    let schema = make_schema(serde_json::json!({
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
    }));
    let coerced = coerce_string_scalars_at_top_level(
        serde_json::json!({"offset": "-7", "limit": "  42  "}),
        &schema,
    );
    assert_eq!(coerced, serde_json::json!({"offset": -7, "limit": 42}));
}

#[test]
fn coerce_string_to_boolean_for_each_case_variant() {
    let schema = make_schema(serde_json::json!({
        "full_page": {"type": "boolean"},
        "headless": {"type": "boolean"},
        "verbose": {"type": "boolean"},
        "untouched": {"type": "boolean"},
    }));
    let coerced = coerce_string_scalars_at_top_level(
        serde_json::json!({
            "full_page": "true",
            "headless": "True",
            "verbose": "FALSE",
            "untouched": "maybe",
        }),
        &schema,
    );
    // Recognised forms become bools; gibberish stays a string so the
    // strict validator still rejects with a useful error.
    assert_eq!(coerced["full_page"], serde_json::json!(true));
    assert_eq!(coerced["headless"], serde_json::json!(true));
    assert_eq!(coerced["verbose"], serde_json::json!(false));
    assert_eq!(coerced["untouched"], serde_json::json!("maybe"));
}

#[test]
fn coerce_string_to_number_for_float_schema() {
    let schema = make_schema(serde_json::json!({
        "temperature": {"type": "number"},
    }));
    let coerced =
        coerce_string_scalars_at_top_level(serde_json::json!({"temperature": "0.7"}), &schema);
    // f64 → Number round-trips through serde_json::Number::from_f64.
    let n = coerced["temperature"].as_f64().expect("number");
    assert!((n - 0.7).abs() < 1e-9);
}

#[test]
fn coerce_leaves_string_fields_alone() {
    let schema = make_schema(serde_json::json!({
        "query": {"type": "string"},
        "count": {"type": "integer"},
    }));
    let coerced = coerce_string_scalars_at_top_level(
        serde_json::json!({"query": "50", "count": "50"}),
        &schema,
    );
    // The string-typed field must NOT be turned into a number even
    // though "50" parses cleanly — schema is the source of truth.
    assert_eq!(coerced["query"], serde_json::json!("50"));
    assert_eq!(coerced["count"], serde_json::json!(50));
}

#[test]
fn coerce_leaves_unparseable_strings_alone() {
    let schema = make_schema(serde_json::json!({
        "item_count": {"type": "integer"},
    }));
    let coerced =
        coerce_string_scalars_at_top_level(serde_json::json!({"item_count": "fifty"}), &schema);
    // Unparseable values pass through so the strict serde path
    // produces the canonical "invalid type" error rather than us
    // silently dropping the value.
    assert_eq!(coerced, serde_json::json!({"item_count": "fifty"}));
}

#[test]
fn coerce_treats_nullable_integer_as_integer() {
    // `Option<usize>` renders as `{"type": ["integer", "null"]}`.
    // The non-null branch is unambiguous, so coercion still applies.
    let schema = make_schema(serde_json::json!({
        "item_count": {"type": ["integer", "null"]},
    }));
    let coerced =
        coerce_string_scalars_at_top_level(serde_json::json!({"item_count": "20"}), &schema);
    assert_eq!(coerced, serde_json::json!({"item_count": 20}));
}

#[test]
fn coerce_skips_ambiguous_multi_type_schemas() {
    // If the schema genuinely accepts both string and integer, leave
    // the value alone — coercion would discard the model's chosen
    // representation. Multi-type schemas wider than `[T, null]` are
    // ambiguous.
    let schema = make_schema(serde_json::json!({
        "value": {"type": ["integer", "string"]},
    }));
    let coerced = coerce_string_scalars_at_top_level(serde_json::json!({"value": "42"}), &schema);
    assert_eq!(coerced, serde_json::json!({"value": "42"}));
}

#[test]
fn coerce_passes_through_object_without_properties() {
    // No schema info → no coercion. Mirrors the safe path for tools
    // that ship a schema without explicit `properties` (e.g. when
    // the args type is `serde_json::Value`).
    let schema = serde_json::json!({"type": "object"});
    let coerced = coerce_string_scalars_at_top_level(serde_json::json!({"x": "50"}), &schema);
    assert_eq!(coerced, serde_json::json!({"x": "50"}));
}

// ---- arg-parse-error enrichment -----------------------------------

fn hint_for(json: Value, expected_target: &str) -> Option<String> {
    // Drive serde with a real schema mismatch so the helper sees a
    // genuine `serde_json::Error`, not a hand-written string. Skip
    // the coercion pass on purpose — we want the strict-path error.
    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct UsizeField {
        n: usize,
    }
    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct BoolField {
        b: bool,
    }
    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct VecField {
        items: Vec<serde_json::Value>,
    }
    let raw = match expected_target {
        "usize" => serde_json::from_value::<UsizeField>(json).unwrap_err(),
        "bool" => serde_json::from_value::<BoolField>(json).unwrap_err(),
        "sequence" => serde_json::from_value::<VecField>(json).unwrap_err(),
        _ => panic!("unknown target {expected_target}"),
    };
    Some(enrich_arg_parse_error_message(&raw))
}

#[test]
fn enrich_appends_integer_hint_for_string_encoded_int() {
    let msg = hint_for(serde_json::json!({"n": "50"}), "usize").unwrap();
    assert!(
        msg.contains("Did you mean the integer 50"),
        "expected integer hint, got: {msg}"
    );
    assert!(msg.contains("Resend without quotes"));
}

#[test]
fn enrich_appends_boolean_hint_for_string_encoded_bool() {
    let msg = hint_for(serde_json::json!({"b": "True"}), "bool").unwrap();
    assert!(
        msg.contains("Did you mean true"),
        "expected boolean hint, got: {msg}"
    );
}

#[test]
fn enrich_appends_sequence_hint_for_string_in_array_slot() {
    let xml_soup = "\n<ref>{\"kind\":\"file\",\"path\":\"x.md\"}</ref></artifact></file_write>";
    let msg = hint_for(serde_json::json!({"items": xml_soup}), "sequence").unwrap();
    assert!(
        msg.contains("Expected a JSON array"),
        "expected sequence hint, got: {msg}"
    );
}

#[test]
fn enrich_passes_through_unrecognised_errors_unchanged() {
    // Errors that don't match a known pattern (e.g. missing field)
    // must surface verbatim; making up a hint would mislead.
    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct R {
        n: usize,
    }
    let err = serde_json::from_value::<R>(serde_json::json!({})).unwrap_err();
    let raw = err.to_string();
    let enriched = enrich_arg_parse_error_message(&err);
    assert_eq!(enriched, raw);
}

#[test]
fn typed_schema_preserves_single_struct_contract() {
    let schema = schema::parameters_schema::<DocVariantArgs>();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"], serde_json::json!(["filename"]));
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema.get("oneOf").is_none());
}
