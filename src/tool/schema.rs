use serde_json::Value;

/// Keep the Rust argument contract intact. Provider adapters own any dialect
/// conversion; flattening here irreversibly loses action-specific requirements.
pub(super) fn parameters_schema<T: schemars::JsonSchema>() -> Value {
    let settings = schemars::gen::SchemaSettings::draft07().with(|s| {
        s.inline_subschemas = true;
    });
    let generator = settings.into_generator();
    let schema = generator.into_root_schema_for::<T>();
    let mut value = serde_json::to_value(schema).expect("typed-tool schema serializes");
    normalize_strict_validator_quirks(&mut value);
    value
}

/// Coerce schemars output into shapes that strict tool-schema
/// validators (Azure's, OpenAI's via Azure proxy, several
/// OpenAI-compatible upstreams) accept. The current quirks list:
///
/// 1. `items: true` (boolean schema, valid in JSON Schema 2020-12 and
///    schemars's default for `Vec<Value>` cells) → rewrite to
///    `items: {}` (empty-object schema, draft-07 compatible). Azure
///    rejects boolean schemas with
///    `array schema items is not an object`.
///
/// Walks the tree once, mutating in place. Idempotent.
pub(super) fn normalize_strict_validator_quirks(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // Coerce `items: true` to `items: {}`.
            if let Some(items) = map.get_mut("items") {
                if matches!(items, Value::Bool(true)) {
                    *items = Value::Object(serde_json::Map::new());
                }
            }
            for v in map.values_mut() {
                normalize_strict_validator_quirks(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                normalize_strict_validator_quirks(v);
            }
        }
        _ => {}
    }
}

