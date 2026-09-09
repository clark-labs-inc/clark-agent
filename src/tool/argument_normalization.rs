use serde_json::Value;

/// Convert top-level string-encoded scalars to their JSON-Schema-declared
/// types when the conversion is unambiguous. Walks `value` (which must be
/// an object) and, for each property whose schema declares a single scalar
/// type (`integer`, `number`, `boolean`), parses the corresponding string
/// value in place. Leaves arrays, objects, nested oneOf branches, and
/// fields with a non-string current value untouched — those go through
/// the strict serde path unchanged. Conservative by design: any
/// ambiguity (multi-type schemas, untyped properties, unparseable
/// strings) preserves the original value so the strict validator still
/// catches genuinely-malformed args.
pub(super) fn coerce_string_scalars_at_top_level(value: Value, schema: &Value) -> Value {
    let Value::Object(mut map) = value else {
        return value;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Value::Object(map);
    };
    for (key, val) in map.iter_mut() {
        let Some(prop_schema) = properties.get(key) else {
            continue;
        };
        coerce_one_scalar_in_place(val, prop_schema);
    }
    Value::Object(map)
}

pub(super) fn coerce_one_scalar_in_place(value: &mut Value, prop_schema: &Value) {
    let Some(text) = value.as_str() else {
        return;
    };
    let Some(target) = scalar_target_from_schema(prop_schema) else {
        return;
    };
    match target {
        ScalarTarget::Integer => {
            let trimmed = text.trim();
            if let Ok(n) = trimmed.parse::<i64>() {
                *value = Value::Number(serde_json::Number::from(n));
            } else if let Ok(n) = trimmed.parse::<u64>() {
                *value = Value::Number(serde_json::Number::from(n));
            }
        }
        ScalarTarget::Number => {
            let trimmed = text.trim();
            if let Ok(n) = trimmed.parse::<f64>() {
                if let Some(num) = serde_json::Number::from_f64(n) {
                    *value = Value::Number(num);
                }
            }
        }
        ScalarTarget::Boolean => match text.trim() {
            "true" | "True" | "TRUE" => *value = Value::Bool(true),
            "false" | "False" | "FALSE" => *value = Value::Bool(false),
            _ => {}
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ScalarTarget {
    Integer,
    Number,
    Boolean,
}

pub(super) fn scalar_target_from_schema(prop_schema: &Value) -> Option<ScalarTarget> {
    let type_field = prop_schema.get("type")?;
    let single = match type_field {
        Value::String(s) => Some(s.as_str()),
        // Optional-shaped schemas often render as ["T", "null"]; pick the
        // non-null entry. Anything wider (e.g. ["string", "integer"]) is
        // genuinely ambiguous — skip and let the strict validator decide.
        Value::Array(arr) => {
            let non_null: Vec<&str> = arr
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|s| *s != "null")
                .collect();
            if non_null.len() == 1 {
                Some(non_null[0])
            } else {
                None
            }
        }
        _ => None,
    }?;
    match single {
        "integer" => Some(ScalarTarget::Integer),
        "number" => Some(ScalarTarget::Number),
        "boolean" => Some(ScalarTarget::Boolean),
        _ => None,
    }
}

/// Append a self-correcting hint to a serde-deserialize error message
/// when the failure pattern is something a model can fix on the next
/// turn (e.g. "string \"50\", expected usize" → "Did you mean the
/// integer 50?"). The base error text is preserved verbatim so the
/// existing format stays diffable; the hint is suffixed after a period.
pub(super) fn enrich_arg_parse_error_message(err: &serde_json::Error) -> String {
    let raw = err.to_string();
    match arg_parse_hint(&raw) {
        Some(hint) => format!("{raw}. {hint}"),
        None => raw,
    }
}

fn arg_parse_hint(raw: &str) -> Option<String> {
    let value = extract_invalid_string_value(raw)?;
    if expects_integer(raw) {
        let parsed: i128 = value.trim().parse().ok()?;
        return Some(format!(
            "Did you mean the integer {parsed}? Resend without quotes."
        ));
    }
    if expects_number(raw) {
        let parsed: f64 = value.trim().parse().ok()?;
        return Some(format!(
            "Did you mean the number {parsed}? Resend without quotes."
        ));
    }
    if expects_boolean(raw) {
        return match value.trim() {
            "true" | "True" | "TRUE" => Some(
                "Did you mean true? Resend as a boolean literal (lowercase, no quotes)."
                    .to_string(),
            ),
            "false" | "False" | "FALSE" => Some(
                "Did you mean false? Resend as a boolean literal (lowercase, no quotes)."
                    .to_string(),
            ),
            _ => None,
        };
    }
    if expects_sequence(raw) {
        return Some(
            "Expected a JSON array (e.g. `[{...}, {...}]`); the field cannot be a string. \
             Resend the value as an array of structured items, not a string of XML-like markup."
                .to_string(),
        );
    }
    None
}

fn extract_invalid_string_value(raw: &str) -> Option<&str> {
    // Serde's `invalid type` errors quote the offending value as
    // `string "X"`. Locate the inner content without pulling in a regex
    // dependency; bail on the first malformed shape.
    let start = raw.find("string \"")? + "string \"".len();
    let rest = &raw[start..];
    let end = rest.find('\"')?;
    Some(&rest[..end])
}

fn expects_integer(raw: &str) -> bool {
    raw.contains("expected usize")
        || raw.contains("expected isize")
        || raw.contains("expected u8")
        || raw.contains("expected u16")
        || raw.contains("expected u32")
        || raw.contains("expected u64")
        || raw.contains("expected i8")
        || raw.contains("expected i16")
        || raw.contains("expected i32")
        || raw.contains("expected i64")
        || raw.contains("expected integer")
}

fn expects_number(raw: &str) -> bool {
    raw.contains("expected f32")
        || raw.contains("expected f64")
        || raw.contains("expected floating point")
}

fn expects_boolean(raw: &str) -> bool {
    raw.contains("expected a boolean") || raw.contains("expected bool")
}

fn expects_sequence(raw: &str) -> bool {
    raw.contains("expected a sequence") || raw.contains("expected an array")
}

pub(super) fn strip_top_level_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.into_iter().filter(|(_, v)| !v.is_null()).collect())
        }
        other => other,
    }
}
