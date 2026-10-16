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
/// integer 50?"). Preserve ordinary diagnostics verbatim, but bound long
/// messages: serde can embed an entire malformed argument in its error.
/// The original arguments remain in the assistant's tool call in history.
pub(super) fn enrich_arg_parse_error_message(err: &serde_json::Error) -> String {
    let raw = err.to_string();
    let message = match arg_parse_hint(&raw) {
        Some(hint) => format!("{raw}. {hint}"),
        None => raw,
    };
    bound_argument_diagnostic(message)
}

fn bound_argument_diagnostic(message: String) -> String {
    if message.len() <= 1024 { return message; }
    // Keep the error kind and beginning of the offending value, plus the
    // expected contract and recovery hint at the end. Never reinterpret args.
    let mut head = 384;
    while !message.is_char_boundary(head) { head -= 1; }
    let mut tail = message.len() - 512;
    while !message.is_char_boundary(tail) { tail += 1; }
    format!("{}\n[... {} diagnostic bytes omitted; original arguments remain in the tool call ...]\n{}",
        &message[..head], tail - head, &message[tail..])
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

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn short_diagnostics_are_byte_for_byte_unchanged() {
        for message in [String::new(), "missing field `path`".into(), "x".repeat(1024)] {
            assert_eq!(bound_argument_diagnostic(message.clone()), message);
        }
    }

    #[test]
    fn long_unicode_diagnostics_preserve_both_ends_with_a_byte_bound() {
        for width in ["a", "é", "界", "🦀"] {
            for offset in 0..4 {
                let message = format!("{}{} expected one of `read`, `write`", "x".repeat(offset), width.repeat(2048));
                let result = bound_argument_diagnostic(message.clone());
                assert!(result.len() <= 1024);
                assert!(result.starts_with(&message[..offset]));
                assert!(result.ends_with("expected one of `read`, `write`"));
                assert!(result.contains("diagnostic bytes omitted"));
            }
        }
    }

    #[test]
    fn long_array_type_errors_retain_expected_type_and_recovery_hint() {
        let error = serde_json::from_value::<Vec<String>>(serde_json::json!("x".repeat(12000))).unwrap_err();
        let result = enrich_arg_parse_error_message(&error);
        assert!(result.len() <= 1024);
        assert!(result.starts_with("invalid type: string"));
        assert!(result.contains("expected a sequence"));
        assert!(result.contains("Expected a JSON array"));
    }
}
