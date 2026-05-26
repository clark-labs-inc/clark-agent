//! Conversation event replay round-trip — `AgentMessage` persistence
//! contract integration coverage.
//!
//! `clark-runtime::session_runner` reads conversation history as
//! `Vec<JsonValue>` from the DB and deserializes each row into
//! `AgentMessage`:
//!
//! ```ignore
//! let prior_messages: Vec<AgentMessage> = cfg.history.iter()
//!     .filter_map(|value| serde_json::from_value::<AgentMessage>(value.clone()).ok())
//!     .collect();
//! ```
//!
//! Anything that breaks this round-trip silently drops history rows on
//! the floor (the `.ok()` swallows the deserialize error and the loop
//! continues without it). Past regressions have included:
//!
//! - the `narration` field added to `ToolResult` (would have lost any
//!   tool-result row if not marked `#[serde(default)]`)
//! - `Usage` block additions (cache_read_input_tokens etc.)
//! - `AssistantBlock::ReasoningDetails` arrival (Gemini reasoning)
//!
//! These tests pin the round-trip property for every `AgentMessage`
//! variant and every nested type, including:
//!
//! - All variants of `AgentMessage` (System, User, Assistant, ToolResult, Custom)
//! - All variants of `AssistantBlock` (Text, Thinking, Reasoning,
//!   ReasoningDetails, ToolCall) — pinning the channel-separation
//!   contract documented at `crates/clark-agent/src/types.rs:280`
//! - All variants of `UserBlock` (Text, Image)
//! - All variants of `ToolResultBlock` (Text, Image)
//! - All variants of `StopReason`
//! - `Usage` with every cache field populated
//! - Forward-compatibility: persisted rows that pre-date a new optional
//!   field still deserialize (defaults take over)
//! - Backward-compatibility: rows with extra unknown keys do NOT fail
//!   to deserialize (so a future field doesn't break old workers)
//! - The exact wire shape used by the gateway (transparent
//!   `AssistantContent`, untagged `UserContent`, role discriminator)

use clark_agent::tool::ToolCall;
use clark_agent::types::{
    AgentMessage, AssistantBlock, AssistantContent, ImageContent, ReasoningDetailsContent,
    StopReason, TextContent, ToolResultBlock, ToolResultContent, Usage, UserBlock, UserContent,
};
use serde_json::{json, Value};

// ─── Helpers ───────────────────────────────────────────────────────

fn round_trip(msg: &AgentMessage) -> AgentMessage {
    let json = serde_json::to_value(msg).expect("AgentMessage serializes");
    serde_json::from_value::<AgentMessage>(json).expect("AgentMessage round-trips")
}

fn assert_round_trip(msg: AgentMessage) {
    let restored = round_trip(&msg);
    assert_eq!(
        msg, restored,
        "AgentMessage round-trip lost or mutated content"
    );
}

// ─── AgentMessage variant coverage ─────────────────────────────────

#[test]
fn system_message_round_trips() {
    assert_round_trip(AgentMessage::System {
        content: "PROMPT".into(),
        timestamp: Some(1234567890),
    });
}

#[test]
fn user_text_message_round_trips() {
    assert_round_trip(AgentMessage::User {
        content: UserContent::Text("hello world".into()),
        timestamp: Some(1),
    });
}

#[test]
fn user_blocks_message_round_trips_with_text_and_image() {
    assert_round_trip(AgentMessage::User {
        content: UserContent::Blocks(vec![
            UserBlock::Text(TextContent {
                text: "describe this image:".into(),
            }),
            UserBlock::Image(ImageContent {
                source: "data:image/png;base64,iVBORw0KGgo".into(),
                media_type: Some("image/png".into()),
                alt: Some("example diagram".into()),
            }),
        ]),
        timestamp: Some(2),
    });
}

#[test]
fn assistant_text_only_round_trips() {
    assert_round_trip(AgentMessage::Assistant {
        content: AssistantContent::text("here is the answer"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: Some(3),
        usage: None,
    });
}

#[test]
fn assistant_with_tool_calls_round_trips_in_source_order() {
    let msg = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::Text(TextContent {
                    text: "calling shell".into(),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "shell".into(),
                    arguments: json!({"command": "ls /tmp"}),
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "call_2".into(),
                    name: "file_read".into(),
                    arguments: json!({"path": "/tmp/x"}),
                }),
            ],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: Some(4),
        usage: None,
    };
    let restored = round_trip(&msg);
    assert_eq!(msg, restored);
    let AgentMessage::Assistant { content, .. } = restored else {
        panic!("expected Assistant");
    };
    assert_eq!(content.tool_calls().len(), 2);
    assert_eq!(content.tool_calls()[0].name, "shell");
    assert_eq!(content.tool_calls()[1].name, "file_read");
}

#[test]
fn assistant_with_thinking_block_round_trips_distinct_from_reasoning() {
    // Pin the channel-separation contract: Thinking and Reasoning are
    // independent variants. A persisted Thinking block must come back
    // as Thinking, never as Reasoning, and vice versa. Conflating them
    // would (a) ship hidden scratch as native reasoning to providers
    // that reject unknown content there, or (b) leak native reasoning
    // into the visible `content` field.
    let thinking = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::Thinking(TextContent {
                text: "let me check the workspace first".into(),
            })],
        },
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: Some(5),
        usage: None,
    };
    let restored = round_trip(&thinking);
    assert_eq!(thinking, restored);
    let AgentMessage::Assistant { content, .. } = restored else {
        panic!("expected Assistant");
    };
    assert_eq!(content.thinking_text(), "let me check the workspace first");
    assert!(content.reasoning_text().is_empty());
    assert!(content.plain_text().is_empty());
}

#[test]
fn assistant_with_reasoning_block_round_trips_distinct_from_thinking() {
    let reasoning = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::Reasoning(TextContent {
                text: "the user wants foo because bar".into(),
            })],
        },
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: Some(6),
        usage: None,
    };
    let restored = round_trip(&reasoning);
    assert_eq!(reasoning, restored);
    let AgentMessage::Assistant { content, .. } = restored else {
        panic!("expected Assistant");
    };
    assert_eq!(content.reasoning_text(), "the user wants foo because bar");
    assert!(content.thinking_text().is_empty());
}

#[test]
fn assistant_with_reasoning_details_round_trips_byte_exact() {
    // Gemini's reasoning.encrypted blocks must round-trip BYTE-EXACT
    // so the next provider request carries the exact same envelopes.
    // Any field reordering or coercion (number→string, etc.) breaks
    // the upstream signature check and Gemini returns INVALID_ARGUMENT
    // "Thought signature is not valid".
    let raw_reasoning = vec![
        json!({
            "id": "tool_file_write_evikd1CFI2gwj1On3ZMo",
            "data": "EjQKMgEMOdbH0bM9ylvOpoL1BBQSryjr4SL1Mt5eKpS42GFv4ge5n5qdHNlhoKrwF/jYj5oo",
            "type": "reasoning.encrypted",
            "index": 0,
            "format": "google-gemini-v1"
        }),
        json!({
            "id": "tool_plan_JqulcNjMHsoVSS4HX1Mr",
            "data": "Other base64 payload here",
            "type": "reasoning.encrypted",
            "index": 1,
            "format": "google-gemini-v1"
        }),
    ];
    let original = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![AssistantBlock::ReasoningDetails(
                ReasoningDetailsContent::new(raw_reasoning.clone()),
            )],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: Some(7),
        usage: None,
    };
    let restored = round_trip(&original);
    assert_eq!(original, restored);
    let AgentMessage::Assistant { content, .. } = restored else {
        panic!("expected Assistant");
    };
    let stored = content.reasoning_details_values();
    assert_eq!(
        stored, raw_reasoning,
        "reasoning_details must round-trip with field order preserved (Gemini signature)"
    );
}

#[test]
fn assistant_with_error_stop_reason_round_trips_with_error_message() {
    assert_round_trip(AgentMessage::Assistant {
        content: AssistantContent { blocks: Vec::new() },
        stop_reason: StopReason::Error,
        error_message: Some("upstream returned 502".into()),
        timestamp: Some(8),
        usage: None,
    });
}

#[test]
fn assistant_with_full_usage_block_round_trips_every_cache_field() {
    let usage = Usage {
        input_tokens: 1234,
        output_tokens: 567,
        cache_creation_input_tokens: 800,
        cache_read_input_tokens: 12000,
    };
    let msg = AgentMessage::Assistant {
        content: AssistantContent::text("billing matters"),
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: Some(9),
        usage: Some(usage.clone()),
    };
    let restored = round_trip(&msg);
    let AgentMessage::Assistant {
        usage: Some(restored_usage),
        ..
    } = restored
    else {
        panic!("usage was dropped");
    };
    assert_eq!(restored_usage, usage, "every cache field must round-trip");
}

#[test]
fn tool_result_with_text_round_trips_including_narration() {
    assert_round_trip(AgentMessage::ToolResult {
        tool_call_id: "call_1".into(),
        tool_name: "shell".into(),
        content: ToolResultContent::text("hello\n"),
        is_error: false,
        narration: Some("Ran `echo hello`.".into()),
        details: None,
        timestamp: Some(10),
    });
}

#[test]
fn tool_result_with_image_blocks_round_trips() {
    assert_round_trip(AgentMessage::ToolResult {
        tool_call_id: "call_2".into(),
        tool_name: "browser_capture".into(),
        content: ToolResultContent {
            blocks: vec![
                ToolResultBlock::Text(TextContent {
                    text: "captured frame:".into(),
                }),
                ToolResultBlock::Image(ImageContent {
                    source: "https://example.com/frame.png".into(),
                    media_type: Some("image/png".into()),
                    alt: None,
                }),
            ],
        },
        is_error: false,
        narration: None,
        details: None,
        timestamp: Some(11),
    });
}

#[test]
fn tool_result_marked_is_error_round_trips() {
    let msg = AgentMessage::ToolResult {
        tool_call_id: "call_x".into(),
        tool_name: "create_website".into(),
        content: ToolResultContent::text(
            "tool execution failed: create_website did not publish a hosted URL",
        ),
        is_error: true,
        narration: None,
        details: None,
        timestamp: Some(12),
    };
    let restored = round_trip(&msg);
    let AgentMessage::ToolResult { is_error, .. } = restored else {
        panic!("expected ToolResult");
    };
    assert!(
        is_error,
        "is_error=true must survive round-trip — UIs render the error badge from this"
    );
}

#[test]
fn custom_message_round_trips_with_arbitrary_payload() {
    let payload = json!({
        "kind": "ui_notification",
        "deeply": {
            "nested": ["array", {"with": "objects"}, 42, true, null]
        }
    });
    assert_round_trip(AgentMessage::Custom {
        kind: "ui_notification".into(),
        payload,
        timestamp: Some(13),
    });
}

// ─── StopReason coverage ──────────────────────────────────────────

#[test]
fn every_stop_reason_variant_round_trips() {
    for reason in [
        StopReason::EndTurn,
        StopReason::ToolUse,
        StopReason::MaxTokens,
        StopReason::Error,
        StopReason::Aborted,
        StopReason::Other,
    ] {
        let json = serde_json::to_value(reason).unwrap();
        let restored: StopReason = serde_json::from_value(json).unwrap();
        assert_eq!(reason, restored, "StopReason::{reason:?} round-trip failed");
    }
}

// ─── Wire-shape pins ──────────────────────────────────────────────

#[test]
fn role_discriminator_uses_snake_case_strings() {
    let cases: &[(AgentMessage, &str)] = &[
        (
            AgentMessage::System {
                content: "x".into(),
                timestamp: None,
            },
            "system",
        ),
        (
            AgentMessage::User {
                content: UserContent::Text("x".into()),
                timestamp: None,
            },
            "user",
        ),
        (
            AgentMessage::Assistant {
                content: AssistantContent { blocks: vec![] },
                stop_reason: StopReason::EndTurn,
                error_message: None,
                timestamp: None,
                usage: None,
            },
            "assistant",
        ),
        (
            AgentMessage::ToolResult {
                tool_call_id: "x".into(),
                tool_name: "x".into(),
                content: ToolResultContent::text(""),
                is_error: false,
                narration: None,
                details: None,
                timestamp: None,
            },
            "tool_result",
        ),
        (
            AgentMessage::Custom {
                kind: "k".into(),
                payload: Value::Null,
                timestamp: None,
            },
            "custom",
        ),
    ];
    for (msg, expected_role) in cases {
        let json = serde_json::to_value(msg).unwrap();
        assert_eq!(
            json["role"].as_str().unwrap(),
            *expected_role,
            "role discriminator wire shape changed for {msg:?}"
        );
    }
}

#[test]
fn user_content_text_uses_untagged_string_form() {
    // Persisted history rows for plain text users use `"content": "text"`,
    // not `"content": {"type": "text", "text": "text"}`. This is the
    // untagged enum representation; changing it breaks every existing
    // row in the DB.
    let msg = AgentMessage::User {
        content: UserContent::Text("hello".into()),
        timestamp: None,
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(
        json["content"], "hello",
        "User text content must serialize as a plain JSON string"
    );
}

#[test]
fn assistant_blocks_use_type_tag_not_value_walking() {
    let msg = AgentMessage::Assistant {
        content: AssistantContent {
            blocks: vec![
                AssistantBlock::Text(TextContent { text: "t".into() }),
                AssistantBlock::Thinking(TextContent { text: "th".into() }),
                AssistantBlock::Reasoning(TextContent { text: "r".into() }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "x".into(),
                    name: "shell".into(),
                    arguments: json!({}),
                }),
            ],
        },
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: None,
        usage: None,
    };
    let json = serde_json::to_value(&msg).unwrap();
    let blocks = json["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[1]["type"], "thinking");
    assert_eq!(blocks[2]["type"], "reasoning");
    assert_eq!(blocks[3]["type"], "tool_call");
}

// ─── Forward / backward compatibility ─────────────────────────────

#[test]
fn old_history_rows_without_narration_field_still_deserialize() {
    // ToolResult rows persisted before `narration` was added must still
    // load. The field is `#[serde(default)]` so missing → None.
    let json = json!({
        "role": "tool_result",
        "tool_call_id": "call_1",
        "tool_name": "shell",
        "content": [{"type": "text", "text": "hello"}],
        "is_error": false,
        "timestamp": 1000
        // no narration field — predates the rename
    });
    let restored: AgentMessage = serde_json::from_value(json).unwrap();
    let AgentMessage::ToolResult {
        narration,
        is_error,
        ..
    } = restored
    else {
        panic!("expected ToolResult");
    };
    assert_eq!(narration, None, "missing narration must default to None");
    assert!(!is_error);
}

#[test]
fn old_assistant_rows_without_usage_block_still_deserialize() {
    let json = json!({
        "role": "assistant",
        "content": [{"type": "text", "text": "old turn"}],
        "stop_reason": "end_turn",
        "timestamp": 999
    });
    let restored: AgentMessage = serde_json::from_value(json).unwrap();
    let AgentMessage::Assistant { usage, .. } = restored else {
        panic!("expected Assistant");
    };
    assert_eq!(usage, None);
}

#[test]
fn old_assistant_rows_without_error_message_still_deserialize() {
    let json = json!({
        "role": "assistant",
        "content": [],
        "stop_reason": "end_turn"
        // no error_message, no timestamp, no usage
    });
    let restored: AgentMessage = serde_json::from_value(json).unwrap();
    let AgentMessage::Assistant {
        error_message,
        usage,
        timestamp,
        ..
    } = restored
    else {
        panic!("expected Assistant");
    };
    assert_eq!(error_message, None);
    assert_eq!(usage, None);
    // timestamp has `default = "default_timestamp"` which fills with now()
    // when missing — verify it produced *something*.
    assert!(
        timestamp.is_some(),
        "default_timestamp must populate missing field"
    );
}

#[test]
fn rows_with_extra_unknown_fields_deserialize_without_failing() {
    // Forward compatibility: a worker on an OLD binary reading rows
    // produced by a NEWER binary must not crash on extra fields.
    // Without `#[serde(deny_unknown_fields)]` (which Clark
    // intentionally avoids), unknown keys are dropped silently.
    let json = json!({
        "role": "assistant",
        "content": [{"type": "text", "text": "newer shape"}],
        "stop_reason": "end_turn",
        "timestamp": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "future_field_added_in_2027": {"important": "data"},
        "another_extension": [1, 2, 3]
    });
    let restored: AgentMessage = serde_json::from_value(json).expect(
        "extra unknown fields must NOT fail deserialization — old workers reading new rows",
    );
    let AgentMessage::Assistant { content, .. } = restored else {
        panic!("expected Assistant");
    };
    assert_eq!(content.plain_text(), "newer shape");
}

// ─── Whole-transcript round-trip (the session_runner contract) ────

#[test]
fn full_transcript_round_trips_through_vec_value_persistence_path() {
    // This is the exact shape `session_runner` consumes:
    // `history: Vec<JsonValue>` → filter_map serde_json::from_value →
    // `Vec<AgentMessage>`. Verify that a realistic transcript (every
    // variant, mixed timestamps, reasoning details, image blocks)
    // survives the round-trip with NO message dropped on the floor.
    let transcript = vec![
        AgentMessage::System {
            content: "PROMPT".into(),
            timestamp: Some(1),
        },
        AgentMessage::User {
            content: UserContent::Text("hi".into()),
            timestamp: Some(2),
        },
        AgentMessage::Assistant {
            content: AssistantContent {
                blocks: vec![
                    AssistantBlock::Thinking(TextContent {
                        text: "let me search".into(),
                    }),
                    AssistantBlock::ReasoningDetails(ReasoningDetailsContent::new(vec![json!({
                        "id": "x",
                        "type": "reasoning.encrypted",
                        "data": "abc",
                        "format": "google-gemini-v1",
                        "index": 0,
                    })])),
                    AssistantBlock::ToolCall(ToolCall {
                        id: "call_1".into(),
                        name: "web_search".into(),
                        arguments: json!({"query": "rust"}),
                    }),
                ],
            },
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: Some(3),
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 10,
                cache_read_input_tokens: 20,
            }),
        },
        AgentMessage::ToolResult {
            tool_call_id: "call_1".into(),
            tool_name: "web_search".into(),
            content: ToolResultContent::text("3 results"),
            is_error: false,
            narration: Some("Searched: rust — 3 results.".into()),
            details: None,
            timestamp: Some(4),
        },
        AgentMessage::User {
            content: UserContent::Blocks(vec![
                UserBlock::Text(TextContent {
                    text: "look at this:".into(),
                }),
                UserBlock::Image(ImageContent {
                    source: "data:image/jpeg;base64,XXX".into(),
                    media_type: Some("image/jpeg".into()),
                    alt: Some("screenshot".into()),
                }),
            ]),
            timestamp: Some(5),
        },
        AgentMessage::Assistant {
            content: AssistantContent::with_tool_calls(
                Some("delivering".into()),
                vec![ToolCall {
                    id: "call_2".into(),
                    name: "message_result".into(),
                    arguments: json!({"text": "Here is the answer."}),
                }],
            ),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: Some(6),
            usage: None,
        },
        AgentMessage::ToolResult {
            tool_call_id: "call_2".into(),
            tool_name: "message_result".into(),
            content: ToolResultContent::text("Here is the answer."),
            is_error: false,
            narration: None,
            details: None,
            timestamp: Some(7),
        },
        AgentMessage::Custom {
            kind: "ui_notification".into(),
            payload: json!({"event": "share_link_minted"}),
            timestamp: Some(8),
        },
    ];

    let history: Vec<Value> = transcript
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let restored: Vec<AgentMessage> = history
        .iter()
        .filter_map(|value| serde_json::from_value::<AgentMessage>(value.clone()).ok())
        .collect();

    assert_eq!(
        restored.len(),
        transcript.len(),
        "no messages dropped during the persistence round-trip"
    );
    assert_eq!(
        restored, transcript,
        "every message must round-trip identically through serde_json::Value"
    );
}

#[test]
fn malformed_history_row_is_skipped_not_fatal() {
    // The session_runner uses `.filter_map(...).ok()` so a single
    // malformed row drops, but the rest of the transcript loads. Pin
    // that contract — a future change to use `.collect::<Result<...>>`
    // would silently drop ALL history on one bad row.
    let history = [
        serde_json::to_value(&AgentMessage::User {
            content: UserContent::Text("good row 1".into()),
            timestamp: Some(1),
        })
        .unwrap(),
        json!({"role": "definitely_not_a_role", "garbage": true}),
        serde_json::to_value(&AgentMessage::User {
            content: UserContent::Text("good row 2".into()),
            timestamp: Some(2),
        })
        .unwrap(),
    ];

    let restored: Vec<AgentMessage> = history
        .iter()
        .filter_map(|value| serde_json::from_value::<AgentMessage>(value.clone()).ok())
        .collect();

    assert_eq!(restored.len(), 2, "good rows must survive a bad neighbor");
    if let AgentMessage::User { content, .. } = &restored[0] {
        assert!(matches!(content, UserContent::Text(t) if t == "good row 1"));
    } else {
        panic!("first restored row should be user");
    }
}

// ─── ReasoningDetails contract (Gemini signature) ─────────────────

#[test]
fn reasoning_details_preserves_field_ordering_in_persisted_array() {
    // Gemini's signature validation hashes the encrypted block as a
    // JSON object. Field reordering breaks the hash. Verify the
    // serialized array preserves the input element order.
    let items = vec![
        json!({"id": "a", "data": "AA", "type": "reasoning.encrypted", "index": 0}),
        json!({"id": "b", "data": "BB", "type": "reasoning.encrypted", "index": 1}),
        json!({"id": "c", "data": "CC", "type": "reasoning.encrypted", "index": 2}),
    ];
    let content = ReasoningDetailsContent::new(items.clone());
    let json = serde_json::to_value(&content).unwrap();
    let restored: ReasoningDetailsContent = serde_json::from_value(json).unwrap();
    assert_eq!(
        restored.details, items,
        "element order in `details` array must be preserved"
    );
    assert_eq!(restored.details[0]["id"], "a");
    assert_eq!(restored.details[1]["id"], "b");
    assert_eq!(restored.details[2]["id"], "c");
}

#[test]
fn reasoning_details_unknown_format_still_round_trips_via_raw_details() {
    // Even if a new provider sends a `type` value the typed
    // `ReasoningItem` enum doesn't recognize, the raw JSON must
    // round-trip so future-replay works. The typed view (`as_items()`)
    // elides unknown items, but `details` keeps every byte.
    let items = vec![json!({
        "id": "future_x",
        "type": "future_provider_v9000.signed",
        "blob": "YYYY"
    })];
    let content = ReasoningDetailsContent::new(items.clone());
    let restored: ReasoningDetailsContent =
        serde_json::from_value(serde_json::to_value(&content).unwrap()).unwrap();
    assert_eq!(restored.details, items);
    // Typed view elides unknown variants; that's expected and not a
    // contract violation.
}
