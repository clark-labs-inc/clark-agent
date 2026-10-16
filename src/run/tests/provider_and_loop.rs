use super::*;

#[test]
fn inconsistent_tool_history_stream_kind_stays_typed_at_loop_boundary() {
    let error = loop_error_from_stream_kind(
        StreamErrorKind::InconsistentToolHistory,
        "interleaved tool result batch".into(),
    );

    assert!(matches!(
        error,
        LoopError::Stream(StreamError::InconsistentToolHistory(message))
            if message == "interleaved tool result batch"
    ));
}
#[tokio::test]
async fn context_overflow_is_recovered_by_shrinking_and_retrying() {
    let stream = Arc::new(OverflowThenTextStream::default());
    let recovery_calls = Arc::new(AtomicUsize::new(0));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .overflow_recovery(KeepLastRecovery {
            calls: recovery_calls.clone(),
        })
        .build()
        .expect("config builds");
    let mut context = AgentContext::new("system").with_messages(vec![
        AgentMessage::User {
            content: UserContent::Text("first".to_string()),
            timestamp: None,
        },
        AgentMessage::User {
            content: UserContent::Text("keep me".to_string()),
            timestamp: None,
        },
    ]);

    let (assistant, _allowlist) =
        stream_with_overflow_recovery(&mut context, &config, &CancellationToken::new(), 0)
            .await
            .expect("overflow recovery should retry");

    let AgentMessage::Assistant { content, .. } = assistant else {
        panic!("expected assistant response");
    };
    assert_eq!(content.plain_text(), "recovered after shrink");
    assert_eq!(stream.calls.load(Ordering::SeqCst), 2, "one retry");
    assert_eq!(recovery_calls.load(Ordering::SeqCst), 1);
    // The shrink is persisted into the live transcript…
    assert_eq!(context.messages.len(), 1);
    // …and the retried request sent only the shrunk history.
    let retried = &stream.requests.lock().unwrap()[1];
    assert_eq!(retried.messages.len(), 1);
}
#[tokio::test]
async fn context_overflow_recovery_outlives_the_legacy_two_attempt_cap() {
    let stream = Arc::new(OverflowThenTextStream::new(3));
    let recovery_calls = Arc::new(AtomicUsize::new(0));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .overflow_recovery(KeepLastRecovery {
            calls: recovery_calls.clone(),
        })
        .build()
        .expect("config builds");
    let mut context = AgentContext::new("system").with_messages(
        ["first", "second", "third", "keep me"]
            .into_iter()
            .map(|text| AgentMessage::User {
                content: UserContent::Text(text.to_string()),
                timestamp: None,
            })
            .collect(),
    );

    let (assistant, _allowlist) =
        stream_with_overflow_recovery(&mut context, &config, &CancellationToken::new(), 0)
            .await
            .expect("fourth stream attempt should recover");

    let AgentMessage::Assistant { content, .. } = assistant else {
        panic!("expected assistant response");
    };
    assert_eq!(content.plain_text(), "recovered after shrink");
    assert_eq!(stream.calls.load(Ordering::SeqCst), 4);
    assert_eq!(recovery_calls.load(Ordering::SeqCst), 3);
    assert_eq!(context.messages.len(), 1);
}
#[tokio::test]
async fn context_overflow_without_recovery_propagates() {
    let stream = Arc::new(OverflowThenTextStream::default());
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .build()
        .expect("config builds");
    let mut context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
        content: UserContent::Text("hi".to_string()),
        timestamp: None,
    }]);

    let result =
        stream_with_overflow_recovery(&mut context, &config, &CancellationToken::new(), 0)
            .await;
    assert!(matches!(
        result,
        Err(LoopError::Stream(StreamError::ContextOverflow(_)))
    ));
    assert_eq!(stream.calls.load(Ordering::SeqCst), 1, "no retry");
}
#[test]
fn done_is_complete() {
    assert!(LoopOutcome::Done.is_complete());
}
#[tokio::test(start_paused = true)]
async fn empty_stream_recovers_within_the_shared_attempt_budget() {
    let stream = Arc::new(EmptyThenTextStream::new(2));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .build()
        .expect("config builds");
    let context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
        content: UserContent::Text("continue".to_string()),
        timestamp: None,
    }]);

    let (assistant, _allowlist) =
        stream_with_max_tokens_recovery(&context, &config, &CancellationToken::new(), 0)
            .await
            .expect("third stream attempt should recover");

    let AgentMessage::Assistant { content, .. } = assistant else {
        panic!("expected assistant response");
    };
    assert_eq!(content.plain_text(), "recovered");
    assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
}
#[tokio::test(start_paused = true)]
async fn provider_rate_limit_outer_retry_has_a_hard_attempt_ceiling() {
    let stream = Arc::new(ProviderRateLimitedStream::immediate());
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .build()
        .expect("config builds");
    let context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
        content: UserContent::Text("continue".to_string()),
        timestamp: None,
    }]);

    let result =
        stream_with_max_tokens_recovery(&context, &config, &CancellationToken::new(), 0).await;

    assert!(matches!(
        result,
        Err(LoopError::Stream(StreamError::ProviderRateLimited(message)))
            if message.contains("shared pool")
    ));
    assert_eq!(
        stream.calls.load(Ordering::SeqCst),
        provider_recovery::PROVIDER_RECOVERY_MAX_ATTEMPTS as usize
    );
}
#[tokio::test(start_paused = true)]
async fn provider_stall_reports_current_attempt_deadline_not_previous_rate_limit() {
    let stream = Arc::new(ProviderRateLimitedStream::then_stall());
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .build()
        .expect("config builds");
    let context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
        content: UserContent::Text("continue".to_string()),
        timestamp: None,
    }]);
    let started_at = tokio::time::Instant::now();

    let result =
        stream_with_max_tokens_recovery(&context, &config, &CancellationToken::new(), 0).await;

    assert!(matches!(
        result,
        Err(LoopError::Stream(StreamError::RequestTimeout { timeout_ms: 900_000 }))
    ));
    assert_eq!(stream.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        started_at.elapsed(),
        std::time::Duration::from_millis(900_500)
    );
}
#[tokio::test]
async fn zero_output_transport_error_is_retried_before_returning() {
    let stream = Arc::new(ZeroOutputThenTextStream::default());
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .reasoning(ReasoningEffort::High)
        .build()
        .expect("config builds");
    let context = AgentContext::new("system").with_messages(vec![AgentMessage::User {
        content: UserContent::Text("continue".to_string()),
        timestamp: None,
    }]);

    let (assistant, _allowlist) =
        stream_with_max_tokens_recovery(&context, &config, &CancellationToken::new(), 0)
            .await
            .expect("second zero-output transport attempt should recover");

    let AgentMessage::Assistant { content, .. } = assistant else {
        panic!("expected assistant response");
    };
    assert_eq!(content.plain_text(), "recovered from transport");
    assert_eq!(stream.calls.load(Ordering::SeqCst), 2);

    let requests = stream.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].reasoning, ReasoningEffort::High);
    assert_eq!(
        requests[1].reasoning,
        ReasoningEffort::High,
        "transport recovery must preserve the caller reasoning policy"
    );
    assert!(
        requests[1].messages.iter().any(|message| matches!(
            message,
            AgentMessage::System { content, .. }
                if content.contains("transport recovery")
                    && content.contains("no visible assistant text")
                    && content.contains("no usable tool call")
                    && content.contains("latest user request")
                    && content.contains("ordinary final assistant response")
                    && !content.contains("final response tool")
        )),
        "zero-output replay must carry explicit recovery context"
    );
}
#[tokio::test]
async fn max_iterations_stops_a_nonterminating_tool_loop() {
    let stream = Arc::new(RepeatingToolStream {
        calls: AtomicUsize::new(0),
    });
    let tools = crate::tool::ToolRegistry::new().with(Arc::new(NonTerminatingTool));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .tools(tools)
        .max_iterations(3)
        .build()
        .expect("config builds");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("work".to_string()),
        timestamp: None,
    }];

    let result = run(
        prompts,
        AgentContext::new("system"),
        &config,
        CancellationToken::new(),
    )
    .await
    .expect("iteration cap is a typed outcome");

    assert_eq!(stream.calls.load(Ordering::SeqCst), 3);
    assert_eq!(result.outcome, LoopOutcome::HitMaxIterations);
    assert!(!result.outcome.is_complete());
}
#[tokio::test]
async fn terminator_vote_skips_post_batch_steering_collection() {
    // Regression: a `SteeringSource` whose firing condition lines
    // up with the same turn the model delivers (e.g.
    // `graceful_turn_limit` reaching its soft limit on the delivery
    // turn) used to re-enter the loop and prompt the model for
    // ANOTHER turn after a clean terminator. The model's drift on
    // that extra turn corrupted the user-visible answer in
    // production. With the fix, a unanimous terminator vote is a
    // hard exit — steering sources are not polled once the run has
    // decided it's done.
    let stream = Arc::new(TerminatorOnlyStream::default());
    let polls = Arc::new(AtomicUsize::new(0));
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry.with(Arc::new(TerminatorTool));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .tools(tool_registry)
        .steering(AlwaysSteer {
            polls: polls.clone(),
        })
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("deliver".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect("run completes after one terminator turn");

    // Exactly one LLM call — the terminator turn.
    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    // Outcome is a clean natural completion.
    assert_eq!(result.outcome, LoopOutcome::Done);
    // Steering source is consulted exactly once — the pre-loop
    // priming poll at the top of `inner_run`. After the terminator
    // batch, `collect_steering` MUST NOT fire again.
    assert_eq!(
        polls.load(Ordering::SeqCst),
        1,
        "steering source polled more than once — terminator vote did not gate post-batch re-entry"
    );
}
#[tokio::test]
async fn terminator_vote_skips_post_batch_follow_up_collection() {
    // Mirror of the steering test for the follow-up source path.
    // `FollowUpSource` exists to nudge the model toward a
    // terminator when it failed to emit one — not to overrule a
    // terminator the model already cast. After a clean delivery,
    // follow-up must be silent.
    let stream = Arc::new(TerminatorOnlyStream::default());
    let polls = Arc::new(AtomicUsize::new(0));
    let mut tool_registry = crate::tool::ToolRegistry::new();
    tool_registry = tool_registry.with(Arc::new(TerminatorTool));
    let config = AgentBuilder::new()
        .stream(stream.clone())
        .model_id("test-model")
        .tools(tool_registry)
        .follow_up(AlwaysFollowUp {
            polls: polls.clone(),
        })
        .build()
        .expect("config builds");
    let context = AgentContext::new("system");
    let prompts = vec![AgentMessage::User {
        content: UserContent::Text("deliver".to_string()),
        timestamp: None,
    }];

    let result = run(prompts, context, &config, CancellationToken::new())
        .await
        .expect("run completes after one terminator turn");

    assert_eq!(stream.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.outcome, LoopOutcome::Done);
    assert_eq!(
        polls.load(Ordering::SeqCst),
        0,
        "follow-up source polled after a terminator vote — terminator did not gate post-batch re-entry"
    );
}
#[tokio::test]
async fn empty_tool_gate_intersection_prefers_delivery_repair_owner() {
    let (sink, mut rx) = crate::event::ChannelSink::new();
    let config = AgentBuilder::new()
        .stream(Arc::new(RepeatedTextStream::default()))
        .event_sink(Arc::new(sink))
        .tool_gate_arc(Arc::new(StaticAllowGate {
            name: "delivery_repair_gate",
            tools: &["browser_interact"],
            priority: 100,
            class: ToolGateClass::Required,
            suppresses_advisory: false,
        }))
        .tool_gate_arc(Arc::new(StaticAllowGate {
            name: "terminal_message_guard",
            tools: &["message_result"],
            priority: 10,
            class: ToolGateClass::Required,
            suppresses_advisory: false,
        }))
        .build()
        .expect("config builds");

    let allow = collect_tool_allowlist_with_events(&config, 3, &[])
        .await
        .expect("conflict repair should keep a non-empty allowlist");

    assert_eq!(
        allow,
        ["browser_interact".to_string()].into_iter().collect()
    );

    let mut saw_conflict = false;
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ToolGateConflictResolved {
            chosen_plugin,
            allow,
            ..
        } = event
        {
            saw_conflict = true;
            assert_eq!(chosen_plugin.as_deref(), Some("delivery_repair_gate"));
            assert_eq!(allow, vec!["browser_interact".to_string()]);
        }
    }
    assert!(saw_conflict, "tool-gate deadlock should be diagnosable");
}
