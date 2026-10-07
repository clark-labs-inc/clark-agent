    use super::*;
    use crate::plugin::{Plugin, PluginCapabilities};
    use crate::stream::{StreamEvent, StreamFn, StreamRequest};
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use futures::StreamExt;

    struct EmptyStream;
    #[async_trait]
    impl StreamFn for EmptyStream {
        async fn stream(
            &self,
            _r: StreamRequest,
            _s: tokio_util::sync::CancellationToken,
        ) -> BoxStream<'static, StreamEvent> {
            futures::stream::empty().boxed()
        }
    }

    struct ParentOnlyPlugin;
    impl Plugin for ParentOnlyPlugin {
        fn name(&self) -> &'static str {
            "parent_only"
        }
        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::event_observer()
        }
    }
    #[async_trait]
    impl crate::EventObserver for ParentOnlyPlugin {
        async fn on_event(&self, _event: &crate::AgentEvent) {}
    }

    struct InheritablePlugin;
    impl Plugin for InheritablePlugin {
        fn name(&self) -> &'static str {
            "inheritable"
        }
        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::event_observer().with_inheritable_to_child()
        }
    }
    #[async_trait]
    impl crate::EventObserver for InheritablePlugin {
        async fn on_event(&self, _event: &crate::AgentEvent) {}
    }

    #[test]
    fn child_builder_inherits_only_opted_in_plugins() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .event_observer(ParentOnlyPlugin)
            .event_observer(InheritablePlugin)
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        let names = child.plugin_names();
        assert_eq!(
            names.event_observer,
            vec!["inheritable"],
            "child must drop parent-only plugins"
        );
    }

    #[test]
    fn child_builder_carries_sampling_knobs() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .temperature(0.3)
            .max_output_tokens(8192)
            .max_tool_calls_per_turn(3)
            .model_id("test-model")
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        assert_eq!(child.temperature, Some(0.3));
        assert_eq!(child.max_output_tokens, Some(8192));
        assert_eq!(child.max_tool_calls_per_turn, Some(3));
        assert_eq!(child.model_id.as_deref(), Some("test-model"));
    }

    #[test]
    fn child_builder_gets_an_independent_iteration_budget() {
        let parent = AgentBuilder::new()
            .stream(Arc::new(EmptyStream))
            .max_iterations(30)
            .build()
            .expect("parent builds");

        let child = parent.child_builder().build().expect("child builds");

        assert_eq!(parent.max_iterations, Some(30));
        assert_eq!(child.max_iterations, None);
    }
