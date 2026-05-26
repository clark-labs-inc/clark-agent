//! Built-in plugins shipped with `clark-agent`.
//!
//! These cover safety-net concerns that every product variant needs and
//! that would otherwise have to be re-implemented per call-site. Each
//! plugin is a regular [`Plugin`](crate::plugin::Plugin) implementation —
//! nothing magic about its placement here.

pub mod graceful_turn_limit;
pub mod opening_gate;

pub use graceful_turn_limit::GracefulTurnLimit;
pub use opening_gate::OpeningGate;
