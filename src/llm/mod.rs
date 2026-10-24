//! Language-model players. `LlmBot` asks a model (headless `claude -p`, on a
//! Claude subscription) for each decision, from a text rendering of its
//! `PlayerView` and a numbered list of the legal options. `run` plays a
//! resumable duplicate match against a bot. See `docs/LLM_MATCH.md`.

pub mod bot;
pub mod client;
pub mod render;
pub mod run;

pub use bot::{Decisions, LlmBot, LlmStats};
pub use client::{Backend, ClaudeCli, Client, OnLimit, Stub};
