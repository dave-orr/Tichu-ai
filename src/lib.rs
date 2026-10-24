//! Tichu engine and bot-testing harness.
//!
//! - `engine`: the rules (a faithful port of the production TS engine)
//! - `view`: what one seat can see; the only thing bots receive
//! - `bot`: the `Bot` trait; `bots`: bundled bots
//! - `sim`: plays rounds/games between four bots
//! - `arena`: seeded, duplicate, multithreaded head-to-head evaluation
//! - `endgame`: exact solver for two-player endgames
//! - `trace`: JSON round traces (used to diff against the TS engine)
//! - `llm`: language-model players and a resumable match runner
//! - `replay`: game logs and the visual replay viewer

pub mod arena;
pub mod bot;
pub mod bots;
pub mod combinations;
pub mod deck;
pub mod endgame;
pub mod engine;
pub mod hand;
pub mod llm;
pub mod replay;
pub mod scoring;
pub mod sim;
pub mod trace;
pub mod types;
pub mod view;
