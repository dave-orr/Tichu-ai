# tichu-ai notes

Rust Tichu engine + bot harness. See README.md for layout and usage.

- The rules authority is the production TS engine in dave-orr/tichu
  (`shared/src/engine.ts`, `combinations.ts`) and its `RULES.md`. `engine.rs`
  and `combinations.rs` are ports of it: keep them in step with it, quirks
  included. Don't "fix" rules from memory.
- After touching `engine.rs` or `combinations.rs`, run `cargo test --release`
  and `scripts/ts-diff.sh 100` (needs the tichu repo at ../tichu or
  `TICHU_REPO`). The diff must report "Rust and TS agree on everything".
- Bots only ever see a `PlayerView`, never `GameState`. Keep it that way, so
  no bot can cheat by accident. `GameState::from_view` is for search bots
  that rebuild a state from a guess at the hidden hands. The one exception
  is the opt-in `Bot::cheats`/`peek` hook, for diagnostic oracles only
  (`oracle`); never use it in a bot meant to play.
- Evaluate with `tichu arena A B` (duplicate, round mode). Quote the ± and
  the number of deals with any result.
