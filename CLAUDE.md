# Tichu Bot — Autoresearch Guide

## Goal

Optimize the ISMCTS Tichu bot to maximize win rate. You may modify both the heuristic logic (used in ISMCTS rollouts) and the ISMCTS search parameters.

## How to Evaluate

```bash
cargo build --release && cargo run --release -- autoresearch --games 100
```

This runs the current bot (Team A) against the baseline bot (Team B) with seat-swapping. Both use 50 MCTS iterations per decision for speed.

**Success criterion:** `win_rate_a > 0.53` AND `significant=true`

Output is parseable:
```
games=200
wins_a=...
wins_b=...
draws=...
win_rate_a=0.XXXX
significant=true/false
```

## Files You CAN Modify

Everything in `src/bots/`:

### Heuristic Bot (`src/bots/heuristic/`)
These drive the ISMCTS rollout policy — better heuristics = better ISMCTS.

- **`constants.rs`** — Tunable weights: `HeuristicWeights` struct with defaults. Hand evaluation weights, leading/following thresholds, tichu calling thresholds.
- **`eval.rs`** — Hand analysis: `analyze_hand()` computes strength score, `estimate_tempo()` counts minimum plays to empty hand. Uses `find_playable_combos()` from the engine.
- **`lead.rs`** — Leading strategy: what to play when starting a trick. Currently: Dog timing, low-lead on strong hands, largest combo on weak hands.
- **`follow.rs`** — Following strategy: minimum beat, partner awareness, bomb management. Uses `total_points()` from scoring module for trick value.
- **`pass_strategy.rs`** — Card passing: what 3 cards to pass left/partner/right. Currently basic (never pass dragon/phoenix/bombs).
- **`tichu_call.rs`** — Grand/Small Tichu decisions based on hand strength and tempo.
- **`mod.rs`** — Wires submodules into `HeuristicBot`. Also has dragon giveaway and MahJong wish logic.

### ISMCTS Bot (`src/bots/ismcts/`)
- **`config.rs`** — `ISMCTSConfig`: `num_iterations` (default 200), `exploration` constant (default 1.41).
- **`mod.rs`** — Main ISMCTS loop: determinize → select (UCB1) → expand → rollout → backpropagate.
- **`determinize.rs`** — Redistributes unknown opponent cards randomly, preserving hand sizes.
- **`rollout.rs`** — Plays out round to completion using heuristic policy. Returns +1/-1/0 based on round outcome.
- **`tree.rs`** — Arena-allocated MCTS tree. UCB1 selection, expansion, backpropagation.

## Files You MUST NOT Modify

- `src/engine.rs` — Game state machine
- `src/types.rs` — Core types (Card, Combo, Phase, etc.)
- `src/combinations.rs` — Combo detection and validation
- `src/scoring.rs` — Point calculation
- `src/deck.rs` — Deck creation
- `src/runner.rs` — Game runner
- `src/tournament.rs` — Tournament infrastructure
- `src/player.rs` — Player trait
- `src/harness.rs` — Training harness
- `src/main.rs` — CLI
- `Cargo.toml` — Dependencies

## Tichu Rules (Brief)

Tichu is a 4-player partnership trick-taking card game (teams: seats 0+2 vs 1+3).

**Cards:** 56 total — 4 suits × 13 ranks (2-A) + 4 specials (MahJong, Dog, Phoenix, Dragon).
**Combinations:** Single, Pair, Triple, Straight (5+), Full House, Consecutive Pairs, Bombs (4-of-a-kind or straight flush).
**Goal:** Go out first (empty your hand). First team to 1000 points wins.

**Key mechanics:**
- Must beat the current trick with same type + higher rank, or bomb
- MahJong holder leads first and can wish for a rank
- Dragon (strongest single) — if your trick, must give it to an opponent
- Dog — gives lead to your partner
- Phoenix — wildcard, acts as 0.5 above current single
- Grand Tichu (+/-200 pts): called before seeing all cards
- Small Tichu (+/-100 pts): called before your first play
- Double victory (both partners out first): 200-0 score

**Scoring per round:** 5s=5pts, 10s/Ks=10pts, Dragon=+25, Phoenix=-25. Total deck = 100pts.

## Strategy Ideas to Try

### Heuristic Improvements
- Better hand decomposition: find the optimal grouping of cards into plays (not just greedy largest-first)
- Positional awareness: play differently based on who has gone out, who has few cards
- Adaptive play based on score differential (play conservatively when ahead, aggressively when behind)
- Smarter passing: analyze which cards break combinations, pass cards that help partner
- Better tichu calling: consider card distribution, not just raw strength
- End-game awareness: when 2 players are out, adjust strategy

### ISMCTS Improvements
- Tune exploration constant (current: sqrt(2), often too high)
- Progressive widening: don't expand all children at once
- Rollout depth limit: stop rollout after N moves, use heuristic evaluation
- Better reward signal: use point differential instead of binary win/loss
- Determinization bias: weight opponent hand samples by likelihood
- Move ordering in expansion: try promising moves first

### Performance
- Reduce calls to `find_playable_combos()` in rollouts (major bottleneck)
- Cache combo computations
- Early termination: if one action dominates, stop searching
