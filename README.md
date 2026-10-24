# tichu-ai

A fast Tichu rules engine in Rust, plus a harness for writing and evaluating
Tichu bots.

The engine is a port of the production engine behind
[tichu.squidbox.com](https://tichu.squidbox.com) (`shared/src/engine.ts` in
[dave-orr/tichu](https://github.com/dave-orr/tichu)). It follows the same
rules and edge cases (Mah Jong wish, Dog, Dragon giveaway, the trick
countdown during which anyone may bomb), and a differential test checks it
against the real TypeScript code (see [Checking against the TS
engine](#checking-against-the-ts-engine)).

## Quick start

```sh
cargo test --release                              # rule tests
cargo run --release --bin tichu -- bots           # list bundled bots
cargo run --release --bin tichu -- arena heuristic random
cargo run --release --bin tichu -- play heuristic random --seed 3   # narrate one round
cargo run --release --example custom_bot          # write-your-own-bot example
```

Sample output (4 cores):

```
heuristic vs random  (2000 deals)
  margin per deal (heuristic − random): +210.9 ± 2.7  (95% CI)
  heuristic    pts/round  187.9 | first out  98% | 1-2  87% | tichu 305/447 (68%) | grand 9/12 (75%)
  random       pts/round  -23.1 | first out   2% | 1-2   0% | tichu 5/634 (1%) | grand 0/221 (0%)
  (1.0s)
```

## Layout

| | |
|---|---|
| `src/types.rs` | cards, combos, seats, settings. Cards serialize to the server's JSON format |
| `src/combinations.rs` | `identify_combo`, `can_beat`, and `playable_combos`, which lists **every** distinct legal play (suit choices and both readings of an ambiguous Phoenix included) |
| `src/engine.rs` | `GameState`, the rules state machine. `pending()` says who acts next; `legal_plays` / `can_pass` / `legal_bombs` list the options; every accepted `Action` is logged |
| `src/view.rs` | `PlayerView`: what one seat can see. Bots only get this. Has `unseen()`, `known_holdings()` and `sample_hands()` for card tracking and search |
| `src/bot.rs` | the `Bot` trait |
| `src/sim.rs` | plays a round or game between four bots |
| `src/arena.rs` | seeded, duplicate, multithreaded head-to-head evaluation |
| `src/bots/` | `random`, `heuristic` (a simple rule-based baseline), `pimcN` (an example search bot), and `eg:BOT` (any bot plus the exact endgame solver) |
| `src/endgame.rs` | exact alpha-beta solver for two-player endgames (see below) |
| `src/trace.rs`, `tools/ts-diff/` | JSON round traces and the TS replayer |
| `src/replay.rs`, `viewer/` | game logs and the visual replay viewer (see below) |
| `src/llm/`, `examples/llm_match.rs` | Claude as a player (via `claude -p` on a subscription) and a resumable Claude-vs-bot match; see [docs/LLM_MATCH.md](docs/LLM_MATCH.md) |

## Writing a bot

Implement `tichu_engine::bot::Bot`. Each method gets your `PlayerView` and,
where there's a choice, the legal options, so a bot can't make an illegal
move:

```rust
impl Bot for MyBot {
    fn name(&self) -> String { "mybot".into() }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo { ... }
    /// Index into `plays`, or None to pass (only when `can_pass`).
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> { ... }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat { ... }
    // Optional, with defaults: grand_tichu, small_tichu, bomb, wish, round_over
}
```

`examples/custom_bot.rs` shows a complete bot and how to run it in the arena.
To use a bot from the CLI, add it to `bots::by_name` in `src/bots/mod.rs`.

Search bots: `view.sample_hands(rng)` guesses the hidden hands,
`GameState::from_view(&view, hands)` rebuilds a full state from that guess,
and `sim::play_from` plays it out. `src/bots/pimc.rs` is a small worked
example of this.

### Two-player endgames

Once only two players hold cards, each knows the other's hand exactly: it's
every card that isn't in their own hand and hasn't been played. From there
the round is a perfect-information game. `endgame::solve(&state, budget)`
solves it exactly with alpha-beta search and a transposition table, and
returns the round margin under best play plus the best decision. The search
follows the simulator's decision points, bomb offers included, so solved
values are exactly what the arena realizes.

`eg:BOT` (e.g. `eg:pimc20`) wraps any bot and switches to the solver in
those positions. `cargo run --release --example endgame_stats` shows how
often they come up and how hard they are: about two thirds of heuristic
rounds reach one, and the median solve takes 0.3 ms.

### How the simulator schedules decisions

The real server is asynchronous. The simulator turns that into fixed decision
points:

- Grand Tichu and passing are simultaneous. Every seat decides from a view
  taken before anyone's choice is applied.
- Small Tichu is offered to all four seats when play begins, then before
  each of a seat's own turns while it's still allowed.
- Bombs are offered after every play or bomb (to the other active seats, in
  play order), and again when a trick is about to be awarded (the winner
  too). On your own turn, bombs are in `plays` as well.
- Conceding isn't modelled.

## Evaluating bots

`tichu arena A B` plays A (seats 0/2) against B (seats 1/3):

- **Duplicate** (on by default): every deal is played twice, with the bots
  swapping seats, so both sides get the same cards. This removes most of the
  luck of the deal.
- **Round mode** (default, `--rounds N`): independent deals from 0–0, scored
  by point margin. It separates bots far faster than win/loss.
  `--games N` plays full games to 1000 and reports the win rate too.
- The ± is a 95% confidence interval on the mean margin. If it includes 0,
  run more deals.
- Everything is seeded (`--seed`), so a run can be reproduced exactly.
  `--threads` defaults to all cores.

Rough speed on 4 cores: ~4,000 heuristic-vs-heuristic rounds/s;
`pimc20` vs `heuristic` is about 2 deals/s (it wins by +62 ± 8 per deal over 400 deals).

## Watching a game

`tichu replay LOG.json [more logs] --out replay.html` builds one
self-contained HTML page: a card table with all four hands, the trick in
the middle, points and calls per seat, and a play log. Step through it
(← → by play, ↑ ↓ by trick, or the scrubber) or press Play to watch it run.
Where Claude played, its reasoning for each decision sits beside the table,
with the options it was offered and the full prompt.

Game logs come from:

- `tichu arena A B --log DIR [--log-deals N]` (round mode): the first N
  deals (default 20), both seatings, in `DIR/deal-NNNN.json`.
- `examples/llm_match.rs`: `games/deal-NNNN.json` in the match directory
  (Claude's games and the baseline bot's).

A log (`replay::game_log`, `version: 1`) is the deck, the seat names, the
settings, the result, and `events`: every action in order, with each LLM
question and reply (`{"type": "llm", ...}`) just before the action it led
to. `tichu replay` finds every game log inside a file, re-runs each through
the engine and embeds per-step snapshots (hands, table, points, calls), so
the page needs no game logic of its own. `--json` prints that replay data
instead; the page can also open such a file. The card art is the production
client's (`client/public/cards/` in the tichu repo), copied to
`viewer/cards/`.

## Checking against the TS engine

```sh
scripts/ts-diff.sh [ROUNDS] [SEED]     # needs Node 22 and ../tichu (or TICHU_REPO=...)
```

The script plays rounds in Rust (random and heuristic bots) and writes each
as a JSON trace. `tools/ts-diff/diff.mjs` then replays the trace through the
TS engine in `tichu/shared`. At every step it checks:

- the state matches (turn, hands, trick, wish, outs, scores, captured points…)
- TS accepts the action Rust took
- the legal plays Rust lists are exactly the subsets of the hand that TS's
  `playCards` accepts (brute force over every subset and Phoenix reading).
  `canPass`, out-of-turn bombs and Small Tichu eligibility must also agree.

`tools/ts-diff/coverage.py traces.jsonl` counts the rule edge cases a trace
file exercises. Re-run the diff whenever the TS engine's rules change, or
when you touch `engine.rs` / `combinations.rs`.

Known, deliberate difference: TS lets a client claim any `phoenixAs` for a
Phoenix pair or triple (`[3, Phoenix]` as a pair of Aces). Rust always gives
the Phoenix the other card's rank.
