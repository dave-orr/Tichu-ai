# tichu-ai: Claude project brief

Paste the "Instructions" section into the project's instructions; attach this
whole file (and README.md) as project knowledge.

## Instructions

We're building a strong Tichu-playing bot.

- Code: `dave-orr/tichu-ai` (Rust engine + bot harness). The rules authority
  is the production engine in `dave-orr/tichu` (`shared/src/engine.ts`,
  `RULES.md`). The Rust engine is a verified port: after touching
  `engine.rs` / `combinations.rs`, run `cargo test --release` and
  `scripts/ts-diff.sh 100`.
- Bots implement `Bot` (src/bot.rs) and only see a `PlayerView`.
  `oracle*` bots cheat on purpose, for diagnostics only.
- Judge every change with `tichu arena A B` (duplicate deals, margin per
  deal). Always report the ± and the number of deals. Don't claim a gain
  whose confidence interval includes 0.
- Priority is play strength. The call rules are frozen for now (see below).
- Commit on a branch and push. Open PRs only when asked.

## Where things stand (Oct 2026)

Bots: `random`, `heuristic` (rule-based baseline), `h2` (stronger rules:
card tracking, hand structure, lead planning, Tichu/1-2 defense), `pimcN`
(search: N guessed deals × rollouts per play; flags `/h` h2 rollouts, `/i`
hand inference, `/b` searched bombs, `/s` searched wish and Dragon gift),
`oracleN` (pimc on the real hands),
wrappers `eg:` (exact two-player endgame solver), `calls:P:` (swap call
policy), `nocalls:`.

Results (duplicate deals):

| match | margin/deal | deals |
|---|---|---|
| heuristic vs random | +211 ± 3 | 2000 |
| pimc20 vs heuristic | +62 ± 8 | 400 |
| oracle20@0.1 vs pimc20@0.1 (value of knowing the hands) | +32 ± 9 | 300 |
| eg:heuristic vs heuristic | +3.4 ± 0.5 | 2000 |
| eg:pimc10 vs pimc10 | +12 ± 10 | 300 |
| nocalls:heuristic vs heuristic (value of v1 calls) | −2.9 ± 0.9 | 4000 |
| h2 vs heuristic | +40.0 ± 3.4 | 4000 |
| pimc10/h vs pimc10 (h2 rollouts) | +23.8 ± 11.4 | 400 |
| pimc10@0.1/h vs pimc10/h (rollout noise) | −0.1 ± 8.9 | 600 |
| oracle10@0.1/h vs pimc10@0.1/h (value of knowing the hands) | +61.3 ± 8.7 | 600 |
| pimc10@0.1/hi vs pimc10@0.1/h (inference) | −0.5 ± 8.9 | 600 |
| pimc10@0.1/hbs vs pimc10@0.1/h (searched bombs, wish, Dragon) | +3.6 ± 9.1 | 600 |

Findings:
- Bots finish 1-2 in ~33% of rounds, likely far above humans: weak defense.
  `npm run export-rounds` in the tichu repo gives the human rate.
- Knowing the hands is worth ~30/deal, so hand inference is the biggest
  lever found so far. The endgame solver helps a little; two-player endgames
  occur in ~65% of rounds and usually solve in under 1 ms.
- Call rules (v2, `src/bots/calls.rs`, the default since Oct 2026): winners
  vs losers for Tichu at threshold 0; card-pattern Grand rules frozen. With
  pimc10 play, the v2 Tichu rule at 0 vs v1 is +4.8 ± 6.1 over 800 deals.
- Real games (148 rounds, 29 games; `examples/human_calls.rs`): humans
  finish 1-2 in 22% of rounds (bots 33%), call Tichu at 15% of seats
  (made 63%) and Grand at 10.5% (made 56%), and bomb 0.30 times per round.
  They call Tichu on the same hands the v2 margin rates highly (mostly
  margin ≥ −1) and make 75–85% of those; pimc makes ~45% of the same
  hands. Humans called Grand on all 7 hands that met the 3-tops rule (made
  6), but bots make 50%. The gap is in play, not in hand evaluation.

Findings from the Oct 2026 play work:
- Most of h2's gain came from card tracking plus the hand split. One rule
  backfired badly: steering leads away from lengths a near-out opponent can
  follow cost ~24/deal (`h2~danger=4` vs `h2`), so it's off. Other knobs
  (`h2~fight_pts=...` etc.) moved results by under 3/deal.
- With exact rollouts (no noise) the oracle is *worse* than pimc
  (oracle10/h vs pimc10/h −12.8 ± 13.9, 300 deals): searching one fixed
  deal against a deterministic model overfits. Inference has to keep the
  guesses varied, so it now draws distinct guesses (no duplicates).
- Inference (`infer.rs`) replays the round from each guessed deal and
  down-weights guesses whose h2 model disagrees with what players did.
  Hidden cards placed right (`examples/infer_check`): vs h2 players 56% →
  64%, vs heuristic 52% → 59%, but vs pimc players only 48.5% → 50.7%. The
  model doesn't predict search players well, which is why it doesn't help
  pimc yet. The oracle gap (+61) says it's still the biggest lever.

## Next steps

1. Hand inference that works against search players: a better opponent
   model (or one learned from pimc / human play), and use passes received;
   measure against the oracle gap.
2. Keep improving h2, since it is the rollout policy: every gain there
   also lifts pimc.
3. Deeper search; endgame solver inside rollouts with a shared cache.
4. Real-game data: compare against human 1-2 / call rates; replay human
   positions as tests.
5. Later: self-play learning, live-server bridge (`docs/AI_API.md` in tichu).
