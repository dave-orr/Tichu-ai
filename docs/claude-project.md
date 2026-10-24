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

Bots: `random`, `heuristic` (rule-based), `pimcN` (search: N guessed
deals × heuristic rollouts per play), `oracleN` (pimc on the real hands),
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

Findings:
- Bots finish 1-2 in ~33% of rounds, likely far above humans: weak defense.
  `npm run export-rounds` in the tichu repo gives the human rate.
- Knowing the hands is worth ~30/deal, so hand inference is the biggest
  lever found so far. The endgame solver helps a little; two-player endgames
  occur in ~65% of rounds and usually solve in under 1 ms.
- Call rules (v2, `src/bots/calls.rs`): winners vs losers for Tichu,
  threshold set to break-even; card-pattern Grand rules frozen.

## Next steps

1. Hand inference: weight `sample_hands` by passes, plays, declined
   beats and calls; measure against the oracle gap.
2. Defense: stop opponents' 1-2s and Tichus (heuristic and rollout
   policy); track the 1-2 rate as a realism check.
3. Better rollout policy (hand decomposition, trick counting), then
   deeper search; endgame solver inside rollouts with a shared cache.
4. Real-game data: compare against human 1-2 / call rates; replay human
   positions as tests.
5. Later: self-play learning, live-server bridge (`docs/AI_API.md` in tichu).
