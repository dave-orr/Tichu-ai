# Endgame audit and human disagreements (Oct 2026)

Two questions: where do our bots make clear, provable mistakes, and what can
real games tell us that self-play can't?

Tools (both in `examples/`, both read `npm run export-rounds` files through
`src/humanlog.rs`, which rebuilds a logged round through the engine):

```sh
# Exact two-player endgames: regret of each bot vs best play
cargo run --release --example endgame_audit -- --human rounds.jsonl \
    --selfplay h2 --selfplay net:models/gen3.tnet --rounds 600 \
    --bots heuristic,h2,net:models/gen3.tnet,pimc10/h --out out/eg
# Every human decision: net / h2 choice vs the human's, scored by paired rollouts
cargo run --release --example human_disagree -- rounds.jsonl \
    --net models/gen3.tnet --samples 100 --out out/hd
```

Models: `gen3.tnet` is the 3rd generation of the Oct 25 overnight expert
iteration run (feature v1, the one main can load).

## Data

- Human: the Oct 24 export, 148 rounds, of which 22 (games 24 to 28) carry a
  play log. All 22 rebuild exactly (finishing order and card points match
  the log; 5 end in a concession, which the log shows as plays stopping).
  That's 1,311 human on-turn decisions, 763 with 2+ distinct options. The
  log doesn't record passes, the countdown, when a Small Tichu was called,
  the table's direction or the Phoenix's reading in `5 5 Ph K K`;
  `humanlog::replay` infers them (see its doc comment) and
  `tests/humanlog.rs` checks the inference by round-tripping 160 bot rounds
  through the log format.
- Self-play: the desktop's self-play files are feature records, not game
  logs, so positions were regenerated: 600 rounds of h2 and 600 of gen3 in
  all seats.

## 1. Two-player endgames (exact)

Once two players hold cards the round is perfect-information, and
`endgame::solve_options` (new: the exact value of every option, not just the
best) scores each decision. Regret = points of round margin lost vs best
play, on the decider's side. Only decisions where the options differ in
value are counted.

Self-play positions (1,634 decisions that matter, from the ~825 of 1,200
rounds that reach a two-player endgame):

| bot | optimal | regret / decision | regret >= 100 |
|---|---|---|---|
| heuristic | 76% | 14.0 | 88 |
| h2 | 82% | 9.4 | 53 |
| net gen0 | 79% | 10.8 | 61 |
| net gen3 | 81% | 9.4 | 47 |
| pimc10/h | 87% | 7.2 | 41 |

The mistakes aren't confined to hard positions. With **6 or fewer cards
left between the two players**, gen3 is still wrong 15% of the time (h2 12%,
pimc10/h 4%). Typical ones:

- Me `3 J A`, opponent `5`, I lead: gen3 leads the 3, the opponent goes out,
  I finish last (−100 vs leading J or A). Same with `4 T` vs `7`.
- Me `9 9 Q`, opponent `J J`, I lead: gen3 leads the pair, which the
  opponent beats and goes out on.
- Me `7 Q` vs `J Q`: gen3 leads the 7 and loses the lead for good (−70).

By type, gen3's 307 losing choices: wrong lead 138 (half of them the wrong
combination type), passing when it should play 76, wrong follow 60, playing
when it should pass 33. pimc10/h almost only errs by passing when it should
take the trick (119 of 211), a rollout-policy artifact: its h2 rollouts
misplay the continuation, so taking control looks worse than it is.

What it's worth: summing per-decision regret overstates the gain (both sides
err, and errors along one line overlap). The arena measures it directly:
**`eg:net:gen3` vs `net:gen3`: +4.4 ± 0.5 per deal (3,000 duplicate
deals)**, in line with `eg:heuristic` vs heuristic (+3.4). Comparable to
Dave's single play rules (+4 to +10).

Human positions: only 20 decisions that matter in 11 rounds. Humans were
optimal in 19 (one 30-point error), gen3 in 18, h2 in 16, pimc10/h in all
20. Too few to rank anyone, but no sign humans are weak here.

## 2. Human vs model disagreements (all decisions)

For every human decision with 2+ distinct options, each option's Q is its
mean round margin over 100 deals guessed from the decider's view, rolled out
by gen3 in all seats, with the same guesses for every option (paired), plus
the standard error of each gap. Two-player endgames use the exact value.

| | net gen3 | h2 | Q argmax |
|---|---|---|---|
| agrees with the human (top-1) | 60.4% | 64.2% | 46.8% |
| leading (224) | 54.0% | 55.8% | 31.2% |
| following (539) | 63.1% | 67.7% | 53.2% |

Net log-loss on the human's move: 1.288.

Mean Q shortfall vs the best option: **human 12.6, net 12.8, h2 16.6**
points per decision. Q comes from the net's own rollouts, which if anything
favors the net, so in these 22 rounds the humans play at least as well as
gen3, and clearly better than h2.

The net and the human disagree on 302 decisions (40%). Counting a gap only
when it's at least 20 points and 2 standard errors (or exact):

- **human clearly worse: 41.** Mostly passing when they should take the
  trick (`A A` on a pair of 10s with 20 points in it; the Dragon on a
  Phoenix), and leading a low single or the Dog instead of a straight.
- **net clearly worse: 39.** Mostly about going out: leading the Dog with
  `Dog Dr` and partner on one card instead of Dragon then Dog (a 1-2 worth
  +340); not winning with the master card when the partner is out and a 1-2
  is on (`J` on `2` with `7 J` left, +120); breaking a winning hand into the
  wrong straight.
- too close to call: 222.

Expect a handful of each list to be noise (2 SE over ~300 tests).

## What to do with it

1. **Exact endgame labels for training.** About two thirds of rounds reach
   a two-player endgame, and the solver labels those decisions exactly and
   cheaply (median solve 0.3 ms, per `endgame_stats`). In
   `selfplay --q`, use `solve_options` values instead of PIMC Q there:
   exact targets with no rollout noise, and they'd teach the net the
   endgame habits it visibly lacks (don't lead a beatable single into an
   opponent's last card). Also run rollouts with `eg:` so the search's
   labels for earlier decisions see correct endgames. At play time, wrap
   the net in `eg:` (+4.4/deal today).
2. **Endgame regret as a fixed eval.** The audited positions are a cheap,
   exact, noise-free regression test for every new net: optimal rate and
   regret per decision on a frozen position set, overall and for ≤6 cards.
   A net that still leads the 3 into a one-card opponent isn't done.
3. **Human games as an eval, not (yet) training data.** 763 decisions is
   far too few to train on, and the net's own search already prefers the
   human's move in most of the "net clearly worse" cases, so expert
   iteration should fix those by itself. Use them as: (a) a human-move
   agreement / log-loss benchmark (today 60% / 1.29), useful for an opponent
   model for inference, where predicting people is the point; (b) the two
   curated lists as spot checks a new model should get right. Revisit
   training on human positions once there are thousands of logged rounds:
   rerun `npm run export-rounds` (play logs started with game 24, and every
   round adds ~60 decisions).

Output from these runs (readable lists of every costly decision) is in the
project files under `endgame-audit/`.
