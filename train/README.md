# Training learned bots

Self-play in Rust, training in PyTorch, inference back in Rust, judged with
`tichu arena`. Only card play is learned; calls, passing, out-of-turn bombs,
wishes and the Dragon stay with the heuristic.

```
tichu selfplay ──► records (.bin) ──► train.py ──► model (.tnet) ──► net:PATH / vpimcN:PATH
```

| | |
|---|---|
| `src/features.rs` | fixed-size byte features of a `PlayerView` (704) and of each candidate play (48), and the round's history as tokens (48 bytes each). Seats are relative, in play order |
| `src/selfplay.rs` | `tichu selfplay`: plays rounds, writes one record per play decision with 2+ options (plus each seat's history tokens once per round) |
| `train/tichu_data.py` | reads record files into numpy arrays |
| `train/train.py` | trains the net, writes `.tnet` (and a `.pt` checkpoint beside it) |
| `train/iterate.py` | the expert-iteration loop: label → train → arena, generation after generation |
| `src/bots/net.rs` | loads `.tnet` (no dependencies), the `net:` and `vpimcN:` bots |

## The model

A shared trunk over the state (704 [+ 64] → 256 → 256, ReLU) with two heads:

- **value**: expected round margin for my team (my team's points − theirs),
  trained by MSE against the final margin of the round the position came
  from. Output is in units of `--value-scale` (100 points). Positions of
  one round share its margin, so a head on the trunk memorizes rounds
  (validation R² falls from ~0.3 to ~0.1 as training goes on, while train
  R² reaches 0.8). The exported value is a separate net over the state
  (`--value-net 128`, dropout 0.3, weight decay 1e-2), kept at its best
  validation MSE; the trunk's head is an auxiliary loss (`--value-weight
  0.25`) that's not exported. On 3k rounds this took validation R² from
  0.10 to 0.28, with the same policy loss.
- **policy**: one logit per candidate from `trunk ⊕ action features`, so a
  turn can have any number of legal plays. Softmax over the candidates.
  Plays that differ only in suits have identical features and are merged.

The extra 64 inputs are a **sequence encoder**: a one-layer GRU over the
round's history, one token per action (who led or followed with what, who
passed on what and against whom, who won each trick and its points, Dragon
gifts, Tichu calls, the wish). It's off by default (`--seq-hidden 64` turns
it on): at gen0 on 16k rounds it was even with the flat net (+0.3 ± 2.8 over
4000 deals) and made net rollouts 30% slower. Rust keeps each bot's walk of the log and GRU states between
decisions, so a growing history (or rollouts from one position) only runs
the new tokens.

What the state features hold beyond the cards (feature version 2): the card
each seat passed me; per seat and combo kind, the lowest rank they passed on
while an opponent held the trick; leads by kind; bomb threats per seat (four
of a kinds expected, P(any bomb), strongest bomb possible) from card tracking
with the unknown cards dealt at random; h2's plan of my hand (combos to go
out, bosses, losers, a clear path); and whether the trick on the table can
still be beaten. Each candidate also gets whether it can be beaten (for a
bomb: by a stronger bomb an opponent might hold), how many unseen ranks beat
it, and h2's plan of the hand that's left. The random-deal bomb odds rank
seats well but run about 3× low (`cargo test --release bomb_threats --
--ignored --nocapture`): passing makes bombs far more common (5% of hands
before the pass, 13% after), and the net learns the scale.

Rust runs it with a hand-written dense forward pass. The net is tiny, inputs
are mostly zeros, and it gets called inside search, so this beats an ONNX
runtime on latency and needs no native libraries. Every `.tnet` carries one
worked example from the trainer; the loader recomputes it and refuses the
file if Rust and PyTorch disagree, and it refuses a model trained on a
different `FEATURE_VERSION`.

## Labels

- `tichu selfplay --bot NAME` records NAME's own choices (imitation). Any
  bot from `tichu bots` works, including the other learned ones.
- `tichu selfplay --q N` runs an N-sample PIMC search (heuristic rollouts) at
  every decision, plays its best option, and writes the mean margin of every
  candidate as Q-values. `train.py --policy q` then distils
  softmax(Q / `--tau`), which carries far more signal per position than the
  single best move. It plays at about pimcN's strength and speed (`--q 10`
  ran 4.5 rounds/s on 4 cores; expect 2–3× that on 16 threads).
- `--rollout NAME` sets the rollout policy for `--q` (default heuristic).
  With a learned net (`--rollout net:models/gen0.tnet`) the search sees
  better continuations, so its labels improve with the net: that's the
  expert-iteration loop below. Net rollouts are ~2× slower (`--q 10` ran
  2.4 rounds/s on 4 cores).
- `--explore P` plays a random option with probability P (the label is still
  the bot's / search's choice), for more varied positions.

Record size is about 0.8 KB, roughly 30 KB per heuristic round and 40 KB per
PIMC round, plus about 4.5 KB of history tokens per seat per round.

Speed on the desktop (16 threads, `--q 10`, h2 playing): heuristic rollouts
about 5.6 rounds/s; rollouts by a feature-v2 net 2.6 rounds/s without the
GRU and 1.8 with it (a v1 net ran 3.6).

## Workflow

```sh
cargo build --release
T=target/release/tichu   # target\release\tichu.exe on Windows

# 1. Data
$T selfplay --bot heuristic --rounds 100000 --out data/h100k.bin
$T selfplay --q 10 --rounds 20000 --seed 2 --out data/q10_20k.bin

# 2. Train (uses the GPU when torch sees one)
python train/train.py data/q10_20k.bin --policy q --out models/q10.tnet --epochs 10

# 3. Judge
$T arena net:models/q10.tnet heuristic --rounds 4000
$T arena vpimc10:models/q10.tnet pimc10 --rounds 400
```

`train.py` prints validation policy loss, label accuracy, and the value
head's MSE with R² against the margin variance. Report arena results with
the ± and the number of deals, as everywhere else in this repo.

Next steps this pipeline is built for:

1. **Value net in search** (`vpimcN`): rollouts stop at my next turn and the
   value head scores the position, so each rollout is a few plies instead of
   a whole round.
2. **Policy imitating PIMC** (`--q`): a fast bot near PIMC strength, which is
   also a better rollout policy for search.
3. **RL self-play**: generate with `--bot net:models/current.tnet --explore
   0.05`, train with `--init models/current.pt`, keep a model only if it
   beats a fixed pool (heuristic, pimc10, earlier nets) in the arena.

## Results so far (CPU-sized runs, Oct 2026)

| match | margin/deal | deals |
|---|---|---|
| net, imitating heuristic (20k rounds, 97% label accuracy) vs heuristic | +0.6 ± 3.0 | 2000 |
| gen0: distils `--q 10` with heuristic rollouts (2k rounds) vs heuristic | +85.6 ± 4.9 | 2000 |
| gen0 vs pimc10 | +46.8 ± 10.9 | 400 |
| gen1: `--q 10` with gen0 rollouts (1.5k rounds) vs gen0 | +11.6 ± 4.0 | 2000 |
| gen1 vs heuristic | +90.1 ± 5.0 | 2000 |
| gen1 vs pimc10 | +68.4 ± 10.7 | 400 |
| gen0 labelled with h2 rollouts (2k rounds) vs gen0 with heuristic rollouts | −14.2 ± 4.2 | 2000 |
| vpimc10 (value head from the 20k heuristic rounds, R² 0.25) vs pimc10 | −37.1 ± 13.0 | 300 |

Desktop, gen0 on 16k `--q 10` rounds (heuristic rollouts, seed 1000), 8 epochs, arena seed 77:

| net | vs h2 | vs heuristic | deals |
|---|---|---|---|
| features v1 (overnight gen0) | −4.3 ± 3.4 | +96.4 ± 3.8 | 4000 |
| features v1, overnight gen2 (best) | +18.4 ± 3.3 | +108.4 ± 3.8 | 4000 |
| features v2, no GRU | +16.5 ± 3.3 | +113.2 ± 3.8 | 4000 |
| features v2 + GRU | +14.4 ± 3.4 | +116.9 ± 3.8 | 4000 |

v2 + GRU vs v2 flat: +0.3 ± 2.8 over 4000 deals. One v2 generation matches
three v1 generations; the GRU adds nothing yet. The value head still
overfits: R² peaks near 0.33 after ~1000 steps and ends near 0.05.

Desktop run `runs/v2` (Nov 2026): 8 generations of 20k `--q 10` rounds with
`--exact-endgame --infer 4`, the separate value net, gen0 rollouts by the
heuristic and then by the previous net. 4000 deals per match:

| gen | vs h2 | vs gen0 | vs previous gen | value R² |
|---|---|---|---|---|
| 0 | +16.6 ± 3.4 | | | 0.34 |
| 1 | +42.9 ± 3.3 | +30.4 ± 3.0 | (= vs gen0) | 0.35 |
| 2 | +51.2 ± 3.4 | +36.0 ± 3.1 | +7.5 ± 3.0 | 0.35 |
| 3 | +47.9 ± 3.4 | +36.3 ± 3.2 | +2.8 ± 3.0 | 0.33 |
| 4 | +47.8 ± 3.4 | +38.3 ± 3.3 | +1.5 ± 3.1 | 0.34 |
| 5 | +45.7 ± 3.3 | +34.6 ± 3.2 | +1.4 ± 3.0 | 0.38 |
| 6 | +47.9 ± 3.4 | +36.1 ± 3.1 | +1.7 ± 3.0 | 0.34 |
| 7 | +46.1 ± 3.3 | +37.3 ± 3.1 | +1.4 ± 3.0 | 0.34 |

gen7 vs gen2 directly: +4.6 ± 3.0. Most of the gain comes in the first two
generations; after that, each generation adds a point or two, too little to
see one step at a time. The exact-endgame/infer labels didn't change gen0
(+16.6 vs +16.5 without them). The likely ceiling is the teacher (10 guessed
deals per decision).

### The teacher (Nov 2026)

Playing the labeler itself (`labelN[/FLAGS]:net:gen7`) against gen7, 500
deals each, seed 900:

| teacher | margin per deal | time |
|---|---|---|
| 10 samples (v2's) | −77.1 ± 10.0 | |
| 24 samples | −40.0 ± 9.2 | 9 min |
| 48 samples | −9.2 ± 9.7 | 18 min |
| 96 samples | +4.6 ± 9.3 | 34 min |
| 24, anchored on top 3 (`/ep3`, `--anchor 3`) | **+19.5 ± 6.5** | 6.6 min |
| 24, anchored on top 6 | +18.8 ± 7.1 | 7.4 min |
| 48, anchored on top 3 | +21.8 ± 6.7 | 9.3 min |

The plain teacher was worse than its student: with every option searched,
the noisiest one often wins by luck. Anchoring searches only the net's top K
options and keeps the net's choice unless another is ahead by over two
paired standard errors, which makes the teacher better than the student for
less time. Adaptive sampling (`/a48`, `--q-max`) beat fixed 24 by +28.7 ± 9.7
but cost as much as fixed 48. Sampled rollouts (`net@T:`) and weighted hand
guesses (`--infer`) didn't help.

### v3: anchored teacher (`--q 24 --exact-endgame --anchor 3`, from gen7)

4000 deals per match:

| gen | vs heuristic | vs h2 | vs previous | vs v2 gen7 | value R² |
|---|---|---|---|---|---|
| 0 | +133.6 ± 3.8 | +47.2 ± 3.4 | | −0.3 ± 3.1 | 0.41 |
| 1 | +139.9 ± 3.7 | +50.1 ± 3.4 | +6.6 ± 2.8 | +0.7 ± 2.9 | 0.36 |
| 2 | +138.8 ± 3.7 | +52.7 ± 3.4 | +0.6 ± 3.0 | +3.7 ± 3.1 | 0.39 |
| 3 | +140.9 ± 3.7 | +49.5 ± 3.4 | −3.3 ± 2.9 | +1.4 ± 3.1 | 0.39 |

About 2 h per generation, nearly all self-play. The stronger teacher didn't
make a clearly stronger student. Retraining gen2 on the same data, vs gen2
(4000 deals): hard labels −10.4 ± 2.9, `--label-mix 0.5` −6.5 ± 2.8, τ 10
−2.3 ± 2.9, hidden 512 −6.1 ± 2.9, hidden 512 depth 3 −1.7 ± 2.9, 16
epochs +1.2 ± 2.9, three generations of data +3.8 ± 2.9, and another
training seed −3.1 ± 3.0 (so training noise is about ±3 too). Label
accuracy stays near 76%: the student is limited by noisy labels, not size.

Two changes to training did help, and add up (vs gen2, 4000 deals, two
training seeds each): `--prior` (target = softmax(prior log-policy + Q
advantage / τ), the previous generation as prior) +5.1 and +4.2 on two
generations of data; three generations with the old target +3.8 and +2.0;
both together **+7.7 ± 2.9 and +6.8 ± 2.9**. `iterate.py --prior0 PT --window
3 --data0 OLD.bin,...` runs the loop that way.

Why the student lags the teacher: at 24 samples 29% of the variance of a
label's Q difference is noise (60% at 6 samples, 9% at 96), and the teacher
overrides the net in only 5% of decisions (`selfplay` prints these as
"teacher: ..."). 36,000 rounds of 4-sample labels were no better than 12,000
rounds of 24-sample ones (−8.9 ± 3.1 vs −10.7 ± 2.9 on one generation of
data), `--policy adv` no better than `q` (+1.3 ± 3.0), and Tichu/Grand
thresholds are at a local optimum (no calls −7.8 ± 2.5, stricter −2.2 ± 1.6,
looser −3.0 ± 1.9). Of the search's gain, exact endgames are +3.2 ± 0.3 and
the rest is mid-round. Net-only self-play runs 62.5 rounds/s on the desktop,
34× the 24-sample teacher.

Search at play time is the reliable gain: `label24/ep3:net:gen2` beats gen2
by +19.6 ± 4.5 and h2 by +62.2 ± 6.7 (1000 deals), at about 93 ms per
decision on one core (12 samples: +11.2 ± 4.2, 51 ms).

The distilled policy beats the search that labelled it, and a second
generation beats the first. The value head isn't good enough to stop
rollouts after one orbit yet: a single position's value is noisy (hidden
cards, the rest of the round), and R² stayed near 0.25.

## Deep Monte Carlo (DouZero-style), no search

`train/dmc.py` is a second loop that uses no search labels at all. It plays a
great many ε-greedy self-play rounds and regresses each played option's value
onto the round's final margin, as DouZero did for Dou Dizhu. The noise of a
single round's outcome is averaged over millions of positions, not over
samples within one.

```sh
python train/dmc.py --dir runs/dmc --init runs/v4/start.pt --chunks 200
```

- **Actor**: `tichu selfplay --bot net:<latest> --explore 0.03 --played-only`
  plays `--chunk-rounds` rounds (default 20k) on every thread. `--played-only`
  writes just the option played, one record per decision with 2+ options
  (state, option, margin; about 0.76 KB, no history tokens).
- **Learner** (the GPU): a replay buffer of the newest `--buffer` records (4M,
  about 3 GB), and MSE steps of Q(state, played option) against margin / 100,
  about `--replay` (4) samples per new record. It trains while the next chunk
  plays, so the actor is one chunk behind. Q is dueling by default: the trunk's
  value head plus the policy head, so the head only learns how options
  differ. The exported `.tnet` is an ordinary `net:` model whose logits are
  those advantages, so `net:` plays argmax Q.
- **Warm start**: `--init` takes any `train.py` checkpoint. Its logits are read
  as advantages of logit × `--init-tau` (20) points. Without `--init` the net
  starts fresh and chunk 0 is played by h2 (`--actor0`).
- Each chunk logs the current model's fit to the new data before training on
  it ("fresh-data R²" of Q and of the state value). A Q that knows the
  options should beat the state value. Every `--eval-every` chunks the export
  plays `--opponents` in the arena and is kept as `snapNNNNN.tnet`/`.pt`. Logs
  go to `log.txt` and `log.jsonl`.
- **Who goes out first**: every record's flags say whether its seat (2) or
  its partner (4) went out first (`tichu_data` reads them as `out_first`).
  The learner trains a small head on the trunk to predict it (opponent / me /
  partner, `--first-weight`), for learning Tichu calls once the card play is
  good: call when P(me first) is high enough. It is saved in `learner.pt` and
  `*.first.pt`, not in the `.tnet`, and logged per chunk as fresh-data log
  loss against the base rate (in a 6k-round smoke run, 0.74 against 1.04 after
  two chunks).
- **Pause and resume**: stop it at any time (Ctrl+C, or kill python and tichu).
  Re-running the same command resumes from the last finished chunk
  (`learner.pt`) and refills the buffer from the chunk files it keeps on disk.

Speed on the desktop: 53–58 rounds/s of self-play, about 750k records per
20k-round chunk (6–6.5 min), and about 25 s of training per chunk. The
learner is idle most of the time.

First test (Nov 2026), warm start from `runs/v4/start` (lr 2e-4, 1000 deals
each):

| after | vs start net | vs h2 |
|---|---|---|
| 2 chunks (40k rounds) | −14.8 ± 6.3 | +42.7 ± 7.0 |
| 4 chunks (80k rounds) | −20.2 ± 6.3 | +49.4 ± 7.2 |

With lr 2e-4 the fresh-data R² of Q was 0.38 against 0.41 for the state
value: the advantages added more noise than they explained. With `--lr 5e-5`
Q pulls ahead (0.44 against 0.43 after 4 chunks) and the spread of Q − V
falls from 65 to 48 points:

| lr 5e-5, after | vs start net | vs h2 |
|---|---|---|
| 2 chunks | −21.6 ± 6.4 | +48.0 ± 6.9 |
| 4 chunks | −15.1 ± 6.6 | +48.5 ± 6.8 |

Both are still below the start net after 80k rounds. DouZero needed far more
games than this.

## Running the loop

```sh
python train/iterate.py --dir runs/a --gens 6 --rounds 20000
```

Generation 0's rollouts use the heuristic (`--rollout0`; h2 rollouts gave
worse labels, above), and every non-play decision (calls, passing, bombs out
of turn, wish, Dragon) is made by h2 (`--base`), here and in the `net:` bots.
Each generation runs `selfplay --q 10 --rollout net:<previous>`, trains on
the last two generations' records, and plays the new net against the
heuristic, h2 and the previous generation (4000 deals each); everything is
logged to `runs/a/log.txt`. Stop it any time; re-running resumes. At ~6
rounds/s on 16 threads (estimated), 20k rounds is about an hour of self-play
per generation; training is minutes on the GPU.

## Setting up the desktop (RTX 2070 Super, Ryzen 7 3700X)

Linux, or Windows with either native tools or WSL2 (Ubuntu). Commands are the
same apart from paths.

1. **NVIDIA driver**: any recent driver (`nvidia-smi` should list the
   2070 Super). The PyTorch wheels bring their own CUDA runtime, so no CUDA
   toolkit install is needed.
2. **Rust**: <https://rustup.rs> (`curl https://sh.rustup.rs -sSf | sh` on
   Linux/WSL, `rustup-init.exe` on Windows; the MSVC build tools are needed
   on native Windows). Then `cargo test --release` in the repo.
3. **Python 3.10+** and a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install numpy
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   That should print `True NVIDIA GeForce RTX 2070 SUPER`. The 2070 Super
   (Turing, compute capability 7.5) is supported by the cu126 and cu128
   wheels; if the driver is too old for them, update the driver.
4. Self-play uses all 16 threads by default (`--threads`). Training keeps the
   whole dataset on the GPU as bytes: 8 GB holds roughly 5M records, so for
   more than that pass `--device cpu` or train on a subset.
