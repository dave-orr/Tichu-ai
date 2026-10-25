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
  from. Output is in units of `--value-scale` (100 points).
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

The distilled policy beats the search that labelled it, and a second
generation beats the first. The value head isn't good enough to stop
rollouts after one orbit yet: a single position's value is noisy (hidden
cards, the rest of the round), and R² stayed near 0.25.

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
