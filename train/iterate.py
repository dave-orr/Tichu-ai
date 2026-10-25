"""Expert iteration: each generation labels self-play with a PIMC search
whose rollouts are played by the previous generation's net, distils those
Q-values into a new net, and judges it in the arena.

    python train/iterate.py --dir runs/a --gens 5 --rounds 20000

Generation 0's rollouts use --rollout0 (default heuristic: in a 2k-round
test, h2 rollouts gave worse labels, −14.2 ± 4.2 over 2000 deals). The
--base rule bot (default h2) makes every non-play decision (calls, passing,
bombs out of turn). Training data is the latest
generation's records plus the previous `--window - 1` generations'.
Results go to <dir>/log.txt. Re-running continues after the last finished
generation.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, log):
    print("+ " + " ".join(cmd), flush=True)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    log.write(out.stdout)
    log.flush()
    return out.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20000, help="self-play rounds per generation")
    ap.add_argument("--base", default="h2", help="rule bot for non-play decisions")
    ap.add_argument("--rollout0", default="heuristic", help="rollout policy for generation 0")
    ap.add_argument("--q", type=int, default=10, help="PIMC samples per decision")
    ap.add_argument("--window", type=int, default=2, help="generations of data to train on")
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--arena-rounds", type=int, default=4000)
    ap.add_argument("--tichu", default=os.path.join(ROOT, "target", "release", "tichu.exe" if os.name == "nt" else "tichu"))
    ap.add_argument("--train-args", default="", help="extra arguments for train.py")
    args = ap.parse_args()
    d = os.path.abspath(args.dir)
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "log.txt"), "a", encoding="utf-8")
    model = lambda g: os.path.join(d, f"gen{g}.tnet")  # noqa: E731
    data = lambda g: os.path.join(d, f"gen{g}.bin")  # noqa: E731

    for g in range(args.gens):
        if os.path.exists(model(g)):
            continue
        rollout = args.rollout0 if g == 0 else "net:" + model(g - 1)
        log.write(f"\n== generation {g} (rollouts: {rollout})\n")
        if not os.path.exists(data(g)):
            run([args.tichu, "selfplay", "--bot", args.base, "--q", str(args.q), "--rollout", rollout, "--rounds", str(args.rounds),
                 "--seed", str(1000 + g), "--out", data(g) + ".part"], log)
            os.replace(data(g) + ".part", data(g))
        files = [data(k) for k in range(max(0, g - args.window + 1), g + 1)]
        run([sys.executable, os.path.join(HERE, "train.py"), *files, "--policy", "q",
             "--epochs", str(args.epochs), "--out", model(g), *args.train_args.split()], log)
        opponents = ["heuristic", args.base] + ([f"net:{model(g - 1)}"] if g > 0 else [])
        for opp in opponents:
            out = run([args.tichu, "arena", f"net:{model(g)}", opp, "--rounds", str(args.arena_rounds),
                       "--seed", str(77 + g)], log)
            m = re.search(r"margin per deal .*?: ([+-][\d.]+ ± [\d.]+)", out)
            print(f"gen{g} vs {opp}: {m.group(1) if m else '?'} over {args.arena_rounds} deals", flush=True)


if __name__ == "__main__":
    main()
