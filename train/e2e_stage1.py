"""Stage 1 of the one-net experiment: does a larger net with a memory of the
round use the same data better than a net of dmc1's shape?

    python train/e2e_stage1.py --dir runs/e2e-s1

1. Speed: each --tichu binary makes the same short data file and the
   fastest is used (to compare allocators; one binary skips this).
2. Data: a held-out file and --parts training files of `tichu e2edata`
   (the current bot without search in every seat, full games, branches).
3. Arms: `train/e2e.py` on the same files in the same order, each file once,
   for every net in --arms (NAME:WIDTH,DEPTH,HEAD,SEQ).
4. A table of the arms' held-out reports in log.txt.

Stop it any time; re-running skips finished files and finished arms.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = "callnet~g:runs/calls3/m3.tcall:callnet~s:runs/calls1/m1.tcall:net:runs/v4/gen3.tnet"
ARMS = "small:256,2,128,0;mid-flat:512,4,256,0;mid:512,4,256,128;big:1024,6,256,256"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bot", default=BOT)
    ap.add_argument("--tichu", nargs="+", default=[os.path.join("target", "release", "tichu.exe" if os.name == "nt" else "tichu")],
                    help="tichu binaries to choose from by speed")
    ap.add_argument("--parts", type=int, default=50)
    ap.add_argument("--games", type=int, default=250, help="games per training file (about 9.6 rounds each)")
    ap.add_argument("--val-games", type=int, default=600)
    ap.add_argument("--arms", default=ARMS)
    ap.add_argument("--speed-games", type=int, default=60)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--extra", default="", help="more arguments for train/e2e.py, in one string")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = args.dir
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "log.txt"), "a", encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def run(cmd, out=None):
        t0 = time.time()
        with open(out, "w", encoding="utf-8") if out else open(os.devnull, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        if rc != 0:
            tail = open(out, encoding="utf-8", errors="replace").read()[-800:] if out else ""
            raise SystemExit(f"failed ({rc}): {' '.join(cmd)}\n{tail}")
        return time.time() - t0

    threads = ["--threads", str(args.threads)] if args.threads else []
    state_path = os.path.join(d, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    for f in os.listdir(d):
        if f.endswith(".tmp"):
            os.remove(os.path.join(d, f))

    # 1. Which binary is fastest?
    if state.get("tichu") not in args.tichu:
        speeds = {}
        for exe in args.tichu:
            tmp = os.path.join(d, "speed.tmp")
            secs = run([exe, "e2edata", "--bot", args.bot, "--games", str(args.speed_games), "--seed", "999", "--out", tmp] + threads)
            os.remove(tmp)
            speeds[exe] = args.speed_games / secs
            say(f"speed: {exe} makes {speeds[exe]:.2f} games/s of data")
        state["tichu"] = max(speeds, key=speeds.get)
        state["speeds"] = speeds
        json.dump(state, open(state_path, "w"))
    exe = state["tichu"]
    say(f"using {exe}")

    # 2. Data.
    def gen(name, games, seed):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            out = os.path.join(d, "e2edata.txt")
            secs = run([exe, "e2edata", "--bot", args.bot, "--games", str(games), "--seed", str(seed), "--out", path + ".tmp"] + threads, out)
            os.replace(path + ".tmp", path)
            m = re.search(r"wrote (\d+) records from \d+ games \((\d+) rounds, (\d+) playouts\)", open(out, encoding="utf-8", errors="replace").read())
            if m:
                say(f"{name}: {m.group(2)} rounds, {m.group(3)} playouts, {m.group(1)} records in {secs / 60:.1f} min "
                    f"({int(m.group(2)) / secs:.1f} rounds/s, {os.path.getsize(path) / 1e6:.0f} MB)")
        return path

    val = gen("val.bin.gz", args.val_games, 1_000_001)
    parts = [gen(f"part{i:03d}.bin.gz", args.games, i) for i in range(1, args.parts + 1)]

    # 3. Arms.
    rows = []
    for arm in args.arms.split(";"):
        name, shape = arm.split(":")
        width, depth, head, seq = (int(x) for x in shape.split(","))
        out = os.path.join(d, f"{name}.pt")
        done = os.path.join(d, f"{name}.done")
        if not os.path.exists(done):
            cmd = [sys.executable, os.path.join(HERE, "e2e.py")] + parts + ["--val", val, "--out", out, "--width", str(width),
                   "--depth", str(depth), "--head", str(head), "--seq", str(seq), "--eval-every", str(max(1, args.parts // 5))]
            if args.device:
                cmd += ["--device", args.device]
            secs = run(cmd + args.extra.split(), os.path.join(d, f"{name}.txt"))
            open(done, "w").write(f"{secs:.0f}\n")
            say(f"arm {name} ({shape}) trained in {secs / 60:.1f} min")
        j = json.load(open(os.path.join(d, f"{name}.json"), encoding="utf-8"))
        rows.append((name, j["parameters"], j["log"][-1]["val"]))

    # 4. The table.
    say("held-out results (same files, each seen once):")
    for name, n_par, rep in rows:
        say(f"  {name}: {n_par / 1e6:.2f}M parameters")
        for kind, r in rep.items():
            line = f"    {kind:7}"
            if "n" in r:
                line += f" R² V {r['v_r2']:.3f} Q {r['q_r2']:.3f}"
            if "teacher_r2" in r:
                line += f" (the bot's net {r['teacher_r2']:.3f})"
            if "cards_right" in r:
                line += f"; cards right {r['cards_right']:.1%} (counting {r['cards_right_counting']:.1%})"
            if "groups" in r:
                line += f"; {r['groups']} branches: best option {r['agree']:.1%}, gain {r['gain']:+.1f} ± {r['gain_ci']:.1f}, R² {r['branch_r2']:.3f}"
            if "same_as_bot" in r:
                line += f", policy head picks the bot's play {r['same_as_bot']:.1%}"
            say(line)
    say("done")


if __name__ == "__main__":
    main()
