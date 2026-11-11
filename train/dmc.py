"""Deep Monte Carlo training, after DouZero: no search, lots of self-play, and
each played option's value regressed onto the round's final margin.

    python train/dmc.py --dir runs/dmc --init runs/v4/start.pt --chunks 200

The net is the usual `TichuNet`, read as a Q-network: the policy head's
output for (state, option) is the expected round margin for my team after
playing that option, in units of --value-scale points. `net:` bots play its
argmax, so every exported .tnet is a playable bot as it is.

The loop has an actor and a learner running at the same time:

- actor: `tichu selfplay --bot net:<latest export> --explore E --played-only`
  plays --chunk-rounds rounds on all CPU threads (ε-greedy over Q) and writes
  one record per decision with 2+ options: the state, the option played and
  the round's margin.
- learner (this process, on the GPU): keeps the newest --buffer records in a
  replay buffer on the device and takes MSE steps of Q(state, played option)
  against the margin, about --replay samples per new record. When a chunk
  lands, its fit is measured before training on it (a fresh-data R²), it
  joins the buffer, the model is exported and the next chunk starts with it.
  The actor is therefore one chunk behind the learner.

Calls, passing, out-of-turn bombs, wishes and the Dragon stay with h2, as in
every `net:` bot. The margin includes Tichu bets, which no play decision
here makes; they are the same for every option and only add noise.

Chunk 0 is played by --actor0 (default: the --init model's .tnet if it has
one, else h2): Q then starts as that bot's values, and the greedy policy
improves on it from there.

Every --eval-every chunks the latest export plays --opponents in the arena
(--eval-rounds deals, a fixed seed) and is kept as snap-<chunk>.tnet; the
results are in log.txt and log.jsonl. Stop it any time: re-running with the
same --dir resumes from the last finished chunk (learner.pt) and refills
the buffer from the chunk files still on disk (--keep-chunks).
"""

import argparse
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tichu_data import BLOCK, FORMAT_VERSION, HEADER, REC  # noqa: E402
from train import TichuNet, write_tnet  # noqa: E402


def read_played(path):
    """A `selfplay --played-only` file -> (states [N, S] u8, actions [N, A]
    u8, margin [N] f32, dims). Vectorised per block: every record has one
    option, no Q and no history tokens."""
    buf = open(path, "rb").read()
    magic, version, fv, S, A, D = HEADER.unpack_from(buf, 0)
    if magic != b"TCHR" or version != FORMAT_VERSION:
        raise ValueError(f"{path}: not a v{FORMAT_VERSION} tichu record file")
    data = np.frombuffer(buf, dtype=np.uint8)
    R = REC.size + S + A
    parts, pos = [], HEADER.size
    while pos < len(buf):
        _rnd, _seat, nt, nr = BLOCK.unpack_from(buf, pos)
        pos += BLOCK.size + nt * D
        parts.append(data[pos:pos + nr * R].reshape(nr, R))
        pos += nr * R
    if pos != len(buf):
        raise ValueError(f"{path}: truncated")
    rows = np.concatenate(parts) if parts else np.zeros((0, R), np.uint8)
    n_opts = rows[:, 1].astype(np.int64) | (rows[:, 2].astype(np.int64) << 8)
    if not ((rows[:, 0] == 0).all() and (n_opts == 1).all()):
        raise ValueError(f"{path}: not a --played-only file (records with several options or Q)")
    margin = np.ascontiguousarray(rows[:, 7:11]).view("<f4").ravel().astype(np.float32)
    dims = dict(feature_version=fv, state_dim=S, action_dim=A, token_dim=D)
    return rows[:, REC.size:REC.size + S], rows[:, REC.size + S:], margin, dims


def arch_of(sd):
    """TichuNet constructor arguments that fit a state dict."""
    if any(k.startswith("gru.") for k in sd):
        raise SystemExit("--init: models with a GRU need history tokens, which --played-only records don't have")
    depth = len([k for k in sd if re.fullmatch(r"trunk\.\d+\.weight", k)])
    vn = sd.get("value_net.0.weight")
    return dict(hidden=sd["trunk.0.weight"].shape[0], depth=depth, head=sd["policy.0.weight"].shape[0],
                value_net=vn.shape[0] if vn is not None else 0)


class Dims(types.SimpleNamespace):
    """The feature layout and record count `write_tnet` asks of its data."""

    def __init__(self, dims, n):
        super().__init__(**dims)
        self.n = n

    def __len__(self):
        return self.n


class Buffer:
    """The newest `cap` records on the device, as a ring."""

    def __init__(self, cap, S, A, dev):
        self.cap, self.size, self.head = cap, 0, 0
        self.s = torch.empty((cap, S), dtype=torch.uint8, device=dev)
        self.a = torch.empty((cap, A), dtype=torch.uint8, device=dev)
        self.y = torch.empty(cap, dtype=torch.float32, device=dev)

    def add(self, s, a, y):
        n = len(s)
        if n > self.cap:
            s, a, y, n = s[-self.cap:], a[-self.cap:], y[-self.cap:], self.cap
        dev = self.s.device
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        s, a, y = T(s), T(a), T(y)
        first = min(n, self.cap - self.head)
        for dst, src in ((self.s, s), (self.a, a), (self.y, y)):
            dst[self.head:self.head + first] = src[:first]
            dst[:n - first] = src[first:]
        self.head = (self.head + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def sample(self, n, gen):
        i = torch.randint(self.size, (n,), device=self.s.device, generator=gen)
        return self.s[i], self.a[i], self.y[i]


def q_and_values(model, s, a, dueling):
    """For each (state, option) row: Q, the policy head's output (the
    exported logit), the exported value and the trunk's value head, all in
    --value-scale units. Dueling: Q = trunk value + policy head, so the head
    only has to learn how options differ."""
    s = s.float() / 255.0
    a = a.float() / 255.0
    v, adv, tv = model(s, a[:, None, :], s.new_zeros(len(s), 0, 0), torch.zeros(len(s), dtype=torch.long, device=s.device))
    adv = adv[:, 0]
    return (tv + adv if dueling else adv), adv, v, tv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--chunks", type=int, default=100, help="stop after this many chunks in total")
    ap.add_argument("--chunk-rounds", type=int, default=20000, help="self-play rounds per chunk")
    ap.add_argument("--init", help="start from this .pt (train.py's checkpoints; architecture read from it)")
    ap.add_argument("--actor0", help="bot that plays chunk 0 (default: net:<init>.tnet if it exists, else h2)")
    ap.add_argument("--explore", type=float, default=0.03, help="actor's chance of a uniformly random option")
    ap.add_argument("--buffer", type=int, default=4_000_000, help="replay buffer size in records (GPU memory: ~0.76 KB each)")
    ap.add_argument("--replay", type=float, default=4.0, help="training samples drawn per new record")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=0.25, help="weight of the state-value losses (exported value and trunk head)")
    ap.add_argument("--value-scale", type=float, default=100.0)
    ap.add_argument("--dueling", type=int, default=1,
                    help="1: Q = trunk value head + policy head (the head learns only how options differ); 0: Q = policy head")
    ap.add_argument("--init-tau", type=float, default=20.0,
                    help="dueling with --init: read the init policy's logits as advantages of logit × this many points")
    ap.add_argument("--hidden", type=int, default=256, help="without --init")
    ap.add_argument("--depth", type=int, default=2, help="without --init")
    ap.add_argument("--head", type=int, default=128, help="without --init")
    ap.add_argument("--value-net", type=int, default=0, help="without --init: units of a separate state-value net (0 = export the trunk's head)")
    ap.add_argument("--eval-every", type=int, default=5, help="chunks between arena checks (0 = never)")
    ap.add_argument("--eval-rounds", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=4321)
    ap.add_argument("--opponents", default="h2", help="arena opponents, comma-separated bot names")
    ap.add_argument("--keep-chunks", type=int, default=-1,
                    help="chunk files kept on disk for resuming (default: enough to refill the buffer)")
    ap.add_argument("--threads", type=int, default=0, help="self-play threads (0 = all)")
    ap.add_argument("--tichu", default=os.path.join(ROOT, "target", "release", "tichu.exe" if os.name == "nt" else "tichu"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # The log has "R²", "±" and "−": never die on a cp1252 console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = os.path.abspath(args.dir)
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "log.txt"), "a", encoding="utf-8")
    jlog = open(os.path.join(d, "log.jsonl"), "a", encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def jot(**kw):
        jlog.write(json.dumps(dict(time=time.time(), **kw)) + "\n")
        jlog.flush()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    chunk_path = lambda k: os.path.join(d, f"chunk{k:05d}.bin")  # noqa: E731
    actor_path = lambda k: os.path.join(d, f"actor{k:05d}.tnet")  # noqa: E731
    state_path = os.path.join(d, "learner.pt")

    # Model: from the learner's own checkpoint, else --init, else fresh.
    resume = torch.load(state_path, map_location=dev) if os.path.exists(state_path) else None
    if resume is not None:
        arch = resume["arch"]
    elif args.init:
        init_sd = torch.load(args.init, map_location=dev)
        arch = dict(arch_of(init_sd), dueling=args.dueling)
        if args.dueling and args.init_tau > 0:
            # A softmax policy's logits, as an advantage: logit · tau points.
            for k in ("policy.1.weight", "policy.1.bias"):
                init_sd[k] = init_sd[k] * (args.init_tau / args.value_scale)
    else:
        arch = dict(hidden=args.hidden, depth=args.depth, head=args.head, value_net=args.value_net, dueling=args.dueling)
    dims = resume["dims"] if resume is not None else None
    model = opt = buf = None

    def build(dims_):
        nonlocal model, opt, buf
        model = TichuNet(dims_["state_dim"], dims_["action_dim"], arch["hidden"], arch["depth"], arch["head"],
                         dims_["token_dim"], 0, arch["value_net"]).to(dev)
        if resume is not None:
            model.load_state_dict(resume["model"])
        elif args.init:
            model.load_state_dict(init_sd)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if resume is not None:
            opt.load_state_dict(resume["opt"])
        buf = Buffer(args.buffer, dims_["state_dim"], dims_["action_dim"], dev)

    st = resume["state"] if resume is not None else dict(chunk=0, steps=0, budget=0.0, records=0, example=None)
    if dims is not None:
        build(dims)
    say(f"dmc: {'resuming at chunk ' + str(st['chunk']) if resume is not None else 'new run'}; arch {arch}; "
        f"buffer {args.buffer}, replay {args.replay}, batch {args.batch}, lr {args.lr}, explore {args.explore} on {dev}")

    keep = args.keep_chunks  # < 0: set from the first chunk's size, enough to refill the buffer

    def export(path):
        ex = st["example"]
        model.eval()
        with torch.no_grad():
            s = torch.tensor(ex["state"], dtype=torch.uint8, device=dev)[None]
            a = torch.tensor(ex["action"], dtype=torch.uint8, device=dev)[None]
            _, adv, v, _ = q_and_values(model, s, a, arch["dueling"])
        model.train()
        check = dict(state=ex["state"], actions=ex["action"], tokens=[],
                     value=float(v[0]) * args.value_scale, logits=[float(adv[0])])
        meta = types.SimpleNamespace(value_scale=args.value_scale, data=[f"dmc chunks 0-{st['chunk'] - 1}"],
                                     policy="dmc", epochs=0)
        data = Dims(dims, st["records"])
        cpu = TichuNet(dims["state_dim"], dims["action_dim"], arch["hidden"], arch["depth"], arch["head"],
                       dims["token_dim"], 0, arch["value_net"])
        cpu.load_state_dict({k: t.detach().cpu() for k, t in model.state_dict().items()})
        write_tnet(cpu, path + ".part", meta, data, check)
        os.replace(path + ".part", path)

    def save():
        tmp = state_path + ".part"
        torch.save(dict(arch=arch, dims=dims, model=model.state_dict(), opt=opt.state_dict(), state=st), tmp)
        os.replace(tmp, state_path)
        torch.save(model.state_dict(), os.path.join(d, "latest.pt"))

    train_time = [0.0]
    last_loss = [float("nan"), float("nan")]

    def train_steps(n):
        t0 = time.time()
        model.train()
        for _ in range(n):
            s, a, y = buf.sample(args.batch, gen)
            q, _, v, tv = q_and_values(model, s, a, arch["dueling"])
            ql = F.mse_loss(q, y)
            vl = F.mse_loss(tv, y) + (F.mse_loss(v, y) if model.value_net is not None else 0)
            loss = ql + args.value_weight * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            st["steps"] += 1
            if st["steps"] % 50 == 0:
                last_loss[:] = [ql.item(), vl.item()]
        train_time[0] += time.time() - t0

    def train_while(proc):
        """Train toward the budget while `proc` runs; then wait for it."""
        while proc.poll() is None:
            if buf is not None and buf.size >= args.batch and st["steps"] < st["budget"]:
                train_steps(min(50, int(st["budget"] - st["steps"]) + 1))
            else:
                time.sleep(0.2)
        return proc.returncode

    def selfplay(k, actor):
        cmd = [args.tichu, "selfplay", "--bot", actor, "--played-only", "--explore", str(args.explore),
               "--rounds", str(args.chunk_rounds), "--seed", str(1_000_000 * (args.seed + 1) + k),
               "--out", chunk_path(k) + ".part"]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        out = open(os.path.join(d, "selfplay.txt"), "a", encoding="utf-8")
        out.write(f"\n== chunk {k}: {' '.join(cmd)}\n")
        out.flush()
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL, cwd=ROOT, env=env), time.time()

    def arena(k, actor_file):
        results = {}
        for opp in [o for o in args.opponents.split(",") if o]:
            cmd = [args.tichu, "arena", "net:" + actor_file, opp, "--rounds", str(args.eval_rounds), "--seed", str(args.eval_seed)]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            out_path = os.path.join(d, "arena.tmp")
            with open(out_path, "w", encoding="utf-8") as out:
                p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
                rc = train_while(p)
            text = open(out_path, encoding="utf-8", errors="replace").read()
            m = re.search(r"margin per deal .*?: ([+-][\d.]+) ± ([\d.]+)", text)
            if rc != 0 or not m:
                say(f"arena vs {opp} failed ({rc}): {text[-500:]}")
                continue
            results[opp] = (float(m.group(1)), float(m.group(2)))
            say(f"chunk {k}: vs {opp} {m.group(1)} ± {m.group(2)} over {args.eval_rounds} deals")
        jot(event="arena", chunk=k, steps=st["steps"], results=results, deals=args.eval_rounds)

    def refill():
        """After a resume: the newest chunk files back into the buffer."""
        files = sorted(glob.glob(os.path.join(d, "chunk*.bin")))
        files = [f for f in files if int(os.path.basename(f)[5:10]) < st["chunk"]]
        loaded, chosen = 0, []
        for f in reversed(files):
            if loaded >= args.buffer:
                break
            chosen.append(f)
            loaded += (os.path.getsize(f) - HEADER.size) // (REC.size + dims["state_dim"] + dims["action_dim"])
        for f in reversed(chosen):
            s, a, y, _ = read_played(f)
            buf.add(s, a, y / args.value_scale)
        say(f"buffer refilled from {len(chosen)} chunk files: {buf.size} records")

    for f in glob.glob(os.path.join(d, "*.part")):
        os.remove(f)
    if resume is not None:
        refill()

    k = st["chunk"]
    if k >= args.chunks:
        say(f"already at chunk {k} of {args.chunks}; nothing to do")
        return
    if k == 0:
        actor = args.actor0
        if actor is None:
            init_tnet = os.path.splitext(args.init)[0] + ".tnet" if args.init else None
            actor = "net:" + init_tnet if init_tnet and os.path.exists(init_tnet) else "h2"
    else:
        actor = "net:" + actor_path(k)
    proc, t_start = selfplay(k, actor)
    run_t0 = time.time()
    while True:
        rc = train_while(proc)
        sp_secs = time.time() - t_start
        if rc != 0:
            raise SystemExit(f"selfplay for chunk {k} failed ({rc}); see {os.path.join(d, 'selfplay.txt')}")
        os.replace(chunk_path(k) + ".part", chunk_path(k))
        s, a, y, dims_k = read_played(chunk_path(k))
        if dims is None:
            dims = dims_k
            build(dims)
        elif dims_k != dims:
            raise SystemExit(f"chunk {k} has feature layout {dims_k}, expected {dims}")
        y = y / args.value_scale
        if keep < 0:
            keep = max(1, -(-args.buffer // max(1, len(s))))
        # How well the current model predicts data it hasn't seen.
        model.eval()
        with torch.no_grad():
            sub = np.linspace(0, len(s) - 1, min(len(s), 200_000)).astype(np.int64)
            ss, aa = torch.from_numpy(s[sub]).to(dev), torch.from_numpy(a[sub]).to(dev)
            yy = torch.from_numpy(y[sub]).to(dev)
            qs, vs = [], []
            for i in range(0, len(sub), 8192):
                q_, _, v_, _ = q_and_values(model, ss[i:i + 8192], aa[i:i + 8192], arch["dueling"])
                qs.append(q_)
                vs.append(v_)
            q_, v_ = torch.cat(qs), torch.cat(vs)
            var = yy.var().item()
            q_r2 = 1 - F.mse_loss(q_, yy).item() / var
            v_r2 = 1 - F.mse_loss(v_, yy).item() / var
            adv_sd = (q_ - v_).std().item() * args.value_scale
        model.train()
        i0 = len(s) // 2
        st["example"] = dict(state=s[i0].tolist(), action=a[i0].tolist())
        buf.add(s, a, y)
        st["records"] += len(s)
        st["budget"] += args.replay * len(s) / args.batch
        st["chunk"] = k + 1
        rounds_s = args.chunk_rounds / sp_secs
        say(f"chunk {k}: {len(s)} records ({rounds_s:.1f} rounds/s, actor {os.path.basename(actor)}); "
            f"fresh-data R² Q {q_r2:.3f}, value {v_r2:.3f}, sd(Q − V) {adv_sd:.1f} pts; "
            f"margin sd {np.sqrt(var) * args.value_scale:.0f} pts; steps {st['steps']} "
            f"(loss Q {last_loss[0]:.3f}, V {last_loss[1]:.3f}; {train_time[0]:.0f}s training); buffer {buf.size}")
        jot(event="chunk", chunk=k, records=len(s), rounds_per_s=rounds_s, q_r2=q_r2, v_r2=v_r2, adv_sd=adv_sd,
            steps=st["steps"], loss_q=last_loss[0], loss_v=last_loss[1], train_secs=train_time[0],
            buffer=buf.size, elapsed=time.time() - run_t0)
        train_time[0] = 0.0
        # Old chunk files: only what a resume needs to refill the buffer.
        for old in range(0, k + 1 - keep):
            if os.path.exists(chunk_path(old)):
                os.remove(chunk_path(old))
        k += 1
        export(actor_path(k))
        for old in glob.glob(os.path.join(d, "actor*.tnet")):
            if old != actor_path(k):
                os.remove(old)
        shutil.copyfile(actor_path(k), os.path.join(d, "latest.tnet"))
        save()
        if args.eval_every and k % args.eval_every == 0:
            snap = os.path.join(d, f"snap{k:05d}.tnet")
            shutil.copyfile(actor_path(k), snap)
            shutil.copyfile(os.path.join(d, "latest.pt"), os.path.splitext(snap)[0] + ".pt")
            arena(k, snap)
        if k >= args.chunks:
            break
        actor = "net:" + actor_path(k)
        proc, t_start = selfplay(k, actor)
    # Finish the last chunk's training budget before the final export.
    if st["steps"] < st["budget"]:
        train_steps(int(st["budget"] - st["steps"]))
        export(os.path.join(d, "latest.tnet"))
        save()
    say(f"done: {k} chunks, {st['records']} records, {st['steps']} steps")


if __name__ == "__main__":
    main()
