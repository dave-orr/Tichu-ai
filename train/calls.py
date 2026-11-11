"""Train a Tichu / Grand Tichu call model from `tichu-calls label` records.

Each record is one call decision with two counterfactual round results (with
and without the call, same deal and bots). Rounds were played at 0-0. Here
each record gets a game score drawn from realistic game states, and the label
is the change in my team's game-win chance (percentage points) that the call
caused, using a win-probability table built from the recorded round results.

    python calls.py --data a.bin [b.bin ...] --out model.tcall
"""

import argparse
import struct
import sys
import time

import numpy as np

STATE_DIM_EXPECTED = None
MODEL_VERSION = 1


def read_records(paths):
    feats, kinds, rounds, heur, res, round_res = [], [], [], [], [], []
    dims = None
    for k, path in enumerate(paths):
        data = open(path, "rb").read()
        assert data[:4] == b"TCAL", path
        fmt, fv, sd, ed = struct.unpack_from("<4I", data, 4)
        if dims is None:
            dims = (fv, sd, ed)
        assert dims == (fv, sd, ed), f"{path}: feature layout differs"
        rec = 1 + 4 + 1 + 1 + 8 + sd + ed
        p = 20
        n = len(data)
        buf = np.frombuffer(data, dtype=np.uint8)
        while p < n:
            kind = data[p]
            rid = struct.unpack_from("<I", data, p + 1)[0] + (k << 32)
            if kind == 2:
                round_res.append(struct.unpack_from("<2h", data, p + 6))
                p += 10
                continue
            kinds.append(kind)
            rounds.append(rid)
            heur.append(data[p + 6])
            res.append(struct.unpack_from("<4h", data, p + 7))
            feats.append(buf[p + 15 : p + rec])
            p += rec
    return dict(
        dims=dims,
        x=np.stack(feats),
        kind=np.array(kinds, np.int8),
        round=np.array(rounds, np.int64),
        heur=np.array(heur, np.int8),
        res=np.array(res, np.int32),  # mine/theirs without, mine/theirs with
        rounds=np.array(round_res, np.int32),
    )


class WinTable:
    """P(my team wins the game | scores before the round), assuming every
    round is an independent draw from the recorded round results."""

    LO, HI, STEP, TARGET = -1500, 995, 5, 1000

    def __init__(self, round_results, sweeps=200):
        r = np.concatenate([round_results, round_results[:, ::-1]])  # both seats' view
        pairs, counts = np.unique(r, axis=0, return_counts=True)
        self.pairs, self.p = pairs, counts / counts.sum()
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        n = (self.HI - self.LO) // self.STEP + 1
        self.n = n
        # Post-round scores on an extended grid, PAD steps either side.
        pad = int(np.abs(pairs).max()) // self.STEP + 1
        m = n + 2 * pad
        s = self.LO + self.STEP * (np.arange(m) - pad)
        us, them = np.meshgrid(s, s, indexing="ij")
        over = torch.tensor(((us >= self.TARGET) | (them >= self.TARGET)) & (us != them), device=dev)
        won = torch.tensor((us > them).astype(np.float64), device=dev)
        ui = torch.tensor(np.clip(np.arange(m) - pad, 0, n - 1), device=dev)
        W = torch.full((n, n), 0.5, dtype=torch.float64, device=dev)
        shifts = [(int(a) // self.STEP + pad, int(b) // self.STEP + pad, float(p)) for (a, b), p in zip(pairs, self.p)]
        for it in range(sweeps):
            V = torch.where(over, won, W[ui][:, ui])
            new = torch.zeros_like(W)
            for i, j, p in shifts:
                new += p * V[i : i + n, j : j + n]
            delta = (new - W).abs().max().item()
            W = new
            if delta < 1e-7:
                break
        self.W = W.cpu().numpy()

    def _after(self, W, u, t):
        over = ((u >= self.TARGET) | (t >= self.TARGET)) & (u != t)
        ui = np.clip((u - self.LO) // self.STEP, 0, self.n - 1)
        ti = np.clip((t - self.LO) // self.STEP, 0, self.n - 1)
        return np.where(over, (u > t).astype(float), W[ui, ti])

    def value_after(self, u, t):
        """Win chance once a round has left the scores at (u, t)."""
        return self._after(self.W, np.asarray(u), np.asarray(t))

    def sample_states(self, n_games, rng):
        """Scores before each round of simulated games (draws from the table's rounds)."""
        out = []
        u = np.zeros(n_games, int)
        t = np.zeros(n_games, int)
        live = np.ones(n_games, bool)
        for _ in range(60):
            idx = np.flatnonzero(live)
            if len(idx) == 0:
                break
            out.append(np.stack([u[idx], t[idx]], 1))
            k = rng.choice(len(self.p), size=len(idx), p=self.p)
            u[idx] += self.pairs[k, 0]
            t[idx] += self.pairs[k, 1]
            over = ((u >= self.TARGET) | (t >= self.TARGET)) & (u != t)
            live &= ~over
        return np.concatenate(out)


def labels(win, res, ctx):
    """Change in win chance (points) from the call, at score contexts ctx."""
    u, t = ctx[:, 0], ctx[:, 1]
    without = win.value_after(u + res[:, 0], t + res[:, 1])
    with_ = win.value_after(u + res[:, 2], t + res[:, 3])
    return 100.0 * (with_ - without)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="calls.tcall")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs=2, default=[256, 128])
    ap.add_argument("--head", type=int, default=64)
    ap.add_argument("--zero-frac", type=float, default=0.25, help="share of samples at 0-0")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    d = read_records(args.data)
    fv, sd, ed = d["dims"]
    n = len(d["kind"])
    print(f"{n} records ({(d['kind'] == 0).sum()} grand, {(d['kind'] == 1).sum()} small), "
          f"{len(d['rounds'])} rounds, {time.time() - t0:.1f}s")

    win = WinTable(d["rounds"])
    states = win.sample_states(20000, rng)
    print(f"win table: P(win | 0-0) {100 * win.value_after(np.array([0]), np.array([0]))[0]:.1f}%, "
          f"{len(states)} game states, {time.time() - t0:.1f}s")

    def contexts(m):
        c = states[rng.integers(len(states), size=m)]
        c[rng.random(m) < args.zero_frac] = 0
        return c

    # Hold out whole rounds.
    rid = np.unique(d["round"])
    val_rounds = set(rng.choice(rid, size=int(len(rid) * args.val), replace=False).tolist())
    is_val = np.array([r in val_rounds for r in d["round"]])
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    X = torch.tensor(d["x"], dtype=torch.uint8, device=dev)
    kind = torch.tensor(d["kind"].astype(np.int64), device=dev)
    res = d["res"]
    # Fixed validation contexts (and 0-0) for comparable numbers.
    va_ctx = contexts(len(va))
    va_y = labels(win, res[va], va_ctx)
    va_y0 = labels(win, res[va], np.zeros((len(va), 2), int))

    in_dim = sd + ed + 3

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = args.hidden
            self.trunk = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
            self.heads = nn.ModuleList(
                [nn.Sequential(nn.Linear(h2, args.head), nn.ReLU(), nn.Linear(args.head, 1)) for _ in range(2)]
            )

        def forward(self, x, k):
            h = self.trunk(x)
            g = self.heads[0](h).squeeze(-1)
            s = self.heads[1](h).squeeze(-1)
            return torch.where(k == 0, g, s)

    def inputs(idx, ctx):
        x = X[idx].float() / 255.0
        extra = torch.tensor(
            np.stack([d["kind"][idx] == 0, ctx[:, 0] / 1000.0, ctx[:, 1] / 1000.0], 1), dtype=torch.float32, device=dev
        )
        return torch.cat([x, extra], 1)

    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def predict(idx, ctx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 8192):
                j = idx[i : i + 8192]
                out.append(model(inputs(j, ctx[i : i + 8192]), kind[j]).cpu().numpy())
        model.train()
        return np.concatenate(out) if out else np.zeros(0)

    def report(tag):
        lines = []
        for ctx, y, name in [(va_ctx, va_y, "game scores"), (np.zeros((len(va), 2), int), va_y0, "0-0")]:
            p = predict(va, ctx)
            for k, kname in [(0, "grand"), (1, "small")]:
                m = d["kind"][va] == k
                if m.sum() == 0:
                    continue
                yk, pk = y[m], p[m]
                r2 = 1 - ((yk - pk) ** 2).mean() / yk.var()
                h = d["heur"][va][m] == 2
                # Offline value of each policy: mean gain of the calls it makes.
                v_model = np.where(pk > 0, yk, 0).mean()
                v_heur = np.where(h, yk, 0).mean()
                v_best = np.maximum(yk, 0).mean()
                lines.append(
                    f"  {kname} @ {name}: R2 {r2:.3f} | win-chance gain per decision: model {v_model:+.3f} "
                    f"(calls {100 * (pk > 0).mean():.1f}%), v2 {v_heur:+.3f} (calls {100 * h.mean():.1f}%), "
                    f"hindsight {v_best:+.3f}"
                )
        print(tag)
        print("\n".join(lines))
        sys.stdout.flush()

    report("before training")
    for ep in range(args.epochs):
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            ctx = contexts(len(idx))
            y = torch.tensor(labels(win, res[idx], ctx), dtype=torch.float32, device=dev)
            loss = ((model(inputs(idx, ctx), kind[idx]) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            report(f"epoch {ep + 1}: train mse {tot / len(tr):.2f} ({time.time() - t0:.0f}s)")

    # Export.
    def layers(seq):
        return [m for m in seq if isinstance(m, nn.Linear)]

    with open(args.out, "wb") as f:
        f.write(b"TCLL")
        f.write(struct.pack("<3I", MODEL_VERSION, fv, in_dim))
        for stack in [layers(model.trunk), layers(model.heads[0]), layers(model.heads[1])]:
            f.write(struct.pack("<I", len(stack)))
            for l in stack:
                w = l.weight.detach().cpu().numpy().astype("<f4")
                b = l.bias.detach().cpu().numpy().astype("<f4")
                f.write(struct.pack("<2I", w.shape[1], w.shape[0]))
                f.write(w.tobytes())
                f.write(b.tobytes())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
