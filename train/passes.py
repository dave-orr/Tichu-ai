"""Train a pass model on `tichu passlabel` records and export it as a .tpass
for the Rust `passnet:` bot.

    python train/passes.py runs/pass1/*.bin --out runs/pass1/p1.tpass

Each record is one pass decision: every candidate partner card with the
round's result when it was passed on the real deal. The model scores a
candidate from its features; the loss is the squared error between the
scores and the margins, both centred over the decision's candidates, so only
differences between candidates are learned (the deal's luck is shared by
all of them).

The last --val-frac of the rounds is held out. On it:
- R²: share of the centred margins' variance the centred scores explain;
- gain: the margin of the model's choice minus the margin of the rule's own
  pass, in points per pass decision, for several thresholds T (the model
  only overrides the rule when it predicts more than T points). The choice
  never sees the held-out margins, so the gain is unbiased; "hindsight" is
  the best candidate picked by its own margin, which no passer can reach.
- the same split by who called Tichu or Grand when the rule's pass was played.
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

MODEL_VERSION = 1
SCALE = 100.0


def read(paths):
    S, A, M, F_, G, R = [], [], [], [], [], []
    dim = None
    for fi, path in enumerate(paths):
        buf = open(path, "rb").read()
        magic, version, d = struct.unpack_from("<4sII", buf, 0)
        if magic != b"TPAS" or version != 1:
            raise ValueError(f"{path}: not a v1 pass record file")
        dim = d
        data = np.frombuffer(buf, np.uint8)
        row = d + 5
        pos = 12
        while pos + 7 <= len(buf):
            rnd, _seat, n, rule = struct.unpack_from("<IBBB", buf, pos)
            pos += 7
            if pos + n * row > len(buf):
                break  # a file still being written
            block = data[pos:pos + n * row].reshape(n, row)
            pos += n * row
            S.append(block[:, :d])
            mt = np.ascontiguousarray(block[:, d:d + 4]).view("<i2").astype(np.float32)
            M.append(mt[:, 0] - mt[:, 1])
            F_.append(block[:, d + 4])
            A.append(n)
            R.append(rule)
            G.append(rnd + (fi << 32))
    n = np.array(A)
    return dict(dim=dim, x=np.concatenate(S), margin=np.concatenate(M), flags=np.concatenate(F_), n=n,
                off=np.concatenate([[0], np.cumsum(n)[:-1]]), rule=np.array(R), group=np.array(G))


def padded(d, idx, K):
    """Decisions `idx` as [B, K] row indices into the candidates and a mask."""
    ar = np.arange(K)
    mask = ar[None, :] < d["n"][idx][:, None]
    rows = np.where(mask, d["off"][idx][:, None] + ar[None, :], 0)
    return rows, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = read(args.data)
    N, K = len(d["n"]), int(d["n"].max())
    groups = np.unique(d["group"])
    val_groups = groups[int(len(groups) * (1 - args.val_frac)):]
    is_val = np.isin(d["group"], val_groups)
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    x = torch.from_numpy(d["x"]).to(dev)
    y = torch.from_numpy(d["margin"] / SCALE).to(dev)
    print(f"{N} pass decisions ({len(groups)} rounds), {K} candidates at most; {len(va)} held out; "
          f"sd of a candidate's margin around its decision's mean "
          f"{np.sqrt(np.mean([np.var(d['margin'][o:o + n]) for o, n in zip(d['off'], d['n'])])):.0f} pts")

    model = nn.Sequential(nn.Linear(d["dim"], args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
                          nn.Linear(args.hidden, args.hidden // 2), nn.ReLU(), nn.Linear(args.hidden // 2, 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * max(1, len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def scores(idx):
        rows, mask = padded(d, idx, K)
        rows, mask = torch.from_numpy(rows).to(dev), torch.from_numpy(mask).to(dev)
        s = model(x[rows].float() / 255.0).squeeze(-1)
        return s, y[rows], mask

    def centred(v, mask):
        m = mask.float()
        return (v - (v * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)) * m

    @torch.no_grad()
    def evaluate(show=False):
        model.eval()
        S, Y, Mk = [], [], []
        for i in range(0, len(va), 2048):
            s, yy, mask = scores(va[i:i + 2048])
            S.append(s)
            Y.append(yy)
            Mk.append(mask)
        s, yy, mask = torch.cat(S), torch.cat(Y), torch.cat(Mk)
        model.train()
        sc, yc = centred(s, mask), centred(yy, mask)
        r2 = 1 - ((sc - yc) ** 2).sum().item() / (yc ** 2).sum().item()
        if not show:
            return r2
        rule = torch.from_numpy(d["rule"][va]).to(dev)[:, None]
        s_rule, y_rule = s.gather(1, rule), yy.gather(1, rule).squeeze(1)
        sm = s.masked_fill(~mask, -1e9)
        best, pick = sm.max(-1)
        fl = torch.from_numpy(d["flags"][d["off"][va] + d["rule"][va]]).to(dev)
        splits = [("all", torch.ones_like(fl, dtype=torch.bool)), ("my team called", (fl & 1) > 0),
                  ("only opponents called", ((fl & 1) == 0) & ((fl & 2) > 0)), ("nobody called", (fl & 3) == 0)]
        print(f"held-out R² {r2:.3f}")
        hind = (yy.masked_fill(~mask, -1e9).max(-1).values - y_rule) * SCALE
        for name, sel in splits:
            line = f"  {name:<22} {int(sel.sum()):>6} decisions; hindsight {hind[sel].mean().item():+.1f};"
            for T in (0, 5, 10, 20):
                use = (best - s_rule.squeeze(1)) * SCALE > T
                g = torch.where(use, yy.gather(1, pick[:, None]).squeeze(1) - y_rule, torch.zeros_like(y_rule))[sel] * SCALE
                ci = 1.96 * g.std().item() / max(1, len(g)) ** 0.5
                line += f"  T={T}: {g.mean().item():+.2f} ± {ci:.2f} ({use[sel].float().mean().item():.0%} changed)"
            print(line)
        return r2

    best = (-1e9, None)
    step = 0
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            s, yy, mask = scores(perm[i:i + args.batch])
            loss = ((centred(s, mask) - centred(yy, mask)) ** 2).sum() / mask.float().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
        r2 = evaluate()
        if r2 > best[0]:
            best = (r2, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"epoch {ep + 1}: train loss {tot / max(1, len(tr) // args.batch):.4f}, held-out R² {r2:.4f}", flush=True)
    model.load_state_dict(best[1])
    evaluate(show=True)

    layers = [m for m in model if isinstance(m, nn.Linear)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"TPSM" + struct.pack("<IIfI", MODEL_VERSION, d["dim"], SCALE, len(layers)))
        for l in layers:
            w = l.weight.detach().cpu().numpy().astype("<f4")
            f.write(struct.pack("<II", w.shape[1], w.shape[0]))
            f.write(w.tobytes())
            f.write(l.bias.detach().cpu().numpy().astype("<f4").tobytes())
    torch.save(model.state_dict(), os.path.splitext(args.out)[0] + ".pt")
    # A worked example for checking the Rust side by hand.
    model.eval()
    with torch.no_grad():
        s0 = model(x[:1].float() / 255.0).item() * SCALE
    print(f"wrote {args.out} (first candidate of the first record scores {s0:.3f} pts)")


if __name__ == "__main__":
    main()
