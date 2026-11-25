"""Offline test of a truncated all-seeing critic, on `examples/oracle_data.rs`
files.

    python train/oracle_probe.py runs/ceiling/orc.bin --init runs/dmc1/snap00190.pt

1. Trains an all-seeing value net (input: the position as all four players
   see it) on final margins, and for comparison the same net on the mover's
   own view only. Reports held-out R².
2. Labels every option of the critic rows three ways: the full playout's
   margin ("full"), the all-seeing value right after the move ("after"), and
   the all-seeing value `horizon` decisions later ("horizon"; the real margin
   when the round ended first).
3. For each labelling: what choosing by the label itself is worth on
   held-out rows, and what the move scores of --init learn from it. "Gain"
   is the full playout's margin of the chosen option minus that of the net's
   own first choice, in points per decision. A choice made without looking
   at a row's full margins gets an unbiased gain.

The last 15% of the file (whole rounds) is held out throughout.
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc import arch_of, critic_loss, option_advantages  # noqa: E402
from train import TichuNet  # noqa: E402


def read(path, K):
    buf = open(path, "rb").read()
    assert buf[:4] == b"TORC"
    S, A = struct.unpack_from("<II", buf, 4)
    data = np.frombuffer(buf, np.uint8)
    O = 4 * S
    opt = A + 4 + 2 * (1 + 4 + O)
    vals, crit, pos, order = [], [], 12, []
    while pos < len(buf):
        if buf[pos] == 0:
            vals.append(pos)
            order.append(0)
            pos += 5 + O
        else:
            crit.append(pos)
            order.append(1)
            pos += 2 + S + buf[pos + 1] * opt
    assert pos == len(buf), "truncated"
    f4 = lambda p: np.frombuffer(buf, "<f4", 1, p)[0]  # noqa: E731
    vx = np.stack([data[p + 5:p + 5 + O] for p in vals])
    vy = np.array([f4(p + 1) for p in vals], np.float32)
    n = len(crit)
    cs = np.zeros((n, S), np.uint8)
    ca = np.zeros((n, K, A), np.uint8)
    full = np.full((n, K), np.nan, np.float32)
    ended = np.zeros((n, 2, K), bool)
    marg = np.zeros((n, 2, K), np.float32)
    feat = np.zeros((n, 2, K, O), np.uint8)
    for i, p in enumerate(crit):
        k = min(buf[p + 1], K)
        cs[i] = data[p + 2:p + 2 + S]
        q = p + 2 + S
        for j in range(k):
            ca[i, j] = data[q:q + A]
            full[i, j] = f4(q + A)
            r = q + A + 4
            for t in range(2):
                ended[i, t, j] = buf[r] == 1
                marg[i, t, j] = f4(r + 1)
                feat[i, t, j] = data[r + 5:r + 5 + O]
                r += 5 + O
            q += opt
    # Position in the file, so the hold-out is whole rounds at the end.
    order = np.array(order)
    vpos, cpos = np.flatnonzero(order == 0), np.flatnonzero(order == 1)
    return S, A, vx, vy, cs, ca, full, ended, marg, feat, vpos, cpos, len(order)


class ValueNet(nn.Module):
    def __init__(self, d, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x.float() / 255.0).squeeze(-1)


def fit_value(x, y, xv, yv, dev, epochs, tag):
    """Train on (x, y / 100), report held-out R² on (xv, yv); returns the net."""
    torch.manual_seed(0)
    net = ValueNet(x.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    n, B = len(x), 1024
    steps = epochs * (n // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    best = (-1e9, None)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        net.train()
        for i in range(0, n - B + 1, B):
            idx = perm[i:i + B]
            loss = F.mse_loss(net(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(xv[i:i + 8192]) for i in range(0, len(xv), 8192)])
            r2 = 1 - F.mse_loss(pred, yv).item() / yv.var().item()
        if r2 > best[0]:
            best = (r2, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    print(f"{tag}: held-out R² {best[0]:.3f} ({n} training positions)", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--init", required=True, help="the .pt whose move scores are trained on each labelling")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--value-epochs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, A, vx, vy, cs, ca, full, ended, marg, feat, vpos, cpos, total = read(args.data, args.k)
    cut = int(total * 0.85)
    vtr, vte = vpos < cut, vpos >= cut
    ctr, cte = cpos < cut, cpos >= cut
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    valid = ~np.isnan(full)
    print(f"{len(vy)} value rows, {len(cs)} critic rows ({valid.sum(1).mean():.1f} options each); "
          f"{vte.sum()} and {cte.sum()} held out; margin sd {vy.std():.0f} pts")

    # 1. Value nets. The critic's playouts are more (position, outcome)
    # pairs for the all-seeing net: each option's two stages with its full margin.
    live = valid[ctr][:, None, :] & ~ended[ctr]
    ex = feat[ctr][live]
    ey = np.broadcast_to(full[ctr][:, None, :], live.shape)[live]
    yv = T(vy[vte] / 100)
    one = fit_value(T(vx[vtr][:, :S]), T(vy[vtr] / 100), T(vx[vte][:, :S]), yv, dev, args.value_epochs, "value from my own view  ")
    del one
    fit_value(T(vx[vtr]), T(vy[vtr] / 100), T(vx[vte]), yv, dev, args.value_epochs, "all-seeing value        ")
    oracle = fit_value(T(np.concatenate([vx[vtr], ex])), T(np.concatenate([vy[vtr], ey]) / 100), T(vx[vte]), yv, dev,
                       args.value_epochs, "all-seeing, + playouts  ")

    # 2. Labels.
    labels = {"full": full.copy()}
    with torch.no_grad():
        for t, name in ((0, "after"), (1, "horizon")):
            f = feat[:, t].reshape(-1, 4 * S)
            pred = np.concatenate([oracle(T(f[i:i + 8192])).cpu().numpy() for i in range(0, len(f), 8192)]) * 100
            lab = np.where(ended[:, t], marg[:, t], pred.reshape(len(cs), args.k))
            labels[name] = np.where(valid, lab, np.nan).astype(np.float32)
    print(f"round already over: right after the move {ended[:, 0][valid].mean():.1%}, by the horizon {ended[:, 1][valid].mean():.1%}")

    def centred(x):
        return x - np.nanmean(x, 1, keepdims=True)

    fc = centred(full[cte])
    print("\nlabels on held-out rows (their own choice, against the full playouts):")
    for name, lab in labels.items():
        l = lab[cte]
        pick = np.nanargmax(l, 1)
        g = full[cte][np.arange(len(l)), pick] - full[cte][:, 0]
        lc = centred(l)
        r2 = np.nansum(lc * fc) ** 2 / (np.nansum(lc * lc) * np.nansum(fc * fc))
        note = "  (hindsight: it is the full playout)" if name == "full" else ""
        print(f"  {name:<8} sd of differences {np.sqrt(np.nanmean(lc ** 2)):5.1f} pts; R² vs full playout {r2:.3f}; "
              f"gain {g.mean():+.2f} ± {1.96 * g.std() / len(g) ** 0.5:.2f}{note}")

    # 3. Move scores trained on each labelling.
    sd = torch.load(args.init, map_location=dev)
    a = arch_of(sd)
    s_tr, a_tr = T(cs[ctr]), T(ca[ctr])
    s_te, a_te, f_te = T(cs[cte]), T(ca[cte]), T(full[cte])

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            adv = torch.cat([option_advantages(model, s_te[i:i + 4096], a_te[i:i + 4096]) for i in range(0, len(s_te), 4096)])
            m = ~torch.isnan(f_te)
            _, se, sst, _ = critic_loss(adv, f_te, 100.0)
            pick = torch.where(m, adv, torch.full_like(adv, -1e9)).argmax(-1)
            g = f_te.gather(1, pick[:, None]).squeeze(1) - f_te[:, 0]
        model.train()
        return 1 - (se / sst).item(), (pick != 0).float().mean().item(), g.mean().item(), 1.96 * g.std().item() / len(g) ** 0.5

    print(f"\nmove scores of {args.init}, trained on each labelling ({ctr.sum()} rows, {args.epochs} epochs, lr {args.lr}):")
    for name in ("(untrained)", "full", "after", "horizon"):
        torch.manual_seed(0)
        model = TichuNet(S, A, a["hidden"], a["depth"], a["head"], 48, 0, a["value_net"]).to(dev)
        model.load_state_dict(sd)
        if name != "(untrained)":
            q = T(labels[name][ctr])
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            n, B = len(s_tr), 512
            for ep in range(args.epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(0, n - B + 1, B):
                    idx = perm[i:i + B]
                    loss = critic_loss(option_advantages(model, s_tr[idx], a_tr[idx]), q[idx], 100.0)[0]
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        r2, changed, g, ci = evaluate(model)
        print(f"  {name:<12} R² vs full playout {r2:.3f}; changes the choice in {changed:.1%}; gain {g:+.2f} ± {ci:.2f} pts", flush=True)


if __name__ == "__main__":
    main()
