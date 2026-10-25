"""Train a policy + value net on `tichu selfplay` records and export it as a
.tnet model for the Rust `net:` / `vpimcN:` bots.

    python train/train.py data/*.bin --out models/h1.tnet

The policy target is the recorded label by default (imitation). With
`--policy q`, files that carry PIMC Q-values are distilled instead: the
target is softmax(Q / --tau) over the candidates. The value head regresses
the final round margin (in units of --value-scale points).
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tichu_data import Records  # noqa: E402


class TichuNet(nn.Module):
    """Shared trunk over the state (and, with `seq_hidden`, a GRU over the
    round's history tokens); value head; per-candidate policy head on
    trunk ⊕ action. Mirrors `Net` in src/bots/net.rs (ReLU after every layer
    but the last of each head, and after the trunk)."""

    def __init__(self, state_dim, action_dim, hidden=256, depth=2, head=128, token_dim=0, seq_hidden=0):
        super().__init__()
        self.seq_hidden = seq_hidden
        self.gru = nn.GRU(token_dim, seq_hidden, batch_first=True) if seq_hidden else None
        dims = [state_dim + seq_hidden] + [hidden] * depth
        self.trunk = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.value = nn.ModuleList([nn.Linear(hidden, head // 2), nn.Linear(head // 2, 1)])
        self.policy = nn.ModuleList([nn.Linear(hidden + action_dim, head), nn.Linear(head, 1)])

    def encode_seq(self, toks, lens):
        """toks [B, L, D] (zero-padded), lens [B] -> the GRU state after each
        sequence's last token, zeros for an empty one [B, H]."""
        if self.gru is None:
            return toks.new_zeros(toks.shape[0], 0)
        if toks.shape[1] == 0:
            return toks.new_zeros(toks.shape[0], self.seq_hidden)
        out, _ = self.gru(toks)
        last = out[torch.arange(len(lens), device=toks.device), (lens - 1).clamp(min=0)]
        return last * (lens > 0).unsqueeze(-1).to(last.dtype)

    @staticmethod
    def _head(layers, x):
        for i, l in enumerate(layers):
            x = l(x)
            if i + 1 < len(layers):
                x = F.relu(x)
        return x

    def forward_trunk(self, s, seq):
        s = torch.cat([s, seq], dim=-1)
        for l in self.trunk:
            s = F.relu(l(s))
        return s

    def forward(self, s, a, toks, lens):
        """s [B, S], a [B, K, A], toks [B, L, D], lens [B] -> value [B], logits [B, K]."""
        h = self.forward_trunk(s, self.encode_seq(toks, lens))
        v = self._head(self.value, h).squeeze(-1)
        hk = h.unsqueeze(1).expand(-1, a.shape[1], -1)
        logits = self._head(self.policy, torch.cat([hk, a], dim=-1)).squeeze(-1)
        return v, logits


def write_tnet(model, path, args, data, check):
    layers = lambda ml: [list(l.weight.shape) for l in ml]  # noqa: E731
    header = dict(
        format=1, feature_version=int(data.feature_version),
        state_dim=int(data.state_dim), action_dim=int(data.action_dim),
        trunk=layers(model.trunk), value=layers(model.value), policy=layers(model.policy),
        seq=dict(token_dim=int(data.token_dim), hidden=model.seq_hidden) if model.gru is not None else None,
        value_scale=args.value_scale, check=check,
        trained=dict(data=args.data, policy=args.policy, records=len(data), epochs=args.epochs),
    )
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(b"TNET" + struct.pack("<I", len(hb)) + hb)
        for ml in (model.trunk, model.value, model.policy):
            for l in ml:
                f.write(l.weight.detach().cpu().float().numpy().astype("<f4").tobytes())
                f.write(l.bias.detach().cpu().float().numpy().astype("<f4").tobytes())
        if model.gru is not None:
            for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                f.write(getattr(model.gru, name).detach().cpu().float().numpy().astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="record files from `tichu selfplay`")
    ap.add_argument("--out", required=True, help="output .tnet path")
    ap.add_argument("--init", help="start from this .pt checkpoint (saved next to every .tnet)")
    ap.add_argument("--policy", choices=["label", "q"], default="label")
    ap.add_argument("--tau", type=float, default=20.0, help="Q temperature in points (--policy q)")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--value-scale", type=float, default=100.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--head", type=int, default=128)
    ap.add_argument("--seq-hidden", type=int, default=64, help="GRU size over the history tokens (0 = no GRU)")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    t0 = time.time()
    data = Records(args.data)
    has_q = ~np.isnan(data.q)
    print(f"{len(data)} records, {len(data.actions)} candidates "
          f"({has_q.mean():.0%} with Q), loaded in {time.time() - t0:.1f}s", flush=True)
    if args.policy == "q" and not has_q.any():
        sys.exit("--policy q needs files written with `tichu selfplay --q N`")

    # Split by round so positions from one round don't straddle train/val.
    rng = np.random.default_rng(args.seed)
    groups = np.unique(data.group)
    val_groups = rng.choice(groups, size=max(1, int(len(groups) * args.val_frac)), replace=False)
    is_val = np.isin(data.group, val_groups)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    # Everything lives on the device as uint8; batches are gathered there.
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
    states, actions, tokens = T(data.states), T(data.actions), T(data.tokens)
    tok_off, tok_len = T(data.tok_off), T(data.tok_len)
    act_off, n_opts, label = T(data.act_off), T(data.n_opts), T(data.label)
    margin = T(data.margin / args.value_scale)
    q = T(np.nan_to_num(data.q, nan=0.0))
    rec_has_q = T(np.add.reduceat(has_q, data.act_off) > 0)

    def batch(idx):
        idx = torch.as_tensor(idx, device=dev)
        n = n_opts[idx]
        k = int(n.max())
        ar = torch.arange(k, device=dev)
        mask = ar[None, :] < n[:, None]
        rows = torch.where(mask, act_off[idx][:, None] + ar[None, :], 0)
        s = states[idx].float() / 255.0
        a = actions[rows].float() / 255.0
        lens = tok_len[idx]
        L = int(lens.max()) if args.seq_hidden else 0
        tr = torch.arange(L, device=dev)
        tmask = tr[None, :] < lens[:, None]
        trows = torch.where(tmask, tok_off[idx][:, None] + tr[None, :], 0)
        toks = tokens[trows].float() / 255.0 * tmask[..., None]
        return idx, s, a, mask, rows, toks, lens

    model = TichuNet(data.state_dim, data.action_dim, args.hidden, args.depth, args.head,
                     data.token_dim, args.seq_hidden).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(args.epochs * len(train_idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def losses(idx):
        idx, s, a, mask, rows, toks, lens = batch(idx)
        v, logits = model(s, a, toks, lens)
        logits = logits.masked_fill(~mask, -1e9)
        logp = F.log_softmax(logits, dim=-1)
        if args.policy == "q":
            qt = q[rows].masked_fill(~mask, -1e9) / args.tau
            target = F.softmax(qt, dim=-1)
            pl = -(target * logp).sum(-1)
            pl = torch.where(rec_has_q[idx], pl, -logp.gather(1, label[idx][:, None]).squeeze(1))
        else:
            pl = -logp.gather(1, label[idx][:, None]).squeeze(1)
        vl = F.mse_loss(v, margin[idx], reduction="none")
        acc = (logits.argmax(-1) == label[idx]).float()
        return pl.mean(), vl.mean(), acc.mean()

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = np.zeros(3)
        chunks = np.array_split(val_idx, max(1, len(val_idx) // 4096))
        for c in chunks:
            tot += np.array([x.item() for x in losses(c)]) * len(c)
        model.train()
        return tot / max(1, len(val_idx))

    var = float(np.var(data.margin[val_idx] / args.value_scale))
    print(f"train {len(train_idx)}, val {len(val_idx)}; val margin variance {var:.3f} "
          f"(value MSE to beat); {steps} steps on {dev}", flush=True)
    t0, step = time.time(), 0
    log_every = max(1, steps // 20)
    while step < steps:
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            pl, vl, _ = losses(perm[i:i + args.batch])
            loss = pl + args.value_weight * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0 or step == steps:
                vp, vv, va = evaluate()
                print(f"step {step}/{steps}  train pol {pl.item():.3f} val {vp:.3f} | "
                      f"value mse {vv:.3f} (R² {1 - vv / var:.3f}) | label acc {va:.1%} | "
                      f"{time.time() - t0:.0f}s", flush=True)
            if step >= steps:
                break

    # A worked example in the header: the Rust loader recomputes it and
    # refuses the model if the two implementations disagree.
    model.eval().cpu()
    # A record with some history, so the check covers the GRU too.
    i = int(next((j for j in val_idx if data.tok_len[j] >= 5), val_idx[0]))
    o, n = int(data.act_off[i]), int(data.n_opts[i])
    t0, tl = int(data.tok_off[i]), (int(data.tok_len[i]) if args.seq_hidden else 0)
    tk = data.tokens[t0:t0 + tl]
    with torch.no_grad():
        s = torch.from_numpy(data.states[i:i + 1]).float() / 255.0
        a = torch.from_numpy(data.actions[o:o + n][None]).float() / 255.0
        toks = torch.from_numpy(tk[None]).float() / 255.0
        v, lg = model(s, a, toks, torch.tensor([tl]))
    check = dict(state=data.states[i].tolist(), actions=data.actions[o:o + n].reshape(-1).tolist(),
                 tokens=tk.reshape(-1).tolist(),
                 value=float(v[0]) * args.value_scale, logits=[float(x) for x in lg[0]])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_tnet(model, args.out, args, data, check)
    torch.save(model.state_dict(), os.path.splitext(args.out)[0] + ".pt")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
