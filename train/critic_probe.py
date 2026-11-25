"""Offline test: how much of the all-seeing critic's labels can a net learn
when it trains on them alone (no game-result target on the move scores)?

    python train/critic_probe.py runs/dmc-critic/chunk*.bin --init runs/dmc1/snap00190.pt

Trains the policy head (and trunk) on `selfplay --critic` records only; the
last file is held out. Reports, on the held-out records:
- R² of the centred advantages against the centred critic margins;
- how often the net's top option is one of the critic's best;
- gain: the critic's margin for the net's top option minus that of the
  option that was played (label 0), in points per graded decision. The pick
  never sees the held-out labels, so this is an unbiased estimate of what
  switching to the pick is worth on real deals (one-step deviation).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc import arch_of, critic_loss, option_advantages, read_played  # noqa: E402
from train import TichuNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--init", help=".pt to start from (else fresh, --hidden/--head)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--head", type=int, default=128)
    ap.add_argument("--loss", choices=["mse", "huber", "rank"], default="mse",
                    help="mse/huber on centred margins; rank = cross-entropy to softmax(margin / tau)")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = [read_played(p, args.k)[5] for p in args.data]
    tr = [torch.from_numpy(np.concatenate([p[i] for p in parts[:-1]])).to(dev) for i in range(3)]
    va = [torch.from_numpy(parts[-1][i]).to(dev) for i in range(3)]
    S, A = tr[0].shape[1], tr[1].shape[2]
    if args.init:
        sd = torch.load(args.init, map_location=dev)
        a = arch_of(sd)
        model = TichuNet(S, A, a["hidden"], a["depth"], a["head"], 48, 0, a["value_net"]).to(dev)
        model.load_state_dict(sd)
    else:
        model = TichuNet(S, A, args.hidden, 2, args.head, 48, 0, 0).to(dev)
    print(f"{len(tr[0])} train, {len(va[0])} held-out critic records; init {args.init}; loss {args.loss}, lr {args.lr}")

    def loss_of(adv, q):
        if args.loss == "mse":
            return critic_loss(adv, q, 100.0)[0]
        m = ~torch.isnan(q)
        mf = m.float()
        k = mf.sum(-1, keepdim=True).clamp(min=1)
        qq = torch.where(m, q / 100.0, torch.zeros_like(q))
        if args.loss == "huber":
            qc = (qq - (qq * mf).sum(-1, keepdim=True) / k) * mf
            ac = (adv - (adv * mf).sum(-1, keepdim=True) / k) * mf
            return (F.huber_loss(ac, qc, delta=0.5, reduction="none") * mf).sum() / mf.sum()
        target = F.softmax((qq * 100.0 / args.tau).masked_fill(~m, -1e9), dim=-1)
        logp = F.log_softmax((adv * 100.0 / args.tau).masked_fill(~m, -1e9), dim=-1)
        return -(target * logp).sum(-1).mean()

    @torch.no_grad()
    def evaluate():
        model.eval()
        se = sst = 0.0
        agree, gain, best_gain = [], [], []
        for i in range(0, len(va[0]), 4096):
            s, a, q = (x[i:i + 4096] for x in va)
            adv = option_advantages(model, s, a)
            _, e, t, _ = critic_loss(adv, q, 100.0)
            se += e.item()
            sst += t.item()
            m = ~torch.isnan(q)
            qm = torch.where(m, q, torch.full_like(q, -1e9))
            pick = torch.where(m, adv, torch.full_like(adv, -1e9)).argmax(-1)
            qp = q.gather(1, pick[:, None]).squeeze(1)
            agree.append((qp == qm.max(-1).values).float())
            gain.append(qp - q[:, 0])
            best_gain.append(qm.max(-1).values - q[:, 0])
        model.train()
        g = torch.cat(gain)
        return (1 - se / sst, torch.cat(agree).mean().item(), g.mean().item(),
                1.96 * g.std().item() / len(g) ** 0.5, torch.cat(best_gain).mean().item())

    def show(tag):
        r2, ag, g, ci, bg = evaluate()
        print(f"{tag}: held-out R² {r2:.3f}, top option is a critic-best {ag:.1%}, "
              f"gain over the played move {g:+.2f} ± {ci:.2f} pts (hindsight best {bg:+.1f})", flush=True)

    show("start")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = len(tr[0])
    steps = args.epochs * (n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_of(option_advantages(model, tr[0][idx], tr[1][idx]), tr[2][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        show(f"epoch {ep + 1} (train loss {tot / (n // args.batch):.3f})")


if __name__ == "__main__":
    main()
