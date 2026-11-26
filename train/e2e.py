"""One net for every kind of decision, trained on `tichu e2edata` records.

    tichu e2edata --bot BOT --games 2000 --seed 1 --out runs/e2e/part001.bin.gz
    python train/e2e.py runs/e2e/part*.bin.gz --val runs/e2e/val.bin.gz --out runs/e2e/a.pt

Every record is one decision of any kind (Grand, Small Tichu, a pass card, a
play, a bomb, the wish, the Dragon gift): what the seat saw, the option
taken, and how the round ended. The net scores an option from a shared body
over the position (and, with --seq, a GRU over the round's history tokens)
plus the option's own features:

- Q(position, option) = V(position) + A(position, option): my team's round
  margin after taking the option, in units of 100 points, regressed onto the
  round's result;
- who goes out first (me, next, partner, previous), for the position and for
  each option;
- where each card I can't see is (next, partner, previous), which self-play
  knows exactly.

Each file is read, trained on once (--epochs passes over the whole list;
keep it at 1 for a large net: every record of a round carries the same
result, and a net that sees rounds repeatedly learns to recognise them) and
dropped, so the data never has to fit in memory at once.

Reported on the held-out file (--val), by kind of decision:
- fit: R² of V and of Q against the round's margin, on the games' own
  decisions;
- cards: hidden cards placed right, against guessing by how many cards each
  seat holds;
- branches: where the generator tried other options on the same deal, how
  often the net's best-scored option was a best one, and the points gained
  per decision by taking the net's choice among the tried options over the
  option the data's bot took (0 = no better than that bot; "hindsight" is
  the ceiling no player can reach).
"""

import argparse
import gzip
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADER = struct.Struct("<4s6I")
BLOCK = struct.Struct("<IHHBB6hHH")
FORMAT_VERSION = 2
KINDS = ["grand", "small", "pass", "play", "bomb", "wish", "dragon"]
SCALE = 100.0
# Teacher log-probabilities below this are all "never".
LP_FLOOR = -8.0


class Data:
    """The records of some e2edata files, as arrays (see src/e2e.rs)."""

    def __init__(self, paths):
        recs, blocks, toks = [], [], []
        dims = None
        tok_at = 0
        for fi, path in enumerate(paths):
            buf = gzip.open(path, "rb").read()
            magic, version, fv, S, C, D, H = HEADER.unpack_from(buf, 0)
            if magic != b"TE2E" or version != FORMAT_VERSION:
                raise ValueError(f"{path}: not a v{FORMAT_VERSION} e2edata file")
            d = dict(feature_version=fv, state_dim=S, cand_dim=C, token_dim=D, hidden_dim=H)
            if dims is not None and d != dims:
                raise ValueError(f"{path}: layout {d}, expected {dims}")
            dims = d
            R = 8 + S + C + H + 4
            data = np.frombuffer(buf, dtype=np.uint8)
            pos = HEADER.size
            while pos < len(buf):
                b = BLOCK.unpack_from(buf, pos)
                nt, nr = b[-2], b[-1]
                pos += BLOCK.size
                toks.append(data[pos:pos + nt * D])
                pos += nt * D
                recs.append(data[pos:pos + nr * R].reshape(nr, R))
                pos += nr * R
                blocks.append((fi,) + b[:-2] + (tok_at, nt, nr))
                tok_at += nt
            if pos != len(buf):
                raise ValueError(f"{path}: truncated")
        self.dims = dims
        S, C, H, D = dims["state_dim"], dims["cand_dim"], dims["hidden_dim"], dims["token_dim"]
        rows = np.concatenate(recs) if recs else np.zeros((0, 8 + S + C + H + 4), np.uint8)
        B = np.array(blocks, dtype=np.int64).reshape(-1, 15)
        of = np.repeat(np.arange(len(B)), B[:, 14])
        self.kind = rows[:, 0].astype(np.int64)
        self.forced = (rows[:, 1] & 1).astype(bool)
        self.n_options = rows[:, 2].astype(np.int64)
        self.step = rows[:, 3].astype(np.int64)
        group = rows[:, 4].astype(np.int64) | (rows[:, 5].astype(np.int64) << 8)
        self.prefix = rows[:, 6].astype(np.int64) | (rows[:, 7].astype(np.int64) << 8)
        self.state = rows[:, 8:8 + S]
        self.cand = rows[:, 8 + S:8 + S + C]
        self.hidden = rows[:, 8 + S + C:8 + S + C + H]
        # What the data's bot's own net said: its value of the position
        # (points) and its policy's log-probability of this option.
        teacher = np.ascontiguousarray(rows[:, 8 + S + C + H:]).view("<i2").astype(np.float32)
        self.has_tv, self.has_tlp = (rows[:, 1] & 2).astype(bool), (rows[:, 1] & 4).astype(bool)
        self.teacher_v, self.teacher_lp = teacher[:, 0] / 10.0, teacher[:, 1] / 1000.0
        self.tokens = np.concatenate(toks).reshape(-1, D) if toks else np.zeros((0, D), np.uint8)
        fi, game, rnd, playout, seat, first = (B[of, k] for k in range(6))
        self.game, self.playout, self.first = game, playout, first
        self.margin = (B[of, 6] - B[of, 7]).astype(np.float32)
        self.tok_off = B[of, 12]
        # A group is a decision and the other options tried there: unique
        # within a round of a game of a file (and a seat's records).
        key = np.stack([fi, game, rnd, group], 1)
        _, gid = np.unique(key, axis=0, return_inverse=True)
        self.group = np.where(group > 0, gid.ravel() + 1, 0)
        self.rounds = len(np.unique(np.stack([fi, game, rnd], 1)[playout == 0], axis=0)) if len(rows) else 0

    def __len__(self):
        return len(self.kind)


class E2ENet(nn.Module):
    def __init__(self, dims, width=512, depth=4, head=256, seq=128):
        super().__init__()
        S, C, D, H = dims["state_dim"], dims["cand_dim"], dims["token_dim"], dims["hidden_dim"]
        self.arch = dict(width=width, depth=depth, head=head, seq=seq)
        self.seq = seq
        self.gru = nn.GRU(D, seq, batch_first=True) if seq else None
        self.inp = nn.Linear(S + seq, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.value = nn.Sequential(nn.Linear(width, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))
        self.first = nn.Linear(width, 4)
        self.belief = nn.Linear(width, H * 3)
        self.q1 = nn.Linear(width + C, head)
        # advantage, first-out logits for the option, and the teacher
        # policy's log-probability of it
        self.q2 = nn.Linear(head, 1 + 4 + 1)
        self.H = H

    def body(self, s, toks, lens):
        if self.gru is not None:
            if toks.shape[1] == 0:
                e = s.new_zeros(len(s), self.seq)
            else:
                out, _ = self.gru(toks)
                e = out[torch.arange(len(lens), device=s.device), (lens - 1).clamp(min=0)]
                e = e * (lens > 0).unsqueeze(-1).to(e.dtype)
            s = torch.cat([s, e], -1)
        h = F.relu(self.inp(s))
        for l in self.blocks:
            h = h + F.relu(l(h))
        return h

    def forward(self, s, c, toks, lens):
        """-> V [B], Q [B], first-out logits for the position [B, 4] and the
        option [B, 4], hidden-card logits [B, H, 3], the option's policy
        log-probability [B]."""
        h = self.body(s, toks, lens)
        v = self.value(h).squeeze(-1)
        o = self.q2(F.relu(self.q1(torch.cat([h, c], -1))))
        return v, v + o[:, 0], self.first(h), o[:, 1:5], self.belief(h).view(len(s), self.H, 3), o[:, 5]


class Batcher:
    """A Data on the device, cut into batches."""

    def __init__(self, data, dev, seq):
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        self.n, self.dev, self.seq = len(data), dev, seq
        self.state, self.cand, self.hidden = T(data.state), T(data.cand), T(data.hidden)
        self.margin, self.first = T(data.margin) / SCALE, T(data.first)
        self.forced, self.kind = T(data.forced), T(data.kind)
        self.prefix, self.tok_off = T(data.prefix), T(data.tok_off)
        self.tokens = T(data.tokens) if seq else None
        self.has_tv, self.has_tlp = T(data.has_tv), T(data.has_tlp)
        self.teacher_v, self.teacher_lp = T(data.teacher_v) / SCALE, T(data.teacher_lp).clamp(min=LP_FLOOR)

    def get(self, idx):
        s = self.state[idx].float() / 255.0
        c = self.cand[idx].float() / 255.0
        lens = self.prefix[idx]
        if self.seq and len(self.tokens):
            L = int(lens.max())
            at = (self.tok_off[idx][:, None] + torch.arange(L, device=self.dev)[None]).clamp(max=len(self.tokens) - 1)
            toks = self.tokens[at].float() / 255.0
            toks = toks * (torch.arange(L, device=self.dev)[None] < lens[:, None]).unsqueeze(-1)
        else:
            toks = s.new_zeros(len(idx), 0, 1)
        return s, c, toks, lens


def losses(model, b, idx, args):
    s, c, toks, lens = b.get(idx)
    v, q, f_s, f_c, bel, lp = model(s, c, toks, lens)
    y, first, own = b.margin[idx], b.first[idx], ~b.forced[idx]
    ql = F.mse_loss(q, y)
    # V and the position's first-out are about the bot's own choice.
    w = own.float()
    n_own = w.sum().clamp(min=1)
    vl = ((v - y) ** 2 * w).sum() / n_own
    fl = F.cross_entropy(f_c, first) + (F.cross_entropy(f_s, first, reduction="none") * w).sum() / n_own
    hid = b.hidden[idx].long()
    m = hid > 0
    bl = F.cross_entropy(bel[m], hid[m] - 1) if m.any() else ql * 0
    # The teacher: its value where it gave one, its policy on plays.
    wv, wp = b.has_tv[idx].float(), b.has_tlp[idx].float()
    tl = ((v - b.teacher_v[idx]) ** 2 * wv).sum() / wv.sum().clamp(min=1)
    pl = (F.smooth_l1_loss(lp, b.teacher_lp[idx], reduction="none") * wp).sum() / wp.sum().clamp(min=1)
    total = ql + args.value_weight * vl + args.first_weight * fl + args.belief_weight * bl + args.teacher_weight * (tl + 0.5 * pl)
    return total, (ql, vl, fl, bl, tl, pl)


@torch.no_grad()
def evaluate(model, data, dev, args):
    """The held-out report (see the module docs), as a dict."""
    model.eval()
    b = Batcher(data, dev, model.seq)
    V, Q, P, hit, n_hid, ll = [], [], [], [], [], []
    for i in range(0, len(data), 4096):
        idx = torch.arange(i, min(i + 4096, len(data)), device=dev)
        s, c, toks, lens = b.get(idx)
        v, q, _, _, bel, plp = model(s, c, toks, lens)
        P.append(plp.cpu())
        V.append(v.cpu())
        Q.append(q.cpu())
        hid = b.hidden[idx].long()
        m = hid > 0
        lp = F.log_softmax(bel, -1)
        hit.append(((bel.argmax(-1) + 1 == hid) & m).sum(1).cpu())
        ll.append((-(lp.gather(2, (hid - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)) * m).sum(1).cpu())
        n_hid.append(m.sum(1).cpu())
    model.train()
    V, Q, P = torch.cat(V).numpy() * SCALE, torch.cat(Q).numpy() * SCALE, torch.cat(P).numpy()
    hit, ll, n_hid = torch.cat(hit).numpy(), torch.cat(ll).numpy(), torch.cat(n_hid).numpy()
    out = {}
    # Guessing from how many cards each seat holds: the best single guess
    # is the fullest seat, and the log loss is the entropy of the shares.
    counts = np.stack([(data.hidden == k).sum(1) for k in (1, 2, 3)], 1).astype(np.float64)
    tot = counts.sum(1)
    share = counts / np.maximum(tot, 1)[:, None]
    base_ll = -(counts * np.log(np.maximum(share, 1e-12))).sum(1)
    own = (data.playout == 0) & ~data.forced
    for k, name in enumerate(KINDS):
        sel = own & (data.kind == k)
        if sel.sum() < 50:
            continue
        y = data.margin[sel]
        var = y.var()
        r = dict(n=int(sel.sum()), v_r2=1 - ((V[sel] - y) ** 2).mean() / var, q_r2=1 - ((Q[sel] - y) ** 2).mean() / var)
        cards = tot[sel].sum()
        if cards > 0:
            r.update(cards_right=hit[sel].sum() / cards, cards_right_counting=counts[sel].max(1).sum() / cards,
                     cards_ll=ll[sel].sum() / cards, cards_ll_counting=base_ll[sel].sum() / cards)
        tv = sel & data.has_tv
        if tv.sum() >= 50:
            yt = data.margin[tv]
            r.update(teacher_r2=1 - ((data.teacher_v[tv] - yt) ** 2).mean() / yt.var(),
                     v_like_teacher=1 - ((V[tv] - data.teacher_v[tv]) ** 2).mean() / data.teacher_v[tv].var())
        out[name] = r
    # Branch groups: a decision and the other options tried on the same deal.
    g = data.group
    order = np.argsort(g, kind="stable")
    order = order[g[order] > 0]
    starts = np.flatnonzero(np.r_[True, g[order][1:] != g[order][:-1]])
    ends = np.r_[starts[1:], len(order)]
    stats = {}
    for a, e in zip(starts, ends):
        m = order[a:e]
        mine = m[~data.forced[m]]
        if len(m) < 2 or len(mine) != 1:
            continue
        y, q = data.margin[m], Q[m]
        pick = m[int(np.argmax(q))]
        st = stats.setdefault(int(data.kind[mine[0]]), dict(gain=[], hind=[], agree=[], same=[], sse=0.0, sst=0.0))
        if data.has_tlp[m].all():
            # The data's bot took its policy's top option: does the policy head too?
            st["same"].append(float(m[int(np.argmax(P[m]))] == mine[0]))
        st["gain"].append(data.margin[pick] - data.margin[mine[0]])
        st["hind"].append(y.max() - data.margin[mine[0]])
        st["agree"].append(float(data.margin[pick] == y.max()))
        st["sse"] += (((q - q.mean()) - (y - y.mean())) ** 2).sum()
        st["sst"] += ((y - y.mean()) ** 2).sum()
    for k, st in stats.items():
        gain = np.array(st["gain"])
        out.setdefault(KINDS[k], {}).update(
            groups=len(gain), gain=gain.mean(), gain_ci=1.96 * gain.std() / max(1, len(gain)) ** 0.5,
            hindsight=float(np.mean(st["hind"])), agree=float(np.mean(st["agree"])), branch_r2=1 - st["sse"] / max(st["sst"], 1e-9))
        if st["same"]:
            out[KINDS[k]].update(same_as_bot=float(np.mean(st["same"])))
    return {k: {a: (float(x) if not isinstance(x, int) else x) for a, x in v.items()} for k, v in out.items()}


def show(report):
    lines = []
    for name, r in report.items():
        line = f"  {name:7}"
        if "n" in r:
            line += f" {r['n']:8d} decisions: R² V {r['v_r2']:.3f} Q {r['q_r2']:.3f}"
        if "teacher_r2" in r:
            line += f" (the bot's net: {r['teacher_r2']:.3f}; V matches it {r['v_like_teacher']:.3f})"
        if "cards_right" in r:
            line += (f"; cards right {r['cards_right']:.1%} (counting {r['cards_right_counting']:.1%}), "
                     f"log loss {r['cards_ll']:.3f} ({r['cards_ll_counting']:.3f})")
        if "groups" in r:
            line += (f"; {r['groups']} branches: picks a best option {r['agree']:.0%}, gain {r['gain']:+.1f} ± {r['gain_ci']:.1f} "
                     f"(hindsight {r['hindsight']:+.1f}), R² {r['branch_r2']:.3f}")
        if "same_as_bot" in r:
            line += f", policy head picks the bot's play {r['same_as_bot']:.0%}"
        lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="e2edata files, trained on in this order")
    ap.add_argument("--val", required=True, help="a held-out e2edata file (other seeds)")
    ap.add_argument("--out", required=True, help="checkpoint (.pt); a .json log is written beside it")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--head", type=int, default=256)
    ap.add_argument("--seq", type=int, default=128, help="GRU units over the history tokens (0 = no history)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--first-weight", type=float, default=0.1)
    ap.add_argument("--belief-weight", type=float, default=0.5)
    ap.add_argument("--teacher-weight", type=float, default=1.0,
                    help="weight of matching the data's bot's own net (its value, and its policy on plays); 0 = results only")
    ap.add_argument("--eval-every", type=int, default=1, help="files between held-out reports")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    val = Data([args.val])
    model = E2ENet(val.dims, args.width, args.depth, args.head, args.seq).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{n_par} parameters, {model.arch}; held out: {len(val)} records from {val.rounds} rounds; on {dev}", flush=True)
    log, steps, seen, t0 = [], 0, 0, time.time()
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    files = [f for _ in range(args.epochs) for f in args.data]
    for fi, path in enumerate(files):
        data = Data([path])
        if data.dims != val.dims:
            raise SystemExit(f"{path}: layout {data.dims}, expected {val.dims}")
        b = Batcher(data, dev, model.seq)
        perm = torch.randperm(len(data), device=dev, generator=gen)
        tot = np.zeros(6)
        nb = 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (steps + 1) / max(1, args.warmup))
            loss, parts = losses(model, b, perm[i:i + args.batch], args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            tot += [p.item() for p in parts]
            nb += 1
        seen += len(data)
        del b
        line = dict(file=os.path.basename(path), records=seen, steps=steps, mins=(time.time() - t0) / 60,
                    train=dict(zip(["q", "v", "first", "cards", "teacher_v", "teacher_policy"], (tot / max(nb, 1)).tolist())))
        msg = (f"{fi + 1}/{len(files)} {line['file']}: {seen} records, {steps} steps, {line['mins']:.1f} min; "
               f"train loss Q {line['train']['q']:.3f} V {line['train']['v']:.3f} first {line['train']['first']:.3f} cards {line['train']['cards']:.3f} "
               f"teacher V {line['train']['teacher_v']:.3f} policy {line['train']['teacher_policy']:.3f}")
        if (fi + 1) % args.eval_every == 0 or fi + 1 == len(files):
            line["val"] = evaluate(model, val, dev, args)
            msg += "\n" + show(line["val"])
        print(msg, flush=True)
        log.append(line)
        torch.save(dict(arch=model.arch, dims=val.dims, model=model.state_dict(), args=vars(args)), args.out)
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(dict(parameters=n_par, arch=model.arch, args=vars(args), log=log), f, indent=1)


if __name__ == "__main__":
    main()
