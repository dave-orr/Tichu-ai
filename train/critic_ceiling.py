"""Numbers from `examples/critic_ceiling.rs` (one JSON line per decision).

    python train/critic_ceiling.py runs/ceiling/d1.jsonl

Every margin vector is centred over the decision's options, so everything
is about how the options differ. "R²" of a predictor x for the real-deal
verdict r is the share of r's variance a scaled x explains, pooled over
decisions: (Σ x·r)² / (Σ x² · Σ r²).

- reliability of a guess distribution: the share of one guessed deal's
  verdict that another guessed deal reproduces (Σ u_i·u_j / Σ u²). It is the
  best R² any predictor could reach *if hidden cards really were drawn that
  way*.
- mean of M guesses vs the real deal: what a search over M guessed deals
  knows about the real verdict; M = ∞ is extrapolated from the same sums.
- net + guesses: the two together (least squares on both).
- real hands, rebuilt state: the real deal put through
  `GameState::from_view_with_history`, as every guessed deal is. The other
  three players then no longer know what they passed and received, so they
  can play differently; this caps what any guessed deal can say about the
  real verdict.
± are 95% bootstrap intervals over decisions.
"""

import json
import sys

import numpy as np


def centre(x):
    return x - x.mean(-1, keepdims=True)


def main():
    path = sys.argv[1]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    M = len(rows[0]["uniform"])
    same = np.mean([r["real"] == r["rebuilt"] for r in rows])
    # Per-decision sufficient statistics (sums over options).
    names = ["rr", "aa", "ar", "bb", "br"]
    stat = {k: np.zeros(n) for k in names}
    per = {}
    for src in ("uniform", "weighted"):
        per[src] = dict(uu=np.zeros(n), uv=np.zeros(n), ur=np.zeros((n, M)), ua=np.zeros((n, M)),
                        mm=np.zeros((n, M)), mr=np.zeros((n, M)), ma=np.zeros((n, M)))
    for i, row in enumerate(rows):
        r = centre(np.array(row["real"]))
        a = centre(np.array(row["logits"]) * 100.0)
        stat["rr"][i], stat["aa"][i], stat["ar"][i] = r @ r, a @ a, a @ r
        b = centre(np.array(row["rebuilt"]))
        stat["bb"][i], stat["br"][i] = b @ b, b @ r
        for src in ("uniform", "weighted"):
            u = centre(np.array(row[src]))  # [M, K]
            p = per[src]
            g = u @ u.T
            p["uu"][i] = np.trace(g) / M
            p["uv"][i] = (g.sum() - np.trace(g)) / (M * (M - 1))
            p["ur"][i], p["ua"][i] = u @ r, u @ a
            cm = np.cumsum(u, 0) / np.arange(1, M + 1)[:, None]  # mean of the first m guesses
            p["mm"][i], p["mr"][i], p["ma"][i] = (cm * cm).sum(1), cm @ r, cm @ a

    def numbers(idx):
        S = lambda x: x[idx].sum(0)  # noqa: E731
        rr, aa, ar = S(stat["rr"]), S(stat["aa"]), S(stat["ar"])
        out = {"net vs real": ar ** 2 / (aa * rr),
               "real hands, rebuilt state, vs real": S(stat["br"]) ** 2 / (S(stat["bb"]) * rr)}
        for src in ("uniform", "weighted"):
            p = per[src]
            uu, uv, ur, ua = S(p["uu"]), S(p["uv"]), S(p["ur"]).mean(), S(p["ua"]).mean()
            out[f"{src}: reliability"] = uv / uu
            out[f"{src}: net's share of its signal"] = ua ** 2 / (aa * uv)
            out[f"{src}: M=inf vs real"] = ur ** 2 / (uv * rr)
            mm, mr, ma = S(p["mm"]), S(p["mr"]), S(p["ma"])
            for m in (1, 2, 4, M):
                j = m - 1
                out[f"{src}: M={m} vs real"] = mr[j] ** 2 / (mm[j] * rr)
                # Least squares of r on (a, mean of m guesses).
                G = np.array([[aa, ma[j]], [ma[j], mm[j]]])
                b = np.array([ar, mr[j]])
                out[f"{src}: net + M={m} vs real"] = b @ np.linalg.solve(G, b) / rr
            # M = ∞: the guess mean becomes its signal (variance uv, same covariances).
            G = np.array([[aa, ua], [ua, uv]])
            b = np.array([ar, ur])
            out[f"{src}: net + M=inf vs real"] = b @ np.linalg.solve(G, b) / rr
        return out

    full = numbers(np.arange(n))
    rng = np.random.default_rng(0)
    boots = [numbers(rng.integers(0, n, n)) for _ in range(300)]
    print(f"{n} decisions, {M} guesses each; rebuilt real state plays out identically in {same:.1%}")
    print(f"sd of a real-deal verdict difference: {np.sqrt(stat['rr'].sum() / sum(len(r['real']) for r in rows)):.0f} pts")
    for k, v in full.items():
        lo, hi = np.percentile([b[k] for b in boots], [2.5, 97.5])
        print(f"  {k:<40} {v:.3f}  ({lo:.3f} to {hi:.3f})")


if __name__ == "__main__":
    main()
