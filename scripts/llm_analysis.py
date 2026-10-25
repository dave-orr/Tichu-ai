#!/usr/bin/env python3
"""Analysis of an llm_match run directory, written as Markdown.

    python3 scripts/llm_analysis.py runs/DIR > runs/DIR/ANALYSIS.md
    python3 scripts/llm_analysis.py runs/NEW --compare runs/OLD   # same seed: paired by deal

Reads results.jsonl, games/deal-*.json and decisions.jsonl. Three "sides" are
reported where it makes sense: the LLM team, the bot it played against (in
the same games), and the fallback bot in the LLM's seats (the baseline games
of the same deals).
"""
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

RANKS = {11: "J", 12: "Q", 13: "K", 14: "A"}


def card_name(c):
    if c["type"] == "special":
        return {"mahjong": "MJ", "dog": "Dog", "phoenix": "Ph", "dragon": "Dr"}[c["name"]]
    return RANKS.get(c["rank"], str(c["rank"]))


def card_points(c):
    if c["type"] == "special":
        return {"dragon": 25, "phoenix": -25}.get(c["name"], 0)
    return {5: 5, 10: 10, 13: 10}.get(c["rank"], 0)


def pass_class(c):
    if c["type"] == "special":
        return card_name(c)
    r = c["rank"]
    if r == 14:
        return "A"
    if r == 13:
        return "K"
    if r >= 11:
        return "J/Q"
    if r >= 7:
        return "7-10"
    return "2-6"


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def pct(a, b):
    return "-" if b == 0 else f"{100 * a / b:.0f}%"


def fmt(m, ci):
    return f"{m:+.1f} ± {ci:.1f}"


class Side:
    """Tallies for one side (a team in a set of games)."""

    def __init__(self):
        self.c = Counter()
        self.card_pts = []          # card points in rounds without a 1-2
        self.margins = []
        self.pass_partner = Counter()
        self.pass_opp = Counter()
        self.wishes = Counter()
        self.dragon_gift = Counter()

    def add_game(self, g, team):
        r = g["result"]
        opp = 1 - team
        self.c["rounds"] += 1
        self.margins.append(r["card_points"][team] + r["tichu_bonuses"][team]
                            - r["card_points"][opp] - r["tichu_bonuses"][opp])
        first = r["out_order"][0]
        self.c["first_out"] += first % 2 == team
        if r["double_victory"] == team:
            self.c["one_two"] += 1
        elif r["double_victory"] == opp:
            self.c["one_two_against"] += 1
        else:
            self.card_pts.append(r["card_points"][team])
        for seat in (team, team + 2):
            call = r["tichu_calls"][seat]
            if call == "none":
                continue
            made = first == seat
            self.c[f"{call}_calls"] += 1
            self.c[f"{call}_made"] += made
            if not made and first == (seat + 2) % 4:
                self.c["call_broken_by_partner"] += 1
        for seat in (opp, opp + 2):
            call = r["tichu_calls"][seat]
            if call != "none":
                self.c["opp_calls"] += 1
                self.c["opp_made"] += first == seat

        # Walk the actions: passes, bombs, Dog, wish, Dragon gifts, overtaking partner.
        counts = [14] * 4
        last = None              # last player on the current trick
        for ev in g["events"]:
            if ev.get("type") != "action":
                continue
            a = ev["action"]
            t, seat = a["type"], a.get("seat")
            mine = seat is not None and seat % 2 == team
            if t == "passCards" and mine:
                p = a["pass"]
                self.pass_partner[pass_class(p["partner"])] += 1
                for side in ("left", "right"):
                    self.pass_opp[pass_class(p[side])] += 1
                    self.c["pass_opp_points"] += card_points(p[side])
            elif t in ("play", "bomb"):
                cards = a["cards"]
                if mine:
                    if t == "bomb" or (len(cards) >= 4 and is_bomb(cards)):
                        self.c["bombs"] += 1
                    if len(cards) == 1 and card_name(cards[0]) == "Dog":
                        self.c["dog_leads"] += 1
                    if last is not None and last != seat and last % 2 == team and counts[last] > 0:
                        self.c["overtook_partner"] += 1
                    self.c["plays"] += 1
                counts[seat] -= len(cards)
                if len(cards) == 1 and card_name(cards[0]) == "Dog":
                    last = None
                else:
                    last = seat
            elif t == "passTurn" and mine:
                self.c["passes"] += 1
            elif t == "wish" and mine:
                self.wishes["none" if a["rank"] is None else RANKS.get(a["rank"], str(a["rank"]))] += 1
            elif t == "giveDragon" and mine:
                to, other = a["to"], (a["to"] + 2) % 4
                if counts[to] == 0:
                    self.dragon_gift["to an opponent already out"] += 1
                elif counts[other] == 0:
                    self.dragon_gift["to the only opponent still in"] += 1
                elif counts[to] > counts[other]:
                    self.dragon_gift["to the opponent with more cards"] += 1
                elif counts[to] < counts[other]:
                    self.dragon_gift["to the opponent with fewer cards"] += 1
                else:
                    self.dragon_gift["equal cards"] += 1
                last = None
            elif t == "resolveTrick":
                last = None


def is_bomb(cards):
    if any(c["type"] == "special" for c in cards):
        return False
    ranks = sorted(c["rank"] for c in cards)
    if len(cards) == 4 and len(set(ranks)) == 1:
        return True
    suits = {c["suit"] for c in cards}
    return len(cards) >= 5 and len(suits) == 1 and all(b - a == 1 for a, b in zip(ranks, ranks[1:]))


def load(run):
    results = [json.loads(l) for l in open(os.path.join(run, "results.jsonl"))]
    results.sort(key=lambda r: r["deal"])
    games = {}
    for p in glob.glob(os.path.join(run, "games", "deal-*.json")):
        g = json.load(open(p))
        games[g["deal"]] = g
    decisions = []
    path = os.path.join(run, "decisions.jsonl")
    if os.path.exists(path):
        decisions = [json.loads(l) for l in open(path)]
    cfg = json.load(open(os.path.join(run, "config.json")))
    return cfg, results, games, decisions


def sides(results, games):
    llm, opp, base = Side(), Side(), Side()
    for r in results:
        g = games.get(r["deal"])
        if not g:
            continue
        for game, s in zip(g["llm"], r["seatings"]):
            llm.add_game(game, s["team"])
            opp.add_game(game, 1 - s["team"])
        for game, s in zip(g["baseline"], r["baseline"]):
            base.add_game(game, s["team"])
    return llm, opp, base


def report(run, compare=None):
    cfg, results, games, decisions = load(run)
    n = len(results)
    out = []
    w = out.append
    llm_name = f"Claude ({cfg['backend'].split(':')[1]})"
    w(f"# {llm_name} vs {cfg['opponent']}: {n} duplicate deals (seed {cfg['seed']})\n")
    d = cfg["decisions"]
    asked = [k for k in ("play", "bomb", "pass", "wish", "dragon", "calls") if d.get(k)]
    w(f"Claude decides: {', '.join(asked)}. Fallback / baseline bot: `{cfg['fallback']}`"
      f"{'' if d.get('calls') else ' (it also makes the Claude seats’ Tichu calls)'}.\n")

    m = [r["margin"] for r in results]
    b = [r["baseline_margin"] for r in results if r.get("baseline_margin") is not None]
    paired = [r["margin"] - r["baseline_margin"] for r in results if r.get("baseline_margin") is not None]
    w("## Headline\n")
    w("| | per deal (95% CI) |\n|---|---|")
    w(f"| Claude − {cfg['opponent']} | **{fmt(*mean_ci(m))}** |")
    if b:
        w(f"| {cfg['fallback']} − {cfg['opponent']} (same deals) | {fmt(*mean_ci(b))} |")
        w(f"| Claude − {cfg['fallback']} (paired) | {fmt(*mean_ci(paired))} |")
    w("")
    seat_m = [s["margin"] for r in results for s in r["seatings"]]
    wins = sum(x > 0 for x in m)
    w(f"Deals won / tied / lost (mean of both seatings): {wins} / {sum(x == 0 for x in m)} / {sum(x < 0 for x in m)}. "
      f"Single games: {sum(x > 0 for x in seat_m)} won, {sum(x < 0 for x in seat_m)} lost of {len(seat_m)}.\n")
    half = n // 2
    if half >= 10:
        w(f"First half / second half of deals: {fmt(*mean_ci(m[:half]))} / {fmt(*mean_ci(m[half:]))} (a stability check).\n")

    llm, opp, base = sides(results, games)
    cols = [(llm_name, llm), (f"{cfg['opponent']} (vs Claude)", opp), (f"{cfg['fallback']} (baseline, Claude's seats)", base)]
    w("## How the points were won\n")
    w("| | " + " | ".join(c for c, _ in cols) + " |")
    w("|---|" + "---|" * len(cols))

    def row(label, f):
        w(f"| {label} | " + " | ".join(f(s) for _, s in cols) + " |")

    row("games", lambda s: str(s.c["rounds"]))
    row("margin per game", lambda s: fmt(*mean_ci(s.margins)))
    row("went out first", lambda s: pct(s.c["first_out"], s.c["rounds"]))
    row("1-2 for", lambda s: pct(s.c["one_two"], s.c["rounds"]))
    row("1-2 against", lambda s: pct(s.c["one_two_against"], s.c["rounds"]))
    row("card points (games without a 1-2)", lambda s: f"{mean_ci(s.card_pts)[0]:.1f}")
    w("")

    w("## Calls\n")
    w("Made = the caller went out first. *Broken by partner* = the caller's partner went out first instead.\n")
    w("| | " + " | ".join(c for c, _ in cols) + " |")
    w("|---|" + "---|" * len(cols))
    row("Tichu made / called", lambda s: f"{s.c['small_made']}/{s.c['small_calls']} ({pct(s.c['small_made'], s.c['small_calls'])})")
    row("Tichu calls per game", lambda s: f"{s.c['small_calls'] / max(s.c['rounds'], 1):.2f}")
    row("Grand made / called", lambda s: f"{s.c['grand_made']}/{s.c['grand_calls']} ({pct(s.c['grand_made'], s.c['grand_calls'])})")
    row("calls broken by partner", lambda s: str(s.c["call_broken_by_partner"]))
    row("opponents' calls made (defense)", lambda s: f"{s.c['opp_made']}/{s.c['opp_calls']} ({pct(s.c['opp_made'], s.c['opp_calls'])})")
    w("")

    w("## Play habits\n")
    w("| | " + " | ".join(c for c, _ in cols) + " |")
    w("|---|" + "---|" * len(cols))
    row("bombs per game", lambda s: f"{s.c['bombs'] / max(s.c['rounds'], 1):.2f}")
    row("Dog leads per game", lambda s: f"{s.c['dog_leads'] / max(s.c['rounds'], 1):.2f}")
    row("plays that overtook partner's winning play", lambda s: f"{s.c['overtook_partner']} ({pct(s.c['overtook_partner'], s.c['plays'])} of plays)")
    row("points passed to opponents per game", lambda s: f"{s.c['pass_opp_points'] / max(s.c['rounds'], 1):.1f}")
    w("")

    def dist(counter):
        tot = sum(counter.values())
        return ", ".join(f"{k} {pct(v, tot)}" for k, v in counter.most_common()) or "-"

    w("**Card passed to partner:**\n")
    for c, s in cols:
        w(f"- {c}: {dist(s.pass_partner)}")
    w("\n**Cards passed to opponents:**\n")
    for c, s in cols:
        w(f"- {c}: {dist(s.pass_opp)}")
    w("\n**Mah Jong wish:**\n")
    for c, s in cols:
        w(f"- {c}: {dist(s.wishes)}")
    w("\n**Dragon trick given:**\n")
    for c, s in cols:
        w(f"- {c}: {dist(s.dragon_gift)}")
    w("")

    if decisions:
        w("## Claude's questions\n")
        kinds = Counter(x["kind"] for x in decisions)
        w(f"{len(decisions)} questions ({len(decisions) / max(n, 1):.0f} per deal): "
          + ", ".join(f"{k} {v}" for k, v in kinds.most_common()) + ".")
        lens = [len(x["reply"]) for x in decisions]
        w(f"Median reply {sorted(lens)[len(lens) // 2]} characters. "
          f"Unusable answers: {sum(1 for x in decisions if x.get('answer') is None)}. "
          f"Time per deal: {sum(r['seconds'] for r in results) / max(n, 1) / 60:.1f} min (3 deals in parallel).\n")

    w("## Deals worth replaying\n")
    w(f"`cargo run --release --example llm_match -- --dir {run} --replay N` (add the same --opponent/--fallback/--decisions as the run).\n")
    if b:
        by = sorted(results, key=lambda r: r["margin"] - r["baseline_margin"])
        w("Claude vs the baseline on the same cards:\n")
        w("- Best for Claude: " + ", ".join(f"deal {r['deal']} ({r['margin'] - r['baseline_margin']:+.0f})" for r in by[::-1][:5]))
        w("- Worst for Claude: " + ", ".join(f"deal {r['deal']} ({r['margin'] - r['baseline_margin']:+.0f})" for r in by[:5]))
    by = sorted(results, key=lambda r: r["margin"])
    w(f"- Biggest losses to {cfg['opponent']}: " + ", ".join(f"deal {r['deal']} ({r['margin']:+.0f})" for r in by[:5]))
    w(f"- Biggest wins over {cfg['opponent']}: " + ", ".join(f"deal {r['deal']} ({r['margin']:+.0f})" for r in by[::-1][:5]))
    w("")

    if compare:
        _, old, _, _ = load(compare)
        old = {r["deal"]: r for r in old}
        diffs = [r["margin"] - old[r["deal"]]["margin"] for r in results if r["deal"] in old]
        w(f"## Compared with `{compare}` (same deals)\n")
        w(f"This run − that run, paired by deal: **{fmt(*mean_ci(diffs))}** over {len(diffs)} deals.\n")

    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--compare")
    a = ap.parse_args()
    print(report(a.run, a.compare))
