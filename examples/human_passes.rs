//! Compares the bots' passing with what humans passed in real games (from
//! `npm run export-rounds` in the tichu repo): how often each rule picks the
//! human's partner card and opponent cards (by rank, suits ignored).
//!
//!     cargo run --release --example human_passes -- rounds-export.jsonl [--show N]

use serde::Deserialize;

use tichu_engine::bot::Bot;
use tichu_engine::bots::passing::pass_v2;
use tichu_engine::bots::HeuristicBot;
use tichu_engine::engine::GameState;
use tichu_engine::types::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    seat: Seat,
    tichu_call: TichuCall,
    initial_hand: Vec<Card>,
    passed_left: Option<Card>,
    passed_partner: Option<Card>,
    passed_right: Option<Card>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Round {
    players: Vec<Entry>,
    scores_before_round: [i32; 2],
}

fn same(a: Card, b: Card) -> bool {
    a.rank().map(|r| Some(r) == b.rank()).unwrap_or(a == b)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(args.get(1).expect("usage: human_passes ROUNDS.jsonl [--show N]")).unwrap();
    let show: usize = args.iter().position(|a| a == "--show").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(0);
    let mut shown = 0;
    let (mut n, mut v1, mut v2) = (0usize, [0usize; 3], [0usize; 3]);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let r: Round = serde_json::from_str(line).unwrap();
        let mut ps: Vec<&Entry> = r.players.iter().collect();
        ps.sort_by_key(|p| p.seat);
        if ps.len() != 4 || ps.iter().any(|p| p.initial_hand.len() != 14) {
            continue;
        }
        let mut g = GameState::mid_play(Settings::default(), [0, 1, 2, 3].map(|s| ps[s].initial_hand.clone()), 0);
        g.phase = Phase::Passing;
        g.scores = r.scores_before_round;
        for s in 0..4 {
            // Only Grand calls are made before the pass.
            g.players[s].tichu = if ps[s].tichu_call == TichuCall::Grand { TichuCall::Grand } else { TichuCall::None };
        }
        for s in 0..4 {
            let p = ps[s];
            let (Some(l), Some(pa), Some(rt)) = (p.passed_left, p.passed_partner, p.passed_right) else { continue };
            n += 1;
            let view = g.view(s);
            let a = HeuristicBot::new().pass_cards(&view);
            let b = pass_v2(&view);
            for (pass, tally) in [(a, &mut v1), (b, &mut v2)] {
                tally[0] += same(pass.partner, pa) as usize;
                let opp = (same(pass.left, l) && same(pass.right, rt)) || (same(pass.left, rt) && same(pass.right, l));
                tally[1] += opp as usize;
                tally[2] += (same(pass.left, l) && same(pass.right, rt)) as usize;
            }
            if shown < show {
                shown += 1;
                let mut h = p.initial_hand.clone();
                h.sort_by_key(|c| c.index());
                let hs: Vec<String> = h.iter().map(|c| c.to_string()).collect();
                println!("{}  [{:?}]", hs.join(" "), p.tichu_call);
                println!("  human  L {}  P {}  R {}", l, pa, rt);
                println!("  v1     L {}  P {}  R {}", a.left, a.partner, a.right);
                println!("  v2     L {}  P {}  R {}", b.left, b.partner, b.right);
            }
        }
    }
    let pct = |k: usize| format!("{:>3.0}%", 100.0 * k as f64 / n as f64);
    println!("{n} seats; matches the human's pass (by rank):");
    println!("        partner  both opps  both opps, same sides");
    println!("  v1     {}      {}       {}", pct(v1[0]), pct(v1[1]), pct(v1[2]));
    println!("  v2     {}      {}       {}", pct(v2[0]), pct(v2[1]), pct(v2[2]));
}
