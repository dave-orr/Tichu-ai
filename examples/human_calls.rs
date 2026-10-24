//! Scores real games (from `npm run export-rounds` in the tichu repo) with
//! the bots' call rules: for each seat, the v2 Tichu margin of its hand after
//! the pass and the Grand rule its first 8 cards meet, against what the human
//! called and whether they went out first.
//!
//!     cargo run --release --example human_calls -- rounds-export.jsonl

use std::collections::BTreeMap;

use serde::Deserialize;

use tichu_engine::bots::calls::{grand_rule, tichu_eval};
use tichu_engine::engine::GameState;
use tichu_engine::types::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    seat: Seat,
    tichu_call: TichuCall,
    out_order: u8,
    initial_hand: Vec<Card>,
    #[serde(default)]
    first_eight: Vec<Card>,
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

/// (calls, made, no-call hands, no-call first out)
type Row = (usize, usize, usize, usize);

fn main() {
    let path = std::env::args().nth(1).expect("usage: human_calls ROUNDS.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut small: BTreeMap<i32, Row> = BTreeMap::new();
    let mut grand: BTreeMap<u8, Row> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let r: Round = serde_json::from_str(line).unwrap();
        let mut ps: Vec<&Entry> = r.players.iter().collect();
        ps.sort_by_key(|p| p.seat);
        let passes: Vec<Option<PassInfo>> = ps
            .iter()
            .map(|p| Some(PassInfo { left: p.passed_left?, partner: p.passed_partner?, right: p.passed_right? }))
            .collect();
        if ps.len() != 4 || passes.iter().any(|p| p.is_none()) || ps.iter().any(|p| p.initial_hand.len() != 14) {
            continue;
        }
        let passes: Vec<PassInfo> = passes.into_iter().map(|p| p.unwrap()).collect();
        // Hands after the pass.
        let mut hands: [Vec<Card>; 4] = [0, 1, 2, 3].map(|s| {
            ps[s].initial_hand.iter().copied().filter(|c| !passes[s].cards().contains(c)).collect()
        });
        for (from, pass) in passes.iter().enumerate() {
            for c in pass.cards() {
                hands[pass.recipient(from, c).unwrap()].push(c);
            }
        }
        let mut g = GameState::mid_play(Settings::default(), hands, 0);
        g.scores = r.scores_before_round;
        for s in 0..4 {
            g.players[s].has_played_first_card = false;
            g.players[s].pass = Some(passes[s]);
        }
        for s in 0..4 {
            let p = ps[s];
            let first = (p.out_order == 1) as usize;
            let margin = tichu_eval(&g.view(s)).margin().round() as i32;
            let row = small.entry(margin).or_default();
            match p.tichu_call {
                TichuCall::Small => {
                    row.0 += 1;
                    row.1 += first;
                }
                TichuCall::None => {
                    row.2 += 1;
                    row.3 += first;
                }
                TichuCall::Grand => {}
            }
            if p.first_eight.len() == 8 {
                let mut g8 = GameState::mid_play(Settings::default(), Default::default(), 0);
                g8.phase = Phase::GrandTichuWindow;
                g8.scores = r.scores_before_round;
                g8.players[s].hand = p.first_eight.clone();
                let row = grand.entry(grand_rule(&g8.view(s))).or_default();
                if p.tichu_call == TichuCall::Grand {
                    row.0 += 1;
                    row.1 += first;
                } else {
                    row.2 += 1;
                    row.3 += first;
                }
            }
        }
    }
    let pct = |n: usize, d: usize| if d == 0 { "    -".to_string() } else { format!("{:>4.0}%", 100.0 * n as f64 / d as f64) };
    println!("Tichu margin of the hand after the pass (Grand callers excluded)");
    println!("  margin | humans called  made | didn't call  went out 1st");
    for (m, (c, made, n, f)) in small.iter().rev() {
        println!("  {m:>+5}  | {c:>8}  {}  | {n:>8}  {}", pct(*made, *c), pct(*f, *n));
    }
    let names = ["no rule", "3 tops incl. special", "A + special + extra", "bomb + special/2A", "A + special only"];
    println!("\nGrand rule met by the first 8 cards (rounds with first-8 logs)");
    println!("  rule                   | humans called  made | didn't call  went out 1st");
    for (r, (c, made, n, f)) in &grand {
        println!("  {:<22} | {c:>8}  {}  | {n:>8}  {}", names[*r as usize], pct(*made, *c), pct(*f, *n));
    }
}
