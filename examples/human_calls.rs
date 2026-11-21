//! Human Tichu / Grand Tichu calls next to a learned call model's and the v2
//! rules', from an `npm run export-rounds` file.
//!
//! For every round of a game that reached `--min-final` (default 1000), it
//! rebuilds the deal and asks, for each seat:
//! - Grand (only when the log has `firstEight`): from the 8-card view,
//!   before anyone's Grand is known;
//! - Small: at the start of play (after the pass, with the Grand calls
//!   known), for seats that didn't call Grand.
//!
//! Each decision also gets a paired rollout from the real deal (call vs no
//! call, `--bot` in every seat, the deciding seat never calling Small in
//! the no-call branch): what the call was worth in that deal with bot play.
//! One deal is a noisy judge; read it in aggregate.
//!
//!     cargo run --release --example human_calls -- ROUNDS.jsonl --model M.tcall
//!         [--grand-model G.tcall] [--bot net:runs/v4/start.tnet] [--min-final 1000] [--out FILE.jsonl]

use std::collections::HashMap;

use serde_json::json;

use tichu_engine::bot::Bot;
use tichu_engine::bots::{self, calls, callnet::CallNet};
use tichu_engine::engine::{Action, GameState, Pending};
use tichu_engine::humanlog::{read_rounds, LogPlayer, RoundLog};
use tichu_engine::sim::play_from;
use tichu_engine::types::*;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn deal(r: &RoundLog) -> Option<(GameState, Vec<&LogPlayer>)> {
    let mut ps: Vec<&LogPlayer> = r.players.iter().collect();
    ps.sort_by_key(|p| p.seat);
    if ps.len() != 4 || ps.iter().enumerate().any(|(i, p)| p.seat != i || p.initial_hand.len() != 14) {
        return None;
    }
    let mut deck = vec![];
    for p in &ps {
        match &p.first_eight {
            Some(e) if e.len() == 8 && e.iter().all(|c| p.initial_hand.contains(c)) => {
                deck.extend_from_slice(e);
                deck.extend(p.initial_hand.iter().filter(|c| !e.contains(c)));
            }
            _ => deck.extend_from_slice(&p.initial_hand),
        }
    }
    let mut g = GameState::new(Settings { clockwise: r.clockwise.unwrap_or(false), ..Settings::default() });
    g.start_round(deck);
    g.scores = r.scores_before_round;
    Some((g, ps))
}

/// Round totals (seat's team, the other) after `first`, with `bot`
/// everywhere and `seat` never calling Small.
fn rollout(state: &GameState, first: Option<Action>, seat: Seat, bot: &str) -> Option<[i32; 2]> {
    let mut s = state.clone();
    if let Some(a) = first {
        s.apply(&a).ok()?;
    }
    let mut bs: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| {
        let name = if k == seat { format!("nocalls:{bot}") } else { bot.to_string() };
        bots::by_name(&name, 7 + k as u64).expect("bot")
    });
    let r = play_from(&mut s, &mut bs, None, true).ok()?;
    let t = seat % 2;
    Some([r.total(t), r.total(1 - t)])
}

fn hand_str(h: &[Card]) -> String {
    let mut h = h.to_vec();
    h.sort_by_key(|c| std::cmp::Reverse(c.index()));
    h.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("ROUNDS.jsonl");
    let text = std::fs::read_to_string(path).expect("read rounds");
    let model = CallNet::load(&arg(&args, "--model").expect("--model")).expect("model");
    let grand_model = arg(&args, "--grand-model").map(|p| CallNet::load(&p).expect("grand model"));
    let gm = grand_model.as_ref().unwrap_or(&model);
    let bot = arg(&args, "--bot").unwrap_or("net:runs/v4/start.tnet".into());
    let min_final: i32 = arg(&args, "--min-final").map_or(1000, |v| v.parse().unwrap());

    // Final score per game, from the raw lines (scoresAfterRound isn't in RoundLog).
    let mut final_score: HashMap<i64, i32> = HashMap::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = match serde_json::from_str(l) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let g = v["gameId"].as_i64().unwrap_or(-1);
        let s = v["scoresAfterRound"].as_array().map_or(0, |a| a.iter().filter_map(|x| x.as_i64()).max().unwrap_or(0)) as i32;
        let e = final_score.entry(g).or_insert(i32::MIN);
        *e = (*e).max(s);
    }
    let rounds = read_rounds(&text);
    let kept: Vec<&RoundLog> = rounds.iter().filter(|r| final_score.get(&r.game_id).is_some_and(|&s| s >= min_final)).collect();
    let games = kept.iter().map(|r| r.game_id).collect::<std::collections::HashSet<_>>().len();
    eprintln!("{} rounds in {} games reach {min_final} (of {} rounds)", kept.len(), games, rounds.len());

    let mut out: Vec<serde_json::Value> = vec![];
    let mut skipped = 0;
    for r in &kept {
        let Some((mut g, ps)) = deal(r) else {
            skipped += 1;
            continue;
        };
        let has_eight = ps.iter().all(|p| p.first_eight.as_ref().is_some_and(|e| e.len() == 8));
        if has_eight {
            for s in 0..4 {
                let v = g.view(s);
                let gain = gm.gain(&v, true);
                let yes = rollout(&g, Some(Action::GrandTichu { seat: s, call: true }), s, &bot);
                let no = rollout(&g, Some(Action::GrandTichu { seat: s, call: false }), s, &bot);
                out.push(json!({
                    "id": r.id(), "seat": s, "kind": "grand",
                    "score": [v.scores[s % 2], v.scores[1 - s % 2]],
                    "hand": hand_str(&v.hand),
                    "human": ps[s].tichu_call == TichuCall::Grand,
                    "model_gain": gain, "model": gain > 0.0,
                    "v2": calls::grand_tichu_v2(&v),
                    "rollout_gain": yes.zip(no).map(|(y, n)| (y[0] - y[1]) - (n[0] - n[1])),
                    "with": yes, "without": no,
                    "first_out": ps[s].out_order == 1,
                }));
            }
        }
        let mut ok = true;
        for s in 0..4 {
            ok &= g.apply(&Action::GrandTichu { seat: s, call: ps[s].tichu_call == TichuCall::Grand }).is_ok();
        }
        for s in 0..4 {
            let p = ps[s];
            match (p.passed_left, p.passed_partner, p.passed_right) {
                (Some(left), Some(partner), Some(right)) => {
                    ok &= g.apply(&Action::PassCards { seat: s, pass: PassInfo { left, partner, right } }).is_ok()
                }
                _ => ok = false,
            }
        }
        if !ok || !matches!(g.pending(), Pending::Turn(_)) {
            skipped += 1;
            continue;
        }
        for s in 0..4 {
            if !g.can_call_small_tichu(s) {
                continue;
            }
            let v = g.view(s);
            let gain = model.gain(&v, false);
            let yes = rollout(&g, Some(Action::SmallTichu { seat: s }), s, &bot);
            let no = rollout(&g, None, s, &bot);
            let from_partner = v.received.iter().find(|(_, f)| *f == v.partner()).map(|(c, _)| c.to_string());
            let eval = calls::tichu_eval(&v);
            out.push(json!({
                "id": r.id(), "seat": s, "kind": "small",
                "score": [v.scores[s % 2], v.scores[1 - s % 2]],
                "hand": hand_str(&v.hand),
                "from_partner": from_partner,
                "partner_call": format!("{:?}", v.players[v.partner()].tichu),
                "opp_calls": v.opponents().iter().map(|&o| format!("{:?}", v.players[o].tichu)).collect::<Vec<_>>(),
                "human": ps[s].tichu_call == TichuCall::Small,
                "model_gain": gain, "model": gain > 0.0,
                "v2_margin": eval.margin() - calls::call_adjustment(&v),
                "v2": eval.margin() >= calls::DEFAULT_TICHU_THRESHOLD + calls::call_adjustment(&v)
                    && v.players[v.partner()].tichu == TichuCall::None,
                "rollout_gain": yes.zip(no).map(|(y, n)| (y[0] - y[1]) - (n[0] - n[1])),
                    "with": yes, "without": no,
                "first_out": ps[s].out_order == 1,
            }));
        }
    }
    eprintln!("{} decisions ({skipped} rounds skipped)", out.len());
    let dest = arg(&args, "--out").unwrap_or("human_calls.jsonl".into());
    let body: String = out.iter().map(|v| v.to_string() + "\n").collect();
    std::fs::write(&dest, body).expect("write");
    eprintln!("wrote {dest}");
}
