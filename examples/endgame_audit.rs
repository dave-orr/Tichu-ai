//! Audits play in two-player endgames, where the exact solver knows the
//! value of every option. Positions come from real games (an
//! `npm run export-rounds` file, rounds with a play log) and/or bot
//! self-play. At each on-turn decision every option is solved, then the
//! player who was actually there and each audited bot are scored by regret:
//! the points (round margin, their team's side) lost against best play.
//!
//!     cargo run --release --example endgame_audit -- [--human ROUNDS.jsonl]
//!         [--selfplay BOT]... [--rounds N] [--bots a,b,c] [--out DIR]
//!         [--seed S] [--threads T]
//!
//! `--out DIR` writes `positions.jsonl` (every position, all option values,
//! every choice) and `mistakes.txt` (the costliest decisions, readable).

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;

use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::endgame::{self, Decision, Node, OptionValue};
use tichu_engine::engine::{Action, GameState, Pending};
use tichu_engine::humanlog;
use tichu_engine::sim::play_round;
use tichu_engine::types::*;

const BUDGET: usize = 5_000_000;

#[derive(Clone)]
struct Position {
    source: String,
    id: String,
    state: GameState,
    /// The action actually taken there.
    actual: Action,
}

struct Audited {
    pos: Position,
    seat: Seat,
    options: Vec<OptionValue>,
    /// Option index chosen by the actual player, then by each audited bot.
    actual: Option<usize>,
    bots: Vec<Option<usize>>,
}

impl Audited {
    fn value(&self, i: usize) -> i32 {
        let v = self.options[i].value;
        if team_for_seat(self.seat) == 0 { v } else { -v }
    }
    fn best(&self) -> i32 {
        (0..self.options.len()).map(|i| self.value(i)).max().unwrap()
    }
    fn worst(&self) -> i32 {
        (0..self.options.len()).map(|i| self.value(i)).min().unwrap()
    }
    fn regret(&self, choice: Option<usize>) -> Option<i32> {
        choice.map(|i| self.best() - self.value(i))
    }
    /// Does the choice matter at all here?
    fn nontrivial(&self) -> bool {
        self.best() != self.worst()
    }
}

fn combo_key(c: &Combo) -> (u64, i32, ComboKind) {
    (CardSet::from_cards(&c.cards).0, (c.rank * 2.0) as i32, c.kind)
}

/// Index of the option matching a play (None = pass).
fn match_play(options: &[OptionValue], play: Option<&Combo>) -> Option<usize> {
    options.iter().position(|o| match (&o.decision, play) {
        (Decision::Pass, None) => true,
        (Decision::Play(c), Some(p)) => combo_key(c) == combo_key(p),
        _ => false,
    })
}

/// Index of the option matching a logged action.
fn match_action(state: &GameState, options: &[OptionValue], action: &Action) -> Option<usize> {
    match action {
        Action::PassTurn { .. } => match_play(options, None),
        Action::Play { seat, cards, .. } => {
            // The logged play may leave the Phoenix's reading to the engine:
            // apply it and read the trick back.
            let mut g = state.clone();
            g.apply(action).ok()?;
            let rank = g.trick_plays.last().map(|t| t.combo.rank).or(g.current_trick.as_ref().map(|c| c.rank));
            let set = CardSet::from_cards(cards).0;
            let cands: Vec<usize> = options
                .iter()
                .enumerate()
                .filter(|(_, o)| matches!(&o.decision, Decision::Play(c) if CardSet::from_cards(&c.cards).0 == set))
                .map(|(i, _)| i)
                .collect();
            let _ = seat;
            if cands.len() <= 1 {
                return cands.first().copied();
            }
            cands
                .iter()
                .copied()
                .find(|&i| matches!(&options[i].decision, Decision::Play(c) if Some(c.rank) == rank))
                .or(cands.first().copied())
        }
        _ => None,
    }
}

fn label(d: &Decision) -> String {
    match d {
        Decision::Pass => "pass".into(),
        Decision::Play(c) | Decision::Bomb(c) => {
            let cards: Vec<String> = c.cards.iter().map(|x| x.short()).collect();
            format!("{} [{}]", cards.join(" "), format!("{:?}", c.kind))
        }
        other => format!("{other:?}"),
    }
}

fn hand_str(h: &[Card]) -> String {
    h.iter().map(|c| c.short()).collect::<Vec<_>>().join(" ")
}

fn audit(pos: Position, bot_names: &[String], seed: u64) -> Option<Audited> {
    let (seat, options) = endgame::solve_options(&pos.state, Node::Engine, BUDGET)?;
    if options.len() < 2 {
        return None;
    }
    let actual = match_action(&pos.state, &options, &pos.actual);
    let plays = pos.state.legal_plays(seat);
    let can_pass = pos.state.can_pass(seat);
    let view = pos.state.view(seat);
    let bots = bot_names
        .iter()
        .map(|name| {
            let mut b = bots::by_name(name, seed).expect("bot");
            let choice = b.play(&view, &plays, can_pass);
            match_play(&options, choice.map(|i| &plays[i]))
        })
        .collect();
    Some(Audited { pos, seat, options, actual, bots })
}

/// On-turn decisions in two-player endgames from `rounds` self-play rounds.
fn selfplay_positions(bot: &str, rounds: usize, seed: u64) -> Vec<Position> {
    let mut out = vec![];
    let mut rng = StdRng::seed_from_u64(seed);
    for r in 0..rounds {
        let mut seats: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| bots::by_name(bot, seed + (r * 4 + k) as u64).unwrap());
        let mut found: Vec<(GameState, Action)> = vec![];
        let mut obs = |s: &GameState, a: &Action| {
            if matches!(a, Action::Play { .. } | Action::PassTurn { .. })
                && matches!(s.pending(), Pending::Turn(_))
                && endgame::is_two_player_endgame(s)
            {
                found.push((s.clone(), a.clone()));
            }
        };
        let mut g = GameState::new(Settings::default());
        if play_round(&mut g, &mut seats, &mut rng, Some(&mut obs)).is_err() {
            continue;
        }
        for (k, (state, actual)) in found.into_iter().enumerate() {
            out.push(Position { source: format!("self:{bot}"), id: format!("r{r}.{k}"), state, actual });
        }
    }
    out
}

fn human_positions(path: &str) -> Vec<Position> {
    let rounds = humanlog::read_rounds(&std::fs::read_to_string(path).expect("read rounds"));
    let mut out = vec![];
    for r in rounds.iter().filter(|r| !r.plays.is_empty()) {
        let Ok(rp) = humanlog::replay(r) else {
            eprintln!("skipping {}: replay failed", r.id());
            continue;
        };
        for (k, st) in rp.steps.iter().enumerate() {
            if st.is_turn_decision() && endgame::is_two_player_endgame(&st.state) {
                out.push(Position { source: "human".into(), id: format!("{}.{k}", r.id()), state: st.state.clone(), actual: st.action.clone() });
            }
        }
    }
    out
}

fn describe(a: &Audited, who: &[String]) -> String {
    let st = &a.pos.state;
    let seat = a.seat;
    let opp = (0..4).find(|&s| s != seat && !st.players[s].is_out()).unwrap();
    let mut s = String::new();
    let calls: Vec<String> = (0..4)
        .filter(|&x| st.players[x].tichu != TichuCall::None)
        .map(|x| format!("seat {x} {:?}", st.players[x].tichu))
        .collect();
    let outs: Vec<String> = (0..4).filter(|&x| st.players[x].is_out()).map(|x| format!("{x}:{}", st.players[x].out_order)).collect();
    s += &format!("== {} {} | seat {seat} (team {}) to play vs seat {opp}\n", a.pos.source, a.pos.id, team_for_seat(seat));
    s += &format!("   mine: {}\n   opp:  {}\n", hand_str(&st.players[seat].hand), hand_str(&st.players[opp].hand));
    let table = match &st.current_trick {
        Some(c) => format!("{} by seat {}", hand_str(&c.cards), st.last_played_by.map(|x| x.to_string()).unwrap_or("?".into())),
        None => "empty (leading)".into(),
    };
    let tp: i32 = st.trick_plays.iter().map(|p| tichu_engine::scoring::total_points(&p.combo.cards)).sum();
    let cap: Vec<i32> = (0..4).map(|x| st.players[x].captured_points()).collect();
    s += &format!(
        "   table: {table} ({tp} pts) | captured {cap:?} | out {} | calls {} | wish {:?}\n",
        outs.join(","),
        if calls.is_empty() { "none".into() } else { calls.join(", ") },
        st.wish
    );
    let mut order: Vec<usize> = (0..a.options.len()).collect();
    order.sort_by_key(|&i| -a.value(i));
    for i in order {
        let mut tags = vec![];
        if a.actual == Some(i) {
            tags.push("ACTUAL".to_string());
        }
        for (k, b) in a.bots.iter().enumerate() {
            if *b == Some(i) {
                tags.push(who[k].clone());
            }
        }
        s += &format!("   {:>+5}  {:<40} {}\n", a.value(i), label(&a.options[i].decision), tags.join(" "));
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let all = |flag: &str| -> Vec<String> {
        args.iter().enumerate().filter(|(_, a)| *a == flag).filter_map(|(i, _)| args.get(i + 1).cloned()).collect()
    };
    let rounds: usize = get("--rounds").and_then(|s| s.parse().ok()).unwrap_or(300);
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let threads: usize = get("--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let bot_names: Vec<String> = get("--bots")
        .unwrap_or_else(|| "heuristic,h2,pimc10/h".into())
        .split(',')
        .map(|s| s.to_string())
        .collect();
    for b in &bot_names {
        bots::by_name(b, 0).unwrap_or_else(|| panic!("unknown bot {b}"));
    }

    let mut positions: Vec<Position> = vec![];
    if let Some(path) = get("--human") {
        positions.extend(human_positions(&path));
    }
    for (k, bot) in all("--selfplay").iter().enumerate() {
        // Split the rounds across threads.
        let per = rounds.div_ceil(threads);
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let bot = bot.clone();
                let n = per.min(rounds.saturating_sub(t * per));
                let s = seed * 1_000_003 + (k * 7919 + t * 104_729) as u64;
                std::thread::spawn(move || {
                    let mut v = selfplay_positions(&bot, n, s);
                    for p in &mut v {
                        p.id = format!("t{t}.{}", p.id);
                    }
                    v
                })
            })
            .collect();
        for h in handles {
            positions.extend(h.join().unwrap());
        }
    }
    eprintln!("{} endgame decisions to audit", positions.len());

    // Audit in parallel.
    let queue = Arc::new(Mutex::new(positions.into_iter().enumerate().collect::<Vec<_>>()));
    let results = Arc::new(Mutex::new(vec![]));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (queue, results, names) = (queue.clone(), results.clone(), bot_names.clone());
            std::thread::spawn(move || loop {
                let Some((i, pos)) = queue.lock().unwrap().pop() else { break };
                if let Some(a) = audit(pos, &names, seed + i as u64) {
                    results.lock().unwrap().push((i, a));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let mut results = Arc::try_unwrap(results).ok().unwrap().into_inner().unwrap();
    results.sort_by_key(|(i, _)| *i);
    let results: Vec<Audited> = results.into_iter().map(|(_, a)| a).collect();

    // Summary per source: the actual player, then each bot.
    let mut sources: BTreeMap<String, Vec<&Audited>> = BTreeMap::new();
    for a in &results {
        sources.entry(a.pos.source.clone()).or_default().push(a);
    }
    println!("Two-player endgame audit: regret = points lost vs exact best play (round margin, own side)\n");
    for (src, list) in &sources {
        let nontrivial: Vec<&&Audited> = list.iter().filter(|a| a.nontrivial()).collect();
        let rounds_seen: std::collections::HashSet<String> =
            list.iter().map(|a| a.pos.id.rsplit_once('.').map(|x| x.0.to_string()).unwrap_or_default()).collect();
        println!(
            "{src}: {} decisions with 2+ options in {} rounds, {} where the choice matters",
            list.len(),
            rounds_seen.len(),
            nontrivial.len()
        );
        println!("  {:<24} {:>8} {:>9} {:>10} {:>9} {:>9} {:>12}", "player", "optimal", "regret/d", "regret/rnd", ">=20 pts", ">=100", "worst");
        let mut rows: Vec<(String, Vec<Option<i32>>)> = vec![];
        rows.push(("actual player".into(), nontrivial.iter().map(|a| a.regret(a.actual)).collect()));
        for (k, b) in bot_names.iter().enumerate() {
            rows.push((b.clone(), nontrivial.iter().map(|a| a.regret(a.bots[k])).collect()));
        }
        for (name, regs) in rows {
            let regs: Vec<i32> = regs.into_iter().flatten().collect();
            if regs.is_empty() {
                continue;
            }
            let n = regs.len() as f64;
            let total: i64 = regs.iter().map(|&r| r as i64).sum();
            println!(
                "  {:<24} {:>7.1}% {:>9.1} {:>10.2} {:>9} {:>9} {:>12}",
                name,
                100.0 * regs.iter().filter(|&&r| r == 0).count() as f64 / n,
                total as f64 / n,
                total as f64 / rounds_seen.len().max(1) as f64,
                regs.iter().filter(|&&r| r >= 20).count(),
                regs.iter().filter(|&&r| r >= 100).count(),
                regs.iter().max().unwrap()
            );
        }
        println!();
    }

    if let Some(dir) = get("--out") {
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/positions.jsonl")).unwrap());
        for a in &results {
            let st = &a.pos.state;
            let opts: Vec<_> = (0..a.options.len())
                .map(|i| json!({"option": label(&a.options[i].decision), "value": a.value(i), "action": a.options[i].action}))
                .collect();
            let bots: serde_json::Map<String, serde_json::Value> =
                bot_names.iter().zip(&a.bots).map(|(n, c)| (n.clone(), json!(c))).collect();
            let line = json!({
                "source": a.pos.source, "id": a.pos.id, "seat": a.seat,
                "hands": (0..4).map(|s| st.players[s].hand.clone()).collect::<Vec<_>>(),
                "view": st.view(a.seat),
                "options": opts, "actual": a.actual, "bots": bots,
                "best": a.best(), "regret_actual": a.regret(a.actual),
            });
            writeln!(f, "{line}").unwrap();
        }
        let mut who = bot_names.clone();
        for w in &mut who {
            *w = format!("<{w}>");
        }
        let mut m = std::fs::File::create(format!("{dir}/mistakes.txt")).unwrap();
        for (src, list) in &sources {
            // Costliest first: the max regret of anyone at the position.
            let mut l: Vec<&&Audited> = list
                .iter()
                .filter(|a| a.regret(a.actual).unwrap_or(0) > 0 || a.bots.iter().any(|&b| a.regret(b).unwrap_or(0) > 0))
                .collect();
            let key = |a: &Audited| {
                std::iter::once(a.regret(a.actual)).chain(a.bots.iter().map(|&b| a.regret(b))).flatten().max().unwrap_or(0)
            };
            l.sort_by_key(|a| -key(a));
            writeln!(m, "##### {src}: {} positions where someone lost points\n", l.len()).unwrap();
            for a in l {
                writeln!(m, "{}", describe(a, &who)).unwrap();
            }
        }
        eprintln!("wrote {dir}/positions.jsonl and {dir}/mistakes.txt");
    }
}
