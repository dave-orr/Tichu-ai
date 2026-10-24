//! How often two-player endgames come up, and how hard they are to solve.
//!
//!     cargo run --release --example endgame_stats -- [BOT] [ROUNDS] [BUDGET]
//!
//! Plays ROUNDS rounds with BOT in all seats (default heuristic, 500) and
//! solves the first on-turn position of every two-player endgame.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::endgame;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::sim::play_round;
use tichu_engine::types::*;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bot = args.first().cloned().unwrap_or_else(|| "heuristic".into());
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let budget: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let mut rng = StdRng::seed_from_u64(1);

    let mut found = 0;
    let mut solved: Vec<(usize, f64, usize)> = vec![]; // (nodes, seconds, cards)
    let mut failed: Vec<usize> = vec![];
    for i in 0..rounds {
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| bots::by_name(&bot, (i * 4 + k) as u64).unwrap());
        let mut first: Option<GameState> = None;
        let mut obs = |s: &GameState, a: &Action| {
            if first.is_none() && matches!(a, Action::Play { .. } | Action::PassTurn { .. }) && endgame::is_two_player_endgame(s) {
                first = Some(s.clone());
            }
        };
        let mut g = GameState::new(Settings::default());
        play_round(&mut g, &mut bots, &mut rng, Some(&mut obs)).unwrap();
        let Some(start) = first else { continue };
        found += 1;
        let cards: usize = start.players.iter().map(|p| p.hand.len()).sum();
        let t = Instant::now();
        match endgame::solve(&start, budget) {
            Some(sol) => solved.push((sol.nodes, t.elapsed().as_secs_f64(), cards)),
            None => failed.push(cards),
        }
    }

    println!("{bot}: {rounds} rounds, {found} reached a two-player endgame ({:.0}%)", 100.0 * found as f64 / rounds as f64);
    println!("  solved {} within {budget} nodes, {} over budget (cards left: {:?})", solved.len(), failed.len(), failed);
    if solved.is_empty() {
        return;
    }
    let mut nodes: Vec<usize> = solved.iter().map(|x| x.0).collect();
    let mut secs: Vec<f64> = solved.iter().map(|x| x.1).collect();
    nodes.sort_unstable();
    secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p) as usize];
    let qn = |p: f64| nodes[((nodes.len() - 1) as f64 * p) as usize];
    println!("  nodes   median {} | p90 {} | p99 {} | max {}", qn(0.5), qn(0.9), qn(0.99), qn(1.0));
    println!(
        "  time    median {:.2}ms | p90 {:.2}ms | p99 {:.1}ms | max {:.1}ms",
        q(&secs, 0.5) * 1e3,
        q(&secs, 0.9) * 1e3,
        q(&secs, 0.99) * 1e3,
        q(&secs, 1.0) * 1e3
    );
    let mut by_cards = std::collections::BTreeMap::new();
    for (n, _, c) in &solved {
        by_cards.entry(*c).or_insert_with(Vec::new).push(*n);
    }
    println!("  cards left -> positions, median nodes:");
    for (c, mut v) in by_cards {
        v.sort_unstable();
        println!("    {c:>2}: {:>4}  {}", v.len(), v[v.len() / 2]);
    }
}
