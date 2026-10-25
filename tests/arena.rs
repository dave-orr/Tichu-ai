//! The arena: results are reproducible from the seed whatever the thread
//! count, duplicate seating cancels the deal, and swapping A and B negates
//! the result. Bot comparisons quoted in PRs depend on all three.

mod common;

use tichu_engine::arena::{self, ArenaConfig, Mode, Report};
use tichu_engine::bots;

fn run(a: &str, b: &str, cfg: &ArenaConfig) -> Report {
    let (na, nb) = (a.to_string(), b.to_string());
    let fa = move |s: u64| bots::by_name(&na, s).unwrap();
    let fb = move |s: u64| bots::by_name(&nb, s).unwrap();
    let r = arena::run(a, &fa, b, &fb, cfg);
    assert!(r.errors.is_empty(), "{a} vs {b}: {:?}", r.errors);
    r
}

fn cfg(count: usize, threads: usize) -> ArenaConfig {
    ArenaConfig { count, threads, seed: 7, ..Default::default() }
}

#[test]
fn results_dont_depend_on_thread_count() {
    for (a, b) in [("heuristic", "random"), ("pimc2", "h2")] {
        let one = run(a, b, &cfg(12, 1));
        let four = run(a, b, &cfg(12, 4));
        assert_eq!(one.samples, four.samples, "{a} vs {b}");
        assert_eq!(one.a.points, four.a.points);
        assert_eq!(one.b.small_calls, four.b.small_calls);
    }
}

#[test]
fn seed_changes_the_deals() {
    let a = run("heuristic", "random", &cfg(20, 2));
    let b = run("heuristic", "random", &ArenaConfig { seed: 8, ..cfg(20, 2) });
    assert_ne!(a.samples, b.samples);
}

/// A seedless bot against itself scores exactly 0 per deal with duplicate:
/// both seatings play the same cards the same way.
#[test]
fn duplicate_mirror_match_is_exactly_even() {
    for bot in ["heuristic", "h2"] {
        let r = run(bot, bot, &cfg(30, 4));
        assert!(r.samples.iter().all(|&x| x == 0.0), "{bot} vs itself: {:?}", r.samples);
        assert_eq!(r.a.points, r.b.points);
    }
}

/// Swapping the sides negates every sample (for seedless bots, whose play
/// doesn't depend on which seed slot they get).
#[test]
fn swapping_sides_negates_the_result() {
    let ab = run("heuristic", "h2", &cfg(30, 4));
    let ba = run("h2", "heuristic", &cfg(30, 4));
    let neg: Vec<f64> = ba.samples.iter().map(|x| -x).collect();
    assert_eq!(ab.samples, neg);
    assert_eq!(ab.a.points, ba.b.points);
    assert_eq!(ab.a.first_out, ba.b.first_out);
}

#[test]
fn without_duplicate_each_deal_is_played_once() {
    let r = run("heuristic", "random", &ArenaConfig { duplicate: false, ..cfg(10, 2) });
    assert_eq!(r.samples.len(), 10);
    assert_eq!(r.a.rounds, 10);
    let d = run("heuristic", "random", &cfg(10, 2));
    assert_eq!(d.a.rounds, 20);
}

#[test]
fn game_mode_plays_full_games() {
    let r = run("heuristic", "random", &ArenaConfig { mode: Mode::Games, ..cfg(4, 2) });
    assert_eq!(r.games, 4);
    assert_eq!(r.samples.len(), 4);
    assert!(r.a.rounds >= 2 * 4 * 2, "games of one round? {} rounds", r.a.rounds);
    assert!(r.a_game_wins > 3.0, "heuristic should beat random: {}", r.a_game_wins);
    assert!(r.summary().contains("game win rate"));
}

/// Per-side tallies agree with the samples they summarize.
#[test]
fn report_totals_match_samples() {
    let r = run("h2", "random", &cfg(25, 3));
    let sum: f64 = r.samples.iter().sum::<f64>() * 2.0; // duplicate: each sample is a two-seating mean
    assert_eq!(sum as i64, r.a.points - r.b.points);
    assert_eq!(r.a.rounds, r.b.rounds);
    assert!(r.a.small_made <= r.a.small_calls && r.a.grand_made <= r.a.grand_calls);
    assert!((r.mean() - r.samples.iter().sum::<f64>() / 25.0).abs() < 1e-9);
    assert!(r.stderr().is_finite() && r.stderr() > 0.0);
}
