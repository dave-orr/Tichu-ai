//! How much the partner card of a pass matters, measured by paired rollouts.
//!
//! For each sampled pass decision (a fresh deal, h2's Grand calls, one seat
//! per deal), keep h2's two opponent cards and try every other card as the
//! partner card. Guess the other three hands `--samples` times; for each
//! guess, play the round out once per candidate with the same guessed deal
//! and the same bots, so only the pass differs (other seats pass as h2 would
//! with those hands). PLAY (default a dmc1 net) plays the cards and h2 makes
//! the calls, so a pass that makes a hand strong enough to call gets the
//! call's points.
//!
//! The gain over h2's partner card is estimated without the max-of-noise
//! bias: the best card is picked on one half of the guesses and scored on
//! the other half (both ways round). Decisions are split by what happened
//! in the rollouts of h2's pass: my team called Tichu/Grand in most of
//! them, the opponents did, or neither.
//!
//! cargo run --release --example pass_value -- [--deals N] [--samples K] [--play BOT] [--seed S] [--threads T] [--out FILE.jsonl]

use std::io::Write;
use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tichu_engine::bot::Bot;
use tichu_engine::bots::{self, H2Bot};
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::scoring::RoundResult;
use tichu_engine::sim::play_from;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

struct Decision {
    seat: Seat,
    h2_partner: Card,
    cands: Vec<Card>,
    /// rows[sample][candidate]: my team's margin.
    rows: Vec<Vec<f64>>,
    /// Over rollouts of h2's pass: share where my team / the opponents called.
    my_call: f64,
    opp_call: f64,
}

fn team_called(r: &RoundResult, team: usize) -> bool {
    (0..4).any(|s| s % 2 == team && r.tichu_calls[s] != TichuCall::None)
}

/// The state just after the Grand window with `hands` dealt and `view`'s
/// Grand calls.
fn pass_state(view: &PlayerView, hands: &[Vec<Card>; 4]) -> GameState {
    let mut deck = Vec::with_capacity(56);
    for h in hands {
        deck.extend_from_slice(h);
    }
    let mut g = GameState::new(view.settings);
    g.start_round(deck);
    for s in 0..4 {
        g.apply(&Action::GrandTichu { seat: s, call: view.players[s].tichu == TichuCall::Grand }).unwrap();
    }
    g
}

fn rollout(view: &PlayerView, hands: &[Vec<Card>; 4], mine: PassInfo, play: &str, seed: u64) -> RoundResult {
    let mut g = pass_state(view, hands);
    for s in 0..4 {
        let pass = if s == view.seat { mine } else { H2Bot::new().pass_cards(&g.view(s)) };
        g.apply(&Action::PassCards { seat: s, pass }).unwrap();
    }
    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| bots::by_name(play, seed * 4 + k).unwrap());
    play_from(&mut g, &mut bots, None, true).expect("rollout failed")
}

fn decide(deal: u64, seed: u64, samples: usize, play: &str) -> Decision {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9).wrapping_add(deal));
    let mut g = GameState::new(Settings::default());
    g.start_round(shuffled_deck(&mut rng));
    for s in 0..4 {
        let call = H2Bot::new().grand_tichu(&g.view(s));
        g.apply(&Action::GrandTichu { seat: s, call }).unwrap();
    }
    let seat = (deal % 4) as Seat;
    let view = g.view(seat);
    let h2 = H2Bot::new().pass_cards(&view);
    let cands: Vec<Card> = view.hand.iter().copied().filter(|&c| c != h2.left && c != h2.right).collect();
    let team = seat % 2;
    let (mut rows, mut my_call, mut opp_call) = (vec![], 0, 0);
    for k in 0..samples {
        let hands = view.sample_hands(&mut rng);
        let row: Vec<f64> = cands
            .iter()
            .map(|&c| {
                let r = rollout(&view, &hands, PassInfo { partner: c, ..h2 }, play, deal * 1000 + k as u64);
                if c == h2.partner {
                    my_call += team_called(&r, team) as usize;
                    opp_call += team_called(&r, 1 - team) as usize;
                }
                (r.total(team) - r.total(1 - team)) as f64
            })
            .collect();
        rows.push(row);
    }
    let n = samples as f64;
    Decision { seat, h2_partner: h2.partner, cands, rows, my_call: my_call as f64 / n, opp_call: opp_call as f64 / n }
}

fn mean_of(rows: &[Vec<f64>], k: usize, par: usize) -> f64 {
    let v: Vec<f64> = rows.iter().skip(par).step_by(2).map(|r| r[k]).collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Unbiased gain of the best card over h2's: pick on one half, score on the other.
fn split_gain(d: &Decision) -> (f64, bool) {
    let h = d.cands.iter().position(|&c| c == d.h2_partner).unwrap();
    let mut gain = 0.0;
    let mut differs = false;
    for par in 0..2 {
        let best = (0..d.cands.len()).max_by(|&a, &b| mean_of(&d.rows, a, par).total_cmp(&mean_of(&d.rows, b, par))).unwrap();
        gain += (mean_of(&d.rows, best, 1 - par) - mean_of(&d.rows, h, 1 - par)) / 2.0;
        differs |= best != h;
    }
    (gain, differs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let deals: u64 = get("--deals").map_or(200, |v| v.parse().unwrap());
    let samples: usize = get("--samples").map_or(24, |v| v.parse().unwrap());
    let seed: u64 = get("--seed").map_or(1, |v| v.parse().unwrap());
    let threads: usize = get("--threads").map_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()), |v| v.parse().unwrap());
    let play = get("--play").unwrap_or_else(|| "net:runs/dmc1/snap00190.tnet".into());
    assert!(bots::by_name(&play, 0).is_some(), "unknown bot {play}");
    let out = get("--out").map(|p| Mutex::new(std::fs::File::create(p).unwrap()));
    let results = Mutex::new(vec![]);
    let next = Mutex::new(0u64);
    let start = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let deal = {
                    let mut n = next.lock().unwrap();
                    if *n >= deals {
                        return;
                    }
                    *n += 1;
                    *n - 1
                };
                let d = decide(deal, seed, samples, &play);
                let (gain, differs) = split_gain(&d);
                if let Some(f) = &out {
                    let means: Vec<f64> = (0..d.cands.len()).map(|k| d.rows.iter().map(|r| r[k]).sum::<f64>() / d.rows.len() as f64).collect();
                    let line = serde_json::json!({
                        "deal": deal, "seat": d.seat, "h2_partner": d.h2_partner, "cands": d.cands, "q": means,
                        "gain": gain, "differs": differs, "my_call": d.my_call, "opp_call": d.opp_call,
                    });
                    writeln!(f.lock().unwrap(), "{line}").unwrap();
                }
                let mut r = results.lock().unwrap();
                r.push((gain, differs, d.my_call, d.opp_call));
                if r.len() % 20 == 0 {
                    eprint!("\r{}/{deals} decisions ({:.0}s)   ", r.len(), start.elapsed().as_secs_f64());
                }
            });
        }
    });
    eprintln!();
    let r = results.into_inner().unwrap();
    let report = |name: &str, sel: &dyn Fn(f64, f64) -> bool| {
        let g: Vec<f64> = r.iter().filter(|x| sel(x.2, x.3)).map(|x| x.0).collect();
        let dif = r.iter().filter(|x| sel(x.2, x.3) && x.1).count();
        let n = g.len() as f64;
        if n < 2.0 {
            println!("{name:<28} {:>4} decisions", g.len());
            return;
        }
        let m = g.iter().sum::<f64>() / n;
        let sd = (g.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt();
        println!(
            "{name:<28} {:>4} decisions ({:>3.0}%): gain over h2's partner card {:+.1} ± {:.1} pts; best differs {:.0}%",
            g.len(),
            100.0 * n / r.len() as f64,
            m,
            1.96 * sd / n.sqrt(),
            100.0 * dif as f64 / n
        );
    };
    println!("partner-card search: {} deals, {samples} guesses each, play {play} ({:.0}s)", r.len(), start.elapsed().as_secs_f64());
    report("all", &|_, _| true);
    report("my team usually calls", &|m, _| m >= 0.5);
    report("opponents usually call", &|m, o| m < 0.5 && o >= 0.5);
    report("neither usually calls", &|m, o| m < 0.5 && o < 0.5);
}
