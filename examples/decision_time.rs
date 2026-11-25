//! How long a bot takes per decision, single-threaded, as a server-hosted
//! bot would see it.
//!
//!     cargo run --release --example decision_time -- [BOT] [OPPONENT] [ROUNDS]
//!
//! Defaults: `best` against `h2`, 40 rounds of whole games. BOT sits in seats 0/2 and
//! every one of its calls is timed (plays, bombs out of turn, calls, passes,
//! wish, Dragon gift). Prints count, mean, median, p90, p99 and max per kind.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::engine::GameState;
use tichu_engine::scoring::RoundResult;
use tichu_engine::sim;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

#[derive(Default)]
struct Times {
    by_kind: std::collections::BTreeMap<&'static str, Vec<f64>>,
}

struct Timed {
    inner: Box<dyn Bot>,
    times: Rc<RefCell<Times>>,
}

impl Timed {
    fn time<T>(&mut self, kind: &'static str, f: impl FnOnce(&mut dyn Bot) -> T) -> T {
        let t = Instant::now();
        let r = f(self.inner.as_mut());
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        self.times.borrow_mut().by_kind.entry(kind).or_default().push(ms);
        r
    }
}

impl Bot for Timed {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        self.time("grand", |b| b.grand_tichu(v))
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        self.time("small", |b| b.small_tichu(v))
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        self.time("pass", |b| b.pass_cards(v))
    }
    fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let kind = if plays.len() + can_pass as usize <= 1 { "play (forced)" } else { "play" };
        self.time(kind, |b| b.play(v, plays, can_pass))
    }
    fn bomb(&mut self, v: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.time("bomb offer", |b| b.bomb(v, bombs))
    }
    fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
        self.time("wish", |b| b.wish(v))
    }
    fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
        self.time("dragon", |b| b.dragon_recipient(v))
    }
    fn round_over(&mut self, v: &PlayerView, r: &RoundResult) {
        self.inner.round_over(v, r)
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = args.first().map(String::as_str).unwrap_or("best");
    let b = args.get(1).map(String::as_str).unwrap_or("h2");
    let rounds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let times = Rc::new(RefCell::new(Times::default()));
    let mk = |name: &str, seed: u64| bots::by_name(name, seed).unwrap_or_else(|| panic!("unknown bot {name}"));
    let t0 = Instant::now();
    // Whole games, as on the server: the same bots play every round of a game.
    let mut played = 0;
    let mut game = 0u64;
    let mut rng = StdRng::seed_from_u64(1000);
    while played < rounds {
        let mut seats: [Box<dyn Bot>; 4] = [
            Box::new(Timed { inner: mk(a, game * 4), times: times.clone() }),
            mk(b, game * 4 + 1),
            Box::new(Timed { inner: mk(a, game * 4 + 2), times: times.clone() }),
            mk(b, game * 4 + 3),
        ];
        let mut state = GameState::new(Settings::default());
        while state.phase != Phase::GameEnd && played < rounds {
            let t = Instant::now();
            sim::play_round(&mut state, &mut seats, &mut rng, None).expect("round");
            eprintln!("game {game} round {}: {:.1} s", state.round_number, t.elapsed().as_secs_f64());
            played += 1;
        }
        game += 1;
    }
    println!("{a} (seats 0/2) vs {b}, {rounds} rounds, {:.1} s, one thread", t0.elapsed().as_secs_f64());
    println!("{:<14} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}", "decision", "n", "mean ms", "median", "p90", "p99", "max");
    let mut total = 0.0;
    for (kind, v) in times.borrow_mut().by_kind.iter_mut() {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        total += v.iter().sum::<f64>();
        println!(
            "{:<14} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            kind, v.len(), mean, pct(v, 0.5), pct(v, 0.9), pct(v, 0.99), v[v.len() - 1]
        );
    }
    println!("thinking time per round per seat: {:.2} s", total / 1000.0 / rounds as f64 / 2.0);
}
