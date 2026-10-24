//! Calibrates the v2 call rules: plays rounds with calls switched off and
//! records, per seat, the Small Tichu margin (winners − losers) at its first
//! turn and the Grand Tichu rule its first 8 cards meet, then how often each
//! group actually went out first. A call is worth roughly
//! 100·(2p − 1) (Small) or 200·(2p − 1) (Grand) when p is that rate.
//!
//!     cargo run --release --example call_calibration -- [BOT] [DEALS]
//!     (defaults: heuristic, 5000; the arena plays each deal twice)

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tichu_engine::arena::{self, ArenaConfig};
use tichu_engine::bot::Bot;
use tichu_engine::bots::calls::{grand_rule, tichu_eval};
use tichu_engine::bots::{self, CallPolicy, CallsBot};
use tichu_engine::scoring::RoundResult;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

#[derive(Default)]
struct Tally {
    small: BTreeMap<i32, (usize, usize)>,
    grand: BTreeMap<u8, (usize, usize)>,
}

struct Recorder {
    inner: Box<dyn Bot>,
    margin: Option<i32>,
    rule: Option<u8>,
    tally: Arc<Mutex<Tally>>,
}

impl Bot for Recorder {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        self.rule = Some(grand_rule(view));
        false
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if self.margin.is_none() && !view.me().has_played_first_card {
            self.margin = Some(tichu_eval(view).margin().round() as i32);
        }
        self.inner.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.inner.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.inner.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.inner.dragon_recipient(view)
    }
    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        let first = (result.out_order[0] == view.seat) as usize;
        let mut t = self.tally.lock().unwrap();
        if let Some(m) = self.margin.take() {
            let e = t.small.entry(m).or_default();
            e.0 += 1;
            e.1 += first;
        }
        if let Some(r) = self.rule.take() {
            let e = t.grand.entry(r).or_default();
            e.0 += 1;
            e.1 += first;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bot = args.first().cloned().unwrap_or_else(|| "heuristic".into());
    let deals: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    bots::by_name(&bot, 0).expect("unknown bot");
    let tally = Arc::new(Mutex::new(Tally::default()));
    let (t2, name) = (tally.clone(), bot.clone());
    let factory = move |seed: u64| -> Box<dyn Bot> {
        let (bot, tally) = (&name, &t2);
        let inner = Box::new(CallsBot::new(CallPolicy::Never, bots::by_name(&bot, seed).unwrap()));
        Box::new(Recorder { inner, margin: None, rule: None, tally: tally.clone() })
    };
    let cfg = ArenaConfig { count: deals, ..Default::default() };
    arena::run(&bot, &factory, &bot, &factory, &cfg);

    let t = tally.lock().unwrap();
    let total: usize = t.small.values().map(|v| v.0).sum();
    println!("Small Tichu: margin (winners − losers) at first turn, {bot} play, no calls");
    println!("  margin   hands   share  went out 1st   EV of calling");
    let mut cum = 0;
    for (m, (n, f)) in t.small.iter().rev() {
        cum += n;
        let p = *f as f64 / *n as f64;
        println!(
            "  {m:>+5}  {n:>7}  {:>5.1}%  {:>6.1}%        {:>+6.0}   (calling at >= {m:+}: {:.1}% of hands)",
            100.0 * *n as f64 / total as f64,
            100.0 * p,
            100.0 * (2.0 * p - 1.0),
            100.0 * cum as f64 / total as f64
        );
    }
    let names = ["no rule", "3 tops incl. special", "A + special + extra", "bomb + special/2A", "A + special only"];
    println!("\nGrand Tichu: rule met by the first 8 cards");
    for (r, (n, f)) in &t.grand {
        let p = *f as f64 / *n as f64;
        println!(
            "  {:<22} {n:>7}  {:>5.2}% of hands  went out 1st {:>5.1}%  EV of calling {:>+5.0}",
            names[*r as usize],
            100.0 * *n as f64 / t.grand.values().map(|v| v.0).sum::<usize>() as f64,
            100.0 * p,
            200.0 * (2.0 * p - 1.0)
        );
    }
}
