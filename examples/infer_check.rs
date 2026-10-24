//! How much does `infer` sharpen the guess at the hidden hands? At random
//! turns of seat 0, compare the share of hidden cards that uniform guesses
//! and inferred guesses put in the right hand.
//!
//!     cargo run --release --example infer_check -- [OTHERS] [ROUNDS] [EPS]
//!
//! OTHERS (default h2) plays seats 1-3; seat 0 plays h2.

use std::cell::RefCell;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tichu_engine::bot::Bot;
use tichu_engine::bots::{by_name, H2Bot};
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::GameState;
use tichu_engine::infer::{sample_hands_weighted, InferConfig};
use tichu_engine::sim::play_dealt_round;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

#[derive(Default)]
struct Tally {
    points: usize,
    uniform: f64,
    inferred: f64,
}

struct Probe {
    inner: H2Bot,
    rng: StdRng,
    cfg: InferConfig,
    truth: Option<[Vec<Card>; 4]>,
    tally: Rc<RefCell<Tally>>,
}

/// Share of the hidden cards placed in the right seat, averaged over `guesses`.
fn accuracy(view: &PlayerView, truth: &[Vec<Card>; 4], guesses: &[[Vec<Card>; 4]]) -> f64 {
    let mut right = 0usize;
    let mut total = 0usize;
    for g in guesses {
        for s in (0..4).filter(|&s| s != view.seat) {
            let t = CardSet::from_cards(&truth[s]);
            right += g[s].iter().filter(|c| t.contains(**c)).count();
            total += g[s].len();
        }
    }
    right as f64 / total.max(1) as f64
}

impl Bot for Probe {
    fn name(&self) -> String {
        "probe".into()
    }
    fn cheats(&self) -> bool {
        true
    }
    fn peek(&mut self, state: &GameState) {
        self.truth = Some([0, 1, 2, 3].map(|s| state.players[s].hand.clone()));
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if view.played_cards.len() > 8 && self.rng.gen_bool(0.2) {
            let truth = self.truth.clone().unwrap();
            let uniform: Vec<_> = (0..20).map(|_| view.sample_hands(&mut self.rng)).collect();
            let inferred = sample_hands_weighted(view, &mut self.rng, 20, &self.cfg);
            let mut t = self.tally.borrow_mut();
            t.points += 1;
            t.uniform += accuracy(view, &truth, &uniform);
            t.inferred += accuracy(view, &truth, &inferred);
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
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let others = args.get(1).cloned().unwrap_or("h2".into());
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut cfg = InferConfig::default();
    if let Some(e) = args.get(3).and_then(|s| s.parse().ok()) {
        cfg.eps_play = e;
    }
    let mut rng = StdRng::seed_from_u64(7);
    let tally = Rc::new(RefCell::new(Tally::default()));
    for _ in 0..rounds {
        let mut g = GameState::new(Settings::default());
        g.start_round(shuffled_deck(&mut rng));
        let probe = Probe { inner: H2Bot::new(), rng: StdRng::seed_from_u64(rng.gen()), cfg, truth: None, tally: tally.clone() };
        let other = |seed: u64| by_name(&others, seed).expect("unknown bot");
        let mut bots: [Box<dyn Bot>; 4] = [Box::new(probe), other(rng.gen()), other(rng.gen()), other(rng.gen())];
        play_dealt_round(&mut g, &mut bots, None).unwrap();
    }
    let t = tally.borrow();
    let n = t.points.max(1) as f64;
    println!(
        "others = {others}, eps_play = {}: {} decision points; hidden cards placed right: uniform {:.1}%, inferred {:.1}%",
        cfg.eps_play,
        t.points,
        100.0 * t.uniform / n,
        100.0 * t.inferred / n
    );
}
