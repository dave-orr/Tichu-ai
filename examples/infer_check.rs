//! Sanity check for `infer`: at sampled decision points in h2 self-play,
//! compare the likelihood of the true hands with that of uniform guesses,
//! and count guesses the replay rejects.
//!
//!     cargo run --release --example infer_check

use std::cell::RefCell;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tichu_engine::bot::Bot;
use tichu_engine::bots::H2Bot;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{GameState, Pending};
use tichu_engine::infer::{likelihood, InferConfig};
use tichu_engine::sim::play_dealt_round;
use tichu_engine::types::*;

/// Plays as h2, and at random turns records (likelihood of the truth, mean likelihood of guesses, zero guesses).
struct Probe {
    inner: H2Bot,
    rng: StdRng,
    truth: Option<[Vec<Card>; 4]>,
    out: Rc<RefCell<Vec<(f64, f64, usize)>>>,
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
    fn pass_cards(&mut self, view: &tichu_engine::view::PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &tichu_engine::view::PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if view.played_cards.len() > 8 && self.rng.gen_bool(0.2) {
            let cfg = InferConfig::default();
            let t = likelihood(view, self.truth.as_ref().unwrap(), &cfg);
            let mut sum = 0.0;
            let mut zeros = 0;
            for _ in 0..50 {
                let h = view.sample_hands(&mut self.rng);
                let l = likelihood(view, &h, &cfg);
                sum += l;
                zeros += (l == 0.0) as usize;
            }
            self.out.borrow_mut().push((t, sum / 50.0, zeros));
        }
        self.inner.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &tichu_engine::view::PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.inner.bomb(view, bombs)
    }
    fn wish(&mut self, view: &tichu_engine::view::PlayerView) -> Option<Rank> {
        self.inner.wish(view)
    }
    fn dragon_recipient(&mut self, view: &tichu_engine::view::PlayerView) -> Seat {
        self.inner.dragon_recipient(view)
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let (mut n, mut truth_ln, mut guess_ln, mut zeros, mut better) = (0usize, 0.0, 0.0, 0usize, 0usize);
    for _ in 0..200 {
        let mut g = GameState::new(Settings::default());
        g.start_round(shuffled_deck(&mut rng));
        let out = Rc::new(RefCell::new(vec![]));
        let probe = Probe { inner: H2Bot::new(), rng: StdRng::seed_from_u64(rng.gen()), truth: None, out: out.clone() };
        let mut bots: [Box<dyn Bot>; 4] =
            [Box::new(probe), Box::new(H2Bot::new()), Box::new(H2Bot::new()), Box::new(H2Bot::new())];
        play_dealt_round(&mut g, &mut bots, None).unwrap();
        assert!(matches!(g.pending(), Pending::RoundOver | Pending::GameOver));
        for &(t, m, z) in out.borrow().iter() {
            n += 1;
            truth_ln += t.max(1e-300).ln();
            guess_ln += m.max(1e-300).ln();
            zeros += z;
            better += (t > m) as usize;
        }
    }
    println!(
        "{n} decision points: truth beats the average guess {:.0}% of the time; mean ln L truth {:.2}, guesses {:.2}; rejected guesses {:.1}%",
        100.0 * better as f64 / n as f64,
        truth_ln / n as f64,
        guess_ln / n as f64,
        100.0 * zeros as f64 / (50 * n) as f64
    );
}
