use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::types::*;
use crate::view::PlayerView;

/// Plays uniformly random legal moves. Calls Tichu and bombs now and then,
/// mostly so random games exercise every rule.
pub struct RandomBot {
    rng: StdRng,
}

impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot { rng: StdRng::seed_from_u64(seed) }
    }
}

impl Bot for RandomBot {
    fn name(&self) -> String {
        "random".into()
    }

    fn grand_tichu(&mut self, _view: &PlayerView) -> bool {
        self.rng.gen_bool(0.03)
    }

    fn small_tichu(&mut self, _view: &PlayerView) -> bool {
        self.rng.gen_bool(0.03)
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        let picks: Vec<Card> = view.hand.choose_multiple(&mut self.rng, 3).copied().collect();
        PassInfo { left: picks[0], partner: picks[1], right: picks[2] }
    }

    fn play(&mut self, _view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if plays.is_empty() || (can_pass && self.rng.gen_bool(0.4)) {
            return None;
        }
        Some(self.rng.gen_range(0..plays.len()))
    }

    fn bomb(&mut self, _view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        if self.rng.gen_bool(0.25) { Some(self.rng.gen_range(0..bombs.len())) } else { None }
    }

    fn wish(&mut self, _view: &PlayerView) -> Option<Rank> {
        if self.rng.gen_bool(0.8) { Some(self.rng.gen_range(2..=14)) } else { None }
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        *view.opponents().choose(&mut self.rng).unwrap()
    }
}
