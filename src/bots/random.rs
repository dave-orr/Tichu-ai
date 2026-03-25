use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::engine::{Action, GameState};
use crate::player::Player;
use crate::types::{Card, Seat};

/// A bot that makes uniformly random legal moves.
pub struct RandomBot;

impl Player for RandomBot {
    fn decide_grand_tichu(&mut self, _state: &GameState, _seat: Seat) -> bool {
        false
    }

    fn choose_pass_cards(&mut self, state: &GameState, seat: Seat) -> [Card; 3] {
        let mut rng = thread_rng();
        let hand = &state.hands[seat];
        let mut indices: Vec<usize> = (0..hand.len()).collect();
        indices.shuffle(&mut rng);
        [hand[indices[0]], hand[indices[1]], hand[indices[2]]]
    }

    fn choose_action(
        &mut self,
        _state: &GameState,
        _seat: Seat,
        legal_actions: &[Action],
    ) -> (Action, Option<u8>) {
        let mut rng = thread_rng();
        let action = legal_actions.choose(&mut rng).unwrap().clone();
        (action, None)
    }
}
