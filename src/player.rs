use crate::engine::{Action, GameState};
use crate::types::{Card, Seat};

/// Trait for implementing Tichu-playing agents.
///
/// All decision methods receive the full game state and the player's seat.
/// For self-play training this is fine; add observation filtering later
/// for competitive play with hidden information.
pub trait Player {
    /// Decide whether to call Grand Tichu (player sees only first 8 cards).
    fn decide_grand_tichu(&mut self, state: &GameState, seat: Seat) -> bool;

    /// Decide whether to call Small Tichu (called once before first play).
    fn decide_small_tichu(&mut self, _state: &GameState, _seat: Seat) -> bool {
        false
    }

    /// Choose 3 cards to pass: \[to_left, to_partner, to_right\].
    fn choose_pass_cards(&mut self, state: &GameState, seat: Seat) -> [Card; 3];

    /// Choose an action during play from the list of legal actions.
    /// Returns (action, optional mah_jong_wish) where the wish is a rank 2..=14,
    /// only relevant when playing a combo that contains the Mah Jong.
    fn choose_action(
        &mut self,
        state: &GameState,
        seat: Seat,
        legal_actions: &[Action],
    ) -> (Action, Option<u8>);

    /// Called at the start of each round (for bookkeeping).
    fn on_round_start(&mut self, _state: &GameState, _seat: Seat) {}

    /// Called when the game ends.
    fn on_game_end(&mut self, _state: &GameState, _seat: Seat) {}
}
