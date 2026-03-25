pub mod constants;
pub mod eval;
pub mod follow;
pub mod lead;
pub mod pass_strategy;
pub mod tichu_call;

use crate::engine::{Action, GameState};
use crate::player::Player;
use crate::types::*;

use constants::HeuristicWeights;

/// A bot that uses Tichu domain heuristics for all decisions.
pub struct HeuristicBot {
    pub weights: HeuristicWeights,
}

impl HeuristicBot {
    pub fn new() -> Self {
        Self {
            weights: HeuristicWeights::default(),
        }
    }

    pub fn with_weights(weights: HeuristicWeights) -> Self {
        Self { weights }
    }
}

impl Default for HeuristicBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Player for HeuristicBot {
    fn decide_grand_tichu(&mut self, state: &GameState, seat: Seat) -> bool {
        tichu_call::should_call_grand_tichu(state, seat, &self.weights)
    }

    fn decide_small_tichu(&mut self, state: &GameState, seat: Seat) -> bool {
        tichu_call::should_call_small_tichu(state, seat, &self.weights)
    }

    fn choose_pass_cards(&mut self, state: &GameState, seat: Seat) -> [Card; 3] {
        pass_strategy::choose_pass_cards(state, seat, &self.weights)
    }

    fn choose_action(
        &mut self,
        state: &GameState,
        seat: Seat,
        legal_actions: &[Action],
    ) -> (Action, Option<u8>) {
        // Handle dragon giveaway: give to the opponent with the most cards.
        if state.dragon_giveaway {
            let action = choose_dragon_target(state, seat, legal_actions);
            return (action, None);
        }

        let action = if state.current_trick.is_none() {
            // Leading.
            lead::choose_lead(&state.hands[seat], legal_actions, &self.weights)
        } else {
            // Following.
            follow::choose_follow(state, seat, legal_actions, &self.weights)
        };

        // If we're playing MahJong, wish for the most common rank among opponents.
        let wish = if contains_mahjong(&action) {
            choose_wish(state, seat)
        } else {
            None
        };

        (action, wish)
    }
}

/// Choose which opponent to give the dragon trick to.
/// Give to the opponent with the most cards (they're further from going out).
fn choose_dragon_target(state: &GameState, _seat: Seat, legal_actions: &[Action]) -> Action {
    let mut best_action = &legal_actions[0];
    let mut most_cards = 0;

    for action in legal_actions {
        if let Action::GiveDragonTrick(target) = action {
            let cards = state.hands[*target].len();
            if cards > most_cards {
                most_cards = cards;
                best_action = action;
            }
        }
    }

    best_action.clone()
}

/// Check if an action plays the MahJong.
fn contains_mahjong(action: &Action) -> bool {
    if let Action::Play(combo) = action {
        combo.cards.iter().any(|c| matches!(c, Card::Special(SpecialKind::MahJong)))
    } else {
        false
    }
}

/// Choose a rank to wish for when playing MahJong.
/// Wish for the rank most commonly held by opponents (and not by us).
fn choose_wish(state: &GameState, seat: Seat) -> Option<u8> {
    let my_team = team_for_seat(seat);
    let mut rank_counts = [0u8; 15]; // index 2..=14

    for s in 0..4 {
        if team_for_seat(s) == my_team {
            continue;
        }
        for card in &state.hands[s] {
            if let Card::Normal { rank, .. } = card {
                if *rank >= 2 && *rank <= 14 {
                    rank_counts[*rank as usize] += 1;
                }
            }
        }
    }

    // Wish for the highest rank that opponents hold (harder for them to play).
    // Prefer high ranks as they're more impactful.
    let best = (2..=14u8).rev().max_by_key(|&r| rank_counts[r as usize]);
    best.filter(|&r| rank_counts[r as usize] > 0)
}
