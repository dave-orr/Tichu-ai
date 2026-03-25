use rand::seq::SliceRandom;
use rand::Rng;

use crate::deck::create_deck;
use crate::engine::GameState;
use crate::types::*;

/// Create a determinized copy of the game state from `perspective` seat's point of view.
///
/// The perspective player keeps their actual hand. All other players' hands
/// are redistributed from the pool of unknown cards (cards not visible to
/// the perspective player), while preserving each player's hand size.
pub fn determinize(state: &GameState, perspective: Seat, rng: &mut impl Rng) -> GameState {
    let mut det = state.clone();

    // Compute the pool of unknown cards and shuffle them.
    let mut unknown = full_deck_minus_known(&state.hands[perspective], state);
    unknown.shuffle(rng);

    // Redistribute unknown cards to other players, preserving hand sizes.
    let mut idx = 0;
    for seat in 0..4 {
        if seat == perspective {
            continue;
        }
        let hand_size = state.hands[seat].len();
        det.hands[seat] = unknown[idx..idx + hand_size].to_vec();
        det.hands[seat].sort_by(|a, b| a.sort_key().partial_cmp(&b.sort_key()).unwrap());
        idx += hand_size;
    }

    det
}

/// Compute the pool of cards not visible to the perspective player.
fn full_deck_minus_known(my_hand: &[Card], state: &GameState) -> Vec<Card> {
    let mut deck = create_deck();

    // Remove my hand.
    for card in my_hand {
        if let Some(pos) = deck.iter().position(|c| c == card) {
            deck.remove(pos);
        }
    }

    // Remove cards from all tricks.
    for seat_tricks in &state.tricks_won {
        for trick in seat_tricks {
            for card in trick {
                if let Some(pos) = deck.iter().position(|c| c == card) {
                    deck.remove(pos);
                }
            }
        }
    }

    // Remove cards in current trick pile.
    for card in &state.current_trick_cards {
        if let Some(pos) = deck.iter().position(|c| c == card) {
            deck.remove(pos);
        }
    }

    deck
}
