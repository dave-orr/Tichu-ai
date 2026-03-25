use crate::engine::GameState;
use crate::types::*;

use super::constants::HeuristicWeights;
use super::eval::analyze_hand;

/// Decide whether to call Grand Tichu (only first 8 cards visible).
pub fn should_call_grand_tichu(state: &GameState, seat: Seat, weights: &HeuristicWeights) -> bool {
    let hand = &state.hands[seat];
    let analysis = analyze_hand(hand, weights);

    // Grand Tichu is risky — need a very strong 8-card hand.
    // Key indicators: dragon, phoenix, multiple high cards, a bomb.
    analysis.strength >= weights.grand_tichu_threshold
}

/// Decide whether to call Small Tichu (full 14 cards visible).
pub fn should_call_small_tichu(state: &GameState, seat: Seat, weights: &HeuristicWeights) -> bool {
    let hand = &state.hands[seat];
    let analysis = analyze_hand(hand, weights);

    // Small Tichu: call if hand is fast (low tempo) with control.
    analysis.tempo as f64 <= weights.small_tichu_tempo_threshold
        && analysis.high_card_count >= 3
}
