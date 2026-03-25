use crate::combinations::find_playable_combos;
use crate::types::*;

use super::constants::HeuristicWeights;

/// Analysis of a hand's composition and strength.
#[derive(Debug, Clone)]
pub struct HandAnalysis {
    pub num_cards: usize,
    pub has_dragon: bool,
    pub has_phoenix: bool,
    pub has_dog: bool,
    pub has_mahjong: bool,
    pub bomb_count: usize,
    pub high_card_count: usize, // rank >= 12 (Q, K, A)
    pub tempo: usize,           // estimated plays to empty hand
    pub strength: f64,          // composite score
}

/// Analyze a hand and compute its strength.
pub fn analyze_hand(hand: &[Card], weights: &HeuristicWeights) -> HandAnalysis {
    let has_dragon = hand.iter().any(|c| matches!(c, Card::Special(SpecialKind::Dragon)));
    let has_phoenix = hand.iter().any(|c| matches!(c, Card::Special(SpecialKind::Phoenix)));
    let has_dog = hand.iter().any(|c| matches!(c, Card::Special(SpecialKind::Dog)));
    let has_mahjong = hand.iter().any(|c| matches!(c, Card::Special(SpecialKind::MahJong)));

    let high_card_count = hand.iter().filter(|c| {
        matches!(c, Card::Normal { rank, .. } if *rank >= 12)
            || matches!(c, Card::Special(SpecialKind::Dragon))
    }).count();

    // Count bombs in hand.
    let all_combos = find_playable_combos(hand, None);
    let bomb_count = all_combos.iter().filter(|c| {
        matches!(c.kind, ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb)
    }).count();

    // Estimate tempo: greedily decompose hand into plays.
    let tempo = estimate_tempo(hand);

    // Composite strength score.
    let mut strength = 0.0;

    // Fewer groups = stronger hand.
    strength -= tempo as f64 * weights.group_penalty;

    // High cards are good.
    strength += high_card_count as f64 * 1.5;

    // Special cards.
    if has_dragon { strength += weights.dragon_value; }
    if has_phoenix { strength += weights.phoenix_value; }

    // Bombs are very valuable.
    strength += bomb_count as f64 * weights.bomb_bonus;

    HandAnalysis {
        num_cards: hand.len(),
        has_dragon,
        has_phoenix,
        has_dog,
        has_mahjong,
        bomb_count,
        high_card_count,
        tempo,
        strength,
    }
}

/// Estimate the minimum number of plays to empty a hand.
/// Uses a greedy approach: repeatedly play the largest available combo.
pub fn estimate_tempo(hand: &[Card]) -> usize {
    let mut remaining = hand.to_vec();
    let mut plays = 0;

    while !remaining.is_empty() {
        let combos = find_playable_combos(&remaining, None);
        if combos.is_empty() {
            // Each remaining card is a single play.
            plays += remaining.len();
            break;
        }

        // Pick the largest combo (most cards cleared per play).
        // Break ties by lowest rank (save high cards).
        let best = combos.iter().max_by(|a, b| {
            a.length.cmp(&b.length)
                .then_with(|| b.rank.partial_cmp(&a.rank).unwrap())
        }).unwrap();

        remaining = remove_cards(&remaining, &best.cards);
        plays += 1;
    }

    plays
}

/// Remove specific cards from a hand, returning the remainder.
pub fn remove_cards(hand: &[Card], to_remove: &[Card]) -> Vec<Card> {
    let mut remaining = hand.to_vec();
    for card in to_remove {
        if let Some(pos) = remaining.iter().position(|c| c == card) {
            remaining.remove(pos);
        }
    }
    remaining
}
