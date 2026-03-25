use crate::engine::GameState;
use crate::types::*;

use super::constants::HeuristicWeights;

/// Choose 3 cards to pass: [to_left, to_partner, to_right].
/// Strategy:
/// - To partner: pass a low card to help them (don't break pairs/bombs)
/// - To opponents (left, right): pass isolated high cards that don't form combos
/// - Never pass Dragon, Phoenix, or cards that are part of a bomb
pub fn choose_pass_cards(state: &GameState, seat: Seat, _weights: &HeuristicWeights) -> [Card; 3] {
    let hand = &state.hands[seat];
    let mut candidates: Vec<Card> = hand.clone();

    // Never pass Dragon or Phoenix.
    candidates.retain(|c| !matches!(c, Card::Special(SpecialKind::Dragon | SpecialKind::Phoenix)));

    // Never pass cards that are part of a four-of-a-kind bomb.
    let bomb_cards = find_bomb_cards(hand);
    candidates.retain(|c| !bomb_cards.contains(c));

    // Sort by rank for selection.
    candidates.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    // If we don't have 3 candidates (unlikely), fall back to first 3 from full hand.
    if candidates.len() < 3 {
        return [hand[0], hand[1], hand[2]];
    }

    // To partner: pass our lowest card (help them, dump our weakness).
    let to_partner = candidates[0];

    // To opponents: pass our next-lowest isolated cards.
    // (In a more sophisticated version, we'd pass high isolated cards to opponents,
    //  but for now passing low junk is safer than accidentally helping them.)
    let mut remaining: Vec<Card> = candidates[1..].to_vec();

    let to_left = remaining.remove(0);
    let to_right = remaining.remove(0);

    [to_left, to_partner, to_right]
}

/// Find all cards that belong to a four-of-a-kind bomb.
fn find_bomb_cards(hand: &[Card]) -> Vec<Card> {
    use std::collections::HashMap;
    let mut rank_cards: HashMap<u8, Vec<Card>> = HashMap::new();
    for card in hand {
        if let Card::Normal { rank, .. } = card {
            rank_cards.entry(*rank).or_default().push(*card);
        }
    }
    let mut bomb_cards = Vec::new();
    for (_rank, cards) in rank_cards {
        if cards.len() == 4 {
            bomb_cards.extend(cards);
        }
    }
    bomb_cards
}
