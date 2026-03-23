use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::types::*;

/// Create the full 56-card Tichu deck.
pub fn create_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(56);
    for &suit in &Suit::ALL {
        for rank in 2..=14u8 {
            deck.push(Card::Normal { suit, rank });
        }
    }
    deck.push(Card::Special(SpecialKind::MahJong));
    deck.push(Card::Special(SpecialKind::Dog));
    deck.push(Card::Special(SpecialKind::Phoenix));
    deck.push(Card::Special(SpecialKind::Dragon));
    deck
}

/// Fisher-Yates shuffle in place.
pub fn shuffle(deck: &mut Vec<Card>) {
    let mut rng = thread_rng();
    deck.shuffle(&mut rng);
}

/// Sort a hand by the canonical Tichu ordering.
pub fn sort_hand(hand: &mut Vec<Card>) {
    hand.sort_by(|a, b| a.sort_key().partial_cmp(&b.sort_key()).unwrap());
}
