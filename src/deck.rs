use rand::seq::SliceRandom;
use rand::Rng;

use crate::types::*;

/// The full 56-card deck, in the same order as TS `createDeck`.
pub fn create_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(56);
    for &suit in &Suit::ALL {
        for rank in 2..=14u8 {
            deck.push(Card::Normal { suit, rank });
        }
    }
    deck.extend_from_slice(&[MAH_JONG, DOG, PHOENIX, DRAGON]);
    deck
}

/// A shuffled deck. Seat `i` is dealt `deck[14i..14i+8]` first and
/// `deck[14i+8..14i+14]` after the Grand Tichu window (same layout as TS).
pub fn shuffled_deck<R: Rng + ?Sized>(rng: &mut R) -> Vec<Card> {
    let mut deck = create_deck();
    deck.shuffle(rng);
    deck
}

/// Sort a hand by the canonical Tichu ordering.
pub fn sort_hand(hand: &mut [Card]) {
    hand.sort_by_key(|c| c.sort_key());
}
