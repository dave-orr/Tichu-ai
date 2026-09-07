use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};

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

/// Fisher-Yates shuffle in place using the thread-local RNG.
pub fn shuffle(deck: &mut Vec<Card>) {
    shuffle_with(deck, &mut thread_rng());
}

/// Fisher-Yates shuffle in place using the supplied RNG (seed it for
/// reproducible deals).
pub fn shuffle_with<R: Rng + ?Sized>(deck: &mut Vec<Card>, rng: &mut R) {
    deck.shuffle(rng);
}

/// A deterministic RNG for reproducible games.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Sort a hand by the canonical Tichu ordering.
pub fn sort_hand(hand: &mut Vec<Card>) {
    hand.sort_by(|a, b| a.sort_key().partial_cmp(&b.sort_key()).unwrap());
}
