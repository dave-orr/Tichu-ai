//! Hand analysis: splitting a hand into the fewest combinations.

use std::collections::HashMap;

use crate::combinations::playable_combos;
use crate::types::*;

/// A partition of a hand into combinations.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub combos: Vec<Combo>,
    /// The Phoenix was kept out of the combos (it's one of the singles).
    pub loose_phoenix: bool,
}

impl Decomposition {
    /// Singles other than the specials (Dragon, Phoenix, Dog, Mah Jong).
    pub fn normal_singles(&self) -> impl Iterator<Item = &Combo> {
        self.combos.iter().filter(|c| c.kind == ComboKind::Single && !c.cards[0].is_special())
    }
}

/// Lexicographic cost: fewest combos, then fewest singles, then (as a tie
/// break) singles as high as possible.
type Cost = (usize, usize, i32);

fn cost_of(combos: &[Combo]) -> Cost {
    let singles: Vec<&Combo> = combos.iter().filter(|c| c.kind == ComboKind::Single).collect();
    (combos.len(), singles.len(), -singles.iter().map(|c| c.rank as i32).sum::<i32>())
}

/// Exact fewest-combinations split of `cards` (memoized search: each step
/// picks a combination containing the lowest remaining card).
fn best_split(cards: CardSet, memo: &mut HashMap<u64, (Cost, Vec<Combo>)>) -> (Cost, Vec<Combo>) {
    if cards.is_empty() {
        return ((0, 0, 0), vec![]);
    }
    if let Some(r) = memo.get(&cards.0) {
        return r.clone();
    }
    let hand = cards.cards();
    let lowest = Card::from_index(cards.0.trailing_zeros() as usize);
    let mut best: Option<(Cost, Vec<Combo>)> = None;
    for combo in playable_combos(&hand, None) {
        if !combo.cards.contains(&lowest) {
            continue;
        }
        let (_, mut rest) = best_split(cards.minus(combo.card_set()), memo);
        rest.push(combo);
        let cost = cost_of(&rest);
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, rest));
        }
    }
    let best = best.expect("every card is at least a single");
    memo.insert(cards.0, best.clone());
    best
}

/// Split `hand` into the fewest combinations. The Phoenix is only used in a
/// combination if that removes at least two singles; otherwise it is kept
/// loose (it's more useful as a flexible high single).
pub fn decompose(hand: &[Card]) -> Decomposition {
    let mut memo = HashMap::new();
    let all = CardSet::from_cards(hand);
    if !all.contains(PHOENIX) {
        let (_, combos) = best_split(all, &mut memo);
        return Decomposition { combos, loose_phoenix: false };
    }
    let mut without = all;
    without.remove(PHOENIX);
    let (_, mut loose) = best_split(without, &mut memo);
    let (_, with) = best_split(all, &mut memo);
    let singles = |cs: &[Combo]| cs.iter().filter(|c| c.kind == ComboKind::Single && c.cards[0] != PHOENIX).count();
    let phoenix_used = !with.iter().any(|c| c.cards == [PHOENIX]);
    if phoenix_used && singles(&loose) >= singles(&with) + 2 {
        return Decomposition { combos: with, loose_phoenix: false };
    }
    loose.push(Combo { kind: ComboKind::Single, cards: vec![PHOENIX], rank: 1.5, length: 1, phoenix_as: None });
    Decomposition { combos: loose, loose_phoenix: true }
}
