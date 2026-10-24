//! Combination identification, comparison, and legal-play generation.
//!
//! `identify_combo` / `can_beat` are a line-for-line port of
//! `shared/src/combinations.ts` in the tichu repo (the production engine), so
//! the two agree on every play; `tests/ts_diff` checks this against the real
//! TS code. `playable_combos` is new: unlike the TS `findPlayableCombos` (a UI
//! hint that picks one card per rank), it enumerates every distinct legal play,
//! including suit choices and both readings of an ambiguous Phoenix.

use std::collections::HashSet;

use crate::types::*;

// ---------------------------------------------------------------------------
// Single-card effective rank
// ---------------------------------------------------------------------------

/// Effective rank of a single card when led. Mah Jong = 1, Dragon = 15,
/// Dog = 0, Phoenix = 1.5 (when following, the engine sets it to the trick's
/// rank + 0.5).
pub fn single_card_rank(card: &Card) -> f32 {
    match card {
        Card::Normal { rank, .. } => *rank as f32,
        Card::Special(SpecialKind::MahJong) => 1.0,
        Card::Special(SpecialKind::Dragon) => 15.0,
        Card::Special(SpecialKind::Dog) => 0.0,
        Card::Special(SpecialKind::Phoenix) => 1.5,
    }
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------

fn combo(kind: ComboKind, cards: &[Card], rank: f32, phoenix_as: Option<u8>) -> Option<Combo> {
    let phoenix_as = if cards.len() > 1 && cards.contains(&PHOENIX) { phoenix_as } else { None };
    Some(Combo { kind, cards: cards.to_vec(), rank, length: cards.len(), phoenix_as })
}

fn is_consecutive(sorted: &[u8]) -> bool {
    sorted.windows(2).all(|w| w[1] == w[0] + 1)
}

/// (rank, count) pairs in ascending rank order.
fn frequencies(sorted_ranks: &[u8]) -> Vec<(u8, usize)> {
    let mut out: Vec<(u8, usize)> = Vec::new();
    for &r in sorted_ranks {
        match out.last_mut() {
            Some((lr, n)) if *lr == r => *n += 1,
            _ => out.push((r, 1)),
        }
    }
    out
}

/// Identify the combination formed by `cards`, or None if they don't form one.
///
/// `phoenix_as`: the rank the Phoenix stands for where the cards could be read
/// more than one way (a straight it could extend at either end, or a full
/// house of two pairs + Phoenix). Omitted, the TS default reading is used.
pub fn identify_combo(cards: &[Card], phoenix_as: Option<u8>) -> Option<Combo> {
    if cards.is_empty() {
        return None;
    }
    let has_phoenix = cards.contains(&PHOENIX);
    let has_dog = cards.contains(&DOG);
    let has_dragon = cards.contains(&DRAGON);
    let has_mahjong = cards.contains(&MAH_JONG);

    if has_dog {
        if cards.len() == 1 {
            return combo(ComboKind::Single, cards, 0.0, None);
        }
        return None;
    }
    if cards.len() == 1 {
        return combo(ComboKind::Single, cards, single_card_rank(&cards[0]), None);
    }
    if has_dragon {
        return None;
    }

    let mut ranks: Vec<u8> = cards.iter().filter_map(|c| c.rank()).collect();
    if has_mahjong {
        ranks.push(1);
    }
    ranks.sort_unstable();

    // The Mah Jong is a 1 only for straights.
    if has_mahjong {
        return try_straight(cards, has_phoenix, &ranks, phoenix_as);
    }

    try_four_of_a_kind(cards, has_phoenix, &ranks)
        .or_else(|| try_straight_flush(cards, has_phoenix))
        .or_else(|| try_pair(cards, has_phoenix, &ranks))
        .or_else(|| try_triple(cards, has_phoenix, &ranks))
        .or_else(|| try_full_house(cards, has_phoenix, &ranks, phoenix_as))
        .or_else(|| try_consecutive_pairs(cards, has_phoenix, &ranks))
        .or_else(|| try_straight(cards, has_phoenix, &ranks, phoenix_as))
}

fn try_pair(cards: &[Card], phoenix: bool, ranks: &[u8]) -> Option<Combo> {
    if cards.len() != 2 {
        return None;
    }
    if !phoenix {
        if ranks.len() == 2 && ranks[0] == ranks[1] {
            return combo(ComboKind::Pair, cards, ranks[0] as f32, None);
        }
        return None;
    }
    // The Phoenix pairs with the other card. (The TS engine takes the rank from
    // phoenixAs when given; a legitimate phoenixAs always equals ranks[0].)
    if ranks.len() == 1 {
        return combo(ComboKind::Pair, cards, ranks[0] as f32, Some(ranks[0]));
    }
    None
}

fn try_triple(cards: &[Card], phoenix: bool, ranks: &[u8]) -> Option<Combo> {
    if cards.len() != 3 {
        return None;
    }
    if !phoenix {
        if ranks.len() == 3 && ranks[0] == ranks[2] {
            return combo(ComboKind::Triple, cards, ranks[0] as f32, None);
        }
        return None;
    }
    if ranks.len() == 2 && ranks[0] == ranks[1] {
        return combo(ComboKind::Triple, cards, ranks[0] as f32, Some(ranks[0]));
    }
    None
}

fn try_full_house(cards: &[Card], phoenix: bool, ranks: &[u8], phoenix_as: Option<u8>) -> Option<Combo> {
    if cards.len() != 5 {
        return None;
    }
    let freq = frequencies(ranks);
    if freq.len() != 2 {
        return None;
    }
    let ((r1, c1), (r2, c2)) = (freq[0], freq[1]);
    if !phoenix {
        return match (c1, c2) {
            (3, 2) => combo(ComboKind::FullHouse, cards, r1 as f32, None),
            (2, 3) => combo(ComboKind::FullHouse, cards, r2 as f32, None),
            _ => None,
        };
    }
    match (c1, c2) {
        (3, 1) => combo(ComboKind::FullHouse, cards, r1 as f32, Some(r2)),
        (1, 3) => combo(ComboKind::FullHouse, cards, r2 as f32, Some(r1)),
        (2, 2) => {
            // Phoenix joins either pair; default (as TS) is the higher one.
            let triple = if phoenix_as == Some(r1) { r1 } else { r2 };
            combo(ComboKind::FullHouse, cards, triple as f32, Some(triple))
        }
        _ => None,
    }
}

fn try_consecutive_pairs(cards: &[Card], phoenix: bool, ranks: &[u8]) -> Option<Combo> {
    if cards.len() < 4 || cards.len() % 2 != 0 {
        return None;
    }
    let num_pairs = cards.len() / 2;
    let freq = frequencies(ranks);
    if freq.len() != num_pairs {
        return None;
    }
    let distinct: Vec<u8> = freq.iter().map(|(r, _)| *r).collect();
    if !is_consecutive(&distinct) {
        return None;
    }
    let top = *distinct.last().unwrap() as f32;
    if !phoenix {
        if freq.iter().all(|(_, c)| *c == 2) {
            return combo(ComboKind::ConsecutivePairs, cards, top, None);
        }
        return None;
    }
    let ones: Vec<u8> = freq.iter().filter(|(_, c)| *c == 1).map(|(r, _)| *r).collect();
    let twos = freq.iter().filter(|(_, c)| *c == 2).count();
    if ones.len() == 1 && twos == num_pairs - 1 {
        return combo(ComboKind::ConsecutivePairs, cards, top, Some(ones[0]));
    }
    None
}

fn try_straight(cards: &[Card], phoenix: bool, ranks: &[u8], phoenix_as: Option<u8>) -> Option<Combo> {
    if cards.len() < 5 {
        return None;
    }
    let mut unique = ranks.to_vec();
    unique.dedup();
    let top = *unique.last().unwrap();

    if !phoenix {
        if unique.len() != cards.len() || !is_consecutive(&unique) {
            return None;
        }
        return combo(ComboKind::Straight, cards, top as f32, None);
    }

    if unique.len() != cards.len() - 1 {
        return None;
    }
    if is_consecutive(&unique) {
        // The Phoenix extends the run: high by default when possible, low
        // when asked (or when the run already reaches the Ace).
        let low = unique[0] as i32 - 1;
        let low_asked = phoenix_as.map(|p| p as i32) == Some(low) && low >= 2;
        let (rank, stands_for) = if top < 14 && !low_asked {
            (top + 1, top + 1)
        } else {
            (top, low.max(0) as u8)
        };
        return combo(ComboKind::Straight, cards, rank as f32, Some(stands_for));
    }
    // Exactly one gap of one rank, which the Phoenix fills.
    let gaps: Vec<(u8, u8)> = unique.windows(2).filter(|w| w[1] - w[0] > 1).map(|w| (w[0], w[1])).collect();
    if gaps.len() == 1 && gaps[0].1 - gaps[0].0 == 2 {
        return combo(ComboKind::Straight, cards, top as f32, Some(gaps[0].0 + 1));
    }
    None
}

fn try_four_of_a_kind(cards: &[Card], phoenix: bool, ranks: &[u8]) -> Option<Combo> {
    if cards.len() != 4 || phoenix || ranks.len() != 4 {
        return None;
    }
    if cards.iter().any(|c| c.is_special()) || ranks[0] != ranks[3] {
        return None;
    }
    combo(ComboKind::FourOfAKindBomb, cards, ranks[0] as f32, None)
}

fn try_straight_flush(cards: &[Card], phoenix: bool) -> Option<Combo> {
    if cards.len() < 5 || phoenix || cards.iter().any(|c| c.is_special()) {
        return None;
    }
    let suit = cards[0].suit();
    if !cards.iter().all(|c| c.suit() == suit) {
        return None;
    }
    let mut ranks: Vec<u8> = cards.iter().filter_map(|c| c.rank()).collect();
    ranks.sort_unstable();
    if !is_consecutive(&ranks) {
        return None;
    }
    combo(ComboKind::StraightFlushBomb, cards, *ranks.last().unwrap() as f32, None)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Can `play` legally be played on top of `current`?
pub fn can_beat(current: &Combo, play: &Combo) -> bool {
    if play.is_bomb() {
        if !current.is_bomb() {
            return true;
        }
        return compare_bombs(current, play) > 0.0;
    }
    if current.is_bomb() {
        return false;
    }
    if current.kind != play.kind || current.length != play.length {
        return false;
    }
    // Phoenix as a single beats everything except the Dragon.
    if play.kind == ComboKind::Single && play.cards[0] == PHOENIX {
        return !current.contains(SpecialKind::Dragon);
    }
    play.rank > current.rank
}

pub fn is_bomb(combo: &Combo) -> bool {
    combo.is_bomb()
}

/// > 0 if `play` beats `current` (both bombs).
fn compare_bombs(current: &Combo, play: &Combo) -> f32 {
    use ComboKind::*;
    match (current.kind, play.kind) {
        (FourOfAKindBomb, StraightFlushBomb) => 1.0,
        (StraightFlushBomb, FourOfAKindBomb) => -1.0,
        (StraightFlushBomb, StraightFlushBomb) if play.length != current.length => {
            play.length as f32 - current.length as f32
        }
        _ => play.rank - current.rank,
    }
}

// ---------------------------------------------------------------------------
// Legal-play generation
// ---------------------------------------------------------------------------

/// All k-element subsets of `items` (k <= 4 in practice).
fn choose<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    fn rec<T: Copy>(items: &[T], k: usize, start: usize, cur: &mut Vec<T>, out: &mut Vec<Vec<T>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..items.len() {
            if items.len() - i < k - cur.len() {
                break;
            }
            cur.push(items[i]);
            rec(items, k, i + 1, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(items, k, 0, &mut Vec::new(), &mut out);
    out
}

struct Gen<'a> {
    current: Option<&'a Combo>,
    seen: HashSet<(u64, i32)>,
    out: Vec<Combo>,
}

impl<'a> Gen<'a> {
    fn push(&mut self, cards: &[Card], phoenix_as: Option<u8>) {
        let Some(mut c) = identify_combo(cards, phoenix_as) else { return };
        if c.kind == ComboKind::Single && c.cards[0] == PHOENIX {
            if let Some(cur) = self.current {
                c.rank = cur.rank + 0.5;
            }
        }
        if let Some(cur) = self.current {
            if !can_beat(cur, &c) {
                return;
            }
        }
        if self.seen.insert((c.card_set().0, (c.rank * 2.0) as i32)) {
            self.out.push(c);
        }
    }
}

/// Every distinct combination in `hand` that may be played on `current`
/// (any combination when `current` is None, i.e. when leading). Includes
/// bombs and, when leading, the Dog. Does **not** apply the Mah Jong wish;
/// see `GameState::legal_plays` for the full turn rules.
pub fn playable_combos(hand: &[Card], current: Option<&Combo>) -> Vec<Combo> {
    let mut by_rank: [Vec<Card>; 15] = Default::default();
    for c in hand {
        if let Some(r) = c.rank() {
            by_rank[r as usize].push(*c);
        }
    }
    let phoenix = hand.contains(&PHOENIX);
    let mahjong = hand.contains(&MAH_JONG);
    let mut g = Gen { current, seen: HashSet::new(), out: Vec::new() };

    let want = |k: ComboKind| match current {
        None => true,
        Some(c) => !c.is_bomb() && c.kind == k,
    };

    if want(ComboKind::Single) {
        for c in hand {
            g.push(&[*c], None);
        }
    }

    if want(ComboKind::Pair) {
        for r in 2..=14usize {
            for p in choose(&by_rank[r], 2) {
                g.push(&p, None);
            }
            if phoenix {
                for c in &by_rank[r] {
                    g.push(&[*c, PHOENIX], Some(r as u8));
                }
            }
        }
    }

    if want(ComboKind::Triple) {
        for r in 2..=14usize {
            for t in choose(&by_rank[r], 3) {
                g.push(&t, None);
            }
            if phoenix {
                for p in choose(&by_rank[r], 2) {
                    g.push(&[p[0], p[1], PHOENIX], Some(r as u8));
                }
            }
        }
    }

    if want(ComboKind::FullHouse) {
        for t in 2..=14usize {
            // (cards, uses phoenix)
            let mut triples: Vec<(Vec<Card>, bool)> = choose(&by_rank[t], 3).into_iter().map(|v| (v, false)).collect();
            if phoenix {
                for mut p in choose(&by_rank[t], 2) {
                    p.push(PHOENIX);
                    triples.push((p, true));
                }
            }
            if triples.is_empty() {
                continue;
            }
            for p in 2..=14usize {
                if p == t || by_rank[p].is_empty() {
                    continue;
                }
                let pairs2 = choose(&by_rank[p], 2);
                for (tc, tp) in &triples {
                    for pc in &pairs2 {
                        let mut cards = tc.clone();
                        cards.extend_from_slice(pc);
                        g.push(&cards, if *tp { Some(t as u8) } else { None });
                    }
                    if phoenix && !tp {
                        for c in &by_rank[p] {
                            let mut cards = tc.clone();
                            cards.push(*c);
                            cards.push(PHOENIX);
                            g.push(&cards, Some(p as u8));
                        }
                    }
                }
            }
        }
    }

    if want(ComboKind::ConsecutivePairs) {
        let (min_pairs, max_pairs) = match current {
            Some(c) => (c.length / 2, c.length / 2),
            None => (2, 7),
        };
        for lo in 2..=14usize {
            for k in min_pairs..=max_pairs {
                let hi = lo + k - 1;
                if hi > 14 {
                    break;
                }
                consecutive_pair_variants(&by_rank, lo, hi, phoenix, &mut g);
            }
        }
    }

    if want(ComboKind::Straight) {
        let (min_len, max_len) = match current {
            Some(c) => (c.length, c.length),
            None => (5, 14),
        };
        for lo in 1..=14usize {
            for len in min_len..=max_len {
                let hi = lo + len - 1;
                if hi > 14 {
                    break;
                }
                straight_variants(&by_rank, mahjong, lo, hi, phoenix, &mut g);
            }
        }
    }

    // Bombs are always worth generating: they beat any non-bomb.
    for r in 2..=14usize {
        if by_rank[r].len() == 4 {
            let cards = by_rank[r].clone();
            g.push(&cards, None);
        }
    }
    for suit in Suit::ALL {
        let has = |r: usize| hand.contains(&Card::Normal { suit, rank: r as u8 });
        for lo in 2..=10usize {
            if !has(lo) {
                continue;
            }
            let mut hi = lo;
            while hi < 14 && has(hi + 1) {
                hi += 1;
                if hi - lo + 1 >= 5 {
                    let cards: Vec<Card> = (lo..=hi).map(|r| Card::Normal { suit, rank: r as u8 }).collect();
                    g.push(&cards, None);
                }
            }
        }
    }

    g.out
}

/// Cartesian product of per-slot card choices, calling `f` with each pick.
fn product(slots: &[Vec<Vec<Card>>], f: &mut dyn FnMut(&[Card])) {
    fn rec(slots: &[Vec<Vec<Card>>], i: usize, cur: &mut Vec<Card>, f: &mut dyn FnMut(&[Card])) {
        if i == slots.len() {
            f(cur);
            return;
        }
        for opt in &slots[i] {
            let n = cur.len();
            cur.extend_from_slice(opt);
            rec(slots, i + 1, cur, f);
            cur.truncate(n);
        }
    }
    rec(slots, 0, &mut Vec::new(), f);
}

fn consecutive_pair_variants(by_rank: &[Vec<Card>; 15], lo: usize, hi: usize, phoenix: bool, g: &mut Gen) {
    let natural: Vec<Vec<Vec<Card>>> = (lo..=hi).map(|r| choose(&by_rank[r], 2)).collect();
    let short: Vec<usize> = (lo..=hi).filter(|&r| by_rank[r].len() < 2).collect();
    if short.is_empty() {
        product(&natural, &mut |cards| g.push(cards, None));
    }
    if !phoenix || short.len() > 1 {
        return;
    }
    // The Phoenix completes one rank's pair (one that's short, or any rank).
    let candidates: Vec<usize> = if short.len() == 1 { short.clone() } else { (lo..=hi).collect() };
    for pr in candidates {
        if by_rank[pr].is_empty() {
            continue;
        }
        let mut slots = natural.clone();
        slots[pr - lo] = by_rank[pr].iter().map(|c| vec![*c, PHOENIX]).collect();
        product(&slots, &mut |cards| g.push(cards, Some(pr as u8)));
    }
}

fn straight_variants(by_rank: &[Vec<Card>; 15], mahjong: bool, lo: usize, hi: usize, phoenix: bool, g: &mut Gen) {
    let options = |r: usize| -> Vec<Vec<Card>> {
        if r == 1 {
            if mahjong { vec![vec![MAH_JONG]] } else { vec![] }
        } else {
            by_rank[r].iter().map(|c| vec![*c]).collect()
        }
    };
    let slots: Vec<Vec<Vec<Card>>> = (lo..=hi).map(options).collect();
    let missing: Vec<usize> = (lo..=hi).filter(|&r| slots[r - lo].is_empty()).collect();
    if missing.is_empty() {
        product(&slots, &mut |cards| g.push(cards, None));
    }
    if !phoenix || missing.len() > 1 {
        return;
    }
    // The Phoenix stands in for one rank >= 2 (a missing one, or any).
    let candidates: Vec<usize> = if missing.len() == 1 { missing.clone() } else { (lo.max(2)..=hi).collect() };
    for pr in candidates {
        if pr < 2 {
            continue;
        }
        let mut s = slots.clone();
        s[pr - lo] = vec![vec![PHOENIX]];
        product(&s, &mut |cards| g.push(cards, Some(pr as u8)));
    }
}

/// Can `hand` make any legal play on `current` that includes a normal card of
/// rank `wish`? (Mirrors TS `canPlayWishedRankFromHand`; bombs count.)
pub fn can_fulfill_wish(hand: &[Card], wish: Rank, current: Option<&Combo>) -> bool {
    if !hand.iter().any(|c| c.rank() == Some(wish)) {
        return false;
    }
    match current {
        None => true,
        Some(_) => playable_combos(hand, current).iter().any(|c| c.contains_rank(wish)),
    }
}
