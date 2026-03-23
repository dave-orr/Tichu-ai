use std::collections::HashMap;

use crate::types::*;

// ---------------------------------------------------------------------------
// Single-card effective rank
// ---------------------------------------------------------------------------

/// Effective rank of a single card.
/// Phoenix = last_played_rank + 0.5  (or 1.5 when leading).
/// MahJong = 1, Dragon = 15, Dog = -1.
pub fn single_card_rank(card: &Card, last_played_rank: Option<f32>) -> f32 {
    match card {
        Card::Normal { rank, .. } => *rank as f32,
        Card::Special(SpecialKind::MahJong) => 1.0,
        Card::Special(SpecialKind::Dragon) => 15.0,
        Card::Special(SpecialKind::Dog) => -1.0,
        Card::Special(SpecialKind::Phoenix) => match last_played_rank {
            Some(r) => r + 0.5,
            None => 1.5,
        },
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normal_rank(card: &Card) -> Option<u8> {
    match card {
        Card::Normal { rank, .. } => Some(*rank),
        _ => None,
    }
}

fn has_phoenix(cards: &[Card]) -> bool {
    cards
        .iter()
        .any(|c| matches!(c, Card::Special(SpecialKind::Phoenix)))
}

fn has_dragon(cards: &[Card]) -> bool {
    cards
        .iter()
        .any(|c| matches!(c, Card::Special(SpecialKind::Dragon)))
}

fn has_dog(cards: &[Card]) -> bool {
    cards
        .iter()
        .any(|c| matches!(c, Card::Special(SpecialKind::Dog)))
}

fn has_mahjong(cards: &[Card]) -> bool {
    cards
        .iter()
        .any(|c| matches!(c, Card::Special(SpecialKind::MahJong)))
}

/// Group normal cards by rank, returning (rank -> count) and the list of
/// normal ranks sorted ascending.
fn rank_groups(cards: &[Card]) -> (HashMap<u8, usize>, Vec<u8>) {
    let mut map: HashMap<u8, usize> = HashMap::new();
    for c in cards {
        if let Some(r) = normal_rank(c) {
            *map.entry(r).or_insert(0) += 1;
        }
    }
    let mut ranks: Vec<u8> = map.keys().cloned().collect();
    ranks.sort();
    (map, ranks)
}

// ---------------------------------------------------------------------------
// identifyCombo  (matches TS logic)
// ---------------------------------------------------------------------------

/// Identify the combination type for the given set of cards.
/// `phoenix_as` can force the Phoenix to act as a specific rank for full
/// house / consecutive pair disambiguation (mirrors TS `phoenixAs`).
///
/// Returns `None` if the cards do not form a valid combination.
pub fn identify_combo(cards: &[Card], phoenix_as: Option<u8>) -> Option<Combo> {
    let n = cards.len();
    if n == 0 {
        return None;
    }

    let phoenix = has_phoenix(cards);
    let dragon = has_dragon(cards);
    let dog = has_dog(cards);
    let mahjong = has_mahjong(cards);

    // --- Dog (alone) ---
    if n == 1 && dog {
        return Some(Combo {
            kind: ComboKind::Single,
            cards: cards.to_vec(),
            rank: -1.0,
            length: 1,
        });
    }

    // --- Single ---
    if n == 1 {
        let r = single_card_rank(&cards[0], None);
        return Some(Combo {
            kind: ComboKind::Single,
            cards: cards.to_vec(),
            rank: r,
            length: 1,
        });
    }

    // Dragon is only valid as a single.
    if dragon {
        return None;
    }
    // Dog is only valid alone.
    if dog {
        return None;
    }

    let (rank_map, sorted_ranks) = rank_groups(cards);
    let normal_count: usize = rank_map.values().sum();

    // --- Four-of-a-kind bomb (no phoenix, no specials) ---
    if n == 4 && !phoenix && !mahjong && sorted_ranks.len() == 1 && rank_map[&sorted_ranks[0]] == 4
    {
        return Some(Combo {
            kind: ComboKind::FourOfAKindBomb,
            cards: cards.to_vec(),
            rank: sorted_ranks[0] as f32,
            length: 4,
        });
    }

    // --- Straight flush bomb (5+, all same suit, consecutive, no phoenix, no mahjong) ---
    if n >= 5 && !phoenix && !mahjong {
        let first_suit = cards[0].suit();
        if let Some(s) = first_suit {
            let all_same_suit = cards.iter().all(|c| c.suit() == Some(s));
            if all_same_suit && sorted_ranks.len() == n {
                let is_consecutive = sorted_ranks
                    .windows(2)
                    .all(|w| w[1] == w[0] + 1);
                if is_consecutive {
                    return Some(Combo {
                        kind: ComboKind::StraightFlushBomb,
                        cards: cards.to_vec(),
                        rank: *sorted_ranks.last().unwrap() as f32,
                        length: n,
                    });
                }
            }
        }
    }

    // --- Pair ---
    if n == 2 {
        if phoenix && normal_count == 1 {
            return Some(Combo {
                kind: ComboKind::Pair,
                cards: cards.to_vec(),
                rank: sorted_ranks[0] as f32,
                length: 2,
            });
        }
        if sorted_ranks.len() == 1 && rank_map[&sorted_ranks[0]] == 2 {
            return Some(Combo {
                kind: ComboKind::Pair,
                cards: cards.to_vec(),
                rank: sorted_ranks[0] as f32,
                length: 2,
            });
        }
        return None;
    }

    // --- Triple ---
    if n == 3 {
        if sorted_ranks.len() == 1 {
            let count = rank_map[&sorted_ranks[0]];
            if count == 3 || (count == 2 && phoenix) {
                return Some(Combo {
                    kind: ComboKind::Triple,
                    cards: cards.to_vec(),
                    rank: sorted_ranks[0] as f32,
                    length: 3,
                });
            }
        }
        return None;
    }

    // --- Full house (5 cards: triple + pair) ---
    if n == 5 {
        if let Some(fh) = try_full_house(cards, &rank_map, &sorted_ranks, phoenix, phoenix_as) {
            return Some(fh);
        }
    }

    // --- Consecutive pairs (even count >= 4) ---
    if n >= 4 && n % 2 == 0 {
        if let Some(cp) =
            try_consecutive_pairs(cards, &rank_map, &sorted_ranks, phoenix, mahjong, phoenix_as)
        {
            return Some(cp);
        }
    }

    // --- Straight (5+ consecutive, possibly with mahjong as rank 1 and/or phoenix as wildcard) ---
    if n >= 5 {
        if let Some(st) = try_straight(cards, &rank_map, &sorted_ranks, phoenix, mahjong) {
            return Some(st);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Full house helper
// ---------------------------------------------------------------------------

fn try_full_house(
    cards: &[Card],
    rank_map: &HashMap<u8, usize>,
    sorted_ranks: &[u8],
    phoenix: bool,
    phoenix_as: Option<u8>,
) -> Option<Combo> {
    // Without phoenix: need exactly two distinct ranks, one with 3 and one with 2.
    if !phoenix {
        if sorted_ranks.len() != 2 {
            return None;
        }
        let (r0, r1) = (sorted_ranks[0], sorted_ranks[1]);
        let (c0, c1) = (rank_map[&r0], rank_map[&r1]);
        if (c0 == 3 && c1 == 2) || (c0 == 2 && c1 == 3) {
            let triple_rank = if c0 == 3 { r0 } else { r1 };
            return Some(Combo {
                kind: ComboKind::FullHouse,
                cards: cards.to_vec(),
                rank: triple_rank as f32,
                length: 5,
            });
        }
        return None;
    }

    // With phoenix: 4 normal cards + phoenix.
    // Possibilities: (3+1) phoenix completes pair, or (2+2) phoenix completes one triple.
    let normal_count: usize = rank_map.values().sum();
    if normal_count != 4 {
        return None;
    }

    match sorted_ranks.len() {
        1 => {
            // 4 of same rank + phoenix -> not a full house (would be bomb candidate, but phoenix
            // disqualifies). Actually this is 4+phoenix which doesn't make a FH.
            None
        }
        2 => {
            let (r0, r1) = (sorted_ranks[0], sorted_ranks[1]);
            let (c0, c1) = (rank_map[&r0], rank_map[&r1]);

            if c0 == 3 && c1 == 1 {
                // Phoenix becomes pair with r1. Triple = r0.
                return Some(Combo {
                    kind: ComboKind::FullHouse,
                    cards: cards.to_vec(),
                    rank: r0 as f32,
                    length: 5,
                });
            }
            if c0 == 1 && c1 == 3 {
                // Phoenix becomes pair with r0. Triple = r1.
                return Some(Combo {
                    kind: ComboKind::FullHouse,
                    cards: cards.to_vec(),
                    rank: r1 as f32,
                    length: 5,
                });
            }
            if c0 == 2 && c1 == 2 {
                // Phoenix can complete either side to a triple.
                // Use phoenix_as hint if provided.
                let triple_rank = if let Some(pa) = phoenix_as {
                    if pa == r0 {
                        r0
                    } else {
                        r1
                    }
                } else {
                    // Default: higher rank is triple.
                    r1
                };
                return Some(Combo {
                    kind: ComboKind::FullHouse,
                    cards: cards.to_vec(),
                    rank: triple_rank as f32,
                    length: 5,
                });
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Consecutive pairs helper
// ---------------------------------------------------------------------------

fn try_consecutive_pairs(
    cards: &[Card],
    rank_map: &HashMap<u8, usize>,
    sorted_ranks: &[u8],
    phoenix: bool,
    _mahjong: bool,
    _phoenix_as: Option<u8>,
) -> Option<Combo> {
    let pair_count = cards.len() / 2;

    if !phoenix {
        // Every rank must appear exactly twice and ranks must be consecutive.
        if sorted_ranks.len() != pair_count {
            return None;
        }
        if !rank_map.values().all(|&c| c == 2) {
            return None;
        }
        let consecutive = sorted_ranks.windows(2).all(|w| w[1] == w[0] + 1);
        if !consecutive {
            return None;
        }
        return Some(Combo {
            kind: ComboKind::ConsecutivePairs,
            cards: cards.to_vec(),
            rank: *sorted_ranks.last().unwrap() as f32,
            length: cards.len(),
        });
    }

    // With phoenix: one rank has count 1, the rest have count 2.
    // Phoenix fills the gap. The single could extend or fill a hole.
    let normal_count: usize = rank_map.values().sum();
    if normal_count + 1 != cards.len() {
        return None;
    }

    let singles: Vec<u8> = sorted_ranks
        .iter()
        .filter(|r| rank_map[r] == 1)
        .cloned()
        .collect();
    let pairs: Vec<u8> = sorted_ranks
        .iter()
        .filter(|r| rank_map[r] == 2)
        .cloned()
        .collect();

    if singles.len() > 1 {
        return None;
    }

    // Build the sequence of ranks we need (pair_count consecutive ranks).
    // The phoenix completes one pair. Try all possible sequences that include all our ranks.

    let mut all_ranks: Vec<u8> = sorted_ranks.to_vec();
    // If singles is empty, phoenix adds a new rank at either end.
    if singles.is_empty() {
        // All are pairs already, phoenix extends by one rank.
        if pairs.len() != pair_count - 1 {
            return None;
        }
        // Try extending low or high.
        let lo = sorted_ranks[0];
        let hi = *sorted_ranks.last().unwrap();
        if lo > 2 {
            let mut test = vec![lo - 1];
            test.extend_from_slice(sorted_ranks);
            if test.windows(2).all(|w| w[1] == w[0] + 1) && test.len() == pair_count {
                return Some(Combo {
                    kind: ComboKind::ConsecutivePairs,
                    cards: cards.to_vec(),
                    rank: *test.last().unwrap() as f32,
                    length: cards.len(),
                });
            }
        }
        if hi < 14 {
            let mut test: Vec<u8> = sorted_ranks.to_vec();
            test.push(hi + 1);
            if test.windows(2).all(|w| w[1] == w[0] + 1) && test.len() == pair_count {
                return Some(Combo {
                    kind: ComboKind::ConsecutivePairs,
                    cards: cards.to_vec(),
                    rank: *test.last().unwrap() as f32,
                    length: cards.len(),
                });
            }
        }
        return None;
    }

    // Exactly one single rank. Phoenix pairs with it.
    // All ranks (including single) must form pair_count consecutive values.
    all_ranks.sort();
    if all_ranks.len() != pair_count {
        // Check if phoenix is filling a gap rank (not present in hand).
        // We need pair_count consecutive ranks. We have pair_count-1 distinct ranks (some pairs, one single).
        // Phoenix adds a card to the single to make a pair, but we still need pair_count ranks total.
        // If we only have pair_count-1 ranks, phoenix might also be filling a missing rank.
        // Actually phoenix is only one card, so it can only help once.

        // We have (pairs.len() pairs) + 1 single = pairs.len()+1 distinct ranks, need pair_count.
        // If pairs.len()+1 < pair_count, can't work (phoenix only fills one spot).
        return None;
    }

    // Check consecutive.
    if all_ranks.windows(2).all(|w| w[1] == w[0] + 1) {
        return Some(Combo {
            kind: ComboKind::ConsecutivePairs,
            cards: cards.to_vec(),
            rank: *all_ranks.last().unwrap() as f32,
            length: cards.len(),
        });
    }

    // Maybe phoenix fills a gap (one missing rank in the sequence).
    // We have pair_count ranks but they're not consecutive.
    // Phoenix can't help here because it's already used to complete the single's pair.
    None
}

// ---------------------------------------------------------------------------
// Straight helper
// ---------------------------------------------------------------------------

fn try_straight(
    cards: &[Card],
    rank_map: &HashMap<u8, usize>,
    sorted_ranks: &[u8],
    phoenix: bool,
    mahjong: bool,
) -> Option<Combo> {
    let n = cards.len();

    // Build the effective list of ranks (including mahjong as 1).
    let mut ranks: Vec<u8> = sorted_ranks.to_vec();
    if mahjong {
        if !ranks.contains(&1) {
            ranks.push(1);
            ranks.sort();
        }
    }

    // Each rank should appear exactly once among normal cards.
    // (MahJong contributes rank 1.)
    let total_normal = if mahjong {
        rank_map.values().sum::<usize>() + 1 // +1 for mahjong
    } else {
        rank_map.values().sum::<usize>()
    };

    if !phoenix {
        // All cards must be normals (+ possibly mahjong). No duplicate ranks.
        if ranks.len() != n {
            return None;
        }
        if total_normal != n {
            return None;
        }
        // Check consecutive.
        if ranks.windows(2).all(|w| w[1] == w[0] + 1) {
            return Some(Combo {
                kind: ComboKind::Straight,
                cards: cards.to_vec(),
                rank: *ranks.last().unwrap() as f32,
                length: n,
            });
        }
        return None;
    }

    // With phoenix: one wildcard can fill a gap or extend.
    // Normal cards + mahjong should give n-1 ranks (phoenix is the n-th card).
    let distinct_normal_ranks = ranks.len();
    // Check no rank has count > 1 among normals.
    if rank_map.values().any(|&c| c > 1) {
        return None;
    }

    if distinct_normal_ranks == n - 1 {
        // Phoenix fills one gap or extends one end.
        // Check if already consecutive (phoenix extends).
        if ranks.windows(2).all(|w| w[1] == w[0] + 1) {
            // Phoenix extends high end.
            let top = *ranks.last().unwrap();
            if top < 14 {
                return Some(Combo {
                    kind: ComboKind::Straight,
                    cards: cards.to_vec(),
                    rank: (top + 1) as f32,
                    length: n,
                });
            }
            // Or low end... but rank is based on highest, so let's try high first.
            // If top == 14, phoenix extends low.
            return Some(Combo {
                kind: ComboKind::Straight,
                cards: cards.to_vec(),
                rank: top as f32,
                length: n,
            });
        }

        // Check for exactly one gap of size 2 (phoenix fills it).
        let mut gap_count = 0;
        let max_rank = *ranks.last().unwrap();
        for w in ranks.windows(2) {
            let diff = w[1] - w[0];
            if diff == 2 {
                gap_count += 1;
            } else if diff != 1 {
                return None; // gap too large
            }
        }
        if gap_count == 1 {
            return Some(Combo {
                kind: ComboKind::Straight,
                cards: cards.to_vec(),
                rank: max_rank as f32,
                length: n,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// canBeat
// ---------------------------------------------------------------------------

fn is_bomb(kind: ComboKind) -> bool {
    matches!(
        kind,
        ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb
    )
}

/// Can `play` beat `current`?
pub fn can_beat(current: &Combo, play: &Combo) -> bool {
    let cur_bomb = is_bomb(current.kind);
    let play_bomb = is_bomb(play.kind);

    if play_bomb && !cur_bomb {
        return true; // bomb beats non-bomb
    }
    if !play_bomb && cur_bomb {
        return false; // non-bomb can't beat bomb
    }

    if play_bomb && cur_bomb {
        return compare_bombs(current, play);
    }

    // Same type, same length, higher rank.
    if play.kind != current.kind {
        return false;
    }
    if play.length != current.length {
        return false;
    }
    play.rank > current.rank
}

fn compare_bombs(current: &Combo, play: &Combo) -> bool {
    match (current.kind, play.kind) {
        (ComboKind::FourOfAKindBomb, ComboKind::StraightFlushBomb) => true,
        (ComboKind::StraightFlushBomb, ComboKind::FourOfAKindBomb) => false,
        (ComboKind::StraightFlushBomb, ComboKind::StraightFlushBomb) => {
            if play.length != current.length {
                play.length > current.length
            } else {
                play.rank > current.rank
            }
        }
        (ComboKind::FourOfAKindBomb, ComboKind::FourOfAKindBomb) => play.rank > current.rank,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// findPlayableCombos – enumerate all legal plays from a hand
// ---------------------------------------------------------------------------

/// Find all valid combinations from `hand` that can beat `current_trick`
/// (or all leading combos if `current_trick` is `None`).
pub fn find_playable_combos(hand: &[Card], current_trick: Option<&Combo>) -> Vec<Combo> {
    let mut results = Vec::new();
    let n = hand.len();

    // Generate all subsets up to size n (but practically we only need 1..=n).
    // For performance we enumerate by expected sizes.
    // Singles
    for i in 0..n {
        if let Some(combo) = identify_combo(&[hand[i]], None) {
            if let Some(ct) = current_trick {
                if can_beat(ct, &combo) {
                    results.push(combo);
                }
            } else {
                results.push(combo);
            }
        }
    }

    // Pairs
    for i in 0..n {
        for j in (i + 1)..n {
            let cards = vec![hand[i], hand[j]];
            if let Some(combo) = identify_combo(&cards, None) {
                if combo.kind == ComboKind::Pair {
                    if let Some(ct) = current_trick {
                        if can_beat(ct, &combo) {
                            results.push(combo);
                        }
                    } else {
                        results.push(combo);
                    }
                }
            }
        }
    }

    // Triples
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let cards = vec![hand[i], hand[j], hand[k]];
                if let Some(combo) = identify_combo(&cards, None) {
                    if combo.kind == ComboKind::Triple {
                        if let Some(ct) = current_trick {
                            if can_beat(ct, &combo) {
                                results.push(combo);
                            }
                        } else {
                            results.push(combo);
                        }
                    }
                }
            }
        }
    }

    // For larger combos (4+ cards), use smarter enumeration.
    // Group cards by rank for efficient combo building.
    find_larger_combos(hand, current_trick, &mut results);

    results
}

/// Enumerate bombs, full houses, straights, consecutive pairs.
fn find_larger_combos(hand: &[Card], current_trick: Option<&Combo>, results: &mut Vec<Combo>) {
    let phoenix = hand
        .iter()
        .any(|c| matches!(c, Card::Special(SpecialKind::Phoenix)));
    let mahjong = hand
        .iter()
        .any(|c| matches!(c, Card::Special(SpecialKind::MahJong)));

    // Group normal cards by rank.
    let mut by_rank: HashMap<u8, Vec<Card>> = HashMap::new();
    for &c in hand {
        if let Card::Normal { rank, .. } = c {
            by_rank.entry(rank).or_default().push(c);
        }
    }

    let mut all_ranks: Vec<u8> = by_rank.keys().cloned().collect();
    all_ranks.sort();

    // Four-of-a-kind bombs
    for &r in &all_ranks {
        if by_rank[&r].len() == 4 {
            let cards = by_rank[&r].clone();
            if let Some(combo) = identify_combo(&cards, None) {
                if let Some(ct) = current_trick {
                    if can_beat(ct, &combo) {
                        results.push(combo);
                    }
                } else {
                    results.push(combo);
                }
            }
        }
    }

    // Straight flush bombs: find consecutive same-suit runs of 5+
    for &suit in &Suit::ALL {
        let mut suit_ranks: Vec<u8> = Vec::new();
        let mut suit_cards: HashMap<u8, Card> = HashMap::new();
        for &c in hand {
            if let Card::Normal { suit: s, rank } = c {
                if s == suit {
                    suit_ranks.push(rank);
                    suit_cards.insert(rank, c);
                }
            }
        }
        suit_ranks.sort();
        suit_ranks.dedup();

        // Find all consecutive runs of length >= 5
        if suit_ranks.len() >= 5 {
            for start in 0..suit_ranks.len() {
                let mut end = start;
                while end + 1 < suit_ranks.len()
                    && suit_ranks[end + 1] == suit_ranks[end] + 1
                {
                    end += 1;
                }
                let run_len = end - start + 1;
                if run_len >= 5 {
                    // Generate all sub-runs of length 5..=run_len
                    for len in 5..=run_len {
                        for s in start..=(end + 1 - len) {
                            let cards: Vec<Card> = (s..s + len)
                                .map(|i| suit_cards[&suit_ranks[i]])
                                .collect();
                            if let Some(combo) = identify_combo(&cards, None) {
                                if is_bomb(combo.kind) {
                                    if let Some(ct) = current_trick {
                                        if can_beat(ct, &combo) {
                                            results.push(combo);
                                        }
                                    } else {
                                        results.push(combo);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Full houses: pick a triple rank and a pair rank.
    let triple_ranks: Vec<u8> = all_ranks
        .iter()
        .filter(|r| by_rank[r].len() >= 3)
        .cloned()
        .collect();
    let pair_ranks: Vec<u8> = all_ranks
        .iter()
        .filter(|r| by_rank[r].len() >= 2)
        .cloned()
        .collect();

    for &tr in &triple_ranks {
        for &pr in &pair_ranks {
            if tr == pr {
                continue;
            }
            let mut cards: Vec<Card> = by_rank[&tr][..3].to_vec();
            cards.extend_from_slice(&by_rank[&pr][..2]);
            if let Some(combo) = identify_combo(&cards, Some(tr)) {
                if combo.kind == ComboKind::FullHouse {
                    check_and_push(combo, current_trick, results);
                }
            }
        }
    }

    // Phoenix full houses: phoenix completes a pair to triple, or a single to pair.
    if phoenix {
        let phoenix_card = Card::Special(SpecialKind::Phoenix);
        // Phoenix as third in triple: need a pair rank + another pair rank.
        for &pr in &pair_ranks {
            // Phoenix makes pr a triple. Need another pair.
            for &pr2 in &pair_ranks {
                if pr == pr2 {
                    continue;
                }
                let mut cards: Vec<Card> = by_rank[&pr][..2].to_vec();
                cards.push(phoenix_card);
                cards.extend_from_slice(&by_rank[&pr2][..2]);
                if let Some(combo) = identify_combo(&cards, Some(pr)) {
                    if combo.kind == ComboKind::FullHouse {
                        check_and_push(combo, current_trick, results);
                    }
                }
            }
            // Phoenix as pair with a single: triple rank (3) + phoenix + single
            for &tr in &triple_ranks {
                if tr == pr {
                    continue;
                }
                // Already covered in non-phoenix case.
            }
        }
        // Phoenix pairs with a single to form the pair part of FH.
        for &tr in &triple_ranks {
            for &sr in &all_ranks {
                if sr == tr {
                    continue;
                }
                if by_rank[&sr].len() >= 1 {
                    let mut cards: Vec<Card> = by_rank[&tr][..3].to_vec();
                    cards.push(by_rank[&sr][0]);
                    cards.push(phoenix_card);
                    if let Some(combo) = identify_combo(&cards, Some(tr)) {
                        if combo.kind == ComboKind::FullHouse {
                            check_and_push(combo, current_trick, results);
                        }
                    }
                }
            }
        }
    }

    // Consecutive pairs: find runs of consecutive ranks each with count >= 2.
    find_consecutive_pair_combos(hand, &by_rank, &all_ranks, phoenix, current_trick, results);

    // Straights: find runs of 5+ consecutive ranks.
    find_straight_combos(hand, &by_rank, &all_ranks, phoenix, mahjong, current_trick, results);
}

fn check_and_push(combo: Combo, current_trick: Option<&Combo>, results: &mut Vec<Combo>) {
    if let Some(ct) = current_trick {
        if can_beat(ct, &combo) {
            results.push(combo);
        }
    } else {
        results.push(combo);
    }
}

fn find_consecutive_pair_combos(
    _hand: &[Card],
    by_rank: &HashMap<u8, Vec<Card>>,
    all_ranks: &[u8],
    phoenix: bool,
    current_trick: Option<&Combo>,
    results: &mut Vec<Combo>,
) {
    // Find consecutive runs of ranks that have >= 2 cards each (or >= 1 with phoenix).
    let phoenix_card = Card::Special(SpecialKind::Phoenix);

    // Without phoenix: find consecutive ranks where each has count >= 2, length >= 2 ranks.
    for start_idx in 0..all_ranks.len() {
        let mut end_idx = start_idx;
        while end_idx < all_ranks.len()
            && by_rank[&all_ranks[end_idx]].len() >= 2
            && (end_idx == start_idx || all_ranks[end_idx] == all_ranks[end_idx - 1] + 1)
        {
            end_idx += 1;
        }
        let run = &all_ranks[start_idx..end_idx];
        if run.len() >= 2 {
            // Generate all sub-runs of length >= 2.
            for len in 2..=run.len() {
                for s in 0..=(run.len() - len) {
                    let sub = &run[s..s + len];
                    let mut cards: Vec<Card> = Vec::new();
                    for &r in sub {
                        cards.extend_from_slice(&by_rank[&r][..2]);
                    }
                    if let Some(combo) = identify_combo(&cards, None) {
                        if combo.kind == ComboKind::ConsecutivePairs {
                            check_and_push(combo, current_trick, results);
                        }
                    }
                }
            }
        }
    }

    // With phoenix: phoenix can fill one pair that's missing a card.
    if phoenix {
        // For each consecutive run of ranks, allow one rank to have count == 1 (phoenix fills it).
        for start_rank in 2u8..=14 {
            for run_len in 2..=7usize {
                let end_rank = start_rank + run_len as u8 - 1;
                if end_rank > 14 {
                    break;
                }
                let mut cards: Vec<Card> = Vec::new();
                let mut phoenix_used = false;
                let mut valid = true;
                for r in start_rank..=end_rank {
                    match by_rank.get(&r) {
                        Some(cs) if cs.len() >= 2 => {
                            cards.extend_from_slice(&cs[..2]);
                        }
                        Some(cs) if cs.len() == 1 && !phoenix_used => {
                            cards.push(cs[0]);
                            cards.push(phoenix_card);
                            phoenix_used = true;
                        }
                        _ => {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid && phoenix_used {
                    if let Some(combo) = identify_combo(&cards, None) {
                        if combo.kind == ComboKind::ConsecutivePairs {
                            check_and_push(combo, current_trick, results);
                        }
                    }
                }
            }
        }
    }
}

fn find_straight_combos(
    _hand: &[Card],
    by_rank: &HashMap<u8, Vec<Card>>,
    all_ranks: &[u8],
    phoenix: bool,
    mahjong: bool,
    current_trick: Option<&Combo>,
    results: &mut Vec<Combo>,
) {
    let phoenix_card = Card::Special(SpecialKind::Phoenix);
    let mahjong_card = Card::Special(SpecialKind::MahJong);

    // Effective ranks available (including mahjong as rank 1).
    let mut available: Vec<u8> = all_ranks.to_vec();
    if mahjong && !available.contains(&1) {
        available.push(1);
        available.sort();
    }

    // For each starting rank and length, try to form a straight.
    let min_rank = if mahjong { 1 } else { 2 };
    for start_rank in min_rank..=14u8 {
        for run_len in 5..=14usize {
            let end_rank = start_rank as usize + run_len - 1;
            if end_rank > 14 {
                break;
            }

            let mut cards: Vec<Card> = Vec::new();
            let mut phoenix_used = false;
            let mut valid = true;

            for r in start_rank..=(end_rank as u8) {
                if r == 1 && mahjong {
                    cards.push(mahjong_card);
                } else if let Some(cs) = by_rank.get(&r) {
                    if !cs.is_empty() {
                        cards.push(cs[0]);
                    } else if phoenix && !phoenix_used {
                        cards.push(phoenix_card);
                        phoenix_used = true;
                    } else {
                        valid = false;
                        break;
                    }
                } else if phoenix && !phoenix_used {
                    cards.push(phoenix_card);
                    phoenix_used = true;
                } else {
                    valid = false;
                    break;
                }
            }

            if valid && cards.len() == run_len {
                if let Some(combo) = identify_combo(&cards, None) {
                    if combo.kind == ComboKind::Straight {
                        check_and_push(combo, current_trick, results);
                    }
                }
            }
        }
    }
}

/// Check if the hand contains any playable combo that includes a card of the wished rank.
pub fn can_fulfill_wish(hand: &[Card], wish_rank: u8, current_trick: Option<&Combo>) -> bool {
    let combos = find_playable_combos(hand, current_trick);
    combos.iter().any(|combo| {
        combo.cards.iter().any(|c| match c {
            Card::Normal { rank, .. } => *rank == wish_rank,
            Card::Special(SpecialKind::MahJong) => wish_rank == 1,
            _ => false,
        })
    })
}
