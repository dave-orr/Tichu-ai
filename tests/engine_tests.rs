use tichu_engine::combinations::*;
use tichu_engine::engine::*;
use tichu_engine::scoring::*;
use tichu_engine::types::*;

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

fn c(suit: Suit, rank: u8) -> Card {
    Card::Normal { suit, rank }
}

fn jade(rank: u8) -> Card {
    c(Suit::Jade, rank)
}
fn sword(rank: u8) -> Card {
    c(Suit::Sword, rank)
}
fn pagoda(rank: u8) -> Card {
    c(Suit::Pagoda, rank)
}
fn star(rank: u8) -> Card {
    c(Suit::Star, rank)
}

fn phoenix() -> Card {
    Card::Special(SpecialKind::Phoenix)
}
fn dragon() -> Card {
    Card::Special(SpecialKind::Dragon)
}
fn mahjong() -> Card {
    Card::Special(SpecialKind::MahJong)
}
fn dog() -> Card {
    Card::Special(SpecialKind::Dog)
}

// ---------------------------------------------------------------------------
// Combination identification tests
// ---------------------------------------------------------------------------

#[test]
fn test_single_normal() {
    let combo = identify_combo(&[jade(5)], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Single);
    assert_eq!(combo.rank, 5.0);
    assert_eq!(combo.length, 1);
}

#[test]
fn test_single_mahjong() {
    let combo = identify_combo(&[mahjong()], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Single);
    assert_eq!(combo.rank, 1.0);
}

#[test]
fn test_single_dragon() {
    let combo = identify_combo(&[dragon()], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Single);
    assert_eq!(combo.rank, 15.0);
}

#[test]
fn test_single_phoenix() {
    let combo = identify_combo(&[phoenix()], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Single);
    assert_eq!(combo.rank, 1.5); // leading: 1.5
}

#[test]
fn test_single_dog() {
    let combo = identify_combo(&[dog()], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Single);
    assert_eq!(combo.rank, -1.0);
}

#[test]
fn test_pair() {
    let combo = identify_combo(&[jade(8), sword(8)], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Pair);
    assert_eq!(combo.rank, 8.0);
}

#[test]
fn test_pair_with_phoenix() {
    let combo = identify_combo(&[jade(10), phoenix()], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Pair);
    assert_eq!(combo.rank, 10.0);
}

#[test]
fn test_triple() {
    let combo = identify_combo(&[jade(7), sword(7), star(7)], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Triple);
    assert_eq!(combo.rank, 7.0);
}

#[test]
fn test_triple_with_phoenix() {
    let combo = identify_combo(&[jade(7), sword(7), phoenix()], None).unwrap();
    assert_eq!(combo.kind, ComboKind::Triple);
    assert_eq!(combo.rank, 7.0);
}

#[test]
fn test_four_of_a_kind_bomb() {
    let combo =
        identify_combo(&[jade(9), sword(9), pagoda(9), star(9)], None).unwrap();
    assert_eq!(combo.kind, ComboKind::FourOfAKindBomb);
    assert_eq!(combo.rank, 9.0);
}

#[test]
fn test_four_of_a_kind_with_phoenix_is_not_bomb() {
    // Phoenix can't be in a bomb.
    let result = identify_combo(&[jade(9), sword(9), pagoda(9), phoenix()], None);
    // Should be identified as something else or None, but NOT a bomb.
    if let Some(combo) = result {
        assert_ne!(combo.kind, ComboKind::FourOfAKindBomb);
    }
}

#[test]
fn test_straight_flush_bomb() {
    let combo = identify_combo(
        &[jade(3), jade(4), jade(5), jade(6), jade(7)],
        None,
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::StraightFlushBomb);
    assert_eq!(combo.rank, 7.0);
    assert_eq!(combo.length, 5);
}

#[test]
fn test_straight() {
    let combo = identify_combo(
        &[jade(3), sword(4), pagoda(5), star(6), jade(7)],
        None,
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::Straight);
    assert_eq!(combo.rank, 7.0);
}

#[test]
fn test_straight_with_mahjong() {
    let combo = identify_combo(
        &[mahjong(), jade(2), sword(3), pagoda(4), star(5)],
        None,
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::Straight);
    assert_eq!(combo.rank, 5.0);
}

#[test]
fn test_straight_with_phoenix() {
    // 3, 4, phoenix, 6, 7 -> phoenix fills 5
    let combo = identify_combo(
        &[jade(3), sword(4), phoenix(), star(6), jade(7)],
        None,
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::Straight);
    assert_eq!(combo.rank, 7.0);
}

#[test]
fn test_full_house() {
    let combo = identify_combo(
        &[jade(5), sword(5), pagoda(5), star(8), jade(8)],
        Some(5),
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::FullHouse);
    assert_eq!(combo.rank, 5.0);
}

#[test]
fn test_consecutive_pairs() {
    let combo = identify_combo(
        &[jade(5), sword(5), pagoda(6), star(6)],
        None,
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::ConsecutivePairs);
    assert_eq!(combo.rank, 6.0);
    assert_eq!(combo.length, 4);
}

#[test]
fn test_consecutive_pairs_three() {
    let combo = identify_combo(
        &[jade(5), sword(5), pagoda(6), star(6), jade(7), sword(7)],
        None,
    )
    .unwrap();
    assert_eq!(combo.kind, ComboKind::ConsecutivePairs);
    assert_eq!(combo.rank, 7.0);
    assert_eq!(combo.length, 6);
}

// ---------------------------------------------------------------------------
// can_beat tests
// ---------------------------------------------------------------------------

#[test]
fn test_higher_single_beats_lower() {
    let c1 = identify_combo(&[jade(5)], None).unwrap();
    let c2 = identify_combo(&[sword(8)], None).unwrap();
    assert!(can_beat(&c1, &c2));
    assert!(!can_beat(&c2, &c1));
}

#[test]
fn test_bomb_beats_non_bomb() {
    let single = identify_combo(&[jade(14)], None).unwrap();
    let bomb = identify_combo(&[jade(3), sword(3), pagoda(3), star(3)], None).unwrap();
    assert!(can_beat(&single, &bomb));
    assert!(!can_beat(&bomb, &single));
}

#[test]
fn test_straight_flush_beats_four_of_a_kind() {
    let four = identify_combo(&[jade(14), sword(14), pagoda(14), star(14)], None).unwrap();
    let sf = identify_combo(&[jade(2), jade(3), jade(4), jade(5), jade(6)], None).unwrap();
    assert!(can_beat(&four, &sf));
}

#[test]
fn test_cannot_beat_different_type() {
    let pair = identify_combo(&[jade(5), sword(5)], None).unwrap();
    let triple = identify_combo(&[jade(3), sword(3), pagoda(3)], None).unwrap();
    assert!(!can_beat(&pair, &triple));
}

// ---------------------------------------------------------------------------
// Scoring tests
// ---------------------------------------------------------------------------

#[test]
fn test_card_points() {
    assert_eq!(card_points(&jade(5)), 5);
    assert_eq!(card_points(&jade(10)), 10);
    assert_eq!(card_points(&jade(13)), 10); // King
    assert_eq!(card_points(&dragon()), 25);
    assert_eq!(card_points(&phoenix()), -25);
    assert_eq!(card_points(&jade(3)), 0);
    assert_eq!(card_points(&mahjong()), 0);
}

#[test]
fn test_double_victory_scoring() {
    let out_order = vec![0, 2, 1, 3]; // Team 0 goes 1-2
    let tricks_won: [Vec<Vec<Card>>; 4] = [vec![], vec![], vec![], vec![]];
    let hands: [Vec<Card>; 4] = [vec![], vec![], vec![], vec![]];
    let tichu_calls: [Option<TichuCall>; 4] = [None, None, None, None];

    let (t0, t1) = score_round(&out_order, &tricks_won, &hands, &tichu_calls);
    assert_eq!(t0, 200);
    assert_eq!(t1, 0);
}

#[test]
fn test_tichu_bonus() {
    let out_order = vec![0, 1, 2, 3];
    let tricks_won: [Vec<Vec<Card>>; 4] = [vec![], vec![], vec![], vec![]];
    let hands: [Vec<Card>; 4] = [vec![], vec![], vec![], vec![]];
    let tichu_calls: [Option<TichuCall>; 4] = [
        Some(TichuCall::Small), // seat 0 called, went out first -> +100
        None,
        None,
        Some(TichuCall::Grand), // seat 3 called, went out last -> -200
    ];

    let (t0, t1) = score_round(&out_order, &tricks_won, &hands, &tichu_calls);
    // Team 0 gets +100 (small tichu bonus for seat 0)
    // Team 1 gets -200 (grand tichu penalty for seat 3)
    assert_eq!(t0, 100);
    assert_eq!(t1, -200);
}

// ---------------------------------------------------------------------------
// Full game flow test
// ---------------------------------------------------------------------------

#[test]
fn test_game_flow_grand_tichu_to_passing() {
    let mut game = GameState::new();
    game.start_new_round();

    assert_eq!(game.phase, Phase::GrandTichuWindow);

    // Each player has 8 cards.
    for seat in 0..4 {
        assert_eq!(game.hands[seat].len(), 8);
    }

    // All decline grand tichu.
    for seat in 0..4 {
        game.decide_grand_tichu(seat, false).unwrap();
    }

    assert_eq!(game.phase, Phase::Passing);

    // Now each player has 14 cards.
    for seat in 0..4 {
        assert_eq!(game.hands[seat].len(), 14);
    }
}

#[test]
fn test_passing_to_playing() {
    let mut game = GameState::new();
    game.start_new_round();
    for seat in 0..4 {
        game.decide_grand_tichu(seat, false).unwrap();
    }

    // Each player passes 3 cards.
    for seat in 0..4 {
        let hand = game.hands[seat].clone();
        let pass = [hand[0], hand[1], hand[2]];
        game.pass_cards(seat, pass).unwrap();
    }

    assert_eq!(game.phase, Phase::Playing);
    // Still 14 cards each (passed 3, received 3).
    for seat in 0..4 {
        assert_eq!(game.hands[seat].len(), 14);
    }
}

#[test]
fn test_deck_has_56_cards() {
    let deck = tichu_engine::deck::create_deck();
    assert_eq!(deck.len(), 56);
}

#[test]
fn test_deck_points_sum_to_100() {
    let deck = tichu_engine::deck::create_deck();
    let total: i32 = deck.iter().map(|c| card_points(c)).sum();
    assert_eq!(total, 100);
}
