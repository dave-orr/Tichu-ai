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

// ---------------------------------------------------------------------------
// Engine rule regression tests
// ---------------------------------------------------------------------------

/// Build a game in the Playing phase with the given hands, seat 0 to lead.
fn playing_game(hands: [Vec<Card>; 4]) -> GameState {
    let mut g = GameState::new();
    g.phase = Phase::Playing;
    g.hands = hands;
    g.current_player = 0;
    g
}

#[test]
fn test_phoenix_offered_as_single_response() {
    let eight = identify_combo(&[jade(8)], None).unwrap();
    let combos = find_playable_combos(&[phoenix(), star(3)], Some(&eight));
    let ph = combos.iter().find(|c| c.cards == vec![phoenix()]).expect("phoenix should be playable on an 8");
    assert_eq!(ph.kind, ComboKind::Single);
    assert_eq!(ph.rank, 8.5);

    // And it is offered through legal_actions.
    let mut g = playing_game([vec![jade(8), jade(2)], vec![phoenix(), star(3)], vec![jade(4)], vec![jade(5)]]);
    g.play_cards(0, &[jade(8)], None, None).unwrap();
    let offered = g.legal_actions().iter().any(|a| matches!(a, Action::Play(c) if c.cards == vec![phoenix()]));
    assert!(offered);
    g.play_cards(1, &[phoenix()], None, None).unwrap();
    assert_eq!(g.current_trick.as_ref().unwrap().rank, 8.5);
}

#[test]
fn test_phoenix_cannot_beat_dragon() {
    let mut g = playing_game([vec![dragon(), jade(2)], vec![phoenix(), star(3)], vec![jade(4)], vec![jade(5)]]);
    g.play_cards(0, &[dragon()], None, None).unwrap();
    assert_eq!(g.play_cards(1, &[phoenix()], None, None), Err(GameError::CannotBeatCurrentTrick));
    let offered = g.legal_actions().iter().any(|a| matches!(a, Action::Play(c) if c.cards == vec![phoenix()]));
    assert!(!offered);
    // Ace-then-Phoenix still works.
    let ace = identify_combo(&[jade(14)], None).unwrap();
    assert!(find_playable_combos(&[phoenix()], Some(&ace)).len() == 1);
}

#[test]
fn test_all_remaining_players_must_pass_when_leader_is_out() {
    // Seat 0 goes out on its lead; seats 1, 2 and 3 all have to pass before the
    // trick is awarded, so seat 3 still gets a chance to beat it.
    let mut g = playing_game([vec![jade(8)], vec![star(3), star(4)], vec![jade(4), jade(5)], vec![jade(9), jade(2)]]);
    g.play_cards(0, &[jade(8)], None, None).unwrap();
    assert!(g.is_out(0));
    g.pass_turn(1).unwrap();
    g.pass_turn(2).unwrap();
    assert!(g.current_trick.is_some(), "trick awarded before seat 3 could act");
    assert_eq!(g.current_player, 3);
    g.play_cards(3, &[jade(9)], None, None).unwrap();
    assert_eq!(g.last_played_by, Some(3));
    // Now seats 1 and 2 pass and seat 3 (still active) wins the trick.
    g.pass_turn(1).unwrap();
    g.pass_turn(2).unwrap();
    assert!(g.current_trick.is_none());
    assert_eq!(g.current_player, 3);
    assert_eq!(g.tricks_won[3].len(), 1);
}

#[test]
fn test_trick_awarded_to_out_leader_after_everyone_passes() {
    let mut g = playing_game([vec![jade(8)], vec![star(3), star(4)], vec![jade(4), jade(5)], vec![jade(6), jade(2)]]);
    g.play_cards(0, &[jade(8)], None, None).unwrap();
    g.pass_turn(1).unwrap();
    g.pass_turn(2).unwrap();
    g.pass_turn(3).unwrap();
    assert!(g.current_trick.is_none());
    assert_eq!(g.tricks_won[0].len(), 1);
    // Next active player after the (out) winner leads.
    assert_eq!(g.current_player, 1);
}

#[test]
fn test_wish_clears_once_fulfilled() {
    let mut g = playing_game([
        vec![mahjong(), jade(2)],
        vec![star(7), star(3)],
        vec![jade(7), jade(4)],
        vec![jade(9), jade(5)],
    ]);
    g.play_cards(0, &[mahjong()], None, Some(7)).unwrap();
    assert_eq!(g.mah_jong_wish, Some(7));
    // Seat 1 holds a 7 and can play it, so it may not pass.
    assert_eq!(g.pass_turn(1), Err(GameError::MustFulfillWish));
    assert_eq!(g.play_cards(1, &[star(3)], None, None), Err(GameError::MustFulfillWish));
    g.play_cards(1, &[star(7)], None, None).unwrap();
    assert_eq!(g.mah_jong_wish, None, "wish must clear once the wished rank is played");
    // Seat 2 also holds a 7 but is no longer forced to play it.
    assert!(g.legal_actions().iter().any(|a| matches!(a, Action::Pass)));
    g.pass_turn(2).unwrap();
}

#[test]
fn test_bomb_containing_wished_rank_fulfills_wish() {
    let mut g = playing_game([
        vec![mahjong(), jade(2)],
        vec![jade(7), sword(7), pagoda(7), star(7), star(3)],
        vec![jade(4), jade(5)],
        vec![jade(9), jade(6)],
    ]);
    g.play_cards(0, &[mahjong()], None, Some(7)).unwrap();
    g.play_bomb(1, &[jade(7), sword(7), pagoda(7), star(7)]).unwrap();
    assert_eq!(g.mah_jong_wish, None);
}

#[test]
fn test_dragon_trick_hands_turn_to_winner_for_giveaway() {
    let mut g = playing_game([vec![dragon(), jade(2)], vec![jade(3)], vec![jade(4)], vec![jade(5)]]);
    g.play_cards(0, &[dragon()], None, None).unwrap();
    for s in 1..4 {
        g.pass_turn(s).unwrap();
    }
    assert!(g.dragon_giveaway);
    assert_eq!(g.dragon_giveaway_seat, Some(0));
    assert_eq!(g.current_player, 0);
    let actions = g.legal_actions();
    assert_eq!(actions.len(), 2);
    assert!(actions.iter().all(|a| matches!(a, Action::GiveDragonTrick(1) | Action::GiveDragonTrick(3))));
    // Nobody may play (even a bomb) until the trick is given away.
    assert_eq!(g.play_cards(0, &[jade(2)], None, None), Err(GameError::WrongPhase));
    assert_eq!(g.give_dragon_trick(0, 2), Err(GameError::InvalidDragonTarget));
    g.give_dragon_trick(0, 3).unwrap();
    assert_eq!(g.tricks_won[3].len(), 1);
    assert_eq!(g.current_player, 0);
    assert!(g.legal_actions().iter().any(|a| matches!(a, Action::Play(_))));
}

#[test]
fn test_dragon_in_final_trick_is_still_given_away() {
    // Seats 1 and 2 are already out; seat 0 goes out with the Dragon, which ends
    // the round. The Dragon trick still has to go to an opponent.
    let mut g = playing_game([vec![dragon()], vec![], vec![], vec![jade(5), jade(10)]]);
    g.out_order = vec![1, 2];
    g.play_cards(0, &[dragon()], None, None).unwrap();
    assert_eq!(g.phase, Phase::Playing);
    assert!(g.dragon_giveaway);
    let actions = g.legal_actions();
    assert_eq!(actions.len(), 2);
    assert!(actions.iter().all(|a| matches!(a, Action::GiveDragonTrick(1) | Action::GiveDragonTrick(3))));
    g.give_dragon_trick(0, 3).unwrap();
    assert_eq!(g.phase, Phase::RoundEnd);
    assert!(g.tricks_won[3][0].contains(&dragon()));
    assert!(g.tricks_won[0].is_empty());
    // Seat 1 went out first (team 1) and seat 3 (team 1) was last. Seat 3's
    // tricks (the Dragon, 25) go to the first-out team (team 1); its remaining
    // hand (5 + 10) goes to the opposing team (team 0).
    assert_eq!(g.round_history[0], (15, 25));
}

#[test]
fn test_small_tichu_window_is_per_player() {
    let mut g = playing_game([vec![jade(8), jade(2)], vec![star(3), star(4)], vec![jade(4), jade(5)], vec![jade(9), jade(6)]]);
    g.play_cards(0, &[jade(8)], None, None).unwrap();
    // Seat 0 has played, so it may no longer call.
    assert_eq!(g.call_small_tichu(0), Err(GameError::WrongPhase));
    // Seat 3 has not played a card yet, so it still may.
    g.call_small_tichu(3).unwrap();
    assert_eq!(g.tichu_calls[3], Some(TichuCall::Small));
}

#[test]
fn test_seeded_deal_is_reproducible() {
    let mut a = GameState::new();
    a.start_new_round_with_rng(&mut tichu_engine::deck::seeded_rng(42));
    let mut b = GameState::new();
    b.start_new_round_with_rng(&mut tichu_engine::deck::seeded_rng(42));
    let mut c = GameState::new();
    c.start_new_round_with_rng(&mut tichu_engine::deck::seeded_rng(43));
    assert_eq!(a.hands, b.hands);
    assert_ne!(a.hands, c.hands);
}

#[test]
fn test_dog_passes_lead_to_partner_and_is_captured() {
    let mut g = playing_game([vec![dog(), jade(2)], vec![star(3)], vec![jade(4)], vec![jade(5)]]);
    assert_eq!(g.play_cards(1, &[star(3)], None, None), Err(GameError::NotYourTurn));
    g.play_cards(0, &[dog()], None, None).unwrap();
    assert_eq!(g.current_player, 2);
    assert!(g.current_trick.is_none());
    assert_eq!(g.tricks_won[0], vec![vec![dog()]]);
    // The Dog cannot be played onto a trick.
    g.play_cards(2, &[jade(4)], None, None).unwrap();
    let mut h = playing_game([vec![dog()], vec![], vec![], vec![]]);
    h.current_trick = Some(identify_combo(&[jade(3)], None).unwrap());
    assert_eq!(h.play_cards(0, &[dog()], None, None), Err(GameError::DogMustBeLead));
}

#[test]
fn test_dragon_beaten_by_bomb_is_captured_by_bomber() {
    // Seat 2 (the Dragon player's partner) bombs the Dragon; the trick, Dragon
    // included, simply goes to seat 2. No giveaway.
    let mut g = playing_game([
        vec![dragon(), jade(2)],
        vec![star(3), star(4)],
        vec![jade(7), sword(7), pagoda(7), star(7), jade(9)],
        vec![jade(5), jade(6)],
    ]);
    g.play_cards(0, &[dragon()], None, None).unwrap();
    g.pass_turn(1).unwrap();
    g.play_cards(2, &[jade(7), sword(7), pagoda(7), star(7)], None, None).unwrap();
    g.pass_turn(3).unwrap();
    g.pass_turn(0).unwrap();
    g.pass_turn(1).unwrap();
    assert!(!g.dragon_giveaway);
    assert_eq!(g.tricks_won[2].len(), 1);
    assert!(g.tricks_won[2][0].contains(&dragon()));
    assert_eq!(g.current_player, 2);
}
