//! Seeded random self-play through the engine, checking rule invariants on
//! every action. This is the safety net for engine changes: any bot is only as
//! good as the rules it is playing under.

use rand::seq::SliceRandom;
use rand::Rng;
use tichu_engine::deck::{create_deck, seeded_rng};
use tichu_engine::engine::*;
use tichu_engine::scoring::card_points;
use tichu_engine::types::*;

fn dragon_card() -> Card {
    Card::Special(SpecialKind::Dragon)
}

fn is_dragon(c: &Card) -> bool {
    *c == dragon_card()
}

fn active_seats(g: &GameState) -> Vec<Seat> {
    (0..4).filter(|&s| !g.is_out(s)).collect()
}

/// Play one full round with uniformly random legal actions, asserting invariants.
fn play_random_round(seed: u64) -> GameState {
    let mut rng = seeded_rng(seed);
    let mut g = GameState::new();
    g.start_new_round_with_rng(&mut rng);

    for seat in 0..4 {
        g.decide_grand_tichu(seat, rng.gen_bool(0.1)).unwrap();
    }
    assert_eq!(g.phase, Phase::Passing);
    for seat in 0..4 {
        if g.tichu_calls[seat].is_none() && rng.gen_bool(0.1) {
            g.call_small_tichu(seat).unwrap();
        }
        let mut hand = g.hands[seat].clone();
        hand.shuffle(&mut rng);
        g.pass_cards(seat, [hand[0], hand[1], hand[2]]).unwrap();
    }
    assert_eq!(g.phase, Phase::Playing);
    assert!(g.hands[g.current_player].contains(&Card::Special(SpecialKind::MahJong)));

    // Seats that have acted (played or passed) since the trick was last raised.
    let mut acted_since_last_play: Vec<Seat> = Vec::new();
    let mut dragon_played_by: Option<Seat> = None;
    let mut dragon_beaten_by_bomb = false;
    let mut steps = 0;

    while g.phase == Phase::Playing {
        steps += 1;
        assert!(steps < 500, "seed {seed}: round did not terminate");

        let seat = g.current_player;
        let actions = g.legal_actions();
        assert!(
            !actions.is_empty(),
            "seed {seed}: no legal action for seat {seat}: trick={:?} wish={:?} hand={:?}",
            g.current_trick,
            g.mah_jong_wish,
            g.hands[seat]
        );
        if g.dragon_giveaway {
            assert_eq!(Some(seat), g.dragon_giveaway_seat, "seed {seed}: turn not handed to dragon giver");
        } else {
            assert!(!g.is_out(seat), "seed {seed}: out seat {seat} to act");
        }

        let action = actions.choose(&mut rng).unwrap().clone();
        match action {
            Action::Pass => {
                let leader = g.last_played_by.expect("pass with no trick leader");
                let active = active_seats(&g);
                let winning_combo = g.current_trick.clone().unwrap();
                let pile_has_dragon = g.current_trick_cards.iter().any(is_dragon);
                g.pass_turn(seat).unwrap();
                acted_since_last_play.push(seat);
                if g.dragon_giveaway {
                    assert_eq!(winning_combo.cards, vec![dragon_card()], "seed {seed}: giveaway without the Dragon winning");
                } else if g.current_trick.is_none() && pile_has_dragon {
                    assert!(
                        matches!(winning_combo.kind, ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb),
                        "seed {seed}: Dragon trick captured without a giveaway"
                    );
                }
                if g.current_trick.is_none() {
                    // Trick awarded: every active seat other than an active leader must have acted.
                    for s in active {
                        if s == leader && !g.is_out(leader) {
                            continue;
                        }
                        assert!(
                            acted_since_last_play.contains(&s),
                            "seed {seed}: trick awarded before seat {s} could act (leader {leader}, out={:?})",
                            g.out_order
                        );
                    }
                    acted_since_last_play.clear();
                } else {
                    assert_ne!(g.current_player, seat);
                }
            }
            Action::Play(combo) => {
                let wish_before = g.mah_jong_wish;
                let phoenix_as = (combo.kind == ComboKind::FullHouse).then_some(combo.rank as u8);
                let wish = combo
                    .cards
                    .iter()
                    .any(|c| matches!(c, Card::Special(SpecialKind::MahJong)))
                    .then(|| rng.gen_range(2..=14u8));
                let trick_before = g.current_trick.clone();
                g.play_cards(seat, &combo.cards, phoenix_as, wish)
                    .unwrap_or_else(|e| panic!("seed {seed}: legal action rejected: {e:?} combo={combo:?} trick={trick_before:?} wish={wish_before:?}"));
                if combo.cards.iter().any(is_dragon) {
                    dragon_played_by = Some(seat);
                }
                let is_bomb = matches!(combo.kind, ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb);
                if is_bomb && trick_before.as_ref().is_some_and(|t| t.cards.iter().any(is_dragon)) {
                    dragon_beaten_by_bomb = true;
                }
                assert!(!g.dragon_giveaway, "seed {seed}: giveaway cannot start on a play");
                if let Some(w) = wish_before {
                    if combo.cards.iter().any(|c| c.rank() == Some(w)) {
                        assert_eq!(g.mah_jong_wish, None, "seed {seed}: wish not cleared after being fulfilled");
                    }
                }
                if let Some(ct) = &g.current_trick {
                    // Phoenix never beats the Dragon.
                    if let Some(prev) = &trick_before {
                        assert!(!(prev.rank >= 15.0 && ct.cards == vec![Card::Special(SpecialKind::Phoenix)]));
                    }
                }
                acted_since_last_play.clear();
                acted_since_last_play.push(seat);
            }
            Action::GiveDragonTrick(target) => {
                assert_ne!(team_for_seat(target), team_for_seat(seat));
                g.give_dragon_trick(seat, target).unwrap();
                assert!(!g.dragon_giveaway);
                acted_since_last_play.clear();
            }
        }
    }

    assert!(matches!(g.phase, Phase::RoundEnd | Phase::GameEnd));
    assert_eq!(g.out_order.len(), 4);
    assert!(!g.dragon_giveaway);
    assert!(g.current_trick_cards.is_empty(), "seed {seed}: trick left on the table at round end");

    // Every card is either in a captured trick or still in a hand, exactly once.
    let mut all: Vec<Card> = g.tricks_won.iter().flatten().flatten().copied().collect();
    all.extend(g.hands.iter().flatten().copied());
    let mut expected = create_deck();
    all.sort_by(|a, b| a.sort_key().partial_cmp(&b.sort_key()).unwrap());
    expected.sort_by(|a, b| a.sort_key().partial_cmp(&b.sort_key()).unwrap());
    let missing: Vec<&Card> = expected.iter().filter(|c| !all.contains(c)).collect();
    assert_eq!(all.len(), 56, "seed {seed}: cards lost or duplicated; missing={missing:?}");
    assert_eq!(all, expected, "seed {seed}: cards lost or duplicated; missing={missing:?}");

    // The Dragon trick never stays with the Dragon player's team (unless a
    // bomb beat it, in which case the bomber simply captures it).
    if let (Some(dp), false) = (dragon_played_by, dragon_beaten_by_bomb) {
        for seat in 0..4 {
            if g.tricks_won[seat].iter().flatten().any(is_dragon) {
                assert_ne!(team_for_seat(seat), team_for_seat(dp), "seed {seed}: Dragon trick kept by own team");
            }
        }
    }

    // Card points sum to 100 (or 200/0 on a double victory), plus Tichu bonuses.
    let (t0, t1) = g.round_history[0];
    let bonus: i32 = (0..4)
        .map(|s| match g.tichu_calls[s] {
            None => 0,
            Some(call) => {
                let won = g.out_order[0] == s;
                let size = if call == TichuCall::Grand { 200 } else { 100 };
                if won { size } else { -size }
            }
        })
        .sum();
    let double_victory = team_for_seat(g.out_order[0]) == team_for_seat(g.out_order[1]);
    let expected_total = if double_victory { 200 } else { 100 };
    assert_eq!(t0 + t1, expected_total + bonus, "seed {seed}: scores {t0}/{t1} out={:?} calls={:?}", g.out_order, g.tichu_calls);
    let _ = card_points;
    g
}

#[test]
fn random_selfplay_rounds_respect_rules() {
    for seed in 0..300u64 {
        play_random_round(seed);
    }
}

#[test]
fn random_selfplay_is_deterministic_for_a_seed() {
    let a = play_random_round(7);
    let b = play_random_round(7);
    assert_eq!(a.round_history, b.round_history);
    assert_eq!(a.out_order, b.out_order);
    assert_eq!(a.tricks_won, b.tricks_won);
}

#[test]
fn random_selfplay_full_game_ends() {
    let mut rng = seeded_rng(99);
    let mut g = GameState::new();
    g.target_score = 300;
    let mut rounds = 0;
    while g.phase != Phase::GameEnd {
        rounds += 1;
        assert!(rounds < 200, "game did not end");
        g.start_new_round_with_rng(&mut rng);
        for seat in 0..4 {
            g.decide_grand_tichu(seat, false).unwrap();
        }
        for seat in 0..4 {
            let h = g.hands[seat].clone();
            g.pass_cards(seat, [h[0], h[1], h[2]]).unwrap();
        }
        while g.phase == Phase::Playing {
            let seat = g.current_player;
            let action = g.legal_actions().choose(&mut rng).unwrap().clone();
            match action {
                Action::Pass => g.pass_turn(seat).unwrap(),
                Action::Play(c) => {
                    let pa = (c.kind == ComboKind::FullHouse).then_some(c.rank as u8);
                    g.play_cards(seat, &c.cards, pa, None).unwrap()
                }
                Action::GiveDragonTrick(t) => g.give_dragon_trick(seat, t).unwrap(),
            }
        }
    }
    assert!(g.winner().is_some());
    assert_eq!(g.round_history.len(), rounds);
}
