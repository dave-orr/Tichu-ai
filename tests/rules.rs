//! Rule tests. Most engine cases mirror `shared/src/engine.test.ts` in the
//! tichu repo; the broad check against the TS engine is `tools/ts-diff`.

use rand::rngs::StdRng;
use rand::SeedableRng;

use tichu_engine::bot::Bot;
use tichu_engine::bots::{HeuristicBot, RandomBot};
use tichu_engine::combinations::*;
use tichu_engine::engine::*;
use tichu_engine::sim::play_round;
use tichu_engine::types::*;

fn j(r: u8) -> Card {
    Card::normal(Suit::Jade, r)
}
fn s(r: u8) -> Card {
    Card::normal(Suit::Sword, r)
}
fn p(r: u8) -> Card {
    Card::normal(Suit::Pagoda, r)
}
fn st(r: u8) -> Card {
    Card::normal(Suit::Star, r)
}

fn id(cards: &[Card]) -> Combo {
    identify_combo(cards, None).unwrap_or_else(|| panic!("not a combo: {}", cards_str(cards)))
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------

#[test]
fn singles() {
    assert_eq!(id(&[j(5)]).rank, 5.0);
    assert_eq!(id(&[MAH_JONG]).rank, 1.0);
    assert_eq!(id(&[DRAGON]).rank, 15.0);
    assert_eq!(id(&[PHOENIX]).rank, 1.5);
    assert_eq!(id(&[DOG]).rank, 0.0);
}

#[test]
fn pairs_triples_and_the_phoenix() {
    assert_eq!(id(&[j(7), s(7)]).kind, ComboKind::Pair);
    assert_eq!(id(&[j(7), PHOENIX]).rank, 7.0);
    assert_eq!(id(&[j(7), s(7), PHOENIX]).kind, ComboKind::Triple);
    assert!(identify_combo(&[j(7), s(8)], None).is_none());
    // The Mah Jong is a 1 only in straights; the Dragon and Dog only alone.
    assert!(identify_combo(&[MAH_JONG, PHOENIX], None).is_none());
    assert!(identify_combo(&[DRAGON, PHOENIX], None).is_none());
    assert!(identify_combo(&[DOG, PHOENIX], None).is_none());
    // A Phoenix's claimed rank can't change what it pairs with.
    assert_eq!(identify_combo(&[j(3), PHOENIX], Some(14)).unwrap().rank, 3.0);
}

#[test]
fn full_houses() {
    assert_eq!(id(&[j(3), s(3), p(3), j(9), s(9)]).rank, 3.0);
    assert_eq!(id(&[j(3), s(3), p(3), j(9), PHOENIX]).rank, 3.0);
    // Two pairs + Phoenix: the higher pair is the triple unless asked.
    let two_pairs = [j(3), s(3), j(9), s(9), PHOENIX];
    assert_eq!(identify_combo(&two_pairs, None).unwrap().rank, 9.0);
    assert_eq!(identify_combo(&two_pairs, Some(3)).unwrap().rank, 3.0);
    assert!(identify_combo(&[j(3), s(3), p(3), st(3), PHOENIX], None).is_none());
}

#[test]
fn consecutive_pairs() {
    let c = id(&[j(5), s(5), j(6), s(6), j(7), s(7)]);
    assert_eq!((c.kind, c.rank, c.length), (ComboKind::ConsecutivePairs, 7.0, 6));
    assert_eq!(id(&[j(5), s(5), j(6), PHOENIX]).rank, 6.0);
    assert!(identify_combo(&[j(5), s(5), j(7), s(7)], None).is_none());
}

#[test]
fn straights() {
    let c = id(&[j(3), s(4), j(5), s(6), p(7)]);
    assert_eq!((c.kind, c.rank), (ComboKind::Straight, 7.0));
    assert_eq!(id(&[MAH_JONG, j(2), s(3), j(4), p(5)]).rank, 5.0);
    // Phoenix fills a gap…
    assert_eq!(id(&[j(3), s(4), PHOENIX, s(6), p(7)]).rank, 7.0);
    // …or extends the run: high by default, low when asked, low at the Ace.
    let run = [j(3), s(4), j(5), s(6), PHOENIX];
    assert_eq!(identify_combo(&run, None).unwrap().rank, 7.0);
    assert_eq!(identify_combo(&run, Some(2)).unwrap().rank, 6.0);
    assert_eq!(id(&[j(11), s(12), j(13), s(14), PHOENIX]).rank, 14.0);
    // Duplicate ranks, two gaps, or the Phoenix as the 1 are not straights.
    assert!(identify_combo(&[j(3), s(3), j(5), s(6), p(7)], None).is_none());
    assert!(identify_combo(&[j(3), PHOENIX, j(6), s(7), p(9)], None).is_none());
    assert!(identify_combo(&[PHOENIX, j(2), s(3), j(4), p(5)], None).map(|c| c.rank) == Some(6.0));
}

#[test]
fn bombs() {
    assert_eq!(id(&[j(9), s(9), p(9), st(9)]).kind, ComboKind::FourOfAKindBomb);
    assert_eq!(id(&[s(4), s(5), s(6), s(7), s(8)]).kind, ComboKind::StraightFlushBomb);
    // No Phoenix (or Mah Jong) in bombs.
    assert_eq!(id(&[s(4), s(5), PHOENIX, s(7), s(8)]).kind, ComboKind::Straight);
    assert!(identify_combo(&[j(9), s(9), p(9), PHOENIX], None).is_none());
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

#[test]
fn beating() {
    let mut ph = id(&[PHOENIX]);
    ph.rank = 14.5;
    assert!(can_beat(&id(&[j(14)]), &ph));
    assert!(!can_beat(&id(&[DRAGON]), &ph));
    assert!(can_beat(&id(&[j(14)]), &id(&[DRAGON])));
    assert!(!can_beat(&id(&[j(5), s(6), j(7), s(8), p(9)]), &id(&[j(6), s(7), j(8), s(9), p(10), j(11)])));

    let four = id(&[j(2), s(2), p(2), st(2)]);
    let sf5 = id(&[s(4), s(5), s(6), s(7), s(8)]);
    let sf5_hi = id(&[j(5), j(6), j(7), j(8), j(9)]);
    let sf6 = id(&[p(2), p(3), p(4), p(5), p(6), p(7)]);
    assert!(can_beat(&id(&[DRAGON]), &four));
    assert!(can_beat(&four, &sf5));
    assert!(!can_beat(&sf5, &four));
    assert!(can_beat(&sf5, &sf5_hi));
    assert!(can_beat(&sf5_hi, &sf6));
    assert!(!can_beat(&four, &id(&[DRAGON])));
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

#[test]
fn generator_finds_both_phoenix_readings_and_suit_choices() {
    let hand = [j(3), s(4), j(5), s(6), PHOENIX, p(6)];
    let plays = playable_combos(&hand, None);
    let straights: Vec<&Combo> = plays.iter().filter(|c| c.kind == ComboKind::Straight).collect();
    // 3-7 and 2-6 readings, each with either 6.
    assert!(straights.iter().any(|c| c.rank == 7.0 && c.phoenix_as == Some(7)));
    assert!(straights.iter().any(|c| c.rank == 6.0 && c.phoenix_as == Some(2)));
    assert_eq!(straights.iter().filter(|c| c.rank == 6.0 && c.cards.len() == 5).count(), 2);

    let trick = id(&[j(10), s(10), p(10), j(2), s(2)]);
    let fh = playable_combos(&[j(3), s(3), j(12), s(12), PHOENIX], Some(&trick));
    assert_eq!(fh.iter().map(|c| c.rank).collect::<Vec<_>>(), vec![12.0]);
}

#[test]
fn generator_matches_brute_force() {
    // Every subset that identifies (with any Phoenix reading) and beats the
    // trick must be generated, and nothing else.
    let mut rng = StdRng::seed_from_u64(5);
    for trial in 0..60 {
        let deck = tichu_engine::deck::shuffled_deck(&mut rng);
        let hand: Vec<Card> = deck[..10].to_vec();
        let current = if trial % 3 == 0 { None } else { playable_combos(&deck[10..24], None).pop() };
        let generated: std::collections::HashSet<(u64, i32)> = playable_combos(&hand, current.as_ref())
            .iter()
            .map(|c| (c.card_set().0, (c.rank * 2.0) as i32))
            .collect();
        let mut brute = std::collections::HashSet::new();
        for mask in 1u32..(1 << hand.len()) {
            let cards: Vec<Card> = (0..hand.len()).filter(|i| mask & (1 << i) != 0).map(|i| hand[i]).collect();
            for pa in std::iter::once(None).chain((2..=14).map(Some)) {
                let Some(mut c) = identify_combo(&cards, pa) else { continue };
                if c.cards == [PHOENIX] {
                    if let Some(cur) = &current {
                        c.rank = cur.rank + 0.5;
                    }
                }
                if current.as_ref().map_or(true, |cur| can_beat(cur, &c)) {
                    brute.insert((c.card_set().0, (c.rank * 2.0) as i32));
                }
            }
        }
        assert_eq!(generated, brute, "hand {} on {:?}", cards_str(&hand), current.map(|c| c.to_string()));
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

fn state(hands: [Vec<Card>; 4], turn: Seat) -> GameState {
    GameState::mid_play(Settings::default(), hands, turn)
}

/// Put `combo` (played by `by`) on the table.
fn on_table(g: &mut GameState, by: Seat, cards: &[Card]) {
    let c = id(cards);
    g.trick_plays.push(TrickPlay { seat: by, combo: c.clone() });
    g.current_trick = Some(c);
    g.last_played_by = Some(by);
}

#[test]
fn trick_needs_every_active_player_to_pass_after_leader_goes_out() {
    let mut g = state([vec![], vec![j(8)], vec![j(9)], vec![]], 1);
    g.players[0].out_order = 2;
    g.players[3].out_order = 1;
    g.out_count = 2;
    on_table(&mut g, 0, &[j(7)]);
    g.pass_turn(1).unwrap();
    assert_eq!(g.trick_countdown, None);
    assert_eq!(g.turn, 2);
    g.pass_turn(2).unwrap();
    assert_eq!(g.trick_countdown, Some(0));
    // Seat 0 is out, so the lead goes to the next active seat.
    g.resolve_trick().unwrap();
    assert_eq!(g.turn, 1);
}

#[test]
fn countdown_locks_the_trick_but_not_bombs() {
    let mut g = state([vec![j(11), j(2)], vec![j(3), s(4), s(5)], vec![j(4), j(9), s(9), p(9), st(9)], vec![j(6)]], 1);
    on_table(&mut g, 0, &[j(10)]);
    g.pass_turn(1).unwrap();
    g.pass_turn(2).unwrap();
    g.pass_turn(3).unwrap();
    assert_eq!(g.pending(), Pending::TrickCountdown(0));
    assert_eq!(g.pass_turn(3), Err(GameError::Blocked));
    assert_eq!(g.play(3, &[j(6)], None), Err(GameError::Blocked));
    g.bomb(2, &[j(9), s(9), p(9), st(9)]).unwrap();
    assert_eq!(g.trick_countdown, None);
    assert_eq!(g.turn, 3);
}

#[test]
fn wish_holds_the_turn_then_binds_leader_and_followers() {
    let mut g = state([vec![MAH_JONG, j(3)], vec![j(8), j(12)], vec![j(9)], vec![j(5), DOG]], 0);
    g.play(0, &[MAH_JONG], None).unwrap();
    assert_eq!(g.pending(), Pending::Wish(0));
    assert_eq!(g.play(1, &[j(12)], None), Err(GameError::Blocked));
    g.set_wish(0, Some(8)).unwrap();
    assert_eq!(g.turn, 1);
    // Seat 1 holds an 8 that beats the Mah Jong: the queen and a pass are out.
    assert!(!g.can_pass(1));
    assert_eq!(g.play(1, &[j(12)], None), Err(GameError::MustFulfillWish));
    assert_eq!(g.legal_plays(1).len(), 1);
    g.play(1, &[j(8)], None).unwrap();
    assert_eq!(g.wish, None);
}

#[test]
fn wish_binds_the_lead_and_the_dog() {
    let mut g = state([vec![j(3), j(8), DOG], vec![j(2)], vec![j(4)], vec![j(5)]], 0);
    g.wish = Some(8);
    assert_eq!(g.play(0, &[DOG], None), Err(GameError::MustFulfillWish));
    assert_eq!(g.play(0, &[j(3)], None), Err(GameError::MustFulfillWish));
    assert!(g.legal_plays(0).iter().all(|c| c.contains_rank(8)));
}

#[test]
fn a_wish_can_force_out_a_bomb() {
    let bomb = [j(8), s(8), p(8), st(8)];
    let mut hand = bomb.to_vec();
    hand.push(j(2));
    let mut g = state([vec![j(3)], hand, vec![j(4)], vec![j(5)]], 1);
    on_table(&mut g, 0, &[j(3), s(4), p(5), j(6), s(7)]);
    g.wish = Some(8);
    assert!(!g.can_pass(1));
    g.play(1, &bomb, None).unwrap();
    assert_eq!(g.wish, None);
}

#[test]
fn dog_passes_the_lead_to_partner_or_onward() {
    let mut g = state([vec![DOG, j(3)], vec![j(4)], vec![j(5)], vec![j(6)]], 0);
    g.play(0, &[DOG], None).unwrap();
    assert_eq!(g.turn, 2);
    assert!(g.current_trick.is_none());

    let mut g = state([vec![DOG, j(3)], vec![j(4)], vec![], vec![j(6)]], 0);
    g.players[2].out_order = 1;
    g.out_count = 1;
    g.play(0, &[DOG], None).unwrap();
    assert_eq!(g.turn, 3, "next active seat after the partner");

    let mut g = state([vec![j(3)], vec![DOG, j(4)], vec![j(5)], vec![j(6)]], 0);
    on_table(&mut g, 3, &[j(2)]);
    g.turn = 1;
    assert_eq!(g.play(1, &[DOG], None), Err(GameError::DogMustBeLead));
}

#[test]
fn bombs_on_an_empty_table_only_from_the_leader() {
    let bomb = [j(8), s(8), p(8), st(8)];
    let mut g = state([vec![j(3)], bomb.to_vec(), vec![j(4)], vec![j(5)]], 0);
    assert_eq!(g.bomb(1, &bomb), Err(GameError::NotYourTurn));
    assert!(g.legal_bombs(1).is_empty());
    g.turn = 1;
    g.bomb(1, &bomb).unwrap();
}

#[test]
fn dragon_trick_goes_to_the_chosen_opponent() {
    let mut g = state([vec![DRAGON, j(3)], vec![j(4)], vec![j(5)], vec![j(6)]], 0);
    on_table(&mut g, 3, &[j(10)]);
    g.play(0, &[DRAGON], None).unwrap();
    for seat in [1, 2, 3] {
        g.pass_turn(seat).unwrap();
    }
    g.resolve_trick().unwrap();
    assert_eq!(g.pending(), Pending::DragonGiveaway(0));
    assert_eq!(g.bomb(1, &[j(4)]), Err(GameError::Blocked));
    assert_eq!(g.give_dragon(0, 2), Err(GameError::InvalidDragonTarget));
    g.give_dragon(0, 3).unwrap();
    assert_eq!(g.players[3].captured_points(), 35);
    assert_eq!(g.turn, 0);
}

#[test]
fn dragon_on_top_at_round_end_still_goes_to_an_opponent() {
    let mut g = state([vec![DRAGON], vec![], vec![], vec![j(3), j(4)]], 0);
    g.players[1].out_order = 1;
    g.players[2].out_order = 2;
    g.out_count = 2;
    on_table(&mut g, 3, &[j(10)]);
    g.play(0, &[DRAGON], None).unwrap();
    assert_eq!(g.pending(), Pending::DragonGiveaway(0));
    g.give_dragon(0, 1).unwrap();
    assert_eq!(g.phase, Phase::RoundEnd);
    let r = g.last_result().unwrap();
    assert_eq!(r.card_points, [0, 35]);
}

#[test]
fn one_two_finish_ends_at_once_even_with_the_dragon_on_top() {
    let mut g = state([vec![DRAGON], vec![j(3)], vec![], vec![j(4)]], 0);
    g.players[2].out_order = 1;
    g.out_count = 1;
    on_table(&mut g, 1, &[j(10)]);
    g.play(0, &[DRAGON], None).unwrap();
    assert_eq!(g.phase, Phase::RoundEnd);
    assert_eq!(g.last_result().unwrap().double_victory, Some(0));
    assert_eq!(g.last_result().unwrap().card_points, [200, 0]);
}

#[test]
fn small_tichu_only_before_first_card_and_before_anyone_is_out() {
    let mut g = state([vec![j(3), j(4)], vec![j(5)], vec![j(6)], vec![j(7)]], 0);
    for p in g.players.iter_mut() {
        p.has_played_first_card = false;
    }
    g.call_small_tichu(1).unwrap();
    assert_eq!(g.call_small_tichu(1), Err(GameError::AlreadyCalledTichu));
    g.play(0, &[j(3)], None).unwrap();
    assert!(!g.can_call_small_tichu(0));
    g.play(1, &[j(5)], None).unwrap(); // seat 1 goes out
    assert!(!g.can_call_small_tichu(2));
}

#[test]
fn clockwise_play_moves_to_the_left() {
    let mut g = state([vec![j(3), j(9)], vec![j(4)], vec![j(5)], vec![j(6)]], 0);
    g.settings.clockwise = true;
    g.play(0, &[j(3)], None).unwrap();
    assert_eq!(g.turn, 3);
}

// ---------------------------------------------------------------------------
// Whole rounds
// ---------------------------------------------------------------------------

#[test]
fn simulated_rounds_finish_and_conserve_points() {
    let mut rng = StdRng::seed_from_u64(9);
    for i in 0..300u64 {
        let mut bots: [Box<dyn Bot>; 4] = if i % 2 == 0 {
            [0, 1, 2, 3].map(|k| Box::new(RandomBot::new(i * 4 + k)) as Box<dyn Bot>)
        } else {
            [0, 1, 2, 3].map(|k| {
                if k % 2 == 0 { Box::new(HeuristicBot::new()) as Box<dyn Bot> } else { Box::new(RandomBot::new(i * 4 + k)) }
            })
        };
        let mut g = GameState::new(Settings::default());
        let r = play_round(&mut g, &mut bots, &mut rng, None).expect("round failed");
        if r.double_victory.is_none() {
            assert_eq!(r.card_points[0] + r.card_points[1], 100, "round {i}: {r:?}");
        }
    }
}

#[test]
fn from_view_rebuilds_a_consistent_state() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut g = GameState::new(Settings::default());
    g.start_round(tichu_engine::deck::shuffled_deck(&mut rng));
    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| Box::new(RandomBot::new(k)) as Box<dyn Bot>);
    // Play into the middle of the round.
    for seat in 0..4 {
        g.call_grand_tichu(seat, false).unwrap();
    }
    for seat in 0..4 {
        let pass = bots[seat].pass_cards(&g.view(seat));
        g.pass_cards(seat, pass).unwrap();
    }
    for _ in 0..12 {
        if let Pending::Turn(seat) = g.pending() {
            let plays = g.legal_plays(seat);
            g.play(seat, &plays[0].cards, plays[0].phoenix_as).unwrap();
        } else if let Pending::Wish(seat) = g.pending() {
            g.set_wish(seat, None).unwrap();
        } else if let Pending::TrickCountdown(_) = g.pending() {
            g.resolve_trick().unwrap();
        } else if let Pending::DragonGiveaway(seat) = g.pending() {
            g.give_dragon(seat, right_seat(seat)).unwrap();
        }
    }
    let view = g.view(1);
    let hands = view.sample_hands(&mut rng);
    for seat in 0..4 {
        assert_eq!(hands[seat].len(), g.players[seat].hand.len());
    }
    assert_eq!(hands[1], view.hand);
    let rebuilt = GameState::from_view(&view, hands);
    assert_eq!(rebuilt.pending(), g.pending());
    assert_eq!(rebuilt.legal_plays(1), g.legal_plays(1));
    for seat in 0..4 {
        assert_eq!(rebuilt.players[seat].captured_points(), g.players[seat].captured_points());
    }
}
