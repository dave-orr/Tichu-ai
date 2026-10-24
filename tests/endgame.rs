//! Two-player endgame solver tests.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use tichu_engine::bot::Bot;
use tichu_engine::bots::{EndgameBot, HeuristicBot, RandomBot};
use tichu_engine::combinations::identify_combo;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::endgame::{self, Decision, Node};
use tichu_engine::engine::*;
use tichu_engine::sim::{play_from, play_round};
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

/// Seats 0 and 1 hold `h0`/`h1`; seats 2 and 3 are out (`first` went out first).
fn endgame_state(h0: Vec<Card>, h1: Vec<Card>, first: Seat, turn: Seat) -> GameState {
    let mut g = GameState::mid_play(Settings::default(), [h0, h1, vec![], vec![]], turn);
    let second = if first == 2 { 3 } else { 2 };
    g.players[first].out_order = 1;
    g.players[second].out_order = 2;
    g.out_count = 2;
    g
}

fn on_table(g: &mut GameState, by: Seat, cards: &[Card]) {
    let c = identify_combo(cards, None).unwrap();
    g.trick_plays.push(TrickPlay { seat: by, combo: c.clone() });
    g.current_trick = Some(c);
    g.last_played_by = Some(by);
}

const BUDGET: usize = 5_000_000;

#[test]
fn bombs_to_take_the_last_trick() {
    // Seat 1 is about to win a King and go out; seat 0's only out is the bomb.
    let mut g = endgame_state(vec![j(9), s(9), p(9), st(9)], vec![j(14)], 3, 0);
    on_table(&mut g, 1, &[j(13)]);
    let sol = endgame::solve(&g, BUDGET).unwrap();
    assert_eq!(sol.seat, 0);
    // The bomb wins the King for team 0 (now or in the countdown, same value).
    assert_eq!(sol.value, 10);
    assert!(matches!(sol.decision, Decision::Play(ref c) if c.is_bomb()) || sol.decision == Decision::Pass);
}

#[test]
fn gives_the_dragon_to_the_opponent_who_finishes_last() {
    // Seat 2 (team 0) went out first, so the last player's tricks go to team 0:
    // the Dragon trick should go to seat 1, who will be last, not seat 3.
    let mut g = endgame_state(vec![DRAGON, j(2)], vec![j(5), s(6), p(7)], 2, 0);
    let sol = endgame::solve(&g, BUDGET).unwrap();
    assert_eq!(sol.value, 30, "25 for the Dragon + 5 left in seat 1's hand");

    g.play(0, &[DRAGON], None).unwrap();
    g.pass_turn(1).unwrap();
    let sol = endgame::solve(&g, BUDGET).unwrap();
    assert_eq!(sol.decision, Decision::GiveDragon(1));
    assert_eq!(sol.value, 30);
}

#[test]
fn not_an_endgame_is_none() {
    let g = GameState::mid_play(Settings::default(), [vec![j(2)], vec![j(3)], vec![j(4)], vec![]], 0);
    assert!(endgame::solve(&g, BUDGET).is_none());
}

/// A random two-player endgame: small hands (sometimes with a bomb or the
/// Mah Jong), seats 2/3 out with random captured points, and sometimes a
/// trick on the table.
fn random_endgame(rng: &mut StdRng, max_cards: usize) -> GameState {
    let mut deck = shuffled_deck(rng);
    let n0 = rng.gen_range(1..=max_cards);
    let n1 = rng.gen_range(1..=max_cards);
    let mut h0: Vec<Card> = deck.drain(..n0).collect();
    let mut h1: Vec<Card> = deck.drain(..n1).collect();
    if rng.gen_bool(0.3) {
        // Hand someone a four-of-a-kind.
        let r = rng.gen_range(2..=14u8);
        let quad: Vec<Card> = Suit::ALL.iter().map(|&s| Card::normal(s, r)).collect();
        h0.retain(|c| c.rank() != Some(r));
        h1.retain(|c| c.rank() != Some(r));
        deck.retain(|c| c.rank() != Some(r));
        if rng.gen_bool(0.5) { h0.extend(quad) } else { h1.extend(quad) }
    }
    for h in [&mut h0, &mut h1] {
        if h.is_empty() {
            h.push(deck.pop().unwrap());
        }
    }
    let first = if rng.gen_bool(0.5) { 2 } else { 3 };
    let turn = rng.gen_range(0..2);
    let mut g = endgame_state(h0, h1, first, turn);
    for seat in [2, 3] {
        let pile: Vec<Card> = deck.drain(..rng.gen_range(0..6)).collect();
        g.players[seat].tricks_won.push(pile);
    }
    if rng.gen_bool(0.4) {
        // The other active seat led a single from the leftover cards.
        let c = *deck.iter().find(|c| c.rank().is_some()).unwrap();
        on_table(&mut g, 1 - turn, &[c]);
    }
    // Everything outside the two hands counts as played, so a bot can deduce
    // its opponent's hand from the view.
    let held = CardSet::from_cards(&g.players[0].hand).union(CardSet::from_cards(&g.players[1].hand));
    g.played_cards = CardSet::FULL.minus(held).cards();
    g
}

#[test]
fn solver_matches_unpruned_reference() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut checked = 0;
    for _ in 0..300 {
        let g = random_endgame(&mut rng, 4);
        let Some(fast) = endgame::solve(&g, BUDGET) else { continue };
        // The unpruned search explodes on bigger hands; skip those.
        let Some(slow) = endgame::solve_reference(&g, Node::Engine, 30_000) else { continue };
        assert_eq!(fast.value, slow.value, "hands {} / {}", cards_str(&g.players[0].hand), cards_str(&g.players[1].hand));
        checked += 1;
    }
    assert!(checked > 100, "only {checked} positions checked");
}

fn endgame_bots() -> [Box<dyn Bot>; 4] {
    [0, 1, 2, 3].map(|_| Box::new(EndgameBot::new(Box::new(HeuristicBot::new()))) as Box<dyn Bot>)
}

#[test]
fn optimal_play_realizes_the_solved_value() {
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..200 {
        let g = random_endgame(&mut rng, 6);
        let Some(sol) = endgame::solve(&g, BUDGET) else { continue };
        let mut played = g.clone();
        let r = play_from(&mut played, &mut endgame_bots(), None, false).unwrap();
        assert_eq!(r.diff(), sol.value, "hands {} / {}", cards_str(&g.players[0].hand), cards_str(&g.players[1].hand));
    }
}

#[test]
fn optimal_play_realizes_the_solved_value_in_real_rounds() {
    // Play heuristic/random rounds; at the first on-turn action of a
    // two-player endgame, solve, then play out with solver bots everywhere.
    let mut rng = StdRng::seed_from_u64(13);
    let mut checked = 0;
    for i in 0..150u64 {
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| {
            if (i + k) % 3 == 0 { Box::new(RandomBot::new(i * 4 + k)) as Box<dyn Bot> } else { Box::new(HeuristicBot::new()) }
        });
        let mut found: Option<GameState> = None;
        let mut obs = |s: &GameState, a: &Action| {
            if found.is_none() && matches!(a, Action::Play { .. } | Action::PassTurn { .. }) && endgame::is_two_player_endgame(s) {
                found = Some(s.clone());
            }
        };
        let mut g = GameState::new(Settings::default());
        play_round(&mut g, &mut bots, &mut rng, Some(&mut obs)).unwrap();
        let Some(start) = found else { continue };
        let Some(sol) = endgame::solve(&start, BUDGET) else { continue };
        let mut played = start.clone();
        played.round_results.clear();
        let r = play_from(&mut played, &mut endgame_bots(), None, false).unwrap();
        assert_eq!(r.diff(), sol.value, "round {i}");
        checked += 1;
    }
    assert!(checked > 40, "only {checked} endgames checked");
}

#[test]
fn wish_choices_are_searched() {
    // Seat 0 leads the Mah Jong and must wish; check the chosen wish is at
    // least as good as every alternative.
    let mut rng = StdRng::seed_from_u64(14);
    let mut checked = 0;
    for _ in 0..200 {
        let mut g = random_endgame(&mut rng, 5);
        if g.current_trick.is_some() || g.turn != 0 {
            continue;
        }
        let other: Vec<Card> = g.players[1].hand.iter().copied().filter(|&c| c != MAH_JONG).collect();
        g.players[1].hand = other;
        if g.players[1].hand.is_empty() {
            continue;
        }
        g.players[0].hand.retain(|&c| c != MAH_JONG);
        g.players[0].hand.push(MAH_JONG);
        if g.play(0, &[MAH_JONG], None).is_err() {
            continue;
        }
        let Some(sol) = endgame::solve(&g, BUDGET) else { continue };
        let Decision::Wish(_) = sol.decision else { panic!("expected a wish decision") };
        let mut ranks: Vec<Option<Rank>> = (2..=14).map(Some).collect();
        ranks.push(None);
        ranks.shuffle(&mut rng);
        for w in ranks {
            let mut h = g.clone();
            h.set_wish(0, w).unwrap();
            if let Some(alt) = endgame::solve(&h, BUDGET) {
                assert!(alt.value <= sol.value, "wish {w:?} beats the solver's choice");
            } else if h.phase != Phase::Playing {
                assert!(h.last_result().unwrap().diff() <= sol.value);
            }
        }
        checked += 1;
    }
    assert!(checked > 20, "only {checked} checked");
}
