//! Engine invariants checked at every decision of whole simulated rounds:
//! cards are conserved, every option the engine offers is one it accepts,
//! a seat's view hides exactly the other hands, and `from_view` with the
//! true hands rebuilds the same position. These catch an engine or view
//! change that the rule tests' hand-built positions don't reach.

mod common;

use rand::rngs::StdRng;
use rand::SeedableRng;

use tichu_engine::bot::Bot;
use tichu_engine::engine::{Action, GameState, Pending};
use tichu_engine::scoring::RoundResult;
use tichu_engine::sim::play_round;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

/// Wraps a bot and, through the diagnostic `peek` hook, checks the full
/// state before each of its decisions. Test-only use of `cheats`.
struct Checker {
    inner: Box<dyn Bot>,
    checks: std::rc::Rc<std::cell::Cell<usize>>,
}

impl Bot for Checker {
    fn name(&self) -> String {
        format!("checked:{}", self.inner.name())
    }
    fn cheats(&self) -> bool {
        true
    }
    fn peek(&mut self, state: &GameState) {
        check_state(state);
        self.checks.set(self.checks.get() + 1);
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.grand_tichu(v)
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.small_tichu(v)
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        self.inner.pass_cards(v)
    }
    fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        self.inner.play(v, plays, can_pass)
    }
    fn bomb(&mut self, v: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.inner.bomb(v, bombs)
    }
    fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
        self.inner.wish(v)
    }
    fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
        self.inner.dragon_recipient(v)
    }
    fn round_over(&mut self, v: &PlayerView, r: &RoundResult) {
        self.inner.round_over(v, r)
    }
}

fn set(cards: impl IntoIterator<Item = Card>) -> CardSet {
    let mut s = CardSet::EMPTY;
    for c in cards {
        assert!(!s.contains(c), "card {c} is in two places");
        s.insert(c);
    }
    s
}

fn check_state(g: &GameState) {
    assert_eq!(set(g.deck.iter().copied()), CardSet::FULL, "the dealt deck isn't 56 distinct cards");
    if g.phase != Phase::Playing {
        for p in &g.players {
            let hand = set(p.hand.iter().copied());
            assert_eq!(hand.minus(CardSet::FULL), CardSet::EMPTY);
        }
        return;
    }

    // Every card is in exactly one place: a hand, a trick pile, or the table.
    // The Dog, once led, is discarded (as in the TS engine; it's worth 0).
    let mut all = Vec::new();
    if g.played_cards.contains(&DOG) {
        all.push(DOG);
    }
    for p in &g.players {
        all.extend(p.hand.iter().copied());
        all.extend(p.tricks_won.iter().flatten().copied());
        assert_eq!(p.is_out(), p.hand.is_empty() && p.out_order != 0, "out_order disagrees with the hand");
    }
    all.extend(g.trick_plays.iter().flat_map(|t| t.combo.cards.iter().copied()));
    assert_eq!(set(all), CardSet::FULL, "cards not conserved");
    let played = set(g.played_cards.iter().copied());
    let in_hands = set(g.players.iter().flat_map(|p| p.hand.iter().copied()));
    assert_eq!(played, CardSet::FULL.minus(in_hands), "played_cards isn't every card out of the hands");

    // Each view hides exactly the other three hands, and what it claims to
    // know about them is true.
    for seat in 0..4 {
        let v = g.view(seat);
        let others = set((0..4).filter(|&s| s != seat).flat_map(|s| g.players[s].hand.iter().copied()));
        assert_eq!(v.unseen(), others, "seat {seat}: unseen() is not the other hands");
        for (card, holder) in v.known_holdings() {
            assert!(g.players[holder].hand.contains(&card), "seat {seat} thinks {holder} holds {card}");
        }
        for (s, p) in v.players.iter().enumerate() {
            assert_eq!(p.card_count, g.players[s].hand.len());
        }
        assert!(v.log.iter().all(|a| !matches!(a, Action::PassCards { seat: s, .. } if *s != seat)), "view leaks a pass");

        // A guessed deal fits the view: right counts, my hand, known cards placed.
        let mut rng = StdRng::seed_from_u64(seat as u64);
        let guess = v.sample_hands(&mut rng);
        assert_eq!(guess[seat], v.hand);
        let guessed = set(guess.iter().flatten().copied());
        assert_eq!(guessed, in_hands, "sampled hands aren't the unplayed cards");
        for s in 0..4 {
            assert_eq!(guess[s].len(), g.players[s].hand.len());
        }
        for (card, holder) in v.known_holdings() {
            assert!(guess[holder].contains(&card), "sample misplaced a known card");
        }

        // from_view with the true hands is the same position.
        let truth = [0, 1, 2, 3].map(|s| g.players[s].hand.clone());
        let r = GameState::from_view(&v, truth);
        assert_eq!(r.pending(), g.pending(), "seat {seat}: from_view pending");
        for s in 0..4 {
            assert_eq!(r.legal_plays(s), g.legal_plays(s), "seat {seat}: from_view legal_plays({s})");
            assert_eq!(r.can_pass(s), g.can_pass(s), "seat {seat}: from_view can_pass({s})");
            assert_eq!(r.legal_bombs(s), g.legal_bombs(s), "seat {seat}: from_view legal_bombs({s})");
            assert_eq!(r.can_call_small_tichu(s), g.can_call_small_tichu(s));
            assert_eq!(r.players[s].captured_points(), g.players[s].captured_points());
        }
    }

    // Every option offered is accepted, and passing exactly when allowed.
    match g.pending() {
        Pending::Turn(seat) => {
            assert!(!g.players[seat].is_out(), "turn of a seat that is out");
            let plays = g.legal_plays(seat);
            assert!(g.can_pass(seat) || !plays.is_empty(), "seat {seat} can neither play nor pass");
            for c in &plays {
                let mut t = g.clone();
                t.apply(&Action::Play { seat, cards: c.cards.clone(), phoenix_as: c.phoenix_as })
                    .unwrap_or_else(|e| panic!("offered play {} rejected: {e}", cards_str(&c.cards)));
            }
            let mut t = g.clone();
            assert_eq!(t.apply(&Action::PassTurn { seat }).is_ok(), g.can_pass(seat), "can_pass disagrees with pass_turn");
        }
        Pending::TrickCountdown(_) => {
            let mut t = g.clone();
            t.apply(&Action::ResolveTrick).expect("countdown trick can't be resolved");
        }
        _ => {}
    }
    for seat in 0..4 {
        for b in g.legal_bombs(seat) {
            assert!(isbomb(&b), "legal_bombs offered a non-bomb");
            let mut t = g.clone();
            t.apply(&Action::Bomb { seat, cards: b.cards.clone() })
                .unwrap_or_else(|e| panic!("offered bomb {} for {seat} rejected: {e}", cards_str(&b.cards)));
        }
    }
}

fn isbomb(c: &Combo) -> bool {
    tichu_engine::combinations::is_bomb(c)
}

fn checked(inner: Box<dyn Bot>, n: &std::rc::Rc<std::cell::Cell<usize>>) -> Box<dyn Bot> {
    Box::new(Checker { inner, checks: n.clone() })
}

#[test]
fn invariants_hold_through_random_and_heuristic_rounds() {
    let n = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..150u64 {
        // Random bots reach odd corners (bombs, wishes, Tichu calls); the
        // heuristic gives realistic tricks.
        let spec = |k: u64| match i % 3 {
            0 => "random",
            1 => if k % 2 == 0 { "heuristic" } else { "random" },
            _ => "h2",
        };
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| checked(common::bot(spec(k), i * 4 + k), &n));
        let mut g = GameState::new(Settings { clockwise: i % 4 == 1, ..Settings::default() });
        let r = play_round(&mut g, &mut bots, &mut rng, None).unwrap_or_else(|e| panic!("round {i}: {e}"));
        let total: i32 = r.card_points.iter().sum();
        assert!(r.double_victory.is_some() || total == 100, "round {i}: card points {total}");
        let mut order = r.out_order.to_vec();
        order.sort();
        order.dedup();
        assert_eq!(order.len(), r.out_order.len(), "round {i}: out_order repeats a seat: {:?}", r.out_order);
    }
    assert!(n.get() > 10_000, "only {} checks ran", n.get());
}

/// Whole games: scores are the running sum of the round totals, and the game
/// ends exactly when a team reaches the target with the higher score.
#[test]
fn games_accumulate_round_totals() {
    let mut rng = StdRng::seed_from_u64(5);
    for i in 0..10u64 {
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| common::bot(if k % 2 == 0 { "heuristic" } else { "random" }, i * 4 + k));
        let g = tichu_engine::sim::play_game(Settings::default(), &mut bots, &mut rng).unwrap();
        let mut sum = [0, 0];
        for r in &g.round_results {
            sum[0] += r.total(0);
            sum[1] += r.total(1);
        }
        assert_eq!(g.scores, sum, "game {i}");
        assert_eq!(g.round_results.len() as u32, g.round_number);
        let w = g.winner().expect("finished game has a winner");
        assert!(g.scores[w] >= g.settings.target_score && g.scores[w] > g.scores[1 - w], "game {i}: {:?}", g.scores);
        // It didn't end a round late.
        let before = [sum[0] - g.round_results.last().unwrap().total(0), sum[1] - g.round_results.last().unwrap().total(1)];
        assert!(!tichu_engine::scoring::is_game_over(before, g.settings.target_score), "game {i} ran past its end");
    }
}
