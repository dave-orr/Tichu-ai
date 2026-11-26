//! `live::Shadow` follows a game it only sees through the server's eyes.
//!
//! Rounds are played by bots on our engine; after every action each seat's
//! shadow gets that seat's server-style state (`ClientGameState` +
//! `roundActions`, as dave-orr/tichu sends it) and must end up in step, with
//! exactly the view the real game gives that seat.

use rand::rngs::StdRng;
use rand::SeedableRng;

use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::humanlog::LogAction;
use tichu_engine::live::{ServerCountdown, ServerPlayer, ServerReceived, ServerSettings, ServerState, ServerTeam, Shadow};
use tichu_engine::sim::play_dealt_round;
use tichu_engine::types::*;

fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::GrandTichuWindow => "grandTichuWindow",
        Phase::Passing => "passing",
        Phase::Playing => "playing",
        Phase::RoundEnd => "roundEnd",
        Phase::GameEnd => "gameEnd",
    }
}

/// What the server would send `seat` in this state.
fn server_state(g: &GameState, seat: Seat) -> ServerState {
    let actions = g
        .log
        .iter()
        .filter_map(|a| {
            Some(match a {
                Action::Play { seat, cards, phoenix_as } => LogAction::Play { seat: *seat, cards: cards.clone(), phoenix_as: *phoenix_as },
                Action::Bomb { seat, cards } => LogAction::Bomb { seat: *seat, cards: cards.clone() },
                Action::PassTurn { seat } => LogAction::Pass { seat: *seat },
                Action::SmallTichu { seat } => LogAction::SmallTichu { seat: *seat },
                Action::Wish { seat, rank } => LogAction::Wish { seat: *seat, rank: *rank },
                Action::GiveDragon { seat, to } => LogAction::Dragon { seat: *seat, to: *to },
                _ => return None,
            })
        })
        .collect();
    ServerState {
        phase: phase_name(g.phase).into(),
        players: g
            .players
            .iter()
            .map(|p| ServerPlayer {
                name: String::new(),
                tichu_call: p.tichu,
                has_played_first_card: p.has_played_first_card,
                out_order: p.out_order,
                grand_tichu_decided: p.grand_decided,
                passed_cards: p.pass.is_some(),
                card_count: p.hand.len(),
                is_ai: true,
            })
            .collect(),
        teams: vec![ServerTeam { score: g.scores[0] }, ServerTeam { score: g.scores[1] }],
        current_trick: g.current_trick.as_ref().map(|c| serde_json::to_value(c).unwrap()),
        turn_index: g.turn,
        last_played_by: g.last_played_by,
        mah_jong_wish: g.wish,
        mah_jong_wish_pending: g.wish_pending,
        dragon_giveaway: g.dragon_giveaway_by.is_some(),
        dragon_giveaway_by: g.dragon_giveaway_by,
        trick_countdown: g.trick_countdown.map(|winner| ServerCountdown { winner }),
        concede_offer: None,
        settings: ServerSettings { clockwise: g.settings.clockwise, target_score: g.settings.target_score },
        round_number: g.round_number,
        round_end_ready: vec![],
        my_hand: g.players[seat].hand.clone(),
        my_seat: seat,
        my_received_cards: g.received[seat].iter().map(|&(card, from_seat)| ServerReceived { card, from_seat }).collect(),
        round_actions: Some(actions),
    }
}

fn without_grand(log: &[Action]) -> Vec<Action> {
    log.iter().filter(|a| !matches!(a, Action::GrandTichu { .. })).cloned().collect()
}

fn follow_rounds(bot: &str, rounds: u64, clockwise: bool) {
    for r in 0..rounds {
        let settings = Settings { clockwise, ..Settings::default() };
        let mut rng = StdRng::seed_from_u64(1000 + r);
        let deck = shuffled_deck(&mut rng);
        let mut g = GameState::new(settings);
        g.round_number = r as u32;
        g.scores = [(r as i32 % 7) * 50, 100];
        g.start_round(deck);
        let start = g.clone();

        let mut actions = vec![];
        let mut bots: [Box<dyn Bot>; 4] = std::array::from_fn(|s| bots::by_name(bot, r * 4 + s as u64).unwrap());
        let mut played = g.clone();
        let mut obs = |_: &GameState, a: &Action| actions.push(a.clone());
        play_dealt_round(&mut played, &mut bots, Some(&mut obs)).unwrap();

        let mut shadows: Vec<Shadow> = (0..4).map(|s| Shadow::new(s, r * 10 + s as u64)).collect();
        let mut g = start;
        for (i, s) in shadows.iter_mut().enumerate() {
            assert_eq!(s.sync(&server_state(&g, i)).unwrap(), Vec::<String>::new());
        }
        for (k, a) in actions.iter().enumerate() {
            g.apply(a).unwrap();
            if let Action::PassCards { seat, pass } = a {
                shadows[*seat].set_my_pass(*pass);
            }
            for (seat, sh) in shadows.iter_mut().enumerate() {
                let st = server_state(&g, seat);
                let warnings = sh.sync(&st).unwrap_or_else(|e| panic!("round {r} action {k} {a:?} seat {seat}: {e}"));
                assert!(warnings.is_empty(), "round {r} action {k} {a:?} seat {seat}: {warnings:?}");
                let want = g.view(seat);
                let got = sh.view().unwrap();
                assert_eq!(without_grand(&got.log), without_grand(&want.log), "round {r} action {k} seat {seat}: log");
                assert_eq!(CardSet::from_cards(&got.hand), CardSet::from_cards(&want.hand));
                assert_eq!(got.played_cards, want.played_cards, "round {r} action {k} seat {seat}: played");
                assert_eq!(got.received, want.received, "round {r} action {k} seat {seat}: received");
                assert_eq!(got.turn, want.turn);
                assert_eq!(got.scores, want.scores);
                assert_eq!(got.round_number, want.round_number);
                assert_eq!(got.players.iter().map(|p| p.card_count).collect::<Vec<_>>(), want.players.iter().map(|p| p.card_count).collect::<Vec<_>>());
                assert_eq!(got.players.iter().map(|p| p.captured_points).collect::<Vec<_>>(), want.players.iter().map(|p| p.captured_points).collect::<Vec<_>>(), "round {r} action {k} seat {seat}: points");
            }
        }
    }
}

#[test]
fn shadow_follows_random_bots() {
    follow_rounds("random", 150, false);
}

#[test]
fn shadow_follows_h2_clockwise() {
    follow_rounds("h2", 60, true);
}

/// Bots take turns at the Grand call: after every person, then in seat
/// order, and never over the partner's call.
#[test]
fn grand_calls_wait_their_turn() {
    use tichu_engine::live::decide;
    let mut rng = StdRng::seed_from_u64(7);
    let mut g = GameState::new(Settings::default());
    g.start_round(shuffled_deck(&mut rng));
    let me = 1;
    /// Calls Grand whenever asked.
    struct Eager;
    impl Bot for Eager {
        fn name(&self) -> String {
            "eager".into()
        }
        fn grand_tichu(&mut self, _: &tichu_engine::view::PlayerView) -> bool {
            true
        }
        fn pass_cards(&mut self, _: &tichu_engine::view::PlayerView) -> PassInfo {
            unreachable!()
        }
        fn play(&mut self, _: &tichu_engine::view::PlayerView, _: &[Combo], _: bool) -> Option<usize> {
            unreachable!()
        }
        fn dragon_recipient(&mut self, _: &tichu_engine::view::PlayerView) -> Seat {
            unreachable!()
        }
    }
    let mut bot = Eager;
    let not_done = |_: &str| false;
    let mut ask = |g: &GameState, humans: &[Seat]| {
        let mut s = server_state(g, me);
        for &h in humans {
            s.players[h].is_ai = false;
        }
        let mut sh = Shadow::new(me, 3);
        sh.sync(&s).unwrap();
        decide(&mut bot, &sh, &s, &not_done).map(|d| d.action)
    };
    // Seat 0 undecided, person or bot (a lower seat): wait.
    assert!(ask(&g, &[0]).is_none());
    assert!(ask(&g, &[]).is_none());
    g.apply(&Action::GrandTichu { seat: 0, call: false }).unwrap();
    // A person in seat 2 still deciding: wait.
    assert!(ask(&g, &[2]).is_none());
    // Only bots in higher seats left: my turn.
    assert_eq!(ask(&g, &[]).unwrap()["call"], true);
    // My partner (seat 3, a person) called Grand: never call over it.
    g.apply(&Action::GrandTichu { seat: 3, call: true }).unwrap();
    assert_eq!(ask(&g, &[3]).unwrap()["call"], false);
}
