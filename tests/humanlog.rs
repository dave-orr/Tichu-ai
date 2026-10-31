//! `humanlog::replay` rebuilds a round from the server's round-log format.
//! Round trip: play bot rounds, write them as the server would log them
//! (plays and bombs only, no passes or countdowns), replay, and compare.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;

use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::endgame::{self, Node};
use tichu_engine::engine::{Action, GameState};
use tichu_engine::humanlog::{self, RoundLog};
use tichu_engine::sim::play_round;
use tichu_engine::types::*;

/// Play one round and return it as a server round log, plus the true final state.
fn logged_round(bot: &str, seed: u64, clockwise: bool) -> (RoundLog, GameState) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seats: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| bots::by_name(bot, seed * 4 + k).unwrap());
    let mut g = GameState::new(Settings { clockwise, ..Settings::default() });
    let mut plays = vec![];
    let mut wishes = vec![];
    let mut dragons = vec![];
    let mut trick: i32 = -1;
    let mut obs = |s: &GameState, a: &Action| match a {
        Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } => {
            if s.trick_plays.is_empty() {
                trick += 1;
            }
            plays.push(json!({"seat": seat, "cards": cards, "trick": trick}));
        }
        Action::Wish { seat, rank } => {
            if rank.is_some() {
                wishes.push(json!({"seat": seat, "rank": rank}));
            }
        }
        Action::GiveDragon { seat, to } => dragons.push(json!({"fromSeat": seat, "toSeat": to})),
        _ => {}
    };
    play_round(&mut g, &mut seats, &mut rng, Some(&mut obs)).unwrap();
    let res = g.last_result().unwrap().clone();
    let players: Vec<_> = (0..4)
        .map(|s| {
            let p = &g.players[s];
            let pass = p.pass.unwrap();
            json!({
                "seat": s, "team": s % 2, "tichuCall": p.tichu, "outOrder": p.out_order,
                "initialHand": g.deck[s * 14..s * 14 + 14],
                "passedLeft": pass.left, "passedPartner": pass.partner, "passedRight": pass.right,
            })
        })
        .collect();
    let log = json!({
        "gameId": 0, "roundNumber": 1, "scoresBeforeRound": [0, 0],
        "roundCardPoints": res.card_points, "outOrder": res.out_order,
        "players": players, "plays": plays, "mahJongWishes": wishes, "dragonGiveaways": dragons,
    });
    (serde_json::from_value(log).unwrap(), g)
}

#[test]
fn replay_rebuilds_bot_rounds() {
    let mut checked = 0;
    for (bot, clockwise) in [("random", false), ("h2", false), ("h2", true), ("heuristic", true)] {
        for seed in 0..40 {
            let (log, truth) = logged_round(bot, seed, clockwise);
            let rp = humanlog::replay(&log).unwrap_or_else(|e| panic!("{bot} seed {seed}: {e}"));
            assert!(!rp.conceded);
            assert_eq!(rp.end.last_result(), truth.last_result(), "{bot} seed {seed}");
            for s in 0..4 {
                assert_eq!(rp.end.players[s].tichu, truth.players[s].tichu);
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 160);
}

#[test]
fn truncated_log_is_a_concession() {
    let (mut log, _) = logged_round("h2", 3, false);
    let n = log.plays.len();
    log.plays.truncate(n / 2);
    let rp = humanlog::replay(&log).unwrap();
    assert!(rp.conceded);
    assert!(rp.steps.iter().filter(|s| matches!(s.action, Action::Play { .. } | Action::Bomb { .. })).count() == n / 2);
}

#[test]
fn solve_options_agrees_with_solve() {
    let mut found = 0;
    for seed in 0..25 {
        let (log, _) = logged_round("h2", seed, false);
        let rp = humanlog::replay(&log).unwrap();
        for st in rp.steps.iter().filter(|s| s.is_turn_decision() && endgame::is_two_player_endgame(&s.state)) {
            let Some(sol) = endgame::solve(&st.state, 2_000_000) else { continue };
            let Some((seat, opts)) = endgame::solve_options(&st.state, Node::Engine, 20_000_000) else { continue };
            assert_eq!(seat, sol.seat);
            let best = if team_for_seat(seat) == 0 {
                opts.iter().map(|o| o.value).max()
            } else {
                opts.iter().map(|o| o.value).min()
            };
            assert_eq!(best, Some(sol.value));
            assert_eq!(opts.len(), st.state.legal_plays(seat).len() + st.state.can_pass(seat) as usize);
            found += 1;
        }
    }
    assert!(found > 20, "only {found} endgame decisions");
}


