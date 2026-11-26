//! `humanlog::replay` rebuilds a round from the server's round-log format.
//! Round trip: play bot rounds, write them as the server would log them
//! (plays and bombs only, no passes or countdowns), replay, and compare.

mod common;

use common::{server_round_log, server_round_log_json};
use rand::SeedableRng;
use tichu_engine::endgame::{self, Node};
use tichu_engine::engine::Action;
use tichu_engine::humanlog::{self, RoundLog};
use tichu_engine::types::*;

#[test]
fn replay_follows_logged_actions_exactly() {
    for (bot, clockwise) in [("random", false), ("h2", true), ("pimc2", false)] {
        for seed in 0..30 {
            let (json, truth) = server_round_log_json(bot, seed, clockwise, true);
            let log: RoundLog = serde_json::from_value(json).unwrap();
            let rp = humanlog::replay(&log).unwrap_or_else(|e| panic!("{bot} seed {seed}: {e}"));
            assert_eq!(rp.end.last_result(), truth.last_result(), "{bot} seed {seed}");
            // Every play-phase decision the bots made is a step, in order.
            let made: Vec<&Action> = truth.log.iter().filter(|a| !matches!(a, Action::GrandTichu { .. } | Action::PassCards { .. } | Action::ResolveTrick)).collect();
            let replayed: Vec<&Action> = rp.steps.iter().map(|s| &s.action).filter(|a| !matches!(a, Action::ResolveTrick)).collect();
            assert_eq!(made, replayed, "{bot} seed {seed}");
        }
    }
}

#[test]
fn replay_rebuilds_bot_rounds() {
    let mut checked = 0;
    for (bot, clockwise) in [("random", false), ("h2", false), ("h2", true), ("heuristic", true)] {
        for seed in 0..40 {
            let (log, truth) = server_round_log(bot, seed, clockwise);
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
    let (mut log, _) = server_round_log("h2", 3, false);
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
        let (log, _) = server_round_log("h2", seed, false);
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



#[test]
fn selfplay_exact_q_matches_the_solver() {
    use rand::rngs::StdRng;
    use tichu_engine::features::Candidates;
    let mut found = 0;
    for seed in 0..12 {
        let (log, _) = server_round_log("h2", seed, false);
        let rp = humanlog::replay(&log).unwrap();
        for st in rp.steps.iter().filter(|s| s.is_turn_decision() && endgame::is_two_player_endgame(&s.state)) {
            let Some(sol) = endgame::solve(&st.state, 2_000_000) else { continue };
            let seat = sol.seat;
            let view = st.state.view(seat);
            let plays = st.state.legal_plays(seat);
            let cands = Candidates::new(&view, &plays, st.state.can_pass(seat));
            let mut rng = StdRng::seed_from_u64(seed);
            let Some(q) = tichu_engine::selfplay::exact_q(&view, &plays, &cands, 20_000_000, &mut rng) else { continue };
            let best = q.iter().cloned().fold(f32::MIN, f32::max);
            assert_eq!(best as i32, sol.value_for(team_for_seat(seat)), "seed {seed}");
            found += 1;
        }
    }
    assert!(found > 10, "only {found}");
}
