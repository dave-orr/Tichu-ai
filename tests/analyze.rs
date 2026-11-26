//! `analyze`: a bot's choice at every decision of a logged round.

mod common;

use common::server_round_log_json;

use tichu_engine::analyze::{analyze, Suggestion};
use tichu_engine::engine::{Action, Pending};
use tichu_engine::humanlog::{self, LogAction, RoundLog};
use tichu_engine::types::*;

fn same(a: &[Card], b: &[Card]) -> bool {
    CardSet::from_cards(a) == CardSet::from_cards(b)
}

/// h2 decides from its view alone, so asked about a round h2 itself played,
/// it gives back exactly what was played, decision for decision.
#[test]
fn h2_agrees_with_its_own_rounds() {
    for (seed, clockwise) in [(1, false), (2, true), (3, false), (4, true), (5, false)] {
        let (json, truth) = server_round_log_json("h2", seed, clockwise, true);
        let log: RoundLog = serde_json::from_value(json).unwrap();
        let a = analyze(&log, "h2", 3).unwrap();
        assert_eq!(a.passes.len(), 4);
        for p in &a.passes {
            assert_eq!(truth.players[p.seat].pass.unwrap().cards(), [p.left, p.partner, p.right], "seed {seed} pass {}", p.seat);
        }
        assert!(a.decisions.len() > 10);
        for d in &a.decisions {
            let ok = match (&log.actions[d.action], &d.suggestion) {
                (LogAction::Play { seat, cards, .. }, Suggestion::Play { cards: c, .. }) => *seat == d.seat && same(cards, c),
                (LogAction::Pass { seat }, Suggestion::Pass) => *seat == d.seat,
                (LogAction::Wish { seat, rank }, Suggestion::Wish { rank: r }) => *seat == d.seat && rank == r,
                (LogAction::Dragon { seat, to }, Suggestion::Dragon { to: t }) => *seat == d.seat && to == t,
                _ => false,
            };
            assert!(ok, "seed {seed}: action {} {:?} vs suggestion {:?}", d.action, log.actions[d.action], d.suggestion);
        }
    }
}

/// Every on-turn play and pass with a choice gets a suggestion, and the
/// answers don't depend on the thread count.
#[test]
fn every_turn_is_answered_deterministically() {
    let (json, _) = server_round_log_json("random", 7, false, true);
    let log: RoundLog = serde_json::from_value(json).unwrap();
    let one = analyze(&log, "pimc2", 1).unwrap();
    let four = analyze(&log, "pimc2", 4).unwrap();
    assert_eq!(one, four);
    // On-turn decisions with something playable (the rest are auto-skips).
    let rp = humanlog::replay(&log).unwrap();
    let turns = rp
        .steps
        .iter()
        .filter(|s| match (&s.action, s.state.pending()) {
            (Action::Play { .. } | Action::PassTurn { .. }, Pending::Turn(t)) => !s.state.legal_plays(t).is_empty(),
            _ => false,
        })
        .count();
    let answered = one.decisions.iter().filter(|d| matches!(d.suggestion, Suggestion::Play { .. } | Suggestion::Pass)).count();
    assert_eq!(answered, turns);
    let s = serde_json::to_value(&one).unwrap();
    assert_eq!(s["passes"].as_array().unwrap().len(), 4);
    assert!(s["decisions"][0]["suggestion"]["type"].is_string());
}

#[test]
fn rejects_unknown_bots_and_old_logs() {
    let (mut json, _) = server_round_log_json("h2", 1, false, true);
    let log: RoundLog = serde_json::from_value(json.clone()).unwrap();
    assert!(analyze(&log, "nope", 1).is_err());
    json["actions"] = serde_json::json!([]);
    let log: RoundLog = serde_json::from_value(json).unwrap();
    assert!(analyze(&log, "h2", 1).is_err());
}
