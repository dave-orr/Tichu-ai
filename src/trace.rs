//! JSON round traces: the deal, every action, and before each action the
//! state and the legal moves. `tools/ts-diff` replays traces through the
//! production TypeScript engine and checks the two agree.

use serde_json::{json, Value};

use crate::bot::Bot;
use crate::engine::{GameState, Pending};
use crate::sim::play_dealt_round;
use crate::types::*;

fn ids(cards: &[Card]) -> Vec<String> {
    let mut v: Vec<String> = cards.iter().map(|c| c.id()).collect();
    v.sort();
    v
}

fn combo_json(c: &Combo) -> Value {
    json!({ "cards": ids(&c.cards), "phoenixAs": c.phoenix_as, "type": c.kind, "rank": c.rank })
}

/// The parts of the state both engines track, in a comparable form.
pub fn snapshot(s: &GameState) -> Value {
    json!({
        "phase": s.phase,
        "turn": s.turn,
        "hands": s.players.iter().map(|p| ids(&p.hand)).collect::<Vec<_>>(),
        "trick": s.current_trick.as_ref().map(|t| json!({ "type": t.kind, "rank": t.rank, "length": t.length })),
        "trickPlays": s.trick_plays.len(),
        "wish": s.wish,
        "wishPending": s.wish_pending,
        "outOrder": s.players.iter().map(|p| p.out_order).collect::<Vec<_>>(),
        "lastPlayedBy": s.last_played_by,
        "passCount": s.pass_count,
        "countdown": s.trick_countdown,
        "dragonBy": s.dragon_giveaway_by,
        "scores": s.scores,
        "captured": s.players.iter().map(|p| p.captured_points()).collect::<Vec<_>>(),
        "tricks": s.players.iter().map(|p| p.tricks_won.len()).collect::<Vec<_>>(),
        "tichu": s.players.iter().map(|p| p.tichu).collect::<Vec<_>>(),
    })
}

/// Legal moves before the next action.
fn checks(s: &GameState) -> Value {
    let turn = match s.pending() {
        Pending::Turn(seat) => json!({
            "seat": seat,
            "plays": s.legal_plays(seat).iter().map(combo_json).collect::<Vec<_>>(),
            "canPass": s.can_pass(seat),
        }),
        _ => Value::Null,
    };
    let bombs: Vec<Vec<Value>> =
        (0..4).map(|seat| s.legal_bombs(seat).iter().map(combo_json).collect()).collect();
    let small: Vec<bool> = (0..4).map(|seat| s.can_call_small_tichu(seat)).collect();
    json!({ "turn": turn, "bombs": bombs, "smallTichu": small })
}

/// Deal from `deck` and play a round with `bots`, recording a trace.
pub fn trace_round(state: &mut GameState, bots: &mut [Box<dyn Bot>; 4], deck: Vec<Card>) -> Value {
    let start_scores = state.scores;
    let round_number = state.round_number + 1;
    state.start_round(deck.clone());
    let mut steps: Vec<Value> = Vec::new();
    let result = {
        let mut obs = |s: &GameState, a: &crate::engine::Action| {
            steps.push(json!({ "before": snapshot(s), "checks": checks(s), "action": a }));
        };
        play_dealt_round(state, bots, Some(&mut obs))
    };
    json!({
        "settings": { "clockwise": state.settings.clockwise, "targetScore": state.settings.target_score },
        "scores": start_scores,
        "roundNumber": round_number,
        "deck": deck,
        "steps": steps,
        "final": snapshot(state),
        "error": result.err().map(|e| e.to_string()),
    })
}
