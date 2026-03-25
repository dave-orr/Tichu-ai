use crate::bots::heuristic::constants::HeuristicWeights;
use crate::bots::heuristic::{follow, lead};
use crate::engine::{Action, GameState};
use crate::types::*;

/// Simulate a game from the current state to round end using heuristic policy.
/// Returns the reward from the perspective of `perspective_seat`'s team:
/// +1 if their team has more points this round, -1 if less, 0 if equal.
pub fn rollout(
    state: &mut GameState,
    perspective: Seat,
    weights: &HeuristicWeights,
) -> f64 {
    let my_team = team_for_seat(perspective);

    // Play out until the round ends.
    let mut safety = 0;
    while state.phase == Phase::Playing {
        safety += 1;
        if safety > 500 {
            // Shouldn't happen, but prevent infinite loops.
            break;
        }

        let legal = state.legal_actions();
        if legal.is_empty() {
            break;
        }

        if state.dragon_giveaway {
            // Give dragon to the opponent with the most cards.
            let seat = state.dragon_giveaway_seat.unwrap();
            let action = heuristic_dragon_giveaway(state, seat, &legal);
            if let Action::GiveDragonTrick(target) = action {
                let _ = state.give_dragon_trick(seat, target);
            }
        } else {
            let seat = state.current_player;
            let action = heuristic_choose_action(state, seat, &legal, weights);
            match &action {
                Action::Pass => {
                    let _ = state.pass_turn(seat);
                }
                Action::Play(combo) => {
                    let wish = if combo.cards.iter().any(|c| matches!(c, Card::Special(SpecialKind::MahJong))) {
                        // Simple wish: wish for a high rank.
                        Some(14u8)
                    } else {
                        None
                    };
                    let _ = state.play_cards(seat, &combo.cards, None, wish);
                }
                Action::GiveDragonTrick(_) => {
                    // Shouldn't happen outside dragon_giveaway state.
                }
            }
        }
    }

    // Evaluate the round outcome.
    if let Some(&(t0, t1)) = state.round_history.last() {
        if my_team == 0 {
            if t0 > t1 { 1.0 } else if t0 < t1 { -1.0 } else { 0.0 }
        } else {
            if t1 > t0 { 1.0 } else if t1 < t0 { -1.0 } else { 0.0 }
        }
    } else {
        // Round didn't end (safety bail) — use score differential from point counts.
        0.0
    }
}

/// Use heuristic policy to choose an action during rollout.
fn heuristic_choose_action(
    state: &GameState,
    seat: Seat,
    legal_actions: &[Action],
    weights: &HeuristicWeights,
) -> Action {
    if state.current_trick.is_none() {
        lead::choose_lead(&state.hands[seat], legal_actions, weights)
    } else {
        follow::choose_follow(state, seat, legal_actions, weights)
    }
}

/// Give dragon trick to opponent with most cards.
fn heuristic_dragon_giveaway(state: &GameState, _seat: Seat, legal_actions: &[Action]) -> Action {
    let mut best = &legal_actions[0];
    let mut most_cards = 0;
    for action in legal_actions {
        if let Action::GiveDragonTrick(target) = action {
            if state.hands[*target].len() > most_cards {
                most_cards = state.hands[*target].len();
                best = action;
            }
        }
    }
    best.clone()
}
