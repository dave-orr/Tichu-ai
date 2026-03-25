use crate::engine::{Action, GameState};
use crate::scoring::total_points;
use crate::types::*;

use super::constants::HeuristicWeights;

/// Choose which action to take when following (there's a current trick to beat).
pub fn choose_follow(
    state: &GameState,
    seat: Seat,
    legal_actions: &[Action],
    weights: &HeuristicWeights,
) -> Action {
    let can_pass = legal_actions.iter().any(|a| matches!(a, Action::Pass));
    let plays: Vec<&Action> = legal_actions.iter().filter(|a| matches!(a, Action::Play(_))).collect();
    let non_bomb_plays: Vec<&Action> = plays.iter().filter(|a| {
        if let Action::Play(combo) = a {
            !matches!(combo.kind, ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb)
        } else {
            false
        }
    }).copied().collect();
    let bomb_plays: Vec<&Action> = plays.iter().filter(|a| {
        if let Action::Play(combo) = a {
            matches!(combo.kind, ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb)
        } else {
            false
        }
    }).copied().collect();

    // If partner is winning the trick, pass (save our cards).
    if can_pass {
        if let Some(last) = state.last_played_by {
            if team_for_seat(last) == team_for_seat(seat) {
                return Action::Pass;
            }
        }
    }

    // Try to beat with minimum non-bomb play.
    if !non_bomb_plays.is_empty() {
        // Find the lowest-rank play (minimum beat).
        let min_play = non_bomb_plays.iter().min_by(|a, b| {
            let rank_a = if let Action::Play(c) = a { c.rank } else { 0.0 };
            let rank_b = if let Action::Play(c) = b { c.rank } else { 0.0 };
            rank_a.partial_cmp(&rank_b).unwrap()
        }).unwrap();

        return (*min_play).clone();
    }

    // No non-bomb plays available. Consider bombing.
    if !bomb_plays.is_empty() {
        let trick_points = total_points(&state.current_trick_cards);

        // Only bomb if the trick is worth it, or if an opponent is about to go out.
        let opponent_about_to_out = (0..4)
            .filter(|&s| team_for_seat(s) != team_for_seat(seat) && !state.is_out(s))
            .any(|s| state.hands[s].len() <= 2);

        if trick_points as f64 >= weights.bomb_point_threshold || opponent_about_to_out {
            // Use the weakest bomb.
            let min_bomb = bomb_plays.iter().min_by(|a, b| {
                let rank_a = if let Action::Play(c) = a { c.rank } else { 0.0 };
                let rank_b = if let Action::Play(c) = b { c.rank } else { 0.0 };
                rank_a.partial_cmp(&rank_b).unwrap()
            }).unwrap();
            return (*min_bomb).clone();
        }
    }

    // Pass if we can.
    if can_pass {
        return Action::Pass;
    }

    // Must play something.
    legal_actions[0].clone()
}
