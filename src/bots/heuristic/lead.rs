use crate::engine::Action;
use crate::types::*;

use super::constants::HeuristicWeights;
use super::eval::analyze_hand;

/// Choose which action to take when leading (no current trick to beat).
/// Returns the chosen action from `legal_actions`.
pub fn choose_lead(
    hand: &[Card],
    legal_actions: &[Action],
    weights: &HeuristicWeights,
) -> Action {
    let analysis = analyze_hand(hand, weights);

    // Filter to only Play actions (no Pass when leading).
    let plays: Vec<&Action> = legal_actions.iter().filter(|a| matches!(a, Action::Play(_))).collect();

    if plays.is_empty() {
        // Shouldn't happen when leading, but fall back.
        return legal_actions[0].clone();
    }

    // If we have Dog and it makes sense to play it (give lead to partner), do so.
    if let Some(dog_action) = plays.iter().find(|a| {
        if let Action::Play(combo) = a {
            combo.cards.len() == 1 && matches!(combo.cards[0], Card::Special(SpecialKind::Dog))
        } else {
            false
        }
    }) {
        // Play dog if we have a decent hand (partner can help).
        if analysis.tempo as f64 <= weights.lead_low_tempo_threshold {
            return (*dog_action).clone();
        }
    }

    // Strong hand (low tempo): lead low singles to bait opponents into wasting cards.
    if analysis.tempo as f64 <= weights.lead_low_tempo_threshold {
        // Find the lowest single.
        let mut best: Option<&Action> = None;
        let mut best_rank = f32::MAX;
        for action in &plays {
            if let Action::Play(combo) = action {
                if combo.kind == ComboKind::Single
                    && combo.rank < best_rank
                    && combo.rank > 0.0 // skip Dog
                {
                    best_rank = combo.rank;
                    best = Some(action);
                }
            }
        }
        if let Some(action) = best {
            return (*action).clone();
        }
    }

    // Weak hand: lead our largest combo to clear the most cards at once.
    let mut best: Option<&Action> = None;
    let mut best_len = 0;
    let mut best_rank = 0.0f32;
    for action in &plays {
        if let Action::Play(combo) = action {
            // Prefer larger combos; break ties by lower rank (save high cards).
            if combo.length > best_len
                || (combo.length == best_len && combo.rank < best_rank)
            {
                best_len = combo.length;
                best_rank = combo.rank;
                best = Some(action);
            }
        }
    }

    best.unwrap_or(&plays[0]).clone()
}
