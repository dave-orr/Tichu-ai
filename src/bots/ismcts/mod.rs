pub mod config;
pub mod determinize;
pub mod rollout;
pub mod tree;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::bots::heuristic::constants::HeuristicWeights;
use crate::bots::heuristic::{pass_strategy, tichu_call};
use crate::engine::{Action, GameState};
use crate::player::Player;
use crate::types::*;

use config::ISMCTSConfig;
use determinize::determinize;
use rollout::rollout;
use tree::{actions_match, MctsTree};

/// Information Set Monte Carlo Tree Search bot.
///
/// Uses ISMCTS for play decisions (handles hidden information by sampling
/// consistent opponent hands). Falls back to heuristic logic for
/// grand tichu, small tichu, and passing decisions.
pub struct ISMCTSBot {
    /// ISMCTS-specific parameters.
    pub config: ISMCTSConfig,
    /// Heuristic weights used for rollout policy and non-play decisions.
    pub weights: HeuristicWeights,
}

impl ISMCTSBot {
    pub fn new(num_iterations: u32) -> Self {
        Self {
            config: ISMCTSConfig {
                num_iterations,
                ..ISMCTSConfig::default()
            },
            weights: HeuristicWeights::default(),
        }
    }

    pub fn from_config(config: ISMCTSConfig, weights: HeuristicWeights) -> Self {
        Self { config, weights }
    }

    /// Run ISMCTS and return the best action.
    fn search(&self, state: &GameState, seat: Seat, legal_actions: &[Action]) -> Action {
        let mut tree = MctsTree::new();
        let mut rng = SmallRng::from_entropy();

        for _ in 0..self.config.num_iterations {
            // 1. Determinize: sample a consistent world.
            let det_state = determinize(state, seat, &mut rng);

            // 2. Select + Expand + Rollout + Backpropagate.
            self.run_iteration(&mut tree, &det_state, seat, legal_actions);
        }

        // Return the most-visited root action.
        tree.best_root_action()
            .unwrap_or_else(|| legal_actions[0].clone())
    }

    fn run_iteration(
        &self,
        tree: &mut MctsTree,
        det_state: &GameState,
        perspective: Seat,
        root_legal_actions: &[Action],
    ) {
        let mut sim_state = det_state.clone();
        let mut node_id = tree.root();

        // Selection: walk down the tree following UCB1.
        loop {
            let legal = if node_id == tree.root() {
                root_legal_actions.to_vec()
            } else {
                sim_state.legal_actions()
            };

            if legal.is_empty() {
                break;
            }

            // Check if there are untried actions.
            let expanded = tree.expanded_actions(node_id);
            let untried: Vec<Action> = legal
                .iter()
                .filter(|a| !expanded.iter().any(|e| actions_match(a, e)))
                .cloned()
                .collect();

            if !untried.is_empty() {
                // Expansion: add one new child.
                let action = untried[0].clone();
                let child_id = tree.expand(node_id, action.clone());

                // Apply the action to the simulation state.
                apply_action(&mut sim_state, &action);

                node_id = child_id;
                break;
            }

            // All actions tried — select best child.
            if let Some(child_id) = tree.select_child(node_id, &legal, self.config.exploration) {
                let action = tree.nodes[child_id].action.clone().unwrap();
                apply_action(&mut sim_state, &action);
                node_id = child_id;

                // If the game state has moved past playing phase, stop selecting.
                if sim_state.phase != Phase::Playing {
                    break;
                }
            } else {
                break;
            }
        }

        // Rollout: simulate to end of round using heuristic policy.
        let reward = rollout(&mut sim_state, perspective, &self.weights);

        // Backpropagation.
        tree.backpropagate(node_id, reward);
    }
}

/// Apply an action to a game state (best-effort, ignoring errors).
fn apply_action(state: &mut GameState, action: &Action) {
    match action {
        Action::Pass => {
            if state.dragon_giveaway {
                return;
            }
            let seat = state.current_player;
            let _ = state.pass_turn(seat);
        }
        Action::Play(combo) => {
            let seat = state.current_player;
            let has_mahjong = combo.cards.iter().any(|c| matches!(c, Card::Special(SpecialKind::MahJong)));
            let wish = if has_mahjong { Some(14u8) } else { None };
            let _ = state.play_cards(seat, &combo.cards, None, wish);
        }
        Action::GiveDragonTrick(target) => {
            if let Some(seat) = state.dragon_giveaway_seat {
                let _ = state.give_dragon_trick(seat, *target);
            }
        }
    }
}

impl Player for ISMCTSBot {
    fn decide_grand_tichu(&mut self, state: &GameState, seat: Seat) -> bool {
        tichu_call::should_call_grand_tichu(state, seat, &self.weights)
    }

    fn decide_small_tichu(&mut self, state: &GameState, seat: Seat) -> bool {
        tichu_call::should_call_small_tichu(state, seat, &self.weights)
    }

    fn choose_pass_cards(&mut self, state: &GameState, seat: Seat) -> [Card; 3] {
        pass_strategy::choose_pass_cards(state, seat, &self.weights)
    }

    fn choose_action(
        &mut self,
        state: &GameState,
        seat: Seat,
        legal_actions: &[Action],
    ) -> (Action, Option<u8>) {
        // If only one legal action, skip the search.
        if legal_actions.len() == 1 {
            return (legal_actions[0].clone(), None);
        }

        // Handle dragon giveaway with heuristic (simple 2-choice decision).
        if state.dragon_giveaway {
            let action = heuristic_dragon_giveaway(state, legal_actions);
            return (action, None);
        }

        let action = self.search(state, seat, legal_actions);

        let wish = if let Action::Play(ref combo) = action {
            if combo.cards.iter().any(|c| matches!(c, Card::Special(SpecialKind::MahJong))) {
                Some(14u8) // wish for Ace
            } else {
                None
            }
        } else {
            None
        };

        (action, wish)
    }
}

fn heuristic_dragon_giveaway(state: &GameState, legal_actions: &[Action]) -> Action {
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
