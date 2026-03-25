use crate::engine::Action;

/// A node in the MCTS search tree, stored in an arena (Vec).
pub struct MctsNode {
    pub action: Option<Action>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub visits: u32,
    pub total_reward: f64,
}

impl MctsNode {
    pub fn root() -> Self {
        Self {
            action: None,
            parent: None,
            children: Vec::new(),
            visits: 0,
            total_reward: 0.0,
        }
    }

    pub fn new(action: Action, parent: usize) -> Self {
        Self {
            action: Some(action),
            parent: Some(parent),
            children: Vec::new(),
            visits: 0,
            total_reward: 0.0,
        }
    }

    /// UCB1 score for selection.
    pub fn ucb1(&self, parent_visits: u32, exploration: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.total_reward / self.visits as f64;
        let exploration_term = exploration * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration_term
    }
}

/// Arena-allocated MCTS tree.
pub struct MctsTree {
    pub nodes: Vec<MctsNode>,
}

impl MctsTree {
    pub fn new() -> Self {
        Self {
            nodes: vec![MctsNode::root()],
        }
    }

    pub fn root(&self) -> usize {
        0
    }

    /// Select the best child of `node_id` using UCB1.
    /// Only considers children whose actions are in `legal_actions`.
    pub fn select_child(
        &self,
        node_id: usize,
        legal_actions: &[Action],
        exploration: f64,
    ) -> Option<usize> {
        let parent_visits = self.nodes[node_id].visits;
        let mut best_id = None;
        let mut best_score = f64::NEG_INFINITY;

        for &child_id in &self.nodes[node_id].children {
            let child = &self.nodes[child_id];
            // Only consider children whose action is legal in this determinization.
            if let Some(ref action) = child.action {
                if !action_in_list(action, legal_actions) {
                    continue;
                }
            }
            let score = child.ucb1(parent_visits, exploration);
            if score > best_score {
                best_score = score;
                best_id = Some(child_id);
            }
        }

        best_id
    }

    /// Add a new child node for the given action.
    pub fn expand(&mut self, parent_id: usize, action: Action) -> usize {
        let child_id = self.nodes.len();
        self.nodes.push(MctsNode::new(action, parent_id));
        self.nodes[parent_id].children.push(child_id);
        child_id
    }

    /// Backpropagate a reward from a leaf up to the root.
    pub fn backpropagate(&mut self, mut node_id: usize, reward: f64) {
        loop {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].total_reward += reward;
            if let Some(parent) = self.nodes[node_id].parent {
                node_id = parent;
            } else {
                break;
            }
        }
    }

    /// Get the most-visited child of the root (the best action to take).
    pub fn best_root_action(&self) -> Option<Action> {
        let root = &self.nodes[self.root()];
        root.children
            .iter()
            .max_by_key(|&&child_id| self.nodes[child_id].visits)
            .and_then(|&child_id| self.nodes[child_id].action.clone())
    }

    /// Get actions that have been expanded as children of `node_id`.
    pub fn expanded_actions(&self, node_id: usize) -> Vec<Action> {
        self.nodes[node_id]
            .children
            .iter()
            .filter_map(|&child_id| self.nodes[child_id].action.clone())
            .collect()
    }
}

/// Check if an action is in a list of legal actions.
/// Uses structural comparison since Action doesn't impl PartialEq.
fn action_in_list(action: &Action, list: &[Action]) -> bool {
    list.iter().any(|a| actions_match(a, action))
}

/// Compare two actions for structural equality.
pub fn actions_match(a: &Action, b: &Action) -> bool {
    match (a, b) {
        (Action::Pass, Action::Pass) => true,
        (Action::GiveDragonTrick(s1), Action::GiveDragonTrick(s2)) => s1 == s2,
        (Action::Play(c1), Action::Play(c2)) => {
            c1.kind == c2.kind && c1.cards == c2.cards
        }
        _ => false,
    }
}
