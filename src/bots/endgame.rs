//! `EndgameBot` wraps any bot and takes over once only two players hold
//! cards: it deduces the opponent's hand (every unseen card) and plays the
//! rest of the round with the exact solver in `crate::endgame`. If the search
//! exceeds its node budget, it falls back to the wrapped bot.

use crate::bot::Bot;
use crate::endgame::{self, Decision, Node, Solution};
use crate::engine::GameState;
use crate::types::*;
use crate::view::PlayerView;

pub struct EndgameBot {
    inner: Box<dyn Bot>,
    pub budget: usize,
}

impl EndgameBot {
    pub const DEFAULT_BUDGET: usize = 2_000_000;

    pub fn new(inner: Box<dyn Bot>) -> Self {
        EndgameBot { inner, budget: Self::DEFAULT_BUDGET }
    }

    /// The exact state, if this is a two-player endgame I'm in.
    pub fn exact_state(view: &PlayerView) -> Option<GameState> {
        if view.phase != Phase::Playing || view.players.iter().filter(|p| !p.is_out()).count() != 2 {
            return None;
        }
        if view.players[view.seat].is_out() {
            return None;
        }
        let other = (0..4).find(|&s| s != view.seat && !view.players[s].is_out())?;
        let unseen = view.unseen().cards();
        if unseen.len() != view.players[other].card_count {
            return None;
        }
        let mut hands: [Vec<Card>; 4] = Default::default();
        hands[view.seat] = view.hand.clone();
        hands[other] = unseen;
        Some(GameState::from_view(view, hands))
    }

    fn solve(&self, view: &PlayerView, node: Node) -> Option<Solution> {
        let state = Self::exact_state(view)?;
        let sol = endgame::solve_node(&state, node, self.budget)?;
        (sol.seat == view.seat).then_some(sol)
    }
}

fn same_play(a: &Combo, b: &Combo) -> bool {
    a.card_set() == b.card_set() && a.rank == b.rank
}

/// Same ranks/specials and rank, ignoring suits (for suit-collapsed solver moves).
fn equivalent_play(a: &Combo, b: &Combo) -> bool {
    let sig = |c: &Combo| {
        let mut v: Vec<i32> = c.cards.iter().map(|x| x.rank().map(|r| r as i32).unwrap_or(-(x.index() as i32))).collect();
        v.sort_unstable();
        v
    };
    a.kind == b.kind && a.rank == b.rank && sig(a) == sig(b)
}

impl Bot for EndgameBot {
    fn name(&self) -> String {
        format!("eg:{}", self.inner.name())
    }

    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        self.inner.grand_tichu(view)
    }

    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        self.inner.small_tichu(view)
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if let Some(sol) = self.solve(view, Node::Engine) {
            match &sol.decision {
                Decision::Pass if can_pass => return None,
                Decision::Play(c) => {
                    let exact = plays.iter().position(|p| same_play(p, c));
                    if let Some(i) = exact.or_else(|| plays.iter().position(|p| equivalent_play(p, c))) {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        self.inner.play(view, plays, can_pass)
    }

    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        if let Some(state) = Self::exact_state(view) {
            // Rebuild the offer the simulator is making to us.
            let node = match (view.trick_countdown, view.last_played_by) {
                (Some(w), _) => Some(Node::countdown(&state, w)),
                (None, Some(by)) => Some(Node::after_play(&state, by)),
                _ => None,
            };
            if let Some(node) = node {
                let sol = endgame::solve_node(&state, node.from_seat(view.seat), self.budget);
                if let Some(sol) = sol.filter(|s| s.seat == view.seat) {
                    return match &sol.decision {
                        Decision::Bomb(b) => bombs.iter().position(|x| same_play(x, b)),
                        _ => None,
                    };
                }
            }
        }
        self.inner.bomb(view, bombs)
    }

    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        match self.solve(view, Node::Engine).map(|s| s.decision) {
            Some(Decision::Wish(w)) => w,
            _ => self.inner.wish(view),
        }
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        match self.solve(view, Node::Engine).map(|s| s.decision) {
            Some(Decision::GiveDragon(to)) => to,
            _ => self.inner.dragon_recipient(view),
        }
    }

    fn round_over(&mut self, view: &PlayerView, result: &crate::scoring::RoundResult) {
        self.inner.round_over(view, result)
    }
}
