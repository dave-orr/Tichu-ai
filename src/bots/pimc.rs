//! An example search bot: Perfect-Information Monte Carlo (PIMC).
//!
//! On each turn it samples `samples` guesses at the hidden hands
//! (`PlayerView::sample_hands`), rebuilds the full state for each
//! (`GameState::from_view`), and for every candidate play simulates the rest
//! of the round with heuristic bots in all four seats. It picks the play with
//! the best average team margin. Everything else (calls, passing, bombs,
//! wishes, the Dragon) is delegated to the heuristic bot.
//!
//! `PimcBot::oracle()` is the cheating version: it is shown the real hands
//! (`Bot::cheats`) and simulates from those instead of guesses.
//!
//! With `noise` > 0, each rollout player makes a uniformly random legal play
//! instead of the heuristic one with that probability. That keeps the search
//! from over-trusting a single deterministic opponent model. It also makes
//! `oracleN@x` vs `pimcN@x` a clean measure of what knowing the hands is
//! worth: same rollouts, same sample count, only the hands differ.
//!
//! It's slow and naive (uniform sampling, a weak rollout policy, no reuse
//! between turns) — a starting point, not a strong player.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::bots::HeuristicBot;
use crate::engine::{Action, GameState};
use crate::sim::play_from;
use crate::types::*;
use crate::view::PlayerView;

pub struct PimcBot {
    pub samples: usize,
    /// Chance that a rollout player plays a random legal move.
    pub noise: f64,
    rng: StdRng,
    base: HeuristicBot,
    /// Some(true hands) for the oracle, refreshed before every decision.
    oracle: Option<[Vec<Card>; 4]>,
}

impl PimcBot {
    pub fn new(samples: usize, seed: u64) -> Self {
        PimcBot { samples, noise: 0.0, rng: StdRng::seed_from_u64(seed), base: HeuristicBot::new(), oracle: None }
    }

    /// Searches the real deal, `samples` rollouts per candidate (with
    /// noise 0 the rollouts are deterministic, so 1 is exact).
    pub fn oracle(samples: usize, seed: u64) -> Self {
        PimcBot { oracle: Some(Default::default()), ..PimcBot::new(samples, seed) }
    }

    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    fn rollout_bots(&mut self) -> [Box<dyn Bot>; 4] {
        let noise = self.noise;
        [0, 1, 2, 3].map(|_| -> Box<dyn Bot> {
            if noise > 0.0 {
                Box::new(NoisyHeuristic { noise, rng: StdRng::seed_from_u64(self.rng.gen()), base: HeuristicBot::new() })
            } else {
                Box::new(HeuristicBot::new())
            }
        })
    }
}

/// Rollout policy: the heuristic, but a random legal play with probability `noise`.
struct NoisyHeuristic {
    noise: f64,
    rng: StdRng,
    base: HeuristicBot,
}

impl Bot for NoisyHeuristic {
    fn name(&self) -> String {
        "noisy-heuristic".into()
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.base.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if self.rng.gen_bool(self.noise) {
            let n = plays.len() + can_pass as usize;
            let k = self.rng.gen_range(0..n);
            return if k < plays.len() { Some(k) } else { None };
        }
        self.base.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.base.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.base.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.base.dragon_recipient(view)
    }
}

/// Plays that differ only in suits (and aren't bombs) are interchangeable for
/// search purposes; keep one of each.
fn distinct_candidates(plays: &[Combo]) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    (0..plays.len())
        .filter(|&i| {
            let c = &plays[i];
            let mut ranks: Vec<i32> =
                c.cards.iter().map(|x| x.rank().map(|r| r as i32).unwrap_or(-(x.index() as i32))).collect();
            ranks.sort_unstable();
            c.is_bomb() || seen.insert((c.kind, (c.rank * 2.0) as i32, ranks))
        })
        .collect()
}

impl Bot for PimcBot {
    fn name(&self) -> String {
        let base = if self.oracle.is_some() { "oracle" } else { "pimc" };
        let noise = if self.noise > 0.0 { format!("@{}", self.noise) } else { String::new() };
        format!("{base}{}{noise}", self.samples)
    }

    fn cheats(&self) -> bool {
        self.oracle.is_some()
    }

    fn peek(&mut self, state: &GameState) {
        self.oracle = Some([0, 1, 2, 3].map(|s| state.players[s].hand.clone()));
    }

    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        self.base.grand_tichu(view)
    }
    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        self.base.small_tichu(view)
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.base.pass_cards(view)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.base.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.base.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.base.dragon_recipient(view)
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        // Options: None = pass, Some(i) = plays[i].
        let mut options: Vec<Option<usize>> = distinct_candidates(plays).into_iter().map(Some).collect();
        if can_pass {
            options.push(None);
        }
        if options.len() == 1 {
            return options[0];
        }
        let team = view.my_team();
        let mut totals = vec![0.0f64; options.len()];
        for _ in 0..self.samples {
            let hands = match &self.oracle {
                Some(h) => h.clone(),
                None => view.sample_hands(&mut self.rng),
            };
            let base = GameState::from_view(view, hands);
            for (k, opt) in options.iter().enumerate() {
                let mut g = base.clone();
                let action = match opt {
                    Some(i) => Action::Play { seat: view.seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                    None => Action::PassTurn { seat: view.seat },
                };
                g.apply(&action).expect("candidate rejected by engine");
                let mut bots = self.rollout_bots();
                if let Ok(r) = play_from(&mut g, &mut bots, None, false) {
                    totals[k] += (r.total(team) - r.total(1 - team)) as f64;
                }
            }
        }
        let best = (0..options.len()).max_by(|&a, &b| totals[a].partial_cmp(&totals[b]).unwrap()).unwrap();
        options[best]
    }
}
