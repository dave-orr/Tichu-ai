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
use crate::bots::{H2Bot, HeuristicBot};
use crate::infer::{sample_hands_weighted, InferConfig};
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
    /// Rollout policy: `h2` instead of `heuristic`.
    pub h2_rollouts: bool,
    /// Weight the guessed hands by the play so far (`infer`).
    pub infer: Option<InferConfig>,
    /// Search out-of-turn bombs instead of using the heuristic's rule.
    pub search_bombs: bool,
    /// Search the Mah Jong wish and the Dragon gift.
    pub search_specials: bool,
}

impl PimcBot {
    pub fn new(samples: usize, seed: u64) -> Self {
        PimcBot {
            samples,
            noise: 0.0,
            rng: StdRng::seed_from_u64(seed),
            base: HeuristicBot::new(),
            oracle: None,
            h2_rollouts: false,
            infer: None,
            search_bombs: false,
            search_specials: false,
        }
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
        let h2 = self.h2_rollouts;
        [0, 1, 2, 3].map(|_| -> Box<dyn Bot> {
            let base: Box<dyn Bot> = if h2 { Box::new(H2Bot::new()) } else { Box::new(HeuristicBot::new()) };
            if noise > 0.0 {
                Box::new(NoisyHeuristic { noise, rng: StdRng::seed_from_u64(self.rng.gen()), base })
            } else {
                base
            }
        })
    }

    /// The guessed deals to search this turn.
    fn deals(&mut self, view: &PlayerView) -> Vec<[Vec<Card>; 4]> {
        if let Some(h) = &self.oracle {
            return vec![h.clone(); self.samples];
        }
        match &self.infer {
            Some(cfg) => sample_hands_weighted(view, &mut self.rng, self.samples, cfg),
            None => (0..self.samples).map(|_| view.sample_hands(&mut self.rng)).collect(),
        }
    }
}

/// Rollout policy: the base bot, but a random legal play with probability `noise`.
struct NoisyHeuristic {
    noise: f64,
    rng: StdRng,
    base: Box<dyn Bot>,
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
        let mut flags = String::new();
        if self.h2_rollouts {
            flags.push('h');
        }
        if let Some(cfg) = &self.infer {
            flags.push('i');
            if cfg.eps_play != InferConfig::default().eps_play {
                flags.push_str(&format!("{}", (cfg.eps_play * 10.0).round()));
            }
        }
        if self.search_bombs {
            flags.push('b');
        }
        if self.search_specials {
            flags.push('s');
        }
        let flags = if flags.is_empty() { flags } else { format!("/{flags}") };
        format!("{base}{}{noise}{flags}", self.samples)
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
        // Dave's v2 rules as written: +22.3 ± 3.9 per deal over v1 passing
        // for pimc10 (4000 deals); Dog always to an opponent was +14.7 ± 3.9.
        crate::bots::passing::pass_v2(view)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        if self.search_bombs {
            return self.search_bomb(view, bombs);
        }
        self.base.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        if self.search_specials {
            return self.search_wish(view);
        }
        self.base.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        if self.search_specials {
            return self.search_dragon(view);
        }
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
        let actions: Vec<Option<Action>> = options
            .iter()
            .map(|opt| {
                Some(match opt {
                    Some(i) => Action::Play { seat: view.seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                    None => Action::PassTurn { seat: view.seat },
                })
            })
            .collect();
        let totals = self.evaluate(view, &actions);
        options[argmax(&totals)]
    }
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap()
}

/// My seat in a rollout of "don't bomb now": declines every bomb offer until
/// the first action after the decision point, then plays as `inner`.
struct DeclineFirstBomb {
    inner: Box<dyn Bot>,
}

impl Bot for DeclineFirstBomb {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        self.inner.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        // Rollout states start with an empty log (`GameState::from_view`).
        if view.log.is_empty() {
            return None;
        }
        self.inner.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.inner.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.inner.dragon_recipient(view)
    }
}

impl PimcBot {
    /// Total team margin over the guessed deals for each option: an action
    /// to apply now, or None to change nothing (only used for declining a
    /// bomb, so my seat then declines the offer it's being made).
    fn evaluate(&mut self, view: &PlayerView, options: &[Option<Action>]) -> Vec<f64> {
        let team = view.my_team();
        let mut totals = vec![0.0f64; options.len()];
        for hands in self.deals(view) {
            let base = GameState::from_view(view, hands);
            for (k, opt) in options.iter().enumerate() {
                let mut g = base.clone();
                if let Some(a) = opt {
                    g.apply(a).expect("candidate rejected by engine");
                }
                let mut bots = self.rollout_bots();
                if opt.is_none() {
                    let me = std::mem::replace(&mut bots[view.seat], Box::new(HeuristicBot::new()));
                    bots[view.seat] = Box::new(DeclineFirstBomb { inner: me });
                }
                if let Ok(r) = play_from(&mut g, &mut bots, None, false) {
                    totals[k] += (r.total(team) - r.total(1 - team)) as f64;
                }
            }
        }
        totals
    }

    fn search_bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        // Only worth a search when an opponent holds the trick.
        let by = view.last_played_by?;
        if view.is_teammate(by) {
            return None;
        }
        let mut options: Vec<Option<Action>> = vec![None];
        let mut seen = std::collections::HashSet::new();
        let mut idx = vec![];
        for (i, b) in bombs.iter().enumerate() {
            // Same-shape bombs of the same rank are interchangeable.
            if seen.insert((b.kind, b.length, (b.rank * 2.0) as i32)) {
                options.push(Some(Action::Bomb { seat: view.seat, cards: b.cards.clone() }));
                idx.push(i);
            }
        }
        let totals = self.evaluate(view, &options);
        let best = argmax(&totals);
        if best == 0 { None } else { Some(idx[best - 1]) }
    }

    fn search_wish(&mut self, view: &PlayerView) -> Option<Rank> {
        let unseen = view.unseen();
        let mut ranks: Vec<Option<Rank>> = vec![None];
        for r in 2..=14u8 {
            if Suit::ALL.iter().any(|s| unseen.contains(Card::normal(*s, r))) {
                ranks.push(Some(r));
            }
        }
        let options: Vec<Option<Action>> = ranks.iter().map(|&rank| Some(Action::Wish { seat: view.seat, rank })).collect();
        let totals = self.evaluate(view, &options);
        ranks[argmax(&totals)]
    }

    fn search_dragon(&mut self, view: &PlayerView) -> Seat {
        let opps = view.opponents();
        let options: Vec<Option<Action>> = opps.iter().map(|&to| Some(Action::GiveDragon { seat: view.seat, to })).collect();
        let totals = self.evaluate(view, &options);
        opps[argmax(&totals)]
    }
}
