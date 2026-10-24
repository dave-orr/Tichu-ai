//! Tichu / Grand Tichu call rules, and a wrapper that swaps them into any bot
//! (so call policies can be compared with play held fixed).
//!
//! v2 rules (from Dave's notes):
//!
//! **Small Tichu.** Split the hand into the fewest combinations (`hand::
//! decompose`; the Phoenix joins a combination only if that removes two
//! singles). Count:
//! - *winners*, each a way to get the lead back: Dragon, a loose Phoenix,
//!   bombs, and combinations almost sure to top their type (pair of Aces, a
//!   straight to the Ace or of 8+ cards, stairs to the Ace, a triple Q+, a
//!   full house on K+). Single Aces count, less one per Dragon/Phoenix that
//!   might be with an opponent. Single Kings count, less one per Ace that
//!   might be with an opponent. Mah Jong +1 (the opening lead); the Dog +1
//!   if I passed it to my partner.
//! - *losers*: the other combinations, minus one (the last one goes out on
//!   its own). A Dog in my own hand is a loser.
//!
//! Call if winners − losers ≥ threshold. The threshold rises (stricter) by 1
//! if an opponent called Tichu, 0.5 if one called Grand, 1 if we lead by
//! 300+; it drops by 1 if we're behind and the opponents have 800+. Never
//! call over a partner's call. The decision is made on my first turn; if
//! that's following a single that no loose single of mine beats, I pass and
//! decide next turn.
//!
//! **Grand Tichu** (first 8 cards), with "top" = Dragon, Phoenix, Ace and
//! "special" = Dragon or Phoenix. Call with:
//! - 3+ tops including a special;
//! - an Ace and a special plus any of: the Mah Jong or Dog; a King; a
//!   straight or full house with no single below 9; a 6+ straight or 3+
//!   stairs;
//! - a bomb plus a special or two Aces.
//!
//! When behind with the opponents at 800+, an Ace and a special is enough;
//! when 300+ ahead, only the first and last rules apply.

use crate::bot::Bot;
use crate::bots::HeuristicBot;
use crate::engine::GameState;
use crate::hand::{decompose, Decomposition};
use crate::scoring::RoundResult;
use crate::types::*;
use crate::view::PlayerView;

/// The parts of the Small Tichu evaluation, for tuning and logging.
#[derive(Debug, Clone)]
pub struct TichuEval {
    pub winners: f32,
    pub losers: f32,
    pub decomposition: Decomposition,
}

impl TichuEval {
    pub fn margin(&self) -> f32 {
        self.winners - self.losers
    }
}

/// Is this combination nearly certain to top its type when led?
fn top_of_type(c: &Combo) -> bool {
    match c.kind {
        ComboKind::Single => c.cards == [DRAGON] || c.cards == [PHOENIX],
        ComboKind::Pair => c.rank >= 14.0,
        ComboKind::ConsecutivePairs => c.rank >= 14.0,
        ComboKind::Straight => c.rank >= 14.0 || c.length >= 8,
        ComboKind::Triple => c.rank >= 12.0,
        ComboKind::FullHouse => c.rank >= 13.0,
        ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb => true,
    }
}

pub fn tichu_eval(view: &PlayerView) -> TichuEval {
    let d = decompose(&view.hand);
    let unseen = view.unseen();
    let with_partner: Vec<Card> =
        view.known_holdings().into_iter().filter(|(_, s)| *s == view.partner()).map(|(c, _)| c).collect();
    let threat = |c: Card| unseen.contains(c) && !with_partner.contains(&c);
    let missing_specials = [DRAGON, PHOENIX].into_iter().filter(|c| threat(*c)).count();
    let unseen_aces = Suit::ALL.into_iter().filter(|s| threat(Card::normal(*s, 14))).count();

    let mut top = 0usize;
    let (mut aces, mut kings) = (0usize, 0usize);
    for c in &d.combos {
        if top_of_type(c) {
            top += 1;
        } else if c.kind == ComboKind::Single && c.rank == 14.0 {
            aces += 1;
        } else if c.kind == ComboKind::Single && c.rank == 13.0 {
            kings += 1;
        }
    }
    let ace_winners = aces.saturating_sub(missing_specials);
    let king_winners = kings.saturating_sub(unseen_aces);
    let winner_combos = top + ace_winners + king_winners;
    let mut winners = winner_combos as f32;
    if view.hand.contains(&MAH_JONG) {
        winners += 1.0;
    }
    if with_partner.contains(&DOG) {
        winners += 1.0;
    }
    let losers = (d.combos.len() - winner_combos) as f32 - 1.0;
    TichuEval { winners, losers, decomposition: d }
}

/// Score-situation adjustment to a call threshold (positive = stricter).
fn score_adjustment(view: &PlayerView) -> f32 {
    let us = view.scores[view.my_team()];
    let them = view.scores[1 - view.my_team()];
    if them >= 800 && us < them {
        -1.0
    } else if us - them >= 300 {
        1.0
    } else {
        0.0
    }
}

/// v2 Small Tichu. None = not deciding yet (ask again next turn).
pub fn small_tichu_v2(view: &PlayerView, threshold: f32) -> Option<bool> {
    let me = view.seat;
    let my_turn = view.turn == me && !view.wish_pending && view.trick_countdown.is_none() && view.dragon_giveaway_by.is_none();
    if !my_turn {
        return None;
    }
    if view.players[view.partner()].tichu != TichuCall::None {
        return Some(false);
    }
    let eval = tichu_eval(view);
    if let Some(t) = &view.current_trick {
        let beats = eval.decomposition.normal_singles().any(|c| c.rank > t.rank);
        if t.kind == ComboKind::Single && !beats {
            return None;
        }
    }
    let mut need = threshold + score_adjustment(view);
    for o in view.opponents() {
        need += match view.players[o].tichu {
            TichuCall::Small => 1.0,
            TichuCall::Grand => 0.5,
            TichuCall::None => 0.0,
        };
    }
    Some(eval.margin() >= need)
}

/// Which v2 Grand Tichu rule (if any) an 8-card hand meets. 0 = none.
pub fn grand_rule(view: &PlayerView) -> u8 {
    let h = &view.hand;
    let aces = h.iter().filter(|c| c.rank() == Some(14)).count();
    let special = h.contains(&DRAGON) || h.contains(&PHOENIX);
    let tops = aces + h.contains(&DRAGON) as usize + h.contains(&PHOENIX) as usize;
    let d = decompose(h);
    let bomb = d.combos.iter().any(|c| c.is_bomb());
    if tops >= 3 && special {
        return 1;
    }
    if bomb && (special || aces >= 2) {
        return 3;
    }
    if aces >= 1 && special {
        let extra = h.contains(&MAH_JONG)
            || h.contains(&DOG)
            || h.iter().any(|c| c.rank() == Some(13))
            || (d.combos.iter().any(|c| matches!(c.kind, ComboKind::Straight | ComboKind::FullHouse))
                && !d.combos.iter().any(|c| c.kind == ComboKind::Single && c.rank < 9.0 && c.cards != [DOG]))
            || d.combos.iter().any(|c| {
                (c.kind == ComboKind::Straight && c.length >= 6) || (c.kind == ComboKind::ConsecutivePairs && c.length >= 6)
            });
        return if extra { 2 } else { 4 };
    }
    0
}

pub fn grand_tichu_v2(view: &PlayerView) -> bool {
    let adj = score_adjustment(view);
    match grand_rule(view) {
        1 | 3 => true,
        2 => adj <= 0.0,
        4 => adj < 0.0,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CallPolicy {
    /// Never call.
    Never,
    /// The original heuristic-bot rules.
    V1,
    /// The v2 rules with this Small Tichu threshold; v2 Grand too if `grand`.
    V2 { threshold: f32, grand: bool },
    /// v2 Grand Tichu only; never Small.
    GrandOnly,
}

impl CallPolicy {
    /// "none", "v1", "g" (v2 Grand only), a number T (v2 Small with
    /// threshold T plus v2 Grand), or "sT" (v2 Small only).
    pub fn parse(s: &str) -> Option<CallPolicy> {
        match s {
            "none" => Some(CallPolicy::Never),
            "v1" => Some(CallPolicy::V1),
            "g" => Some(CallPolicy::GrandOnly),
            _ => match s.strip_prefix('s') {
                Some(t) => Some(CallPolicy::V2 { threshold: t.parse().ok()?, grand: false }),
                None => Some(CallPolicy::V2 { threshold: s.parse().ok()?, grand: true }),
            },
        }
    }
}

/// Wraps a bot, replacing its Grand/Small Tichu decisions with `policy`.
pub struct CallsBot {
    pub policy: CallPolicy,
    inner: Box<dyn Bot>,
    v1: HeuristicBot,
}

impl CallsBot {
    pub fn new(policy: CallPolicy, inner: Box<dyn Bot>) -> Self {
        CallsBot { policy, inner, v1: HeuristicBot::new() }
    }
}

impl Bot for CallsBot {
    fn name(&self) -> String {
        let p = match self.policy {
            CallPolicy::Never => "none".to_string(),
            CallPolicy::V1 => "v1".to_string(),
            CallPolicy::V2 { threshold, grand: true } => format!("{threshold}"),
            CallPolicy::V2 { threshold, grand: false } => format!("s{threshold}"),
            CallPolicy::GrandOnly => "g".to_string(),
        };
        format!("calls:{p}:{}", self.inner.name())
    }
    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        match self.policy {
            CallPolicy::Never => false,
            CallPolicy::V1 => self.v1.grand_tichu(view),
            CallPolicy::V2 { grand, .. } => grand && grand_tichu_v2(view),
            CallPolicy::GrandOnly => grand_tichu_v2(view),
        }
    }
    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        match self.policy {
            CallPolicy::Never | CallPolicy::GrandOnly => false,
            CallPolicy::V1 => self.v1.small_tichu(view),
            CallPolicy::V2 { threshold, .. } => small_tichu_v2(view, threshold).unwrap_or(false),
        }
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        // While a v2 Small Tichu decision is deferred, pass if allowed.
        if let CallPolicy::V2 { threshold: t, .. } = self.policy {
            let eligible = !view.me().has_played_first_card
                && view.me().tichu == TichuCall::None
                && !view.players.iter().any(|p| p.is_out());
            if eligible && can_pass && small_tichu_v2(view, t).is_none() {
                return None;
            }
        }
        self.inner.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.inner.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.inner.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.inner.dragon_recipient(view)
    }
    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        self.inner.round_over(view, result)
    }
    fn cheats(&self) -> bool {
        self.inner.cheats()
    }
    fn peek(&mut self, state: &GameState) {
        self.inner.peek(state)
    }
}
