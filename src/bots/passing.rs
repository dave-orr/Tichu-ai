//! v2 card passing (from Dave's rules), and a wrapper that swaps it into any
//! bot (so passing can be compared with play held fixed).
//!
//! **Opponents** get my lowest singletons (from `hand::decompose`) below 9.
//! With fewer than two, I split my lowest pair (the lowest rank I hold
//! exactly twice), one card to each. Odd cards
//! go left and even cards right; if the two lowest singletons share a parity,
//! I pass the lowest odd one left and the lowest even one right instead (if
//! both parities exist among them).
//!
//! **Partner.** If an opponent called Grand Tichu, the Mah Jong goes to the
//! seat just before the caller in play order: partner gets it, or I keep it
//! if that's me. Otherwise, I'm *strong* if my hand would meet the v2 Tichu
//! rule after this pass plus a Dragon, Phoenix or Ace from partner (or if I
//! called Grand). Strong: partner gets the Dog, else my lowest singleton,
//! else a medium card. Weak: partner gets my best card (Dragon, Phoenix, then
//! the highest card not in a bomb).

use crate::bot::Bot;
use crate::bots::calls::{self, tichu_eval};
use crate::engine::GameState;
use crate::hand::decompose;
use crate::scoring::RoundResult;
use crate::types::*;
use crate::view::PlayerView;

fn rank_counts(hand: &[Card]) -> [usize; 15] {
    let mut n = [0; 15];
    for c in hand {
        if let Some(r) = c.rank() {
            n[r as usize] += 1;
        }
    }
    n
}

fn without(hand: &[Card], gone: &[Card]) -> Vec<Card> {
    hand.iter().copied().filter(|c| !gone.contains(c)).collect()
}

fn rank_of(c: &Card) -> u8 {
    c.rank().unwrap_or(0)
}

/// The two opponent cards: (left, right).
fn opponent_cards(hand: &[Card]) -> (Card, Card) {
    let d = decompose(hand);
    let mut low: Vec<Card> =
        d.normal_singles().map(|c| c.cards[0]).filter(|c| rank_of(c) < 9).collect();
    low.sort_by_key(rank_of);
    let odd = |c: &Card| rank_of(c) % 2 == 1;
    if low.len() >= 2 {
        let (a, b) = (low[0], low[1]);
        if odd(&a) != odd(&b) {
            return if odd(&a) { (a, b) } else { (b, a) };
        }
        if let (Some(o), Some(e)) = (low.iter().find(|c| odd(c)), low.iter().find(|c| !odd(c))) {
            return (*o, *e);
        }
        // All one parity: the lower one goes to that parity's side.
        return if odd(&a) { (a, b) } else { (b, a) };
    }
    // Split the lowest pair: the lowest rank I hold exactly twice (else three times).
    let counts = rank_counts(hand);
    let pair_rank = (2..=14u8)
        .find(|&r| counts[r as usize] == 2)
        .or_else(|| (2..=14u8).find(|&r| counts[r as usize] == 3));
    if let Some(r) = pair_rank {
        let two: Vec<Card> = hand.iter().copied().filter(|c| c.rank() == Some(r)).take(2).collect();
        return (two[0], two[1]);
    }
    // No pair at all (only bombs and specials, or the like): lowest normal cards.
    let mut rest: Vec<Card> = hand.iter().copied().filter(|c| !c.is_special()).collect();
    rest.sort_by_key(|c| (counts[rank_of(c) as usize] == 4, rank_of(c)));
    match rest.len() {
        0 | 1 => {
            // Hand of specials: give away whatever isn't Dragon/Phoenix first.
            let mut all = hand.to_vec();
            all.sort_by_key(|c| match c {
                Card::Normal { rank, .. } => *rank as i32,
                Card::Special(SpecialKind::Dog) => 20,
                Card::Special(SpecialKind::MahJong) => 21,
                Card::Special(SpecialKind::Phoenix) => 22,
                Card::Special(SpecialKind::Dragon) => 23,
            });
            (all[0], all[1])
        }
        _ => (rest[0], rest[1]),
    }
}

/// My best card for a partner: Dragon, Phoenix, else the highest normal card
/// not in a four of a kind.
fn best_card(hand: &[Card]) -> Card {
    if hand.contains(&DRAGON) {
        return DRAGON;
    }
    if hand.contains(&PHOENIX) {
        return PHOENIX;
    }
    let counts = rank_counts(hand);
    hand.iter()
        .copied()
        .filter(|c| !c.is_special())
        .max_by_key(|c| (counts[rank_of(c) as usize] != 4, rank_of(c)))
        .or_else(|| hand.iter().copied().find(|c| *c != MAH_JONG))
        .unwrap_or(hand[0])
}

/// What a strong hand passes partner: the Dog, else the lowest singleton
/// below a King, else the medium card whose loss costs the fewest combos.
fn strong_card(hand: &[Card], dog_ok: bool) -> Card {
    if dog_ok && hand.contains(&DOG) {
        return DOG;
    }
    let d = decompose(hand);
    if let Some(c) = d.normal_singles().filter(|c| c.rank < 13.0).min_by_key(|c| c.rank as i32) {
        return c.cards[0];
    }
    let counts = rank_counts(hand);
    let medium = hand
        .iter()
        .copied()
        .filter(|c| (6..=11).contains(&rank_of(c)) && counts[rank_of(c) as usize] < 4)
        .min_by_key(|c| (decompose(&without(hand, &[*c])).combos.len(), (rank_of(c) as i32 - 8).abs()));
    medium
        .or_else(|| {
            hand.iter()
                .copied()
                .filter(|c| !c.is_special() && counts[rank_of(c) as usize] < 4)
                .min_by_key(rank_of)
        })
        .unwrap_or_else(|| best_card(hand))
}

/// Would `hand` (14 cards, minus the pass, plus a gift) meet the v2 Tichu rule?
fn strong_with_gift(view: &PlayerView, after_pass: &[Card]) -> bool {
    let gifts: Vec<Card> = [DRAGON, PHOENIX]
        .into_iter()
        .chain(Suit::ALL.into_iter().map(|s| Card::normal(s, 14)))
        .filter(|c| !view.hand.contains(c))
        .collect();
    let need = calls::DEFAULT_TICHU_THRESHOLD + calls::call_adjustment(view);
    let mut v = view.clone();
    let try_hand = |v: &mut PlayerView, h: Vec<Card>| {
        v.hand = h;
        tichu_eval(v).margin() >= need
    };
    if gifts.is_empty() {
        return try_hand(&mut v, after_pass.to_vec());
    }
    gifts.into_iter().any(|g| {
        let mut h = after_pass.to_vec();
        h.push(g);
        try_hand(&mut v, h)
    })
}

/// Which v2 rules to use; the parts switched off use the v1 rule
/// (`HeuristicBot::pass_cards`). For ablations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassConfig {
    /// Opponents get low singletons, odd left / even right.
    pub opp: bool,
    /// The Mah Jong goes to the seat before an opposing Grand.
    pub mj: bool,
    /// Partner gets my best card unless a gift would make me a Tichu hand.
    pub partner: bool,
    /// A Dog that isn't going to partner goes to an opponent (in place of the
    /// higher opponent card).
    pub dog: bool,
    /// Never pass the Dog to partner (a strong hand passes a singleton instead).
    pub no_dog_partner: bool,
}

impl PassConfig {
    pub const ALL: PassConfig = PassConfig { opp: true, mj: true, partner: true, dog: false, no_dog_partner: false };
    /// All rules, but the Dog always goes to an opponent (`pass2~ompdx`):
    /// what `h2` passes. Against v1 passing: +10.3 ± 2.5 per deal with h2,
    /// +2.3 ± 2.5 with heuristic (10000 deals each).
    pub const TUNED: PassConfig = PassConfig { dog: true, no_dog_partner: true, ..PassConfig::ALL };

    /// "" = all rules (o, m, p), else the letters of the rules to use (o, m,
    /// p, d, x).
    pub fn parse(s: &str) -> Option<PassConfig> {
        if s.is_empty() {
            return Some(PassConfig::ALL);
        }
        let mut c = PassConfig { opp: false, mj: false, partner: false, dog: false, no_dog_partner: false };
        for ch in s.chars() {
            match ch {
                'o' => c.opp = true,
                'm' => c.mj = true,
                'p' => c.partner = true,
                'd' => c.dog = true,
                'x' => c.no_dog_partner = true,
                _ => return None,
            }
        }
        Some(c)
    }

    fn tag(&self) -> String {
        if *self == PassConfig::ALL {
            return String::new();
        }
        let mut t = String::from("~");
        for (on, ch) in [(self.opp, 'o'), (self.mj, 'm'), (self.partner, 'p'), (self.dog, 'd'), (self.no_dog_partner, 'x')] {
            if on {
                t.push(ch);
            }
        }
        t
    }
}

/// v1 junk order: the Dog, then low lone non-point cards; specials last.
fn v1_junk_sorted(hand: &[Card], counts: &[usize; 15]) -> Vec<Card> {
    let junk = |c: &Card| -> i32 {
        match c {
            Card::Special(SpecialKind::Dog) => -10,
            Card::Special(_) => 100,
            Card::Normal { rank, .. } => {
                let r = *rank as i32;
                r + if counts[r as usize] == 1 { 0 } else { 20 } + if crate::scoring::card_points(c) > 0 { 8 } else { 0 }
            }
        }
    };
    let mut h = hand.to_vec();
    h.sort_by_key(junk);
    h
}

/// v1 opponent cards (the two junkiest): (left, right).
fn v1_opponent_cards(hand: &[Card]) -> (Card, Card) {
    let h = v1_junk_sorted(hand, &rank_counts(hand));
    (h[1], h[0])
}

/// v1 partner card from what's left after the opponent cards.
fn v1_partner_card(view: &PlayerView, rest: &[Card]) -> Card {
    let counts = rank_counts(&view.hand);
    if view.me().tichu != TichuCall::None {
        return v1_junk_sorted(rest, &counts)[0];
    }
    let partner_called = view.players[view.partner()].tichu != TichuCall::None;
    let strength = |c: &Card| -> i32 {
        match c {
            Card::Special(SpecialKind::Dragon) => if partner_called { 100 } else { 0 },
            Card::Special(SpecialKind::Phoenix) => if partner_called { 90 } else { 0 },
            Card::Special(_) => -1,
            Card::Normal { rank, .. } => if counts[*rank as usize] == 4 { 0 } else { *rank as i32 },
        }
    };
    *rest.iter().max_by_key(|c| strength(c)).unwrap()
}

pub fn pass_v2(view: &PlayerView) -> PassInfo {
    pass_with(view, PassConfig::ALL)
}

pub fn pass_with(view: &PlayerView, cfg: PassConfig) -> PassInfo {
    let hand = &view.hand;
    let partner_called = view.players[view.partner()].tichu != TichuCall::None;

    // Mah Jong against a Grand: to the seat just before the caller (partner,
    // or me, and then I keep it: opponents never get the Mah Jong anyway).
    let mj_to_partner = cfg.mj
        && hand.contains(&MAH_JONG)
        && view
            .opponents()
            .into_iter()
            .filter(|&o| view.players[o].tichu == TichuCall::Grand)
            .any(|g| view.next_seat(view.partner()) == g);

    let (left, right) = if cfg.opp { opponent_cards(hand) } else { v1_opponent_cards(hand) };
    let rest = without(hand, &[left, right]);
    let partner = if mj_to_partner {
        MAH_JONG
    } else if !cfg.partner {
        v1_partner_card(view, &rest)
    } else if partner_called {
        best_card(&rest)
    } else {
        let s = strong_card(&rest, !cfg.no_dog_partner);
        let strong = view.me().tichu == TichuCall::Grand || strong_with_gift(view, &without(&rest, &[s]));
        if strong { s } else { best_card(&rest) }
    };
    let (mut left, mut right) = (left, right);
    if cfg.dog && hand.contains(&DOG) && ![left, right, partner].contains(&DOG) {
        if rank_of(&left) > rank_of(&right) {
            left = DOG;
        } else {
            right = DOG;
        }
    }
    PassInfo { left, partner, right }
}

/// Wraps a bot, replacing its passing with `pass_with(cfg)`.
pub struct PassBot {
    cfg: PassConfig,
    inner: Box<dyn Bot>,
}

impl PassBot {
    pub fn new(cfg: PassConfig, inner: Box<dyn Bot>) -> Self {
        PassBot { cfg, inner }
    }
}

impl Bot for PassBot {
    fn name(&self) -> String {
        format!("pass2{}:{}", self.cfg.tag(), self.inner.name())
    }
    fn cheats(&self) -> bool {
        self.inner.cheats()
    }
    fn peek(&mut self, state: &GameState) {
        self.inner.peek(state)
    }
    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        self.inner.grand_tichu(view)
    }
    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        self.inner.small_tichu(view)
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        pass_with(view, self.cfg)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
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
}
