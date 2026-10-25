//! The round so far, replayed from one seat's `PlayerView`: a token per
//! event for the sequence encoder (`bots::net`'s GRU), and fixed-size
//! summaries for `features` (declined beats, bombs thrown, late calls…).
//!
//! Tokens are built from the log in order and each depends only on what
//! came before it, so the tokens of an earlier view of the same seat are a
//! prefix of a later view's. `selfplay` relies on that: it writes each
//! seat's whole-round sequence once and gives every decision its prefix
//! length.
//!
//! Seats are relative, in play order, as in `features` (0 = me).

use crate::combinations::identify_combo;
use crate::engine::Action;
use crate::features::rel;
use crate::scoring::total_points;
use crate::types::*;
use crate::view::PlayerView;

pub const TOKEN_DIM: usize = 48;

/// Event types (token bytes 4..13).
#[derive(Clone, Copy)]
enum Ev {
    Play = 0,
    PassTurn = 1,
    Bomb = 2,
    SmallTichu = 3,
    GrandTichu = 4,
    Wish = 5,
    GiveDragon = 6,
    TrickWon = 7,
    /// A card I passed (actor me, target the recipient) or received (actor the sender).
    PassCard = 8,
}

const KINDS: [ComboKind; 8] = [
    ComboKind::Single,
    ComboKind::Pair,
    ComboKind::Triple,
    ComboKind::FullHouse,
    ComboKind::Straight,
    ComboKind::ConsecutivePairs,
    ComboKind::FourOfAKindBomb,
    ComboKind::StraightFlushBomb,
];

fn kind_index(k: ComboKind) -> usize {
    KINDS.iter().position(|&x| x == k).unwrap()
}

fn byte(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// What `features` reads from the history.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    /// Per relative seat and non-bomb kind: the lowest table rank the seat
    /// passed on while an opponent of theirs held the trick (0 = never).
    pub declined: [[f32; 6]; 4],
    /// Such passes, per relative seat.
    pub declines: [u32; 4],
    /// Bombs thrown, per relative seat.
    pub bombs: [u32; 4],
    /// Small Tichu called after the first card of the round was played.
    pub late_tichu: [bool; 4],
    /// Relative seat that received a Dragon trick.
    pub dragon_to: Option<usize>,
    /// The Mah Jong wish, once made (also after it's fulfilled).
    pub wished: Option<Rank>,
}

pub struct History {
    pub tokens: Vec<u8>,
    pub summary: Summary,
}

impl History {
    pub fn len(&self) -> usize {
        self.tokens.len() / TOKEN_DIM
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

struct Builder<'a> {
    view: &'a PlayerView,
    tokens: Vec<u8>,
    /// The combination on the table and who played it.
    table: Option<(Combo, Seat)>,
    trick_points: i32,
    trick_no: usize,
    cards_left: [usize; 4],
}

impl Builder<'_> {
    fn push(&mut self, actor: Seat, ev: Ev, combo: Option<&Combo>, cards: &[Card], target: Option<Seat>, extra: f32) {
        let mut t = [0u8; TOKEN_DIM];
        t[rel(self.view, actor)] = 255;
        t[4 + ev as usize] = 255;
        if let Some(c) = combo {
            t[13 + kind_index(c.kind)] = 255;
            t[21] = byte(c.rank / 16.0);
            t[22] = byte(c.length as f32 / 14.0);
        } else {
            t[21] = byte(extra / 16.0);
        }
        for c in cards {
            match c {
                Card::Normal { rank, .. } => t[23 + *rank as usize - 2] = t[23 + *rank as usize - 2].saturating_add(64),
                Card::Special(SpecialKind::MahJong) => t[36] = 255,
                Card::Special(SpecialKind::Dog) => t[37] = 255,
                Card::Special(SpecialKind::Phoenix) => t[38] = 255,
                Card::Special(SpecialKind::Dragon) => t[39] = 255,
            }
        }
        t[40] = if self.table.is_none() { 255 } else { 0 };
        t[41] = match self.table {
            Some((_, by)) if by != actor && partner_seat(by) == actor => 255,
            _ => 0,
        };
        t[42] = byte((total_points(cards) as f32 + 25.0) / 125.0);
        t[43] = byte(self.trick_no as f32 / 20.0);
        t[44] = byte(self.cards_left[actor] as f32 / 14.0);
        if let Some(to) = target {
            let r = rel(self.view, to);
            if r > 0 {
                t[44 + r] = 255;
            }
        }
        self.tokens.extend_from_slice(&t);
    }
}

/// Replay `view.log` (plus `view.received`, placed where play begins).
pub fn replay(view: &PlayerView) -> History {
    let mut b = Builder {
        view,
        tokens: Vec::with_capacity(TOKEN_DIM * 160),
        table: None,
        trick_points: 0,
        trick_no: 0,
        cards_left: [14; 4],
    };
    let mut s = Summary::default();
    let mut received_done = false;
    let mut played_any = false;
    for a in &view.log {
        let in_play = !matches!(a, Action::GrandTichu { .. } | Action::PassCards { .. });
        if in_play && !received_done && !view.received.is_empty() {
            received_done = true;
            for &(card, from) in &view.received {
                b.push(from, Ev::PassCard, None, &[card], None, 0.0);
            }
        }
        match a {
            Action::GrandTichu { seat, call } => {
                if *call {
                    b.push(*seat, Ev::GrandTichu, None, &[], None, 0.0);
                }
            }
            Action::SmallTichu { seat } => {
                if played_any {
                    s.late_tichu[rel(view, *seat)] = true;
                }
                b.push(*seat, Ev::SmallTichu, None, &[], None, 0.0);
            }
            Action::PassCards { seat, pass } => {
                for c in pass.cards() {
                    b.push(*seat, Ev::PassCard, None, &[c], pass.recipient(*seat, c), 0.0);
                }
            }
            Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } if !cards.is_empty() => {
                played_any = true;
                let bomb = matches!(a, Action::Bomb { .. });
                let pa = if let Action::Play { phoenix_as, .. } = a { *phoenix_as } else { None };
                let mut combo = identify_combo(cards, pa);
                if let (Some(c), Some((t, _))) = (combo.as_mut(), b.table.as_ref()) {
                    // A lone Phoenix is worth half a rank over the table.
                    if c.kind == ComboKind::Single && c.cards == [PHOENIX] {
                        c.rank = t.rank + 0.5;
                    }
                }
                b.cards_left[*seat] = b.cards_left[*seat].saturating_sub(cards.len());
                let is_bomb = bomb || combo.as_ref().is_some_and(|c| c.is_bomb());
                if is_bomb {
                    s.bombs[rel(view, *seat)] += 1;
                }
                let ev = if is_bomb { Ev::Bomb } else { Ev::Play };
                b.push(*seat, ev, combo.as_ref(), cards, None, 0.0);
                b.trick_points += total_points(cards);
                if cards == &[DOG] {
                    // The Dog hands the lead on and is gone.
                    b.table = None;
                    b.trick_points = 0;
                    b.trick_no += 1;
                } else if let Some(c) = combo {
                    b.table = Some((c, *seat));
                }
            }
            Action::Play { .. } | Action::Bomb { .. } => {}
            Action::PassTurn { seat } => {
                if let Some((t, by)) = &b.table {
                    if team_for_seat(*by) != team_for_seat(*seat) && !t.is_bomb() {
                        let r = rel(view, *seat);
                        let k = kind_index(t.kind);
                        let v = &mut s.declined[r][k];
                        if *v == 0.0 || t.rank < *v {
                            *v = t.rank;
                        }
                        s.declines[r] += 1;
                    }
                }
                b.push(*seat, Ev::PassTurn, None, &[], None, 0.0);
            }
            Action::Wish { seat, rank } => {
                if rank.is_some() {
                    s.wished = *rank;
                }
                b.push(*seat, Ev::Wish, None, &[], None, rank.map_or(0.0, |r| r as f32));
            }
            Action::ResolveTrick => {
                if let Some((_, by)) = b.table {
                    let pts = b.trick_points;
                    b.push(by, Ev::TrickWon, None, &[], None, 0.0);
                    // The points ride in the token's points byte.
                    let last = b.tokens.len() - TOKEN_DIM;
                    b.tokens[last + 42] = byte((pts as f32 + 25.0) / 125.0);
                }
                b.table = None;
                b.trick_points = 0;
                b.trick_no += 1;
            }
            Action::GiveDragon { seat, to } => {
                s.dragon_to = Some(rel(view, *to));
                b.push(*seat, Ev::GiveDragon, None, &[], Some(*to), 0.0);
            }
        }
    }
    if !received_done && !view.received.is_empty() {
        for &(card, from) in &view.received {
            b.push(from, Ev::PassCard, None, &[card], None, 0.0);
        }
    }
    History { tokens: b.tokens, summary: s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::Bot;
    use crate::bots::{H2Bot, RandomBot};
    use crate::deck::shuffled_deck;
    use crate::engine::GameState;
    use crate::sim::play_dealt_round;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Records the token sequence at every decision, and the final one.
    struct Watch(Box<dyn Bot>, Rc<RefCell<Vec<Vec<u8>>>>, Rc<RefCell<Option<Vec<u8>>>>);
    impl Bot for Watch {
        fn name(&self) -> String {
            "watch".into()
        }
        fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
            self.0.pass_cards(v)
        }
        fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
            self.1.borrow_mut().push(replay(v).tokens);
            self.0.dragon_recipient(v)
        }
        fn bomb(&mut self, v: &PlayerView, b: &[Combo]) -> Option<usize> {
            self.0.bomb(v, b)
        }
        fn small_tichu(&mut self, v: &PlayerView) -> bool {
            self.0.small_tichu(v)
        }
        fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
            self.0.wish(v)
        }
        fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
            self.1.borrow_mut().push(replay(v).tokens);
            self.0.play(v, plays, can_pass)
        }
        fn round_over(&mut self, v: &PlayerView, _r: &crate::scoring::RoundResult) {
            *self.2.borrow_mut() = Some(replay(v).tokens);
        }
    }

    #[test]
    fn earlier_sequences_are_prefixes() {
        let mut rng = StdRng::seed_from_u64(9);
        for i in 0..40 {
            let mut state = GameState::new(Settings { clockwise: i % 2 == 1, ..Settings::default() });
            state.start_round(shuffled_deck(&mut rng));
            let seen: Vec<_> = (0..4).map(|_| Rc::new(RefCell::new(vec![]))).collect();
            let fin: Vec<_> = (0..4).map(|_| Rc::new(RefCell::new(None))).collect();
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                let inner: Box<dyn Bot> = if i % 3 == 0 { Box::new(RandomBot::new(s as u64)) } else { Box::new(H2Bot::new()) };
                Box::new(Watch(inner, seen[s].clone(), fin[s].clone()))
            });
            play_dealt_round(&mut state, &mut bots, None).unwrap();
            for s in 0..4 {
                let full = fin[s].borrow().clone().unwrap();
                assert_eq!(full.len() % TOKEN_DIM, 0);
                for t in seen[s].borrow().iter() {
                    assert!(full.starts_with(t), "seat {s}: a decision's tokens aren't a prefix of the round's");
                }
            }
        }
    }

    #[test]
    fn summary_tracks_declines_and_bombs() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut any_decline = false;
        for _ in 0..20 {
            let mut state = GameState::new(Settings::default());
            state.start_round(shuffled_deck(&mut rng));
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(H2Bot::new()) as Box<dyn Bot>);
            play_dealt_round(&mut state, &mut bots, None).unwrap();
            let v = state.view(0);
            let h = replay(&v);
            let bombs: u32 = h.summary.bombs.iter().sum();
            let logged = v.log.iter().filter(|a| match a {
                Action::Bomb { .. } => true,
                Action::Play { cards, phoenix_as, .. } => identify_combo(cards, *phoenix_as).is_some_and(|c| c.is_bomb()),
                _ => false,
            });
            assert_eq!(bombs as usize, logged.count());
            any_decline |= h.summary.declines.iter().any(|&d| d > 0);
            for r in 0..4 {
                for &x in &h.summary.declined[r] {
                    assert!(x == 0.0 || (1.0..=15.5).contains(&x));
                }
            }
        }
        assert!(any_decline);
    }
}
