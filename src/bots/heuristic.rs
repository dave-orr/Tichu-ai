//! A simple rule-based baseline. Deliberately unsophisticated: it exists so
//! there's something better than random to measure new bots against.

use crate::bot::Bot;
use crate::bots::calls;
use crate::scoring::card_points;
use crate::types::*;
use crate::view::PlayerView;

pub struct HeuristicBot;

impl HeuristicBot {
    pub fn new() -> Self {
        HeuristicBot
    }
}

impl Default for HeuristicBot {
    fn default() -> Self {
        Self::new()
    }
}

fn rank_counts(hand: &[Card]) -> [usize; 15] {
    let mut n = [0; 15];
    for c in hand {
        if let Some(r) = c.rank() {
            n[r as usize] += 1;
        }
    }
    n
}

/// How many same-rank groups a play splits (e.g. playing one 7 of a pair).
fn breaks(hand_counts: &[usize; 15], combo: &Combo) -> usize {
    let used = rank_counts(&combo.cards);
    (2..=14).filter(|&r| used[r] > 0 && used[r] < hand_counts[r]).count()
}

/// Uses a bomb's cards without playing the bomb itself.
fn breaks_bomb(hand_counts: &[usize; 15], combo: &Combo) -> bool {
    !combo.is_bomb() && combo.cards.iter().any(|c| c.rank().is_some_and(|r| hand_counts[r as usize] == 4))
}

/// The original (v1) call rules, kept for comparison (`calls:v1:BOT`).
pub fn grand_tichu_v1(view: &PlayerView) -> bool {
    controls(&view.hand) >= 4.5
}

pub fn small_tichu_v1(view: &PlayerView) -> bool {
    view.players[view.partner()].tichu == TichuCall::None
        && controls(&view.hand) >= 4.0
        && low_singles(&view.hand) <= 2
}

/// Rough count of "control" cards: ones likely to win a trick.
fn controls(hand: &[Card]) -> f32 {
    let counts = rank_counts(hand);
    let mut s = 0.0;
    for c in hand {
        s += match c {
            Card::Special(SpecialKind::Dragon) => 1.5,
            Card::Special(SpecialKind::Phoenix) => 1.0,
            Card::Normal { rank: 14, .. } => 1.0,
            Card::Normal { rank: 13, .. } => 0.4,
            _ => 0.0,
        };
    }
    s + 2.0 * (2..=14).filter(|&r| counts[r] == 4).count() as f32
}

/// Low ranks held as lone cards: each needs a trick (or a lucky lead) to shed.
fn low_singles(hand: &[Card]) -> usize {
    let counts = rank_counts(hand);
    (2..=9).filter(|&r| counts[r] == 1).count()
}

impl Bot for HeuristicBot {
    fn name(&self) -> String {
        "heuristic".into()
    }

    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        calls::grand_tichu_v2(view)
    }

    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        calls::small_tichu_v2(view, calls::DEFAULT_TICHU_THRESHOLD).unwrap_or(false)
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        let mut hand = view.hand.clone();
        let counts = rank_counts(&hand);
        let i_called = view.me().tichu != TichuCall::None;
        let partner_called = view.players[view.partner()].tichu != TichuCall::None;

        // Opponents get junk: the Dog, else the lowest lone non-point cards.
        let junk_value = |c: &Card| -> i32 {
            match c {
                Card::Special(SpecialKind::Dog) => -10,
                Card::Special(_) => 100,
                Card::Normal { rank, .. } => {
                    let r = *rank as i32;
                    let lone = if counts[r as usize] == 1 { 0 } else { 20 };
                    r + lone + if card_points(c) > 0 { 8 } else { 0 }
                }
            }
        };
        hand.sort_by_key(junk_value);
        let right = hand.remove(0);
        let left = hand.remove(0);

        // Partner: something strong, unless I'm the one going for Tichu.
        let partner = if i_called {
            hand[0]
        } else {
            let strength = |c: &Card| -> i32 {
                match c {
                    Card::Special(SpecialKind::Dragon) => if partner_called { 100 } else { 0 },
                    Card::Special(SpecialKind::Phoenix) => if partner_called { 90 } else { 0 },
                    Card::Special(_) => -1,
                    Card::Normal { rank, .. } => {
                        // Don't break a bomb.
                        if counts[*rank as usize] == 4 { 0 } else { *rank as i32 }
                    }
                }
            };
            *hand.iter().max_by_key(|c| strength(c)).unwrap()
        };
        PassInfo { left, partner, right }
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let hand = &view.hand;
        let counts = rank_counts(hand);
        let partner = view.partner();
        let opp_min_cards = view
            .opponents()
            .iter()
            .filter(|&&s| !view.players[s].is_out())
            .map(|&s| view.players[s].card_count)
            .min()
            .unwrap_or(14);
        let danger = opp_min_cards <= 3;

        // The whole hand in one go?
        if let Some(i) = plays.iter().position(|c| c.cards.len() == hand.len()) {
            return Some(i);
        }

        if view.is_leading() {
            // Hand the lead to a partner who's going for Tichu.
            if let Some(i) = plays.iter().position(|c| c.cards == [DOG]) {
                if !view.players[partner].is_out()
                    && (view.players[partner].tichu != TichuCall::None
                        || view.players[partner].card_count < hand.len())
                {
                    return Some(i);
                }
            }
            let score = |c: &Combo| -> f32 {
                let mut s = c.cards.len() as f32 * 2.0 - c.rank * 0.6;
                s -= breaks(&counts, c) as f32 * 3.0;
                if breaks_bomb(&counts, c) {
                    s -= 20.0;
                }
                if c.is_bomb() {
                    s -= 15.0;
                }
                if c.contains(SpecialKind::Phoenix) {
                    s -= if hand.len() <= 6 { 1.0 } else { 6.0 };
                }
                if c.cards == [DRAGON] {
                    s -= 10.0;
                }
                if c.cards == [DOG] {
                    s -= 8.0;
                }
                s
            };
            return best_by(plays, score);
        }

        let trick = view.current_trick.as_ref().unwrap();
        let partner_winning = view.last_played_by == Some(partner);
        if partner_winning && can_pass {
            return None;
        }
        let points = view.trick_points();
        let worth_it = points >= 10 || danger || hand.len() <= 4;
        let candidates: Vec<usize> = (0..plays.len())
            .filter(|&i| {
                let c = &plays[i];
                if c.is_bomb() {
                    // Bomb opponents only when it matters.
                    return !partner_winning && (danger || points >= 20) && !trick.is_bomb();
                }
                if breaks_bomb(&counts, c) {
                    return false;
                }
                if c.cards == [DRAGON] || c.cards == [PHOENIX] {
                    return worth_it || !can_pass;
                }
                // Keep Aces for when they count, unless the trick is already high.
                if c.kind == ComboKind::Single && c.rank >= 14.0 && trick.rank < 11.0 && !worth_it && can_pass {
                    return false;
                }
                true
            })
            .collect();
        if candidates.is_empty() {
            return if can_pass { None } else { Some(0) };
        }
        let score = |c: &Combo| -> f32 {
            let mut s = -c.rank - breaks(&counts, c) as f32 * 2.0;
            if c.contains(SpecialKind::Phoenix) && c.length > 1 {
                s -= 3.0;
            }
            if c.is_bomb() {
                s -= 10.0;
            }
            s
        };
        let best = candidates
            .iter()
            .copied()
            .max_by(|&a, &b| score(&plays[a]).partial_cmp(&score(&plays[b])).unwrap())
            .unwrap();
        Some(best)
    }

    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        let by = view.last_played_by?;
        if view.is_teammate(by) {
            return None;
        }
        let opp_cards = view.players[by].card_count;
        if view.trick_countdown.is_some() && (view.trick_points() >= 15 || opp_cards <= 2) {
            return best_by(bombs, |c| -c.rank - c.length as f32);
        }
        None
    }

    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        // Wish for a card I know an opponent holds.
        view.known_holdings()
            .iter()
            .filter(|(_, s)| !view.is_teammate(*s))
            .filter_map(|(c, _)| c.rank())
            .max()
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        let [a, b] = view.opponents();
        // The opponent with more cards is likelier to finish last, and a last
        // player's tricks go to whoever went out first.
        if view.players[a].card_count >= view.players[b].card_count { a } else { b }
    }
}

fn best_by(plays: &[Combo], score: impl Fn(&Combo) -> f32) -> Option<usize> {
    (0..plays.len()).max_by(|&a, &b| score(&plays[a]).partial_cmp(&score(&plays[b])).unwrap())
}
