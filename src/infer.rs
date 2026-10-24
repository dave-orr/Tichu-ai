//! Hand inference for search bots: guess the hidden hands in proportion to
//! how well each guess explains what the other players have done.
//!
//! `sample_hands_weighted` draws `k` uniform guesses per wanted sample
//! (`PlayerView::sample_hands`) and scores each by replaying the round so far
//! from the guessed hands. At every decision another seat made (a play or a
//! pass on its turn, a Small Tichu call, judged by
//! `quick_tichu_margin`), a model player (`h2`) is asked what it would have done there. A
//! disagreement multiplies the guess's weight by `eps_play` / `eps_call`.
//! The wanted samples are then drawn by weight.
//!
//! This assumes the others play roughly like `h2`. The `eps` values keep it
//! soft: a guess that disagrees a few times is down-weighted, not ruled out,
//! so the model being wrong about a player costs little.

use rand::Rng;

use crate::bot::Bot;
use crate::bots::h2::{could_beat, shape, Counts};
use crate::bots::H2Bot;
use crate::engine::{Action, GameState};
use crate::types::*;
use crate::view::PlayerView;

#[derive(Debug, Clone, Copy)]
pub struct InferConfig {
    /// Uniform guesses scored per sample returned.
    pub k: usize,
    /// Weight factor per play/pass the model would have made differently.
    pub eps_play: f64,
    /// Weight factor per Small Tichu decision the model would have made differently.
    pub eps_call: f64,
}

impl Default for InferConfig {
    fn default() -> Self {
        InferConfig { k: 4, eps_play: 0.3, eps_call: 0.2 }
    }
}

/// `n` guesses at all four hands, drawn by how well they explain the play so far.
pub fn sample_hands_weighted<R: Rng + ?Sized>(
    view: &PlayerView,
    rng: &mut R,
    n: usize,
    cfg: &InferConfig,
) -> Vec<[Vec<Card>; 4]> {
    let m = n * cfg.k.max(1);
    let cands: Vec<[Vec<Card>; 4]> = (0..m).map(|_| view.sample_hands(rng)).collect();
    if cfg.k <= 1 {
        return cands;
    }
    let weights: Vec<f64> = cands.iter().map(|h| likelihood(view, h, cfg)).collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return cands.into_iter().take(n).collect();
    }
    // Systematic resampling.
    let step = total / n as f64;
    let mut u = rng.gen::<f64>() * step;
    let mut out = Vec::with_capacity(n);
    let mut acc = 0.0;
    for (h, w) in cands.iter().zip(&weights) {
        acc += w;
        while u < acc && out.len() < n {
            out.push(h.clone());
            u += step;
        }
    }
    while out.len() < n {
        out.push(cands[0].clone());
    }
    out
}

/// What a play looks like regardless of suits (bombs keep their suits out
/// too: a different bomb of the same shape is a different hand anyway).
fn play_key(cards: &[Card], phoenix_as: Option<u8>) -> (Vec<u8>, Option<u8>) {
    let mut k: Vec<u8> = cards.iter().map(|c| c.rank().unwrap_or(100 + c.index() as u8)).collect();
    k.sort_unstable();
    (k, phoenix_as)
}

/// The relative likelihood of the round's actions so far, if the hidden hands
/// were `hands`, under the model player.
pub fn likelihood(view: &PlayerView, hands: &[Vec<Card>; 4], cfg: &InferConfig) -> f64 {
    // Each seat's hand when play began: what it holds now plus what it played.
    let mut start: [Vec<Card>; 4] = hands.clone();
    let mut first_play = 0;
    for (i, a) in view.log.iter().enumerate() {
        match a {
            Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } => {
                start[*seat].extend(cards.iter().copied())
            }
            Action::GrandTichu { .. } | Action::PassCards { .. } => first_play = i + 1,
            _ => {}
        }
    }
    if start.iter().any(|h| h.len() != 14) {
        return 1.0; // not in the playing phase yet (or a short deal): nothing to weigh
    }
    let turn = (0..4).find(|&s| start[s].contains(&MAH_JONG)).unwrap_or(0);
    let mut g = GameState::mid_play(view.settings, start, turn);
    g.scores = view.scores;
    g.round_number = view.round_number;
    for s in 0..4 {
        g.players[s].has_played_first_card = false;
    }
    for a in &view.log[..first_play] {
        match a {
            Action::GrandTichu { seat, call: true } => g.players[*seat].tichu = TichuCall::Grand,
            Action::SmallTichu { seat } => g.players[*seat].tichu = TichuCall::Small,
            _ => {}
        }
    }

    let mut model = H2Bot::new();
    let mut w = 1.0f64;
    for a in &view.log[first_play..] {
        match a {
            Action::SmallTichu { seat } if *seat != view.seat => {
                let m = quick_tichu_margin(&g.players[*seat].hand);
                if m < -1.0 {
                    w *= cfg.eps_call;
                } else if m < 0.0 {
                    w *= cfg.eps_call.sqrt();
                }
            }
            Action::Play { seat, cards, phoenix_as } if *seat != view.seat => {
                let s = *seat;
                let plays = g.legal_plays(s);
                let can_pass = g.can_pass(s);
                if plays.len() + can_pass as usize > 1 {
                    let agree = match model.play(&g.view_without_log(s), &plays, can_pass) {
                        Some(i) => plays
                            .get(i)
                            .is_some_and(|c| play_key(&c.cards, c.phoenix_as) == play_key(cards, *phoenix_as)),
                        None => false,
                    };
                    if !agree {
                        w *= cfg.eps_play;
                    }
                }
            }
            Action::PassTurn { seat } if *seat != view.seat => {
                let s = *seat;
                let plays = g.legal_plays(s);
                if !plays.is_empty() && model.play(&g.view_without_log(s), &plays, true).is_some() {
                    w *= cfg.eps_play;
                }
            }
            _ => {}
        }
        if g.apply(a).is_err() {
            return 0.0;
        }
    }
    w
}

/// A fast stand-in for the v2 Tichu margin (`calls::tichu_eval`, whose exact
/// split is too slow to run on every guessed deal): winners are the Dragon,
/// a loose Phoenix, the Mah Jong (the opening lead) and combinations nothing
/// else in the deck could beat; losers the other combinations, less one.
pub fn quick_tichu_margin(hand: &[Card]) -> f32 {
    let mine = Counts::of(hand.iter().copied());
    let others = Counts::of(CardSet::FULL.minus(CardSet::from_cards(hand)).cards());
    let sh = shape(&mine);
    let winners = sh.combos.iter().filter(|c| !could_beat(c, &others)).count();
    let losers = sh.combos.len() - winners + mine.dog as usize;
    let w = winners as f32 + mine.dragon as u8 as f32 + mine.phoenix as u8 as f32 + (mine.n[1] > 0) as u8 as f32;
    w - (losers as f32 - 1.0)
}
