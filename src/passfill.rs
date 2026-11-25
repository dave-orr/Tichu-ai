//! Pass knowledge for the other seats of a state rebuilt from one seat's
//! view (`GameState::from_view`).
//!
//! A rebuilt state knows only the viewer's own pass and what the viewer
//! received, so the other three players have forgotten what they passed and
//! were passed. Bots that read that (the learned nets' "card each seat passed
//! me" features, card tracking through `known_holdings`) then play
//! differently from how they would in the real round: with the real hands
//! put back, a rebuilt state played out like the real one in only 37% of
//! decisions (`examples/critic_ceiling.rs`).
//!
//! `guess_passes` fills the gap. Part is known for sure: the card the viewer
//! gave each seat, and the card each seat gave the viewer. The six passes
//! among the other three are guessed: each is a card the recipient holds or
//! has played, drawn by how typical it is for a pass to a partner or to an
//! opponent (`PassModel`, counted from h2's passing).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::bots::H2Bot;
use crate::deck::shuffled_deck;
use crate::engine::{Action, GameState};
use crate::types::*;
use crate::view::PlayerView;

fn class(c: Card) -> usize {
    match c {
        Card::Normal { rank, .. } => rank as usize - 2,
        Card::Special(SpecialKind::MahJong) => 13,
        Card::Special(SpecialKind::Dog) => 14,
        Card::Special(SpecialKind::Phoenix) => 15,
        Card::Special(SpecialKind::Dragon) => 16,
    }
}

/// How much more often a card of each class is the one passed (to a
/// partner, to an opponent) than a card the recipient just happens to hold.
pub struct PassModel {
    /// `ratio[is_partner][class]`.
    pub ratio: [[f64; 17]; 2],
}

impl PassModel {
    /// Counted over `deals` deals passed by h2 (after h2's Grand calls).
    pub fn from_h2(deals: usize, seed: u64) -> PassModel {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut passed = [[1.0f64; 17]; 2];
        let mut held = [1.0f64; 17];
        for _ in 0..deals {
            let mut g = GameState::new(Settings::default());
            g.start_round(shuffled_deck(&mut rng));
            for s in 0..4 {
                let call = H2Bot::new().grand_tichu(&g.view(s));
                g.apply(&Action::GrandTichu { seat: s, call }).unwrap();
            }
            let passes: Vec<PassInfo> = (0..4).map(|s| H2Bot::new().pass_cards(&g.view(s))).collect();
            for (s, p) in passes.iter().enumerate() {
                passed[1][class(p.partner)] += 1.0;
                passed[0][class(p.left)] += 1.0;
                passed[0][class(p.right)] += 1.0;
                g.apply(&Action::PassCards { seat: s, pass: *p }).unwrap();
            }
            for s in 0..4 {
                for &c in &g.players[s].hand {
                    held[class(c)] += 1.0;
                }
            }
        }
        let mut ratio = [[0.0; 17]; 2];
        for rel in 0..2 {
            for k in 0..17 {
                ratio[rel][k] = passed[rel][k] / held[k];
            }
        }
        PassModel { ratio }
    }

    /// The shared model: `from_h2(1500, 0x9A55)`, stored because h2's calls
    /// and passes take about 0.1 s a deal. Classes are ranks 2..A, then Mah
    /// Jong, Dog, Phoenix, Dragon; first row to an opponent, second to
    /// partner.
    pub fn shared() -> &'static PassModel {
        static MODEL: PassModel = PassModel {
            ratio: [
                [0.461, 0.360, 0.257, 0.198, 0.146, 0.121, 0.100, 0.033, 0.023, 0.015, 0.014, 0.011, 0.011, 0.001, 1.000, 0.001, 0.001],
                [0.032, 0.021, 0.045, 0.031, 0.047, 0.047, 0.056, 0.065, 0.054, 0.050, 0.066, 0.094, 0.242, 0.080, 0.001, 0.240, 0.284],
            ],
        };
        &MODEL
    }

    fn weight(&self, partner: bool, c: Card) -> f64 {
        self.ratio[partner as usize][class(c)]
    }
}

/// Put all four passes into a state rebuilt from `view`: each seat's own
/// pass, what each seat received, and the `PassCards` actions in the log
/// (each seat's view shows only its own).
pub fn restore_passes(state: &mut GameState, view: &PlayerView, passes: [PassInfo; 4]) {
    for (s, p) in passes.iter().enumerate() {
        state.players[s].pass = Some(*p);
    }
    for r in 0..4 {
        let (l, p, rt) = (left_seat(r), partner_seat(r), right_seat(r));
        state.received[r] = vec![(passes[l].right, l), (passes[p].partner, p), (passes[rt].left, rt)];
    }
    let me = view.seat;
    if let Some(at) = state.log.iter().position(|a| matches!(a, Action::PassCards { seat, .. } if *seat == me)) {
        state.log.remove(at);
        for (k, s) in (0..4).enumerate() {
            state.log.insert(at + k, Action::PassCards { seat: s, pass: passes[s] });
        }
    }
}

/// Every card `seat` has held since the pass: its hand in `state` plus what
/// it has played.
fn holdings(state: &GameState, seat: Seat) -> Vec<Card> {
    let mut h = state.players[seat].hand.clone();
    for a in &state.log {
        match a {
            Action::Play { seat: s, cards, .. } | Action::Bomb { seat: s, cards } if *s == seat => h.extend_from_slice(cards),
            _ => {}
        }
    }
    h
}

/// Give the other seats of a state rebuilt from `view` their pass knowledge:
/// what is known for sure, and a guess at the rest (see the module docs).
/// Does nothing if the view has no pass yet.
pub fn guess_passes(state: &mut GameState, view: &PlayerView, rng: &mut StdRng) {
    let me = view.seat;
    let Some(mine) = view.my_pass else { return };
    if view.received.len() != 3 {
        return;
    }
    let model = PassModel::shared();
    // to[a][b]: the card a passed to b.
    let mut to: [[Option<Card>; 4]; 4] = Default::default();
    to[me][left_seat(me)] = Some(mine.left);
    to[me][partner_seat(me)] = Some(mine.partner);
    to[me][right_seat(me)] = Some(mine.right);
    for &(c, from) in &view.received {
        to[from][me] = Some(c);
    }
    for b in (0..4).filter(|&b| b != me) {
        let mut pool: Vec<Card> = holdings(state, b).into_iter().filter(|&c| Some(c) != to[me][b]).collect();
        for a in (0..4).filter(|&a| a != me && a != b) {
            let partner = a == partner_seat(b);
            let w: Vec<f64> = pool.iter().map(|&c| model.weight(partner, c)).collect();
            let total: f64 = w.iter().sum();
            if pool.is_empty() || total <= 0.0 {
                return;
            }
            let mut x = rng.gen::<f64>() * total;
            let mut pick = pool.len() - 1;
            for (i, wi) in w.iter().enumerate() {
                x -= wi;
                if x <= 0.0 {
                    pick = i;
                    break;
                }
            }
            to[a][b] = Some(pool.swap_remove(pick));
        }
    }
    let pass = |a: Seat| PassInfo {
        left: to[a][left_seat(a)].unwrap(),
        partner: to[a][partner_seat(a)].unwrap(),
        right: to[a][right_seat(a)].unwrap(),
    };
    restore_passes(state, view, [pass(0), pass(1), pass(2), pass(3)]);
}
