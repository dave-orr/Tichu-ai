//! Training data for one net that makes every decision (`tichu e2edata`).
//!
//! Plays full games with `bot` in every seat and writes one record for every
//! decision with a real choice, whatever its kind: Grand Tichu, Small Tichu,
//! each of the three pass cards, a play, an out-of-turn bomb, the wish, the
//! Dragon gift. A record is what the seat could see, the one option taken,
//! and (in its block's header) how the round ended.
//!
//! At some decisions the generator also *branches*: it copies the real state,
//! takes another option there, and plays that round out with fresh copies of
//! `bot` in every seat. Players in a branch still see only their own views;
//! only the generator knows the deal. The decision and its alternatives share
//! a group number, so a trainer can compare their results on the same deal
//! (the real-deal critic of `selfplay --critic`, `calllabel` and
//! `passlabel`, for every kind of decision). A share of the decisions inside
//! a branch are recorded too.
//!
//! Each record also carries where every card the seat can't see really is,
//! as a training target (never an input).
//!
//! File: gzip of (little endian) a 28-byte header — `b"TE2E"`, format
//! version, `FEATURE_VERSION`, `E_STATE_DIM`, `CAND_DIM`, `TOKEN_DIM`,
//! `HIDDEN_DIM` (u32 each) — then blocks, one per seat per playout:
//!
//! ```text
//! block:  u32 game | u16 round | u16 playout (0 = the game itself) | u8 seat
//!         u8 who went out first (relative seat) | i16 my team's round total
//!         i16 theirs | i16 my team's card points (200 for a double win)
//!         i16 theirs | i16 my team's game score before the round | i16 theirs
//!         u16 n tokens | u16 n records
//!         n tokens × TOKEN_DIM bytes (the seat's history of this playout)
//!         n records × RECORD_SIZE bytes
//! record: u8 kind | u8 flags (1 = the option was forced by the generator:
//!         the first decision of a branch) | u8 options there were (capped
//!         at 255) | u8 pass step | u16 group (0 = none; unique within the
//!         game's round) | u16 history tokens seen
//!         E_STATE_DIM bytes | CAND_DIM bytes | HIDDEN_DIM bytes
//!         i16 the bot's net's value of the position, in tenths of a
//!         point (flag 2; plays, bombs, wishes and Dragon gifts)
//!         i16 its policy's log-probability of this option, in
//!         thousandths (flag 4; plays)
//! ```
//!
//! Candidate features (`CAND_DIM` bytes; each kind has its own slots, the
//! rest are zero): 0..48 a play, a bomb, or passing / not bombing
//! (`features::encode_action`); 48..50 don't call / call; 50..112 a pass
//! card (see `encode_pass_card`); 112..128 a wish; 128..136 a Dragon gift.
//!
//! Hidden-card target: one byte per card (`Card::index`): 0 = nothing to
//! predict (I hold it, it was played, or it isn't dealt yet), else the
//! relative seat (1..=3) that holds it.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::{mpsc, Arc};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::bots::h2::{self, Counts};
use crate::bots::net::{Net, SeqCache};
use crate::calllabel::{encode_call_extra, CALL_EXTRA_DIM};
use crate::deck::shuffled_deck;
use crate::engine::{Action, GameState};
use crate::features::{encode_state_with, rel, Candidates, HistoryWalk, ACTION_DIM, FEATURE_VERSION, STATE_DIM, TOKEN_DIM};
use crate::scoring::RoundResult;
use crate::sim::{bomb_round, play_dealt_round, play_from, resume_from};
use crate::types::*;
use crate::view::PlayerView;

pub const FORMAT_VERSION: u32 = 2;
/// The decision's kind (one-hot, 8) and the call features.
pub const CTX_DIM: usize = 8 + CALL_EXTRA_DIM;
pub const E_STATE_DIM: usize = STATE_DIM + CTX_DIM;
pub const CAND_DIM: usize = 136;
pub const HIDDEN_DIM: usize = 56;
pub const RECORD_SIZE: usize = 8 + E_STATE_DIM + CAND_DIM + HIDDEN_DIM + 4;

const CALL_AT: usize = 48;
const PASS_AT: usize = 50;
const WISH_AT: usize = 112;
const DRAGON_AT: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Grand = 0,
    Small = 1,
    Pass = 2,
    Play = 3,
    Bomb = 4,
    Wish = 5,
    Dragon = 6,
}

pub const KINDS: usize = 7;
pub const KIND_NAMES: [&str; KINDS] = ["grand", "small", "pass", "play", "bomb", "wish", "dragon"];

/// What `view.seat` sees at a decision of `kind`: `encode_state`, the kind,
/// and for calls the call features.
pub fn encode_decision_state(view: &PlayerView, walk: &HistoryWalk, kind: Kind, out: &mut Vec<u8>) {
    let start = out.len();
    encode_state_with(view, walk, out);
    for k in 0..8 {
        out.push(if k == kind as usize { 255 } else { 0 });
    }
    if matches!(kind, Kind::Grand | Kind::Small) {
        encode_call_extra(view, out);
    }
    out.resize(start + E_STATE_DIM, 0);
}

fn frac(x: f32, max: f32) -> u8 {
    ((x / max).clamp(0.0, 1.0) * 255.0).round() as u8
}

fn class(c: Card) -> usize {
    match c {
        Card::Normal { rank, .. } => rank as usize - 2,
        Card::Special(SpecialKind::MahJong) => 13,
        Card::Special(SpecialKind::Dog) => 14,
        Card::Special(SpecialKind::Phoenix) => 15,
        Card::Special(SpecialKind::Dragon) => 16,
    }
}

/// A play, a bomb, or passing / not bombing: `encode_action`'s bytes.
pub fn cand_action(action: &[u8]) -> Vec<u8> {
    let mut c = vec![0u8; CAND_DIM];
    c[..ACTION_DIM].copy_from_slice(action);
    c
}

/// Calling (`true`) or not.
pub fn cand_call(call: bool) -> Vec<u8> {
    let mut c = vec![0u8; CAND_DIM];
    c[CALL_AT + call as usize] = 255;
    c
}

/// The recipients of the three pass steps, in order: partner, left
/// (seat + 3), right (seat + 1).
pub const PASS_STEPS: usize = 3;

pub fn pass_from_steps(cards: [Card; 3]) -> PassInfo {
    PassInfo { partner: cards[0], left: cards[1], right: cards[2] }
}

pub fn pass_steps(pass: &PassInfo) -> [Card; 3] {
    [pass.partner, pass.left, pass.right]
}

/// Passing `card` at `step`, after `earlier` (the cards of the steps before):
/// the step, the card's class, how it sits in my hand (copies of its rank,
/// part of a four of a kind or of a same-suit run of five), how the hand I
/// would keep plays out (h2's plan against every other card), and the
/// classes of the cards already chosen.
pub fn encode_pass_card(view: &PlayerView, step: usize, card: Card, earlier: &[Card]) -> Vec<u8> {
    let mut c = vec![0u8; CAND_DIM];
    let at = PASS_AT;
    c[at + step] = 255;
    c[at + 3 + class(card)] = 255;
    let hand = CardSet::from_cards(&view.hand);
    let copies = card.rank().map_or(1, |r| view.hand.iter().filter(|x| x.rank() == Some(r)).count());
    c[at + 20] = frac(copies as f32, 4.0);
    if let Card::Normal { suit, rank } = card {
        let (mut lo, mut hi) = (rank, rank);
        while lo > 2 && hand.contains(Card::normal(suit, lo - 1)) {
            lo -= 1;
        }
        while hi < 14 && hand.contains(Card::normal(suit, hi + 1)) {
            hi += 1;
        }
        c[at + 21] = if hi - lo + 1 >= 5 { 255 } else { 0 };
    }
    c[at + 22] = if copies == 4 { 255 } else { 0 };
    let mut kept = hand;
    kept.remove(card);
    for e in earlier {
        kept.remove(*e);
    }
    let mine = Counts::of(kept.cards());
    let rest = Counts::of(CardSet::FULL.minus(kept).cards());
    let cost = h2::cost(&mine);
    let plan = h2::plan(&mine, &rest, cost);
    c[at + 23] = frac(cost as f32, 14.0);
    c[at + 24] = frac(plan.lead_bosses as f32, 8.0);
    c[at + 25] = frac(plan.regain as f32, 8.0);
    c[at + 26] = frac(plan.losers as f32, 14.0);
    c[at + 27] = if h2::clear_path(&mine, &rest) { 255 } else { 0 };
    for (k, e) in earlier.iter().enumerate().take(2) {
        c[at + 28 + 17 * k + class(*e)] = 255;
    }
    c
}

/// Wish options: 0 = no wish, 1..=13 = ranks 2..=14.
pub const WISH_OPTIONS: usize = 14;

pub fn wish_of(option: usize) -> Option<Rank> {
    (option > 0).then_some(option as Rank + 1)
}

pub fn wish_option(rank: Option<Rank>) -> usize {
    rank.map_or(0, |r| r as usize - 1)
}

/// A wish: the option, how many of the rank I hold, and whether I passed a
/// card of that rank that is still in its recipient's hand.
pub fn encode_wish(view: &PlayerView, option: usize) -> Vec<u8> {
    let mut c = vec![0u8; CAND_DIM];
    c[WISH_AT + option] = 255;
    if let Some(r) = wish_of(option) {
        c[WISH_AT + 14] = frac(view.hand.iter().filter(|x| x.rank() == Some(r)).count() as f32, 4.0);
        c[WISH_AT + 15] = if view.known_holdings().iter().any(|(x, _)| x.rank() == Some(r)) { 255 } else { 0 };
    }
    c
}

/// Giving the Dragon trick to `to`: which opponent (the next player or the
/// one before me), their cards left, whether they called, whether they are
/// out, the points they hold.
pub fn encode_dragon(view: &PlayerView, to: Seat) -> Vec<u8> {
    let mut c = vec![0u8; CAND_DIM];
    let p = &view.players[to];
    c[DRAGON_AT + if rel(view, to) == 1 { 0 } else { 1 }] = 255;
    c[DRAGON_AT + 2] = frac(p.card_count as f32, 14.0);
    c[DRAGON_AT + 3] = if p.tichu != TichuCall::None { 255 } else { 0 };
    c[DRAGON_AT + 4] = if p.is_out() { 255 } else { 0 };
    c[DRAGON_AT + 5] = frac(p.captured_points as f32 + 25.0, 150.0);
    c
}

/// Where the cards `view.seat` can't see are in `real` (see the module docs).
pub fn hidden_targets(view: &PlayerView, real: &GameState) -> Vec<u8> {
    let mut h = vec![0u8; HIDDEN_DIM];
    if real.phase == Phase::GrandTichuWindow {
        return h;
    }
    for s in 0..4 {
        if s != view.seat {
            let r = rel(view, s) as u8;
            for c in &real.players[s].hand {
                h[c.index()] = r;
            }
        }
    }
    h
}

#[derive(Debug, Clone)]
pub struct E2eConfig {
    /// Bot in all four seats (by `bots::by_name`).
    pub bot: String,
    pub games: usize,
    pub seed: u64,
    pub threads: usize,
    /// Chance of branching at a decision, by kind (`Kind as usize`).
    pub branch: [f64; KINDS],
    /// Alternatives tried at a branching play or pass decision.
    pub play_alts: usize,
    pub pass_alts: usize,
    /// Share of the decisions inside a branch that are recorded.
    pub inner_keep: f64,
}

impl Default for E2eConfig {
    fn default() -> Self {
        E2eConfig {
            bot: "h2".into(),
            games: 100,
            seed: 0,
            threads: 1,
            branch: [0.25, 0.12, 0.25, 0.04, 0.2, 0.5, 0.5],
            play_alts: 2,
            pass_alts: 1,
            inner_keep: 0.25,
        }
    }
}

/// Parse `--branch` (`grand=0.25,play=0.04,...`; `all=0` sets every kind).
pub fn parse_branch(spec: &str, into: &mut [f64; KINDS]) -> Option<()> {
    for part in spec.split(',').filter(|p| !p.is_empty()) {
        let (name, p) = part.split_once('=')?;
        let p: f64 = p.parse().ok().filter(|p| (0.0..=1.0).contains(p))?;
        if name == "all" {
            *into = [p; KINDS];
        } else {
            into[KIND_NAMES.iter().position(|n| *n == name)?] = p;
        }
    }
    Some(())
}

struct Rec {
    kind: Kind,
    flags: u8,
    n_options: usize,
    step: u8,
    group: u16,
    prefix: usize,
    state: Vec<u8>,
    cand: Vec<u8>,
    hidden: Vec<u8>,
    teacher_v: i16,
    teacher_lp: i16,
}

/// What a round's playouts share.
#[derive(Default)]
struct Shared {
    game: u32,
    round: u16,
    next_group: u16,
    next_playout: u16,
    buf: Vec<u8>,
    records: usize,
    playouts: usize,
}

struct Common {
    cfg: E2eConfig,
    /// Ranks a play decision's alternatives (the net behind `cfg.bot`, if it
    /// names one).
    anchor: Option<Arc<Net>>,
}

/// How a branch goes on after its first action.
enum After {
    /// From before play began: the usual round, with its Small Tichu offers.
    Full,
    /// After a play or bomb: the others' bomb offers, then on.
    Bombs,
    Resume,
}

struct Recorder {
    inner: Box<dyn Bot>,
    common: Arc<Common>,
    shared: Rc<RefCell<Shared>>,
    rng: StdRng,
    walk: HistoryWalk,
    real: Option<GameState>,
    recs: Vec<Rec>,
    /// 0 = the game itself.
    playout: u16,
    /// In a branch whose first move was "not now": turn down Small Tichu
    /// and bomb offers while the seat's log still has this length.
    hold: Option<usize>,
    scores_before: [i32; 2],
    seed: u64,
    seq: SeqCache,
}

impl Recorder {
    fn main(&self) -> bool {
        self.playout == 0
    }

    fn rec(&mut self, view: &PlayerView, kind: Kind, n_options: usize, step: usize, cand: Vec<u8>) -> Rec {
        self.walk.update(view);
        let mut state = Vec::with_capacity(E_STATE_DIM);
        encode_decision_state(view, &self.walk, kind, &mut state);
        let hidden = match &self.real {
            Some(g) => hidden_targets(view, g),
            None => vec![0; HIDDEN_DIM],
        };
        Rec { kind, flags: 0, n_options, step: step as u8, group: 0, prefix: self.walk.n_tokens(), state, cand, hidden, teacher_v: 0, teacher_lp: 0 }
    }

    /// Keep a decision's record: always in the game itself, a share of them
    /// inside a branch.
    fn keep(&mut self, mut rec: Rec, option: Option<(&Candidates, usize)>) {
        if self.main() || rec.group != 0 || self.rng.gen_bool(self.common.cfg.inner_keep) {
            self.teach(&mut rec, option);
            self.recs.push(rec);
        }
    }

    /// What the bot's own net says here, for a learner to start from: its
    /// value of the position and, for a play, its policy's log-probability
    /// of option `k`. Call it while `self.walk` is at the record's view.
    fn teach(&mut self, rec: &mut Rec, option: Option<(&Candidates, usize)>) {
        let Some(net) = self.common.anchor.clone() else { return };
        if !matches!(rec.kind, Kind::Play | Kind::Bomb | Kind::Wish | Kind::Dragon) || rec.flags & 6 != 0 {
            return;
        }
        let state = &rec.state[..STATE_DIM];
        let h = net.trunk(state, &net.encode_seq_cached(self.walk.tokens(), &mut self.seq));
        rec.teacher_v = (net.value(state, &h) * 10.0).round().clamp(-32000.0, 32000.0) as i16;
        rec.flags |= 2;
        if let (Kind::Play, Some((cands, k))) = (rec.kind, option) {
            let logits = net.logits(&h, &cands.features);
            let top = logits.iter().copied().fold(f32::MIN, f32::max);
            let lse = top + logits.iter().map(|l| (l - top).exp()).sum::<f32>().ln();
            rec.teacher_lp = ((logits[k] - lse) * 1000.0).round().clamp(-32000.0, 0.0) as i16;
            rec.flags |= 4;
        }
    }

    fn branching(&mut self, kind: Kind) -> bool {
        self.main() && self.real.is_some() && self.rng.gen_bool(self.common.cfg.branch[kind as usize])
    }

    fn new_group(&mut self) -> u16 {
        let mut sh = self.shared.borrow_mut();
        sh.next_group += 1;
        sh.next_group
    }

    /// Play the round out from the real state with `action` taken instead
    /// (`None`: with the offer turned down for now), recording `root` as
    /// this seat's first decision of the playout.
    fn branch(&mut self, view: &PlayerView, mut root: Rec, action: Option<Action>, after: After) {
        self.teach(&mut root, None);
        let Some(mut g) = self.real.clone() else { return };
        let seat = view.seat;
        let hold = action.is_none().then_some(view.log.len());
        if let Some(a) = &action {
            if g.apply(a).is_err() {
                return;
            }
        }
        let playout = {
            let mut sh = self.shared.borrow_mut();
            sh.next_playout += 1;
            sh.next_playout
        };
        root.flags |= 1;
        let mut root = Some(root);
        let base = self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(playout as u64 * 8);
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| -> Box<dyn Bot> {
            Box::new(Recorder {
                inner: crate::bots::by_name(&self.common.cfg.bot, base + s as u64).expect("unknown bot"),
                common: self.common.clone(),
                shared: self.shared.clone(),
                rng: StdRng::seed_from_u64(base + 4 + s as u64),
                walk: HistoryWalk::default(),
                real: None,
                recs: if s == seat { root.take().into_iter().collect() } else { vec![] },
                playout,
                hold: if s == seat { hold } else { None },
                scores_before: self.scores_before,
                seed: base,
                seq: SeqCache::default(),
            })
        });
        let done = match after {
            After::Full => play_from(&mut g, &mut bots, None, true),
            After::Bombs => bomb_round(&mut g, &mut bots, seat, &mut None).and_then(|_| resume_from(&mut g, &mut bots, None)),
            After::Resume => resume_from(&mut g, &mut bots, None),
        };
        if done.is_ok() {
            self.shared.borrow_mut().playouts += 1;
        }
    }

    fn held(&self, view: &PlayerView) -> bool {
        self.hold == Some(view.log.len())
    }
}

impl Bot for Recorder {
    fn name(&self) -> String {
        format!("e2e-record:{}", self.inner.name())
    }

    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        self.scores_before = view.scores;
        let call = self.inner.grand_tichu(view);
        let mut rec = self.rec(view, Kind::Grand, 2, 0, cand_call(call));
        if self.branching(Kind::Grand) {
            rec.group = self.new_group();
            let mut alt = self.rec(view, Kind::Grand, 2, 0, cand_call(!call));
            alt.group = rec.group;
            self.branch(view, alt, Some(Action::GrandTichu { seat: view.seat, call: !call }), After::Full);
        }
        self.keep(rec, None);
        call
    }

    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        if self.held(view) {
            return false;
        }
        let call = self.inner.small_tichu(view);
        let mut rec = self.rec(view, Kind::Small, 2, 0, cand_call(call));
        if self.branching(Kind::Small) {
            rec.group = self.new_group();
            let mut alt = self.rec(view, Kind::Small, 2, 0, cand_call(!call));
            alt.group = rec.group;
            let action = (!call).then_some(Action::SmallTichu { seat: view.seat });
            self.branch(view, alt, action, After::Resume);
        }
        self.keep(rec, None);
        call
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        let pass = self.inner.pass_cards(view);
        let steps = pass_steps(&pass);
        let mut recs: Vec<Rec> =
            (0..PASS_STEPS).map(|k| self.rec(view, Kind::Pass, view.hand.len() - k, k, encode_pass_card(view, k, steps[k], &steps[..k]))).collect();
        if self.branching(Kind::Pass) {
            for _ in 0..self.common.cfg.pass_alts.max(1) {
                // Another card at one step. Opponents mostly get low cards,
                // so weight their alternatives that way; if the card was
                // going to a later recipient, the two swap.
                let k = self.rng.gen_range(0..PASS_STEPS);
                let pool: Vec<Card> = view.hand.iter().copied().filter(|c| *c != steps[k] && !steps[..k].contains(c)).collect();
                let weight = |c: &Card| if k == 0 { 1.0 } else { (-(class(*c).min(13) as f64) / 4.0).exp() };
                let total: f64 = pool.iter().map(weight).sum();
                let mut x = self.rng.gen_range(0.0..total);
                let mut card = pool[pool.len() - 1];
                for c in &pool {
                    x -= weight(c);
                    if x <= 0.0 {
                        card = *c;
                        break;
                    }
                }
                let mut alt_steps = steps;
                if let Some(j) = (k + 1..PASS_STEPS).find(|&j| steps[j] == card) {
                    alt_steps[j] = steps[k];
                }
                alt_steps[k] = card;
                if recs[k].group == 0 {
                    recs[k].group = self.new_group();
                }
                let mut alt = self.rec(view, Kind::Pass, view.hand.len() - k, k, encode_pass_card(view, k, card, &steps[..k]));
                alt.group = recs[k].group;
                let action = Action::PassCards { seat: view.seat, pass: pass_from_steps(alt_steps) };
                self.branch(view, alt, Some(action), After::Full);
            }
        }
        for rec in recs {
            self.keep(rec, None);
        }
        pass
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let cands = Candidates::new(view, plays, can_pass);
        let choice = self.inner.play(view, plays, can_pass);
        if cands.len() < 2 {
            return choice;
        }
        let Some(opt) = cands.option_of(choice) else { return choice };
        let feat = |k: usize| cand_action(&cands.features[k * ACTION_DIM..(k + 1) * ACTION_DIM]);
        let mut rec = self.rec(view, Kind::Play, cands.len(), 0, feat(opt));
        if self.branching(Kind::Play) {
            let mut order: Vec<usize> = (0..cands.len()).filter(|&k| k != opt).collect();
            match &self.common.anchor {
                Some(net) => {
                    let logits = net.option_logits(view, &cands);
                    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
                }
                None => {
                    for i in (1..order.len()).rev() {
                        order.swap(i, self.rng.gen_range(0..=i));
                    }
                }
            }
            order.truncate(self.common.cfg.play_alts.max(1));
            rec.group = self.new_group();
            for k in order {
                let mut alt = self.rec(view, Kind::Play, cands.len(), 0, feat(k));
                alt.group = rec.group;
                self.teach(&mut alt, Some((&cands, k)));
                match cands.options[k] {
                    Some(i) => {
                        let action = Action::Play { seat: view.seat, cards: plays[i].cards.clone(), phoenix_as: plays[i].phoenix_as };
                        self.branch(view, alt, Some(action), After::Bombs)
                    }
                    None => self.branch(view, alt, Some(Action::PassTurn { seat: view.seat }), After::Resume),
                }
            }
        }
        self.keep(rec, Some((&cands, opt)));
        choice
    }

    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        if self.held(view) {
            return None;
        }
        let choice = self.inner.bomb(view, bombs);
        let cands = Candidates::new(view, bombs, true);
        let Some(opt) = cands.option_of(choice) else { return choice };
        let feat = |k: usize| cand_action(&cands.features[k * ACTION_DIM..(k + 1) * ACTION_DIM]);
        let mut rec = self.rec(view, Kind::Bomb, cands.len(), 0, feat(opt));
        if self.branching(Kind::Bomb) {
            // The other way: wait if the bot bombed, else throw its lowest bomb.
            let k = if choice.is_some() { cands.len() - 1 } else { 0 };
            rec.group = self.new_group();
            let mut alt = self.rec(view, Kind::Bomb, cands.len(), 0, feat(k));
            alt.group = rec.group;
            match cands.options[k] {
                Some(i) => self.branch(view, alt, Some(Action::Bomb { seat: view.seat, cards: bombs[i].cards.clone() }), After::Bombs),
                None => self.branch(view, alt, None, After::Resume),
            }
        }
        self.keep(rec, None);
        choice
    }

    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        let rank = self.inner.wish(view);
        let opt = wish_option(rank);
        let mut rec = self.rec(view, Kind::Wish, WISH_OPTIONS, 0, encode_wish(view, opt));
        if self.branching(Kind::Wish) {
            let k = (opt + self.rng.gen_range(1..WISH_OPTIONS)) % WISH_OPTIONS;
            rec.group = self.new_group();
            let mut alt = self.rec(view, Kind::Wish, WISH_OPTIONS, 0, encode_wish(view, k));
            alt.group = rec.group;
            self.branch(view, alt, Some(Action::Wish { seat: view.seat, rank: wish_of(k) }), After::Resume);
        }
        self.keep(rec, None);
        rank
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        let to = self.inner.dragon_recipient(view);
        let [a, b] = view.opponents();
        let other = if to == a { b } else { a };
        let mut rec = self.rec(view, Kind::Dragon, 2, 0, encode_dragon(view, to));
        if self.branching(Kind::Dragon) {
            rec.group = self.new_group();
            let mut alt = self.rec(view, Kind::Dragon, 2, 0, encode_dragon(view, other));
            alt.group = rec.group;
            self.branch(view, alt, Some(Action::GiveDragon { seat: view.seat, to: other }), After::Resume);
        }
        self.keep(rec, None);
        to
    }

    fn cheats(&self) -> bool {
        // Not to play: the real state gives the hidden-card targets and the
        // branches. Every decision comes from `inner`, which sees its view.
        true
    }

    fn peek(&mut self, state: &GameState) {
        self.real = Some(state.clone());
    }

    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        self.inner.round_over(view, result);
        self.real = None;
        self.walk.update(view);
        let recs = std::mem::take(&mut self.recs);
        if recs.is_empty() {
            return;
        }
        let team = view.my_team();
        let mut sh = self.shared.borrow_mut();
        let (game, round) = (sh.game, sh.round);
        let b = &mut sh.buf;
        b.extend_from_slice(&game.to_le_bytes());
        b.extend_from_slice(&round.to_le_bytes());
        b.extend_from_slice(&self.playout.to_le_bytes());
        b.push(view.seat as u8);
        b.push(rel(view, result.out_order[0]) as u8);
        for x in [
            result.total(team),
            result.total(1 - team),
            result.card_points[team],
            result.card_points[1 - team],
            self.scores_before[team],
            self.scores_before[1 - team],
        ] {
            b.extend_from_slice(&(x as i16).to_le_bytes());
        }
        b.extend_from_slice(&(self.walk.n_tokens() as u16).to_le_bytes());
        b.extend_from_slice(&(recs.len() as u16).to_le_bytes());
        b.extend_from_slice(self.walk.tokens());
        for r in &recs {
            assert!(r.prefix <= self.walk.n_tokens(), "history isn't a prefix of the playout's");
            b.push(r.kind as u8);
            b.push(r.flags);
            b.push(r.n_options.min(255) as u8);
            b.push(r.step);
            b.extend_from_slice(&r.group.to_le_bytes());
            b.extend_from_slice(&(r.prefix as u16).to_le_bytes());
            b.extend_from_slice(&r.state);
            b.extend_from_slice(&r.cand);
            b.extend_from_slice(&r.hidden);
            b.extend_from_slice(&r.teacher_v.to_le_bytes());
            b.extend_from_slice(&r.teacher_lp.to_le_bytes());
        }
        sh.records += recs.len();
    }
}

pub fn write_header(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"TE2E")?;
    for x in [FORMAT_VERSION, FEATURE_VERSION, E_STATE_DIM as u32, CAND_DIM as u32, TOKEN_DIM as u32, HIDDEN_DIM as u32] {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

/// Totals of a run.
#[derive(Debug, Default, Clone, Copy)]
pub struct E2eStats {
    pub games: usize,
    pub rounds: usize,
    pub playouts: usize,
    pub records: usize,
}

/// Play `cfg.games` games and write their records (gzipped) to `w`.
pub fn run(cfg: &E2eConfig, w: impl Write) -> std::io::Result<E2eStats> {
    let mut w = flate2::write::GzEncoder::new(std::io::BufWriter::new(w), flate2::Compression::new(3));
    write_header(&mut w)?;
    let common = Arc::new(Common { cfg: cfg.clone(), anchor: anchor_of(&cfg.bot) });
    let threads = cfg.threads.max(1).min(cfg.games.max(1));
    // (records, playouts, bytes) per round; an empty message ends a game.
    let (tx, rx) = mpsc::sync_channel::<(usize, usize, Vec<u8>, bool)>(threads * 4);
    let mut stats = E2eStats::default();
    let start = std::time::Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        for t in 0..threads {
            let tx = tx.clone();
            let common = common.clone();
            scope.spawn(move || {
                for i in (t..cfg.games).step_by(threads) {
                    let seed = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64);
                    let shared = Rc::new(RefCell::new(Shared { game: i as u32, ..Default::default() }));
                    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                        let sd = seed.wrapping_mul(31).wrapping_add(s);
                        Box::new(Recorder {
                            inner: crate::bots::by_name(&cfg.bot, sd).expect("unknown bot"),
                            common: common.clone(),
                            shared: shared.clone(),
                            rng: StdRng::seed_from_u64(sd ^ 0xE2E),
                            walk: HistoryWalk::default(),
                            real: None,
                            recs: vec![],
                            playout: 0,
                            hold: None,
                            scores_before: [0, 0],
                            seed: sd,
                            seq: SeqCache::default(),
                        })
                    });
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut state = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                    while state.phase != Phase::GameEnd && state.round_number < 200 {
                        {
                            let mut sh = shared.borrow_mut();
                            sh.round = state.round_number as u16;
                            sh.next_group = 0;
                            sh.next_playout = 0;
                        }
                        state.start_round(shuffled_deck(&mut rng));
                        let ok = play_dealt_round(&mut state, &mut bots, None).is_ok();
                        let mut sh = shared.borrow_mut();
                        let msg = (sh.records, sh.playouts + 1, std::mem::take(&mut sh.buf), false);
                        sh.records = 0;
                        sh.playouts = 0;
                        drop(sh);
                        if !ok || tx.send(msg).is_err() {
                            break;
                        }
                    }
                    if tx.send((0, 0, vec![], true)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        for (records, playouts, buf, game_over) in rx {
            if game_over {
                stats.games += 1;
                let secs = start.elapsed().as_secs_f64();
                eprint!(
                    "\r{}/{} games, {} rounds, {} playouts, {} records ({:.1} rounds/s)   ",
                    stats.games,
                    cfg.games,
                    stats.rounds,
                    stats.playouts,
                    stats.records,
                    stats.rounds as f64 / secs
                );
                continue;
            }
            w.write_all(&buf)?;
            stats.rounds += 1;
            stats.playouts += playouts;
            stats.records += records;
        }
        eprintln!();
        Ok(())
    })?;
    w.finish()?.flush()?;
    Ok(stats)
}

/// The play net inside a bot name (`...:net:PATH` or `net:PATH`), if any.
fn anchor_of(name: &str) -> Option<Arc<Net>> {
    let at = name.rfind("net:")?;
    if at > 0 && !name[..at].ends_with(':') {
        return None;
    }
    Net::cached(&name[at + 4..]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn generate(cfg: &E2eConfig) -> (Vec<u8>, E2eStats) {
        let mut gz = vec![];
        let stats = run(cfg, &mut gz).unwrap();
        let mut raw = vec![];
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut raw).unwrap();
        (raw, stats)
    }

    fn u16_at(b: &[u8], at: usize) -> usize {
        u16::from_le_bytes([b[at], b[at + 1]]) as usize
    }

    /// (kind, flags, group, playout, state, cand, hidden) of every record.
    #[allow(clippy::type_complexity)]
    fn records(raw: &[u8]) -> Vec<(u8, u8, usize, usize, (u32, usize), Vec<u8>, Vec<u8>)> {
        assert_eq!(&raw[..4], b"TE2E");
        let (mut pos, mut out) = (28, vec![]);
        while pos < raw.len() {
            let game = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
            let round = u16_at(raw, pos + 4);
            let playout = u16_at(raw, pos + 6);
            assert!(raw[pos + 8] < 4 && raw[pos + 9] < 4);
            let (nt, nr) = (u16_at(raw, pos + 22), u16_at(raw, pos + 24));
            pos += 26 + nt * TOKEN_DIM;
            for _ in 0..nr {
                let r = &raw[pos..pos + RECORD_SIZE];
                assert!(u16_at(r, 6) <= nt);
                out.push((r[0], r[1], u16_at(r, 4), playout, (game, round), r[8 + E_STATE_DIM..8 + E_STATE_DIM + CAND_DIM].to_vec(), r[RECORD_SIZE - 4 - HIDDEN_DIM..RECORD_SIZE - 4].to_vec()));
                pos += RECORD_SIZE;
            }
        }
        assert_eq!(pos, raw.len());
        out
    }

    #[test]
    fn records_every_kind_and_branches() {
        let cfg = E2eConfig { games: 3, seed: 5, branch: [0.5; KINDS], inner_keep: 0.5, ..Default::default() };
        let (raw, stats) = generate(&cfg);
        let recs = records(&raw);
        assert_eq!(recs.len(), stats.records);
        assert!(stats.playouts > stats.rounds);
        for kind in [Kind::Grand, Kind::Small, Kind::Pass, Kind::Play, Kind::Wish, Kind::Dragon] {
            assert!(recs.iter().any(|r| r.0 == kind as u8), "no {kind:?} records");
        }
        // Every forced record starts a branch and belongs to a group whose
        // other members include the game's own decision.
        let forced: Vec<_> = recs.iter().filter(|r| r.1 & 1 != 0).collect();
        assert!(!forced.is_empty());
        for f in &forced {
            assert!(f.3 > 0 && f.2 > 0);
            let own = recs.iter().filter(|r| r.4 == f.4 && r.2 == f.2 && r.3 == 0).count();
            assert_eq!(own, 1, "a group has one decision from the game itself");
        }
        // Each candidate uses only its kind's slots.
        for r in &recs {
            let used: Vec<usize> = (0..CAND_DIM).filter(|&i| r.5[i] != 0).collect();
            let (lo, hi) = match r.0 {
                0 | 1 => (CALL_AT, PASS_AT),
                2 => (PASS_AT, WISH_AT),
                3 | 4 => (0, CALL_AT),
                5 => (WISH_AT, DRAGON_AT),
                _ => (DRAGON_AT, CAND_DIM),
            };
            assert!(used.iter().all(|&i| (lo..hi).contains(&i)), "kind {} uses {used:?}", r.0);
            // Hidden cards: nothing during Grand Tichu, else the three other seats.
            assert!(r.6.iter().all(|&h| h <= 3));
            assert_eq!(r.0 == 0, r.6.iter().all(|&h| h == 0) && r.0 == 0);
        }
    }

    #[test]
    fn deterministic_and_branch_free_matches() {
        let cfg = E2eConfig { games: 2, seed: 9, ..Default::default() };
        assert_eq!(generate(&cfg).0, generate(&cfg).0);
        // Branching doesn't change the games themselves.
        let plain = E2eConfig { branch: [0.0; KINDS], ..cfg.clone() };
        let a: Vec<_> = records(&generate(&cfg).0).into_iter().filter(|r| r.3 == 0).map(|r| (r.0, r.4, r.5)).collect();
        let b: Vec<_> = records(&generate(&plain).0).into_iter().map(|r| (r.0, r.4, r.5)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn branch_spec_parses() {
        let mut b = [0.5; KINDS];
        assert!(parse_branch("all=0,play=0.1,pass=1", &mut b).is_some());
        assert_eq!(b, [0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0]);
        assert!(parse_branch("plays=0.1", &mut b).is_none());
        assert!(parse_branch("play=2", &mut b).is_none());
    }
}
