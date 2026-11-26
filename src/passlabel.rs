//! Training data for a learned passer (`tichu passlabel`), and the features
//! a pass model reads.
//!
//! Deals a round, lets `bot` make the Grand Tichu calls, and then, for every
//! seat, tries each candidate pass on that real deal: the other three seats
//! pass what `bot` passes, and the round is played to the end with `bot` in
//! every seat (calls included, so a pass is credited with the Tichu it makes
//! possible or helps to beat). Every candidate is played from the same state
//! with the same bots, so only the pass differs. The players see only their
//! own hands; the labeler knows the deal, which is what makes the comparison
//! fair for it. One playout per candidate: the noise of a single deal is
//! averaged by the model over many deals, not here.
//!
//! Candidates (`partner_candidates`): `bot`'s own two opponent cards, with
//! each other card of the hand as the partner card.
//!
//! File layout (little endian): `b"TPAS"`, format version, `PASS_DIM` (u32
//! each), then one record per seat per round:
//!
//! ```text
//! u32 round id | u8 seat | u8 n candidates | u8 index of bot's own pass
//! n × ( PASS_DIM feature bytes | i16 my team's total | i16 theirs
//!       | u8 flags: 1 = my team called Tichu or Grand, 2 = the opponents did,
//!                   4 = my team went out first )
//! ```

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::mpsc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::bots::h2::{self, Counts};
use crate::deck::shuffled_deck;
use crate::engine::{Action, GameState};
use crate::scoring::RoundResult;
use crate::sim::{play_dealt_round, play_from};
use crate::types::*;
use crate::view::PlayerView;

pub const FORMAT_VERSION: u32 = 1;
pub const PASS_DIM: usize = 368;

fn class(c: Card) -> usize {
    match c {
        Card::Normal { rank, .. } => rank as usize - 2,
        Card::Special(SpecialKind::MahJong) => 13,
        Card::Special(SpecialKind::Dog) => 14,
        Card::Special(SpecialKind::Phoenix) => 15,
        Card::Special(SpecialKind::Dragon) => 16,
    }
}

/// What a pass model sees of one candidate pass: `PASS_DIM` bytes (the
/// model reads `byte / 255`). `view` is the passer's view in the passing
/// phase. Seats are by offset (me, seat+1 = right, partner, seat+3 = left),
/// as the pass itself is; one bit says which way play runs.
pub fn encode_pass(view: &PlayerView, pass: &PassInfo, out: &mut Vec<u8>) {
    let start = out.len();
    let bit = |out: &mut Vec<u8>, b: bool| out.push(if b { 255 } else { 0 });
    let frac = |out: &mut Vec<u8>, x: f32, max: f32| out.push(((x / max).clamp(0.0, 1.0) * 255.0).round() as u8);
    let hand = CardSet::from_cards(&view.hand);
    let kept = hand.minus(CardSet::from_cards(&pass.cards()));
    let set = |out: &mut Vec<u8>, s: CardSet| (0..56).for_each(|i| bit(out, s.0 & (1u64 << i) != 0));
    set(out, hand);
    set(out, kept);
    for c in [pass.partner, pass.left, pass.right] {
        set(out, CardSet::from_cards(&[c]));
    }
    for c in [pass.partner, pass.left, pass.right] {
        (0..17).for_each(|k| bit(out, class(c) == k));
    }
    // The hand I keep: counts by rank, and how it plays out (h2's plan
    // against every card I don't hold).
    let mut n = [0u8; 15];
    for c in kept.cards() {
        if let Some(r) = c.rank() {
            n[r as usize] += 1;
        }
    }
    for r in 2..=14 {
        frac(out, n[r] as f32, 4.0);
    }
    let mine = Counts::of(kept.cards());
    let rest = Counts::of(CardSet::FULL.minus(kept).cards());
    let cost = h2::cost(&mine);
    let plan = h2::plan(&mine, &rest, cost);
    frac(out, cost as f32, 14.0);
    frac(out, plan.lead_bosses as f32, 8.0);
    frac(out, plan.regain as f32, 8.0);
    frac(out, plan.losers as f32, 14.0);
    bit(out, h2::clear_path(&mine, &rest));
    // Grand Tichu calls so far, the direction of play, the game score.
    for k in 0..4 {
        bit(out, view.players[(view.seat + k) % 4].tichu == TichuCall::Grand);
    }
    bit(out, view.settings.clockwise);
    let team = view.my_team();
    frac(out, view.scores[team] as f32 + 500.0, 2000.0);
    frac(out, view.scores[1 - team] as f32 + 500.0, 2000.0);
    let len = out.len() - start;
    assert!(len <= PASS_DIM, "pass features overflow: {len}");
    out.resize(start + PASS_DIM, 0);
}

/// The passes a model chooses among: `rule`'s two opponent cards, with each
/// other card of the hand as the partner card. `rule` itself is one of them.
pub fn partner_candidates(view: &PlayerView, rule: &PassInfo) -> Vec<PassInfo> {
    view.hand.iter().filter(|&&c| c != rule.left && c != rule.right).map(|&c| PassInfo { partner: c, ..*rule }).collect()
}

#[derive(Debug, Clone)]
pub struct PassLabelConfig {
    /// Bot in all four seats (by `bots::by_name`): calls, the other seats'
    /// passes, and all card play.
    pub bot: String,
    pub rounds: usize,
    pub seed: u64,
    pub threads: usize,
}

struct Cand {
    features: Vec<u8>,
    mine: i16,
    theirs: i16,
    flags: u8,
}

struct Rec {
    seat: u8,
    rule: u8,
    cands: Vec<Cand>,
}

struct Labeler {
    inner: Box<dyn Bot>,
    bot: String,
    seed: u64,
    real: Option<GameState>,
    /// The four seats' own passes for this deal (computed once).
    passes: Rc<RefCell<Option<[PassInfo; 4]>>>,
    out: Rc<RefCell<Vec<Rec>>>,
}

impl Labeler {
    fn fresh(&self) -> [Box<dyn Bot>; 4] {
        [0, 1, 2, 3].map(|k| crate::bots::by_name(&self.bot, self.seed.wrapping_add(k)).expect("unknown bot"))
    }
}

impl Bot for Labeler {
    fn name(&self) -> String {
        format!("passlabel:{}", self.inner.name())
    }
    fn cheats(&self) -> bool {
        // Not to play: the labels' playouts start from the real deal.
        true
    }
    fn peek(&mut self, state: &GameState) {
        self.real = (state.phase == Phase::Passing).then(|| state.clone());
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.grand_tichu(v)
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.small_tichu(v)
    }
    fn bomb(&mut self, v: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.inner.bomb(v, bombs)
    }
    fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
        self.inner.wish(v)
    }
    fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
        self.inner.dragon_recipient(v)
    }
    fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        self.inner.play(v, plays, can_pass)
    }
    fn round_over(&mut self, v: &PlayerView, r: &RoundResult) {
        self.inner.round_over(v, r)
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        let Some(real) = self.real.take() else { return self.inner.pass_cards(view) };
        let passes = *self.passes.borrow_mut().get_or_insert_with(|| {
            let mut bots = self.fresh();
            [0, 1, 2, 3].map(|s| bots[s].pass_cards(&real.view(s)))
        });
        let me = view.seat;
        let team = view.my_team();
        let cands = partner_candidates(view, &passes[me]);
        let mut rec = Rec { seat: me as u8, rule: 0, cands: vec![] };
        for (i, cand) in cands.iter().enumerate() {
            if *cand == passes[me] {
                rec.rule = i as u8;
            }
            let mut g = real.clone();
            let ok = (0..4).all(|s| g.apply(&Action::PassCards { seat: s, pass: if s == me { *cand } else { passes[s] } }).is_ok());
            let Some(r) = ok.then(|| play_from(&mut g, &mut self.fresh(), None, true).ok()).flatten() else { return passes[me] };
            let called = |t: usize| (0..4).any(|s| s % 2 == t && r.tichu_calls[s] != TichuCall::None);
            let mut features = Vec::with_capacity(PASS_DIM);
            encode_pass(view, cand, &mut features);
            rec.cands.push(Cand {
                features,
                mine: r.total(team) as i16,
                theirs: r.total(1 - team) as i16,
                flags: called(team) as u8 | (called(1 - team) as u8) << 1 | ((r.out_order[0] % 2 == team) as u8) << 2,
            });
        }
        self.out.borrow_mut().push(rec);
        passes[me]
    }
}

/// Label `cfg.rounds` rounds and write the records to `w`. Returns the
/// number of pass decisions written.
pub fn run(cfg: &PassLabelConfig, w: &mut impl Write) -> std::io::Result<usize> {
    w.write_all(b"TPAS")?;
    for x in [FORMAT_VERSION, PASS_DIM as u32] {
        w.write_all(&x.to_le_bytes())?;
    }
    let threads = cfg.threads.max(1).min(cfg.rounds.max(1));
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(threads * 4);
    let mut written = 0;
    let start = std::time::Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        for t in 0..threads {
            let tx = tx.clone();
            scope.spawn(move || {
                for i in (t..cfg.rounds).step_by(threads) {
                    let seed = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64);
                    let out = Rc::new(RefCell::new(vec![]));
                    let passes = Rc::new(RefCell::new(None));
                    let mut bots: [Box<dyn Bot>; 4] = [0u64, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                        Box::new(Labeler {
                            inner: crate::bots::by_name(&cfg.bot, seed.wrapping_mul(31).wrapping_add(s)).expect("unknown bot"),
                            bot: cfg.bot.clone(),
                            seed: seed.wrapping_mul(131),
                            real: None,
                            passes: passes.clone(),
                            out: out.clone(),
                        })
                    });
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut state = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                    state.start_round(shuffled_deck(&mut rng));
                    let (mut buf, mut n) = (vec![], 0);
                    if play_dealt_round(&mut state, &mut bots, None).is_ok() {
                        for rec in out.borrow().iter() {
                            buf.extend_from_slice(&(i as u32).to_le_bytes());
                            buf.extend_from_slice(&[rec.seat, rec.cands.len() as u8, rec.rule]);
                            for c in &rec.cands {
                                buf.extend_from_slice(&c.features);
                                buf.extend_from_slice(&c.mine.to_le_bytes());
                                buf.extend_from_slice(&c.theirs.to_le_bytes());
                                buf.push(c.flags);
                            }
                            n += 1;
                        }
                    }
                    if tx.send((n, buf)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut rounds = 0;
        for (n, buf) in rx {
            w.write_all(&buf)?;
            written += n;
            rounds += 1;
            if rounds % 50 == 0 || rounds == cfg.rounds {
                let secs = start.elapsed().as_secs_f64();
                eprint!("\r{rounds}/{} rounds, {written} pass decisions ({:.2} rounds/s)   ", cfg.rounds, rounds as f64 / secs);
            }
        }
        eprintln!();
        Ok(())
    })?;
    w.flush()?;
    Ok(written)
}
