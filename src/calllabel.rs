//! Training data for learned Tichu / Grand Tichu calls (`tichu calllabel`).
//!
//! Plays rounds with `bot` in every seat. At every call decision (each
//! seat's Grand Tichu choice, and every Small Tichu offer to an eligible
//! seat) it plays the round out twice from that exact state, once with the
//! call and once without, with the same bots. The difference between the
//! two results is what the call was worth in that deal. Rollouts see the
//! real hidden hands (that's what makes the label fair for this deal), but
//! the recorded features are only the deciding seat's `PlayerView`.
//!
//! - Grand Tichu: "call Grand" against "don't", after which the seat makes
//!   its later Small Tichu decision with `bot`'s own policy.
//! - Small Tichu: "call now" against "never call this round".
//!
//! The main round goes on with `bot`'s own decisions, so the decision states
//! come from ordinary play. Rounds start at 0-0, and `train/calls.py`
//! attaches game scores afterwards (card play doesn't depend on the score),
//! unless `scores` gives game scores to start rounds from. Use those when
//! the bots' calls depend on the score (e.g. `callnet:` bots), so the other
//! seats' calls in the rollouts fit the score.
//!
//! File layout (little endian): a 20-byte header (`b"TCAL"`, format
//! version, `FEATURE_VERSION`, `STATE_DIM`, `CALL_EXTRA_DIM`, u32 each),
//! then records:
//!
//! ```text
//! u8 kind (0 = Grand, 1 = Small, 2 = round) | u32 round id | u8 seat
//! kind 2:    i16 team 0 total | i16 team 1 total     (the main round)
//! kind 0, 1: u8 heuristic (0 = undecided, 1 = no, 2 = call)
//!            i16 mine, theirs without the call | i16 mine, theirs with it
//!            i16 my team's game score, theirs (before the round)
//!            STATE_DIM + CALL_EXTRA_DIM feature bytes (encoded at 0-0;
//!            the score goes to the model separately)
//! ```

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::mpsc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;

use crate::bots::calls;
use crate::deck::shuffled_deck;
use crate::engine::{Action, GameState};
use crate::features::{encode_state, FEATURE_VERSION, STATE_DIM};
use crate::scoring::RoundResult;
use crate::sim::{play_dealt_round, play_from};
use crate::types::*;
use crate::view::PlayerView;

pub const FORMAT_VERSION: u32 = 2;
pub const CALL_EXTRA_DIM: usize = 32;
/// Inputs of a call model: state bytes, extra bytes (each / 255), then
/// is-Grand, my score / 1000, theirs / 1000.
pub const CALL_INPUT_DIM: usize = STATE_DIM + CALL_EXTRA_DIM + 3;

fn frac(out: &mut Vec<u8>, x: f32, max: f32) {
    out.push(((x / max).clamp(0.0, 1.0) * 255.0).round() as u8);
}

fn bit(out: &mut Vec<u8>, b: bool) {
    out.push(if b { 255 } else { 0 });
}

/// Call-specific features on top of `encode_state`: the v2 rules' hand
/// evaluation, and how the wish meets my hand.
pub fn encode_call_extra(view: &PlayerView, out: &mut Vec<u8>) {
    let start = out.len();
    let eval = calls::tichu_eval(view);
    let d = &eval.decomposition;
    frac(out, eval.winners, 8.0);
    frac(out, eval.losers + 1.0, 12.0);
    frac(out, eval.margin() + 8.0, 16.0);
    let g = calls::grand_rule(view);
    for r in 0..5 {
        bit(out, g == r);
    }
    frac(out, d.combos.len() as f32, 14.0);
    let singles: Vec<&Combo> = d.combos.iter().filter(|c| c.kind == ComboKind::Single).collect();
    frac(out, singles.len() as f32, 14.0);
    frac(out, singles.iter().filter(|c| c.rank < 9.0 && c.cards[0].rank().is_some()).count() as f32, 10.0);
    frac(out, d.combos.iter().filter(|c| c.is_bomb()).count() as f32, 3.0);
    frac(out, d.combos.iter().filter(|c| c.length >= 5 && !c.is_bomb()).count() as f32, 3.0);
    frac(out, d.combos.iter().map(|c| c.length).max().unwrap_or(0) as f32, 14.0);

    // The wish: is it pending, do I hold the rank, and what would it break?
    let w = view.wish;
    let held = w.map_or(0, |r| view.hand.iter().filter(|c| c.rank() == Some(r)).count());
    bit(out, w.is_some());
    frac(out, held as f32, 4.0);
    bit(out, held > 0);
    let combo_with = |f: &dyn Fn(&Combo) -> bool| w.is_some_and(|r| d.combos.iter().any(|c| c.contains_rank(r) && f(c)));
    bit(out, combo_with(&|c| c.is_bomb()));
    bit(out, combo_with(&|c| c.length >= 5 && !c.is_bomb()));
    bit(out, combo_with(&|c| c.kind != ComboKind::Single && !c.is_bomb()));
    // Ranks of bombs: does the wished rank hit a four of a kind I hold?
    bit(out, held == 4);

    // Where the decision falls.
    let me = view.me();
    bit(out, view.turn == view.seat);
    bit(out, me.has_played_first_card);
    frac(out, view.played_cards.len() as f32, 56.0);
    bit(out, view.players[view.partner()].tichu != TichuCall::None);
    frac(out, view.opponents().iter().filter(|&&o| view.players[o].tichu != TichuCall::None).count() as f32, 2.0);
    bit(out, view.players.iter().any(|p| p.is_out()));
    let n = out.len() - start;
    assert!(n <= CALL_EXTRA_DIM, "call features overflow: {n}");
    out.resize(start + CALL_EXTRA_DIM, 0);
}

/// The model inputs for a call decision (scores taken from `view`).
pub fn call_inputs(view: &PlayerView, grand: bool) -> Vec<f32> {
    let mut v0 = view.clone();
    v0.scores = [0, 0];
    let mut bytes = Vec::with_capacity(STATE_DIM + CALL_EXTRA_DIM);
    encode_state(&v0, &mut bytes);
    encode_call_extra(&v0, &mut bytes);
    let mut x: Vec<f32> = bytes.iter().map(|&b| b as f32 / 255.0).collect();
    let team = view.my_team();
    x.push(grand as u8 as f32);
    x.push(view.scores[team] as f32 / 1000.0);
    x.push(view.scores[1 - team] as f32 / 1000.0);
    x
}

#[derive(Debug, Clone)]
pub struct CallLabelConfig {
    /// Bot in all four seats, in the main round and the rollouts.
    pub bot: String,
    pub rounds: usize,
    pub seed: u64,
    pub threads: usize,
    /// Chance of labelling each Small Tichu offer after the first one a
    /// seat gets (later offers are many and alike).
    pub later_small: f64,
    /// Game scores (team 0, team 1) to start rounds from, one picked at
    /// random per round. Empty = 0-0.
    pub scores: Vec<[i32; 2]>,
}

struct Recorder {
    inner: Box<dyn Bot>,
    cfg: CallLabelConfig,
    round: u32,
    rollout_seed: u64,
    rng: StdRng,
    state: Option<GameState>,
    small_offers: usize,
    out: Rc<RefCell<Vec<u8>>>,
}

fn team_totals(r: &RoundResult, team: Team) -> [i16; 2] {
    [r.total(team) as i16, r.total(1 - team) as i16]
}

impl Recorder {
    fn rollout(&self, state: &GameState, first: Action, seat: Seat, never_small: bool) -> Option<[i16; 2]> {
        let mut s = state.clone();
        s.apply(&first).ok()?;
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| {
            let name = if never_small && k == seat { format!("nocalls:{}", self.cfg.bot) } else { self.cfg.bot.clone() };
            crate::bots::by_name(&name, self.rollout_seed.wrapping_add(k as u64)).expect("unknown bot")
        });
        let r = play_from(&mut s, &mut bots, None, true).ok()?;
        Some(team_totals(&r, seat % 2))
    }

    fn record(&self, kind: u8, view: &PlayerView, heuristic: u8, without: [i16; 2], with: [i16; 2]) {
        let mut out = self.out.borrow_mut();
        out.push(kind);
        out.extend_from_slice(&self.round.to_le_bytes());
        out.push(view.seat as u8);
        out.push(heuristic);
        let team = view.my_team();
        let score = [view.scores[team] as i16, view.scores[1 - team] as i16];
        for v in [without[0], without[1], with[0], with[1], score[0], score[1]] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut v0 = view.clone();
        v0.scores = [0, 0];
        encode_state(&v0, &mut out);
        encode_call_extra(&v0, &mut out);
    }
}

impl Bot for Recorder {
    fn name(&self) -> String {
        format!("calllabel:{}", self.inner.name())
    }
    fn cheats(&self) -> bool {
        true
    }
    fn peek(&mut self, state: &GameState) {
        // Only call decisions need the state; skip the copy otherwise.
        let deciding = state.phase == Phase::GrandTichuWindow || (0..4).any(|s| state.can_call_small_tichu(s));
        self.state = deciding.then(|| state.clone());
    }
    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        if let Some(st) = self.state.take() {
            let seat = view.seat;
            let no = self.rollout(&st, Action::GrandTichu { seat, call: false }, seat, false);
            let yes = self.rollout(&st, Action::GrandTichu { seat, call: true }, seat, false);
            if let (Some(no), Some(yes)) = (no, yes) {
                let h = if calls::grand_tichu_v2(view) { 2 } else { 1 };
                self.record(0, view, h, no, yes);
            }
        }
        self.inner.grand_tichu(view)
    }
    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        if let Some(st) = self.state.take() {
            self.small_offers += 1;
            let label = self.small_offers == 1 || self.rng.gen_bool(self.cfg.later_small);
            if label && st.can_call_small_tichu(view.seat) {
                let seat = view.seat;
                // Without: play on from the offer with this seat declining
                // now and at every later offer.
                let no = {
                    let mut s = st.clone();
                    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| {
                        let name = if k == seat { format!("nocalls:{}", self.cfg.bot) } else { self.cfg.bot.clone() };
                        crate::bots::by_name(&name, self.rollout_seed.wrapping_add(k as u64)).expect("unknown bot")
                    });
                    play_from(&mut s, &mut bots, None, true).ok().map(|r| team_totals(&r, seat % 2))
                };
                let yes = self.rollout(&st, Action::SmallTichu { seat }, seat, true);
                if let (Some(no), Some(yes)) = (no, yes) {
                    let h = match calls::small_tichu_v2(view, calls::DEFAULT_TICHU_THRESHOLD) {
                        None => 0,
                        Some(false) => 1,
                        Some(true) => 2,
                    };
                    self.record(1, view, h, no, yes);
                }
            }
        }
        self.inner.small_tichu(view)
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.state = None;
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        self.state = None;
        self.inner.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.state = None;
        self.inner.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.state = None;
        self.inner.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.state = None;
        self.inner.dragon_recipient(view)
    }
}

fn write_header(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"TCAL")?;
    for v in [FORMAT_VERSION, FEATURE_VERSION, STATE_DIM as u32, CALL_EXTRA_DIM as u32] {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Play `cfg.rounds` rounds and write their call records to `w`. Returns
/// (Grand records, Small records).
pub fn run(cfg: &CallLabelConfig, w: &mut impl Write) -> std::io::Result<(usize, usize)> {
    write_header(w)?;
    let threads = cfg.threads.max(1).min(cfg.rounds.max(1));
    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(threads * 4);
    let (mut grand, mut small) = (0, 0);
    let start = std::time::Instant::now();
    std::thread::scope(|scope| -> std::io::Result<()> {
        for t in 0..threads {
            let tx = tx.clone();
            scope.spawn(move || {
                for i in (t..cfg.rounds).step_by(threads) {
                    let seed = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64);
                    let out = Rc::new(RefCell::new(vec![]));
                    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                        let sd = seed.wrapping_mul(31).wrapping_add(s);
                        Box::new(Recorder {
                            inner: crate::bots::by_name(&cfg.bot, sd).expect("unknown bot"),
                            cfg: cfg.clone(),
                            round: i as u32,
                            rollout_seed: seed.wrapping_mul(7),
                            rng: StdRng::seed_from_u64(sd ^ 0x5EED),
                            state: None,
                            small_offers: 0,
                            out: out.clone(),
                        })
                    });
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut state = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                    if !cfg.scores.is_empty() {
                        state.scores = cfg.scores[rng.gen_range(0..cfg.scores.len())];
                    }
                    state.start_round(shuffled_deck(&mut rng));
                    let mut buf = vec![];
                    if let Ok(r) = play_dealt_round(&mut state, &mut bots, None) {
                        buf = out.borrow().clone();
                        buf.push(2);
                        buf.extend_from_slice(&(i as u32).to_le_bytes());
                        buf.push(0);
                        for v in team_totals(&r, 0) {
                            buf.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    if tx.send(buf).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let rec = 1 + 4 + 1 + 1 + 12 + STATE_DIM + CALL_EXTRA_DIM;
        let mut rounds = 0;
        for buf in rx {
            // Count records by walking the buffer.
            let mut p = 0;
            while p < buf.len() {
                match buf[p] {
                    0 => grand += 1,
                    1 => small += 1,
                    _ => {}
                }
                p += if buf[p] == 2 { 10 } else { rec };
            }
            w.write_all(&buf)?;
            rounds += 1;
            if rounds % 500 == 0 {
                eprintln!(
                    "  {rounds}/{} rounds, {grand} grand + {small} small records ({:.0}s)",
                    cfg.rounds,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        Ok(())
    })?;
    Ok((grand, small))
}
