//! Self-play export: plays rounds and writes one training record per play
//! decision (state features, candidate features, the label, the final round
//! margin, and optionally search Q-values), for `train/`.
//!
//! The label comes from one of two sources:
//! - the bot in every seat (`--bot NAME`): its chosen play;
//! - a built-in PIMC labeler (`--q N`): N guessed deals × rollouts for every
//!   candidate, with the heuristic (or `--rollout NAME`, e.g. a learned net)
//!   in all four seats. Its mean margins are written as Q-values and it plays
//!   the best one, so the round is a pimcN self-play game.
//!
//! The bot still makes every other decision (calls, passing, out-of-turn
//! bombs, wishes, the Dragon). Decisions with a single option are skipped.
//!
//! File layout (little endian): a 24-byte header — `b"TCHR"`, format
//! version, `FEATURE_VERSION`, `STATE_DIM`, `ACTION_DIM`, `TOKEN_DIM` (u32
//! each) — then blocks, each starting with a tag byte. Per round, every seat
//! that made a recorded decision gets one sequence block (its whole-round
//! history tokens, `history::replay` of its final view) before its records:
//!
//! ```text
//! tag 1: u32 round id | u8 seat | u16 n tokens | n × TOKEN_DIM bytes
//! tag 0: u32 round id | u8 seat | u8 flags (1 = has Q) | u16 n options | u16 label
//!        f32 margin (my team − theirs, round points)
//!        u16 history length (a prefix of the seat's sequence)
//!        56 × u8 owner of each card (relative seat 1..3, or 255 if the
//!        card wasn't unseen by me at the decision)
//!        STATE_DIM bytes | n × ACTION_DIM bytes | (has Q) n × f32 Q
//! ```
//!
//! The owners come from the finished round (who later played each card, or
//! still held it), never from the bots' views, so the belief labels are
//! hindsight only.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::mpsc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::bots;
use crate::deck::shuffled_deck;
use crate::engine::{Action, GameState};
use crate::features::*;
use crate::history::{self, TOKEN_DIM};
use crate::scoring::RoundResult;
use crate::sim::{play_dealt_round, play_from};
use crate::types::*;
use crate::view::PlayerView;

pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone)]
pub struct SelfplayConfig {
    /// Bot in all four seats (by `bots::by_name`).
    pub bot: String,
    pub rounds: usize,
    pub seed: u64,
    pub threads: usize,
    /// PIMC labeler samples per decision (0 = label with the bot's choice).
    pub q_samples: usize,
    /// Rollout policy for the PIMC labeler (by `bots::by_name`).
    pub rollout: String,
    /// Chance of playing a uniformly random option instead of the label
    /// (the label is still recorded), for more varied positions.
    pub explore: f64,
}

struct Rec {
    seat: u8,
    /// History tokens seen at the decision.
    seq_len: u16,
    unseen: CardSet,
    /// Relative seat of each absolute seat, from the decider's view.
    rel_of: [u8; 4],
    state: Vec<u8>,
    cands: Candidates,
    label: usize,
    q: Option<Vec<f32>>,
}

/// Mean round margin for `view.seat`'s team after each option, over
/// `samples` guessed deals, with `rollout` playing all four seats.
pub fn pimc_q(
    view: &PlayerView,
    plays: &[Combo],
    cands: &Candidates,
    samples: usize,
    rollout: &mut [Box<dyn Bot>; 4],
    rng: &mut StdRng,
) -> Vec<f32> {
    let team = view.my_team();
    let mut totals = vec![0.0f64; cands.len()];
    for _ in 0..samples {
        let base = GameState::from_view(view, view.sample_hands(rng));
        for (k, opt) in cands.options.iter().enumerate() {
            let mut g = base.clone();
            let action = match opt {
                Some(i) => Action::Play { seat: view.seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                None => Action::PassTurn { seat: view.seat },
            };
            g.apply(&action).expect("candidate rejected by engine");
            if let Ok(r) = play_from(&mut g, rollout, None, false) {
                totals[k] += (r.total(team) - r.total(1 - team)) as f64;
            }
        }
    }
    totals.iter().map(|t| (t / samples.max(1) as f64) as f32).collect()
}

struct Recorder {
    inner: Box<dyn Bot>,
    rng: StdRng,
    q_samples: usize,
    rollout: [Box<dyn Bot>; 4],
    explore: f64,
    recs: Vec<Rec>,
    out: Rc<RefCell<Vec<(Rec, f32)>>>,
    seqs: Rc<RefCell<Vec<(u8, Vec<u8>)>>>,
}

impl Bot for Recorder {
    fn name(&self) -> String {
        format!("record:{}", self.inner.name())
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.grand_tichu(v)
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.small_tichu(v)
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        self.inner.pass_cards(v)
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

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let cands = Candidates::new(view, plays, can_pass);
        if cands.len() < 2 {
            return self.inner.play(view, plays, can_pass);
        }
        let (label, q) = if self.q_samples > 0 {
            let q = pimc_q(view, plays, &cands, self.q_samples, &mut self.rollout, &mut self.rng);
            let best = (0..q.len()).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap();
            (best, Some(q))
        } else {
            let choice = self.inner.play(view, plays, can_pass);
            (cands.option_of(choice).expect("bot chose an unknown option"), None)
        };
        let act = if self.explore > 0.0 && self.rng.gen_bool(self.explore) {
            self.rng.gen_range(0..cands.len())
        } else {
            label
        };
        let choice = cands.options[act];
        let hist = history::replay(view);
        let mut state = Vec::with_capacity(STATE_DIM);
        encode_state_with(view, &hist.summary, &mut state);
        self.recs.push(Rec {
            seat: view.seat as u8,
            seq_len: hist.len() as u16,
            unseen: view.unseen(),
            rel_of: [0, 1, 2, 3].map(|s| rel(view, s) as u8),
            state,
            cands,
            label,
            q,
        });
        choice
    }

    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        self.inner.round_over(view, result);
        let team = view.my_team();
        let margin = (result.total(team) - result.total(1 - team)) as f32;
        if !self.recs.is_empty() {
            self.seqs.borrow_mut().push((view.seat as u8, history::replay(view).tokens));
        }
        self.out.borrow_mut().extend(self.recs.drain(..).map(|r| (r, margin)));
    }
}

fn write_header(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"TCHR")?;
    for x in [FORMAT_VERSION, FEATURE_VERSION, STATE_DIM as u32, ACTION_DIM as u32, TOKEN_DIM as u32] {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn write_seq(buf: &mut Vec<u8>, round: u32, seat: u8, tokens: &[u8]) {
    buf.push(1);
    buf.extend_from_slice(&round.to_le_bytes());
    buf.push(seat);
    buf.extend_from_slice(&((tokens.len() / TOKEN_DIM) as u16).to_le_bytes());
    buf.extend_from_slice(tokens);
}

/// Who held each card once play began: whoever played it, else whoever
/// still holds it at the end.
fn owners(state: &GameState) -> [Option<Seat>; 56] {
    let mut own = [None; 56];
    for a in &state.log {
        if let Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } = a {
            for c in cards {
                own[c.index()] = Some(*seat);
            }
        }
    }
    for (s, p) in state.players.iter().enumerate() {
        for c in &p.hand {
            own[c.index()] = Some(s);
        }
    }
    own
}

fn write_rec(buf: &mut Vec<u8>, round: u32, rec: &Rec, margin: f32, own: &[Option<Seat>; 56]) {
    buf.push(0);
    buf.extend_from_slice(&round.to_le_bytes());
    buf.push(rec.seat);
    buf.push(rec.q.is_some() as u8);
    buf.extend_from_slice(&(rec.cands.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(rec.label as u16).to_le_bytes());
    buf.extend_from_slice(&margin.to_le_bytes());
    buf.extend_from_slice(&rec.seq_len.to_le_bytes());
    for (i, o) in own.iter().enumerate() {
        buf.push(match o {
            Some(s) if rec.unseen.contains(Card::from_index(i)) => rec.rel_of[*s],
            _ => 255,
        });
    }
    buf.extend_from_slice(&rec.state);
    buf.extend_from_slice(&rec.cands.features);
    if let Some(q) = &rec.q {
        for x in q {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
}

/// Play `cfg.rounds` rounds and write their records to `w`. Returns the
/// number of records written.
pub fn run(cfg: &SelfplayConfig, w: &mut impl Write) -> std::io::Result<usize> {
    write_header(w)?;
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
                    let seqs = Rc::new(RefCell::new(vec![]));
                    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                        let sd = seed.wrapping_mul(31).wrapping_add(s);
                        Box::new(Recorder {
                            inner: bots::by_name(&cfg.bot, sd).expect("unknown bot"),
                            rng: StdRng::seed_from_u64(sd ^ 0xABCD),
                            q_samples: cfg.q_samples,
                            rollout: [0, 1, 2, 3].map(|k| bots::by_name(&cfg.rollout, sd * 4 + k).expect("unknown bot")),
                            explore: cfg.explore,
                            recs: vec![],
                            out: out.clone(),
                            seqs: seqs.clone(),
                        })
                    });
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut state = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                    state.start_round(shuffled_deck(&mut rng));
                    let mut buf = vec![];
                    if play_dealt_round(&mut state, &mut bots, None).is_ok() {
                        let own = owners(&state);
                        for (seat, tokens) in seqs.borrow().iter() {
                            write_seq(&mut buf, i as u32, *seat, tokens);
                            for (rec, margin) in out.borrow().iter().filter(|(r, _)| r.seat == *seat) {
                                write_rec(&mut buf, i as u32, rec, *margin, &own);
                            }
                        }
                    }
                    let n = out.borrow().len();
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
            if rounds % 1000 == 0 || rounds == cfg.rounds {
                let secs = start.elapsed().as_secs_f64();
                eprint!("\r{rounds}/{} rounds, {written} records ({:.1} rounds/s)   ", cfg.rounds, rounds as f64 / secs);
            }
        }
        eprintln!();
        Ok(())
    })?;
    w.flush()?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::H2Bot;

    /// The hindsight owners match the hands as play begins.
    #[test]
    fn owners_match_hands_after_passing() {
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..20 {
            let deck = shuffled_deck(&mut rng);
            let mut g = GameState::new(Settings::default());
            g.start_round(deck.clone());
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(H2Bot::new()) as Box<dyn Bot>);
            play_dealt_round(&mut g, &mut bots, None).unwrap();
            let own = owners(&g);
            let mut r = GameState::new(Settings::default());
            r.start_round(deck);
            for a in g.log.iter().take_while(|a| !matches!(a, Action::Play { .. })) {
                r.apply(a).unwrap();
            }
            for (s, p) in r.players.iter().enumerate() {
                for c in &p.hand {
                    assert_eq!(own[c.index()], Some(s), "{c:?}");
                }
            }
        }
    }
}
