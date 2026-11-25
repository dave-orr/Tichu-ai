//! Data for an all-seeing value net and a truncated critic.
//!
//! Self-play with NET. Two kinds of rows, each for a random share of the
//! decisions with 2+ options:
//!
//! - value rows: the position as all four players see it (`oracle
//!   features`: the mover's `encode_state`, then the next three seats' in
//!   play order of seats) and the round's final margin for the mover's team.
//!   An "all-seeing" value net is trained on these.
//! - critic rows: the mover's own state features and, for each of the net's
//!   top K options, the option's features and what happens on the real deal
//!   when NET plays on from it in every seat (players see only their own
//!   hands): the final margin of the full playout, the oracle features right
//!   after the move, and the oracle features `--horizon` card-play decisions
//!   later (or the final margin if the round ends first).
//!
//! File (little endian): `b"TORC"`, u32 STATE_DIM, u32 ACTION_DIM, then rows
//!   value:  u8 0 | f32 margin | 4·S bytes
//!   critic: u8 1 | u8 n | S bytes | n × option
//!   option: A bytes | f32 full margin
//!           | u8 ended0 | f32 margin0 | 4·S bytes   (right after the move)
//!           | u8 ended1 | f32 margin1 | 4·S bytes   (after the horizon)
//! `ended` = 1 means the round was over by then: the margin is the result and
//! the features are zeros.
//!
//! cargo run --release --example oracle_data -- --net PATH --out FILE [--rounds N] [--p-value 0.05] [--p-critic 0.05] [--k 4] [--horizon 12] [--seed S] [--threads T]

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tichu_engine::bot::Bot;
use tichu_engine::bots::{self, net::Net};
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState, Pending};
use tichu_engine::features::{encode_state, Candidates, ACTION_DIM, STATE_DIM};
use tichu_engine::scoring::RoundResult;
use tichu_engine::sim::{play_dealt_round, play_from};
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

struct Cfg {
    name: String,
    net: Arc<Net>,
    p_value: f64,
    p_critic: f64,
    k: usize,
    horizon: usize,
}

/// The position as all four players see it, starting with `seat`.
fn oracle_features(g: &GameState, seat: Seat, out: &mut Vec<u8>) {
    for i in 0..4 {
        encode_state(&g.view((seat + i) % 4), out);
    }
}

/// A rollout player that stops the playout (by an out-of-range answer,
/// which `play_from` rejects, leaving the state at that turn) once `left`
/// card-play decisions have been made.
struct Limited {
    inner: Box<dyn Bot>,
    left: Rc<Cell<usize>>,
}

impl Bot for Limited {
    fn name(&self) -> String {
        self.inner.name()
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
    fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if self.left.get() == 0 {
            return Some(usize::MAX);
        }
        self.left.set(self.left.get() - 1);
        self.inner.play(v, plays, can_pass)
    }
}

struct Probe {
    inner: Box<dyn Bot>,
    cfg: Arc<Cfg>,
    rng: StdRng,
    real: Option<GameState>,
    /// This round's value rows, waiting for the margin.
    pending: Vec<Vec<u8>>,
    out: Rc<RefCell<Vec<u8>>>,
    counts: Rc<Cell<(usize, usize)>>,
}

impl Probe {
    fn stage(buf: &mut Vec<u8>, g: &GameState, seat: Seat, result: Option<f32>) {
        buf.push(result.is_some() as u8);
        buf.extend_from_slice(&result.unwrap_or(0.0).to_le_bytes());
        if result.is_some() {
            buf.resize(buf.len() + 4 * STATE_DIM, 0);
        } else {
            oracle_features(g, seat, buf);
        }
    }

    fn critic_row(&mut self, view: &PlayerView, plays: &[Combo], cands: &Candidates, real: &GameState) -> Option<Vec<u8>> {
        let cfg = self.cfg.clone();
        let team = view.my_team();
        let margin = |r: &RoundResult| (r.total(team) - r.total(1 - team)) as f32;
        let logits = cfg.net.option_logits(view, cands);
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(cfg.k);
        let mut row = vec![1u8, order.len() as u8];
        encode_state(view, &mut row);
        for &k in &order {
            let mut g = real.clone();
            let action = match cands.options[k] {
                Some(i) => Action::Play { seat: view.seat, cards: plays[i].cards.clone(), phoenix_as: plays[i].phoenix_as },
                None => Action::PassTurn { seat: view.seat },
            };
            g.apply(&action).ok()?;
            let left = Rc::new(Cell::new(0usize));
            let mut bots: [Box<dyn Bot>; 4] =
                [0, 1, 2, 3].map(|s| -> Box<dyn Bot> { Box::new(Limited { inner: bots::by_name(&cfg.name, s).unwrap(), left: left.clone() }) });
            // Up to the first card-play decision: the state right after the move.
            let mut stages = vec![];
            let mut full = None;
            for limit in [0, cfg.horizon] {
                if full.is_none() {
                    left.set(limit);
                    match play_from(&mut g, &mut bots, None, true) {
                        Ok(r) => full = Some(margin(&r)),
                        Err(_) if matches!(g.pending(), Pending::Turn(_)) => {}
                        Err(_) => return None,
                    }
                }
                Self::stage(&mut stages, &g, view.seat, full);
            }
            if full.is_none() {
                left.set(usize::MAX);
                full = Some(margin(&play_from(&mut g, &mut bots, None, true).ok()?));
            }
            row.extend_from_slice(&cands.features[k * ACTION_DIM..(k + 1) * ACTION_DIM]);
            row.extend_from_slice(&full?.to_le_bytes());
            row.extend_from_slice(&stages);
        }
        Some(row)
    }
}

impl Bot for Probe {
    fn name(&self) -> String {
        "probe".into()
    }
    fn cheats(&self) -> bool {
        true
    }
    fn peek(&mut self, state: &GameState) {
        self.real = matches!(state.pending(), Pending::Turn(_)).then(|| state.clone());
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
        let choice = self.inner.play(view, plays, can_pass);
        let cands = Candidates::new(view, plays, can_pass);
        if cands.len() < 2 {
            return choice;
        }
        let Some(real) = self.real.clone().filter(|g| g.pending() == Pending::Turn(view.seat)) else { return choice };
        if self.rng.gen_bool(self.cfg.p_value) {
            let mut row = vec![0u8, 0, 0, 0, 0];
            oracle_features(&real, view.seat, &mut row);
            self.pending.push(row);
        }
        if self.rng.gen_bool(self.cfg.p_critic) {
            if let Some(row) = self.critic_row(view, plays, &cands, &real) {
                self.out.borrow_mut().extend_from_slice(&row);
                let (v, c) = self.counts.get();
                self.counts.set((v, c + 1));
            }
        }
        choice
    }
    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        let team = view.my_team();
        let margin = (result.total(team) - result.total(1 - team)) as f32;
        for mut row in self.pending.drain(..) {
            row[1..5].copy_from_slice(&margin.to_le_bytes());
            self.out.borrow_mut().extend_from_slice(&row);
            let (v, c) = self.counts.get();
            self.counts.set((v + 1, c));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let path = get("--net").expect("--net PATH");
    let out_path = get("--out").expect("--out FILE");
    let rounds: usize = get("--rounds").map_or(2000, |v| v.parse().unwrap());
    let seed: u64 = get("--seed").map_or(1, |v| v.parse().unwrap());
    let threads: usize = get("--threads").map_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()), |v| v.parse().unwrap());
    let cfg = Arc::new(Cfg {
        name: format!("net:{path}"),
        net: Net::cached(&path).expect("can't load the net"),
        p_value: get("--p-value").map_or(0.05, |v| v.parse().unwrap()),
        p_critic: get("--p-critic").map_or(0.05, |v| v.parse().unwrap()),
        k: get("--k").map_or(4, |v| v.parse().unwrap()),
        horizon: get("--horizon").map_or(12, |v| v.parse().unwrap()),
    });
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    f.write_all(b"TORC").unwrap();
    f.write_all(&(STATE_DIM as u32).to_le_bytes()).unwrap();
    f.write_all(&(ACTION_DIM as u32).to_le_bytes()).unwrap();
    let file = Mutex::new(f);
    let next = Mutex::new((0usize, 0usize, 0usize));
    let start = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = {
                    let mut n = next.lock().unwrap();
                    if n.0 >= rounds {
                        return;
                    }
                    n.0 += 1;
                    n.0 - 1
                };
                let rs = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64);
                let out = Rc::new(RefCell::new(vec![]));
                let counts = Rc::new(Cell::new((0, 0)));
                let mut bots: [Box<dyn Bot>; 4] = [0u64, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                    Box::new(Probe {
                        inner: bots::by_name(&cfg.name, rs * 4 + s).unwrap(),
                        cfg: cfg.clone(),
                        rng: StdRng::seed_from_u64(rs.wrapping_mul(31).wrapping_add(s)),
                        real: None,
                        pending: vec![],
                        out: out.clone(),
                        counts: counts.clone(),
                    })
                });
                let mut rng = StdRng::seed_from_u64(rs);
                let mut g = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                g.start_round(shuffled_deck(&mut rng));
                if play_dealt_round(&mut g, &mut bots, None).is_ok() {
                    file.lock().unwrap().write_all(&out.borrow()).unwrap();
                    let mut n = next.lock().unwrap();
                    n.1 += counts.get().0;
                    n.2 += counts.get().1;
                    if n.0 % 200 == 0 {
                        eprint!("\r{}/{rounds} rounds, {} value rows, {} critic rows ({:.0}s)   ", n.0, n.1, n.2, start.elapsed().as_secs_f64());
                    }
                }
            });
        }
    });
    file.lock().unwrap().flush().unwrap();
    let n = next.lock().unwrap();
    eprintln!("\nwrote {} value rows and {} critic rows to {out_path} ({:.0}s)", n.1, n.2, start.elapsed().as_secs_f64());
}
