//! How much of the all-seeing critic's verdict can be known from one hand?
//!
//! Self-play with NET; at random decisions (2+ options) take the net's top K
//! options and play each out to the end of the round (NET in every seat,
//! players see only their own hands), from:
//! - the real deal (what `selfplay --critic` labels with);
//! - the real hands put through `GameState::from_view_with_history`, as a
//!   check that rebuilt states play out like real ones;
//! - M uniform guesses of the hidden cards (`view.sample_hands`);
//! - M guesses weighted by how well they explain the play so far
//!   (`infer::sample_hands_weighted`, `--infer-k` candidates per guess).
//!
//! Writes one JSON line per decision (the net's logits and every margin);
//! `train/critic_ceiling.py` turns them into the numbers.
//!
//! cargo run --release --example critic_ceiling -- --net PATH --out FILE [--rounds N] [--p 0.05] [--k 4] [--m 4] [--infer-k 16] [--seed S] [--threads T]

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tichu_engine::bot::Bot;
use tichu_engine::bots::{self, net::Net};
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState, Pending};
use tichu_engine::features::Candidates;
use tichu_engine::infer::{self, InferConfig};
use tichu_engine::sim::{play_dealt_round, play_from};
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

struct Cfg {
    name: String,
    net: Arc<Net>,
    p: f64,
    k: usize,
    m: usize,
    infer_k: usize,
}

struct Probe {
    inner: Box<dyn Bot>,
    cfg: Arc<Cfg>,
    rng: StdRng,
    real: Option<GameState>,
    out: Rc<RefCell<Vec<String>>>,
}

fn margins(cfg: &Cfg, base: &GameState, view: &PlayerView, plays: &[Combo], opts: &[Option<usize>]) -> Option<Vec<f32>> {
    let team = view.my_team();
    opts.iter()
        .map(|o| {
            let mut g = base.clone();
            let action = match o {
                Some(i) => Action::Play { seat: view.seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                None => Action::PassTurn { seat: view.seat },
            };
            g.apply(&action).ok()?;
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| bots::by_name(&cfg.name, s).unwrap());
            let r = play_from(&mut g, &mut bots, None, true).ok()?;
            Some((r.total(team) - r.total(1 - team)) as f32)
        })
        .collect()
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
        let cfg = self.cfg.clone();
        if cands.len() < 2 || !self.rng.gen_bool(cfg.p) {
            return choice;
        }
        let Some(real) = self.real.clone().filter(|g| g.pending() == Pending::Turn(view.seat)) else { return choice };
        let logits = cfg.net.option_logits(view, &cands);
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(cfg.k);
        let opts: Vec<Option<usize>> = order.iter().map(|&k| cands.options[k]).collect();
        let lg: Vec<f32> = order.iter().map(|&k| logits[k]).collect();
        let hands: [Vec<Card>; 4] = [0, 1, 2, 3].map(|s| real.players[s].hand.clone());
        let rebuilt = GameState::from_view_with_history(view, hands);
        let uniform: Vec<[Vec<Card>; 4]> = (0..cfg.m).map(|_| view.sample_hands(&mut self.rng)).collect();
        let weighted = infer::sample_hands_weighted(view, &mut self.rng, cfg.m, &InferConfig { k: cfg.infer_k, ..Default::default() });
        let many = |gs: &[[Vec<Card>; 4]]| -> Option<Vec<Vec<f32>>> {
            gs.iter().map(|g| margins(&cfg, &GameState::from_view_with_history(view, g.clone()), view, plays, &opts)).collect()
        };
        let (Some(qr), Some(qb), Some(qu), Some(qw)) =
            (margins(&cfg, &real, view, plays, &opts), margins(&cfg, &rebuilt, view, plays, &opts), many(&uniform), many(&weighted))
        else {
            return choice;
        };
        let line = serde_json::json!({
            "cards_left": view.hand.len(), "plays_so_far": view.log.len(), "logits": lg,
            "real": qr, "rebuilt": qb, "uniform": qu, "weighted": qw,
        });
        self.out.borrow_mut().push(line.to_string());
        choice
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
        p: get("--p").map_or(0.05, |v| v.parse().unwrap()),
        k: get("--k").map_or(4, |v| v.parse().unwrap()),
        m: get("--m").map_or(4, |v| v.parse().unwrap()),
        infer_k: get("--infer-k").map_or(16, |v| v.parse().unwrap()),
    });
    let file = Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap()));
    let next = Mutex::new((0usize, 0usize));
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
                let mut bots: [Box<dyn Bot>; 4] = [0u64, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                    Box::new(Probe {
                        inner: bots::by_name(&cfg.name, rs * 4 + s).unwrap(),
                        cfg: cfg.clone(),
                        rng: StdRng::seed_from_u64(rs.wrapping_mul(31).wrapping_add(s)),
                        real: None,
                        out: out.clone(),
                    })
                });
                let mut rng = StdRng::seed_from_u64(rs);
                let mut g = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                g.start_round(shuffled_deck(&mut rng));
                let _ = play_dealt_round(&mut g, &mut bots, None);
                let mut f = file.lock().unwrap();
                for l in out.borrow().iter() {
                    writeln!(f, "{l}").unwrap();
                }
                let mut n = next.lock().unwrap();
                n.1 += out.borrow().len();
                if n.0 % 100 == 0 {
                    eprint!("\r{}/{rounds} rounds, {} decisions ({:.0}s)   ", n.0, n.1, start.elapsed().as_secs_f64());
                }
            });
        }
    });
    file.lock().unwrap().flush().unwrap();
    eprintln!("\nwrote {} decisions to {out_path} ({:.0}s)", next.lock().unwrap().1, start.elapsed().as_secs_f64());
}
