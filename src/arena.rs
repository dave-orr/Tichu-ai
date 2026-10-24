//! Head-to-head evaluation of two bots, with seeded deals, duplicate seating
//! and multithreading.
//!
//! Tichu is noisy: the deal decides most rounds. Two things help:
//! - **Duplicate**: every deal is played twice, with the bots swapping seats,
//!   so both sides get the same cards. The per-deal score is the average.
//! - **Round mode** (the default): score independent rounds by point
//!   differential instead of whole games by win/loss. A game to 1000 is ~10
//!   rounds but yields one bit; per-round differential is far more efficient
//!   for telling bots apart. Use game mode for the final word.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::bot::{Bot, BotFactory};
use crate::engine::GameState;
use crate::scoring::RoundResult;
use crate::sim::{play_game, play_round, SimError};
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Rounds,
    Games,
}

#[derive(Debug, Clone)]
pub struct ArenaConfig {
    pub mode: Mode,
    /// Number of deals (round mode) or games (game mode). With duplicate on,
    /// each is played twice.
    pub count: usize,
    pub seed: u64,
    pub threads: usize,
    pub duplicate: bool,
    pub settings: Settings,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        ArenaConfig {
            mode: Mode::Rounds,
            count: 1000,
            seed: 1,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            duplicate: true,
            settings: Settings::default(),
        }
    }
}

/// Per-side tallies over every round played.
#[derive(Debug, Clone, Default)]
pub struct SideStats {
    pub rounds: usize,
    pub points: i64,
    pub small_calls: usize,
    pub small_made: usize,
    pub grand_calls: usize,
    pub grand_made: usize,
    pub double_victories: usize,
    pub first_out: usize,
}

impl SideStats {
    fn add_round(&mut self, r: &RoundResult, team: Team) {
        self.rounds += 1;
        self.points += r.total(team) as i64;
        for seat in [team, team + 2] {
            let made = r.out_order[0] == seat;
            match r.tichu_calls[seat] {
                TichuCall::Small => {
                    self.small_calls += 1;
                    self.small_made += made as usize;
                }
                TichuCall::Grand => {
                    self.grand_calls += 1;
                    self.grand_made += made as usize;
                }
                TichuCall::None => {}
            }
        }
        if r.double_victory == Some(team) {
            self.double_victories += 1;
        }
        if team_for_seat(r.out_order[0]) == team {
            self.first_out += 1;
        }
    }

    fn merge(&mut self, o: &SideStats) {
        self.rounds += o.rounds;
        self.points += o.points;
        self.small_calls += o.small_calls;
        self.small_made += o.small_made;
        self.grand_calls += o.grand_calls;
        self.grand_made += o.grand_made;
        self.double_victories += o.double_victories;
        self.first_out += o.first_out;
    }
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub a_name: String,
    pub b_name: String,
    pub mode: Option<Mode>,
    /// One sample per deal (round mode) or game (game mode): A's points minus
    /// B's (averaged over the two seatings with duplicate).
    pub samples: Vec<f64>,
    pub games: usize,
    pub a_game_wins: f64,
    pub a: SideStats,
    pub b: SideStats,
    pub errors: Vec<String>,
}

impl Report {
    pub fn mean(&self) -> f64 {
        self.samples.iter().sum::<f64>() / self.samples.len().max(1) as f64
    }
    /// Standard error of the mean.
    pub fn stderr(&self) -> f64 {
        let n = self.samples.len() as f64;
        if n < 2.0 {
            return f64::NAN;
        }
        let m = self.mean();
        let var = self.samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        (var / n).sqrt()
    }

    pub fn summary(&self) -> String {
        let mut s = String::new();
        let unit = match self.mode {
            Some(Mode::Games) => "game",
            _ => "deal",
        };
        s += &format!("{} vs {}  ({} {}s)\n", self.a_name, self.b_name, self.samples.len(), unit);
        s += &format!(
            "  margin per {unit} ({} − {}): {:+.1} ± {:.1}  (95% CI)\n",
            self.a_name,
            self.b_name,
            self.mean(),
            1.96 * self.stderr()
        );
        if self.games > 0 {
            s += &format!(
                "  game win rate for {}: {:.1}%  ({} games)\n",
                self.a_name,
                100.0 * self.a_game_wins / self.games as f64,
                self.games
            );
        }
        for (name, st) in [(&self.a_name, &self.a), (&self.b_name, &self.b)] {
            let pct = |n: usize, d: usize| if d == 0 { "-".to_string() } else { format!("{:.0}%", 100.0 * n as f64 / d as f64) };
            s += &format!(
                "  {:<12} pts/round {:>6.1} | first out {:>4} | 1-2 {:>4} | tichu {}/{} ({}) | grand {}/{} ({})\n",
                name,
                st.points as f64 / st.rounds.max(1) as f64,
                pct(st.first_out, st.rounds),
                pct(st.double_victories, st.rounds),
                st.small_made,
                st.small_calls,
                pct(st.small_made, st.small_calls),
                st.grand_made,
                st.grand_calls,
                pct(st.grand_made, st.grand_calls),
            );
        }
        if !self.errors.is_empty() {
            s += &format!("  {} errors, first: {}\n", self.errors.len(), self.errors[0]);
        }
        s
    }
}

/// Seed derivation so every (deal, seating, seat) gets its own stream.
pub fn mix(seed: u64, a: u64, b: u64) -> u64 {
    let mut x = seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x
}

/// Seats 0/2 get `team0`, seats 1/3 `team1`.
fn make_bots(team0: &BotFactory, team1: &BotFactory, seed: u64) -> [Box<dyn Bot>; 4] {
    [team0(mix(seed, 0, 1)), team1(mix(seed, 1, 1)), team0(mix(seed, 2, 1)), team1(mix(seed, 3, 1))]
}

struct Sample {
    diff: f64,
    a_win: f64,
    a: SideStats,
    b: SideStats,
    error: Option<String>,
}

/// Play one deal (round mode) or game (game mode), once per seating.
fn run_one(i: usize, cfg: &ArenaConfig, a: &BotFactory, b: &BotFactory) -> Sample {
    let mut out = Sample { diff: 0.0, a_win: 0.0, a: SideStats::default(), b: SideStats::default(), error: None };
    let seatings: &[bool] = if cfg.duplicate { &[false, true] } else { &[false] };
    for &swapped in seatings {
        // Same deck stream for both seatings; bots seeded by seating too.
        let mut deck_rng = StdRng::seed_from_u64(mix(cfg.seed, i as u64, 0));
        let bot_seed = mix(cfg.seed, i as u64, 1 + swapped as u64);
        let mut bots = if swapped { make_bots(b, a, bot_seed) } else { make_bots(a, b, bot_seed) };
        let a_team: Team = if swapped { 1 } else { 0 };
        let result: Result<Vec<RoundResult>, SimError> = match cfg.mode {
            Mode::Rounds => {
                let mut state = GameState::new(cfg.settings);
                play_round(&mut state, &mut bots, &mut deck_rng, None).map(|r| vec![r])
            }
            Mode::Games => play_game(cfg.settings, &mut bots, &mut deck_rng).map(|s| {
                out.a_win += if s.winner() == Some(a_team) { 1.0 } else { 0.0 };
                s.round_results
            }),
        };
        match result {
            Ok(rounds) => {
                for r in &rounds {
                    out.a.add_round(r, a_team);
                    out.b.add_round(r, 1 - a_team);
                    out.diff += (r.total(a_team) - r.total(1 - a_team)) as f64;
                }
            }
            Err(e) => {
                out.error = Some(format!("#{i} (swapped: {swapped}): {e}"));
                return out;
            }
        }
    }
    out.diff /= seatings.len() as f64;
    out.a_win /= seatings.len() as f64;
    out
}

/// Pit bot A against bot B.
pub fn run(a_name: &str, a: &BotFactory, b_name: &str, b: &BotFactory, cfg: &ArenaConfig) -> Report {
    let threads = cfg.threads.max(1).min(cfg.count.max(1));
    let mut samples: Vec<(usize, Sample)> = Vec::with_capacity(cfg.count);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t..cfg.count).step_by(threads).map(|i| (i, run_one(i, cfg, a, b))).collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            samples.extend(h.join().expect("arena worker panicked"));
        }
    });
    samples.sort_by_key(|(i, _)| *i);

    let mut report = Report {
        a_name: a_name.to_string(),
        b_name: b_name.to_string(),
        mode: Some(cfg.mode),
        ..Default::default()
    };
    for (_, s) in samples {
        if let Some(e) = s.error {
            report.errors.push(e);
            continue;
        }
        report.samples.push(s.diff);
        report.a.merge(&s.a);
        report.b.merge(&s.b);
        if cfg.mode == Mode::Games {
            report.games += 1;
            report.a_game_wins += s.a_win;
        }
    }
    report
}
