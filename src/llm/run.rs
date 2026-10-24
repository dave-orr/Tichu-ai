//! A resumable duplicate match: an LLM team against a bot, one deal at a
//! time, with every finished deal appended to `results.jsonl`.
//!
//! Deals and seeds are exactly those of `tichu arena`: deal `i` of
//! `--seed S` has the same cards, and the bots in each seat get the same
//! seeds, as in `tichu arena X OPPONENT --seed S`. That allows a paired
//! baseline: with `baseline` on, each deal is also played by the fallback
//! bot in the LLM's seats (with the same seeds), and the report compares the
//! two on the same cards.
//!
//! Restarting with the same directory skips the finished deals and replays
//! an unfinished one from the answer cache (see `client`).

use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::arena::mix;
use crate::bot::Bot;
use crate::bots;
use crate::engine::GameState;
use crate::scoring::RoundResult;
use crate::sim::play_round;
use crate::types::*;

use super::bot::{Decisions, LlmBot, LlmStats};
use super::client::Client;

/// Settings that must stay fixed across restarts of one match directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchConfig {
    pub opponent: String,
    /// Bot that makes the LLM seats' calls and covers for unusable answers.
    pub fallback: String,
    pub seed: u64,
    pub backend: String,
    pub decisions: Decisions,
}

/// One seating of one deal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeatingResult {
    /// The team the LLM (or, for the baseline, the fallback bot) played.
    pub team: Team,
    /// That team's points minus the opponents'.
    pub margin: i32,
    pub result: RoundResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DealResult {
    pub deal: usize,
    /// Mean of the two seatings' margins (LLM team − opponent).
    pub margin: f64,
    pub seatings: Vec<SeatingResult>,
    /// The same deal with the fallback bot in the LLM's seats.
    pub baseline_margin: Option<f64>,
    pub baseline: Vec<SeatingResult>,
    pub llm: LlmStats,
    pub seconds: f64,
}

/// Plays one seating of deal `i`, seeded exactly like `arena::run_one`.
/// `make(seat, seed, is_a)` builds the bot for a seat; A sits at 0/2 unless
/// `swapped`.
fn play_seating(
    seed: u64,
    i: usize,
    swapped: bool,
    make: &mut dyn FnMut(Seat, u64, bool) -> Box<dyn Bot>,
) -> Result<SeatingResult, String> {
    let mut deck_rng = StdRng::seed_from_u64(mix(seed, i as u64, 0));
    let bot_seed = mix(seed, i as u64, 1 + swapped as u64);
    let a_team: Team = if swapped { 1 } else { 0 };
    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| make(s, mix(bot_seed, s as u64, 1), team_for_seat(s) == a_team));
    let mut state = GameState::new(Settings::default());
    let r = play_round(&mut state, &mut bots, &mut deck_rng, None).map_err(|e| format!("deal {i}: {e}"))?;
    Ok(SeatingResult { team: a_team, margin: r.total(a_team) - r.total(1 - a_team), result: r })
}

fn named(name: &str, seed: u64) -> Box<dyn Bot> {
    bots::by_name(name, seed).unwrap_or_else(|| panic!("unknown bot {name}"))
}

/// Plays deal `i` both ways with the LLM team, then (optionally) the baseline.
pub fn play_deal(cfg: &MatchConfig, client: &Arc<Client>, i: usize, baseline: bool) -> Result<DealResult, String> {
    let start = Instant::now();
    let stats = Arc::new(Mutex::new(LlmStats::default()));
    let mut seatings = Vec::new();
    for swapped in [false, true] {
        let mut make = |seat: Seat, s: u64, is_a: bool| -> Box<dyn Bot> {
            if is_a {
                let tag = serde_json::json!({ "deal": i, "swapped": swapped });
                Box::new(
                    LlmBot::new(client.clone(), named(&cfg.fallback, s), cfg.decisions)
                        .with_tag(tag)
                        .with_shared_stats(stats.clone()),
                )
            } else {
                let _ = seat;
                named(&cfg.opponent, s)
            }
        };
        seatings.push(play_seating(cfg.seed, i, swapped, &mut make)?);
    }
    let mut base = Vec::new();
    if baseline {
        for swapped in [false, true] {
            let mut make = |_: Seat, s: u64, is_a: bool| named(if is_a { &cfg.fallback } else { &cfg.opponent }, s);
            base.push(play_seating(cfg.seed, i, swapped, &mut make)?);
        }
    }
    let mean = |v: &[SeatingResult]| v.iter().map(|s| s.margin as f64).sum::<f64>() / v.len() as f64;
    let llm = *stats.lock().unwrap();
    Ok(DealResult {
        deal: i,
        margin: mean(&seatings),
        baseline_margin: if baseline { Some(mean(&base)) } else { None },
        seatings,
        baseline: base,
        llm,
        seconds: start.elapsed().as_secs_f64(),
    })
}

pub fn read_results(dir: &Path) -> Vec<DealResult> {
    let Ok(f) = File::open(dir.join("results.jsonl")) else { return vec![] };
    let mut by_deal = BTreeMap::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if let Ok(r) = serde_json::from_str::<DealResult>(&line) {
            by_deal.insert(r.deal, r);
        }
    }
    by_deal.into_values().collect()
}

/// Checks `dir/config.json` against `cfg` (writing it on first use).
pub fn check_config(dir: &Path, cfg: &MatchConfig) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("config.json");
    if let Ok(s) = std::fs::read_to_string(&path) {
        let old: MatchConfig = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
        if &old != cfg {
            return Err(format!(
                "{} was started with different settings:\n  then: {old:?}\n  now:  {cfg:?}\n\
                 Use the same settings to resume, or a new --dir for a new match.",
                path.display()
            ));
        }
        return Ok(());
    }
    std::fs::write(&path, serde_json::to_string_pretty(cfg).unwrap()).map_err(|e| e.to_string())
}

pub struct RunOptions {
    pub dir: PathBuf,
    /// Total deals wanted (finished ones count).
    pub deals: usize,
    pub jobs: usize,
    pub baseline: bool,
}

/// Plays every unfinished deal below `opts.deals`, appending each to
/// `results.jsonl` as it finishes. Returns all results so far.
pub fn run(cfg: &MatchConfig, client: Arc<Client>, opts: &RunOptions) -> Result<Vec<DealResult>, String> {
    check_config(&opts.dir, cfg)?;
    let done: Vec<usize> = read_results(&opts.dir).iter().map(|r| r.deal).collect();
    let todo: VecDeque<usize> = (0..opts.deals).filter(|i| !done.contains(i)).collect();
    eprintln!(
        "{} deals finished, {} to play ({} cached answers on file).",
        done.len(),
        todo.len(),
        client.cached_answers()
    );
    let queue = Mutex::new(todo);
    let out = Mutex::new(
        OpenOptions::new().create(true).append(true).open(opts.dir.join("results.jsonl")).map_err(|e| e.to_string())?,
    );
    let errors = Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for _ in 0..opts.jobs.max(1) {
            scope.spawn(|| loop {
                let Some(i) = queue.lock().unwrap().pop_front() else { return };
                match play_deal(cfg, &client, i, opts.baseline) {
                    Ok(r) => {
                        let mut f = out.lock().unwrap();
                        let _ = writeln!(f, "{}", serde_json::to_string(&r).unwrap());
                        let _ = f.flush();
                        drop(f);
                        let all = read_results(&opts.dir);
                        let base = r.baseline_margin.map(|b| format!(" | baseline {b:+.1}")).unwrap_or_default();
                        eprintln!(
                            "deal {i}: {:+.1} (seatings {:+} / {:+}){base} | {} asked, {} fallbacks, {:.0}s  ==  {}",
                            r.margin,
                            r.seatings[0].margin,
                            r.seatings[1].margin,
                            r.llm.asked,
                            r.llm.fallbacks,
                            r.seconds,
                            short_summary(&all)
                        );
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        errors.lock().unwrap().push(e);
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        return Err(format!("{} deals failed; first: {}", errors.len(), errors[0]));
    }
    Ok(read_results(&opts.dir))
}

/// Mean and 95% half-width.
pub fn mean_ci(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n.max(1.0);
    if n < 2.0 {
        return (m, f64::NAN);
    }
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, 1.96 * (var / n).sqrt())
}

fn short_summary(rs: &[DealResult]) -> String {
    let (m, ci) = mean_ci(&rs.iter().map(|r| r.margin).collect::<Vec<_>>());
    format!("{m:+.1} ± {ci:.1} over {} deals", rs.len())
}

/// Per-side tallies over seatings.
fn side_line(name: &str, seatings: &[&SeatingResult], mine: bool) -> String {
    let (mut first, mut dv, mut sc, mut sm, mut gc, mut gm, mut pts) = (0, 0, 0, 0, 0, 0, 0i64);
    for s in seatings {
        let team = if mine { s.team } else { 1 - s.team };
        let r = &s.result;
        pts += r.total(team) as i64;
        first += (team_for_seat(r.out_order[0]) == team) as usize;
        dv += (r.double_victory == Some(team)) as usize;
        for seat in [team, team + 2] {
            let made = (r.out_order[0] == seat) as usize;
            match r.tichu_calls[seat] {
                TichuCall::Small => (sc, sm) = (sc + 1, sm + made),
                TichuCall::Grand => (gc, gm) = (gc + 1, gm + made),
                TichuCall::None => {}
            }
        }
    }
    let n = seatings.len().max(1);
    let pct = |a: usize, b: usize| if b == 0 { "-".to_string() } else { format!("{:.0}%", 100.0 * a as f64 / b as f64) };
    format!(
        "  {name:<22} pts/round {:>6.1} | first out {:>4} | 1-2 {:>4} | tichu {sm}/{sc} ({}) | grand {gm}/{gc} ({})\n",
        pts as f64 / n as f64,
        pct(first, n),
        pct(dv, n),
        pct(sm, sc),
        pct(gm, gc)
    )
}

/// The full report over finished deals.
pub fn report(cfg: &MatchConfig, rs: &[DealResult]) -> String {
    let mut s = String::new();
    let llm = format!("llm[{}]", cfg.backend);
    s += &format!("{llm} vs {}  ({} deals, duplicate, seed {})\n", cfg.opponent, rs.len(), cfg.seed);
    if rs.is_empty() {
        return s;
    }
    let (m, ci) = mean_ci(&rs.iter().map(|r| r.margin).collect::<Vec<_>>());
    s += &format!("  margin per deal ({llm} − {}): {m:+.1} ± {ci:.1}  (95% CI)\n", cfg.opponent);
    let paired: Vec<(f64, f64)> = rs.iter().filter_map(|r| r.baseline_margin.map(|b| (r.margin, b))).collect();
    if !paired.is_empty() {
        let (bm, bci) = mean_ci(&paired.iter().map(|p| p.1).collect::<Vec<_>>());
        let (dm, dci) = mean_ci(&paired.iter().map(|p| p.0 - p.1).collect::<Vec<_>>());
        s += &format!(
            "  baseline, same deals ({} − {}): {bm:+.1} ± {bci:.1}\n  llm − baseline (paired): {dm:+.1} ± {dci:.1}  ({} deals)\n",
            cfg.fallback,
            cfg.opponent,
            paired.len()
        );
    }
    let seat: Vec<&SeatingResult> = rs.iter().flat_map(|r| r.seatings.iter()).collect();
    s += &side_line(&llm, &seat, true);
    s += &side_line(&cfg.opponent, &seat, false);
    let base: Vec<&SeatingResult> = rs.iter().flat_map(|r| r.baseline.iter()).collect();
    if !base.is_empty() {
        s += &side_line(&format!("{} (baseline)", cfg.fallback), &base, true);
    }
    let mut st = LlmStats::default();
    for r in rs {
        st.add(&r.llm);
    }
    let secs: f64 = rs.iter().map(|r| r.seconds).sum();
    s += &format!(
        "  questions {} ({:.0}/deal), forced {}, fallbacks {} | {:.1} min/deal\n",
        st.asked,
        st.asked as f64 / rs.len() as f64,
        st.forced,
        st.fallbacks,
        secs / 60.0 / rs.len() as f64
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{self, ArenaConfig};
    use crate::llm::client::Stub;

    fn cfg(fallback: &str, opponent: &str) -> MatchConfig {
        MatchConfig {
            opponent: opponent.into(),
            fallback: fallback.into(),
            seed: 7,
            backend: "stub".into(),
            decisions: Decisions::default(),
        }
    }

    /// The baseline is exactly `tichu arena FALLBACK OPPONENT`, deal by deal.
    #[test]
    fn baseline_matches_arena() {
        let c = cfg("heuristic", "random");
        let client = Arc::new(Client::in_memory(Box::new(Stub::new(0))));
        let acfg = ArenaConfig { count: 12, seed: 7, threads: 2, ..Default::default() };
        let rep = arena::run("heuristic", &|s| named("heuristic", s), "random", &|s| named("random", s), &acfg);
        for i in 0..12 {
            let r = play_deal(&c, &client, i, true).unwrap();
            assert_eq!(r.baseline_margin.unwrap(), rep.samples[i], "deal {i}");
        }
    }

    /// An LLM whose answers are all unusable plays exactly like its fallback.
    #[test]
    fn garbage_answers_fall_back() {
        let c = cfg("heuristic", "random");
        let client = Arc::new(Client::in_memory(Box::new(Stub::new(1))));
        for i in 0..5 {
            let r = play_deal(&c, &client, i, true).unwrap();
            assert_eq!(Some(r.margin), r.baseline_margin);
            assert_eq!(r.llm.fallbacks, r.llm.asked);
        }
    }

    /// Stopping and restarting replays from the cache and gives the same result.
    #[test]
    fn resumes_from_cache() {
        let dir = std::env::temp_dir().join(format!("llm-run-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = cfg("heuristic", "heuristic");
        let opts = |deals| RunOptions { dir: dir.clone(), deals, jobs: 2, baseline: false };

        let stub = Arc::new(Client::with_dir(Box::new(Stub::new(0)), &dir).unwrap());
        let first = run(&c, stub.clone(), &opts(3)).unwrap();
        let fresh = stub.stats().fresh;
        assert!(fresh > 0);
        assert_eq!(first.len(), 3);
        // The stub always answers in range, so nothing falls back.
        assert!(first.iter().all(|r| r.llm.fallbacks == 0 && r.llm.asked > 0));

        // Drop deal 2 from the results, as if the run had been killed mid-deal.
        let lines: Vec<String> = std::fs::read_to_string(dir.join("results.jsonl"))
            .unwrap()
            .lines()
            .filter(|l| !l.contains("\"deal\":2,"))
            .map(String::from)
            .collect();
        std::fs::write(dir.join("results.jsonl"), lines.join("\n") + "\n").unwrap();

        let again = Arc::new(Client::with_dir(Box::new(Stub::new(0)), &dir).unwrap());
        let second = run(&c, again.clone(), &opts(4)).unwrap();
        assert_eq!(second.len(), 4);
        assert_eq!(second[2].margin, first[2].margin);
        assert_eq!(second[2].seatings[0].result, first[2].seatings[0].result);
        // Deal 2 came entirely from the cache; only deal 3 asked anything new.
        let st = again.stats();
        assert!(st.cached >= first[2].llm.asked);
        assert_eq!(st.fresh + st.cached, second[2].llm.asked + second[3].llm.asked);

        let changed = MatchConfig { opponent: "random".into(), ..c };
        assert!(check_config(&dir, &changed).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
