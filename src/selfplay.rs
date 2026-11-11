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
//! each) — then one block per seat per round that has records: the seat's
//! history tokens for the whole round (`features::encode_history`), then its
//! records, each of which saw the first `prefix` tokens.
//!
//! ```text
//! block:  u32 round id | u8 seat | u16 n tokens | u32 n records
//!         n tokens × TOKEN_DIM bytes | records
//! record: u8 flags (1 = has Q) | u16 n options | u16 label | u16 prefix
//!         f32 margin (my team − theirs, round points)
//!         STATE_DIM bytes | n × ACTION_DIM bytes | (has Q) n × f32 Q
//! ```

use std::cell::RefCell;
use std::io::Write;
use std::sync::Arc;
use crate::bots::net::Net;
use std::rc::Rc;
use std::sync::mpsc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bot::Bot;
use crate::bots;
use crate::deck::shuffled_deck;
use crate::endgame;
use crate::infer;
use crate::engine::{Action, GameState};
use crate::features::*;
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
    /// With `q_samples`: in two-player endgames (perfect information) label
    /// with the solver's exact value of every option
    /// (`endgame::solve_options`) instead of rollouts. Falls back to
    /// rollouts if the solve runs past its node budget.
    pub exact_endgame: bool,
    /// With `q_samples`: weight the guessed deals by how well they explain
    /// the other players' plays and calls (`infer::sample_hands_weighted`,
    /// this many uniform candidates per sample). 0 or 1 = uniform guesses.
    pub infer: usize,
    /// With `q_samples`: keep sampling past `q_samples`, up to this many,
    /// until the best option is clearly ahead of the runner-up (paired
    /// difference over 2 standard errors). 0 = fixed `q_samples`.
    pub q_max: usize,
    /// With `q_samples` and a `net:` rollout: search only that net's top
    /// this-many options and keep its choice unless another is clearly
    /// ahead (`Teacher::anchor`). 0 = search every option.
    pub anchor: usize,
}

/// The policy net of a `net:PATH` / `net@T:PATH` bot name.
pub fn anchor_net(name: &str) -> Option<Arc<Net>> {
    let path = name.strip_prefix("net:").or_else(|| name.strip_prefix("net@").and_then(|r| r.split_once(':').map(|x| x.1)))?;
    Net::cached(path).ok()
}

/// Node budget for one exact endgame label (median solve is well under a
/// millisecond; this caps the rare deep one).
const ENDGAME_BUDGET: usize = 2_000_000;

struct Rec {
    state: Vec<u8>,
    cands: Candidates,
    label: usize,
    q: Option<Vec<f32>>,
    /// History tokens seen at this decision.
    prefix: usize,
}

/// One seat's records from a round, with its history tokens.
struct Block {
    seat: u8,
    tokens: Vec<u8>,
    n_tokens: usize,
    recs: Vec<Rec>,
    margin: f32,
}

/// Mean round margin for `view.seat`'s team after each option, over
/// `samples` guessed deals, with `rollout` playing all four seats. With
/// `infer` > 1 the guesses are weighted by the others' play so far
/// (`infer::sample_hands_weighted` with `infer` candidates per sample).
///
/// With `max_samples` > `samples` (uniform guesses only) sampling is
/// adaptive: after `samples` deals it keeps drawing, up to `max_samples`,
/// while the best option isn't clearly ahead of the runner-up (their paired
/// difference within two standard errors). Clear decisions stay cheap and
/// close ones get the samples.
pub fn pimc_q(
    view: &PlayerView,
    plays: &[Combo],
    cands: &Candidates,
    samples: usize,
    infer: usize,
    max_samples: usize,
    rollout: &mut [Box<dyn Bot>; 4],
    rng: &mut StdRng,
) -> Vec<f32> {
    let rows = pimc_rows(view, plays, cands, samples, infer, max_samples, rollout, rng);
    let m = rows.len().max(1) as f64;
    (0..cands.len()).map(|k| (rows.iter().map(|r| r[k]).sum::<f64>() / m) as f32).collect()
}

/// `pimc_q`'s per-deal margins: one row per guessed deal, one column per
/// option.
#[allow(clippy::too_many_arguments)]
fn pimc_rows(
    view: &PlayerView,
    plays: &[Combo],
    cands: &Candidates,
    samples: usize,
    infer: usize,
    max_samples: usize,
    rollout: &mut [Box<dyn Bot>; 4],
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let team = view.my_team();
    let n = cands.len();
    let run = |guess: [Vec<Card>; 4], rollout: &mut [Box<dyn Bot>; 4]| -> Vec<f64> {
        let base = GameState::from_view_with_history(view, guess);
        cands
            .options
            .iter()
            .map(|opt| {
                let mut g = base.clone();
                let action = match opt {
                    Some(i) => Action::Play { seat: view.seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                    None => Action::PassTurn { seat: view.seat },
                };
                g.apply(&action).expect("candidate rejected by engine");
                match play_from(&mut g, rollout, None, false) {
                    Ok(r) => (r.total(team) - r.total(1 - team)) as f64,
                    Err(_) => 0.0,
                }
            })
            .collect()
    };
    let mut rows: Vec<Vec<f64>> = vec![];
    if infer > 1 {
        for guess in infer::sample_hands_weighted(view, rng, samples, &infer::InferConfig { k: infer, ..Default::default() }) {
            rows.push(run(guess, rollout));
        }
    } else {
        let max = max_samples.max(samples);
        while rows.len() < max {
            rows.push(run(view.sample_hands(rng), rollout));
            if rows.len() >= samples.max(2) && rows.len() < max && clearly_best(&rows, n) {
                break;
            }
        }
    }
    rows
}

/// Whether option `b` beats option `a` by more than two standard errors,
/// paired by deal.
fn clearly_ahead(rows: &[Vec<f64>], b: usize, a: usize) -> bool {
    let m = rows.len() as f64;
    if m < 2.0 {
        return false;
    }
    let d: Vec<f64> = rows.iter().map(|r| r[b] - r[a]).collect();
    let dm = d.iter().sum::<f64>() / m;
    let var = d.iter().map(|x| (x - dm) * (x - dm)).sum::<f64>() / (m - 1.0);
    dm > 2.0 * (var / m).sqrt()
}

/// Running totals over anchored decisions, for judging how noisy the labels
/// are (`teacher_stats`).
#[derive(Default, Clone, Copy)]
pub struct TeacherStats {
    pub decisions: u64,
    /// The option with the best mean wasn't the net's choice.
    pub best_not_anchor: u64,
    /// ... and it was clearly ahead, so the teacher overrode the net.
    pub overrides: u64,
    /// Best-by-mean of the odd deals == best-by-mean of the even deals.
    pub split_agree: u64,
    /// Over (anchor, alternative) pairs: the mean paired difference squared,
    /// and its squared standard error (the noise's share of it).
    pub pairs: u64,
    pub sum_d2: f64,
    pub sum_se2: f64,
}

static TEACHER_STATS: std::sync::Mutex<TeacherStats> = std::sync::Mutex::new(TeacherStats {
    decisions: 0,
    best_not_anchor: 0,
    overrides: 0,
    split_agree: 0,
    pairs: 0,
    sum_d2: 0.0,
    sum_se2: 0.0,
});

impl TeacherStats {
    fn record(rows: &[Vec<f64>], mean: &[f64], best: usize, pick: usize) {
        let m = rows.len();
        if m < 4 {
            return;
        }
        let n = mean.len();
        let half = |par: usize| -> usize {
            let sums: Vec<f64> = (0..n).map(|k| rows.iter().skip(par).step_by(2).map(|r| r[k]).sum()).collect();
            (0..n).max_by(|&a, &b| sums[a].total_cmp(&sums[b])).unwrap()
        };
        let mut st = TEACHER_STATS.lock().unwrap();
        st.decisions += 1;
        st.best_not_anchor += (best != 0) as u64;
        st.overrides += (pick != 0) as u64;
        st.split_agree += (half(0) == half(1)) as u64;
        for k in 1..n {
            let d: Vec<f64> = rows.iter().map(|r| r[k] - r[0]).collect();
            let dm = d.iter().sum::<f64>() / m as f64;
            let var = d.iter().map(|x| (x - dm) * (x - dm)).sum::<f64>() / (m as f64 - 1.0);
            st.pairs += 1;
            st.sum_d2 += dm * dm;
            st.sum_se2 += var / m as f64;
        }
    }
}

/// A one-paragraph summary of the anchored teacher's noise so far in this
/// process, or `None` if it made no anchored decisions.
pub fn teacher_stats() -> Option<String> {
    let st = *TEACHER_STATS.lock().unwrap();
    if st.decisions == 0 {
        return None;
    }
    let n = st.decisions as f64;
    let (d2, se2) = (st.sum_d2 / st.pairs as f64, st.sum_se2 / st.pairs as f64);
    let signal = (d2 - se2).max(0.0);
    Some(format!(
        "teacher: {} anchored decisions; best-by-mean isn't the net's move in {:.1}%, overrides {:.1}%; \
         split-half agreement on the best move {:.1}%; Q difference to the net's move: rms {:.1} pts, \
         of which noise {:.1} (se) and signal {:.1}; reliability {:.2}",
        st.decisions,
        100.0 * st.best_not_anchor as f64 / n,
        100.0 * st.overrides as f64 / n,
        100.0 * st.split_agree as f64 / n,
        d2.sqrt(),
        se2.sqrt(),
        signal.sqrt(),
        signal / d2.max(1e-9),
    ))
}

/// The search settings shared by the selfplay labeler and `LabelerBot`.
pub struct Teacher {
    pub q_samples: usize,
    pub exact_endgame: bool,
    pub infer: usize,
    pub q_max: usize,
    /// Anchor on a policy net: search only its top `anchor_k` options and
    /// keep its choice unless another is clearly ahead (`clearly_ahead`).
    pub anchor: Option<Arc<Net>>,
    pub anchor_k: usize,
}

impl Teacher {
    /// The option to play and every option's Q (NaN where not searched).
    pub fn label(
        &self,
        view: &PlayerView,
        plays: &[Combo],
        cands: &Candidates,
        rollout: &mut [Box<dyn Bot>; 4],
        rng: &mut StdRng,
    ) -> (usize, Vec<f32>) {
        let argmax = |q: &[f32]| (0..q.len()).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap();
        if self.exact_endgame {
            if let Some(q) = exact_q(view, plays, cands, ENDGAME_BUDGET, rng) {
                let best = argmax(&q);
                explain_exact(view, plays, cands, &q, best);
                return (best, q);
            }
        }
        let Some(net) = &self.anchor else {
            let q = pimc_q(view, plays, cands, self.q_samples, self.infer, self.q_max, rollout, rng);
            return (argmax(&q), q);
        };
        let logits = net.option_logits(view, cands);
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(self.anchor_k.max(2));
        let sub = Candidates {
            options: order.iter().map(|&k| cands.options[k]).collect(),
            features: order.iter().flat_map(|&k| cands.features[k * ACTION_DIM..(k + 1) * ACTION_DIM].iter().copied()).collect(),
            group_of: vec![],
        };
        let rows = pimc_rows(view, plays, &sub, self.q_samples, self.infer, self.q_max, rollout, rng);
        let m = rows.len().max(1) as f64;
        let mean: Vec<f64> = (0..sub.len()).map(|k| rows.iter().map(|r| r[k]).sum::<f64>() / m).collect();
        let best = (0..sub.len()).max_by(|&a, &b| mean[a].total_cmp(&mean[b])).unwrap();
        let pick = if best != 0 && clearly_ahead(&rows, best, 0) { best } else { 0 };
        TeacherStats::record(&rows, &mean, best, pick);
        explain_search(view, plays, cands, &logits, &order, &rows, &mean, pick);
        let mut q = vec![f32::NAN; cands.len()];
        for (j, &k) in order.iter().enumerate() {
            q[k] = mean[j] as f32;
        }
        (order[pick], q)
    }
}

thread_local! {
    /// Decision notes, collected only while `explain_start` has switched
    /// them on (`tichu arena --log`), for the replay viewer.
    static EXPLAIN: RefCell<Option<Vec<serde_json::Value>>> = const { RefCell::new(None) };
}

/// Start collecting decision notes on this thread.
pub fn explain_start() {
    EXPLAIN.with(|e| *e.borrow_mut() = Some(Vec::new()));
}

/// Stop collecting decision notes on this thread.
pub fn explain_stop() {
    EXPLAIN.with(|e| *e.borrow_mut() = None);
}

/// The notes collected since the last call, as replay events
/// (`{"type": "llm", ...}`, so the viewer shows them as reasoning).
pub fn explain_take() -> Vec<serde_json::Value> {
    EXPLAIN.with(|e| e.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default())
}

fn explaining() -> bool {
    EXPLAIN.with(|e| e.borrow().is_some())
}

fn option_text(plays: &[Combo], o: Option<usize>) -> String {
    match o {
        None => "pass".into(),
        Some(i) => crate::llm::render::describe(&plays[i]),
    }
}

fn push_note(view: &PlayerView, menu: Vec<String>, answer: usize, reply: String) {
    let menu: Vec<String> = menu.into_iter().enumerate().map(|(k, l)| format!("{}: {l}", k + 1)).collect();
    let prompt = format!("Search bot decision for S{} ({} options shown)\nDECISION\n{}", view.seat, menu.len(), menu.join("\n"));
    let ev = serde_json::json!({
        "type": "llm", "seat": view.seat, "kind": "play",
        "answer": answer + 1, "prompt": prompt, "reply": reply,
    });
    EXPLAIN.with(|e| {
        if let Some(v) = e.borrow_mut().as_mut() {
            v.push(ev);
        }
    });
}

fn explain_exact(view: &PlayerView, plays: &[Combo], cands: &Candidates, q: &[f32], best: usize) {
    if !explaining() {
        return;
    }
    let mut order: Vec<usize> = (0..q.len()).collect();
    order.sort_by(|&a, &b| q[b].total_cmp(&q[a]));
    let menu: Vec<String> =
        order.iter().map(|&k| format!("{}  exact {:+.0}", option_text(plays, cands.options[k]), q[k])).collect();
    let pick = order.iter().position(|&k| k == best).unwrap();
    let alts = order.iter().filter(|&&k| q[k] < q[best]).count();
    let reply = format!(
        "**Two-player endgame, solved exactly.** Best play is worth **{:+.0}** to my team's round margin. {}",
        q[best],
        if alts == 0 { "Every option is equally good here.".to_string() } else { format!("{alts} of the {} options do worse.", q.len()) }
    );
    push_note(view, menu, pick, reply);
}

#[allow(clippy::too_many_arguments)]
fn explain_search(
    view: &PlayerView,
    plays: &[Combo],
    cands: &Candidates,
    logits: &[f32],
    order: &[usize],
    rows: &[Vec<f64>],
    mean: &[f64],
    pick: usize,
) {
    if !explaining() {
        return;
    }
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
    let prob = |k: usize| 100.0 * (logits[k] - mx).exp() / z;
    let m = rows.len().max(1) as f64;
    // Each option against the net's top choice, paired by guessed deal.
    let vs_top = |j: usize| -> (f64, f64) {
        let d: Vec<f64> = rows.iter().map(|r| r[j] - r[0]).collect();
        let dm = d.iter().sum::<f64>() / m;
        let var = if m > 1.0 { d.iter().map(|x| (x - dm) * (x - dm)).sum::<f64>() / (m - 1.0) } else { 0.0 };
        (dm, (var / m).sqrt())
    };
    let mut menu = Vec::new();
    for (j, &k) in order.iter().enumerate() {
        let tail = if j == 0 {
            String::new()
        } else {
            let (d, se) = vs_top(j);
            format!(", {d:+.1} ± {:.1} vs net's pick", 2.0 * se)
        };
        menu.push(format!("{}  net {:.0}%, search {:+.1}{tail}", option_text(plays, cands.options[k]), prob(k), mean[j]));
    }
    let rest: Vec<usize> = (0..cands.len()).filter(|k| !order.contains(k)).collect();
    for &k in rest.iter().take(5) {
        menu.push(format!("{}  net {:.0}% (not searched)", option_text(plays, cands.options[k]), prob(k)));
    }
    if rest.len() > 5 {
        menu.push(format!("... {} more options, net {:.0}% together", rest.len() - 5, rest[5..].iter().map(|&k| prob(k)).sum::<f32>()));
    }
    let top = order[0];
    let reply = if pick == 0 {
        let best = (0..mean.len()).max_by(|&a, &b| mean[a].total_cmp(&mean[b])).unwrap();
        let why = if best == 0 {
            "The search agrees".to_string()
        } else {
            let (d, se) = vs_top(best);
            format!(
                "The search rated **{}** {d:+.1} better, but that's within the noise (± {:.1}), so it keeps the net's move",
                option_text(plays, cands.options[order[best]]),
                2.0 * se
            )
        };
        format!(
            "The net's choice: **{}** ({:.0}%). {why}. {} guessed deals, {} options searched.",
            option_text(plays, cands.options[top]),
            prob(top),
            rows.len(),
            order.len()
        )
    } else {
        let (d, se) = vs_top(pick);
        format!(
            "**Search overrides the net.** The net preferred **{}** ({:.0}%), but **{}** came out {d:+.1} ± {:.1} points better over {} guessed deals.",
            option_text(plays, cands.options[top]),
            prob(top),
            option_text(plays, cands.options[order[pick]]),
            2.0 * se,
            rows.len()
        )
    };
    push_note(view, menu, pick, reply);
}

/// Whether the best option's lead over the runner-up, paired by deal, is
/// more than two standard errors.
fn clearly_best(rows: &[Vec<f64>], n: usize) -> bool {
    if n < 2 {
        return true;
    }
    let m = rows.len() as f64;
    let mean: Vec<f64> = (0..n).map(|k| rows.iter().map(|r| r[k]).sum::<f64>() / m).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mean[b].total_cmp(&mean[a]));
    let (b, c) = (order[0], order[1]);
    let d: Vec<f64> = rows.iter().map(|r| r[b] - r[c]).collect();
    let dm = d.iter().sum::<f64>() / m;
    let var = d.iter().map(|x| (x - dm) * (x - dm)).sum::<f64>() / (m - 1.0);
    dm > 2.0 * (var / m).sqrt()
}

/// Exact value of each option for `view.seat`'s team, as round margin, if
/// this is a two-player endgame (perfect information: the one other hand is
/// every unseen card) and the solver finishes within `budget` nodes.
pub fn exact_q(view: &PlayerView, plays: &[Combo], cands: &Candidates, budget: usize, rng: &mut StdRng) -> Option<Vec<f32>> {
    if view.players.iter().filter(|p| !p.is_out()).count() != 2 {
        return None;
    }
    let state = GameState::from_view_with_history(view, view.sample_hands(rng));
    let (seat, opts) = endgame::solve_options(&state, endgame::Node::Engine, budget)?;
    if seat != view.seat {
        return None;
    }
    let sign = if view.my_team() == 0 { 1.0 } else { -1.0 };
    cands
        .options
        .iter()
        .map(|o| {
            opts.iter()
                .find(|ov| match (&ov.decision, o) {
                    (endgame::Decision::Pass, None) => true,
                    (endgame::Decision::Play(c), Some(i)) => {
                        CardSet::from_cards(&c.cards) == CardSet::from_cards(&plays[*i].cards) && c.rank == plays[*i].rank
                    }
                    _ => false,
                })
                .map(|v| sign * v.value as f32)
        })
        .collect()
}

struct Recorder {
    inner: Box<dyn Bot>,
    rng: StdRng,
    teacher: Teacher,
    rollout: [Box<dyn Bot>; 4],
    explore: f64,
    recs: Vec<Rec>,
    out: Rc<RefCell<Vec<Block>>>,
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
        let (label, q) = if self.teacher.q_samples > 0 {
            let (best, q) = self.teacher.label(view, plays, &cands, &mut self.rollout, &mut self.rng);
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
        let mut state = Vec::with_capacity(STATE_DIM);
        encode_state(view, &mut state);
        let prefix = encode_history(view, &mut vec![]);
        self.recs.push(Rec { state, cands, label, q, prefix });
        choice
    }

    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        self.inner.round_over(view, result);
        if self.recs.is_empty() {
            return;
        }
        let team = view.my_team();
        let margin = (result.total(team) - result.total(1 - team)) as f32;
        let mut tokens = vec![];
        let n_tokens = encode_history(view, &mut tokens);
        assert!(self.recs.iter().all(|r| r.prefix <= n_tokens), "history isn't a prefix of the round's");
        let recs = std::mem::take(&mut self.recs);
        self.out.borrow_mut().push(Block { seat: view.seat as u8, tokens, n_tokens, recs, margin });
    }
}

/// The `--q` labeler as a bot (`labelN[/FLAGS]:ROLLOUT`), so the arena can
/// play the teacher itself: it plays the best of `pimc_q` (or `exact_q`)
/// exactly as `selfplay --q N --rollout ROLLOUT` labels it, and leaves every
/// other decision to h2 (`selfplay --bot h2`). Flags: `e` = `--exact-endgame`,
/// `iK` = `--infer K`, `aM` = `--q-max M` (adaptive samples), `pK` = `--anchor K` (needs a
/// `net:` rollout).
pub struct LabelerBot {
    base: crate::bots::H2Bot,
    rng: StdRng,
    teacher: Teacher,
    rollout: [Box<dyn Bot>; 4],
    name: String,
}

impl LabelerBot {
    pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
        let (head, rollout_name) = name.split_once(':')?;
        let (head, flags) = head.split_once('/').unwrap_or((head, ""));
        let q_samples: usize = head.strip_prefix("label")?.parse().ok().filter(|&n| n > 0)?;
        let (mut exact_endgame, mut infer, mut q_max, mut anchor_k) = (false, 0, 0, 0);
        let mut chars = flags.chars().peekable();
        while let Some(f) = chars.next() {
            match f {
                'e' => exact_endgame = true,
                'i' | 'a' | 'p' => {
                    let mut k = String::new();
                    while let Some(d) = chars.peek().filter(|c| c.is_ascii_digit()) {
                        k.push(*d);
                        chars.next();
                    }
                    let k: usize = k.parse().ok()?;
                    match f {
                        'i' => infer = k,
                        'a' => q_max = k,
                        _ => anchor_k = k,
                    }
                }
                _ => return None,
            }
        }
        let mut rollout_bots = Vec::with_capacity(4);
        for k in 0..4 {
            rollout_bots.push(bots::by_name(rollout_name, seed.wrapping_mul(4).wrapping_add(k))?);
        }
        let rollout: [Box<dyn Bot>; 4] = rollout_bots.try_into().ok()?;
        let anchor = match anchor_k {
            0 => None,
            1 => return None,
            _ => Some(anchor_net(rollout_name)?),
        };
        Some(Box::new(LabelerBot {
            base: crate::bots::H2Bot::new(),
            rng: StdRng::seed_from_u64(seed ^ 0xABCD),
            teacher: Teacher { q_samples, exact_endgame, infer, q_max, anchor, anchor_k },
            rollout,
            name: name.to_string(),
        }))
    }
}

impl Bot for LabelerBot {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        self.base.grand_tichu(v)
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        self.base.small_tichu(v)
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        self.base.pass_cards(v)
    }
    fn bomb(&mut self, v: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.base.bomb(v, bombs)
    }
    fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
        self.base.wish(v)
    }
    fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
        self.base.dragon_recipient(v)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let cands = Candidates::new(view, plays, can_pass);
        if cands.len() < 2 {
            return self.base.play(view, plays, can_pass);
        }
        let (best, _) = self.teacher.label(view, plays, &cands, &mut self.rollout, &mut self.rng);
        cands.options[best]
    }
}

fn write_header(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"TCHR")?;
    for x in [FORMAT_VERSION, FEATURE_VERSION, STATE_DIM as u32, ACTION_DIM as u32, TOKEN_DIM as u32] {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn write_block(buf: &mut Vec<u8>, round: u32, b: &Block) {
    buf.extend_from_slice(&round.to_le_bytes());
    buf.push(b.seat);
    buf.extend_from_slice(&(b.n_tokens as u16).to_le_bytes());
    buf.extend_from_slice(&(b.recs.len() as u32).to_le_bytes());
    buf.extend_from_slice(&b.tokens);
    for rec in &b.recs {
        buf.push(rec.q.is_some() as u8);
        buf.extend_from_slice(&(rec.cands.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(rec.label as u16).to_le_bytes());
        buf.extend_from_slice(&(rec.prefix as u16).to_le_bytes());
        buf.extend_from_slice(&b.margin.to_le_bytes());
        buf.extend_from_slice(&rec.state);
        buf.extend_from_slice(&rec.cands.features);
        if let Some(q) = &rec.q {
            for x in q {
                buf.extend_from_slice(&x.to_le_bytes());
            }
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
                    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| -> Box<dyn Bot> {
                        let sd = seed.wrapping_mul(31).wrapping_add(s);
                        Box::new(Recorder {
                            inner: bots::by_name(&cfg.bot, sd).expect("unknown bot"),
                            rng: StdRng::seed_from_u64(sd ^ 0xABCD),
                            teacher: Teacher {
                                q_samples: cfg.q_samples,
                                exact_endgame: cfg.exact_endgame,
                                infer: cfg.infer,
                                q_max: cfg.q_max,
                                anchor: (cfg.anchor > 0).then(|| anchor_net(&cfg.rollout).expect("--anchor needs a net rollout")),
                                anchor_k: cfg.anchor,
                            },
                            rollout: [0, 1, 2, 3].map(|k| bots::by_name(&cfg.rollout, sd * 4 + k).expect("unknown bot")),
                            explore: cfg.explore,
                            recs: vec![],
                            out: out.clone(),
                        })
                    });
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut state = GameState::new(Settings { clockwise: rng.gen_bool(0.5), ..Settings::default() });
                    state.start_round(shuffled_deck(&mut rng));
                    let (mut buf, mut n) = (vec![], 0);
                    if play_dealt_round(&mut state, &mut bots, None).is_ok() {
                        for b in out.borrow().iter() {
                            write_block(&mut buf, i as u32, b);
                            n += b.recs.len();
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
