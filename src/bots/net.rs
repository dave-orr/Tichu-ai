//! Learned bots: a small MLP trained by `train/` on self-play records
//! (`selfplay`), run here with no dependencies.
//!
//! The model has a shared trunk over the state features, a value head (the
//! expected round margin for my team) and a policy head that scores each
//! candidate from `trunk ⊕ action features`. Everything except card play
//! (calls, passing, out-of-turn bombs, wishes, the Dragon) stays with the
//! h2 rule bot.
//!
//! - `net:PATH` plays the policy's top choice.
//! - `vpimcN:PATH` is PIMC whose rollouts stop at my next turn and score
//!   that position with the value head, instead of playing to the end.
//!
//! With a sequence encoder (`"seq"` in the header), a one-layer GRU reads
//! the round's history tokens (`features::encode_history`) and its last
//! hidden state is appended to the state features at the trunk's input.
//!
//! Model file (`.tnet`, written by `train/train.py`): `b"TNET"`, u32 header
//! length, a JSON header (dims, feature version, layer shapes), then each
//! layer's weight (`[out][in]`) and bias as little-endian f32, in the order
//! trunk, value, policy; then, with a GRU, PyTorch's `weight_ih_l0`
//! (`[3H][TOKEN_DIM]`, gates r, z, n), `weight_hh_l0` (`[3H][H]`),
//! `bias_ih_l0` and `bias_hh_l0`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;

use crate::bot::Bot;
use crate::bots::H2Bot;
use crate::engine::{Action, GameState};
use crate::features::*;
use crate::sim::{play_from, SimError};
use crate::types::*;
use crate::view::PlayerView;

/// A dense layer, stored transposed (`w[i * out + o]`) so sparse inputs can
/// skip their zero entries.
struct Layer {
    inp: usize,
    out: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Layer {
    fn forward(&self, x: &[f32], y: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), self.inp);
        y.clear();
        y.extend_from_slice(&self.b);
        add_rows(y, &self.w, self.out, 0, x.iter().copied());
    }
}

/// `y += Σ x_i · w[from + i]` over the rows of a transposed weight matrix.
fn add_rows(y: &mut [f32], w: &[f32], out: usize, from: usize, x: impl Iterator<Item = f32>) {
    for (i, xi) in x.enumerate() {
        if xi != 0.0 {
            let row = &w[(from + i) * out..(from + i + 1) * out];
            for (yo, wo) in y.iter_mut().zip(row) {
                *yo += xi * wo;
            }
        }
    }
}

fn mlp(layers: &[Layer], x: &[f32]) -> Vec<f32> {
    let mut cur = x.to_vec();
    let mut next = vec![];
    for (k, l) in layers.iter().enumerate() {
        l.forward(&cur, &mut next);
        if k + 1 < layers.len() {
            next.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        std::mem::swap(&mut cur, &mut next);
    }
    cur
}

/// A one-layer GRU over history tokens (PyTorch's `nn.GRU` equations).
struct Gru {
    hidden: usize,
    /// Transposed from PyTorch's `[3H][in]`: `w_ih[i * 3H + g]`,
    /// `w_hh[j * 3H + g]` (gates r, z, n), as in `Layer`.
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `[rows][cols]` row-major → `[cols][rows]`.
fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; w.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

impl Gru {
    /// One step: `h` ← GRU(token, h).
    fn step(&self, token: &[u8], h: &mut [f32]) {
        let hd = self.hidden;
        let mut gi = self.b_ih.clone();
        add_rows(&mut gi, &self.w_ih, 3 * hd, 0, token.iter().map(|&t| t as f32 / 255.0));
        let mut gh = self.b_hh.clone();
        add_rows(&mut gh, &self.w_hh, 3 * hd, 0, h.iter().copied());
        for j in 0..hd {
            let r = sigmoid(gi[j] + gh[j]);
            let z = sigmoid(gi[hd + j] + gh[hd + j]);
            let n = (gi[2 * hd + j] + r * gh[2 * hd + j]).tanh();
            h[j] = (1.0 - z) * n + z * h[j];
        }
    }
}

/// Hidden states after each token of the last sequence seen, so a bot that
/// is asked about a growing history (or the same history again, as in
/// rollouts from one position) only runs the new tokens.
#[derive(Default)]
pub struct SeqCache {
    tokens: Vec<u8>,
    states: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct SeqHeader {
    token_dim: usize,
    hidden: usize,
}

#[derive(Deserialize)]
struct Header {
    feature_version: u32,
    state_dim: usize,
    action_dim: usize,
    seq: Option<SeqHeader>,
    /// `[out, in]` per layer.
    trunk: Vec<[usize; 2]>,
    value: Vec<[usize; 2]>,
    policy: Vec<[usize; 2]>,
    /// The value head predicts margin / value_scale.
    value_scale: f32,
    /// "state": the value layers are a separate net over the state features
    /// (trained apart from the trunk and early-stopped); otherwise they
    /// read the trunk's output.
    #[serde(default)]
    value_input: Option<String>,
    /// One position with the trainer's outputs, recomputed on load.
    check: Option<Check>,
}

#[derive(Deserialize)]
struct Check {
    state: Vec<u8>,
    actions: Vec<u8>,
    #[serde(default)]
    tokens: Vec<u8>,
    value: f32,
    logits: Vec<f32>,
}

pub struct Net {
    trunk: Vec<Layer>,
    value: Vec<Layer>,
    policy: Vec<Layer>,
    gru: Option<Gru>,
    value_scale: f32,
    hidden: usize,
    value_on_state: bool,
}

pub fn to_inputs(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| b as f32 / 255.0).collect()
}

impl Net {
    pub fn load(path: &str) -> Result<Net, String> {
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if data.len() < 8 || &data[0..4] != b"TNET" {
            return Err(format!("{path}: not a .tnet model"));
        }
        let hlen = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let header = data.get(8..8 + hlen).ok_or_else(|| format!("{path}: truncated header"))?;
        let header: Header = serde_json::from_slice(header).map_err(|e| format!("{path}: bad header: {e}"))?;
        if header.feature_version != FEATURE_VERSION || header.state_dim != STATE_DIM || header.action_dim != ACTION_DIM {
            return Err(format!(
                "{path}: trained on feature version {} ({}x{}), this build has {FEATURE_VERSION} ({STATE_DIM}x{ACTION_DIM})",
                header.feature_version, header.state_dim, header.action_dim
            ));
        }
        let mut floats = data[8 + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        let mut read = |shapes: &[[usize; 2]]| -> Result<Vec<Layer>, String> {
            shapes
                .iter()
                .map(|&[out, inp]| {
                    let w: Vec<f32> = floats.by_ref().take(out * inp).collect();
                    let b: Vec<f32> = floats.by_ref().take(out).collect();
                    if w.len() != out * inp || b.len() != out {
                        return Err(format!("{path}: truncated weights"));
                    }
                    let mut wt = vec![0.0; out * inp];
                    for o in 0..out {
                        for i in 0..inp {
                            wt[i * out + o] = w[o * inp + i];
                        }
                    }
                    Ok(Layer { inp, out, w: wt, b })
                })
                .collect()
        };
        let trunk = read(&header.trunk)?;
        let value = read(&header.value)?;
        let policy = read(&header.policy)?;
        let gru = match &header.seq {
            None => None,
            Some(s) => {
                if s.token_dim != TOKEN_DIM {
                    return Err(format!("{path}: tokens of {} bytes, this build has {TOKEN_DIM}", s.token_dim));
                }
                let h = s.hidden;
                let mut take = |n: usize| -> Result<Vec<f32>, String> {
                    let v: Vec<f32> = floats.by_ref().take(n).collect();
                    if v.len() == n { Ok(v) } else { Err(format!("{path}: truncated GRU weights")) }
                };
                let (w_ih, w_hh) = (take(3 * h * TOKEN_DIM)?, take(3 * h * h)?);
                Some(Gru {
                    hidden: h,
                    w_ih: transpose(&w_ih, 3 * h, TOKEN_DIM),
                    w_hh: transpose(&w_hh, 3 * h, h),
                    b_ih: take(3 * h)?,
                    b_hh: take(3 * h)?,
                })
            }
        };
        let value_on_state = match header.value_input.as_deref() {
            None | Some("trunk") => false,
            Some("state") => true,
            Some(x) => return Err(format!("{path}: unknown value_input {x:?}")),
        };
        let seq_dim = gru.as_ref().map_or(0, |g| g.hidden);
        let hidden = trunk.last().map_or(STATE_DIM + seq_dim, |l| l.out);
        let chain_ok = |ls: &[Layer], first: usize| {
            ls.first().is_some_and(|l| l.inp == first)
                && ls.windows(2).all(|w| w[0].out == w[1].inp)
                && ls.last().is_some_and(|l| l.out == 1)
        };
        if trunk.first().is_some_and(|l| l.inp != STATE_DIM + seq_dim)
            || trunk.windows(2).any(|w| w[0].out != w[1].inp)
            || !chain_ok(&value, if value_on_state { STATE_DIM } else { hidden })
            || !chain_ok(&policy, hidden + ACTION_DIM)
        {
            return Err(format!("{path}: layer shapes don't chain"));
        }
        let net = Net { trunk, value, policy, gru, value_scale: header.value_scale, hidden, value_on_state };
        if let Some(c) = &header.check {
            let h = net.trunk(&c.state, &net.encode_seq(&c.tokens));
            let (v, l) = (net.value(&c.state, &h), net.logits(&h, &c.actions));
            let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + b.abs());
            if !close(v, c.value) || l.len() != c.logits.len() || l.iter().zip(&c.logits).any(|(a, b)| !close(*a, *b)) {
                return Err(format!("{path}: outputs differ from the trainer's (value {v} vs {})", c.value));
            }
        }
        Ok(net)
    }

    /// Loaded once per path and shared (bots are built per seat per deal).
    pub fn cached(path: &str) -> Result<Arc<Net>, String> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<Net>>>> = OnceLock::new();
        let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
        if let Some(n) = cache.get(path) {
            return Ok(n.clone());
        }
        let n = Arc::new(Net::load(path)?);
        cache.insert(path.to_string(), n.clone());
        Ok(n)
    }

    /// The GRU's last hidden state over `tokens` (zeros for none; empty
    /// without a GRU).
    pub fn encode_seq(&self, tokens: &[u8]) -> Vec<f32> {
        let Some(g) = &self.gru else { return vec![] };
        let mut h = vec![0.0; g.hidden];
        tokens.chunks_exact(TOKEN_DIM).for_each(|t| g.step(t, &mut h));
        h
    }

    /// `encode_seq`, reusing `cache` from earlier calls where the tokens
    /// share a prefix.
    pub fn encode_seq_cached(&self, tokens: &[u8], cache: &mut SeqCache) -> Vec<f32> {
        let Some(g) = &self.gru else { return vec![] };
        let common = cache
            .tokens
            .chunks_exact(TOKEN_DIM)
            .zip(tokens.chunks_exact(TOKEN_DIM))
            .take_while(|(a, b)| a == b)
            .count();
        cache.tokens.truncate(common * TOKEN_DIM);
        cache.states.truncate(common);
        let mut h = cache.states.last().cloned().unwrap_or_else(|| vec![0.0; g.hidden]);
        for t in tokens[common * TOKEN_DIM..].chunks_exact(TOKEN_DIM) {
            g.step(t, &mut h);
            cache.tokens.extend_from_slice(t);
            cache.states.push(h.clone());
        }
        h
    }

    /// The trunk over the state features and the GRU's state (`encode_seq`).
    pub fn trunk(&self, state: &[u8], seq: &[f32]) -> Vec<f32> {
        let mut x = to_inputs(state);
        x.extend_from_slice(seq);
        let mut h = mlp(&self.trunk, &x);
        if !self.trunk.is_empty() {
            h.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        h
    }

    /// Expected round margin (points) for my team, from the state features
    /// and the trunk's output for them.
    pub fn value(&self, state: &[u8], h: &[f32]) -> f32 {
        let v = if self.value_on_state { mlp(&self.value, &to_inputs(state)) } else { mlp(&self.value, h) };
        v[0] * self.value_scale
    }

    /// One logit per candidate (`actions` holds `ACTION_DIM` bytes each).
    pub fn logits(&self, h: &[f32], actions: &[u8]) -> Vec<f32> {
        let (first, rest) = self.policy.split_first().unwrap();
        // The trunk half of the first layer is the same for every candidate.
        let mut base = first.b.clone();
        add_rows(&mut base, &first.w, first.out, 0, h.iter().copied());
        let mut y = vec![];
        actions
            .chunks_exact(ACTION_DIM)
            .map(|a| {
                y.clear();
                y.extend_from_slice(&base);
                add_rows(&mut y, &first.w, first.out, self.hidden, a.iter().map(|&b| b as f32 / 255.0));
                y.iter_mut().for_each(|v| *v = v.max(0.0));
                mlp(rest, &y)[0]
            })
            .collect()
    }

    pub fn value_of(&self, view: &PlayerView) -> f32 {
        let walk = HistoryWalk::of(view);
        let mut s = Vec::with_capacity(STATE_DIM);
        encode_state_with(view, &walk, &mut s);
        if self.value_on_state {
            return self.value(&s, &[]);
        }
        self.value(&s, &self.trunk(&s, &self.encode_seq(walk.tokens())))
    }
}

/// Plays the policy's top choice.
pub struct NetBot {
    net: Arc<Net>,
    base: H2Bot,
    memo: NetMemo,
}

/// What a net bot carries from one decision to the next: its walk of the
/// log and the GRU's states over it. Rollouts from one position keep asking
/// about histories that share a long prefix.
#[derive(Default)]
pub struct NetMemo {
    walk: HistoryWalk,
    seq: SeqCache,
}

impl NetBot {
    pub fn new(net: Arc<Net>) -> Self {
        NetBot { net, base: H2Bot::new(), memo: NetMemo::default() }
    }

    pub fn choose(net: &Net, view: &PlayerView, plays: &[Combo], can_pass: bool, memo: &mut NetMemo) -> Option<usize> {
        let cands = Candidates::new(view, plays, can_pass);
        if cands.len() == 1 {
            return cands.options[0];
        }
        memo.walk.update(view);
        let mut s = Vec::with_capacity(STATE_DIM);
        encode_state_with(view, &memo.walk, &mut s);
        let seq = net.encode_seq_cached(memo.walk.tokens(), &mut memo.seq);
        let logits = net.logits(&net.trunk(&s, &seq), &cands.features);
        let best = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
        cands.options[best]
    }
}

macro_rules! rule_bot_rest {
    () => {
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
    };
}

impl Bot for NetBot {
    fn name(&self) -> String {
        "net".into()
    }
    rule_bot_rest!();
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        Self::choose(&self.net, view, plays, can_pass, &mut self.memo)
    }
}

/// Rollout stand-in for the searching seat: plays like h2 until
/// its next turn, then aborts the rollout by returning an out-of-range play,
/// which `play_from` rejects, leaving the state at that turn.
struct StopAtMyTurn {
    base: H2Bot,
}

impl Bot for StopAtMyTurn {
    fn name(&self) -> String {
        "stop".into()
    }
    rule_bot_rest!();
    fn play(&mut self, _view: &PlayerView, _plays: &[Combo], _can_pass: bool) -> Option<usize> {
        Some(usize::MAX)
    }
}

/// PIMC with rollouts cut at my next turn and scored by the value head.
pub struct ValuePimcBot {
    net: Arc<Net>,
    samples: usize,
    rng: StdRng,
    base: H2Bot,
}

impl ValuePimcBot {
    pub fn new(net: Arc<Net>, samples: usize, seed: u64) -> Self {
        ValuePimcBot { net, samples, rng: StdRng::seed_from_u64(seed), base: H2Bot::new() }
    }
}

impl Bot for ValuePimcBot {
    fn name(&self) -> String {
        format!("vpimc{}", self.samples)
    }
    rule_bot_rest!();
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let cands = Candidates::new(view, plays, can_pass);
        if cands.len() == 1 {
            return cands.options[0];
        }
        let team = view.my_team();
        let mut totals = vec![0.0f64; cands.len()];
        for _ in 0..self.samples {
            let base = GameState::from_view_with_history(view, view.sample_hands(&mut self.rng));
            for (k, opt) in cands.options.iter().enumerate() {
                let mut g = base.clone();
                let action = match opt {
                    Some(i) => Action::Play { seat: view.seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                    None => Action::PassTurn { seat: view.seat },
                };
                g.apply(&action).expect("candidate rejected by engine");
                let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(H2Bot::new()) as Box<dyn Bot>);
                bots[view.seat] = Box::new(StopAtMyTurn { base: H2Bot::new() });
                let v = match play_from(&mut g, &mut bots, None, false) {
                    Ok(r) => (r.total(team) - r.total(1 - team)) as f32,
                    Err(SimError::BadChoice { seat, .. }) if seat == view.seat => {
                        self.net.value_of(&g.view(view.seat))
                    }
                    Err(e) => panic!("rollout failed: {e}"),
                };
                totals[k] += v as f64;
            }
        }
        let best = (0..totals.len()).max_by(|&a, &b| totals[a].total_cmp(&totals[b])).unwrap();
        cands.options[best]
    }
}

/// `net:PATH` or `vpimcN:PATH`.
pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    let load = |p: &str| match Net::cached(p) {
        Ok(n) => Some(n),
        Err(e) => {
            eprintln!("{e}");
            None
        }
    };
    if let Some(path) = name.strip_prefix("net:") {
        return Some(Box::new(NetBot::new(load(path)?)));
    }
    let (head, path) = name.split_once(':')?;
    let n: usize = head.strip_prefix("vpimc")?.parse().ok()?;
    Some(Box::new(ValuePimcBot::new(load(path)?, n, seed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::shuffled_deck;
    use crate::sim::play_dealt_round;
    use rand::Rng;

    /// A random model in the .tnet format: loads, plays whole rounds, and
    /// gives the same outputs as a direct dense forward pass.
    #[test]
    fn random_model_plays() {
        let mut rng = StdRng::seed_from_u64(3);
        let (h, k) = (16, 8);
        let shapes = [vec![[h, STATE_DIM], [h, h]], vec![[k, h], [1, k]], vec![[k, h + ACTION_DIM], [1, k]]];
        let header = serde_json::json!({
            "feature_version": FEATURE_VERSION, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "trunk": shapes[0], "value": shapes[1], "policy": shapes[2], "value_scale": 100.0,
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let mut file = b"TNET".to_vec();
        file.extend_from_slice(&(hb.len() as u32).to_le_bytes());
        file.extend_from_slice(&hb);
        let mut dense = vec![];
        for group in &shapes {
            let mut ls = vec![];
            for &[out, inp] in group {
                let w: Vec<f32> = (0..out * inp + out).map(|_| rng.gen_range(-0.3..0.3)).collect();
                w.iter().for_each(|x| file.extend_from_slice(&x.to_le_bytes()));
                ls.push((out, inp, w));
            }
            dense.push(ls);
        }
        let path = std::env::temp_dir().join(format!("tichu-test-{}.tnet", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let net = Net::cached(path.to_str().unwrap()).unwrap();

        // Reference forward pass straight from the [out][in] weights.
        let fwd = |ls: &[(usize, usize, Vec<f32>)], x: &[f32], relu_last: bool| {
            let mut x = x.to_vec();
            for (n, (out, inp, w)) in ls.iter().enumerate() {
                x = (0..*out)
                    .map(|o| {
                        let y = w[out * inp + o] + (0..*inp).map(|i| w[o * inp + i] * x[i]).sum::<f32>();
                        if n + 1 < ls.len() || relu_last { y.max(0.0) } else { y }
                    })
                    .collect();
            }
            x
        };
        let mut state = GameState::new(Settings::default());
        state.start_round(shuffled_deck(&mut rng));
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(NetBot::new(net.clone())) as Box<dyn Bot>);
        play_dealt_round(&mut state, &mut bots, None).unwrap();

        let v = state.view(0);
        let mut s = vec![];
        encode_state(&v, &mut s);
        let mut a = vec![];
        encode_action(&v, None, &mut a);
        let x = to_inputs(&s);
        let hid = fwd(&dense[0], &x, true);
        let want_v = fwd(&dense[1], &hid, false)[0] * 100.0;
        let mut ha = hid.clone();
        ha.extend(to_inputs(&a));
        let want_l = fwd(&dense[2], &ha, false)[0];
        let got_h = net.trunk(&s, &[]);
        assert!((net.value(&s, &got_h) - want_v).abs() < 1e-3);
        assert!((net.logits(&got_h, &a)[0] - want_l).abs() < 1e-4);
        std::fs::remove_file(path).ok();
    }

    /// A random model with a GRU loads and plays, and the cached sequence
    /// encoding matches a fresh one as histories grow and branch.
    #[test]
    fn random_gru_model_plays_and_caches() {
        let mut rng = StdRng::seed_from_u64(4);
        let (h, k, g) = (16, 8, 12);
        let shapes = [vec![[h, STATE_DIM + g]], vec![[k, h], [1, k]], vec![[k, h + ACTION_DIM], [1, k]]];
        let header = serde_json::json!({
            "feature_version": FEATURE_VERSION, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "trunk": shapes[0], "value": shapes[1], "policy": shapes[2], "value_scale": 100.0,
            "seq": {"token_dim": TOKEN_DIM, "hidden": g},
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let mut file = b"TNET".to_vec();
        file.extend_from_slice(&(hb.len() as u32).to_le_bytes());
        file.extend_from_slice(&hb);
        let n_layers: usize = shapes.iter().flatten().map(|&[o, i]| o * i + o).sum();
        let n_gru = 3 * g * TOKEN_DIM + 3 * g * g + 6 * g;
        for _ in 0..n_layers + n_gru {
            file.extend_from_slice(&rng.gen_range(-0.3f32..0.3).to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!("tichu-test-gru-{}.tnet", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let net = Net::cached(path.to_str().unwrap()).unwrap();

        let mut state = GameState::new(Settings::default());
        state.start_round(shuffled_deck(&mut rng));
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(NetBot::new(net.clone())) as Box<dyn Bot>);
        play_dealt_round(&mut state, &mut bots, None).unwrap();

        let mut tokens = vec![];
        let n = encode_history(&state.view(1), &mut tokens);
        assert!(n > 10);
        let mut cache = SeqCache::default();
        for cut in [n / 2, n, n / 3, n, 0, n - 1] {
            let mut t = tokens[..cut * TOKEN_DIM].to_vec();
            if cut == n - 1 {
                // A branch: same prefix, different last token.
                t.extend(std::iter::repeat(255).take(TOKEN_DIM));
            }
            let fresh = net.encode_seq(&t);
            let cached = net.encode_seq_cached(&t, &mut cache);
            assert_eq!(fresh, cached, "cut {cut}");
        }
        assert!(net.encode_seq(&tokens).iter().any(|&x| x != 0.0));
        std::fs::remove_file(path).ok();
    }
}
