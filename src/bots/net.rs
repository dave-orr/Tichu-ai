//! Learned bots: a small MLP trained by `train/` on self-play records
//! (`selfplay`), run here with no dependencies.
//!
//! The model has a shared trunk over the state features (plus, optionally,
//! a GRU's summary of the round's history tokens from `history`), a value
//! head (the expected round margin for my team) and a policy head that
//! scores each candidate from `trunk ⊕ action features`. Everything except card play
//! (calls, passing, out-of-turn bombs, wishes, the Dragon) stays with the
//! h2 rule bot.
//!
//! - `net:PATH` plays the policy's top choice.
//! - `vpimcN:PATH` is PIMC whose rollouts stop at my next turn and score
//!   that position with the value head, instead of playing to the end.
//!
//! Model file (`.tnet`, written by `train/export.py`): `b"TNET"`, u32 header
//! length, a JSON header (dims, feature version, layer shapes), then each
//! layer's weight (`[out][in]`) and bias as little-endian f32, in the order
//! trunk, value, policy, then (with a GRU) PyTorch's `weight_ih`, `bias_ih`,
//! `weight_hh`, `bias_hh`, gates in the order r, z, n.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;

use crate::bot::Bot;
use crate::bots::H2Bot;
use crate::engine::{Action, GameState};
use crate::features::*;
use crate::history::{self, TOKEN_DIM};
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

/// One-layer GRU, PyTorch's equations:
/// r = σ(W_ir x + b_ir + W_hr h + b_hr), z likewise,
/// n = tanh(W_in x + b_in + r ∘ (W_hn h + b_hn)), h' = (1 − z) ∘ n + z ∘ h.
struct Gru {
    hidden: usize,
    /// `[3H, D]` and `[3H, H]`, stored transposed like `Layer`.
    ih: Layer,
    hh: Layer,
}

impl Gru {
    fn step(&self, h: &mut [f32], token: &[u8]) {
        let n = self.hidden;
        let mut gi = self.ih.b.clone();
        add_rows(&mut gi, &self.ih.w, 3 * n, 0, token.iter().map(|&b| b as f32 / 255.0));
        let mut gh = vec![];
        self.hh.forward(h, &mut gh);
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        for k in 0..n {
            let r = sig(gi[k] + gh[k]);
            let z = sig(gi[n + k] + gh[n + k]);
            let c = (gi[2 * n + k] + r * gh[2 * n + k]).tanh();
            h[k] = (1.0 - z) * c + z * h[k];
        }
    }
}

/// Hidden states after each history token, so a later decision of the same
/// seat (or a rollout from the same position) only runs the new tokens.
#[derive(Default)]
pub struct HistoryCache {
    tokens: Vec<u8>,
    /// `states[t]` is the hidden state after `t` tokens (`states[0]` = 0).
    states: Vec<Vec<f32>>,
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

#[derive(Deserialize)]
struct Header {
    feature_version: u32,
    state_dim: usize,
    action_dim: usize,
    /// `[out, in]` per layer.
    trunk: Vec<[usize; 2]>,
    value: Vec<[usize; 2]>,
    policy: Vec<[usize; 2]>,
    /// The value head predicts margin / value_scale.
    value_scale: f32,
    /// One position with the trainer's outputs, recomputed on load.
    check: Option<Check>,
    /// `[hidden, token dim]` of the history GRU, if the model has one.
    #[serde(default)]
    gru: Option<[usize; 2]>,
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
    value_scale: f32,
    hidden: usize,
    gru: Option<Gru>,
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
        let gru = match header.gru {
            None => None,
            Some([h, d]) => {
                if d != TOKEN_DIM {
                    return Err(format!("{path}: history tokens are {d} bytes, this build has {TOKEN_DIM}"));
                }
                let mut l = read(&[[3 * h, d], [3 * h, h]])?;
                let (hh, ih) = (l.pop().unwrap(), l.pop().unwrap());
                Some(Gru { hidden: h, ih, hh })
            }
        };
        let gru_dim = gru.as_ref().map_or(0, |g| g.hidden);
        let hidden = trunk.last().map_or(STATE_DIM, |l| l.out);
        let chain_ok = |ls: &[Layer], first: usize| {
            ls.first().is_some_and(|l| l.inp == first)
                && ls.windows(2).all(|w| w[0].out == w[1].inp)
                && ls.last().is_some_and(|l| l.out == 1)
        };
        if trunk.first().is_some_and(|l| l.inp != STATE_DIM + gru_dim)
            || trunk.windows(2).any(|w| w[0].out != w[1].inp)
            || !chain_ok(&value, hidden)
            || !chain_ok(&policy, hidden + ACTION_DIM)
        {
            return Err(format!("{path}: layer shapes don't chain"));
        }
        let net = Net { trunk, value, policy, value_scale: header.value_scale, hidden, gru };
        if let Some(c) = &header.check {
            let h = net.trunk(&c.state, &net.history_state(&c.tokens, &mut HistoryCache::default()));
            let (v, l) = (net.value(&h), net.logits(&h, &c.actions));
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

    /// The GRU's state after `tokens` (empty without a GRU), reusing and
    /// updating `cache` for their longest prefix already run.
    pub fn history_state(&self, tokens: &[u8], cache: &mut HistoryCache) -> Vec<f32> {
        let Some(g) = &self.gru else { return vec![] };
        let common = cache.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count() / TOKEN_DIM;
        let common = common.min(cache.states.len().saturating_sub(1));
        cache.tokens.truncate(common * TOKEN_DIM);
        cache.states.truncate(common + 1);
        if cache.states.is_empty() {
            cache.states.push(vec![0.0; g.hidden]);
        }
        let mut h = cache.states[common].clone();
        for t in tokens[common * TOKEN_DIM..].chunks_exact(TOKEN_DIM) {
            g.step(&mut h, t);
            cache.tokens.extend_from_slice(t);
            cache.states.push(h.clone());
        }
        h
    }

    /// Trunk output for state features and the GRU state from `history_state`.
    pub fn trunk(&self, state: &[u8], hist: &[f32]) -> Vec<f32> {
        let mut x = to_inputs(state);
        x.extend_from_slice(hist);
        let mut h = mlp(&self.trunk, &x);
        if !self.trunk.is_empty() {
            h.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        h
    }

    /// Expected round margin (points) for my team.
    pub fn value(&self, h: &[f32]) -> f32 {
        mlp(&self.value, h)[0] * self.value_scale
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

    /// Trunk output for a position.
    pub fn trunk_of(&self, view: &PlayerView, cache: &mut HistoryCache) -> Vec<f32> {
        let hist = history::replay(view);
        let mut s = Vec::with_capacity(STATE_DIM);
        encode_state_with(view, &hist.summary, &mut s);
        self.trunk(&s, &self.history_state(&hist.tokens, cache))
    }

    pub fn value_of(&self, view: &PlayerView, cache: &mut HistoryCache) -> f32 {
        self.value(&self.trunk_of(view, cache))
    }
}

/// Plays the policy's top choice.
pub struct NetBot {
    net: Arc<Net>,
    base: H2Bot,
    cache: HistoryCache,
}

impl NetBot {
    pub fn new(net: Arc<Net>) -> Self {
        NetBot { net, base: H2Bot::new(), cache: HistoryCache::default() }
    }

    pub fn choose(net: &Net, view: &PlayerView, plays: &[Combo], can_pass: bool, cache: &mut HistoryCache) -> Option<usize> {
        let cands = Candidates::new(view, plays, can_pass);
        if cands.len() == 1 {
            return cands.options[0];
        }
        let logits = net.logits(&net.trunk_of(view, cache), &cands.features);
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
        Self::choose(&self.net, view, plays, can_pass, &mut self.cache)
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
    cache: HistoryCache,
}

impl ValuePimcBot {
    pub fn new(net: Arc<Net>, samples: usize, seed: u64) -> Self {
        ValuePimcBot { net, samples, rng: StdRng::seed_from_u64(seed), base: H2Bot::new(), cache: HistoryCache::default() }
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
            let base = GameState::from_view(view, view.sample_hands(&mut self.rng));
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
                        self.net.value_of(&g.view(view.seat), &mut self.cache)
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
        assert!((net.value(&got_h) - want_v).abs() < 1e-3);
        assert!((net.logits(&got_h, &a)[0] - want_l).abs() < 1e-4);
        std::fs::remove_file(path).ok();
    }

    /// The prefix cache gives the same GRU state as running from scratch,
    /// across sequences that share prefixes and ones that diverge.
    #[test]
    fn history_cache_matches_a_fresh_run() {
        let mut rng = StdRng::seed_from_u64(8);
        let h = 6;
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect() };
        let layer = |inp: usize, out: usize, w: Vec<f32>, b: Vec<f32>| Layer { inp, out, w, b };
        let g = Gru { hidden: h, ih: layer(TOKEN_DIM, 3 * h, w(3 * h * TOKEN_DIM), w(3 * h)), hh: layer(h, 3 * h, w(3 * h * h), w(3 * h)) };
        let net = Net { trunk: vec![], value: vec![], policy: vec![], value_scale: 1.0, hidden: STATE_DIM, gru: Some(g) };
        let mut toks = |n: usize| -> Vec<u8> { (0..n * TOKEN_DIM).map(|_| rng.gen_range(0..=255u8)).collect() };
        let a = toks(10);
        let mut b = a[..4 * TOKEN_DIM].to_vec();
        b.extend(toks(3));
        let mut cache = HistoryCache::default();
        for seq in [&a[..5 * TOKEN_DIM], &a[..], &b[..], &a[..2 * TOKEN_DIM], &[][..], &a[..]] {
            let fresh = net.history_state(seq, &mut HistoryCache::default());
            assert_eq!(net.history_state(seq, &mut cache), fresh);
            assert_eq!(fresh.len(), h);
        }
    }
}
