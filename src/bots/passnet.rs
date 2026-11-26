//! A learned passer: `passnet[@T]:PATH:BOT` is BOT with its partner card
//! chosen by the pass model at PATH (`train/passes.py`, labels from
//! `tichu passlabel`).
//!
//! The model scores every candidate pass (`passlabel::partner_candidates`:
//! BOT's own two opponent cards, each other card as the partner card) with
//! the round margin it expects, relative to the other candidates. The bot
//! keeps BOT's own pass unless the best candidate is predicted to beat it by
//! more than T points (default 0).
//!
//! Model file (`.tpass`, little endian): `b"TPSM"`, version, `PASS_DIM`,
//! value scale (f32: points per output unit), layer count, then per layer
//! `in`, `out` (u32), the weight `[out][in]` and the bias (f32). ReLU
//! between layers; one output.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::bot::Bot;
use crate::passlabel::{encode_pass, partner_candidates, PASS_DIM};
use crate::scoring::RoundResult;
use crate::types::*;
use crate::view::PlayerView;

pub const MODEL_VERSION: u32 = 1;

struct Layer {
    inp: usize,
    out: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

pub struct PassNet {
    layers: Vec<Layer>,
    scale: f32,
}

impl PassNet {
    pub fn load(path: &str) -> Result<PassNet, String> {
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if data.get(0..4) != Some(b"TPSM") {
            return Err(format!("{path}: not a .tpass model"));
        }
        let mut p = 4usize;
        let word = |p: &mut usize| -> Result<[u8; 4], String> {
            let b = data.get(*p..*p + 4).ok_or(format!("{path}: truncated"))?;
            *p += 4;
            Ok(b.try_into().unwrap())
        };
        let (v, dim) = (u32::from_le_bytes(word(&mut p)?), u32::from_le_bytes(word(&mut p)?) as usize);
        if v != MODEL_VERSION || dim != PASS_DIM {
            return Err(format!("{path}: model v{v} with {dim} inputs; want v{MODEL_VERSION}, {PASS_DIM}"));
        }
        let scale = f32::from_le_bytes(word(&mut p)?);
        let n = u32::from_le_bytes(word(&mut p)?) as usize;
        let mut layers = vec![];
        let mut want = dim;
        for _ in 0..n {
            let (inp, out) = (u32::from_le_bytes(word(&mut p)?) as usize, u32::from_le_bytes(word(&mut p)?) as usize);
            if inp != want {
                return Err(format!("{path}: layer shapes don't chain"));
            }
            let mut floats = |k: usize| -> Result<Vec<f32>, String> { (0..k).map(|_| Ok(f32::from_le_bytes(word(&mut p)?))).collect() };
            let w = floats(inp * out)?;
            let b = floats(out)?;
            layers.push(Layer { inp, out, w, b });
            want = out;
        }
        if p != data.len() || want != 1 {
            return Err(format!("{path}: bad layout"));
        }
        Ok(PassNet { layers, scale })
    }

    pub fn cached(path: &str) -> Result<Arc<PassNet>, String> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<PassNet>>>> = OnceLock::new();
        let mut c = CACHE.get_or_init(Default::default).lock().unwrap();
        if let Some(n) = c.get(path) {
            return Ok(n.clone());
        }
        let n = Arc::new(PassNet::load(path)?);
        c.insert(path.to_string(), n.clone());
        Ok(n)
    }

    /// The model's score of one candidate, in points (meaningful only
    /// against the other candidates of the same decision).
    pub fn score(&self, view: &PlayerView, pass: &PassInfo) -> f32 {
        let mut bytes = Vec::with_capacity(PASS_DIM);
        encode_pass(view, pass, &mut bytes);
        self.score_features(&bytes)
    }

    /// `score` of already encoded features (`PASS_DIM` bytes).
    pub fn score_features(&self, bytes: &[u8]) -> f32 {
        let mut x: Vec<f32> = bytes.iter().map(|&b| b as f32 / 255.0).collect();
        for (i, l) in self.layers.iter().enumerate() {
            let mut y = l.b.clone();
            for (j, xj) in x.iter().enumerate().filter(|(_, v)| **v != 0.0) {
                for o in 0..l.out {
                    y[o] += l.w[o * l.inp + j] * xj;
                }
            }
            if i + 1 < self.layers.len() {
                y.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            x = y;
        }
        x[0] * self.scale
    }

    /// The pass to make: `rule` unless another candidate scores more than
    /// `threshold` points above it.
    pub fn choose(&self, view: &PlayerView, rule: PassInfo, threshold: f32) -> PassInfo {
        let base = self.score(view, &rule);
        let mut best = (rule, base + threshold);
        for cand in partner_candidates(view, &rule) {
            let s = self.score(view, &cand);
            if s > best.1 {
                best = (cand, s);
            }
        }
        best.0
    }
}

pub struct PassNetBot {
    net: Arc<PassNet>,
    threshold: f32,
    inner: Box<dyn Bot>,
    name: String,
}

impl PassNetBot {
    pub fn by_name(name: &str, seed: u64) -> Option<PassNetBot> {
        let (head, rest) = name.split_once(':')?;
        let threshold = match head.strip_prefix("passnet")? {
            "" => 0.0,
            t => t.strip_prefix('@')?.parse().ok()?,
        };
        // PATH may hold a drive colon (C:\...): the model path ends at ".tpass:".
        let cut = rest.find(".tpass:")? + ".tpass".len();
        let net = PassNet::cached(&rest[..cut]).map_err(|e| eprintln!("{e}")).ok()?;
        let inner = crate::bots::by_name(&rest[cut + 1..], seed)?;
        Some(PassNetBot { net, threshold, inner, name: name.to_string() })
    }
}

impl Bot for PassNetBot {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.grand_tichu(v)
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        self.inner.small_tichu(v)
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        let rule = self.inner.pass_cards(v);
        self.net.choose(v, rule, self.threshold)
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
    fn cheats(&self) -> bool {
        self.inner.cheats()
    }
    fn peek(&mut self, state: &crate::engine::GameState) {
        self.inner.peek(state)
    }
    fn round_over(&mut self, v: &PlayerView, r: &RoundResult) {
        self.inner.round_over(v, r)
    }
}
