//! A learned call model (`.tcall`, written by `train/calls.py`) and the bot
//! wrapper that makes Grand / Small Tichu calls with it.
//!
//! The model predicts how much a call changes my team's chance of winning
//! the game, in percentage points, from what the seat can see and the game
//! score. The bot calls when that is above a threshold (0 by default).
//!
//! File: `b"TCLL"`, u32 version, u32 `FEATURE_VERSION`, u32 input dim, then
//! the trunk and two heads (Grand, Small), each `u32 n layers` and layers
//! of `u32 in, u32 out, f32 w[out][in], f32 b[out]`. ReLU after every
//! layer but a head's last.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::bot::Bot;
use crate::engine::GameState;
use crate::features::FEATURE_VERSION;
use crate::scoring::RoundResult;
use crate::types::*;
use crate::view::PlayerView;

use crate::calllabel::{call_inputs, CALL_INPUT_DIM};

pub const MODEL_VERSION: u32 = 1;

struct Layer {
    inp: usize,
    out: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

fn forward(layers: &[Layer], x: &[f32], relu_last: bool) -> Vec<f32> {
    let mut x = x.to_vec();
    for (i, l) in layers.iter().enumerate() {
        let mut y = l.b.clone();
        for o in 0..l.out {
            let row = &l.w[o * l.inp..(o + 1) * l.inp];
            y[o] += row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>();
        }
        if relu_last || i + 1 < layers.len() {
            for v in &mut y {
                *v = v.max(0.0);
            }
        }
        x = y;
    }
    x
}

pub struct CallNet {
    trunk: Vec<Layer>,
    heads: [Vec<Layer>; 2],
}

impl CallNet {
    pub fn load(path: &str) -> Result<CallNet, String> {
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let mut p = 0usize;
        let u32_at = |p: &mut usize| -> Result<u32, String> {
            let b = data.get(*p..*p + 4).ok_or(format!("{path}: truncated"))?;
            *p += 4;
            Ok(u32::from_le_bytes(b.try_into().unwrap()))
        };
        if data.get(0..4) != Some(b"TCLL") {
            return Err(format!("{path}: not a .tcall model"));
        }
        p += 4;
        let v = u32_at(&mut p)?;
        let fv = u32_at(&mut p)?;
        let dim = u32_at(&mut p)? as usize;
        if v != MODEL_VERSION || fv != FEATURE_VERSION || dim != CALL_INPUT_DIM {
            return Err(format!("{path}: model v{v}, features v{fv}, {dim} inputs; want v{MODEL_VERSION}, v{FEATURE_VERSION}, {CALL_INPUT_DIM}"));
        }
        let read_stack = |p: &mut usize, want_in: usize| -> Result<Vec<Layer>, String> {
            let n = u32_at(p)? as usize;
            let mut layers = vec![];
            let mut inp_expected = want_in;
            for _ in 0..n {
                let inp = u32_at(p)? as usize;
                let out = u32_at(p)? as usize;
                if inp != inp_expected {
                    return Err(format!("{path}: layer shapes don't chain"));
                }
                let f = |k: usize, p: &mut usize| -> Result<Vec<f32>, String> {
                    let b = data.get(*p..*p + 4 * k).ok_or(format!("{path}: truncated"))?;
                    *p += 4 * k;
                    Ok(b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
                };
                let w = f(inp * out, p)?;
                let b = f(out, p)?;
                layers.push(Layer { inp, out, w, b });
                inp_expected = out;
            }
            Ok(layers)
        };
        let trunk = read_stack(&mut p, dim)?;
        let t_out = trunk.last().map_or(dim, |l| l.out);
        let g = read_stack(&mut p, t_out)?;
        let s = read_stack(&mut p, t_out)?;
        if p != data.len() {
            return Err(format!("{path}: trailing bytes"));
        }
        Ok(CallNet { trunk, heads: [g, s] })
    }

    pub fn cached(path: &str) -> Result<Arc<CallNet>, String> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<CallNet>>>> = OnceLock::new();
        let mut c = CACHE.get_or_init(Default::default).lock().unwrap();
        if let Some(n) = c.get(path) {
            return Ok(n.clone());
        }
        let n = Arc::new(CallNet::load(path)?);
        c.insert(path.to_string(), n.clone());
        Ok(n)
    }

    /// Predicted change in my team's game-win chance (percentage points) if
    /// I make this call now.
    pub fn gain(&self, view: &PlayerView, grand: bool) -> f32 {
        let x = call_inputs(view, grand);
        let h = forward(&self.trunk, &x, true);
        forward(&self.heads[if grand { 0 } else { 1 }], &h, false)[0]
    }
}

/// `callnet[@T][~F]:PATH:BOT`: BOT, with calls from the model at PATH.
/// T: call when the predicted gain is above T points (default 0). Flags F:
/// g = model Grand only (BOT's Small), s = model Small only (BOT's Grand).
pub struct CallNetBot {
    net: Arc<CallNet>,
    threshold: f32,
    grand: bool,
    small: bool,
    inner: Box<dyn Bot>,
}

impl CallNetBot {
    pub fn by_name(name: &str, seed: u64) -> Option<CallNetBot> {
        let rest = name.strip_prefix("callnet")?;
        let (head, rest) = rest.split_once(':')?;
        // A Windows drive letter ("C:\..." or "C:/...") belongs to the path.
        let skip = match rest.as_bytes() {
            [d, b':', b'\\' | b'/', ..] if d.is_ascii_alphabetic() => 2,
            _ => 0,
        };
        let cut = skip + rest[skip..].find(':')?;
        let (path, inner) = (&rest[..cut], &rest[cut + 1..]);
        let (head, flags) = head.split_once('~').unwrap_or((head, ""));
        let threshold = match head.strip_prefix('@') {
            Some(t) => t.parse().ok()?,
            None if head.is_empty() => 0.0,
            None => return None,
        };
        let (grand, small) = match flags {
            "" => (true, true),
            "g" => (true, false),
            "s" => (false, true),
            _ => return None,
        };
        let net = CallNet::cached(path).map_err(|e| eprintln!("{e}")).ok()?;
        Some(CallNetBot { net, threshold, grand, small, inner: crate::bots::by_name(inner, seed)? })
    }
}

impl Bot for CallNetBot {
    fn name(&self) -> String {
        format!("callnet:{}", self.inner.name())
    }
    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        if !self.grand {
            return self.inner.grand_tichu(view);
        }
        self.net.gain(view, true) > self.threshold
    }
    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        if !self.small {
            return self.inner.small_tichu(view);
        }
        self.net.gain(view, false) > self.threshold
    }
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.inner.pass_cards(view)
    }
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        self.inner.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        self.inner.bomb(view, bombs)
    }
    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        self.inner.wish(view)
    }
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        self.inner.dragon_recipient(view)
    }
    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        self.inner.round_over(view, result)
    }
    fn cheats(&self) -> bool {
        self.inner.cheats()
    }
    fn peek(&mut self, state: &GameState) {
        self.inner.peek(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::shuffled_deck;
    use crate::sim::play_dealt_round;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// A random model in the .tcall format: loads, rejects a bad header, and
    /// plays whole rounds with both kinds of call.
    #[test]
    fn random_model_plays_rounds() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut file = b"TCLL".to_vec();
        for v in [MODEL_VERSION, FEATURE_VERSION, CALL_INPUT_DIM as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let mut stack = |file: &mut Vec<u8>, shapes: &[(usize, usize)]| {
            file.extend_from_slice(&(shapes.len() as u32).to_le_bytes());
            for &(i, o) in shapes {
                file.extend_from_slice(&(i as u32).to_le_bytes());
                file.extend_from_slice(&(o as u32).to_le_bytes());
                for _ in 0..(i * o + o) {
                    file.extend_from_slice(&rng.gen_range(-0.5f32..0.5).to_le_bytes());
                }
            }
        };
        stack(&mut file, &[(CALL_INPUT_DIM, 16)]);
        stack(&mut file, &[(16, 8), (8, 1)]);
        stack(&mut file, &[(16, 1)]);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("tichu-test-{}.tcall", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let bad = dir.join(format!("tichu-test-bad-{}.tcall", std::process::id()));
        std::fs::write(&bad, &file[..file.len() - 4]).unwrap();
        assert!(CallNet::load(bad.to_str().unwrap()).is_err());
        let net = CallNet::cached(path.to_str().unwrap()).unwrap();
        assert!(CallNetBot::by_name("callnet@x:m.tcall:h2", 0).is_none());
        for seed in 0..3 {
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| -> Box<dyn Bot> {
                let inner = crate::bots::by_name("h2", 0).unwrap();
                Box::new(CallNetBot { net: net.clone(), threshold: -1000.0, grand: true, small: true, inner })
            });
            let mut state = GameState::new(Settings::default());
            state.start_round(shuffled_deck(&mut StdRng::seed_from_u64(seed)));
            let r = play_dealt_round(&mut state, &mut bots, None).unwrap();
            // Threshold -1000: everyone calls Grand.
            assert!(r.tichu_calls.iter().all(|c| *c == TichuCall::Grand));
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bad);
    }
}
