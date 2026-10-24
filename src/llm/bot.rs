//! `LlmBot`: a seat played by a language model, one fresh question per
//! decision. Calls (Grand / Small Tichu) are left to the wrapped `inner` bot,
//! so a match measures play only; `inner` also answers whenever the model's
//! reply can't be used.

use std::sync::Arc;

use serde_json::json;

use crate::bot::Bot;
use crate::scoring::RoundResult;
use crate::types::*;
use crate::view::PlayerView;

use super::client::Client;
use super::render::{self, Question, SYSTEM_PROMPT};

/// Which decisions go to the model (the rest go to `inner`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Decisions {
    pub play: bool,
    pub bomb: bool,
    pub pass: bool,
    pub wish: bool,
    pub dragon: bool,
}

impl Default for Decisions {
    fn default() -> Self {
        Decisions { play: true, bomb: true, pass: true, wish: true, dragon: true }
    }
}

impl Decisions {
    /// Comma-separated subset of `play,bomb,pass,wish,dragon`, or `all`.
    pub fn parse(s: &str) -> Option<Self> {
        if s == "all" {
            return Some(Self::default());
        }
        let mut d = Decisions { play: false, bomb: false, pass: false, wish: false, dragon: false };
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part {
                "play" => d.play = true,
                "bomb" => d.bomb = true,
                "pass" => d.pass = true,
                "wish" => d.wish = true,
                "dragon" => d.dragon = true,
                _ => return None,
            }
        }
        Some(d)
    }
}

/// Per-bot tallies.
#[derive(Debug, Default, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LlmStats {
    /// Questions put to the model (cached replays included).
    pub asked: u64,
    /// Decisions with a single legal option, made without asking.
    pub forced: u64,
    /// Decisions handed to `inner` because no usable answer came back.
    pub fallbacks: u64,
}

impl LlmStats {
    pub fn add(&mut self, o: &LlmStats) {
        self.asked += o.asked;
        self.forced += o.forced;
        self.fallbacks += o.fallbacks;
    }
}

pub struct LlmBot {
    client: Arc<Client>,
    inner: Box<dyn Bot>,
    decisions: Decisions,
    /// Free-form tag written into every decision record (deal, seating…).
    tag: serde_json::Value,
    pub stats: LlmStats,
    shared_stats: Option<Arc<std::sync::Mutex<LlmStats>>>,
}

impl LlmBot {
    pub fn new(client: Arc<Client>, inner: Box<dyn Bot>, decisions: Decisions) -> Self {
        LlmBot { client, inner, decisions, tag: serde_json::Value::Null, stats: LlmStats::default(), shared_stats: None }
    }

    pub fn with_tag(mut self, tag: serde_json::Value) -> Self {
        self.tag = tag;
        self
    }

    /// Also add this bot's tallies into `s` as they happen (the runner reads
    /// them after the round, when the bot itself is gone).
    pub fn with_shared_stats(mut self, s: Arc<std::sync::Mutex<LlmStats>>) -> Self {
        self.shared_stats = Some(s);
        self
    }

    fn bump(&mut self, f: impl Fn(&mut LlmStats)) {
        f(&mut self.stats);
        if let Some(s) = &self.shared_stats {
            f(&mut s.lock().unwrap());
        }
    }

    /// Ask until `accept` maps the answer's numbers to a decision; one extra
    /// try with a reminder if the first reply has no usable answer.
    fn ask<T>(&mut self, view: &PlayerView, q: Question, accept: impl Fn(&[i64]) -> Option<T>) -> Option<T> {
        self.bump(|s| s.asked += 1);
        let mut prompt = q.prompt.clone();
        for attempt in 0..2 {
            let (reply, cached) = match self.client.ask(SYSTEM_PROMPT, &prompt) {
                Ok(r) => r,
                Err(e) => {
                    self.record(view, &q, &prompt, &format!("ERROR: {e}"), None, false);
                    break;
                }
            };
            let nums = render::parse_answer(&reply);
            let choice = nums.as_deref().and_then(&accept);
            self.record(view, &q, &prompt, &reply, nums.clone(), cached);
            if choice.is_some() {
                return choice;
            }
            if attempt == 0 {
                prompt = format!(
                    "{}\n\n(A previous reply to this request had no valid final ANSWER line. \
                     End with exactly one line: ANSWER: <choice>.)",
                    q.prompt
                );
            }
        }
        self.bump(|s| s.fallbacks += 1);
        None
    }

    fn record(&self, view: &PlayerView, q: &Question, prompt: &str, reply: &str, nums: Option<Vec<i64>>, cached: bool) {
        self.client.log(&json!({
            "tag": self.tag,
            "seat": view.seat,
            "kind": q.kind,
            "cached": cached,
            "prompt": prompt,
            "reply": reply,
            "answer": nums,
        }));
    }
}

impl Bot for LlmBot {
    fn name(&self) -> String {
        format!("llm[{}]", self.client.backend_id())
    }

    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        self.inner.grand_tichu(view)
    }

    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        self.inner.small_tichu(view)
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        if self.decisions.pass {
            let (q, hand) = render::pass_question(view);
            let n = hand.len() as i64;
            let pick = self.ask(view, q, |a| {
                if a.len() != 3 || a.iter().any(|&x| x < 1 || x > n) || a[0] == a[1] || a[1] == a[2] || a[0] == a[2] {
                    return None;
                }
                let c = |i: i64| hand[(i - 1) as usize];
                Some(PassInfo { left: c(a[0]), partner: c(a[1]), right: c(a[2]) })
            });
            if let Some(p) = pick {
                return p;
            }
        }
        self.inner.pass_cards(view)
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if plays.is_empty() {
            return None;
        }
        if plays.len() == 1 && !can_pass {
            self.bump(|s| s.forced += 1);
            return Some(0);
        }
        if self.decisions.play {
            let (q, order) = render::play_question(view, plays, can_pass);
            let pick = self.ask(view, q, |a| match a {
                [0] if can_pass => Some(None),
                [k] if *k >= 1 && (*k as usize) <= order.len() => Some(Some(order[*k as usize - 1])),
                _ => None,
            });
            if let Some(p) = pick {
                return p;
            }
        }
        self.inner.play(view, plays, can_pass)
    }

    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        if self.decisions.bomb {
            let q = render::bomb_question(view, bombs);
            let pick = self.ask(view, q, |a| match a {
                [0] => Some(None),
                [k] if *k >= 1 && (*k as usize) <= bombs.len() => Some(Some(*k as usize - 1)),
                _ => None,
            });
            if let Some(p) = pick {
                return p;
            }
        }
        self.inner.bomb(view, bombs)
    }

    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        if self.decisions.wish {
            let q = render::wish_question(view);
            let pick = self.ask(view, q, |a| match a {
                [0] => Some(None),
                [r] if (2..=14).contains(r) => Some(Some(*r as Rank)),
                _ => None,
            });
            if let Some(p) = pick {
                return p;
            }
        }
        self.inner.wish(view)
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        if self.decisions.dragon {
            let q = render::dragon_question(view);
            let (right, left) = (right_seat(view.seat), left_seat(view.seat));
            let pick = self.ask(view, q, |a| match a {
                [1] => Some(right),
                [2] => Some(left),
                _ => None,
            });
            if let Some(p) = pick {
                return p;
            }
        }
        self.inner.dragon_recipient(view)
    }

    fn round_over(&mut self, view: &PlayerView, result: &RoundResult) {
        self.inner.round_over(view, result);
    }
}
