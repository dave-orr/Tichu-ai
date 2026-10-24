//! Narrates a logged game (`games/deal-NNNN.json`): the deal, the passes,
//! every play trick by trick with all four hands, and, at each LLM decision,
//! the question's options and the model's full reply.
//!
//! The log holds the deck and the action list, so the narration re-runs the
//! game through the engine: hands and tricks shown are the engine's, not a
//! second bookkeeping.

use std::fmt::Write;

use serde_json::Value;

use crate::engine::{Action, GameState};
use crate::scoring::{total_points, RoundResult};
use crate::types::*;

use super::render::{cards, describe};

pub struct ReplayOptions {
    /// Print the model's whole reply (else only its last paragraph).
    pub full: bool,
    /// Show all four hands before each play.
    pub hands: bool,
}

fn label(players: &[String], seat: Seat) -> String {
    format!("S{seat} {}", players[seat])
}

/// The menu part of a prompt (after "DECISION").
fn decision_text(prompt: &str) -> &str {
    prompt.split("\nDECISION\n").nth(1).unwrap_or("")
}

fn last_paragraph(reply: &str) -> String {
    let paras: Vec<&str> = reply.trim().split("\n\n").collect();
    let n = paras.len();
    paras[n.saturating_sub(2)..].join("\n\n")
}

fn indent(s: &str, pad: &str) -> String {
    s.lines().map(|l| format!("{pad}{l}\n")).collect()
}

/// Narrate one game log (one seating).
pub fn narrate(game: &Value, opts: &ReplayOptions) -> Result<String, String> {
    let players: Vec<String> = serde_json::from_value(game["players"].clone()).map_err(|e| e.to_string())?;
    let deck: Vec<Card> = serde_json::from_value(game["deck"].clone()).map_err(|e| e.to_string())?;
    let events = game["events"].as_array().ok_or("no events")?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== Deal {}, {} ===",
        game["deal"],
        if game["swapped"].as_bool() == Some(true) { "seating B" } else { "seating A" }
    );
    let _ = writeln!(out, "Teams: S0 & S2 = {}; S1 & S3 = {}", players[0], players[1]);
    let _ = writeln!(out, "\nDealt hands (first 8 | last 6):");
    for s in 0..4 {
        let _ = writeln!(out, "  {:<28} {} | {}", label(&players, s), cards(&deck[s * 14..s * 14 + 8]), cards(&deck[s * 14 + 8..s * 14 + 14]));
    }

    let mut st = GameState::new(Settings::default());
    st.start_round(deck);
    let mut trick_no = 0;
    let mut trick_cards: Vec<Card> = Vec::new();
    let mut started_play = false;
    let mut passes = Vec::new();
    for ev in events {
        match ev["type"].as_str() {
            Some("llm") => {
                let seat = ev["seat"].as_u64().unwrap_or(0) as usize;
                let kind = ev["kind"].as_str().unwrap_or("?");
                let reply = ev["reply"].as_str().unwrap_or("");
                let answer = &ev["answer"];
                let _ = writeln!(out, "\n    ┌ {} asked ({kind}), answered {answer}:", label(&players, seat));
                if kind != "pass" || opts.full {
                    let _ = write!(out, "{}", indent(decision_text(ev["prompt"].as_str().unwrap_or("")).trim_end(), "    │ ? "));
                }
                let body = if opts.full { reply.trim().to_string() } else { last_paragraph(reply) };
                let _ = write!(out, "{}", indent(&body, "    │ "));
                let _ = writeln!(out, "    └");
            }
            Some("fallback") => {
                let seat = ev["seat"].as_u64().unwrap_or(0) as usize;
                let _ = writeln!(out, "    (no usable answer: {}'s fallback bot decides this {})", label(&players, seat), ev["kind"]);
            }
            Some("action") => {
                let a: Action = serde_json::from_value(ev["action"].clone()).map_err(|e| e.to_string())?;
                match &a {
                    Action::GrandTichu { seat, call: true } => {
                        let _ = writeln!(out, "{} calls GRAND TICHU", label(&players, *seat));
                    }
                    Action::PassCards { seat, pass } => passes.push((*seat, *pass)),
                    _ => {}
                }
                if !started_play && matches!(a, Action::Play { .. } | Action::SmallTichu { .. } | Action::PassTurn { .. }) {
                    started_play = true;
                    let _ = writeln!(out, "\nPasses (left / partner / right):");
                    for (seat, p) in &passes {
                        let _ = writeln!(out, "  {:<28} {} / {} / {}", label(&players, *seat), p.left, p.partner, p.right);
                    }
                    let _ = writeln!(out, "\nHands after passing:");
                    for s in 0..4 {
                        let _ = writeln!(out, "  {:<28} {}", label(&players, s), cards(&st.players[s].hand));
                    }
                }
                let new_trick = matches!(a, Action::Play { .. }) && st.current_trick.is_none();
                if new_trick {
                    trick_no += 1;
                    let _ = writeln!(out, "\nTrick {trick_no}:");
                }
                if opts.hands && matches!(a, Action::Play { .. }) {
                    let hs: Vec<String> = (0..4).map(|s| format!("S{s}: {}", cards(&st.players[s].hand))).collect();
                    let _ = writeln!(out, "  [{}]", hs.join(" | "));
                }
                match &a {
                    Action::SmallTichu { seat } => {
                        let _ = writeln!(out, "  {} calls TICHU", label(&players, *seat));
                    }
                    Action::Play { seat, cards: cs, phoenix_as } => {
                        let combo = crate::combinations::identify_combo(cs, *phoenix_as);
                        let d = combo.map(|c| describe(&c)).unwrap_or_else(|| cards(cs));
                        let _ = writeln!(out, "  {} plays {d}", label(&players, *seat));
                        if cs != &[DOG] {
                            trick_cards.extend_from_slice(cs);
                        }
                    }
                    Action::Bomb { seat, cards: cs } => {
                        let _ = writeln!(out, "  {} BOMBS {}", label(&players, *seat), cards(cs));
                        trick_cards.extend_from_slice(cs);
                    }
                    Action::PassTurn { seat } => {
                        let _ = writeln!(out, "  {} passes", label(&players, *seat));
                    }
                    Action::Wish { seat, rank } => {
                        let w = rank.map(|r| ["2","3","4","5","6","7","8","9","10","J","Q","K","A"][r as usize - 2].to_string()).unwrap_or("nothing".into());
                        let _ = writeln!(out, "  {} wishes for {w}", label(&players, *seat));
                    }
                    Action::ResolveTrick if !trick_cards.contains(&DRAGON) => {
                        let w = st.last_played_by.unwrap_or(0);
                        let _ = writeln!(out, "  -> {} wins the trick ({} pts)", label(&players, w), total_points(&trick_cards));
                        trick_cards.clear();
                    }
                    Action::GiveDragon { seat, to } => {
                        let _ = writeln!(
                            out,
                            "  -> {} wins with the Dragon and gives the trick ({} pts) to {}",
                            label(&players, *seat),
                            total_points(&trick_cards),
                            label(&players, *to)
                        );
                        trick_cards.clear();
                    }
                    _ => {}
                }
                let was_out: Vec<bool> = (0..4).map(|s| st.players[s].is_out()).collect();
                st.apply(&a).map_err(|e| format!("replay diverged at {a:?}: {e:?}"))?;
                for s in 0..4 {
                    if !was_out[s] && st.players[s].is_out() {
                        let _ = writeln!(out, "  ** {} is out ({}) **", label(&players, s), st.players[s].out_order);
                    }
                }
                if matches!(a, Action::Play { ref cards, .. } if cards == &[DOG]) {
                    let _ = writeln!(out, "  -> the Dog passes the lead");
                }
            }
            _ => {}
        }
    }
    let r: RoundResult = serde_json::from_value(game["result"].clone()).map_err(|e| e.to_string())?;
    let _ = writeln!(out, "\nResult:");
    let order: Vec<String> = r.out_order.iter().map(|&s| label(&players, s)).collect();
    let _ = writeln!(out, "  finishing order: {}", order.join(", "));
    if let Some(t) = r.double_victory {
        let _ = writeln!(out, "  1-2 finish for team {t}");
    }
    let _ = writeln!(
        out,
        "  team 0 (S0/S2, {}): {} card pts {:+} calls = {}\n  team 1 (S1/S3, {}): {} card pts {:+} calls = {}",
        players[0],
        r.card_points[0],
        r.tichu_bonuses[0],
        r.total(0),
        players[1],
        r.card_points[1],
        r.tichu_bonuses[1],
        r.total(1)
    );
    Ok(out)
}
