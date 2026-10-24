//! Game logs and the visual replay viewer.
//!
//! A *game log* is one played round: the deck, the seat names, every action,
//! and (from LLM seats) each question and reply, interleaved with the
//! actions they led to. It is enough to replay the round exactly through the
//! engine. `game_log` writes one; `llm_match` and `tichu arena --log` both
//! use it, and a log file may hold one game or several (any JSON object or
//! array with game logs inside, e.g. `{"deal": 3, "llm": [g, g], ...}`).
//!
//! `build` re-runs a game log through the engine and turns it into replay
//! steps: after each event, all four hands, the trick on the table, the
//! points taken, the calls, and any model reasoning that led to it. `html`
//! embeds replays in the self-contained viewer (`viewer/replay.html`).
//! Hands and tricks shown are the engine's, not a second bookkeeping.

use serde_json::{json, Value};

use crate::combinations::identify_combo;
use crate::engine::{Action, GameState};
use crate::llm::render::describe;
use crate::scoring::{total_points, RoundResult};
use crate::types::*;

/// The game-log schema version (`version` in each log).
pub const LOG_VERSION: u32 = 1;

/// A game log. `events` holds `{"type": "action", "action": ...}` entries,
/// plus any extra entries a player adds (`llm`, `fallback`).
pub fn game_log(
    deal: usize,
    swapped: bool,
    players: &[&str],
    settings: &Settings,
    deck: &[Card],
    result: &RoundResult,
    events: Vec<Value>,
) -> Value {
    json!({
        "version": LOG_VERSION,
        "deal": deal,
        "swapped": swapped,
        "players": players,
        "settings": { "clockwise": settings.clockwise },
        "deck": deck,
        "result": result,
        "events": events,
    })
}

/// The event entry for one action.
pub fn action_event(a: &Action) -> Value {
    json!({ "type": "action", "action": a })
}

/// Every game log inside `v`, with a label naming where it was found
/// (`deal 0 · llm · seating A`).
pub fn games_in(v: &Value) -> Vec<(String, &Value)> {
    fn walk<'a>(v: &'a Value, path: &mut Vec<String>, out: &mut Vec<(String, &'a Value)>) {
        match v {
            Value::Object(m) if m.contains_key("deck") && m.contains_key("events") => {
                let mut parts = path.clone();
                if let Some(d) = m.get("deal") {
                    if !parts.iter().any(|p| p.starts_with("deal ")) {
                        parts.insert(0, format!("deal {d}"));
                    }
                }
                if let Some(s) = m.get("swapped").and_then(Value::as_bool) {
                    parts.push(if s { "seating B".into() } else { "seating A".into() });
                }
                out.push((parts.join(" · "), v));
            }
            Value::Object(m) => {
                let mut deal_part = false;
                if let Some(d) = m.get("deal").filter(|d| d.is_number()) {
                    path.push(format!("deal {d}"));
                    deal_part = true;
                }
                for (k, x) in m {
                    if x.is_object() || x.is_array() {
                        path.push(k.clone());
                        walk(x, path, out);
                        path.pop();
                    }
                }
                if deal_part {
                    path.pop();
                }
            }
            Value::Array(xs) => {
                for x in xs {
                    walk(x, path, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(v, &mut Vec::new(), &mut out);
    out
}

fn ids(cards: &[Card]) -> Vec<String> {
    let mut v = cards.to_vec();
    v.sort_by_key(|c| c.sort_key());
    v.iter().map(|c| c.id()).collect()
}

fn rank_name(r: u8) -> &'static str {
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"][r as usize - 1]
}

fn ordinal(n: u8) -> &'static str {
    ["", "1st", "2nd", "3rd", "4th"][n as usize]
}

/// The menu part of an LLM prompt (after "DECISION").
fn decision_text(prompt: &str) -> &str {
    prompt.split("\nDECISION\n").nth(1).unwrap_or("").trim_end()
}

/// The menu lines an answer picked (`3` -> the line `3: ...`).
fn chosen(menu: &str, answer: &Value) -> Vec<String> {
    let picks: Vec<u64> = match answer {
        Value::Array(xs) => xs.iter().filter_map(Value::as_u64).collect(),
        Value::Number(n) => n.as_u64().into_iter().collect(),
        _ => vec![],
    };
    picks
        .iter()
        .map(|n| {
            let head = format!("{n}: ");
            menu.lines()
                .find_map(|l| l.trim_start().strip_prefix(&head).map(str::to_string))
                .unwrap_or_else(|| n.to_string())
        })
        .collect()
}

/// A model decision, for the step it led to.
fn thought(ev: &Value) -> Value {
    let prompt = ev["prompt"].as_str().unwrap_or("");
    let menu = decision_text(prompt);
    json!({
        "seat": ev["seat"],
        "kind": ev["kind"],
        "fallback": ev["type"] == "fallback",
        "answer": ev["answer"],
        "chose": chosen(menu, &ev["answer"]),
        "menu": menu,
        "prompt": prompt,
        "reply": ev["reply"],
        "cached": ev["cached"],
    })
}

/// One entry of the table display: a seat's play or pass in the current trick.
fn table_entry(seat: Seat, kind: &str, cards: &[Card], text: &str) -> Value {
    json!({ "seat": seat, "kind": kind, "cards": ids(cards), "text": text })
}

/// Replays a game log through the engine and returns the viewer's replay:
/// `{title, players, deal, dealt, steps, result}`.
pub fn build(game: &Value, label: &str) -> Result<Value, String> {
    let players: Vec<String> = serde_json::from_value(game["players"].clone()).map_err(|e| e.to_string())?;
    let deck: Vec<Card> = serde_json::from_value(game["deck"].clone()).map_err(|e| e.to_string())?;
    let events = game["events"].as_array().ok_or("no events")?;
    let clockwise = game["settings"]["clockwise"].as_bool().unwrap_or(false);
    let mut st = GameState::new(Settings { clockwise, ..Settings::default() });
    st.start_round(deck.clone());
    let name = |s: Seat| format!("S{s}");

    let mut steps: Vec<Value> = Vec::new();
    let mut thoughts: Vec<Value> = Vec::new();
    // The current trick as the table shows it, passes included.
    let mut table: Vec<Value> = Vec::new();
    // Cards that changed hands in the pass: per receiver, (card id, from).
    let mut received: [Vec<(String, Seat)>; 4] = Default::default();
    let mut trick_no = 0;
    let mut events_left = events.len();

    let snapshot = |st: &GameState| {
        json!({
            "hands": st.players.iter().map(|p| ids(&p.hand)).collect::<Vec<_>>(),
            "taken": st.players.iter().map(|p| p.captured_points()).collect::<Vec<_>>(),
            "tricks": st.players.iter().map(|p| p.tricks_won.len()).collect::<Vec<_>>(),
            "calls": st.players.iter().map(|p| p.tichu).collect::<Vec<_>>(),
            "out": st.players.iter().map(|p| p.out_order).collect::<Vec<_>>(),
            "wish": st.wish,
        })
    };

    steps.push({
        let mut s = snapshot(&st);
        s["kind"] = json!("deal");
        s["text"] = json!("Cards dealt (first 8 shown bright; the last 6 come after the Grand Tichu call)");
        s["table"] = json!([]);
        s["firstEight"] = json!((0..4).map(|p| ids(&deck[p * 14..p * 14 + 8])).collect::<Vec<_>>());
        s["hands"] = json!((0..4).map(|p| ids(&deck[p * 14..p * 14 + 14])).collect::<Vec<_>>());
        s
    });

    for ev in events {
        events_left -= 1;
        match ev["type"].as_str() {
            Some("llm") | Some("fallback") => {
                thoughts.push(thought(ev));
                continue;
            }
            Some("action") => {}
            _ => continue,
        }
        let a: Action = serde_json::from_value(ev["action"].clone()).map_err(|e| e.to_string())?;
        let before = st.clone();
        let seat: Option<Seat>;
        let mut cards: Vec<Card> = Vec::new();
        let (kind, mut text) = match &a {
            Action::GrandTichu { seat: s, call } => {
                seat = Some(*s);
                ("grandTichu", if *call { format!("{} calls GRAND TICHU", name(*s)) } else { format!("{} does not call Grand Tichu", name(*s)) })
            }
            Action::SmallTichu { seat: s } => {
                seat = Some(*s);
                ("tichu", format!("{} calls TICHU", name(*s)))
            }
            Action::PassCards { seat: s, pass } => {
                seat = Some(*s);
                for (card, to) in [(pass.left, (s + 3) % 4), (pass.partner, (s + 2) % 4), (pass.right, (s + 1) % 4)] {
                    received[to].push((card.id(), *s));
                }
                ("passCards", format!("{} passes {} left, {} to partner, {} right", name(*s), pass.left, pass.partner, pass.right))
            }
            Action::Play { seat: s, cards: cs, phoenix_as } => {
                seat = Some(*s);
                cards = cs.clone();
                if st.current_trick.is_none() {
                    trick_no += (cs != &[DOG]) as usize;
                    table.clear();
                }
                let combo = identify_combo(cs, *phoenix_as);
                let d = combo.as_ref().map(describe).unwrap_or_else(|| cards_str(cs));
                let short = d.split(':').next().unwrap_or(&d).to_string();
                table.push(table_entry(*s, "play", cs, &short));
                let lead = if table.len() == 1 { "leads" } else { "plays" };
                ("play", format!("{} {lead} {d}", name(*s)))
            }
            Action::Bomb { seat: s, cards: cs } => {
                seat = Some(*s);
                cards = cs.clone();
                let d = identify_combo(cs, None).as_ref().map(describe).unwrap_or_else(|| cards_str(cs));
                table.push(table_entry(*s, "bomb", cs, "BOMB"));
                ("bomb", format!("{} BOMBS — {d}", name(*s)))
            }
            Action::PassTurn { seat: s } => {
                seat = Some(*s);
                table.push(table_entry(*s, "pass", &[], "Pass"));
                ("passTurn", format!("{} passes", name(*s)))
            }
            Action::Wish { seat: s, rank } => {
                seat = Some(*s);
                ("wish", match rank {
                    Some(r) => format!("{} wishes for a {}", name(*s), rank_name(*r)),
                    None => format!("{} makes no wish", name(*s)),
                })
            }
            Action::ResolveTrick => {
                let w = st.trick_countdown.or(st.last_played_by).unwrap_or(0);
                seat = Some(w);
                let pts: i32 = st.trick_plays.iter().map(|p| total_points(&p.combo.cards)).sum();
                if st.trick_plays.iter().any(|p| p.combo.cards.contains(&DRAGON)) {
                    ("resolve", format!("{} wins trick {trick_no} with the Dragon and must give it to an opponent", name(w)))
                } else {
                    ("resolve", format!("{} wins trick {trick_no} ({pts} pts)", name(w)))
                }
            }
            Action::GiveDragon { seat: s, to } => {
                seat = Some(*s);
                let pts: i32 = st.trick_plays.iter().map(|p| total_points(&p.combo.cards)).sum();
                ("giveDragon", format!("{} gives the Dragon trick ({pts} pts) to {}", name(*s), name(*to)))
            }
        };
        st.apply(&a).map_err(|e| format!("replay diverged at {a:?}: {e:?}"))?;

        for s in 0..4 {
            if !before.players[s].is_out() && st.players[s].is_out() {
                text += &format!(" — {} is out ({})", name(s), ordinal(st.players[s].out_order));
            }
        }
        if cards == [DOG] {
            text += &format!("; the lead passes to {}", name(st.turn));
        }

        // Quiet steps fold into the next one: Grand Tichu declines, and every
        // pass but the last (the exchange happens when all four have chosen).
        let quiet = match &a {
            Action::GrandTichu { call: false, .. } => true,
            Action::PassCards { .. } => st.phase == Phase::Passing,
            _ => false,
        };
        if quiet && thoughts.is_empty() && events_left > 0 {
            continue;
        }

        let mut step = snapshot(&st);
        step["kind"] = json!(kind);
        step["seat"] = json!(seat);
        step["text"] = json!(text);
        step["cards"] = json!(ids(&cards));
        step["trickNo"] = json!(trick_no);
        if matches!(a, Action::PassCards { .. }) && st.phase != Phase::Passing {
            step["kind"] = json!("exchange");
            step["text"] = json!("Cards exchanged (received cards are marked with the passer's seat)");
            step["received"] = json!(received
                .iter()
                .map(|r| r.iter().map(|(c, from)| json!({ "card": c, "from": from })).collect::<Vec<_>>())
                .collect::<Vec<_>>());
        }
        if st.phase == Phase::GrandTichuWindow {
            // All 14 are shown during the call window, the last 6 faded.
            step["firstEight"] = json!((0..4).map(|p| ids(&deck[p * 14..p * 14 + 8])).collect::<Vec<_>>());
            step["hands"] = json!((0..4).map(|p| ids(&deck[p * 14..p * 14 + 14])).collect::<Vec<_>>());
        }
        // The table keeps a finished trick on show for the step that awards it.
        step["table"] = json!(table);
        if matches!(a, Action::ResolveTrick | Action::GiveDragon { .. }) {
            step["won"] = json!(true);
            if matches!(a, Action::GiveDragon { .. }) || !before.trick_plays.iter().any(|p| p.combo.cards.contains(&DRAGON)) {
                table.clear();
            }
        }
        if cards == [DOG] {
            table.clear();
        }
        step["turn"] = match st.pending() {
            crate::engine::Pending::Turn(s) | crate::engine::Pending::Wish(s) | crate::engine::Pending::DragonGiveaway(s) => json!(s),
            _ => Value::Null,
        };
        step["thoughts"] = json!(std::mem::take(&mut thoughts));
        steps.push(step);
    }

    let r: RoundResult = serde_json::from_value(game["result"].clone()).map_err(|e| e.to_string())?;
    let mut last = snapshot(&st);
    last["kind"] = json!("result");
    last["table"] = json!([]);
    let order: Vec<String> = r.out_order.iter().map(|&s| name(s)).collect();
    last["text"] = json!(format!(
        "Round over. Out order {}{}. Team S0/S2 {:+}, team S1/S3 {:+}",
        order.join(", "),
        if r.double_victory.is_some() { " (1-2 finish)" } else { "" },
        r.total(0),
        r.total(1)
    ));
    last["thoughts"] = json!(thoughts);
    steps.push(last);

    let title = if label.is_empty() { format!("Deal {}", game["deal"]) } else { label.to_string() };
    Ok(json!({
        "title": title,
        "players": players,
        "deal": game["deal"],
        "swapped": game["swapped"],
        "clockwise": clockwise,
        "result": r,
        "totals": [r.total(0), r.total(1)],
        "steps": steps,
    }))
}

const VIEWER: &str = include_str!("../viewer/replay.html");
const DATA_MARK: &str = "/*REPLAY_DATA*/[]";
const ART_MARK: &str = "/*CARD_ART*/{}";
const WASM_MARK: &str = "/*REPLAY_WASM*/";
/// This module's `build_from_text`, compiled to WebAssembly so the page can
/// open raw game logs. Rebuild it with `scripts/build-viewer-wasm.sh` after
/// changing the replay builder or the rules.
const WASM: &[u8] = include_bytes!("../viewer/replay.wasm");

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}
/// Card art from the production client (`client/public/cards/` in the tichu
/// repo), embedded so the page is one self-contained file.
const ART: [(&str, &str); 8] = [
    ("pip-jade", include_str!("../viewer/cards/pip-jade.svg")),
    ("pip-sword", include_str!("../viewer/cards/pip-sword.svg")),
    ("pip-pagoda", include_str!("../viewer/cards/pip-pagoda.svg")),
    ("pip-star", include_str!("../viewer/cards/pip-star.svg")),
    ("mahjong", include_str!("../viewer/cards/mahjong.svg")),
    ("dog", include_str!("../viewer/cards/dog.svg")),
    ("phoenix", include_str!("../viewer/cards/phoenix.svg")),
    ("dragon", include_str!("../viewer/cards/dragon.svg")),
];

/// `v` as JSON that is safe inside a `<script>` element: `<` only appears
/// inside JSON strings, where `\u003c` means the same.
fn script_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap().replace('<', "\\u003c")
}

/// The viewer's content with `replays` (from `build`) embedded, without the
/// document head (for hosts that supply their own).
pub fn html_body(replays: &[Value]) -> String {
    let art: serde_json::Map<String, Value> = ART.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
    VIEWER
        .replacen(DATA_MARK, &script_json(&json!(replays)), 1)
        .replacen(ART_MARK, &script_json(&Value::Object(art)), 1)
        .replacen(WASM_MARK, &base64(WASM), 1)
}

/// The viewer page with `replays` (from `build`) embedded: one
/// self-contained HTML file.
pub fn html(replays: &[Value]) -> String {
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n{}\n</html>\n",
        html_body(replays)
    )
}

/// Every replay in a log file's text: one JSON document, or JSONL with one
/// per line. Ok(empty) if the file holds no game logs.
pub fn build_from_text(text: &str) -> Result<Vec<Value>, String> {
    let docs: Vec<Value> = match serde_json::from_str(text) {
        Ok(v) => vec![v],
        Err(e) => {
            let docs: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect();
            if docs.is_empty() {
                return Err(format!("not JSON: {e}"));
            }
            docs
        }
    };
    let mut out = Vec::new();
    for d in &docs {
        out.extend(build_all(d)?);
    }
    Ok(out)
}

/// Every replay in a log file's JSON (one per game log found).
pub fn build_all(v: &Value) -> Result<Vec<Value>, String> {
    games_in(v).into_iter().map(|(label, g)| build(g, &label)).collect()
}
