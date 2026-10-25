//! On-disk and on-wire formats other code depends on: game logs (replay
//! viewer, llm_match), action/card JSON (the TS server's format), the
//! feature encoding (trained `.tnet` models), and self-play record files
//! (`train/tichu_data.py`). A change that breaks one of these should fail
//! here, not in a training run or a viewer.

mod common;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};

use tichu_engine::bot::Bot;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::features::*;
use tichu_engine::llm::client::fnv1a;
use tichu_engine::replay;
use tichu_engine::scoring::RoundResult;
use tichu_engine::selfplay::{self, SelfplayConfig};
use tichu_engine::sim::play_dealt_round;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

// ---------------------------------------------------------------------------
// Game logs
// ---------------------------------------------------------------------------

/// Re-run a game log's actions through a fresh engine; return its result.
fn replay_actions(log: &Value) -> RoundResult {
    let deck: Vec<Card> = serde_json::from_value(log["deck"].clone()).unwrap();
    let clockwise = log["settings"]["clockwise"].as_bool().unwrap_or(false);
    let mut g = GameState::new(Settings { clockwise, ..Settings::default() });
    g.start_round(deck);
    for ev in log["events"].as_array().unwrap() {
        if ev["type"] == "action" {
            let a: Action = serde_json::from_value(ev["action"].clone()).unwrap();
            g.apply(&a).unwrap_or_else(|e| panic!("logged action {a:?} rejected: {e}"));
        }
    }
    g.last_result().expect("log doesn't finish the round").clone()
}

#[test]
fn game_logs_round_trip_through_json_and_the_engine() {
    for seed in 0..30u64 {
        let clockwise = seed % 3 == 0;
        let (deck, actions, r) = common::logged_round("random", "h2", seed, clockwise);
        let settings = Settings { clockwise, ..Settings::default() };
        let events: Vec<Value> = actions.iter().map(replay::action_event).collect();
        let log = replay::game_log(seed as usize, false, &["random", "h2", "random", "h2"], &settings, &deck, &r, events);
        // Through text and back, as the files are written and read.
        let log: Value = serde_json::from_str(&log.to_string()).unwrap();
        assert_eq!(log["version"], replay::LOG_VERSION);
        let logged: RoundResult = serde_json::from_value(log["result"].clone()).unwrap();
        assert_eq!(logged, r);
        assert_eq!(replay_actions(&log), r, "seed {seed}: the log replays to a different result");
        let rep = replay::build(&log, "").unwrap();
        assert_eq!(rep["clockwise"], clockwise);
        assert_eq!(rep["totals"], json!([r.total(0), r.total(1)]));
        // The final replay step shows what the engine ended with.
        let last = rep["steps"].as_array().unwrap().last().unwrap().clone();
        let out: Vec<u8> = serde_json::from_value(last["out"].clone()).unwrap();
        for (pos, &seat) in r.out_order.iter().enumerate() {
            if out[seat] != 0 {
                assert_eq!(out[seat] as usize, pos + 1, "seed {seed}: replay out order");
            }
        }
    }
}

/// A Claude-vs-bots log from the first match (written before logs carried
/// `version` and `settings`): still builds, and still replays through the
/// current engine to the result it recorded.
#[test]
fn old_llm_match_logs_still_replay() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llm-match-deal.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    let games = replay::games_in(&v);
    assert_eq!(games.len(), 4, "two LLM seatings and two baseline seatings");
    for (label, g) in &games {
        let recorded: RoundResult = serde_json::from_value(g["result"].clone()).unwrap();
        assert_eq!(replay_actions(g), recorded, "{label}");
    }
    let reps = replay::build_from_text(&text).unwrap();
    assert_eq!(reps.len(), 4);
    let thoughts: usize = reps.iter().flat_map(|r| r["steps"].as_array().unwrap()).map(|s| s["thoughts"].as_array().map_or(0, |t| t.len())).sum();
    assert!(thoughts > 10, "model reasoning didn't attach to the steps ({thoughts})");
    // JSONL with one document per line works too.
    let jsonl = format!("{}\n{}\n", v["llm"][0], v["baseline"][1]);
    assert_eq!(replay::build_from_text(&jsonl).unwrap().len(), 2);
    assert!(replay::build_from_text("not json").is_err());
}

/// The JSON for cards and actions is the TS server's format, which the
/// logs, the viewer and tools/ts-diff all read. Pin it.
#[test]
fn card_and_action_json_is_stable() {
    let j = |v: &dyn erased::Ser| v.json();
    assert_eq!(j(&Card::normal(Suit::Jade, 5)), json!({"type": "normal", "suit": "jade", "rank": 5}));
    assert_eq!(j(&MAH_JONG), json!({"type": "special", "name": "mahjong"}));
    assert_eq!(j(&DOG), json!({"type": "special", "name": "dog"}));
    assert_eq!(j(&PHOENIX), json!({"type": "special", "name": "phoenix"}));
    assert_eq!(j(&DRAGON), json!({"type": "special", "name": "dragon"}));
    let c = Card::normal(Suit::Star, 14);
    let cj = j(&c);
    let pass = PassInfo { left: c, partner: DOG, right: MAH_JONG };
    let cases: Vec<(Action, Value)> = vec![
        (Action::GrandTichu { seat: 1, call: true }, json!({"type": "grandTichu", "seat": 1, "call": true})),
        (Action::SmallTichu { seat: 2 }, json!({"type": "smallTichu", "seat": 2})),
        (Action::PassCards { seat: 0, pass }, json!({"type": "passCards", "seat": 0, "pass": {"left": cj, "partner": j(&DOG), "right": j(&MAH_JONG)}})),
        (Action::Play { seat: 3, cards: vec![c, PHOENIX], phoenix_as: Some(14) }, json!({"type": "play", "seat": 3, "cards": [cj, j(&PHOENIX)], "phoenixAs": 14})),
        (Action::Play { seat: 3, cards: vec![c], phoenix_as: None }, json!({"type": "play", "seat": 3, "cards": [cj], "phoenixAs": null})),
        (Action::PassTurn { seat: 1 }, json!({"type": "passTurn", "seat": 1})),
        (Action::Bomb { seat: 0, cards: vec![c] }, json!({"type": "bomb", "seat": 0, "cards": [cj]})),
        (Action::Wish { seat: 0, rank: Some(9) }, json!({"type": "wish", "seat": 0, "rank": 9})),
        (Action::Wish { seat: 0, rank: None }, json!({"type": "wish", "seat": 0, "rank": null})),
        (Action::ResolveTrick, json!({"type": "resolveTrick"})),
        (Action::GiveDragon { seat: 2, to: 1 }, json!({"type": "giveDragon", "seat": 2, "to": 1})),
    ];
    for (a, want) in cases {
        assert_eq!(serde_json::to_value(&a).unwrap(), want, "{a:?}");
        assert_eq!(serde_json::from_value::<Action>(want).unwrap(), a);
    }
}

/// `dyn`-friendly serialization for the table above.
mod erased {
    pub trait Ser {
        fn json(&self) -> serde_json::Value;
    }
    impl<T: serde::Serialize> Ser for T {
        fn json(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// Feature encoding
// ---------------------------------------------------------------------------

/// Plays the first listed option and passes whenever allowed, so positions
/// depend only on the engine, not on any bot's tuning.
struct FirstChoice {
    hash: std::rc::Rc<std::cell::Cell<u64>>,
}

impl Bot for FirstChoice {
    fn name(&self) -> String {
        "first".into()
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        PassInfo { left: v.hand[0], partner: v.hand[1], right: v.hand[2] }
    }
    fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let mut bytes = vec![];
        encode_state(v, &mut bytes);
        let c = Candidates::new(v, plays, can_pass);
        bytes.extend_from_slice(&c.features);
        let h = self.hash.get();
        self.hash.set(fnv1a(&bytes) ^ h.rotate_left(5));
        let choice = if plays.is_empty() || (can_pass && v.played_cards.len() % 3 != 0) { None } else { Some(v.played_cards.len() % plays.len()) };
        assert!(c.option_of(choice).is_some());
        choice
    }
    fn bomb(&mut self, v: &PlayerView, _bombs: &[Combo]) -> Option<usize> {
        (v.played_cards.len() % 2 == 0).then_some(0)
    }
    fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
        Some(v.hand.len() as u8 % 13 + 2)
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        v.hand.iter().filter(|c| c.is_special()).count() >= 2
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        v.hand.contains(&DRAGON) && v.hand.contains(&PHOENIX)
    }
    fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
        v.opponents()[0]
    }
}

/// The feature bytes for a fixed set of positions. Trained models read
/// exactly this layout: if this test fails, the encoding changed, so bump
/// `FEATURE_VERSION` (old .tnet files then refuse to load instead of playing
/// garbage) and update the hash below.
#[test]
fn feature_encoding_is_pinned_to_its_version() {
    assert_eq!((FEATURE_VERSION, STATE_DIM, ACTION_DIM), (2, 704, 48), "dims changed: bump FEATURE_VERSION and update this test");
    let hash = std::rc::Rc::new(std::cell::Cell::new(0u64));
    let mut rng = StdRng::seed_from_u64(2024);
    for i in 0..60 {
        let mut g = GameState::new(Settings { clockwise: i % 2 == 1, ..Settings::default() });
        g.start_round(shuffled_deck(&mut rng));
        let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(FirstChoice { hash: hash.clone() }) as Box<dyn Bot>);
        play_dealt_round(&mut g, &mut bots, None).unwrap();
    }
    assert_eq!(
        hash.get(),
        FEATURE_HASH,
        "feature bytes changed (got {:#x}): if intended, bump FEATURE_VERSION and update FEATURE_HASH",
        hash.get()
    );
}

const FEATURE_HASH: u64 = 0x4f80_3057_2d4c_5ccc;

// ---------------------------------------------------------------------------
// Self-play records
// ---------------------------------------------------------------------------

struct Rec {
    round: u32,
    seat: u8,
    n: usize,
    label: usize,
    margin: f32,
    q: Option<Vec<f32>>,
    bytes: Vec<u8>,
}

/// Parses a record file the way train/tichu_data.py does.
fn parse_records(buf: &[u8]) -> Vec<Rec> {
    let u32_at = |p: usize| u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
    assert_eq!(&buf[0..4], b"TCHR");
    assert_eq!(
        [u32_at(4), u32_at(8), u32_at(12), u32_at(16)],
        [selfplay::FORMAT_VERSION, FEATURE_VERSION, STATE_DIM as u32, ACTION_DIM as u32]
    );
    let mut out = vec![];
    let mut p = 20;
    while p < buf.len() {
        let start = p;
        let round = u32_at(p);
        let seat = buf[p + 4];
        let flags = buf[p + 5];
        let n = u16::from_le_bytes([buf[p + 6], buf[p + 7]]) as usize;
        let label = u16::from_le_bytes([buf[p + 8], buf[p + 9]]) as usize;
        let margin = f32::from_le_bytes(buf[p + 10..p + 14].try_into().unwrap());
        p += 14 + STATE_DIM + n * ACTION_DIM;
        let q = (flags & 1 == 1).then(|| {
            let q = (0..n).map(|k| f32::from_le_bytes(buf[p + 4 * k..p + 4 * k + 4].try_into().unwrap())).collect();
            p += 4 * n;
            q
        });
        assert!(p <= buf.len(), "truncated record");
        out.push(Rec { round, seat, n, label, margin, q, bytes: buf[start..p].to_vec() });
    }
    out
}

fn selfplay_file(cfg: &SelfplayConfig) -> (Vec<u8>, usize) {
    let mut buf = vec![];
    let n = selfplay::run(cfg, &mut buf).unwrap();
    (buf, n)
}

#[test]
fn selfplay_records_parse_and_are_reproducible() {
    let cfg = SelfplayConfig {
        bot: "heuristic".into(),
        rounds: 8,
        seed: 3,
        threads: 1,
        q_samples: 0,
        rollout: "heuristic".into(),
        explore: 0.1,
    };
    let (one, n) = selfplay_file(&cfg);
    let recs = parse_records(&one);
    assert_eq!(recs.len(), n);
    assert!(n > 8 * 10, "only {n} records from 8 rounds");
    for r in &recs {
        assert!(r.seat < 4 && (r.round as usize) < cfg.rounds);
        assert!(r.n >= 2, "single-option decisions are skipped");
        assert!(r.label < r.n);
        assert!(r.q.is_none());
        assert!(r.margin.fract() == 0.0 && r.margin.abs() <= 800.0, "margin {}", r.margin);
    }
    // Teammates share a margin, opponents get its negation.
    for a in &recs {
        for b in recs.iter().filter(|b| b.round == a.round) {
            let want = if (a.seat + b.seat) % 2 == 0 { a.margin } else { -a.margin };
            assert_eq!(b.margin, want);
        }
    }
    // Threads only change the order rounds are written in.
    let (four, _) = selfplay_file(&SelfplayConfig { threads: 4, ..cfg.clone() });
    let by_round = |recs: Vec<Rec>| {
        let mut v: Vec<(u32, Vec<u8>)> = recs.into_iter().map(|r| (r.round, r.bytes)).collect();
        v.sort_by_key(|x| x.0);
        v
    };
    assert_eq!(by_round(recs), by_round(parse_records(&four)));
}

#[test]
fn selfplay_q_labels_are_written() {
    let cfg = SelfplayConfig {
        bot: "heuristic".into(),
        rounds: 2,
        seed: 1,
        threads: 2,
        q_samples: 2,
        rollout: "heuristic".into(),
        explore: 0.0,
    };
    let (buf, n) = selfplay_file(&cfg);
    let recs = parse_records(&buf);
    assert_eq!(recs.len(), n);
    for r in &recs {
        let q = r.q.as_ref().expect("--q records carry Q-values");
        assert_eq!(q.len(), r.n);
        assert!(q.iter().all(|x| x.is_finite()));
        // The labeler plays the best Q.
        let best = q.iter().cloned().fold(f32::MIN, f32::max);
        assert_eq!(q[r.label], best);
    }
}
