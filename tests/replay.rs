//! Game logs and the replay builder.

use rand::rngs::StdRng;
use rand::SeedableRng;

use tichu_engine::arena::{self, ArenaConfig};
use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::replay;
use tichu_engine::sim::play_dealt_round;
use tichu_engine::types::*;

fn logged_round(seed: u64) -> serde_json::Value {
    let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|s| bots::by_name("heuristic", seed + s).unwrap());
    let settings = Settings::default();
    let mut state = GameState::new(settings);
    let deck = shuffled_deck(&mut StdRng::seed_from_u64(seed));
    state.start_round(deck.clone());
    let mut events = Vec::new();
    let mut obs = |_: &GameState, a: &Action| events.push(replay::action_event(a));
    let r = play_dealt_round(&mut state, &mut bots, Some(&mut obs)).unwrap();
    replay::game_log(seed as usize, false, &["heuristic"; 4], &settings, &deck, &r, events)
}

#[test]
fn replay_steps_follow_the_engine() {
    for seed in 0..20 {
        let log = logged_round(seed);
        let rep = replay::build(&log, "").unwrap();
        let steps = rep["steps"].as_array().unwrap();
        assert_eq!(steps[0]["kind"], "deal");
        assert_eq!(steps.last().unwrap()["kind"], "result");
        // Hands only shrink once play starts, by exactly the cards played.
        let total = |s: &serde_json::Value| -> usize { s["hands"].as_array().unwrap().iter().map(|h| h.as_array().unwrap().len()).sum() };
        assert_eq!(total(&steps[0]), 56);
        for w in steps.windows(2) {
            let played = w[1]["cards"].as_array().map_or(0, |c| c.len());
            if w[1]["kind"] == "play" || w[1]["kind"] == "bomb" {
                assert_eq!(total(&w[0]) - total(&w[1]), played, "{}", w[1]["text"]);
            } else if w[1]["kind"] != "result" {
                assert_eq!(total(&w[0]), total(&w[1]), "{}", w[1]["text"]);
            }
        }
        // Every action in the log is accounted for.
        let plays = log["events"].as_array().unwrap().iter().filter(|e| e["action"]["type"] == "play").count();
        assert_eq!(steps.iter().filter(|s| s["kind"] == "play").count(), plays);
    }
}

#[test]
fn model_reasoning_attaches_to_the_next_step() {
    let mut log = logged_round(7);
    let events = log["events"].as_array_mut().unwrap();
    let i = events.iter().position(|e| e["action"]["type"] == "play").unwrap();
    let seat = events[i]["action"]["seat"].clone();
    events.insert(
        i,
        serde_json::json!({ "type": "llm", "seat": seat, "kind": "play", "answer": [2],
            "prompt": "STATE\n\nDECISION\nChoose:\n0: Pass\n1: Single: 2j\n2: Pair of 3s: 3j 3s\n",
            "reply": "Pairs first.\n\nANSWER: 2", "cached": false }),
    );
    let rep = replay::build(&log, "").unwrap();
    let step = rep["steps"].as_array().unwrap().iter().find(|s| s["kind"] == "play").unwrap();
    let t = &step["thoughts"][0];
    assert_eq!(t["chose"][0], "Pair of 3s: 3j 3s");
    assert_eq!(t["seat"], seat);
}

#[test]
fn arena_logs_replay_and_find_both_seatings() {
    let dir = std::env::temp_dir().join(format!("tichu-replay-test-{}", std::process::id()));
    let cfg = ArenaConfig { count: 2, threads: 1, log_dir: Some(dir.clone()), log_deals: 1, ..Default::default() };
    let f = |s: u64| bots::by_name("heuristic", s).unwrap();
    let r = arena::run("heuristic", &f, "heuristic", &f, &cfg);
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    assert!(!dir.join("deal-0001.json").exists());
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("deal-0000.json")).unwrap()).unwrap();
    let reps = replay::build_all(&v).unwrap();
    assert_eq!(reps.len(), 2);
    assert_eq!(reps[1]["title"], "deal 0 · games · seating B");
    let page = replay::html(&reps);
    assert!(!page.contains("/*REPLAY_DATA*/") && !page.contains("/*CARD_ART*/"));
    let _ = std::fs::remove_dir_all(dir);
}
