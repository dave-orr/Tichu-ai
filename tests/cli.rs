//! Smoke tests for the `tichu` command line: each subcommand runs, exits 0,
//! and writes output the next tool in the chain can read.

mod common;

use std::process::{Command, Output};

fn tichu(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_tichu")).args(args).output().expect("running tichu");
    out
}

fn ok(args: &[&str]) -> String {
    let out = tichu(args);
    assert!(out.status.success(), "tichu {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn bots_lists_every_name() {
    let out = tichu(&["bots"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    for (name, _) in tichu_engine::bots::NAMES {
        assert!(text.contains(name), "`tichu bots` doesn't list {name}");
    }
}

#[test]
fn arena_reports_a_margin() {
    let out = ok(&["arena", "heuristic", "random", "--rounds", "20", "--seed", "4", "--threads", "2"]);
    assert!(out.contains("heuristic vs random  (20 deals)"), "{out}");
    assert!(out.contains("margin per deal"), "{out}");
    // Same seed, same report (minus the timing line).
    let again = ok(&["arena", "heuristic", "random", "--rounds", "20", "--seed", "4", "--threads", "1"]);
    let strip = |s: &str| s.lines().filter(|l| !l.trim_start().starts_with('(')).collect::<Vec<_>>().join("\n");
    assert_eq!(strip(&out), strip(&again));
}

#[test]
fn bad_arguments_exit_2() {
    assert_eq!(tichu(&["arena", "heuristic", "nosuchbot", "--rounds", "1"]).status.code(), Some(2));
    assert_eq!(tichu(&["frobnicate"]).status.code(), Some(2));
    assert_eq!(tichu(&["arena", "heuristic", "random", "--rounds"]).status.code(), Some(2));
}

#[test]
fn play_narrates_a_round() {
    let out = ok(&["play", "h2", "random", "--seed", "3"]);
    assert!(out.lines().count() > 20, "{out}");
}

#[test]
fn arena_logs_feed_the_replay_viewer() {
    let dir = common::temp_path("cli-logs");
    let d = dir.to_str().unwrap();
    ok(&["arena", "h2", "random", "--rounds", "3", "--log", d, "--log-deals", "2", "--threads", "1"]);
    let log = dir.join("deal-0001.json");
    assert!(log.exists() && !dir.join("deal-0002.json").exists());
    let json = ok(&["replay", log.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v.as_array().unwrap().len(), 2, "both seatings");
    let html = dir.join("replay.html");
    let out = tichu(&["replay", dir.join("deal-0000.json").to_str().unwrap(), log.to_str().unwrap(), "--out", html.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let page = std::fs::read_to_string(&html).unwrap();
    assert!(page.starts_with("<!doctype html>") && !page.contains("/*REPLAY_DATA*/"));
    assert!(!page.contains("/*REPLAY_WASM*/"), "viewer wasm not embedded");
    // A file with no game logs is flagged; one that isn't JSON is an error.
    let junk = dir.join("junk.json");
    std::fs::write(&junk, "{\"hello\": 1}").unwrap();
    let out = tichu(&["replay", junk.to_str().unwrap(), "--json"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("no game logs found"));
    std::fs::write(&junk, "hello").unwrap();
    assert!(!tichu(&["replay", junk.to_str().unwrap(), "--json"]).status.success());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn selfplay_writes_a_record_file() {
    let path = common::temp_path("cli.bin");
    ok(&["selfplay", "--out", path.to_str().unwrap(), "--rounds", "3", "--threads", "1"]);
    let buf = std::fs::read(&path).unwrap();
    assert_eq!(&buf[..4], b"TCHR");
    assert!(buf.len() > 20 + 10 * tichu_engine::features::STATE_DIM);
    std::fs::remove_file(path).ok();
}

#[test]
fn trace_emits_one_json_round_per_line() {
    let out = ok(&["trace", "--rounds", "3", "--seed", "5", "--bot", "heuristic"]);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3);
    for l in lines {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        assert!(v.is_object());
    }
}
