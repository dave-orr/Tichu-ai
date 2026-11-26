//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::features::{ACTION_DIM, FEATURE_VERSION, STATE_DIM};
use tichu_engine::humanlog::RoundLog;
use tichu_engine::sim::{play_dealt_round, play_round};
use tichu_engine::deck::shuffled_deck;
use tichu_engine::scoring::RoundResult;
use tichu_engine::types::*;

/// A small random model in the `.tnet` format (see `src/bots/net.rs`),
/// written once per test binary. Returns its path.
pub fn random_tnet() -> &'static str {
    static PATH: OnceLock<String> = OnceLock::new();
    PATH.get_or_init(|| {
        let path = temp_path("random.tnet");
        std::fs::write(&path, tnet_bytes(5, true)).unwrap();
        path.to_str().unwrap().to_string()
    })
}

/// A `.tnet` file with a 16-wide trunk and 8-wide heads, random weights.
pub fn tnet_bytes(seed: u64, with_weights: bool) -> Vec<u8> {
    tnet_with_header(seed, with_weights, |_| {})
}

pub fn tnet_with_header(seed: u64, with_weights: bool, edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let (h, k) = (16, 8);
    let shapes = [vec![[h, STATE_DIM], [h, h]], vec![[k, h], [1, k]], vec![[k, h + ACTION_DIM], [1, k]]];
    let mut header = serde_json::json!({
        "feature_version": FEATURE_VERSION, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
        "trunk": shapes[0], "value": shapes[1], "policy": shapes[2], "value_scale": 100.0,
    });
    edit(&mut header);
    let hb = serde_json::to_vec(&header).unwrap();
    let mut file = b"TNET".to_vec();
    file.extend_from_slice(&(hb.len() as u32).to_le_bytes());
    file.extend_from_slice(&hb);
    if with_weights {
        for group in &shapes {
            for &[out, inp] in group {
                for _ in 0..out * inp + out {
                    file.extend_from_slice(&rng.gen_range(-0.3f32..0.3).to_le_bytes());
                }
            }
        }
    }
    file
}

/// A fresh path under the system temp dir, unique to this process.
pub fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tichu-test-{}-{name}", std::process::id()))
}

/// Every kind of bundled bot, in the cheapest configuration that still
/// exercises its code paths. Learned bots use `random_tnet()`.
pub fn bot_specs() -> Vec<String> {
    let mut v: Vec<String> = [
        "random",
        "heuristic",
        "h2",
        "h2~fight_pts=5,danger=3,bomb_mode=0",
        "pimc2",
        "pimc2@0.3",
        "pimc2/h",
        "pimc2/hibs",
        "pimc2/i7",
        "oracle1",
        "eg:heuristic",
        "eg:h2",
        "calls:v1:h2",
        "calls:0:heuristic",
        "calls:s-10:h2",
        "calls:g:random",
        "nocalls:random",
        "pass2:h2",
        "pass2~o:heuristic",
        "pass2~mpdx:random",
        "pass2:pimc2",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    v.push(format!("net:{}", random_tnet()));
    v.push(format!("vpimc2:{}", random_tnet()));
    v.push(format!("label2/p3:net:{}", random_tnet()));
    v.push(format!("search:{}", random_tnet()));
    v
}

pub fn bot(spec: &str, seed: u64) -> Box<dyn Bot> {
    bots::by_name(spec, seed).unwrap_or_else(|| panic!("by_name rejected '{spec}'"))
}

/// Play one dealt round, logging every action. Seats 0/2 get `a`, 1/3 `b`.
pub fn logged_round(a: &str, b: &str, seed: u64, clockwise: bool) -> (Vec<Card>, Vec<Action>, RoundResult) {
    let mut bots: [Box<dyn Bot>; 4] = [0u64, 1, 2, 3].map(|s| bot(if s % 2 == 0 { a } else { b }, seed * 4 + s));
    let mut state = GameState::new(Settings { clockwise, ..Settings::default() });
    let deck = shuffled_deck(&mut StdRng::seed_from_u64(seed));
    state.start_round(deck.clone());
    let mut actions = Vec::new();
    let mut obs = |_: &GameState, a: &Action| actions.push(a.clone());
    let r = play_dealt_round(&mut state, &mut bots, Some(&mut obs))
        .unwrap_or_else(|e| panic!("{a} vs {b}, seed {seed}: {e}"));
    (deck, actions, r)
}

// ----- Server round logs (dave-orr/tichu's RoundLog format) -----

/// Play one round and return it as a server round log, plus the true final state.
pub fn server_round_log(bot: &str, seed: u64, clockwise: bool) -> (RoundLog, GameState) {
    let (log, g) = server_round_log_json(bot, seed, clockwise, false);
    (serde_json::from_value(log).unwrap(), g)
}

/// The same, as JSON; with `actions`, in the newer format that also logs
/// every action (`RoundLog.actions`) and the table direction.
pub fn server_round_log_json(bot: &str, seed: u64, clockwise: bool, with_actions: bool) -> (serde_json::Value, GameState) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seats: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| bots::by_name(bot, seed * 4 + k).unwrap());
    let mut g = GameState::new(Settings { clockwise, ..Settings::default() });
    let mut plays = vec![];
    let mut wishes = vec![];
    let mut dragons = vec![];
    let mut actions = vec![];
    let mut trick: i32 = -1;
    let mut obs = |s: &GameState, a: &Action| {
        let at = 1_700_000_000_000u64 + actions.len() as u64;
        match a {
            Action::Play { seat, cards, phoenix_as } => {
                let mut v = serde_json::json!({"type": "play", "at": at, "seat": seat, "cards": cards, "trick": trick + s.trick_plays.is_empty() as i32});
                if let Some(r) = phoenix_as {
                    v["phoenixAs"] = serde_json::json!(r);
                }
                actions.push(v);
            }
            Action::Bomb { seat, cards } => actions.push(serde_json::json!({"type": "bomb", "at": at, "seat": seat, "cards": cards, "trick": trick})),
            Action::PassTurn { seat } => actions.push(serde_json::json!({"type": "pass", "at": at, "seat": seat})),
            Action::SmallTichu { seat } if s.phase == Phase::Playing => actions.push(serde_json::json!({"type": "smallTichu", "at": at, "seat": seat})),
            Action::Wish { seat, rank } => actions.push(serde_json::json!({"type": "wish", "at": at, "seat": seat, "rank": rank})),
            Action::GiveDragon { seat, to } => actions.push(serde_json::json!({"type": "dragon", "at": at, "seat": seat, "to": to})),
            _ => {}
        }
        match a {
        Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } => {
            if s.trick_plays.is_empty() {
                trick += 1;
            }
            plays.push(serde_json::json!({"seat": seat, "cards": cards, "trick": trick}));
        }
        Action::Wish { seat, rank } => {
            if rank.is_some() {
                wishes.push(serde_json::json!({"seat": seat, "rank": rank}));
            }
        }
        Action::GiveDragon { seat, to } => dragons.push(serde_json::json!({"fromSeat": seat, "toSeat": to})),
        _ => {}
        }
    };
    play_round(&mut g, &mut seats, &mut rng, Some(&mut obs)).unwrap();
    let res = g.last_result().unwrap().clone();
    let players: Vec<_> = (0..4)
        .map(|s| {
            let p = &g.players[s];
            let pass = p.pass.unwrap();
            serde_json::json!({
                "seat": s, "team": s % 2, "tichuCall": p.tichu, "outOrder": p.out_order,
                "initialHand": g.deck[s * 14..s * 14 + 14],
                "passedLeft": pass.left, "passedPartner": pass.partner, "passedRight": pass.right,
            })
        })
        .collect();
    let mut log = serde_json::json!({
        "gameId": 0, "roundNumber": 1, "scoresBeforeRound": [0, 0],
        "roundCardPoints": res.card_points, "outOrder": res.out_order,
        "players": players, "plays": plays, "mahJongWishes": wishes, "dragonGiveaways": dragons,
    });
    if with_actions {
        log["actions"] = serde_json::json!(actions);
        log["clockwise"] = serde_json::json!(clockwise);
    }
    (log, g)
}
