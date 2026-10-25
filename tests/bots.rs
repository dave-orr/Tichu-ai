//! Every bundled bot, in every configuration the CLI accepts, plays whole
//! rounds with only legal moves, and is deterministic given its seed (the
//! arena's reproducibility and the LLM match's paired baseline rely on it).

mod common;

use tichu_engine::bots;

/// Each bot spec against a random and a heuristic table, both directions.
#[test]
fn every_bot_plays_legal_rounds() {
    for spec in common::bot_specs() {
        for (i, other) in ["random", "heuristic"].iter().enumerate() {
            for seed in 0..3u64 {
                let clockwise = seed == 1;
                // The simulator rejects any illegal choice with an error,
                // which logged_round turns into a panic naming the bot.
                common::logged_round(&spec, other, 100 + seed * 2 + i as u64, clockwise);
                common::logged_round(other, &spec, 200 + seed * 2 + i as u64, clockwise);
            }
        }
    }
}

#[test]
fn bots_are_deterministic_given_their_seed() {
    for spec in common::bot_specs() {
        for seed in [3u64, 11] {
            let a = common::logged_round(&spec, "random", seed, false);
            let b = common::logged_round(&spec, "random", seed, false);
            assert_eq!(a.1, b.1, "{spec}: two runs with seed {seed} played differently");
        }
    }
}

/// The examples the docs and `tichu bots` advertise all parse.
#[test]
fn advertised_names_parse() {
    let examples = [
        "random", "heuristic", "h2", "h2~shed=2", "pimc20", "pimc20@0.1", "pimc10/hibs", "pimc5/i7", "oracle1",
        "oracle20@0.1", "eg:heuristic", "eg:pimc20", "calls:1:pimc20", "calls:none:h2", "calls:v1:h2", "calls:s0:h2",
        "calls:g:h2", "nocalls:h2", "eg:calls:v1:h2", "pass2:h2", "pass2~o:h2", "calls:v1:pass2:h2",
    ];
    for name in examples {
        assert!(bots::by_name(name, 0).is_some(), "by_name rejected advertised name '{name}'");
    }
    // Every NAMES entry without a placeholder is a working name.
    for (name, _) in bots::NAMES {
        if !name.contains(|c: char| c.is_ascii_uppercase() || c == '[') {
            assert!(bots::by_name(name, 0).is_some(), "NAMES lists '{name}' but by_name rejects it");
        }
    }
}

/// Malformed names are rejected with None (the CLI then lists the bots)
/// rather than panicking or silently building some other bot.
#[test]
fn malformed_names_are_rejected() {
    let bad = [
        "", "nope", "pimc", "pimcx", "pimc2@", "pimc2@1.5", "pimc2/z", "pimc2/hz", "oracle1/q", "oraclex",
        "h2~", "h2~nope=1", "h2~shed", "h2~shed=x", "calls:h2", "calls:zz:h2", "calls:1:nope", "nocalls:nope",
        "eg:", "eg:nope", "net:/no/such/model.tnet", "vpimc2:/no/such/model.tnet", "vpimcx:/x", "random2", "pass2", "pass2:", "pass2:nope", "pass2~q:h2", "pass2z:h2",
    ];
    for name in bad {
        // "h2~" (empty params) is fine to accept; everything else must fail.
        if name == "h2~" {
            continue;
        }
        assert!(bots::by_name(name, 0).is_none(), "by_name accepted malformed name '{name}'");
    }
}

/// Learned bots refuse models they can't run instead of panicking.
#[test]
fn net_rejects_bad_models() {
    use tichu_engine::bots::net::Net;
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("bad-magic", b"NOPE\x02\x00\x00\x00{}".to_vec()),
        ("header-past-end", {
            let mut f = b"TNET".to_vec();
            f.extend_from_slice(&1000u32.to_le_bytes());
            f.extend_from_slice(b"{}");
            f
        }),
        ("no-weights", common::tnet_bytes(1, false)),
        ("truncated-weights", {
            let f = common::tnet_bytes(1, true);
            f[..f.len() - 4].to_vec()
        }),
        ("old-features", common::tnet_with_header(1, true, |h| h["feature_version"] = (tichu_engine::features::FEATURE_VERSION + 1).into())),
        ("wrong-chain", common::tnet_with_header(1, true, |h| h["value"] = serde_json::json!([[8, 16], [2, 8]]))),
        ("failed-check", common::tnet_with_header(1, true, |h| {
            h["check"] = serde_json::json!({ "state": vec![0u8; tichu_engine::features::STATE_DIM],
                "actions": vec![0u8; tichu_engine::features::ACTION_DIM], "value": 12345.0, "logits": [0.0] })
        })),
    ];
    for (name, bytes) in cases {
        let path = common::temp_path(&format!("{name}.tnet"));
        std::fs::write(&path, &bytes).unwrap();
        let p = path.to_str().unwrap().to_string();
        let r = std::panic::catch_unwind(|| Net::load(&p).err());
        std::fs::remove_file(&path).ok();
        match r {
            Ok(Some(_)) => {}
            Ok(None) => panic!("{name}: a bad model loaded"),
            Err(_) => panic!("{name}: Net::load panicked instead of returning an error"),
        }
    }
    assert!(Net::load(common::random_tnet()).is_ok());
}
