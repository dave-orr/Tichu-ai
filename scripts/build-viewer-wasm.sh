#!/usr/bin/env bash
# Rebuilds viewer/replay.wasm: the replay builder (src/replay.rs and the
# engine under it) for the browser, embedded by `tichu replay` so the page
# can open raw game logs. Run after changing the replay builder or the rules.
set -euo pipefail
cd "$(dirname "$0")/../viewer/wasm"
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/tichu_replay_wasm.wasm ../replay.wasm
ls -l ../replay.wasm
