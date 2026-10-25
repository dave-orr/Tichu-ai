#!/usr/bin/env bash
# Checks that the committed viewer/replay.wasm is up to date: builds some
# game logs (arena logs, plus the old LLM-match fixture) and compares its
# replays with `tichu replay --json`. Needs Node.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -q
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
./target/release/tichu arena random h2 --rounds 6 --seed 9 --log "$dir" --log-deals 6 >/dev/null
./target/release/tichu arena heuristic random --rounds 4 --seed 3 --clockwise --log "$dir/cw" --log-deals 4 >/dev/null
node tools/viewer-check/check.mjs viewer/replay.wasm target/release/tichu "$dir"/deal-*.json "$dir"/cw/deal-*.json tests/fixtures/llm-match-deal.json
