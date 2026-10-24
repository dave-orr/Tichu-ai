#!/usr/bin/env bash
# Check the Rust engine against the production TypeScript engine: play random
# and heuristic rounds in Rust, replay each through the TS engine, and compare
# the state and the full set of legal moves at every step.
#
#   scripts/ts-diff.sh [ROUNDS] [SEED]      (default 100 rounds, random seed)
#
# Needs Node 22 and a checkout of dave-orr/tichu at $TICHU_REPO (default
# ../tichu). Builds its shared/ package on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
TICHU_REPO=$(cd "${TICHU_REPO:-../tichu}" && pwd)
ROUNDS=${1:-100}
SEED=${2:-$RANDOM}
SHARED="$TICHU_REPO/shared/dist/index.js"

if [ ! -f "$SHARED" ]; then
  echo "building $TICHU_REPO/shared ..."
  (cd "$TICHU_REPO" && npm ci --ignore-scripts --engine-strict=false >/dev/null && npm run build --workspace=shared >/dev/null)
fi
cargo build --release -q
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
status=0
for bot in random heuristic; do
  echo "== $bot bots, $ROUNDS rounds, seed $SEED"
  ./target/release/tichu trace --rounds "$ROUNDS" --seed "$SEED" --bot "$bot" > "$tmp"
  TICHU_SHARED="$SHARED" node tools/ts-diff/diff.mjs "$tmp" || status=1
done
exit $status
