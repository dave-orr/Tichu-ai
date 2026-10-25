#!/usr/bin/env bash
# End-to-end check of the Claude-vs-bots runner through the real `claude -p`
# code path, with scripts/fake-claude.sh standing in for claude: plays a
# deal, hits a usage limit mid-deal with --on-limit exit (exit code 75),
# resumes, and checks the finished deals are kept and the report prints.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -q --example llm_match
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
export FAKE_CLAUDE_STATE="$dir/count"
run() {
  ./target/release/examples/llm_match --claude-bin scripts/fake-claude.sh --dir "$dir/run" \
    --opponent heuristic --fallback heuristic --no-baseline --on-limit exit "$@"
}

run --deals 1 >/dev/null
[ "$(wc -l < "$dir/run/results.jsonl")" -eq 1 ] || { echo "deal 0 not recorded"; exit 1; }
first=$(head -1 "$dir/run/results.jsonl")

# A usage limit a few calls into deal 1 stops the run with code 75.
export FAKE_CLAUDE_LIMIT_AT=$(( $(cat "$FAKE_CLAUDE_STATE") + 5 ))
set +e
run --deals 2 >/dev/null 2>&1
code=$?
set -e
[ "$code" -eq 75 ] || { echo "expected exit 75 on a usage limit, got $code"; exit 1; }
[ "$(wc -l < "$dir/run/results.jsonl")" -eq 1 ] || { echo "unfinished deal 1 was recorded"; exit 1; }

# Resuming finishes deal 1 and leaves deal 0 alone.
unset FAKE_CLAUDE_LIMIT_AT
run --deals 2 >/dev/null
[ "$(wc -l < "$dir/run/results.jsonl")" -eq 2 ] || { echo "resume didn't finish deal 1"; exit 1; }
[ "$(head -1 "$dir/run/results.jsonl")" = "$first" ] || { echo "resume changed deal 0"; exit 1; }
[ -f "$dir/run/games/deal-0001.json" ] || { echo "no game log for deal 1"; exit 1; }
./target/release/tichu replay "$dir/run/games/deal-0001.json" --json >/dev/null
run --report | grep -q "margin per deal" || { echo "report missing"; exit 1; }
# Changing a fixed setting for the directory is refused.
if ./target/release/examples/llm_match --claude-bin scripts/fake-claude.sh --dir "$dir/run" \
    --opponent random --fallback heuristic --no-baseline --deals 3 >/dev/null 2>&1; then
  echo "a changed opponent was accepted for an existing match directory"; exit 1
fi
echo "llm_match: play, usage-limit exit, resume and report all work"
