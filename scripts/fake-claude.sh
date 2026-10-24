#!/usr/bin/env bash
# Stand-in for `claude -p --output-format json`, for testing llm_match without
# making model calls:
#   cargo run --release --example llm_match -- --claude-bin scripts/fake-claude.sh --dir runs/fake --deals 2
# Answers 1 (or 1 2 3 for passes). With FAKE_CLAUDE_LIMIT_AT=N, call N reports
# a usage limit that resets 5 seconds later (counter in $FAKE_CLAUDE_STATE).
prompt=$(cat)
state=${FAKE_CLAUDE_STATE:-/tmp/fake-claude-count}
n=$(( $(cat "$state" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$state"
if [ -n "$FAKE_CLAUDE_LIMIT_AT" ] && [ "$n" -eq "$FAKE_CLAUDE_LIMIT_AT" ]; then
  printf '{"type":"result","subtype":"success","is_error":true,"result":"Claude AI usage limit reached|%s"}\n' $(( $(date +%s) + 5 ))
  exit 1
fi
case "$prompt" in
  *"three different numbers"*) answer="1 2 3" ;;
  *) answer="1" ;;
esac
printf '{"type":"result","subtype":"success","is_error":false,"result":"Fake reasoning.\\nANSWER: %s","total_cost_usd":0.001}\n' "$answer"
