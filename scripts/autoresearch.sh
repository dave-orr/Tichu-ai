#!/bin/bash
set -e

# Autoresearch loop for Tichu bot optimization.
# Uses Claude Code to iteratively improve the ISMCTS bot.

MAX_ITERATIONS=${1:-50}
GAMES_PER_EVAL=100
EVAL_ITERATIONS=50
WIN_RATE_THRESHOLD=0.53

echo "=== Tichu Autoresearch ==="
echo "Max iterations: $MAX_ITERATIONS"
echo "Games per eval: $GAMES_PER_EVAL"
echo "MCTS iterations: $EVAL_ITERATIONS"
echo ""

# Initialize baseline if needed.
if [ ! -f baseline_weights.json ] || [ ! -f baseline_config.json ]; then
    echo "Initializing baseline..."
    cargo build --release
    cargo run --release -- autoresearch --games "$GAMES_PER_EVAL" --iterations "$EVAL_ITERATIONS"
    echo ""
fi

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "========================================="
    echo "Iteration $i / $MAX_ITERATIONS"
    echo "========================================="

    # Ask Claude to make ONE improvement.
    claude --print --dangerously-skip-permissions \
        "Read CLAUDE.md for full context. You are optimizing the Tichu ISMCTS bot.

Make ONE specific, targeted improvement to the bot code in src/bots/heuristic/ or src/bots/ismcts/.

Look at the current code, identify the weakest area, and improve it.
Do NOT modify files outside src/bots/.
After making changes, run: cargo build --release
If the build fails, fix it before finishing."

    # Check if it compiles.
    if ! cargo build --release 2>/dev/null; then
        echo "BUILD FAILED — reverting"
        git checkout -- src/bots/heuristic/ src/bots/ismcts/
        continue
    fi

    # Run evaluation: current vs baseline.
    echo "Evaluating..."
    RESULT=$(cargo run --release -- autoresearch --games "$GAMES_PER_EVAL" --iterations "$EVAL_ITERATIONS" 2>/dev/null)
    echo "$RESULT"

    WIN_RATE=$(echo "$RESULT" | grep "^win_rate_a=" | cut -d= -f2)
    SIGNIFICANT=$(echo "$RESULT" | grep "^significant=" | cut -d= -f2)

    echo "Win rate: $WIN_RATE, Significant: $SIGNIFICANT"

    if [ "$SIGNIFICANT" = "true" ] && [ "$(echo "$WIN_RATE > $WIN_RATE_THRESHOLD" | bc -l)" -eq 1 ]; then
        echo "IMPROVEMENT FOUND — committing"
        git add src/bots/heuristic/ src/bots/ismcts/
        git commit -m "autoresearch #$i: win_rate=$WIN_RATE"

        # Update baseline to new version.
        cargo run --release -- autoresearch --games 1 --iterations 1 2>/dev/null || true
        # Force-refresh baseline files from current defaults.
        rm -f baseline_weights.json baseline_config.json
        cargo run --release -- autoresearch --games "$GAMES_PER_EVAL" --iterations "$EVAL_ITERATIONS" 2>/dev/null
        echo "Baseline updated."
    else
        echo "No significant improvement — reverting"
        git checkout -- src/bots/heuristic/ src/bots/ismcts/
    fi

    echo ""
done

echo "=== Autoresearch complete ==="
