# Claude vs the bots

`examples/llm_match.rs` seats Claude at two seats (one team) and a bot at the
other two, and plays duplicate deals, like `tichu arena`. Each Claude
decision is a separate `claude -p` call (headless Claude Code). It runs on
whatever account the CLI is logged in with, so on a Max plan it uses your
subscription's usage allowance, not API billing.

The run is built to be stopped and restarted. Expect it to take days for a
meaningful sample, pausing whenever you hit a usage limit.

## What Claude decides

- Every play on its turn (when there's more than one legal option), every
  out-of-turn bomb chance, the three-card pass, the Mah Jong wish and the
  Dragon recipient.
- Grand and Small Tichu calls are **not** Claude's. The fallback bot makes
  them with the frozen v2 call rules, so the match measures play only.
- Claude sees exactly what a player sees (the `PlayerView`): its hand, every
  player's card count, calls, captured points and finishing position, what it
  passed and received, the whole round so far trick by trick, the table, the
  open wish, and the unseen cards. Options come as a numbered list of the
  legal plays and Claude answers with a number, so it can't make an illegal
  move. Plays that differ only in suits are listed once (suits only matter for
  straight-flush bombs; the version kept is the one that leaves the most bombs
  in hand).
- Each call starts fresh. No memory carries between decisions, but the
  round history in every prompt covers everything that happened.
- If Claude's reply has no usable `ANSWER:` line, it's asked once more, and
  then the fallback bot decides. The report counts these fallbacks.

`--show-prompt` prints the system prompt and sample questions without making
any calls.

## The numbers you get

Deal `i` with `--seed S` is the same deal `tichu arena` plays, with the same
bot seeds. Each deal is played twice (Claude's team on each side of the
cards). Unless you pass `--no-baseline`, each deal is also played a third
way: the fallback bot in Claude's seats, against the same opponent with the
same seeds. That's free (no calls), and it makes the paired comparison
possible:

```
llm[claude-cli:opus:default] vs pimc20  (40 deals, duplicate, seed 1)
  margin per deal (llm − pimc20): ...
  baseline, same deals (h2 − pimc20): ...
  llm − baseline (paired): ...          <- the number to watch
```

As with every arena result, quote the ± and the number of deals. For scale:
bot-vs-bot margins have a standard deviation around 80 points per deal, so
40 deals gives roughly ±25 and 100 deals roughly ±16. The paired line should
be tighter than that, since both sides play the same cards.

## Usage

Each deal asks about 50 questions: 2 Claude seats, about 12 decisions each
per round, and 2 seatings. Prompts are short (about 1,500 characters on
average, up to about 6,000 when leading with a big hand). Most of the cost is
Claude's own reasoning, so `--model` and `--effort` matter most. Rough
timing is 20 to 50 minutes per deal with `--jobs 1`, depending on model and
effort.

The calls count against the same usage limits as your interactive Claude use
(the rolling session window and the weekly cap). Opus uses the allowance
faster than Sonnet.

## Setup on your computer

1. Install Rust (https://rustup.rs) and get the branch:

   ```sh
   git clone https://github.com/dave-orr/Tichu-ai.git && cd Tichu-ai
   git checkout claude/project-thread-v88zno
   cargo test --release
   ```

2. Check that Claude Code is installed and logged in with your Max account:
   run `claude`, then `/status`. It should name your subscription, not an API
   key. If `ANTHROPIC_API_KEY` is set in your shell, the runner removes it
   from the `claude` calls, so they still use the subscription. Pass
   `--keep-api-key` only if you want API billing.

3. Try it without using anything:

   ```sh
   cargo run --release --example llm_match -- --show-prompt
   cargo run --release --example llm_match -- --backend stub --deals 3 --dir runs/stub
   ```

   The stub plays semi-random answers, so its margin is meaningless. It only
   checks that everything runs.

4. Optional: one real call, to confirm the CLI works headless:
   `echo "Say hi" | claude -p --model opus`.

## Running the match

```sh
caffeinate -i cargo run --release --example llm_match -- --deals 40 --opponent pimc20
```

(`caffeinate -i` keeps a Mac awake. On Linux, leave it out or use
`systemd-inhibit`. Running it inside `tmux` or `screen` lets you close the
terminal.)

Progress prints after every deal: the deal's margin, the running mean ± CI,
and how many questions it took. When the run ends it prints the full report.

### Stopping, limits, resuming

- **Usage limit.** By default the runner sleeps until the limit resets, then
  carries on by itself. If the message has no reset time, it retries every
  20 minutes (`--limit-poll-min`). With `--on-limit exit` it stops instead,
  with exit code 75.
- **Ctrl-C, sleep, crash.** Every answer is saved the moment it arrives
  (`answers.jsonl`) and every finished deal is saved too (`results.jsonl`).
  Rerun the **same command** to resume. Finished deals are skipped. An
  unfinished deal is replayed from the saved answers (the bots are seeded,
  so the game goes the same way), so no call is ever repeated.
- **More deals.** Rerun with a bigger `--deals`.
- **Report only.** Run the same command with `--report` added.
- **Replay a deal.** Add `--replay N`: it re-runs deal N's two games through
  the engine from the log and narrates them trick by trick: dealt hands, the
  passes, every play, and at each Claude decision the options and its reply.
  `--full` prints Claude's whole reasoning (default: its last paragraphs),
  `--hands` shows all four hands before each play, and `--baseline` narrates
  the fallback bot's games of the same deal instead.
- **Watch a deal.** `tichu replay DIR/games/deal-0000.json --out deal0.html`
  builds a visual replay of all of that deal's games: the four hands on a
  card table, step by step or auto-play, with Claude's reasoning beside each
  of its decisions (see "Watching a game" in the README).
- **Settings are locked per directory.** The opponent, fallback, seed, model,
  effort and decision set are stored in `config.json`. A rerun with different
  ones is refused, so a match never mixes setups. Use `--dir` for a second
  match (for example `--model sonnet --dir runs/sonnet-vs-pimc20`).
- **Don't pull new bot code mid-match.** The config stores bot names, not
  code. If `pimc20` changes between runs, the deals no longer compare.
- If `claude` can't be found, or three questions in a row get no reply at all
  (not logged in, say), the runner stops instead of quietly playing the
  fallback bot.

### Options

| | |
|---|---|
| `--deals N` | total deals wanted (default 20) |
| `--opponent BOT` | any `tichu bots` name (default `pimc20`) |
| `--fallback BOT` | makes the calls, covers unusable answers, plays the baseline (default `h2`) |
| `--seed S` | deal seed (default 1) |
| `--jobs N` | deals played at once (default 1). More finishes faster within a usage window but doesn't stretch the window |
| `--model M`, `--effort E` | passed to `claude` (default `opus`, CLI-default effort) |
| `--decisions LIST` | `all`, or a subset of `play,bomb,pass,wish,dragon`; the rest go to the fallback bot |
| `--dir DIR` | match directory (default `runs/llm-vs-OPPONENT-sSEED`) |
| `--claude-bin`, `--claude-arg`, `--timeout-s` | how `claude` is run |

### Files in the match directory

| | |
|---|---|
| `config.json` | the locked settings |
| `results.jsonl` | one line per finished deal: margins, both seatings' full round results, the baseline, question counts |
| `answers.jsonl` | every reply, keyed by a hash of the prompt (the resume cache) |
| `games/deal-NNNN.json` | the full log of each game of that deal (both seatings, plus the baseline games): the deck, every action in order, and each question Claude was asked with its full reply, in the order they happened. Enough to replay the game exactly |
| `decisions.jsonl` | every decision: deal, seat, prompt, Claude's full reply (its reasoning), the answer. Useful for seeing *why* it played what it did |

## How it calls Claude

```
claude -p --output-format json --model opus --system-prompt <rules>
       --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence
```

The prompt goes on stdin. It runs in an empty temp directory, so no
CLAUDE.md, tools or MCP servers load and the system prompt is only the Tichu
rules (`src/llm/render.rs`). `scripts/fake-claude.sh` mimics this interface
(including a usage-limit reply) for testing the runner without the real CLI:
`--claude-bin scripts/fake-claude.sh`.

Code: `src/llm/` (`render.rs` prompts and parsing, `client.rs` backends,
cache and limit handling, `bot.rs` the `LlmBot`, `run.rs` the resumable
match).
