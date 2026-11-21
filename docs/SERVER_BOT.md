# Putting the bot on tichu.squidbox.com

What it would take to let people play against `best` on the live server.
Nothing here has been done; this is the plan and its costs.

## The bot

`best[:NET]` (in `src/bots/mod.rs`) is:

- card play: `search:NET`, the net's top 3 moves checked by 24 guessed deals
  each, exact two-player endgames (NET defaults to `runs/v4/gen3.tnet`);
- Grand Tichu: learned model `runs/calls3/m3.tcall`;
- Small Tichu: learned model `runs/calls1/m1.tcall`;
- passing, bombs out of turn, the wish and the Dragon gift: h2's rules.

The three model files (1.5 MB + 2 × 0.9 MB) live only on the desktop. A bot
host needs copies of them and the `tichu` binary; nothing else (no Python,
no GPU).

## Decision speed

`cargo run --release --example decision_time -- best h2 40` times every
decision on one thread. See the README section on `best` for the numbers.
Plays are the only slow decisions; calls, passes, bombs, wishes and Dragon
gifts take under 0.1 s. Forced plays (one legal option) are instant.

One table with three bots needs about one core while a bot is thinking, and
bots think one at a time (it's one seat's turn), so a small VPS core per
concurrent table is plenty. Search could use several threads per decision if
the tail ever matters.

## What the server offers today

dave-orr/tichu already has an API for external bots (`docs/AI_API.md`,
`server/src/api.ts`): the room's organizer marks seats "Open for AI", a bot
joins with `POST /api/join` (matchmaking) or `/api/rooms/:code/join`, reads
`game-state` events from an SSE stream and posts actions. So no new
protocol is needed. What's missing:

1. **The action history.** The bots read the whole round's history (the
   net's features and the hand inference replay `PlayerView::log`), but
   `ClientGameState` only has the current trick and the set of played cards.
   The server already records exactly the needed log
   (`room.accumulator.actions`, the `RoundLog.actions` of the round export,
   which `src/humanlog.rs` replays). Cheapest fix: include this round's
   actions (and, for the API seat, its own pass) in the state sent to API
   players. Reconstructing the history client-side by diffing successive
   SSE states would also work but is fragile (bursts of events, bombs in the
   countdown, server auto-passes).
2. **A bridge process** (Rust, in this repo): keeps an SSE connection per
   seat, turns each `game-state` into a `PlayerView` (hand, public players,
   trick, wish, countdown, scores, log), decides when it's this seat's move
   (the flowchart in `AI_API.md`), asks the bot, and posts the action. It
   needs a card/combo JSON mapping (`types.rs` already serializes cards in
   the server's format) and handling for the parts the simulator schedules
   differently from the real server:
   - bombs can be played at any time; offer the bot a bomb decision on every
     new play and when a trick countdown starts (the server's window is ~2 s,
     and the bot's bomb decision is instant);
   - Small Tichu: ask before each of the seat's turns while allowed, as the
     simulator does;
   - Grand Tichu: the server asks everyone at once; the bot only needs its 8
     cards;
   - concessions: accept (the server already auto-accepts for AI seats).
   The bridge should check every action against the engine before posting
   it, so a mapping bug shows up as a logged error, not a stuck game.
3. **Getting bots into rooms.** Today a bot must already be running and call
   `/api/join` after a human opens a seat. Either the bridge polls
   `/api/join` every few seconds (simple, no server change), or the server
   gets an "add bot" button that calls the bot host.
4. **Security, before letting the public at it.** `SECURITY_TODO.md` lists
   the AI API as unauthenticated: anyone who knows a room code can open the
   SSE stream of an API seat (and see that bot's hand) or post its actions.
   With bots playing against strangers that is a real cheating hole. A
   per-seat secret returned by `join` and required on `stream`/`action`
   closes it. Also decide whether games with bots count toward Elo and
   stats (they do today).
5. **Pacing.** Bots answer in about a tenth of a second. A small delay
   (0.5–1 s) per play makes the table readable for humans.

## Testing before going live

Run the server locally (`npm run dev` in dave-orr/tichu), open a room, mark
three seats for AI, and let the bridge fill them; then a full game with a
human. Then the same on staging if there is one. Only then on the live site.
