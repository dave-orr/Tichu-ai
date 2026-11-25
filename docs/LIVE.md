# Playing on tichu.squidbox.com

`tichu live` puts a bot at real tables on the server in dave-orr/tichu,
through its AI player API (`docs/AI_API.md` there). People create a room as
usual, mark seats "Open for AI" in the waiting room, and the bot fills them.
One `tichu live` process serves every open seat on the server, each on its
own thread.

```sh
BOT_TOKEN=... tichu live --bot best --server http://localhost:3000
```

## How it works

For each seat it claims, it opens the seat's event stream and keeps a
*shadow* game (`src/live.rs`): the real round replayed through our engine
from the server's `roundActions` log, with the hidden hands filled in by a
guess and cards swapped between hidden hands whenever someone plays a card
the guess didn't give them. The bot is handed `shadow.view(seat)`, so it
sees exactly the `PlayerView` (log included) it would see in `tichu arena`,
and every move is checked by the engine before it is sent. `tests/live.rs`
checks the shadow against real engine games, action by action.

Grand Tichu is called by everyone at once on the server, so the bridge makes
its bots take turns: a bot decides only once every person at the table has,
and after any bot in a lower seat, so it sees every call made before it (as
in `tichu arena`, which asks the seats in order). It never calls Grand over
its partner's call.

The server lets anyone bomb at any instant. Like the simulator, the bridge
offers its bot a bomb after every play by another seat and during each trick
countdown (sent at once; the countdown is 3 s). Small Tichu is offered
before each of its turns while allowed. Other moves wait `--delay-ms`
(default 800) so people can follow the table; if the table changes during
the wait, it thinks again.

It needs the server from dave-orr/tichu#148 or later: the bot token, the
`roundActions` log, 409 for refused moves. A seat can move before the first
deal (the organizer swaps seats, or random partners reshuffles the table as
the game starts): the bridge follows `mySeat` and starts its shadow afresh.
With dave-orr/tichu#154 or later it addresses the server by the `player` id
the join returns, which follows it to the new seat; older servers only know
seats, and a moved bot there loses its stream.

## Options

See `tichu` with no arguments. The ones that matter:

- `--bot NAME`: any bundled bot spec. Model paths are relative to the
  working directory.
- `--server URL`: default `http://localhost:3000` (the bot on the same host
  as the server, so nothing goes through the proxy).
- `--name NAME`: shown at the table, default `Tichu AI`.
- `--room CODE`: join one room and exit when its game ends (for testing).
- `BOT_TOKEN` (environment): the server's `BOT_TOKEN` from its `server/.env`.

## Running it on squidbox

Once (on the host):

1. Deploy dave-orr/tichu#148 (`scripts/deploy.sh`), then add the token to
   `server/.env` and restart so the server reads it:
   `echo "BOT_TOKEN=$(openssl rand -hex 24)" >> ~/tichu/server/.env`.
2. Install Rust (`curl https://sh.rustup.rs -sSf | sh`), clone this repo next
   to `~/tichu`, and `cargo build --release`.
3. Copy the model files the bot spec names from the desktop into the clone,
   at the same paths. The default `best` (search on the v4 gen3 net plus the
   learned Grand and Small calls; about 0.12 s a play on one core) needs
   `runs/v4/gen3.tnet`, `runs/calls3/m3.tcall` and `runs/calls1/m1.tcall`
   (about 3.3 MB together). `best:PATH` swaps the net, e.g.
   `best:runs/dmc1/snap00190.tnet` for dmc1. (As nets alone gen3 is +4.4 ± 2.7
   a deal vs the v4 start net, and dmc1 +0.3 ± 3.0; under search dmc1 and the
   start net tie.)
4. `pm2 start scripts/live.config.js && pm2 save`. The config reads
   `BOT_TOKEN` and `PORT` from `../tichu/server/.env`, so the secret lives in
   one place; edit `BOT` in it to change the bot.

Logs: `pm2 logs tichu-bot`. Each seat logs when it joins and finishes, and
anything odd: a move the server refused, or the shadow out of step with the
server (both should never happen; they mean a bug in the bridge).

To update: `git pull && cargo build --release && pm2 restart tichu-bot`.
A restart abandons the seats it is playing (they can't be taken back over
mid-game yet), so do it between games. Restarting the *server* is fine: the
bot reconnects its streams and carries on.

## Testing locally

`scripts/live-e2e.mjs` plays a full game against three `tichu live` seats on
a local server, with the organizer's seat played by a simple script over
the socket:

```sh
# in dave-orr/tichu: npm run build; then
BOT_TOKEN=test-token-123456 PORT=3999 node server/dist/index.js &
BOT_TOKEN=test-token-123456 tichu live --bot h2 --server http://localhost:3999 --delay-ms 0 &
node scripts/live-e2e.mjs --server http://localhost:3999 --target 500
```

At the end, `tichu live --games N` prints how many seat-games had refused
moves or divergences (it exits non-zero if any did).

## Known limits

- A bot joins only while the room is waiting, so it can't take over a
  dropped player's seat mid-game, and a restarted `tichu live` doesn't pick
  its old seats back up.
- Games with an AI seat aren't Elo-rated (dave-orr/tichu `stats.ts`).
