// End-to-end check of `tichu live` against a local dave-orr/tichu server.
//
// Plays the organizer's seat itself (a simple scripted player over the
// socket, like a person in the browser would), marks the other three seats
// "Open for AI", waits for `tichu live` to claim them, starts the game and
// plays it to the end. Run it alongside a server and `tichu live`; see
// scripts/live-e2e.sh, which starts all three.
//
//   node scripts/live-e2e.mjs [--server URL] [--target N] [--clockwise] [--timeout-s S]
//
// Needs the tichu repo's node_modules (socket.io-client): TICHU_REPO or ../tichu.

import { createRequire } from 'node:module';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const server = opt('server', 'http://localhost:3000');
const target = Number(opt('target', '1000'));
const clockwise = args.includes('--clockwise');
const timeoutS = Number(opt('timeout-s', '900'));

const repo = process.env.TICHU_REPO || path.resolve(import.meta.dirname, '../../tichu');
const require = createRequire(path.join(repo, 'package.json'));
const { io } = require('socket.io-client');

const rank = c => (c.type === 'normal' ? c.rank : { dog: 0, mahjong: 1, phoenix: 15, dragon: 16 }[c.name]);
const byRank = h => [...h].sort((a, b) => rank(a) - rank(b));

const sock = io(server, { transports: ['websocket'] });
let room = null;
let marked = false;
let started = false;
let rounds = 0;
// The organizer's move candidates at the current decision, tried in order:
// the server answers a refused move with an error and no new state.
let key = '';
let cands = [];
let next = 0;
let errors = 0;

function fail(msg) {
  console.error(`e2e: FAIL: ${msg}`);
  process.exit(1);
}
setTimeout(() => fail(`no game end within ${timeoutS}s`), timeoutS * 1000);

sock.on('connect', () => {
  sock.emit('create-room', { playerName: 'Host', settings: { targetScore: target, clockwise } });
});
sock.on('room-created', ({ roomCode }) => {
  room = roomCode;
  console.error(`e2e: room ${room}`);
});
sock.on('error', ({ message }) => {
  // Scripted guesses (a pass the wish forbids, a single that doesn't beat)
  // are expected to bounce; the next state tries something else.
  errors++;
  if (errors > 5000) fail(`too many errors (last: ${message})`);
  tryNext();
});

let sentAt = 0;
function tryNext() {
  if (next >= cands.length) return;
  const [ev, data] = cands[next++];
  sentAt = Date.now();
  sock.emit(ev, data);
}
// A refused move changes nothing (and may not even be answered): if the
// decision is still open a second later, try the next candidate.
setInterval(() => {
  if (key && next < cands.length && Date.now() - sentAt > 1000) tryNext();
}, 250);

/** Every single and same-rank pair in the hand, lowest first. */
function plays(hand) {
  const out = byRank(hand).filter(c => !(c.type === 'special' && c.name === 'dog')).map(c => [c]);
  const normal = byRank(hand.filter(c => c.type === 'normal'));
  for (let i = 0; i + 1 < normal.length; i++) {
    if (normal[i].rank === normal[i + 1].rank) out.push([normal[i], normal[i + 1]]);
  }
  return out;
}

sock.on('game-state', ({ state, aiOpenSeats }) => {
  const me = state.mySeat;
  if (state.phase === 'waiting') {
    if (!marked) {
      marked = true;
      for (const s of [1, 2, 3]) sock.emit('mark-seat-ai', { seat: s });
    }
    const full = state.players.every(p => p.name);
    if (full && !started && (aiOpenSeats ?? []).length === 0) {
      started = true;
      console.error(`e2e: players ${state.players.map(p => p.name).join(', ')}; starting`);
      sock.emit('start-game');
    }
    return;
  }
  const p = state.players[me];
  switch (state.phase) {
    case 'grandTichuWindow':
      key = '';
      if (!p.grandTichuDecided) sock.emit('call-grand-tichu', { call: false });
      break;
    case 'passing':
      if (!p.passedCards && state.myHand.length === 14) {
        const [a, b, c] = byRank(state.myHand);
        sock.emit('pass-cards', { left: a, partner: b, right: c });
      }
      break;
    case 'playing': {
      if (state.concedeOffer) {
        sock.emit('concede-respond', { accept: true });
        break;
      }
      if (state.mahJongWishPending && state.lastPlayedBy === me) {
        sock.emit('mah-jong-wish', { rank: null });
        break;
      }
      if (state.dragonGiveaway && state.dragonGiveawayBy === me) {
        sock.emit('give-dragon-trick', { to: (me + 1) % 4 });
        break;
      }
      if (state.turnIndex !== me || state.trickCountdown || state.mahJongWishPending || state.dragonGiveaway) {
        key = '';
        break;
      }
      const k = `${state.roundNumber}:${state.playedCards.length}:${state.passCount}:${state.currentTrickPlays?.length}:${state.mahJongWish}`;
      if (k === key) break;
      key = k;
      next = 0;
      // Leading: the Mah Jong if we have it, else the lowest card. Following:
      // pass, unless the wish forbids it.
      const mj = state.myHand.filter(c => c.type === 'special' && c.name === 'mahjong');
      const leads = [...(mj.length ? [mj] : []), ...plays(state.myHand), ...state.myHand.map(c => [c])];
      cands = state.currentTrick
        ? [['pass-turn', {}], ...plays(state.myHand).map(cards => ['play-cards', { cards }])]
        : leads.map(cards => ['play-cards', { cards }]);
      tryNext();
      break;
    }
    case 'roundEnd':
      key = '';
      if (!state.roundEndReady.includes(me) && rounds < state.roundNumber) {
        rounds = state.roundNumber;
        const s = state.teams.map(t => t.score).join(' : ');
        console.error(`e2e: round ${state.roundNumber} done, score ${s}`);
        sock.emit('next-round');
      }
      break;
    case 'gameEnd': {
      const s = state.teams.map(t => t.score).join(' : ');
      console.error(`e2e: game over after ${state.roundNumber} rounds, score ${s}`);
      if (!state.roundEndReady.includes(me)) sock.emit('next-round');
      setTimeout(() => process.exit(0), 3000);
      break;
    }
  }
});
