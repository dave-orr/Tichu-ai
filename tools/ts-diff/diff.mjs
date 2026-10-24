// Replays Rust round traces (from `tichu trace`) through the production
// TypeScript engine (dave-orr/tichu, shared/) and reports any disagreement:
//   - state before every action (turn, hands, trick, wish, outs, scores, ...)
//   - every action Rust took is accepted by TS
//   - the legal plays Rust offers on each turn == every subset of the hand
//     TS's playCards accepts (brute force), and canPass == TS passTurn accepts
//   - out-of-turn bombs and Small Tichu eligibility
//
// usage: node tools/ts-diff/diff.mjs [traces.jsonl] [--quick]
//   TICHU_SHARED=path/to/tichu/shared/dist/index.js (default ../tichu/shared/dist)
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const here = path.dirname(new URL(import.meta.url).pathname);
const sharedPath = process.env.TICHU_SHARED ?? path.resolve(here, '../../../tichu/shared/dist/index.js');
const T = await import(pathToFileURL(sharedPath).href);

const args = process.argv.slice(2);
const quick = args.includes('--quick');
const file = args.find(a => !a.startsWith('--'));
const input = file ? createReadStream(file) : process.stdin;

const ids = cards => cards.map(T.cardId).sort();
const key = cards => ids(cards).join(',');

function snapshot(s) {
  return {
    phase: s.phase,
    turn: s.turnIndex,
    hands: s.players.map(p => ids(p.hand)),
    trick: s.currentTrick ? { type: s.currentTrick.type, rank: s.currentTrick.rank, length: s.currentTrick.length } : null,
    trickPlays: s.currentTrickPlays.length,
    wish: s.mahJongWish,
    wishPending: s.mahJongWishPending,
    outOrder: s.players.map(p => p.outOrder),
    lastPlayedBy: s.lastPlayedBy,
    passCount: s.passCount,
    countdown: s.trickCountdown ? s.trickCountdown.winner : null,
    dragonBy: s.dragonGiveaway ? s.dragonGiveawayBy : null,
    scores: s.teams.map(t => t.score),
    captured: s.players.map(p => T.sumPoints(p.tricksWon.flat())),
    tricks: s.players.map(p => p.tricksWon.length),
    tichu: s.players.map(p => p.tichuCall),
  };
}

// TS keeps the last trick on the table in its final state (it is only folded
// in for scoring), so compare just the outcome at round end.
const FINAL_KEYS = ['phase', 'scores', 'outOrder', 'tichu', 'hands'];

// Key-order-independent JSON.
const canon = v => JSON.stringify(v, (_, x) => (x && typeof x === 'object' && !Array.isArray(x)
  ? Object.fromEntries(Object.keys(x).sort().map(k => [k, x[k]])) : x));

function diffObjects(a, b, keys) {
  const out = [];
  for (const k of keys ?? Object.keys(b)) {
    if (canon(a[k]) !== canon(b[k])) out.push(`${k}: ts=${JSON.stringify(a[k])} rust=${JSON.stringify(b[k])}`);
  }
  return out;
}

function toCard(id) {
  if (!id.includes('-')) return { type: 'special', name: id };
  const [suit, rank] = id.split('-');
  return { type: 'normal', suit, rank: Number(rank) };
}

// Every play TS accepts from `seat` now: "ids|rank" strings.
function tsLegalPlays(s, seat) {
  const hand = s.players[seat].hand;
  const n = hand.length;
  const out = new Set();
  for (let mask = 1; mask < (1 << n); mask++) {
    const cards = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) cards.push(hand[i]);
    if (!T.identifyCombo(cards)) continue;
    const hasPhoenix = cards.some(c => c.type === 'special' && c.name === 'phoenix');
    const readings = hasPhoenix && cards.length >= 5 ? [undefined, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] : [undefined];
    for (const pa of readings) {
      const r = T.playCards(s, seat, cards, pa);
      if (r.state === s) continue;
      // A Dog play leaves no trick; record it with rank 0 like Rust.
      const rank = r.state.currentTrick && r.state.currentTrickPlays.length > s.currentTrickPlays.length
        ? r.state.currentTrick.rank : 0;
      out.add(`${key(cards)}|${rank}`);
    }
  }
  return out;
}

// Every bomb TS accepts from `seat` now (4-of-a-kinds and straight flushes).
function tsLegalBombs(s, seat) {
  const hand = s.players[seat].hand.filter(c => c.type === 'normal');
  const cands = [];
  for (let r = 2; r <= 14; r++) {
    const q = hand.filter(c => c.rank === r);
    if (q.length === 4) cands.push(q);
  }
  for (const suit of ['jade', 'sword', 'pagoda', 'star']) {
    const has = r => hand.some(c => c.suit === suit && c.rank === r);
    for (let lo = 2; lo <= 10; lo++) {
      for (let hi = lo + 4; hi <= 14; hi++) {
        let ok = true;
        for (let r = lo; r <= hi; r++) if (!has(r)) { ok = false; break; }
        if (!ok) break;
        const run = [];
        for (let r = lo; r <= hi; r++) run.push({ type: 'normal', suit, rank: r });
        cands.push(run);
      }
    }
  }
  const out = new Set();
  for (const cards of cands) {
    const r = T.playBomb(s, seat, cards);
    if (r.state !== s) out.add(`${key(cards)}|${r.state.currentTrick.rank}`);
  }
  return out;
}

function setDiff(ts, rust) {
  const missing = [...ts].filter(x => !rust.has(x));
  const extra = [...rust].filter(x => !ts.has(x));
  const out = [];
  if (missing.length) out.push(`TS allows but Rust lacks: ${missing.slice(0, 5).join('  ')}${missing.length > 5 ? ` (+${missing.length - 5})` : ''}`);
  if (extra.length) out.push(`Rust allows but TS rejects: ${extra.slice(0, 5).join('  ')}${extra.length > 5 ? ` (+${extra.length - 5})` : ''}`);
  return out;
}

function checkMoves(s, checks) {
  const errs = [];
  if (checks.turn) {
    const { seat, plays, canPass } = checks.turn;
    if (!quick) {
      const rust = new Set(plays.map(p => `${p.cards.join(',')}|${p.rank}`));
      errs.push(...setDiff(tsLegalPlays(s, seat), rust).map(e => `plays(seat ${seat}): ${e}`));
    }
    const tsPass = T.passTurn(s, seat).state !== s;
    if (tsPass !== canPass) errs.push(`canPass(seat ${seat}): ts=${tsPass} rust=${canPass}`);
  }
  if (s.phase === 'playing') {
    for (let seat = 0; seat < 4; seat++) {
      const rust = new Set(checks.bombs[seat].map(p => `${p.cards.join(',')}|${p.rank}`));
      errs.push(...setDiff(tsLegalBombs(s, seat), rust).map(e => `bombs(seat ${seat}): ${e}`));
    }
  }
  for (let seat = 0; seat < 4; seat++) {
    const ts = T.callSmallTichu(s, seat) !== s;
    if (ts !== checks.smallTichu[seat]) errs.push(`smallTichu(seat ${seat}): ts=${ts} rust=${checks.smallTichu[seat]}`);
  }
  return errs;
}

function apply(s, a, passes) {
  switch (a.type) {
    case 'grandTichu': return T.callGrandTichu(s, a.seat, a.call);
    case 'smallTichu': return T.callSmallTichu(s, a.seat);
    case 'passCards': {
      const next = T.passCards(s, a.seat, a.pass);
      if (next === s) return s;
      passes[a.seat] = a.pass;
      return Object.keys(passes).length === 4 ? T.applyPasses(next, passes) : next;
    }
    case 'play': return T.playCards(s, a.seat, a.cards, a.phoenixAs ?? undefined).state;
    case 'passTurn': return T.passTurn(s, a.seat).state;
    case 'bomb': return T.playBomb(s, a.seat, a.cards).state;
    case 'wish': return T.setMahJongWish(s, a.seat, a.rank);
    case 'resolveTrick': return T.awardTrick(s).state;
    case 'giveDragon': return T.giveDragonTrick(s, a.seat, a.to).state;
    default: throw new Error(`unknown action ${a.type}`);
  }
}

function replay(tr) {
  let s = T.createInitialState({ ...T.DEFAULT_SETTINGS, clockwise: tr.settings.clockwise, targetScore: tr.settings.targetScore });
  s.teams = [{ ...s.teams[0], score: tr.scores[0] }, { ...s.teams[1], score: tr.scores[1] }];
  s.roundNumber = tr.roundNumber - 1;
  s = T.startNewRound(s);
  s.deck = tr.deck;
  s.players.forEach((p, i) => { p.hand = T.sortHand(tr.deck.slice(i * 14, i * 14 + 8)); });
  const passes = {};
  for (const [i, step] of tr.steps.entries()) {
    const where = `step ${i} (${JSON.stringify(step.action)})`;
    const d = diffObjects(snapshot(s), step.before);
    if (d.length) return [`${where}: state differs before action: ${d.join('; ')}`];
    const m = checkMoves(s, step.checks);
    if (m.length) return m.map(e => `${where}: ${e}`);
    const next = apply(s, step.action, passes);
    if (next === s) return [`${where}: TS rejected the action`];
    s = next;
  }
  const d = diffObjects(snapshot(s), tr.final, FINAL_KEYS);
  return d.length ? [`final state differs: ${d.join('; ')}`] : [];
}

let rounds = 0, bad = 0, steps = 0;
const rl = createInterface({ input });
for await (const line of rl) {
  if (!line.trim()) continue;
  const tr = JSON.parse(line);
  rounds++;
  steps += tr.steps.length;
  if (tr.error) { bad++; console.log(`round ${rounds}: rust sim error: ${tr.error}`); continue; }
  const errs = replay(tr);
  if (errs.length) {
    bad++;
    if (bad <= 10) console.log(`round ${rounds}:\n  ${errs.slice(0, 5).join('\n  ')}`);
  }
}
console.log(`${rounds} rounds, ${steps} actions replayed: ${bad ? `${bad} with differences` : 'Rust and TS agree on everything'}`);
process.exit(bad ? 1 : 0);
