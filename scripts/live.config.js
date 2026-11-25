// pm2 process for `tichu live` on the server's host (see docs/LIVE.md):
//
//   pm2 start scripts/live.config.js && pm2 save
//
// Reads BOT_TOKEN and PORT from the tichu server's .env (TICHU_ENV, default
// ../tichu/server/.env next to this repo), so the token is kept in one place.

const fs = require('fs');
const path = require('path');

const repo = path.join(__dirname, '..');
// The bot spec (default: `best`, the strongest assembled bot; see
// src/bots/mod.rs). Model paths are relative to the repo (pm2's cwd below).
const BOT = process.env.BOT || 'best';

function readEnv(file) {
  const out = {};
  if (!fs.existsSync(file)) return out;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (m) out[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
  return out;
}

const envFile = process.env.TICHU_ENV || path.join(repo, '../tichu/server/.env');
const serverEnv = readEnv(envFile);
if (!serverEnv.BOT_TOKEN) {
  throw new Error(`no BOT_TOKEN in ${envFile} (see docs/LIVE.md)`);
}
const port = serverEnv.PORT || '3000';

module.exports = {
  apps: [
    {
      name: 'tichu-bot',
      script: path.join(repo, 'target/release/tichu'),
      interpreter: 'none',
      args: ['live', '--bot', BOT, '--server', `http://localhost:${port}`],
      cwd: repo,
      env: { BOT_TOKEN: serverEnv.BOT_TOKEN },
      exp_backoff_restart_delay: 2000,
      max_restarts: 20,
      min_uptime: 10000,
      error_file: path.join(repo, 'logs/tichu-bot.log'),
      out_file: path.join(repo, 'logs/tichu-bot.log'),
      merge_logs: true,
      time: true,
    },
  ],
};
