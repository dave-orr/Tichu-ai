// Checks that the committed viewer/replay.wasm builds the same replays as
// the current Rust code (`tichu replay --json`), i.e. that it was rebuilt
// (scripts/build-viewer-wasm.sh) after the last change to the replay builder
// or the rules.
//
//   node tools/viewer-check/check.mjs viewer/replay.wasm target/release/tichu LOG.json...
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { isDeepStrictEqual } from 'node:util';

const [wasmPath, tichu, ...logs] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const w = instance.exports;

function wasmBuild(text) {
  const bytes = new TextEncoder().encode(text);
  const p = w.alloc(bytes.length);
  new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
  w.build(p, bytes.length);
  const out = new Uint8Array(w.memory.buffer, w.out_ptr(), w.out_len());
  return JSON.parse(new TextDecoder().decode(out));
}

let bad = 0;
for (const log of logs) {
  const want = JSON.parse(execFileSync(tichu, ['replay', log, '--json'], { maxBuffer: 1 << 28 }).toString());
  const got = wasmBuild(readFileSync(log, 'utf8'));
  if (!isDeepStrictEqual(got, want)) {
    bad++;
    console.error(`${log}: viewer/replay.wasm disagrees with tichu replay`);
  }
}
if (bad) {
  console.error(`${bad}/${logs.length} logs differ: run scripts/build-viewer-wasm.sh and commit viewer/replay.wasm`);
  process.exit(1);
}
console.log(`viewer/replay.wasm matches tichu replay on ${logs.length} logs`);
