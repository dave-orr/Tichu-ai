#!/usr/bin/env bash
# End-to-end check of the learning pipeline: `tichu selfplay` records are
# read by train/tichu_data.py, and (if PyTorch is installed) train.py trains
# a tiny model whose .tnet the Rust net: and vpimc bots load and play. The
# loader recomputes the trainer's worked example, so this catches any drift
# between the Python and Rust sides of the record or model formats.
#
#   scripts/check-training.sh [PYTHON]      (default python3)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python3}
cargo build --release -q
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
./target/release/tichu selfplay --out "$dir/label.bin" --rounds 40 --seed 1 --threads 2 2>/dev/null
./target/release/tichu selfplay --out "$dir/q.bin" --rounds 4 --q 2 --seed 2 --threads 2 2>/dev/null

"$PY" - "$dir/label.bin" "$dir/q.bin" <<'PYEOF'
import sys
import numpy as np
sys.path.insert(0, "train")
from tichu_data import Records

label, q = sys.argv[1:3]
r = Records([label, q])
assert (r.state_dim, r.action_dim, r.token_dim) == (640, 48, 48), (r.state_dim, r.action_dim, r.token_dim)
assert len(r) > 400, len(r)
assert (r.n_opts >= 2).all() and (r.label < r.n_opts).all() and (r.label >= 0).all()
assert r.act_off[-1] + r.n_opts[-1] == len(r.actions)
assert np.isnan(r.q[: int(Records([label]).n_opts.sum())]).all(), "label-only file has Q"
rq = Records([q])
assert not np.isnan(rq.q).any(), "--q file is missing Q-values"
assert len(np.unique(r.group)) == 44, "round ids not unique across files"
print(f"tichu_data: {len(r)} records read")
PYEOF

if "$PY" -c "import torch" 2>/dev/null; then
  "$PY" train/train.py "$dir/label.bin" "$dir/q.bin" --out "$dir/m.tnet" --hidden 32 --head 16 --epochs 1 --device cpu >/dev/null
  ./target/release/tichu arena "net:$dir/m.tnet" random --rounds 6 --threads 2 | grep -q "margin per deal"
  ./target/release/tichu arena "vpimc2:$dir/m.tnet" heuristic --rounds 2 --threads 2 | grep -q "margin per deal"
  echo "train.py model loads and plays in Rust"
else
  echo "PyTorch not installed: skipped the train.py -> .tnet check"
fi
