"""Reader for `tichu selfplay` record files (layout in src/selfplay.rs)."""

import struct

import numpy as np

HEADER = struct.Struct("<4sIIIII")
BLOCK = struct.Struct("<IBHI")  # round, seat, n tokens, n records
REC = struct.Struct("<BHHHf")  # flags, n options, label, prefix, margin
FORMAT_VERSION = 2


class Records:
    """Every record of one or more files, as flat numpy arrays.

    states  [N, S] uint8      state features
    actions [M, A] uint8      candidate features, records back to back
    act_off [N] int64         first row of each record in `actions`
    n_opts  [N] int64         candidates per record
    label   [N] int64         index of the label among its candidates
    margin  [N] float32       final round margin for the record's team
    q       [M] float32       per-candidate Q (NaN where the file has none)
    group   [N] int64         round id, unique across files (for splits)
    tokens  [T, D] uint8      history tokens, one sequence per seat per round
    tok_off [N] int64         first row of the record's sequence in `tokens`
    tok_len [N] int64         how many of those tokens the record saw
    """

    def __init__(self, paths):
        parts = [read_file(p, i) for i, p in enumerate(paths)]
        dims = {(p["feature_version"], p["state_dim"], p["action_dim"], p["token_dim"]) for p in parts}
        if len(dims) != 1:
            raise ValueError(f"files disagree on feature layout: {dims}")
        (self.feature_version, self.state_dim, self.action_dim, self.token_dim), = dims
        self.states = np.concatenate([p["states"] for p in parts])
        self.actions = np.concatenate([p["actions"] for p in parts])
        self.tokens = np.concatenate([p["tokens"] for p in parts])
        shift = lambda key: np.cumsum([0] + [len(p[key]) for p in parts[:-1]])  # noqa: E731
        self.act_off = np.concatenate([p["act_off"] + s for p, s in zip(parts, shift("actions"))])
        self.tok_off = np.concatenate([p["tok_off"] + s for p, s in zip(parts, shift("tokens"))])
        for k in ("n_opts", "label", "margin", "q", "group", "tok_len"):
            setattr(self, k, np.concatenate([p[k] for p in parts]))

    def __len__(self):
        return len(self.states)


def read_file(path, file_index=0):
    buf = open(path, "rb").read()
    magic, version, fv, S, A, D = HEADER.unpack_from(buf, 0)
    if magic != b"TCHR" or version != FORMAT_VERSION:
        raise ValueError(f"{path}: not a v{FORMAT_VERSION} tichu record file")
    data = np.frombuffer(buf, dtype=np.uint8)
    # First pass: block and record boundaries.
    meta, seqs, pos, n_tok = [], [], HEADER.size, 0
    while pos < len(buf):
        rnd, _seat, nt, nr = BLOCK.unpack_from(buf, pos)
        pos += BLOCK.size
        seqs.append((pos, nt))
        pos += nt * D
        for _ in range(nr):
            flags, n, label, prefix, margin = REC.unpack_from(buf, pos)
            body = pos + REC.size
            meta.append((rnd, flags, n, label, margin, body, n_tok, prefix))
            pos = body + S + n * A + (4 * n if flags & 1 else 0)
        n_tok += nt
    if pos != len(buf):
        raise ValueError(f"{path}: truncated")
    N = len(meta)
    n_opts = np.array([m[2] for m in meta], dtype=np.int64)
    act_off = np.concatenate([[0], np.cumsum(n_opts)[:-1]]).astype(np.int64)
    states = np.empty((N, S), dtype=np.uint8)
    actions = np.empty((int(n_opts.sum()), A), dtype=np.uint8)
    q = np.full(len(actions), np.nan, dtype=np.float32)
    for i, (_, flags, n, _, _, body, _, _) in enumerate(meta):
        states[i] = data[body:body + S]
        a = body + S
        o = act_off[i]
        actions[o:o + n] = data[a:a + n * A].reshape(n, A)
        if flags & 1:
            q[o:o + n] = np.frombuffer(buf, dtype="<f4", count=n, offset=a + n * A)
    tokens = np.empty((n_tok, D), dtype=np.uint8)
    t = 0
    for start, nt in seqs:
        tokens[t:t + nt] = data[start:start + nt * D].reshape(nt, D)
        t += nt
    return dict(
        feature_version=fv, state_dim=S, action_dim=A, token_dim=D,
        states=states, actions=actions, act_off=act_off, n_opts=n_opts, q=q, tokens=tokens,
        label=np.array([m[3] for m in meta], dtype=np.int64),
        margin=np.array([m[4] for m in meta], dtype=np.float32),
        group=np.array([m[0] for m in meta], dtype=np.int64) + (file_index << 32),
        tok_off=np.array([m[6] for m in meta], dtype=np.int64),
        tok_len=np.array([m[7] for m in meta], dtype=np.int64),
    )
