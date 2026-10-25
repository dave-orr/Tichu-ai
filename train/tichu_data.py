"""Reader for `tichu selfplay` record files (layout in src/selfplay.rs)."""

import struct

import numpy as np

HEADER = struct.Struct("<4sIIIII")
SEQ = struct.Struct("<BIBH")  # tag 1, round, seat, n tokens
REC = struct.Struct("<BIBBHHfH")  # tag 0, round, seat, flags, n options, label, margin, history length
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
    tokens  [T, D] uint8      history tokens, one whole-round sequence per seat
    seq_off [N] int64         first row of the record's sequence in `tokens`
    seq_len [N] int64         tokens the record saw (a prefix of that sequence)
    owner   [N, 56] uint8     relative seat 1..3 holding each unseen card, 255 = not unseen
    """

    def __init__(self, paths):
        parts = [read_file(p, i) for i, p in enumerate(paths)]
        dims = {(p["feature_version"], p["state_dim"], p["action_dim"], p["token_dim"]) for p in parts}
        if len(dims) != 1:
            raise ValueError(f"files disagree on feature layout: {dims}")
        (self.feature_version, self.state_dim, self.action_dim, self.token_dim), = dims
        self.states = np.concatenate([p["states"] for p in parts])
        self.actions = np.concatenate([p["actions"] for p in parts])
        shifts = np.cumsum([0] + [len(p["actions"]) for p in parts[:-1]])
        self.act_off = np.concatenate([p["act_off"] + s for p, s in zip(parts, shifts)])
        self.tokens = np.concatenate([p["tokens"] for p in parts])
        tshifts = np.cumsum([0] + [len(p["tokens"]) for p in parts[:-1]])
        self.seq_off = np.concatenate([p["seq_off"] + s for p, s in zip(parts, tshifts)])
        for k in ("n_opts", "label", "margin", "q", "group", "seq_len", "owner"):
            setattr(self, k, np.concatenate([p[k] for p in parts]))

    def __len__(self):
        return len(self.states)


def read_file(path, file_index=0):
    buf = open(path, "rb").read()
    magic, version, fv, S, A, D = HEADER.unpack_from(buf, 0)
    if magic != b"TCHR" or version != FORMAT_VERSION:
        raise ValueError(f"{path}: not a v{FORMAT_VERSION} tichu record file")
    data = np.frombuffer(buf, dtype=np.uint8)
    # First pass: block boundaries.
    meta, seqs, pos = [], [], HEADER.size
    while pos < len(buf):
        if buf[pos] == 1:
            _, rnd, seat, n = SEQ.unpack_from(buf, pos)
            seqs.append((pos + SEQ.size, n))
            pos += SEQ.size + n * D
            continue
        if buf[pos] != 0:
            raise ValueError(f"{path}: bad block tag {buf[pos]} at {pos}")
        _, rnd, seat, flags, n, label, margin, hlen = REC.unpack_from(buf, pos)
        body = pos + REC.size
        meta.append((rnd, flags, n, label, margin, body, len(seqs) - 1, hlen))
        pos = body + 56 + S + n * A + (4 * n if flags & 1 else 0)
    if pos != len(buf):
        raise ValueError(f"{path}: truncated")
    N = len(meta)
    seq_n = np.array([n for _, n in seqs], dtype=np.int64)
    seq_start = np.concatenate([[0], np.cumsum(seq_n)[:-1]]).astype(np.int64)
    tokens = np.empty((int(seq_n.sum()), D), dtype=np.uint8)
    for (p, n), o in zip(seqs, seq_start):
        tokens[o:o + n] = data[p:p + n * D].reshape(n, D)
    n_opts = np.array([m[2] for m in meta], dtype=np.int64)
    act_off = np.concatenate([[0], np.cumsum(n_opts)[:-1]]).astype(np.int64)
    states = np.empty((N, S), dtype=np.uint8)
    owner = np.empty((N, 56), dtype=np.uint8)
    actions = np.empty((int(n_opts.sum()), A), dtype=np.uint8)
    q = np.full(len(actions), np.nan, dtype=np.float32)
    for i, (_, flags, n, _, _, body, _, _) in enumerate(meta):
        owner[i] = data[body:body + 56]
        states[i] = data[body + 56:body + 56 + S]
        a = body + 56 + S
        o = act_off[i]
        actions[o:o + n] = data[a:a + n * A].reshape(n, A)
        if flags & 1:
            q[o:o + n] = np.frombuffer(buf, dtype="<f4", count=n, offset=a + n * A)
    seq_len = np.array([m[7] for m in meta], dtype=np.int64)
    seq_of = np.array([m[6] for m in meta], dtype=np.int64)
    if (seq_of < 0).any() or (seq_len > seq_n[seq_of]).any():
        raise ValueError(f"{path}: record history longer than its sequence")
    return dict(
        feature_version=fv, state_dim=S, action_dim=A, token_dim=D,
        states=states, actions=actions, act_off=act_off, n_opts=n_opts, q=q,
        label=np.array([m[3] for m in meta], dtype=np.int64),
        margin=np.array([m[4] for m in meta], dtype=np.float32),
        group=np.array([m[0] for m in meta], dtype=np.int64) + (file_index << 32),
        tokens=tokens, seq_off=seq_start[seq_of], seq_len=seq_len, owner=owner,
    )
