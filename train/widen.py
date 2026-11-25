"""Grow a trained net without changing what it computes (Net2Net-style), so
a bigger model can start from a smaller one's play instead of from scratch.

    python train/widen.py runs/dmc1/snap00190.pt --hidden 512 --depth 3 --out runs/big/start.pt

- wider layers: the new units get small random input weights and zero
  output weights, so every output is unchanged but they get gradients;
- deeper trunk: extra layers are the identity (the trunk's activations are
  ReLU outputs, so ReLU(I·x) = x).

Writes the .pt and a matching .tnet (playable as `net:`). With --check it
also confirms on random inputs that the outputs didn't move.
"""

import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TichuNet, write_tnet  # noqa: E402
from dmc import Dims, arch_of  # noqa: E402

FEATURE_VERSION_DEFAULT = None


def grow(w, b, out_n, in_map, in_n, noise, gen):
    """A Linear's weight [out, in] and bias grown to [out_n, in_n]: old rows
    keep their weights at the input positions `in_map` (old column -> new
    column), new columns are zero, new rows small random."""
    W = torch.zeros(out_n, in_n)
    W[:w.shape[0]] = 0.0
    W[:w.shape[0], in_map] = w
    if out_n > w.shape[0]:
        W[w.shape[0]:] = torch.randn(out_n - w.shape[0], in_n, generator=gen) * noise / in_n ** 0.5
    B = torch.zeros(out_n)
    B[:b.shape[0]] = b
    return W, B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="a TichuNet .pt (train.py or dmc.py)")
    ap.add_argument("--out", required=True, help="output .pt (a .tnet is written beside it)")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--head", type=int)
    ap.add_argument("--value-net", type=int)
    ap.add_argument("--noise", type=float, default=1.0, help="scale of the new units' input weights (× 1/√fan-in)")
    ap.add_argument("--feature-version", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    sd = torch.load(args.src, map_location="cpu")
    a = arch_of(sd)
    S = sd["trunk.0.weight"].shape[1]
    A = sd["policy.0.weight"].shape[1] - a["hidden"]
    H = args.hidden or a["hidden"]
    D = args.depth or a["depth"]
    K = args.head or a["head"]
    VN = a["value_net"] if args.value_net is None else args.value_net
    if H < a["hidden"] or D < a["depth"] or K < a["head"] or (a["value_net"] and VN < a["value_net"]):
        sys.exit("widen.py only grows nets")
    if VN and not a["value_net"]:
        sys.exit("the source has no separate value net to grow")
    h0 = a["hidden"]
    new = {}
    # Trunk: old layers widened, then identity layers.
    in_map, in_n = list(range(S)), S
    for i in range(a["depth"]):
        new[f"trunk.{i}.weight"], new[f"trunk.{i}.bias"] = grow(
            sd[f"trunk.{i}.weight"], sd[f"trunk.{i}.bias"], H, in_map, in_n, args.noise, gen)
        in_map, in_n = list(range(h0)), H
    for i in range(a["depth"], D):
        new[f"trunk.{i}.weight"], new[f"trunk.{i}.bias"] = torch.eye(H), torch.zeros(H)
    trunk_cols = list(range(h0))
    # Trunk value head (hidden -> head//2 -> 1).
    hv0, hv = sd["value.0.weight"].shape[0], K // 2
    new["value.0.weight"], new["value.0.bias"] = grow(sd["value.0.weight"], sd["value.0.bias"], hv, trunk_cols, H, args.noise, gen)
    new["value.1.weight"], new["value.1.bias"] = grow(sd["value.1.weight"], sd["value.1.bias"], 1, list(range(hv0)), hv, args.noise, gen)
    # Policy head: input is trunk ⊕ action.
    k0 = sd["policy.0.weight"].shape[0]
    pcols = trunk_cols + list(range(H, H + A))
    new["policy.0.weight"], new["policy.0.bias"] = grow(sd["policy.0.weight"], sd["policy.0.bias"], K, pcols, H + A, args.noise, gen)
    new["policy.1.weight"], new["policy.1.bias"] = grow(sd["policy.1.weight"], sd["policy.1.bias"], 1, list(range(k0)), K, args.noise, gen)
    # Separate value net over the state.
    if VN:
        v0 = a["value_net"]
        new["value_net.0.weight"], new["value_net.0.bias"] = grow(sd["value_net.0.weight"], sd["value_net.0.bias"], VN, list(range(S)), S, args.noise, gen)
        new["value_net.1.weight"], new["value_net.1.bias"] = grow(sd["value_net.1.weight"], sd["value_net.1.bias"], VN, list(range(v0)), VN, args.noise, gen)
        new["value_net.2.weight"], new["value_net.2.bias"] = grow(sd["value_net.2.weight"], sd["value_net.2.bias"], 1, list(range(v0)), VN, args.noise, gen)

    small = TichuNet(S, A, a["hidden"], a["depth"], a["head"], 0, 0, a["value_net"])
    small.load_state_dict(sd)
    big = TichuNet(S, A, H, D, K, 0, 0, VN)
    big.load_state_dict(new)
    small.eval()
    big.eval()
    # The outputs on random sparse inputs must not move.
    s = (torch.rand(64, S, generator=gen) < 0.1).float() * torch.rand(64, S, generator=gen)
    x = (torch.rand(64, 5, A, generator=gen) < 0.2).float() * torch.rand(64, 5, A, generator=gen)
    z = torch.zeros(64, 0, 0)
    lens = torch.zeros(64, dtype=torch.long)
    with torch.no_grad():
        outs = [m(s, x, z, lens) for m in (small, big)]
    err = max((o1 - o2).abs().max().item() for o1, o2 in zip(*outs))
    print(f"{a} -> hidden {H}, depth {D}, head {K}, value net {VN}; max output change {err:.2e}")
    if err > 1e-4:
        sys.exit("widening changed the outputs")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(big.state_dict(), args.out)
    xs = (torch.rand(S, generator=gen) < 0.1).to(torch.uint8) * torch.randint(0, 256, (S,), generator=gen).to(torch.uint8)
    xa = torch.randint(0, 256, (A,), generator=gen).to(torch.uint8)
    with torch.no_grad():
        v, lg, _ = big(xs[None].float() / 255, xa[None, None].float() / 255, torch.zeros(1, 0, 0), torch.zeros(1, dtype=torch.long))
    check = dict(state=xs.tolist(), actions=xa.tolist(), tokens=[], value=float(v[0]) * 100.0, logits=[float(lg[0, 0])])
    meta = types.SimpleNamespace(value_scale=100.0, data=[f"widened from {args.src}"], policy="widen", epochs=0)
    write_tnet(big, os.path.splitext(args.out)[0] + ".tnet", meta,
               Dims(dict(feature_version=args.feature_version, state_dim=S, action_dim=A, token_dim=48), 0), check)
    print(f"wrote {args.out} and its .tnet")


if __name__ == "__main__":
    main()
