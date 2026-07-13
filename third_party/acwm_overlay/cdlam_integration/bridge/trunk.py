"""Encode both-frame trunk features for action-to-latent bridge training."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

import torch  # noqa: E402

from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
from cdlam_integration.lam.train import load_init_ckpt  # noqa: E402
from cdlam_integration.bridge.decoding import mlp_decode, r2_dims  # noqa: E402
from cdlam_integration.bridge.encoder import encode_full_with_trunk  # noqa: E402

H, W = 240, 320


@torch.no_grad()
def encode_trunks(lam, frames, dev, bs=64):
    z_l, t0_l, t1_l = [], [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(frames), bs):
            v = torch.from_numpy(frames[s : s + bs]).to(dev).float() / 255.0
            e = encode_full_with_trunk(lam, v, sample=False)
            z_l.append(e["z_mu"].float().cpu().numpy())
            t0_l.append(e["trunk_ot"].float().cpu().numpy())
            t1_l.append(e["trunk_ot1"].float().cpu().numpy())
    return np.concatenate(z_l), np.concatenate(t0_l), np.concatenate(t1_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lam-ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lam = build_lam("CD_LAM_BASE", device=dev).lam
    load_init_ckpt(
        lam, Path(args.lam_ckpt), dev, lambda s: print("[load]", s, flush=True)
    )
    lam.eval().requires_grad_(False)

    d = np.load(args.cache, allow_pickle=True)
    frames = d["frames"]
    ED = d["ee_delta"]
    QE = d["q_ee"]
    split = d["split"]
    tr = np.where(split == "train")[0]
    te = np.where(split == "eval")[0]
    print(
        f"[idm] pairs={len(frames)} train={len(tr)} eval={len(te)}; encoding trunks ...",
        flush=True,
    )
    Z, T0, T1 = encode_trunks(lam, frames, dev)
    print(f"[idm] encoded ({time.time() - t0:.0f}s)", flush=True)

    pos = [0, 1, 2, 10, 11, 12]
    grip = [9, 19]
    rot = [i for i in range(20) if i not in pos and i not in grip]
    amean, astd = ED[tr].mean(0), ED[tr].std(0)
    astd[astd < 1e-6] = 1.0
    qmean, qstd = QE[tr].mean(0), QE[tr].std(0)
    qstd[qstd < 1e-6] = 1.0
    An = (ED - amean) / astd
    Qn = (QE - qmean) / qstd

    feats = {
        "z32+ctx (current)": np.concatenate([Z, T0, Qn], 1),  # what the probe uses
        "trunk_ot only (o_t)": np.concatenate([T0, Qn], 1),  # no z, no future
        "IDM trunk_ot+ot1 (upper bound)": np.concatenate(
            [T0, T1, Qn], 1
        ),  # both frames, no 32-D bottleneck
    }
    res = {}
    for name, X in feats.items():
        pm = mlp_decode(X[tr], An[tr], X[te], dev, epochs=400)
        gt = An[te]
        res[name] = {
            "ee_pos6": round(r2_dims(pm, gt, pos), 3),
            "ee_rot12": round(r2_dims(pm, gt, rot), 3),
            "ee_all20": round(r2_dims(pm, gt, list(range(20))), 3),
        }
        print(
            f"[idm] {name:34s} ee_pos={res[name]['ee_pos6']:.3f} rot={res[name]['ee_rot12']:.3f} all={res[name]['ee_all20']:.3f}",
            flush=True,
        )

    json.dump(
        {
            "results": res,
            "n_eval": len(te),
            "interp": "IDM (both-frame trunk) is the upper bound any z-decoder can reach. "
            "IDM >> z32 => 32-D bottleneck drops EE info (EXP-3 will help). "
            "IDM ~= z32 => 0.72 near the data limit.",
        },
        open(out / "idm.json", "w"),
        indent=2,
    )
    print(f"[idm] saved -> {out}/idm.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
