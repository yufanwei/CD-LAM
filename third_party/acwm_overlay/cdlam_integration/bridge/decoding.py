"""Decode joint-space and end-effector actions from one frozen LAM latent."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
from cdlam_integration.lam.train import load_init_ckpt  # noqa: E402
from cdlam_integration.bridge.encoder import encode_full_with_trunk  # noqa: E402

H, W = 240, 320


def ridge_pred(Ztr, Atr, Zte, alpha=1.0):
    Zb = np.concatenate([Ztr, np.ones((len(Ztr), 1))], 1)
    Zbte = np.concatenate([Zte, np.ones((len(Zte), 1))], 1)
    reg = alpha * np.eye(Zb.shape[1])
    reg[-1, -1] = 0
    W_ = np.linalg.solve(Zb.T @ Zb + reg, Zb.T @ Atr)
    return Zbte @ W_


def r2_dims(pred, gt, dims):
    d = [i for i in dims if gt[:, i].std() > 1e-6]
    if not d:
        return float("nan")
    ssr = ((gt[:, d] - pred[:, d]) ** 2).sum(0)
    sst = ((gt[:, d] - gt[:, d].mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ssr / sst).mean())


def mlp_decode(Xtr, Atr, Xte, dev, epochs=400, wd=1e-5):
    """conditioned decoder [z,ctx]->a (normalized). Returns test preds (normalized)."""
    din, dout = Xtr.shape[1], Atr.shape[1]
    net = nn.Sequential(
        nn.Linear(din, 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, dout),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    xt = torch.tensor(Xtr, device=dev)
    at = torch.tensor(Atr, device=dev)
    xe = torch.tensor(Xte, device=dev)
    for ep in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for s in range(0, len(xt), 2048):
            idx = perm[s : s + 2048]
            loss = F.mse_loss(net(xt[idx]), at[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(xe).cpu().numpy()


@torch.no_grad()
def encode_all(lam, frames, dev, bs=64):
    z_l, t_l = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(frames), bs):
            v = torch.from_numpy(frames[s : s + bs]).to(dev).float() / 255.0
            e = encode_full_with_trunk(lam, v, sample=False)
            z_l.append(e["z_mu"].float().cpu().numpy())
            t_l.append(e["trunk_ot"].float().cpu().numpy())
    return np.concatenate(z_l), np.concatenate(t_l)


def run_target(name, A, Q, Z, T, tr, te, dims_map, dev):
    """A:(N,D) target, Q:(N,Dq) proprio. Returns metrics dict."""
    amean, astd = A[tr].mean(0), A[tr].std(0)
    astd[astd < 1e-6] = 1.0
    qmean, qstd = Q[tr].mean(0), Q[tr].std(0)
    qstd[qstd < 1e-6] = 1.0
    An = (A - amean) / astd
    Qn = (Q - qmean) / qstd
    ctx = np.concatenate([T, Qn], 1)  # [trunk_ot(1024), q_norm]
    X = np.concatenate([Z, ctx], 1)  # [z, ctx]
    # M1 linear z->a (normalized)
    pr = ridge_pred(Z[tr], An[tr], Z[te])
    # M2 conditioned MLP
    pm = mlp_decode(X[tr], An[tr], X[te], dev)
    gt = An[te]
    out = {
        "M1": {},
        "M2": {},
        "per_dim_M2_R2": [
            round(float(x), 3)
            for x in (
                1 - ((gt - pm) ** 2).sum(0) / (((gt - gt.mean(0)) ** 2).sum(0) + 1e-9)
            )
        ],
    }
    for gname, dims in dims_map.items():
        out["M1"][gname] = round(r2_dims(pr, gt, dims), 3)
        out["M2"][gname] = round(r2_dims(pm, gt, dims), 3)
    return out


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
    ck_raw = torch.load(args.lam_ckpt, map_location=dev, weights_only=False)
    if isinstance(ck_raw, dict) and "lam" in ck_raw and isinstance(ck_raw["lam"], dict):
        miss, unexp = lam.load_state_dict(ck_raw["lam"], strict=False)
        print(
            f"[load] co-trained 'lam' key: missing={len(miss)} unexpected={len(unexp)} step={ck_raw.get('step')}",
            flush=True,
        )
    else:
        load_init_ckpt(
            lam, Path(args.lam_ckpt), dev, lambda s: print("[load]", s, flush=True)
        )
    lam.eval().requires_grad_(False)

    d = np.load(args.cache, allow_pickle=True)
    frames = d["frames"]
    JD = d["joint_delta"]
    ED = d["ee_delta"]
    QJ = d["q_joint"]
    QE = d["q_ee"]
    split = d["split"]
    tr = np.where(split == "train")[0]
    te = np.where(split == "eval")[0]
    print(
        f"[ee] pairs={len(frames)} train={len(tr)} eval={len(te)} JD={JD.shape} ED={ED.shape}",
        flush=True,
    )
    print("[ee] encoding z for all pairs ...", flush=True)
    Z, T = encode_all(lam, frames, dev)
    print(f"[ee] encoded ({time.time() - t0:.0f}s)", flush=True)

    # JOINT: 14-D arm joints (agibot_beta joint = 7+7, no gripper in joint)
    joint = run_target(
        "joint",
        JD,
        QJ,
        Z,
        T,
        tr,
        te,
        {
            "arm14": list(range(14)),
            "left7": list(range(7)),
            "right7": list(range(7, 14)),
        },
        dev,
    )
    # EE: per-arm [Δp(3) Δr6(6) grip(1)] x2 -> 20. pos={0,1,2,10,11,12} rot=rest grip={9,19}
    pos = [0, 1, 2, 10, 11, 12]
    grip = [9, 19]
    rot = [i for i in range(20) if i not in pos and i not in grip]
    ee = run_target(
        "ee",
        ED,
        QE,
        Z,
        T,
        tr,
        te,
        {
            "ee_all20": list(range(20)),
            "ee_pos6": pos,
            "ee_rot12": rot,
            "ee_grip2": grip,
        },
        dev,
    )

    res = {
        "joint": joint,
        "ee": ee,
        "n_train": len(tr),
        "n_eval": len(te),
        "interp": "JOINT vs EE decoded from identical frozen release-B z (agibot_beta, episode-disjoint). "
        "joint-arm14 vs ee_pos6 is the headline; ee_pos>>0.44 => 0.44 wall was a wrong-target artifact.",
    }
    json.dump(res, open(out / "ee_vs_joint.json", "w"), indent=2)
    print("\n=== RESULT (M2 conditioned decoder R^2) ===", flush=True)
    print(
        f"  JOINT arm14 : M1={joint['M1']['arm14']}  M2={joint['M2']['arm14']}",
        flush=True,
    )
    print(
        f"  EE  pos6    : M1={ee['M1']['ee_pos6']}  M2={ee['M2']['ee_pos6']}   <-- headline",
        flush=True,
    )
    print(
        f"  EE  rot12   : M2={ee['M2']['ee_rot12']}   grip2: M2={ee['M2']['ee_grip2']}   all20: M2={ee['M2']['ee_all20']}",
        flush=True,
    )
    print(f"[ee] saved -> {out}/ee_vs_joint.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
