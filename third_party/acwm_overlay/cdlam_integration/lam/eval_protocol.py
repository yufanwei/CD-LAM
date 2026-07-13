#!/usr/bin/env python3
"""Unified LAM core eval — runs the M0 baseline metric suite on any encoder.

Encoder selection:
    --encoder-mode baseline            : use baseline LAM (build_lam("CD_LAM"))
    --encoder-mode f1 --ckpt PATH: load F1-trained LAM ckpt (full state_dict
                                    with encoder/decoder/fc/action_prompt)

Metric suite (task book §12 / NOTE §11):
    1. z geometry on real pairs:
         z_mu_norm_mean, z_mu_norm_std, pairwise_cos_mean,
         effective_rank (entropy of singular values)
    2. Identity ratio: ||E(o,o)|| / ||E_real||  p10 p50 p90
    3. Fake-camera ratio for shift_x / shift_y / zoom / rotation
    4. Single-sided drift / cos for shift_x / shift_y / zoom / rotation
    5. Cross-episode primitive top-1 / top-5
    6. action-vs-video gap = mean cos(same_prim_diff_ep) - mean cos(diff_prim_same_ep)
    7. Same-episode share top-5
    8. Cross-dataset retrieval top-1 / top-5
    9. Dataset leakage top-5

Output: a single JSON with all numbers + sample size + provenance.

Usage:
    # M0 baseline on baseline
    python cdlam_integration/tools/eval_cdlam_full.py \
        --pair-index outputs/cdlam_data/pair_index_val.parquet \
        --out outputs/cdlam_train/F1_all_encoder_gen_gap_guard/M0_baseline.json \
        --encoder-mode baseline \
        --n-pairs-real 800 --n-pairs-id 200

    # F1 checkpoint eval
    python cdlam_integration/tools/eval_cdlam_full.py \
        --pair-index outputs/cdlam_data/pair_index_val.parquet \
        --out .../eval_step_001000.json \
        --encoder-mode f1 --ckpt .../checkpoints/step_001000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
from cdlam_integration.lam.model_ops import encode_full  # noqa: E402
from cdlam_integration.lam.eval_metrics import (  # noqa: E402
    decode_pairs_parallel,
    cos_matrix,
    topk_primitive_purity,
    action_vs_video_gap,
    same_episode_share_topk,
    cross_dataset_retrieval,
    dataset_leakage_topk,
)
from cdlam_integration.lam.eval_guardrails import (  # noqa: E402
    apply_camera_torch,
    norm_ratio_to_real,
    cosine_to,
)


CAMERA_TESTS_FAKE = [
    ("shift_x", 0.10),
    ("shift_y", 0.05),
    ("zoom", 0.10),
    ("rotation", 5.0),
]
CAMERA_TESTS_SS = [
    ("shift_x", 0.10),
    ("shift_y", 0.05),
    ("zoom", 0.10),
    ("rotation", 5.0),
]


def sample_pairs(
    df: pd.DataFrame, n_per_primitive: int, seed: int = 0, require_eligible: bool = True
) -> pd.DataFrame:
    """Balanced sample of REAL pairs across canonical primitives × datasets.
    Only takes pair_type=='real' AND eligible_for_lgap=True (so primitive is canonical).
    """
    sub = df[df["pair_type"] == "real"]
    if require_eligible:
        sub = sub[sub["eligible_for_lgap"]]
    sub = sub[sub["primitive"].astype(str) != ""]
    chunks = []
    for (ds, prim), g in sub.groupby(["dataset", "primitive"]):
        n = min(n_per_primitive, len(g))
        if n == 0:
            continue
        chunks.append(g.sample(n=n, random_state=seed))
    if not chunks:
        return sub.head(0)
    return (
        pd.concat(chunks, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def sample_identity(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    sub = df[df["pair_type"] == "identity"]
    if len(sub) == 0:
        return sub
    return sub.sample(n=min(n, len(sub)), random_state=seed).reset_index(drop=True)


def effective_rank(z: np.ndarray) -> float:
    """Entropy-based effective rank from singular values."""
    if len(z) < 2:
        return float("nan")
    z_centered = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(z_centered, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-12]
    H = -np.sum(s * np.log(s))
    return float(np.exp(H))


def pairwise_cos_mean(z: np.ndarray) -> float:
    n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sim = n @ n.T
    iu = np.triu_indices(len(sim), k=1)
    return float(sim[iu].mean())


def encode_lam(
    lam_inner, pairs_uint8: np.ndarray, device: str, batch: int = 64
) -> np.ndarray:
    """pairs_uint8: (N, 2, H, W, 3) uint8 RGB. Returns z_mu (N, latent_dim) numpy float32."""
    out = []
    lam_inner.eval()
    for i in range(0, len(pairs_uint8), batch):
        chunk = pairs_uint8[i : i + batch]
        videos = torch.from_numpy(chunk).float().to(device) / 255.0
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc = encode_full(lam_inner, videos, sample=False)
            z_mu = enc["z_mu"].float()
        out.append(z_mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_encoder(args, device: str):
    if args.encoder_mode == "baseline":
        lam = build_lam("CD_LAM", device=device)
        return lam.lam, "baseline"
    elif args.encoder_mode == "f1":
        # build skeleton then load F1 ckpt over it
        lam = build_lam("CD_LAM", device=device)  # gives correct architecture
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        # Strip "lam." prefix if present (matching how baseline was saved)
        cleaned = {(k[4:] if k.startswith("lam.") else k): v for k, v in sd.items()}
        missing, unexpected = lam.lam.load_state_dict(cleaned, strict=False)
        print(
            f"[eval] loaded F1 ckpt {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        return lam.lam, f"f1@{Path(args.ckpt).name}"
    else:
        raise ValueError(args.encoder_mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder-mode", choices=["baseline", "f1"], default="baseline")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-pairs-real", type=int, default=800)
    ap.add_argument("--n-pairs-id", type=int, default=200)
    ap.add_argument("--n-per-primitive", type=int, default=80)
    ap.add_argument("--decode-workers", type=int, default=16)
    ap.add_argument("--target-h", type=int, default=240)
    ap.add_argument("--target-w", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda:0"

    t_global = time.time()
    print(f"[eval] loading pair_index {args.pair_index}", flush=True)
    df = pd.read_parquet(args.pair_index)
    print(
        f"[eval]   rows={len(df):,}  pair_type={df['pair_type'].value_counts().to_dict()}",
        flush=True,
    )

    # ---- sample real (balanced over canonical primitives × datasets)
    real_df = sample_pairs(df, args.n_per_primitive, seed=args.seed)
    if len(real_df) > args.n_pairs_real:
        real_df = real_df.head(args.n_pairs_real)
    print(
        f"[eval] real sample={len(real_df)}  primitives={real_df['primitive'].value_counts().to_dict()}",
        flush=True,
    )
    print(
        f"[eval]   datasets={real_df['dataset'].value_counts().to_dict()}", flush=True
    )

    id_df = sample_identity(df, args.n_pairs_id, seed=args.seed)
    print(f"[eval] identity sample={len(id_df)}", flush=True)

    # ---- decode
    print("[eval] decoding real pairs ...", flush=True)
    real_pairs, real_valid = decode_pairs_parallel(
        real_df, target_hw=(args.target_h, args.target_w), workers=args.decode_workers
    )
    real_df = real_df.loc[real_valid].reset_index(drop=True)
    real_pairs = real_pairs[real_valid]
    print(f"[eval]   {len(real_df)} valid real after decode", flush=True)

    print("[eval] decoding identity pairs ...", flush=True)
    id_pairs, id_valid = decode_pairs_parallel(
        id_df, target_hw=(args.target_h, args.target_w), workers=args.decode_workers
    )
    id_df = id_df.loc[id_valid].reset_index(drop=True)
    id_pairs = id_pairs[id_valid]
    print(f"[eval]   {len(id_df)} valid identity after decode", flush=True)

    # ---- build encoder
    print(f"[eval] building encoder mode={args.encoder_mode}", flush=True)
    lam_inner, enc_tag = build_encoder(args, device=device)

    # ---- encode real
    print(f"[eval] encoding {len(real_pairs)} real pairs ...", flush=True)
    t0 = time.time()
    z_real = encode_lam(lam_inner, real_pairs, device=device, batch=64)
    print(f"[eval]   {time.time() - t0:.1f}s, z_real shape={z_real.shape}", flush=True)
    z_real_norms = np.linalg.norm(z_real, axis=1)

    # ---- encode identity
    print(f"[eval] encoding {len(id_pairs)} identity pairs ...", flush=True)
    z_id = encode_lam(lam_inner, id_pairs, device=device, batch=64)

    # ---- build fake-camera & single-sided pairs
    fake_metrics = {}
    ss_metrics = {}
    frames_a = real_pairs[:, 0]
    frames_b = real_pairs[:, 1]
    for kind, mag in CAMERA_TESTS_FAKE:
        cam_a = apply_camera_torch(frames_a, kind, mag, device=device)
        fake_pairs = np.stack([frames_a, cam_a], axis=1)
        z_fk = encode_lam(lam_inner, fake_pairs, device=device, batch=64)
        fake_metrics[f"{kind}_{mag}"] = norm_ratio_to_real(z_fk, z_real_norms)
        del cam_a, fake_pairs

    for kind, mag in CAMERA_TESTS_SS:
        cam_b = apply_camera_torch(frames_b, kind, mag, device=device)
        ss_pairs = np.stack([frames_a, cam_b], axis=1)
        z_ss = encode_lam(lam_inner, ss_pairs, device=device, batch=64)
        drift = np.linalg.norm(z_ss - z_real, axis=1) / (z_real_norms + 1e-8)
        ss_metrics[f"{kind}_{mag}"] = {
            "drift_p10": float(np.percentile(drift, 10)),
            "drift_p50": float(np.percentile(drift, 50)),
            "drift_p90": float(np.percentile(drift, 90)),
            "cos_to_real": cosine_to(z_ss, z_real),
        }
        del cam_b, ss_pairs

    # ---- z-geometry on real
    z_norm_mean = float(z_real_norms.mean())
    z_norm_std = float(z_real_norms.std())
    eff_rank = effective_rank(z_real)
    pcos = pairwise_cos_mean(z_real)

    # ---- retrieval / gap / leakage
    primitives = real_df["primitive"].astype(str).values
    episodes = (
        real_df["dataset"].astype(str) + "|" + real_df["episode_id"].astype(str)
    ).values
    datasets = real_df["dataset"].astype(str).values

    sim = cos_matrix(z_real)
    topk = topk_primitive_purity(sim.copy(), primitives, episodes, ks=(1, 5))
    gap = action_vs_video_gap(sim.copy(), primitives, episodes)
    se_share = same_episode_share_topk(sim.copy(), episodes, k=5)
    cd_top1 = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=1)
    cd_top5 = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=5)
    leakage = dataset_leakage_topk(sim.copy(), datasets, k=5)

    out = {
        "encoder": enc_tag,
        "encoder_mode": args.encoder_mode,
        "ckpt": args.ckpt,
        "pair_index": args.pair_index,
        "n_real": int(len(real_df)),
        "n_identity": int(len(id_df)),
        "primitive_breakdown": real_df["primitive"].value_counts().to_dict(),
        "dataset_breakdown": real_df["dataset"].value_counts().to_dict(),
        "z_geometry": {
            "z_mu_norm_mean": z_norm_mean,
            "z_mu_norm_std": z_norm_std,
            "pairwise_cos_mean": pcos,
            "effective_rank": eff_rank,
            "latent_dim": int(z_real.shape[1]),
        },
        "identity_ratio": norm_ratio_to_real(
            z_id,
            z_real_norms[: len(z_id)]
            if len(z_real_norms) >= len(z_id)
            else z_real_norms,
        ),
        # Note: identity ratio uses real z norm scale (compares to overall median).
        "identity_ratio_vs_median": {
            "p10": float(
                np.percentile(np.linalg.norm(z_id, axis=1), 10)
                / max(np.median(z_real_norms), 1e-8)
            ),
            "p50": float(
                np.percentile(np.linalg.norm(z_id, axis=1), 50)
                / max(np.median(z_real_norms), 1e-8)
            ),
            "p90": float(
                np.percentile(np.linalg.norm(z_id, axis=1), 90)
                / max(np.median(z_real_norms), 1e-8)
            ),
            "median_real_norm": float(np.median(z_real_norms)),
            "median_id_norm": float(np.median(np.linalg.norm(z_id, axis=1))),
        },
        "fake_camera_ratio": fake_metrics,
        "single_sided": ss_metrics,
        "retrieval": {
            "top1": float(topk[1]),
            "top5": float(topk[5]),
            "same_episode_share_top5": float(se_share),
            "dataset_leakage_top5": float(leakage),
            **gap,
        },
        "cross_dataset_top1": cd_top1,
        "cross_dataset_top5": cd_top5,
        "elapsed_s": round(time.time() - t_global, 1),
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2, default=str))
    print(f"[eval] wrote {out_p}", flush=True)

    print("\n=== eval summary ===")
    print(f"encoder              {enc_tag}")
    print(f"n_real / n_id        {len(real_df)} / {len(id_df)}")
    g = out["z_geometry"]
    print(f"z_norm mean/std      {g['z_mu_norm_mean']:.3f} / {g['z_mu_norm_std']:.3f}")
    print(f"pairwise cos mean    {g['pairwise_cos_mean']:.3f}")
    print(f"effective rank       {g['effective_rank']:.2f} / {g['latent_dim']}")
    print(
        f"primitive top1/top5  {out['retrieval']['top1']:.3f} / {out['retrieval']['top5']:.3f}"
    )
    print(
        f"action-vs-video gap  {out['retrieval']['gap']:.4f}  ("
        f"same_p_diff_e={out['retrieval']['mean_cos_same_primitive_diff_episode']:.3f}, "
        f"diff_p_same_e={out['retrieval']['mean_cos_diff_primitive_same_episode']:.3f})"
    )
    print(f"same_ep@5            {out['retrieval']['same_episode_share_top5']:.3f}")
    print(f"dataset leakage @5   {out['retrieval']['dataset_leakage_top5']:.3f}")
    print(
        f"identity p50 / median_real_norm  {out['identity_ratio_vs_median']['p50']:.3f}"
    )
    for k, mag in CAMERA_TESTS_FAKE:
        print(
            f"fake-cam {k:<10}{mag:<5} p50  {out['fake_camera_ratio'][f'{k}_{mag}']['p50']:.3f}"
        )
    for k, mag in CAMERA_TESTS_SS:
        v = out["single_sided"][f"{k}_{mag}"]
        print(
            f"ss       {k:<10}{mag:<5} drift p50  {v['drift_p50']:.3f}  cos {v['cos_to_real']['p50']:.3f}"
        )
    print(f"cross-dataset top1   {out['cross_dataset_top1']}")
    print(f"cross-dataset top5   {out['cross_dataset_top5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
