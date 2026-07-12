#!/usr/bin/env python3
"""G1-specific eval — multi-decoder recon + usage_gap + action metric + z health.

Per task book §17 + user supplements §A/§B/§C, each ckpt eval reports:

1. Action metric (z_mu only):
     top1, top5, gap, same_episode_share@5, dataset_leakage@5, cross_dataset
2. z health:
     z_norm (mean/std), pairwise_cos_mean, effective_rank
3. Identity guardrail:
     identity_ratio_vs_median (p10/p50/p90)
4. Camera guardrail (eval-only, NO train loss):
     fake-cam shift_x/shift_y/zoom/rotation
     single-sided shift_x/shift_y/zoom/rotation drift + cos_to_real
5. Reconstruction multi-decoder:
     v1_own       : v1 encoder + v1 decoder  (baseline)
     f1_v3_frozen : F1_v3 encoder + v1 decoder (frozen-decoder line baseline)
     g1_own       : G1 encoder + G1 decoder  (the run being evaluated)
     g1_zero_z    : G1 (encode), zero out z, G1 decoder
     g1_shuffle_z : G1 (encode), shuffle z across batch, G1 decoder
   For each: MSE / PSNR / SSIM (mean + median + p10/p90).
6. usage_gap:
     zero      = MSE(g1_zero_z) - MSE(g1_own)
     shuffle   = MSE(g1_shuffle_z) - MSE(g1_own)
   Both should be > 0; report mean / median / p10.
7. Tile PNG: frame_i | frame_j (gt) | v1_own recon | f1 recon | g1_own recon | g1_zero recon | g1_shuf recon.

Usage (called by G1 trainer, but also runnable standalone):
    python LAM_V2/tools/eval_lam_v2_g1.py \\
        --pair-index outputs/lam_v2_data_v2/pair_index_val.parquet \\
        --out outputs/lam_v2_train/G1_main/eval/step_001000.json \\
        --g1-ckpt outputs/lam_v2_train/G1_main/checkpoints/step_001000.pt \\
        --f1-baseline-ckpt outputs/lam_v2_train/F1_v3_late_block_ddp4_step7000/checkpoints/step_007000.pt \\
        --n-pairs-real 600 --n-pairs-id 200 --n-per-primitive 60 --n-recon-tile 8
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

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetune_4-30/scripts"))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from LAM_V2.tools.train_lam_action_readout import build_lam  # noqa: E402
from LAM_V2.tools._lam_v2_forward import (  # noqa: E402
    encode_full,
    decode_full,
    forward_full,
)
from LAM_V2.tools.run_action_metric_probe import (  # noqa: E402
    decode_pairs_parallel,
    cos_matrix,
    topk_primitive_purity,
    action_vs_video_gap,
    same_episode_share_topk,
    cross_dataset_retrieval,
    dataset_leakage_topk,
)
from LAM_V2.tools.run_guardrail_probe import (  # noqa: E402
    apply_camera_torch,
    norm_ratio_to_real,
    cosine_to,
)
from LAM_V2.tools.eval_lam_v2_full import (  # noqa: E402
    sample_pairs,
    sample_identity,
    effective_rank,
    pairwise_cos_mean,
)


CAMERA_TESTS_FAKE = [
    ("shift_x", 0.10),
    ("shift_y", 0.05),
    ("zoom", 0.10),
    ("rotation", 5.0),
]


# =============== ckpt loading =============================================


def load_ckpt_into(lam_inner: torch.nn.Module, ckpt_path: str, label: str = "ckpt"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    cleaned = {(k[4:] if k.startswith("lam.") else k): v for k, v in sd.items()}
    missing, unexpected = lam_inner.load_state_dict(cleaned, strict=False)
    print(
        f"[{label}] loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


# =============== forward helpers ==========================================


@torch.no_grad()
def encode_only(lam_inner, videos: torch.Tensor, device: str) -> dict:
    """Returns z_mu (B, D), patches (B, T, N, P) for downstream decoder."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = encode_full(lam_inner, videos, sample=False, use_ckpt=False)
    return {
        "z_mu": out["z_mu"].float(),
        "z_rep": out["z_rep"].float(),
        "patches": out["patches"],
    }


@torch.no_grad()
def decode_with_z(
    lam_inner, patches: torch.Tensor, z_rep: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        recon = decode_full(lam_inner, patches, z_rep, H, W, use_ckpt=False)
    return recon.float().clamp(0, 1)


@torch.no_grad()
def encode_decode(lam_inner, videos: torch.Tensor, device: str) -> dict:
    """Full forward (own encoder + own decoder), z_mu (no reparam)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = forward_full(lam_inner, videos, sample=False, use_ckpt=False)
    return {
        "z_mu": out["z_mu"].float(),
        "z_rep": out["z_rep"].float(),
        "patches": out["patches"],
        "recon": out["recon"].float().clamp(0, 1),
    }


def encode_lam(
    lam_inner, pairs_uint8: np.ndarray, device: str, batch: int = 64
) -> np.ndarray:
    out = []
    lam_inner.eval()
    for i in range(0, len(pairs_uint8), batch):
        chunk = pairs_uint8[i : i + batch]
        videos = torch.from_numpy(chunk).float().to(device) / 255.0
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc = encode_full(lam_inner, videos, sample=False, use_ckpt=False)
            z_mu = enc["z_mu"].float()
        out.append(z_mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def ssim_single(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as _ssim

        return float(_ssim(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        return 1.0 - float(np.abs(a - b).mean())


def psnr_per_pixel(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    if mse < 1e-12:
        return 99.0
    return 10.0 * float(np.log10(1.0 / mse))


def agg_dict(rows: list, keys=("mse", "psnr", "ssim")):
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows])
        out[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
        }
    return out


# =============== main =====================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g1-ckpt", required=True)
    ap.add_argument("--f1-baseline-ckpt", default=None)
    ap.add_argument("--n-pairs-real", type=int, default=600)
    ap.add_argument("--n-pairs-id", type=int, default=200)
    ap.add_argument("--n-per-primitive", type=int, default=60)
    ap.add_argument("--n-recon-tile", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--decode-workers", type=int, default=16)
    ap.add_argument("--target-h", type=int, default=240)
    ap.add_argument("--target-w", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    device = args.device
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_dir = out_path.parent / "recon_tiles"
    tile_dir.mkdir(exist_ok=True)

    print(f"[g1-eval] loading pair_index {args.pair_index}", flush=True)
    df = pd.read_parquet(args.pair_index)

    real_df = sample_pairs(df, args.n_per_primitive, seed=args.seed)
    if len(real_df) > args.n_pairs_real:
        real_df = real_df.head(args.n_pairs_real)
    id_df = sample_identity(df, args.n_pairs_id, seed=args.seed)
    print(
        f"[g1-eval] real={len(real_df)} id={len(id_df)} "
        f"prim={real_df['primitive'].value_counts().to_dict()}",
        flush=True,
    )

    real_pairs, real_valid = decode_pairs_parallel(
        real_df, target_hw=(args.target_h, args.target_w), workers=args.decode_workers
    )
    real_df = real_df.loc[real_valid].reset_index(drop=True)
    real_pairs = real_pairs[real_valid]
    id_pairs, id_valid = decode_pairs_parallel(
        id_df, target_hw=(args.target_h, args.target_w), workers=args.decode_workers
    )
    id_df = id_df.loc[id_valid].reset_index(drop=True)
    id_pairs = id_pairs[id_valid]
    print(f"[g1-eval] decoded real={len(real_df)} id={len(id_df)}", flush=True)

    # ---- build all 3 LAMs (each with its own (encoder, decoder) state)
    print("[g1-eval] building v1 LAM ...", flush=True)
    v1_lam = build_lam("CD_LAM_V1", device=device).lam
    v1_lam.eval()

    print("[g1-eval] building G1 LAM (init from v1, then load G1 ckpt) ...", flush=True)
    g1_lam = build_lam("CD_LAM_V1", device=device).lam
    load_ckpt_into(g1_lam, args.g1_ckpt, "g1")
    g1_lam.eval()

    f1_lam = None
    if args.f1_baseline_ckpt:
        print("[g1-eval] building F1_v3 baseline LAM ...", flush=True)
        f1_lam = build_lam("CD_LAM_V1", device=device).lam
        load_ckpt_into(f1_lam, args.f1_baseline_ckpt, "f1_v3")
        f1_lam.eval()

    # ---- recon comparison (per-pair MSE/PSNR/SSIM for each variant)
    H, W = args.target_h, args.target_w
    BATCH = 16
    rec_v1 = []
    rec_f1 = []
    rec_g1 = []
    rec_g1_zero = []
    rec_g1_shuf = []
    g1_recon_imgs = []
    g1_zero_imgs = []
    g1_shuf_imgs = []
    v1_recon_imgs = []
    f1_recon_imgs = []
    z_mu_g1_all = []
    print(
        f"[g1-eval] running recon comparison on {len(real_pairs)} pairs ...", flush=True
    )
    for i in range(0, len(real_pairs), BATCH):
        chunk = real_pairs[i : i + BATCH]
        videos = torch.from_numpy(chunk).float().to(device) / 255.0
        gt = videos[:, 1:]  # (B, 1, H, W, 3)

        # v1 own
        v1_out = encode_decode(v1_lam, videos, device)
        # G1 own
        g1_out = encode_decode(g1_lam, videos, device)
        z_mu_g1_all.append(g1_out["z_mu"].cpu().numpy())
        # G1 zero z (re-decode with zeros)
        z_zero = torch.zeros_like(g1_out["z_rep"])
        rec_g1z = decode_with_z(g1_lam, g1_out["patches"], z_zero, H, W)
        # G1 shuffled z
        B = videos.shape[0]
        if B >= 2:
            perm = torch.randperm(B, device=device)
            while torch.any(perm == torch.arange(B, device=device)):
                perm = torch.randperm(B, device=device)
        else:
            perm = torch.zeros(B, dtype=torch.long, device=device)
        z_shuf = g1_out["z_rep"].index_select(0, perm).contiguous()
        rec_g1s = decode_with_z(g1_lam, g1_out["patches"], z_shuf, H, W)

        # F1_v3 baseline
        if f1_lam is not None:
            f1_out = encode_decode(f1_lam, videos, device)
        else:
            f1_out = None

        gt_np = gt.cpu().numpy()  # (B, 1, H, W, 3)
        for j in range(B):
            gt_j = gt_np[j, 0]
            v1_j = v1_out["recon"][j, 0].cpu().numpy()
            g1_j = g1_out["recon"][j, 0].cpu().numpy()
            gz_j = rec_g1z[j, 0].cpu().numpy()
            gs_j = rec_g1s[j, 0].cpu().numpy()
            rec_v1.append(
                {
                    "mse": float(((gt_j - v1_j) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, v1_j),
                    "ssim": ssim_single(gt_j, v1_j),
                }
            )
            rec_g1.append(
                {
                    "mse": float(((gt_j - g1_j) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, g1_j),
                    "ssim": ssim_single(gt_j, g1_j),
                }
            )
            rec_g1_zero.append(
                {
                    "mse": float(((gt_j - gz_j) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, gz_j),
                    "ssim": ssim_single(gt_j, gz_j),
                }
            )
            rec_g1_shuf.append(
                {
                    "mse": float(((gt_j - gs_j) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, gs_j),
                    "ssim": ssim_single(gt_j, gs_j),
                }
            )
            if f1_out is not None:
                f1_j = f1_out["recon"][j, 0].cpu().numpy()
                rec_f1.append(
                    {
                        "mse": float(((gt_j - f1_j) ** 2).mean()),
                        "psnr": psnr_per_pixel(gt_j, f1_j),
                        "ssim": ssim_single(gt_j, f1_j),
                    }
                )
            else:
                f1_j = None
            # save tile imgs (first n_recon_tile only)
            if len(v1_recon_imgs) < args.n_recon_tile:
                v1_recon_imgs.append(v1_j)
                g1_recon_imgs.append(g1_j)
                g1_zero_imgs.append(gz_j)
                g1_shuf_imgs.append(gs_j)
                f1_recon_imgs.append(f1_j if f1_j is not None else np.zeros_like(g1_j))

    z_mu_g1_arr = np.concatenate(z_mu_g1_all, axis=0)

    # ---- per-pair usage_gap
    usage_gap_zero = np.array(
        [z["mse"] - r["mse"] for r, z in zip(rec_g1, rec_g1_zero)]
    )
    usage_gap_shuf = np.array(
        [s["mse"] - r["mse"] for r, s in zip(rec_g1, rec_g1_shuf)]
    )

    # ---- z health (g1)
    z_norm_mean = float(np.linalg.norm(z_mu_g1_arr, axis=1).mean())
    z_norm_std = float(np.linalg.norm(z_mu_g1_arr, axis=1).std())
    z_pcos = pairwise_cos_mean(z_mu_g1_arr)
    z_eff_rank = effective_rank(z_mu_g1_arr)

    # ---- action retrieval / gap / leakage (g1, z_mu)
    primitives = real_df["primitive"].astype(str).values
    episodes = (
        real_df["dataset"].astype(str) + "|" + real_df["episode_id"].astype(str)
    ).values
    datasets = real_df["dataset"].astype(str).values
    sim = cos_matrix(z_mu_g1_arr)
    topk = topk_primitive_purity(sim.copy(), primitives, episodes, ks=(1, 5))
    gap = action_vs_video_gap(sim.copy(), primitives, episodes)
    se_share = same_episode_share_topk(sim.copy(), episodes, k=5)
    cd_top1 = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=1)
    cd_top5 = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=5)
    leakage = dataset_leakage_topk(sim.copy(), datasets, k=5)

    # ---- identity (g1)
    z_id_g1 = encode_lam(g1_lam, id_pairs, device=device, batch=64)
    z_real_norms = np.linalg.norm(z_mu_g1_arr, axis=1)
    median_real = float(np.median(z_real_norms))
    median_id = float(np.median(np.linalg.norm(z_id_g1, axis=1)))
    id_p10 = float(
        np.percentile(np.linalg.norm(z_id_g1, axis=1), 10) / max(median_real, 1e-8)
    )
    id_p50 = float(
        np.percentile(np.linalg.norm(z_id_g1, axis=1), 50) / max(median_real, 1e-8)
    )
    id_p90 = float(
        np.percentile(np.linalg.norm(z_id_g1, axis=1), 90) / max(median_real, 1e-8)
    )

    # ---- camera guardrail (eval-only) on g1
    fake_metrics = {}
    ss_metrics = {}
    frames_a = real_pairs[:, 0]
    frames_b = real_pairs[:, 1]
    for kind, mag in CAMERA_TESTS_FAKE:
        cam_a = apply_camera_torch(frames_a, kind, mag, device=device)
        fake_in = np.stack([frames_a, cam_a], axis=1)
        z_fk = encode_lam(g1_lam, fake_in, device=device, batch=64)
        fake_metrics[f"{kind}_{mag}"] = norm_ratio_to_real(z_fk, z_real_norms)
        cam_b = apply_camera_torch(frames_b, kind, mag, device=device)
        ss_in = np.stack([frames_a, cam_b], axis=1)
        z_ss = encode_lam(g1_lam, ss_in, device=device, batch=64)
        drift = np.linalg.norm(z_ss - z_mu_g1_arr, axis=1) / (z_real_norms + 1e-8)
        ss_metrics[f"{kind}_{mag}"] = {
            "drift_p50": float(np.percentile(drift, 50)),
            "drift_p10": float(np.percentile(drift, 10)),
            "drift_p90": float(np.percentile(drift, 90)),
            "cos_to_real": cosine_to(z_ss, z_mu_g1_arr),
        }

    # ---- assemble JSON
    out = {
        "g1_ckpt": args.g1_ckpt,
        "f1_baseline_ckpt": args.f1_baseline_ckpt,
        "pair_index": args.pair_index,
        "n_real": int(len(real_df)),
        "n_id": int(len(id_df)),
        "primitive_breakdown": real_df["primitive"].value_counts().to_dict(),
        "dataset_breakdown": real_df["dataset"].value_counts().to_dict(),
        "elapsed_s": round(time.time() - t_start, 1),
        # per-decoder recon
        "recon": {
            "v1_own": agg_dict(rec_v1),
            **({"f1_v3_frozen": agg_dict(rec_f1)} if rec_f1 else {}),
            "g1_own": agg_dict(rec_g1),
            "g1_zero_z": agg_dict(rec_g1_zero),
            "g1_shuffle_z": agg_dict(rec_g1_shuf),
        },
        # usage_gap
        "g1": {
            "usage_gap": {
                "zero_mean": float(usage_gap_zero.mean()),
                "zero_p10": float(np.percentile(usage_gap_zero, 10)),
                "zero_p50": float(np.percentile(usage_gap_zero, 50)),
                "zero_p90": float(np.percentile(usage_gap_zero, 90)),
                "shuffle_mean": float(usage_gap_shuf.mean()),
                "shuffle_p10": float(np.percentile(usage_gap_shuf, 10)),
                "shuffle_p50": float(np.percentile(usage_gap_shuf, 50)),
                "shuffle_p90": float(np.percentile(usage_gap_shuf, 90)),
            },
            "z_geometry": {
                "z_mu_norm_mean": z_norm_mean,
                "z_mu_norm_std": z_norm_std,
                "pairwise_cos_mean": float(z_pcos),
                "effective_rank": float(z_eff_rank),
                "latent_dim": int(z_mu_g1_arr.shape[1]),
            },
            "retrieval": {
                "top1": float(topk[1]),
                "top5": float(topk[5]),
                "same_episode_share_top5": float(se_share),
                "dataset_leakage_top5": float(leakage),
                **gap,
            },
            "identity_ratio_vs_median": {
                "p10": id_p10,
                "p50": id_p50,
                "p90": id_p90,
                "median_real_norm": median_real,
                "median_id_norm": median_id,
            },
            "fake_camera_ratio": fake_metrics,
            "single_sided": ss_metrics,
            "cross_dataset_top1": cd_top1,
            "cross_dataset_top5": cd_top5,
        },
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[g1-eval] wrote {out_path}", flush=True)

    # ---- compact summary
    print()
    rec = out["recon"]
    print("=== recon (PSNR mean / MSE mean) ===")
    for label in ("v1_own", "f1_v3_frozen", "g1_own", "g1_zero_z", "g1_shuffle_z"):
        if label in rec:
            print(
                f"  {label:<14}: PSNR={rec[label]['psnr']['mean']:.2f} dB  MSE={rec[label]['mse']['mean']:.4f}"
            )
    ug = out["g1"]["usage_gap"]
    print("=== usage_gap (mean / p50) ===")
    print(f"  zero    : mean={ug['zero_mean']:.4f}  p50={ug['zero_p50']:.4f}")
    print(f"  shuffle : mean={ug['shuffle_mean']:.4f}  p50={ug['shuffle_p50']:.4f}")
    g = out["g1"]["z_geometry"]
    r = out["g1"]["retrieval"]
    print("=== g1 ===")
    print(f"  z_norm  : {g['z_mu_norm_mean']:.3f} ± {g['z_mu_norm_std']:.3f}")
    print(f"  eff_rank: {g['effective_rank']:.2f} / {g['latent_dim']}")
    print(f"  top1/top5: {r['top1']:.3f} / {r['top5']:.3f}")
    print(
        f"  gap     : {r['gap']:+.4f}  (same_p_diff_e={r['mean_cos_same_primitive_diff_episode']:.3f}, "
        f"diff_p_same_e={r['mean_cos_diff_primitive_same_episode']:.3f})"
    )
    print(f"  identity p50: {out['g1']['identity_ratio_vs_median']['p50']:.3f}")
    print(
        f"  same_ep@5: {r['same_episode_share_top5']:.3f}  leakage@5: {r['dataset_leakage_top5']:.3f}"
    )

    # ---- tile PNG: rows = sample pair, cols = (frame_i, gt frame_j, v1_own, f1_frozen?, g1_own, g1_zero, g1_shuf)
    n_tile = min(args.n_recon_tile, len(real_pairs))
    if n_tile <= 0:
        print(f"[g1-eval] n_recon_tile={args.n_recon_tile} → skip tile gen")
        return 0
    cols = ["frame_i", "frame_j_gt", "v1_own"]
    if rec_f1:
        cols.append("f1_frozen")
    cols += ["g1_own", "g1_zero_z", "g1_shuffle_z"]
    n_cols = len(cols)
    tile = np.zeros((n_tile * H, n_cols * W, 3), dtype=np.uint8)
    for r_idx in range(n_tile):
        col = 0
        # frame_i
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = real_pairs[
            r_idx, 0
        ]
        col += 1
        # frame_j gt
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = real_pairs[
            r_idx, 1
        ]
        col += 1
        # v1
        v1_img = (v1_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = v1_img
        col += 1
        if rec_f1:
            f1_img = (f1_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
            tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = f1_img
            col += 1
        g1_img = (g1_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = g1_img
        col += 1
        gz_img = (g1_zero_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = gz_img
        col += 1
        gs_img = (g1_shuf_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = gs_img
        col += 1

    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    tile_path = tile_dir / f"{out_path.stem}_recon_tile.png"
    cv2.imwrite(str(tile_path), tile_bgr)
    print(f"[g1-eval] tile: {tile_path}  cols={cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
