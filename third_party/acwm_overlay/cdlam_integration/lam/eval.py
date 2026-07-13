#!/usr/bin/env python3
"""Stage-1 evaluation: reconstruction, latent usage, and action metrics.

Each checkpoint evaluation reports:

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
     base       : baseline encoder + baseline decoder  (baseline)
     reference_checkpoint : baseline encoder + baseline decoder (frozen-decoder line baseline)
     trained       : trained model encoder + trained model decoder  (the run being evaluated)
     zero_latent    : trained model (encode), zero out z, trained model decoder
     shuffled_latent : trained model (encode), shuffle z across batch, trained model decoder
   For each: MSE / PSNR / SSIM (mean + median + p10/p90).
6. usage_gap:
     zero      = MSE(zero_latent) - MSE(trained)
     shuffle   = MSE(shuffled_latent) - MSE(trained)
   Both should be > 0; report mean / median / p10.
7. Tile PNG: frame_i | frame_j (gt) | base recon | reference recon | trained recon | zero_latent recon | shuffled_latent recon.

Usage (called by trained model trainer, but also runnable standalone):
    python -m cdlam_runtime.entries.stage1_eval \\
        --pair-index outputs/cdlam_data/pair_index_val.parquet \\
        --out outputs/stage1/eval/step_001000.json \\
        --checkpoint outputs/stage1/checkpoints/step_001000.pt \\
        --baseline-checkpoint outputs/cdlam_train/baseline_late_block_ddp4_step7000/checkpoints/step_007000.pt \\
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

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
from cdlam_integration.lam.model_ops import (  # noqa: E402
    encode_full,
    decode_full,
    forward_full,
)
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
from cdlam_integration.lam.eval_protocol import (  # noqa: E402
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline-checkpoint", default=None)
    ap.add_argument("--n-pairs-real", type=int, default=600)
    ap.add_argument("--n-pairs-id", type=int, default=200)
    ap.add_argument("--n-per-primitive", type=int, default=60)
    ap.add_argument("--n-recon-tile", type=int, default=8)
    ap.add_argument("--decode-workers", type=int, default=16)
    ap.add_argument("--target-h", type=int, default=240)
    ap.add_argument("--target-w", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    device = "cuda:0"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_dir = out_path.parent / "recon_tiles"
    tile_dir.mkdir(exist_ok=True)

    print(f"[stage1-eval] loading pair_index {args.pair_index}", flush=True)
    df = pd.read_parquet(args.pair_index)

    real_df = sample_pairs(df, args.n_per_primitive, seed=args.seed)
    if len(real_df) > args.n_pairs_real:
        real_df = real_df.head(args.n_pairs_real)
    id_df = sample_identity(df, args.n_pairs_id, seed=args.seed)
    print(
        f"[stage1-eval] real={len(real_df)} id={len(id_df)} "
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
    print(f"[stage1-eval] decoded real={len(real_df)} id={len(id_df)}", flush=True)

    # ---- build all 3 LAMs (each with its own (encoder, decoder) state)
    print("[stage1-eval] building baseline LAM ...", flush=True)
    base_lam = build_lam("CD_LAM", device=device).lam
    base_lam.eval()

    print("[stage1-eval] building trained model LAM (init from baseline, then load trained model ckpt) ...", flush=True)
    trained_lam = build_lam("CD_LAM", device=device).lam
    load_ckpt_into(trained_lam, args.checkpoint, "trained")
    trained_lam.eval()

    reference_lam = None
    if args.baseline_checkpoint:
        print("[stage1-eval] building reference checkpoint LAM ...", flush=True)
        reference_lam = build_lam("CD_LAM", device=device).lam
        load_ckpt_into(reference_lam, args.baseline_checkpoint, "baseline")
        reference_lam.eval()

    # ---- recon comparison (per-pair MSE/PSNR/SSIM for each variant)
    H, W = args.target_h, args.target_w
    BATCH = 16
    rec_base = []
    rec_reference = []
    rec_trained = []
    rec_zero_latent = []
    rec_shuffled_latent = []
    trained_recon_imgs = []
    zero_latent_recon_imgs = []
    shuffled_latent_recon_imgs = []
    base_recon_imgs = []
    reference_recon_imgs = []
    z_mu_trained_all = []
    print(
        f"[stage1-eval] running recon comparison on {len(real_pairs)} pairs ...", flush=True
    )
    for i in range(0, len(real_pairs), BATCH):
        chunk = real_pairs[i : i + BATCH]
        videos = torch.from_numpy(chunk).float().to(device) / 255.0
        gt = videos[:, 1:]  # (B, 1, H, W, 3)

        # baseline own
        base_out = encode_decode(base_lam, videos, device)
        # trained model own
        trained_out = encode_decode(trained_lam, videos, device)
        z_mu_trained_all.append(trained_out["z_mu"].cpu().numpy())
        # trained model zero z (re-decode with zeros)
        z_zero = torch.zeros_like(trained_out["z_rep"])
        rec_trainedz = decode_with_z(trained_lam, trained_out["patches"], z_zero, H, W)
        # trained model shuffled z
        B = videos.shape[0]
        if B >= 2:
            perm = torch.randperm(B, device=device)
            while torch.any(perm == torch.arange(B, device=device)):
                perm = torch.randperm(B, device=device)
        else:
            perm = torch.zeros(B, dtype=torch.long, device=device)
        z_shuf = trained_out["z_rep"].index_select(0, perm).contiguous()
        rec_traineds = decode_with_z(trained_lam, trained_out["patches"], z_shuf, H, W)

        # reference checkpoint
        if reference_lam is not None:
            reference_out = encode_decode(reference_lam, videos, device)
        else:
            reference_out = None

        gt_np = gt.cpu().numpy()  # (B, 1, H, W, 3)
        for j in range(B):
            gt_j = gt_np[j, 0]
            base_frame = base_out["recon"][j, 0].cpu().numpy()
            trained_frame = trained_out["recon"][j, 0].cpu().numpy()
            zero_latent_frame = rec_trainedz[j, 0].cpu().numpy()
            shuffled_latent_frame = rec_traineds[j, 0].cpu().numpy()
            rec_base.append(
                {
                    "mse": float(((gt_j - base_frame) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, base_frame),
                    "ssim": ssim_single(gt_j, base_frame),
                }
            )
            rec_trained.append(
                {
                    "mse": float(((gt_j - trained_frame) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, trained_frame),
                    "ssim": ssim_single(gt_j, trained_frame),
                }
            )
            rec_zero_latent.append(
                {
                    "mse": float(((gt_j - zero_latent_frame) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, zero_latent_frame),
                    "ssim": ssim_single(gt_j, zero_latent_frame),
                }
            )
            rec_shuffled_latent.append(
                {
                    "mse": float(((gt_j - shuffled_latent_frame) ** 2).mean()),
                    "psnr": psnr_per_pixel(gt_j, shuffled_latent_frame),
                    "ssim": ssim_single(gt_j, shuffled_latent_frame),
                }
            )
            if reference_out is not None:
                reference_frame = reference_out["recon"][j, 0].cpu().numpy()
                rec_reference.append(
                    {
                        "mse": float(((gt_j - reference_frame) ** 2).mean()),
                        "psnr": psnr_per_pixel(gt_j, reference_frame),
                        "ssim": ssim_single(gt_j, reference_frame),
                    }
                )
            else:
                reference_frame = None
            # save tile imgs (first n_recon_tile only)
            if len(base_recon_imgs) < args.n_recon_tile:
                base_recon_imgs.append(base_frame)
                trained_recon_imgs.append(trained_frame)
                zero_latent_recon_imgs.append(zero_latent_frame)
                shuffled_latent_recon_imgs.append(shuffled_latent_frame)
                reference_recon_imgs.append(reference_frame if reference_frame is not None else np.zeros_like(trained_frame))

    z_mu_trained = np.concatenate(z_mu_trained_all, axis=0)

    # ---- per-pair usage_gap
    usage_gap_zero = np.array(
        [z["mse"] - r["mse"] for r, z in zip(rec_trained, rec_zero_latent)]
    )
    usage_gap_shuf = np.array(
        [s["mse"] - r["mse"] for r, s in zip(rec_trained, rec_shuffled_latent)]
    )

    # ---- z health (trained model)
    z_norm_mean = float(np.linalg.norm(z_mu_trained, axis=1).mean())
    z_norm_std = float(np.linalg.norm(z_mu_trained, axis=1).std())
    z_pcos = pairwise_cos_mean(z_mu_trained)
    z_eff_rank = effective_rank(z_mu_trained)

    # ---- action retrieval / gap / leakage (trained model, z_mu)
    primitives = real_df["primitive"].astype(str).values
    episodes = (
        real_df["dataset"].astype(str) + "|" + real_df["episode_id"].astype(str)
    ).values
    datasets = real_df["dataset"].astype(str).values
    sim = cos_matrix(z_mu_trained)
    topk = topk_primitive_purity(sim.copy(), primitives, episodes, ks=(1, 5))
    gap = action_vs_video_gap(sim.copy(), primitives, episodes)
    se_share = same_episode_share_topk(sim.copy(), episodes, k=5)
    cd_top1 = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=1)
    cd_top5 = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=5)
    leakage = dataset_leakage_topk(sim.copy(), datasets, k=5)

    # ---- identity (trained model)
    z_id_trained = encode_lam(trained_lam, id_pairs, device=device, batch=64)
    z_real_norms = np.linalg.norm(z_mu_trained, axis=1)
    median_real = float(np.median(z_real_norms))
    median_id = float(np.median(np.linalg.norm(z_id_trained, axis=1)))
    id_p10 = float(
        np.percentile(np.linalg.norm(z_id_trained, axis=1), 10) / max(median_real, 1e-8)
    )
    id_p50 = float(
        np.percentile(np.linalg.norm(z_id_trained, axis=1), 50) / max(median_real, 1e-8)
    )
    id_p90 = float(
        np.percentile(np.linalg.norm(z_id_trained, axis=1), 90) / max(median_real, 1e-8)
    )

    # ---- camera guardrail (eval-only) on the trained model
    fake_metrics = {}
    ss_metrics = {}
    frames_a = real_pairs[:, 0]
    frames_b = real_pairs[:, 1]
    for kind, mag in CAMERA_TESTS_FAKE:
        cam_a = apply_camera_torch(frames_a, kind, mag, device=device)
        fake_in = np.stack([frames_a, cam_a], axis=1)
        z_fk = encode_lam(trained_lam, fake_in, device=device, batch=64)
        fake_metrics[f"{kind}_{mag}"] = norm_ratio_to_real(z_fk, z_real_norms)
        cam_b = apply_camera_torch(frames_b, kind, mag, device=device)
        ss_in = np.stack([frames_a, cam_b], axis=1)
        z_ss = encode_lam(trained_lam, ss_in, device=device, batch=64)
        drift = np.linalg.norm(z_ss - z_mu_trained, axis=1) / (z_real_norms + 1e-8)
        ss_metrics[f"{kind}_{mag}"] = {
            "drift_p50": float(np.percentile(drift, 50)),
            "drift_p10": float(np.percentile(drift, 10)),
            "drift_p90": float(np.percentile(drift, 90)),
            "cos_to_real": cosine_to(z_ss, z_mu_trained),
        }

    # ---- assemble JSON
    out = {
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint,
        "pair_index": args.pair_index,
        "n_real": int(len(real_df)),
        "n_id": int(len(id_df)),
        "primitive_breakdown": real_df["primitive"].value_counts().to_dict(),
        "dataset_breakdown": real_df["dataset"].value_counts().to_dict(),
        "elapsed_s": round(time.time() - t_start, 1),
        # per-decoder recon
        "recon": {
            "base": agg_dict(rec_base),
            **({"reference_checkpoint": agg_dict(rec_reference)} if rec_reference else {}),
            "trained": agg_dict(rec_trained),
            "zero_latent": agg_dict(rec_zero_latent),
            "shuffled_latent": agg_dict(rec_shuffled_latent),
        },
        # usage_gap
        "trained": {
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
                "latent_dim": int(z_mu_trained.shape[1]),
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
    print(f"[stage1-eval] wrote {out_path}", flush=True)

    # ---- compact summary
    print()
    rec = out["recon"]
    print("=== recon (PSNR mean / MSE mean) ===")
    for label in ("base", "reference_checkpoint", "trained", "zero_latent", "shuffled_latent"):
        if label in rec:
            print(
                f"  {label:<14}: PSNR={rec[label]['psnr']['mean']:.2f} dB  MSE={rec[label]['mse']['mean']:.4f}"
            )
    ug = out["trained"]["usage_gap"]
    print("=== usage_gap (mean / p50) ===")
    print(f"  zero    : mean={ug['zero_mean']:.4f}  p50={ug['zero_p50']:.4f}")
    print(f"  shuffle : mean={ug['shuffle_mean']:.4f}  p50={ug['shuffle_p50']:.4f}")
    g = out["trained"]["z_geometry"]
    r = out["trained"]["retrieval"]
    print("=== trained model ===")
    print(f"  z_norm  : {g['z_mu_norm_mean']:.3f} ± {g['z_mu_norm_std']:.3f}")
    print(f"  eff_rank: {g['effective_rank']:.2f} / {g['latent_dim']}")
    print(f"  top1/top5: {r['top1']:.3f} / {r['top5']:.3f}")
    print(
        f"  gap     : {r['gap']:+.4f}  (same_p_diff_e={r['mean_cos_same_primitive_diff_episode']:.3f}, "
        f"diff_p_same_e={r['mean_cos_diff_primitive_same_episode']:.3f})"
    )
    print(f"  identity p50: {out['trained']['identity_ratio_vs_median']['p50']:.3f}")
    print(
        f"  same_ep@5: {r['same_episode_share_top5']:.3f}  leakage@5: {r['dataset_leakage_top5']:.3f}"
    )

    # ---- tile PNG: rows = sample pair, cols = (frame_i, gt frame_j, base, reference_checkpoint?, trained, zero_latent, shuffled_latent)
    n_tile = min(args.n_recon_tile, len(real_pairs))
    if n_tile <= 0:
        print(f"[stage1-eval] n_recon_tile={args.n_recon_tile} → skip tile gen")
        return 0
    cols = ["frame_i", "frame_j_gt", "base"]
    if rec_reference:
        cols.append("reference_checkpoint")
    cols += ["trained", "zero_latent", "shuffled_latent"]
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
        # baseline
        base_image = (base_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = base_image
        col += 1
        if rec_reference:
            reference_image = (reference_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
            tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = reference_image
            col += 1
        trained_image = (trained_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = trained_image
        col += 1
        gz_img = (zero_latent_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = gz_img
        col += 1
        gs_img = (shuffled_latent_recon_imgs[r_idx] * 255).clip(0, 255).astype(np.uint8)
        tile[r_idx * H : (r_idx + 1) * H, col * W : (col + 1) * W] = gs_img
        col += 1

    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    tile_path = tile_dir / f"{out_path.stem}_recon_tile.png"
    cv2.imwrite(str(tile_path), tile_bgr)
    print(f"[stage1-eval] tile: {tile_path}  cols={cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
