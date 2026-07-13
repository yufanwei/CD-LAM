#!/usr/bin/env python3
"""Guardrail probe — identity / fake-camera / single-sided invariance check
for the LAM core readout head. Mirrors the baseline LAM camera-clean criteria from
NOTE §11.3. Runs on val split.

Three properties to verify:

  L_id   :  E_new(o, o)  ≈ 0          (identity pair → no action)
  L_cam  :  E_new(o, T_cam(o)) ≈ 0    (pure camera motion → no action)
  L_ss   :  E_new(o_i, T_cam(o_j)) ≈ E_new(o_i, o_j)
                                      (camera should add no extra action signal)

Reports ratios relative to a real pair's z norm so the absolute scale of
the head doesn't matter.

Usage:
    python cdlam_integration/tools/run_guardrail_probe.py \\
        --pair-index outputs/cdlam_data/pair_index_scored.parquet \\
        --head-ckpt outputs/cdlam_train/T0_R2_anti_collapse/checkpoints/final.pt \\
        --base-lam CD_LAM \\
        --out outputs/cdlam_probe/guardrail_T0_R2_anti_collapse.json \\
        --n-pairs 800
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from cdlam_integration.lam.eval_metrics import sample_balanced, decode_pairs_parallel  # noqa: E402
from cdlam_integration.lam.model_loader import (  # noqa: E402
    build_lam,
    encoder_forward_with_layers,
    pool_action_token,
    LayerMixerReadout,
)
from cdlam_integration.lam.residual_adapter import ResidualAdapter  # noqa: E402


# Camera magnitudes — same defaults as the camera guardrail benchmark.
CAMERA_TESTS = [
    ("shift_x", 0.10),
    ("shift_y", 0.05),
    ("zoom", 0.10),
    ("rotation", 5.0),  # degrees
]


def apply_camera_torch(
    frames_uint8: np.ndarray, kind: str, mag: float, device: str = "cuda"
) -> np.ndarray:
    """frames_uint8: (N, H, W, 3) uint8. Return same shape uint8."""
    from cdlam_integration.lam.transforms import (
        _apply,
        rotation_theta,
        shift_theta,
        zoom_theta,
    )

    x = torch.from_numpy(frames_uint8).float().to(device) / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    N, _, H, W = x.shape
    if kind == "shift_x":
        th = shift_theta(torch.full((N,), mag), torch.zeros(N), H, W, device)
    elif kind == "shift_y":
        th = shift_theta(torch.zeros(N), torch.full((N,), mag), H, W, device)
    elif kind == "zoom":
        th = zoom_theta(torch.full((N,), mag), device)
    elif kind == "rotation":
        th = rotation_theta(torch.full((N,), mag), device)
    else:
        raise ValueError(kind)
    y = _apply(x, th)
    y = y.permute(0, 2, 3, 1).clamp(0, 1) * 255.0
    return y.to(torch.uint8).cpu().numpy()


def encode_pairs(
    lam, head, pairs_uint8: np.ndarray, device: str = "cuda", batch: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """pairs_uint8: (N, 2, H, W, 3). Returns (z_mu, z_metric)."""
    z_mu_all, z_metric_all = [], []
    for i in range(0, len(pairs_uint8), batch):
        chunk = pairs_uint8[i : i + batch]
        videos = torch.from_numpy(chunk).float().to(device) / 255.0
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                layer_acts, z_mu = encoder_forward_with_layers(lam, videos)
            h = torch.stack(
                [pool_action_token(h_l) for h_l in layer_acts], dim=1
            ).float()
            z_mu_f = z_mu.float()
            z_metric = head(h, z_mu_f) if head is not None else z_mu_f
        z_mu_all.append(z_mu_f.cpu().numpy())
        z_metric_all.append(z_metric.cpu().numpy())
    return np.concatenate(z_mu_all, 0), np.concatenate(z_metric_all, 0)


def norm_ratio_to_real(z_test: np.ndarray, z_real_norms: np.ndarray) -> dict:
    n = np.linalg.norm(z_test, axis=1)
    ratios = n / (z_real_norms + 1e-8)
    return {
        "p10": float(np.percentile(ratios, 10)),
        "p50": float(np.percentile(ratios, 50)),
        "p90": float(np.percentile(ratios, 90)),
        "mean": float(ratios.mean()),
    }


def cosine_to(z_a: np.ndarray, z_b: np.ndarray) -> dict:
    a = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + 1e-8)
    b = z_b / (np.linalg.norm(z_b, axis=1, keepdims=True) + 1e-8)
    cos = (a * b).sum(axis=1)
    return {
        "p10": float(np.percentile(cos, 10)),
        "p50": float(np.percentile(cos, 50)),
        "p90": float(np.percentile(cos, 90)),
        "mean": float(cos.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--head-ckpt", default=None)
    ap.add_argument("--base-lam", default="CD_LAM")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--n-pairs",
        type=int,
        default=800,
        help="number of real pairs to draw from val split",
    )
    ap.add_argument("--split", default="val")
    ap.add_argument("--decode-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"

    df = pd.read_parquet(args.pair_index)
    sample = sample_balanced(
        df, args.n_pairs // 28 + 1, split=args.split, seed=args.seed
    )
    sample = sample.head(args.n_pairs).reset_index(drop=True)
    print(f"[guardrail] sampled {len(sample)} pairs from split={args.split}")

    pairs, valid = decode_pairs_parallel(
        sample, target_hw=(240, 320), workers=args.decode_workers
    )
    sample = sample.loc[valid].reset_index(drop=True)
    pairs = pairs[valid]
    print(f"[guardrail] {len(sample)} valid pairs after decode")

    # frame0 sample to use for identity / fake-camera tests
    frames_a = pairs[:, 0]  # (N, H, W, 3) — first frame of each pair
    frames_b = pairs[:, 1]  # second frame of each pair

    print(f"[guardrail] building LAM {args.base_lam} ...")
    lam = build_lam(args.base_lam, device=device)
    head = None
    if args.head_ckpt:
        ck = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        mode = cfg.get("readout_mode", "R2")
        sd = ck["head"]
        is_residual = any(k.startswith("delta.") for k in sd.keys())
        if is_residual:
            alpha_init = float(cfg.get("alpha_init", -3.0))
            head = ResidualAdapter(mode=mode, alpha_init=alpha_init).to(device)
            head.load_state_dict(sd)
            head.eval()
            print(
                f"[guardrail] loaded T1 residual (mode={mode}, alpha_sig={head.alpha_sigmoid.item():.4f}) from {args.head_ckpt} (step {ck.get('step', '?')})"
            )
        else:
            head = LayerMixerReadout(
                n_layers=24, d_layer=1024, d_z=32, d_out=32, mode=mode
            ).to(device)
            head.load_state_dict(sd)
            head.eval()
            print(
                f"[guardrail] loaded T0 head mode={mode} from {args.head_ckpt} (step {ck.get('step', '?')})"
            )

    # 1. Real pair encoding (baseline reference for norm scale)
    z_mu_real, z_metric_real = encode_pairs(lam, head, pairs, device=device)
    z_mu_real_norms = np.linalg.norm(z_mu_real, axis=1)
    z_metric_real_norms = np.linalg.norm(z_metric_real, axis=1)
    print(f"[guardrail] real-pair z_mu norm p50: {np.median(z_mu_real_norms):.3f}")
    print(
        f"[guardrail] real-pair z_metric norm p50: {np.median(z_metric_real_norms):.3f}"
    )

    out: dict = {
        "n_pairs": len(sample),
        "real_z_mu_norm_p50": float(np.median(z_mu_real_norms)),
        "real_z_metric_norm_p50": float(np.median(z_metric_real_norms)),
        "tests": {},
    }

    # 2. Identity (o, o)
    id_pairs = np.stack([frames_a, frames_a], axis=1)  # (N, 2, H, W, 3)
    z_mu_id, z_metric_id = encode_pairs(lam, head, id_pairs, device=device)
    out["tests"]["identity"] = {
        "z_mu_ratio": norm_ratio_to_real(z_mu_id, z_mu_real_norms),
        "z_metric_ratio": norm_ratio_to_real(z_metric_id, z_metric_real_norms),
    }

    # 3. Fake camera (o, T(o))
    for kind, mag in CAMERA_TESTS:
        cam_a = apply_camera_torch(frames_a, kind, mag, device=device)
        fake_pairs = np.stack([frames_a, cam_a], axis=1)
        z_mu_fk, z_metric_fk = encode_pairs(lam, head, fake_pairs, device=device)
        out["tests"][f"fake_camera_{kind}_{mag}"] = {
            "z_mu_ratio": norm_ratio_to_real(z_mu_fk, z_mu_real_norms),
            "z_metric_ratio": norm_ratio_to_real(z_metric_fk, z_metric_real_norms),
        }

    # 4. Single-sided (o_i, T(o_j)) vs (o_i, o_j)  — drift / cos
    for kind, mag in CAMERA_TESTS[:2]:  # limit to shifts to keep budget manageable
        cam_b = apply_camera_torch(frames_b, kind, mag, device=device)
        ss_pairs = np.stack([frames_a, cam_b], axis=1)
        z_mu_ss, z_metric_ss = encode_pairs(lam, head, ss_pairs, device=device)
        # drift = ||z_ss - z_real|| / ||z_real||
        z_mu_drift = np.linalg.norm(z_mu_ss - z_mu_real, axis=1) / (
            z_mu_real_norms + 1e-8
        )
        z_metric_drift = np.linalg.norm(z_metric_ss - z_metric_real, axis=1) / (
            z_metric_real_norms + 1e-8
        )
        out["tests"][f"single_sided_{kind}_{mag}"] = {
            "z_mu_drift": {
                "p10": float(np.percentile(z_mu_drift, 10)),
                "p50": float(np.percentile(z_mu_drift, 50)),
                "p90": float(np.percentile(z_mu_drift, 90)),
            },
            "z_metric_drift": {
                "p10": float(np.percentile(z_metric_drift, 10)),
                "p50": float(np.percentile(z_metric_drift, 50)),
                "p90": float(np.percentile(z_metric_drift, 90)),
            },
            "z_mu_cos_to_real": cosine_to(z_mu_ss, z_mu_real),
            "z_metric_cos_to_real": cosine_to(z_metric_ss, z_metric_real),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    # compact summary
    print("\n=== guardrail summary ===")
    print(
        f"identity z_mu p50:     {out['tests']['identity']['z_mu_ratio']['p50']:.3f}  (lower = better, < 0.2 ideal)"
    )
    print(
        f"identity z_metric p50: {out['tests']['identity']['z_metric_ratio']['p50']:.3f}"
    )
    for kind, mag in CAMERA_TESTS:
        k = f"fake_camera_{kind}_{mag}"
        print(f"{k} z_mu p50:     {out['tests'][k]['z_mu_ratio']['p50']:.3f}")
        print(f"{k} z_metric p50: {out['tests'][k]['z_metric_ratio']['p50']:.3f}")
    for kind, mag in CAMERA_TESTS[:2]:
        k = f"single_sided_{kind}_{mag}"
        print(f"{k} z_mu drift p50:     {out['tests'][k]['z_mu_drift']['p50']:.3f}")
        print(f"{k} z_metric drift p50: {out['tests'][k]['z_metric_drift']['p50']:.3f}")
        print(
            f"{k} z_metric cos to real p50: {out['tests'][k]['z_metric_cos_to_real']['p50']:.3f}"
        )
    print(f"\n[guardrail] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
