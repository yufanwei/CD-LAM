#!/usr/bin/env python3
"""Phase T1 — residual adapter training (NOTE §10.3).

  z_new = z_mu + sigmoid(alpha) * delta_z
  delta_z = LayerMixerReadout(layer_acts, z_mu)

When sigmoid(alpha)=0 (init), z_new == z_mu, so v1 LAM's identity / fake-camera
guardrail is preserved by construction. As training progresses, alpha is allowed
to grow but is bounded; explicit L_id / L_cam_fake0 / L_cam_ss losses are added
to keep the residual benign on guardrail inputs.

The sampler now mines AgiBot same-episode different-primitive hard negatives
(NOTE §9.4) — AgiBot episodes have a median of 2 distinct primitive segments.

Usage (4 GPUs in parallel for variants):
    CUDA_VISIBLE_DEVICES=N python LAM_V2/tools/train_lam_residual_adapter.py \\
        --pair-index outputs/lam_v2_data/pair_index_scored.parquet \\
        --sampler-config LAM_V2/configs/samplers/action_metric_balanced.yaml \\
        --base-lam CD_LAM_V1 \\
        --init-from outputs/lam_v2_train/T0_R2_anti_collapse/checkpoints/final.pt \\
        --out outputs/lam_v2_train/T1_residual_<variant> \\
        --readout-mode R2 \\
        --steps 2000 --w-id 0.10 --w-cam 0.05 \\
        --hard-neg-per-batch 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from concurrent.futures import ThreadPoolExecutor

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetune_4-30/scripts"))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from LAM_V2.tools.train_lam_action_readout import (  # noqa: E402
    LAM_NAME_CHOICES,
    decode_pair,
    build_lam,
    encoder_forward_with_layers,
    pool_action_token,
    LayerMixerReadout,
    gap_loss,
    keep_anchor_loss,
    norm_keep_loss,
    cone_spread_loss,
)


# ---------- camera transform (re-uses finetune_4-30/scripts/transforms.py) ---


def apply_camera_torch(
    frames_uint8: np.ndarray, kind: str, mag: float, device: str = "cuda"
) -> np.ndarray:
    from transforms import shift_theta, zoom_theta, rotation_theta, _apply

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


# ---------- ResidualAdapter ------------------------------------------------


class ResidualAdapter(torch.nn.Module):
    """z_new = z_mu + sigmoid(alpha) * delta_z, where delta_z = LayerMixer head.

    alpha starts at -3 (sigmoid -> 0.047) so init is near identity to z_mu.
    """

    def __init__(self, mode: str = "R2", alpha_init: float = -3.0):
        super().__init__()
        self.delta = LayerMixerReadout(
            n_layers=24, d_layer=1024, d_z=32, d_out=32, mode=mode
        )
        self.alpha_residual = torch.nn.Parameter(torch.tensor(alpha_init))

    @property
    def alpha_sigmoid(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_residual)

    def forward(self, h_layers: torch.Tensor, z_mu: torch.Tensor) -> torch.Tensor:
        delta_z = self.delta(h_layers, z_mu)
        return z_mu + self.alpha_sigmoid * delta_z


# ---------- Hard-negative aware sampler ------------------------------------


class HardNegBalancedSampler:
    """Yields a list of pair-row indices. Per batch:
    - K_primitives × M_episodes × N_pairs anchors (as before)
    - + hard_neg_per_batch extra rows: same episode but DIFFERENT primitive
    """

    def __init__(
        self, df: pd.DataFrame, cfg: dict, hard_neg_per_batch: int = 16, seed: int = 0
    ):
        self.cfg = cfg
        self.K = int(cfg["batch"]["K_primitives"])
        self.M = int(cfg["batch"]["M_episodes_per_primitive"])
        self.N = int(cfg["batch"]["N_pairs_per_episode"])
        self.hard_neg = int(hard_neg_per_batch)

        elig = cfg["eligibility"]
        mask = (
            (df["primitive_label"] != "")
            & (df["label_confidence"] >= elig["min_label_confidence"])
            & (df["phase_confidence"] >= elig["min_phase_confidence"])
            & (df["motion_score"] >= elig["min_motion_score"])
        )
        if elig.get("exclude_camera_dominant", True):
            mask &= ~df["is_camera_dominant"].fillna(False).astype(bool)
        self.df = df.loc[mask].reset_index(drop=True)
        self.rng = np.random.default_rng(seed)

        # primitive -> {episode -> [row_idx, ...]}
        self.by_prim: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # episode -> {primitive -> [row_idx, ...]}  (for hard-negative lookup)
        self.by_ep: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx, r in enumerate(self.df.itertuples(index=False)):
            ep_key = f"{r.dataset}|{r.episode_id}"
            self.by_prim[r.primitive_label][ep_key].append(idx)
            self.by_ep[ep_key][r.primitive_label].append(idx)

        self.primitives = [p for p, eps in self.by_prim.items() if len(eps) >= self.M]
        self.multi_prim_episodes = [
            ep for ep, prims in self.by_ep.items() if len(prims) >= 2
        ]
        if len(self.primitives) < self.K:
            self.K = len(self.primitives)
        print(
            f"[sampler] {len(self.df)} eligible pairs, {len(self.primitives)} eligible primitives, "
            f"{len(self.multi_prim_episodes)} multi-primitive episodes; "
            f"K={self.K} M={self.M} N={self.N} hard_neg={self.hard_neg}"
        )

    def __iter__(self):
        return self

    def __next__(self) -> list[int]:
        prims = self.rng.choice(self.primitives, size=self.K, replace=False)
        out_idx: list[int] = []
        used_eps: set[str] = set()
        for p in prims:
            eps = list(self.by_prim[p].keys())
            picked_eps = self.rng.choice(eps, size=min(self.M, len(eps)), replace=False)
            for ep in picked_eps:
                used_eps.add(ep)
                pool = self.by_prim[p][ep]
                pick = self.rng.choice(pool, size=min(self.N, len(pool)), replace=False)
                out_idx.extend(pick.tolist())

        # Hard-neg mining: pick multi-prim episodes (already in batch) and grab a pair from
        # a primitive different from any one already represented in batch for that ep.
        if self.hard_neg > 0 and self.multi_prim_episodes:
            cand_eps = [ep for ep in used_eps if len(self.by_ep[ep]) >= 2]
            if cand_eps:
                tries = 0
                added = 0
                while added < self.hard_neg and tries < 4 * self.hard_neg:
                    tries += 1
                    ep = cand_eps[self.rng.integers(0, len(cand_eps))]
                    prims_in_ep = list(self.by_ep[ep].keys())
                    pp = prims_in_ep[self.rng.integers(0, len(prims_in_ep))]
                    pool = self.by_ep[ep][pp]
                    if not pool:
                        continue
                    out_idx.append(int(self.rng.choice(pool)))
                    added += 1
        return out_idx


# ---------- Loss helpers (extra) -------------------------------------------


def identity_loss(z_id: torch.Tensor, real_norm_p50: float) -> torch.Tensor:
    """penalty = (||z_id|| / real_norm_p50)^2 — wants identity-pair z near 0."""
    n = z_id.norm(dim=1)
    return (n / max(real_norm_p50, 1e-3)).pow(2).mean()


def cam_fake_loss(z_fake: torch.Tensor, real_norm_p50: float) -> torch.Tensor:
    """penalty for ||z(o, T(o))|| — wants pure-camera z near 0."""
    n = z_fake.norm(dim=1)
    return (n / max(real_norm_p50, 1e-3)).pow(2).mean()


def cam_ss_loss(
    z_ss: torch.Tensor, z_real_stopgrad: torch.Tensor, real_norm_p50: float
) -> torch.Tensor:
    """penalty for ||z(o, T(o')) - stopgrad(z(o, o'))|| — camera should not add new action."""
    diff = (z_ss - z_real_stopgrad.detach()).norm(dim=1)
    return (diff / max(real_norm_p50, 1e-3)).pow(2).mean()


# ---------- Training -------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--sampler-config", required=True)
    ap.add_argument("--base-lam", default="CD_LAM_V1", choices=LAM_NAME_CHOICES)
    ap.add_argument(
        "--init-from",
        default=None,
        help="optional: T0 head ckpt to seed the delta_z module",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout-mode", choices=["R0", "R1", "R2"], default="R2")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument(
        "--lr-alpha",
        type=float,
        default=2e-3,
        help="separate LR for alpha (residual gate)",
    )
    ap.add_argument("--alpha-init", type=float, default=-3.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--keep-tau", type=float, default=0.08)
    ap.add_argument("--w-gap", type=float, default=1.0)
    ap.add_argument("--w-keep", type=float, default=0.50)
    ap.add_argument("--w-norm", type=float, default=0.10)
    ap.add_argument("--w-spread", type=float, default=0.05)
    ap.add_argument("--w-id", type=float, default=0.10)
    ap.add_argument("--w-cam-fake0", type=float, default=0.05)
    ap.add_argument("--w-cam-ss", type=float, default=0.05)
    ap.add_argument("--hard-neg-per-batch", type=int, default=16)
    ap.add_argument(
        "--cam-aug-frac",
        type=float,
        default=0.25,
        help="fraction of batch reused as camera-aug guardrail samples each step",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-workers", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(vars(args), f)

    with open(args.sampler_config) as f:
        sampler_cfg = yaml.safe_load(f)

    print(f"[T1] loading pair index {args.pair_index} ...")
    df = pd.read_parquet(args.pair_index)
    df = df[df["split"] == "train"]
    sampler = HardNegBalancedSampler(
        df, sampler_cfg, hard_neg_per_batch=args.hard_neg_per_batch, seed=args.seed
    )

    device = args.device
    lam = build_lam(args.base_lam, device=device)
    for p in lam.parameters():
        p.requires_grad = False

    head = ResidualAdapter(mode=args.readout_mode, alpha_init=args.alpha_init).to(
        device
    )
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck["head"]
        # map keys: T0 head was a bare LayerMixerReadout; T1 wraps it as `delta.*`.
        delta_sd = {}
        for k, v in sd.items():
            delta_sd[f"delta.{k}"] = v
        # only load delta_*; alpha_residual stays at our chosen init
        missing, unexpected = head.load_state_dict(delta_sd, strict=False)
        print(
            f"[T1] init delta from {args.init_from}: missing={len(missing)} (expect ['alpha_residual']), unexpected={len(unexpected)}"
        )
    print(
        f"[T1] head params: {sum(p.numel() for p in head.parameters())}, "
        f"alpha_init_sigmoid={head.alpha_sigmoid.item():.4f}"
    )

    optim = torch.optim.AdamW(
        [
            {"params": head.delta.parameters(), "lr": args.lr},
            {"params": [head.alpha_residual], "lr": args.lr_alpha},
        ],
        weight_decay=1e-4,
    )

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("a")
    print(f"[T1] starting {args.steps} steps")
    t0 = time.time()

    CAMERA_KINDS = [("shift_x", 0.10), ("shift_y", 0.05)]

    for step in range(1, args.steps + 1):
        idxs = next(sampler)
        rows = sampler.df.iloc[idxs]

        # decode all pairs
        decoded = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=args.decode_workers) as ex:
            futs = {
                ex.submit(decode_pair, r.video_path, int(r.frame_i), int(r.frame_j)): k
                for k, r in enumerate(rows.itertuples(index=False))
            }
            for fut in futs:
                k = futs[fut]
                try:
                    decoded[k] = fut.result()
                except Exception:
                    decoded[k] = None
        ok_mask = [d is not None for d in decoded]
        if sum(ok_mask) < 4:
            continue
        decoded_arr = np.stack([d for d, m in zip(decoded, ok_mask) if m], 0)
        rows = rows.iloc[ok_mask].reset_index(drop=True)
        videos = torch.from_numpy(decoded_arr).float().to(device) / 255.0

        # forward LAM (frozen)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                layer_acts, z_mu = encoder_forward_with_layers(lam, videos)
            h = torch.stack(
                [pool_action_token(h_l) for h_l in layer_acts], dim=1
            ).float()
            z_mu_f = z_mu.float()
        z_metric = head(h, z_mu_f)

        primitives = rows["primitive_label"].astype(str).values
        episodes = (
            rows["dataset"].astype(str) + "|" + rows["episode_id"].astype(str)
        ).values

        # core relation losses
        L_gap = gap_loss(z_metric, primitives, episodes, margin=args.margin)
        L_keep = keep_anchor_loss(z_metric, z_mu_f, tau=args.keep_tau)
        L_norm = (
            norm_keep_loss(z_metric, z_mu_f)
            if args.w_norm > 0
            else torch.zeros((), device=z_metric.device)
        )
        L_spread = (
            cone_spread_loss(z_metric)
            if args.w_spread > 0
            else torch.zeros((), device=z_metric.device)
        )

        # guardrail losses on a subset of frames from the batch
        L_id = torch.zeros((), device=device)
        L_cam0 = torch.zeros((), device=device)
        L_camss = torch.zeros((), device=device)
        if args.w_id > 0 or args.w_cam_fake0 > 0 or args.w_cam_ss > 0:
            n_aug = max(2, int(args.cam_aug_frac * len(decoded_arr)))
            sel = np.random.choice(len(decoded_arr), size=n_aug, replace=False)
            frames_a = decoded_arr[sel, 0]  # (n_aug, H, W, 3)
            frames_b = decoded_arr[sel, 1]
            real_norm_p50 = float(z_metric.detach().norm(dim=1).median().item())

            # identity (o, o)
            if args.w_id > 0:
                id_pairs = np.stack([frames_a, frames_a], 1)
                v_id = torch.from_numpy(id_pairs).float().to(device) / 255.0
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        la_id, zm_id = encoder_forward_with_layers(lam, v_id)
                    h_id = torch.stack(
                        [pool_action_token(x) for x in la_id], dim=1
                    ).float()
                    zm_id_f = zm_id.float()
                z_id = head(h_id, zm_id_f)
                L_id = identity_loss(z_id, real_norm_p50)

            # fake camera (o, T(o))
            if args.w_cam_fake0 > 0 or args.w_cam_ss > 0:
                kind, mag = CAMERA_KINDS[step % len(CAMERA_KINDS)]
                cam_a = apply_camera_torch(frames_a, kind, mag, device=device)
                if args.w_cam_fake0 > 0:
                    fake0 = np.stack([frames_a, cam_a], 1)
                    v_f0 = torch.from_numpy(fake0).float().to(device) / 255.0
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            la_f0, zm_f0 = encoder_forward_with_layers(lam, v_f0)
                        h_f0 = torch.stack(
                            [pool_action_token(x) for x in la_f0], dim=1
                        ).float()
                        zm_f0_f = zm_f0.float()
                    z_f0 = head(h_f0, zm_f0_f)
                    L_cam0 = cam_fake_loss(z_f0, real_norm_p50)

                if args.w_cam_ss > 0:
                    cam_b = apply_camera_torch(frames_b, kind, mag, device=device)
                    ss_pairs = np.stack([frames_a, cam_b], 1)
                    real_pairs = np.stack([frames_a, frames_b], 1)
                    v_ss = torch.from_numpy(ss_pairs).float().to(device) / 255.0
                    v_re = torch.from_numpy(real_pairs).float().to(device) / 255.0
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            la_ss, zm_ss = encoder_forward_with_layers(lam, v_ss)
                            la_re, zm_re = encoder_forward_with_layers(lam, v_re)
                        h_ss = torch.stack(
                            [pool_action_token(x) for x in la_ss], dim=1
                        ).float()
                        h_re = torch.stack(
                            [pool_action_token(x) for x in la_re], dim=1
                        ).float()
                    z_ss = head(h_ss, zm_ss.float())
                    z_re = head(h_re, zm_re.float())
                    L_camss = cam_ss_loss(z_ss, z_re.detach(), real_norm_p50)

        L = (
            args.w_gap * L_gap
            + args.w_keep * L_keep
            + args.w_norm * L_norm
            + args.w_spread * L_spread
            + args.w_id * L_id
            + args.w_cam_fake0 * L_cam0
            + args.w_cam_ss * L_camss
        )

        optim.zero_grad()
        L.backward()
        optim.step()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            entry = {
                "step": step,
                "loss": float(L.item()),
                "L_gap": float(L_gap.item()),
                "L_keep": float(L_keep.item()),
                "L_norm": float(L_norm.item()),
                "L_spread": float(L_spread.item()),
                "L_id": float(L_id.item()),
                "L_cam0": float(L_cam0.item()),
                "L_camss": float(L_camss.item()),
                "alpha_sigmoid": float(head.alpha_sigmoid.item()),
                "B": int(videos.shape[0]),
                "elapsed_s": elapsed,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[T1] step {step:4d}  L={L.item():.4f}  "
                f"gap={L_gap.item():.4f}  keep={L_keep.item():.3f}  "
                f"id={L_id.item():.3f}  cam0={L_cam0.item():.3f}  ss={L_camss.item():.3f}  "
                f"α_sig={entry['alpha_sigmoid']:.4f}  B={entry['B']}  ({elapsed:.0f}s)",
                flush=True,
            )
        if step % args.save_every == 0:
            ck = {
                "step": step,
                "head": head.state_dict(),
                "config": vars(args),
                "sampler_cfg": sampler_cfg,
            }
            torch.save(ck, out_dir / "checkpoints" / f"step_{step:06d}.pt")

    ck = {
        "step": args.steps,
        "head": head.state_dict(),
        "config": vars(args),
        "sampler_cfg": sampler_cfg,
    }
    torch.save(ck, out_dir / "checkpoints" / "final.pt")
    log_f.close()
    print(f"[T1] done. ckpt -> {out_dir / 'checkpoints' / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
