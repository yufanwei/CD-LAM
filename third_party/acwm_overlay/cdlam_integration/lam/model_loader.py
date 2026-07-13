#!/usr/bin/env python3
"""Phase 7 — T0 readout-only training (NOTE §10.2).

Architecture:
    LAM (frozen)  ─┬─ block0..block23 hidden states
                   └─ final z_mu (32D)
                           |
                  LayerMixer + final_z gate
                           |
                       z_metric (32D)

Loss:
    L = L_gap   (action-vs-video margin; main)
        + 0.2 * L_rel  (weighted SupCon-style; optional)
        + 0.05 * L_keep (||z_metric - z_mu||^2  light norm anchor)

Sampler (Phase 6):
    Batch is built by sampling K primitives × M episodes × N pairs (per
    cdlam_integration/configs/samplers/action_metric_balanced.yaml).

Outputs:
    outputs/cdlam_train/T0_readout/{config.yaml, train_log.jsonl,
                                     checkpoints/, eval_summary.md}

Usage:
    python cdlam_integration/tools/train_lam_action_readout.py \\
        --pair-index outputs/cdlam_data/pair_index_scored.parquet \\
        --sampler-config cdlam_integration/configs/samplers/action_metric_balanced.yaml \\
        --base-lam CD_LAM \\
        --out outputs/cdlam_train/T0_readout \\
        --steps 2000
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
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)


# ----------- pair decoder (matches Phase 5 probe) ---------------------------


def decode_pair(
    video_path: str, fi: int, fj: int, target_hw=(240, 320)
) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    out = [None, None]
    needed = {fi: 0}
    if fi != fj:
        needed[fj] = 1
    max_idx = max(fi, fj)
    i = 0
    try:
        while i <= max_idx:
            ok, fr = cap.read()
            if not ok:
                break
            if i in needed:
                fr = cv2.resize(
                    fr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA
                )
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                slot = needed[i]
                out[slot] = fr
                if fi == fj:
                    out[1] = fr
            i += 1
    finally:
        cap.release()
    if out[0] is None or out[1] is None:
        return None
    return np.stack(out, 0).astype(np.uint8)


# ----------- LAM with layer hooks (mirrors layers_probe) --------------------

LAM_NAME_ALIASES = {
    "DreamDojo_LAM": "CD_LAM_BASE",
    "CD_LAM": "CD_LAM",
}
LAM_NAME_CHOICES = tuple(sorted({"CD_LAM_BASE", "CD_LAM", *LAM_NAME_ALIASES}))


def _patch_sdpa():
    from external.lam.modules.blocks import SelfAttention

    if getattr(SelfAttention, "_sdpa_patched", False):
        return

    def patched(self, query, key, value, is_causal=False):
        return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

    SelfAttention.scaled_dot_product_attention = patched
    SelfAttention._sdpa_patched = True


def build_lam(name: str, device: str = "cuda") -> nn.Module:
    name = LAM_NAME_ALIASES.get(name, name)
    _patch_sdpa()
    from external.lam.model import LAM

    base_checkpoint = Path(
        os.environ.get(
            "CDLAM_LAM_BASE_CKPT",
            os.environ.get(
                "LAM_400K_LOCAL",
                str(REPO / "lammodel/checkpoints/CD-LAM/LAM_400k.ckpt"),
            ),
        )
    )
    if name == "CD_LAM_BASE":
        ckpt = base_checkpoint
        m = LAM(
            image_channels=3,
            lam_model_dim=1024,
            lam_latent_dim=32,
            lam_patch_size=16,
            lam_enc_blocks=24,
            lam_dec_blocks=24,
            lam_num_heads=16,
            ckpt_path=str(ckpt),
        )
    elif name == "CD_LAM":
        checkpoint_value = os.environ.get("CDLAM_STAGE1_LAM")
        if not checkpoint_value:
            raise FileNotFoundError("CDLAM_STAGE1_LAM is required for CD_LAM")
        ckpt = Path(checkpoint_value).expanduser().resolve()
        m = LAM(
            image_channels=3,
            lam_model_dim=1024,
            lam_latent_dim=32,
            lam_patch_size=16,
            lam_enc_blocks=24,
            lam_dec_blocks=24,
            lam_num_heads=16,
            ckpt_path=str(base_checkpoint),
        )
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        cleaned = {
            (k[8:] if k.startswith("encoder.") else k): v for k, v in cleaned.items()
        }
        missing, unexpected = m.load_state_dict(cleaned, strict=False)
        if len(missing) > 100:
            cleaned2 = {
                (k[4:] if k.startswith("lam.") else k): v for k, v in cleaned.items()
            }
            missing, unexpected = m.load_state_dict(cleaned2, strict=False)
        print(
            f"[lam] checkpoint loaded with {len(missing)} missing / "
            f"{len(unexpected)} unexpected",
            flush=True,
        )
    else:
        raise ValueError(f"unknown LAM ckpt name: {name}")
    return m.to(device).eval()


def encoder_forward_with_layers(lam_model, videos: torch.Tensor):
    """videos: (B, 2, H, W, 3) float in [0,1]. Returns layer_acts list (24 × (B, T, S, E)) and z_mu (B, 32)."""
    from external.lam.modules.blocks import patchify

    inner = lam_model.lam
    B, T = videos.shape[:2]
    assert T == 2
    patches = patchify(videos, inner.patch_size)
    action_pad = inner.action_prompt.expand(B, T, -1, -1)
    x = torch.cat([action_pad, patches], dim=2)
    x = inner.encoder.ffn(x)
    x = inner.encoder.pos_enc(x)
    layer_acts = []
    for blk in inner.encoder.transformer_blocks:
        x = blk(x, inner.encoder.causal_temporal)
        layer_acts.append(x)
    x_out = inner.encoder.out(x)
    z = x_out[:, 1:, 0]
    z = z.reshape(B * (T - 1), inner.model_dim)
    moments = inner.fc(z)
    z_mu, _ = torch.chunk(moments, 2, dim=1)
    return layer_acts, z_mu


def pool_action_token(h: torch.Tensor) -> torch.Tensor:
    return h[:, 1, 0, :]


# ----------- Readout head ---------------------------------------------------


class LayerMixerReadout(nn.Module):
    """Three modes (NOTE §10.2 R0/R1/R2 ablation):
    R0: only final z_mu through a Linear (no per-layer features).
    R1: only layer mixer (no z_mu shortcut).
    R2: layer mixer + final z_mu, gated by sigmoid(alpha).
    """

    def __init__(
        self,
        n_layers: int = 24,
        d_layer: int = 1024,
        d_z: int = 32,
        d_out: int = 32,
        mode: str = "R2",
    ):
        super().__init__()
        assert mode in ("R0", "R1", "R2")
        self.mode = mode
        if mode != "R0":
            self.layer_proj = nn.Linear(d_layer, d_out, bias=False)
            self.layer_w = nn.Parameter(torch.zeros(n_layers))
        if mode != "R1":
            self.z_mu_proj = nn.Linear(d_z, d_out)
        if mode == "R2":
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_layers: torch.Tensor, z_mu: torch.Tensor) -> torch.Tensor:
        if self.mode == "R0":
            return self.z_mu_proj(z_mu)
        w = F.softmax(self.layer_w, dim=0)
        mix = (h_layers * w[None, :, None]).sum(dim=1)
        zl = self.layer_proj(mix)
        if self.mode == "R1":
            return zl
        zf = self.z_mu_proj(z_mu)
        gate = torch.sigmoid(self.alpha)
        return zl + gate * zf


# ----------- Batch sampler --------------------------------------------------


class BalancedBatchSampler:
    """Yields a list of pair-row indices per batch:
    K primitives × M episodes × N pairs."""

    def __init__(self, df: pd.DataFrame, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.K = int(cfg["batch"]["K_primitives"])
        self.M = int(cfg["batch"]["M_episodes_per_primitive"])
        self.N = int(cfg["batch"]["N_pairs_per_episode"])
        self.B_d_min = int(cfg["batch"].get("B_d_min_datasets", 1))
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
        # Index: primitive -> {episode -> [row_idx, ...]}
        self.by_prim: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx, row in enumerate(self.df.itertuples(index=False)):
            self.by_prim[row.primitive_label][f"{row.dataset}|{row.episode_id}"].append(
                idx
            )
        self.primitives = [p for p, eps in self.by_prim.items() if len(eps) >= self.M]
        if len(self.primitives) < self.K:
            print(
                f"[sampler] WARNING only {len(self.primitives)} primitives have >= {self.M} episodes; "
                f"reducing K from {self.K} to {len(self.primitives)}",
                flush=True,
            )
            self.K = len(self.primitives)
        print(
            f"[sampler] {len(self.df)} eligible pairs, {len(self.primitives)} eligible primitives "
            f"(K={self.K}, M={self.M}, N={self.N})"
        )

    def __iter__(self):
        return self

    def __next__(self) -> list[int]:
        prims = self.rng.choice(self.primitives, size=self.K, replace=False)
        out_idx: list[int] = []
        for p in prims:
            eps = list(self.by_prim[p].keys())
            picked_eps = self.rng.choice(eps, size=min(self.M, len(eps)), replace=False)
            for ep in picked_eps:
                pool = self.by_prim[p][ep]
                pick = self.rng.choice(pool, size=min(self.N, len(pool)), replace=False)
                out_idx.extend(pick.tolist())
        return out_idx


# ----------- Loss helpers ---------------------------------------------------


def gap_loss(
    z: torch.Tensor, primitives: np.ndarray, episodes: np.ndarray, margin: float = 0.10
) -> torch.Tensor:
    """For each anchor, find a same-prim/diff-ep positive and a diff-prim/same-ep negative
    in the batch; if not present, fall back to diff-prim/diff-ep negative.
    L = max(0, margin - cos(a, p) + cos(a, n)) averaged."""
    z_n = F.normalize(z, dim=1)
    sim = z_n @ z_n.t()  # (B, B)
    B = z.shape[0]
    device = z.device
    losses = []
    for i in range(B):
        same_p = primitives == primitives[i]
        same_e = episodes == episodes[i]
        pos_mask = same_p & ~same_e
        if not pos_mask.any():
            continue
        # negative: prefer diff-prim, same-episode (hardest)
        neg_mask = (~same_p) & same_e
        if not neg_mask.any():
            neg_mask = (~same_p) & ~same_e
        if not neg_mask.any():
            continue
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]
        # use HARDEST positive (lowest sim) and HARDEST negative (highest sim)
        pos_sims = sim[i, pos_idx]
        neg_sims = sim[i, neg_idx]
        p = pos_sims.min()
        n = neg_sims.max()
        losses.append(F.relu(margin - p + n))
    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


def keep_anchor_loss(
    z_metric: torch.Tensor, z_mu: torch.Tensor, tau: float = 0.10
) -> torch.Tensor:
    """Hinge: penalize ||z_new - z_reference|| / ||z_reference|| above tau. Light anchor to baseline final_z."""
    diff = (z_metric - z_mu).norm(dim=1)
    base = z_mu.norm(dim=1) + 1e-6
    rel = diff / base
    return F.relu(rel - tau).pow(2).mean()


def norm_keep_loss(z_metric: torch.Tensor, z_mu: torch.Tensor) -> torch.Tensor:
    """Force z_metric norm to match z_mu norm; counters cone-collapse."""
    n_metric = z_metric.norm(dim=1)
    n_mu = z_mu.norm(dim=1) + 1e-6
    ratio = n_metric / n_mu
    return (ratio - 1.0).pow(2).mean()


def cone_spread_loss(z_metric: torch.Tensor) -> torch.Tensor:
    """Penalty if all z_metric vectors collapse to a narrow cone — pushes mean cosine of off-diag to 0.
    L = mean_off_diag(cos(z_i, z_j))^2 — small if directions are isotropic."""
    z_n = F.normalize(z_metric, dim=1)
    sim = z_n @ z_n.t()
    B = sim.shape[0]
    if B < 2:
        return torch.zeros((), device=z_metric.device, requires_grad=True)
    off_diag = sim - torch.eye(B, device=sim.device)
    mean_cos = off_diag.sum() / (B * (B - 1))
    return mean_cos.pow(2)


# ----------- Training -------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--sampler-config", required=True)
    ap.add_argument("--base-lam", default="CD_LAM", choices=LAM_NAME_CHOICES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--keep-tau", type=float, default=0.10)
    ap.add_argument("--w-gap", type=float, default=1.0)
    ap.add_argument("--w-keep", type=float, default=0.05)
    ap.add_argument(
        "--w-norm",
        type=float,
        default=0.0,
        help="weight on L_norm (keep ||z_metric|| close to ||z_mu||)",
    )
    ap.add_argument(
        "--w-spread",
        type=float,
        default=0.0,
        help="weight on cone-spread penalty (mean off-diag cosine -> 0)",
    )
    ap.add_argument(
        "--readout-mode",
        choices=["R0", "R1", "R2"],
        default="R2",
        help="head architecture: R0=final_z only, R1=layer_mixer only, R2=both",
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument(
        "--mode",
        choices=["full", "smoke"],
        default="full",
        help="smoke = 200 steps, smaller batch, for sanity",
    )
    args = ap.parse_args()
    device = "cuda"

    if args.mode == "smoke":
        args.steps = min(args.steps, 200)

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
    if args.mode == "smoke":
        sampler_cfg["batch"]["K_primitives"] = min(
            sampler_cfg["batch"]["K_primitives"], 6
        )
        sampler_cfg["batch"]["M_episodes_per_primitive"] = min(
            sampler_cfg["batch"]["M_episodes_per_primitive"], 2
        )

    print(f"[train] loading pair index {args.pair_index} ...")
    df = pd.read_parquet(args.pair_index)
    df = df[df["split"] == "train"]
    sampler = BalancedBatchSampler(df, sampler_cfg, seed=args.seed)

    print(f"[train] building LAM {args.base_lam} on {device} ...")
    lam = build_lam(args.base_lam, device=device)
    for p in lam.parameters():
        p.requires_grad = False

    head = LayerMixerReadout(
        n_layers=24, d_layer=1024, d_z=32, d_out=32, mode=args.readout_mode
    ).to(device)
    print(
        f"[train] head mode={args.readout_mode}, params: {sum(p.numel() for p in head.parameters())}"
    )

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("a")
    print(f"[train] starting {args.steps} steps")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        idxs = next(sampler)
        rows = sampler.df.iloc[idxs]

        # decode
        from concurrent.futures import ThreadPoolExecutor

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
            print(f"[train] step {step} too few decoded ({sum(ok_mask)}); skip")
            continue
        decoded = [d for d, m in zip(decoded, ok_mask) if m]
        rows = rows.iloc[ok_mask].reset_index(drop=True)
        videos = (
            torch.from_numpy(np.stack(decoded, 0)).float().to(device) / 255.0
        )  # (B, 2, H, W, 3)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                layer_acts, z_mu = encoder_forward_with_layers(lam, videos)
            h = torch.stack(
                [pool_action_token(h_l) for h_l in layer_acts], dim=1
            ).float()  # (B, 24, 1024)
            z_mu_f = z_mu.float()
        # head + losses run in fp32 with grad
        z_metric = head(h, z_mu_f)

        primitives = rows["primitive_label"].astype(str).values
        episodes = (
            rows["dataset"].astype(str) + "|" + rows["episode_id"].astype(str)
        ).values

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
        L = (
            args.w_gap * L_gap
            + args.w_keep * L_keep
            + args.w_norm * L_norm
            + args.w_spread * L_spread
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
                "alpha_sigmoid": float(torch.sigmoid(head.alpha).item())
                if hasattr(head, "alpha")
                else None,
                "layer_w_argmax": int(head.layer_w.argmax().item())
                if hasattr(head, "layer_w")
                else None,
                "layer_w_max": float(F.softmax(head.layer_w, dim=0).max().item())
                if hasattr(head, "layer_w")
                else None,
                "B": int(videos.shape[0]),
                "elapsed_s": elapsed,
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            extra = ""
            if entry["alpha_sigmoid"] is not None:
                extra = f"α_sig={entry['alpha_sigmoid']:.3f}  argmax_layer={entry['layer_w_argmax']}"
            elif entry["layer_w_argmax"] is not None:
                extra = f"argmax_layer={entry['layer_w_argmax']}"
            print(
                f"[train] step {step:4d}  L={L.item():.4f}  "
                f"gap={L_gap.item():.4f}  keep={L_keep.item():.4f}  "
                f"norm={L_norm.item():.4f}  spread={L_spread.item():.4f}  "
                f"{extra}  B={entry['B']}  ({elapsed:.0f}s)",
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

    # final save
    ck = {
        "step": args.steps,
        "head": head.state_dict(),
        "config": vars(args),
        "sampler_cfg": sampler_cfg,
    }
    torch.save(ck, out_dir / "checkpoints" / "final.pt")
    log_f.close()
    print(f"[train] done. ckpt -> {out_dir / 'checkpoints' / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
