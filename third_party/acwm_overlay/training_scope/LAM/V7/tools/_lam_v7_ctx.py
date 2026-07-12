"""Context plumbing for the joint LAM g/h trainer (New-LAM §4.2).

g and h must be FiLM-conditioned on ctx = (pooled o_t visual token from E's trunk,
read-only) + proprioception q_t. This module exposes the trunk token (which
encode_full discards) and assembles ctx, WITHOUT touching the canonical forward.

Key structural guarantee (verified `_lam_v2_forward.py:80-81`):
  encoder produces h = enc.out(x) of shape (B, T, 1+N, 1024); z is taken from
  h[:, 1:, 0] (FUTURE frame, action slot 0). The o_t visual token we use is
  trunk_ot = h[:, 0, 1:, :].mean(1)  -> frame 0 (o_t), patch slots 1: (EXCLUDING
  action slot 0). It therefore CANNOT contain o_{t+1} information. ctx is detached
  before g/h so no gradient flows into E through the conditioning path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from external.lam.modules.blocks import patchify
from LAM_V2.tools._lam_v2_forward import _enc_blocks_forward

STATE_DIM = 20
RUNTIME_ROOT = Path(
    os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4]))
)
STATS_JSON = Path(
    os.environ.get(
        "CDLAM_AGIBOT_LEROBOT_STATS",
        str(RUNTIME_ROOT / "datasets/agibot/meta/stats.json"),
    )
)


def encode_full_with_trunk(
    lam_inner, videos: torch.Tensor, sample: bool, use_ckpt: bool = False
):
    """encode_full + an extra 'trunk_ot' key (B, model_dim) = pooled o_t patch token.

    Mirrors `_lam_v2_forward.encode_full` exactly up to h=enc.out(x), then pools the
    o_t (frame 0) patch slots. Returns the same dict plus 'trunk_ot'.
    """
    B, T = videos.shape[:2]
    assert T == 2, f"LAM expects T=2, got T={T}"
    patches = patchify(videos, lam_inner.patch_size)
    action_pad = lam_inner.action_prompt.expand(B, T, -1, -1)
    padded = torch.cat([action_pad, patches], dim=2)
    enc = lam_inner.encoder
    x = enc.ffn(padded)
    x = enc.pos_enc(x)
    x = _enc_blocks_forward(enc, x, use_ckpt)
    h = enc.out(x)  # (B, T, 1+N, model_dim)
    trunk_ot = h[:, 0, 1:, :].mean(dim=1)  # (B, model_dim) pooled o_t, no action slot
    trunk_ot1 = h[:, 1, 1:, :].mean(
        dim=1
    )  # (B, model_dim) pooled o_{t+1} (IDM upper-bound use)
    z = h[:, 1:, 0].reshape(B * (T - 1), lam_inner.model_dim)
    moments = lam_inner.fc(z)
    z_mu, z_var = torch.chunk(moments, 2, dim=1)
    z_mu_f, z_var_f = z_mu.float(), z_var.float()
    if sample:
        z_var_clamped = torch.clamp(z_var_f, min=-10.0, max=10.0)
        z_rep_flat = z_mu_f + torch.randn_like(z_var_f) * torch.exp(0.5 * z_var_clamped)
    else:
        z_rep_flat = z_mu_f
    z_rep = z_rep_flat.reshape(B, T - 1, 1, lam_inner.latent_dim)
    return {
        "patches": patches,
        "z_mu": z_mu_f,
        "z_var": z_var_f,
        "z_rep": z_rep,
        "z_rep_flat": z_rep_flat,
        "trunk_ot": trunk_ot,
        "trunk_ot1": trunk_ot1,
    }


def load_qt_stats(device="cpu", dead_std_thresh=0.04):
    """Return (mean(20), std(20)) for observation.state from LeRobot stats.json.
    Dead dims (std < thresh) get std->1 so they normalize to ~0 (contribute nothing)."""
    st = json.load(open(STATS_JSON))
    mean = np.asarray(st["observation.state"]["mean"], dtype=np.float32)
    std = np.asarray(st["observation.state"]["std"], dtype=np.float32)
    assert mean.shape == (STATE_DIM,) and std.shape == (STATE_DIM,)
    std = std.copy()
    std[std < dead_std_thresh] = 1.0
    return torch.tensor(mean, device=device), torch.tensor(std, device=device)


def normalize_qt(q, q_mean, q_std):
    return (q - q_mean) / q_std


def build_ctx(trunk_ot, q_t, q_mean, q_std):
    """ctx = [trunk_ot.detach() (1024), q_t_norm (20)] -> (B, 1044). Read-only."""
    q_norm = normalize_qt(q_t, q_mean, q_std)
    return torch.cat([trunk_ot.detach(), q_norm], dim=-1)


CTX_DIM = 1024 + STATE_DIM  # 1044


@torch.no_grad()
def recompute_s_vec(lam_inner, frames_t, dev, n=1024, bs=64, eps=1e-6):
    """s_i = sqrt(E[z_mu_i^2]) over a frame sample. frames_t: (N,2,H,W,3) uint8/float CPU."""
    N = len(frames_t)
    idx = np.random.choice(N, size=min(n, N), replace=False)
    zs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s0 in range(0, len(idx), bs):
            v = frames_t[idx[s0 : s0 + bs]].to(dev).float()
            if v.max() > 1.5:
                v = v / 255.0
            zs.append(
                encode_full_with_trunk(lam_inner, v, sample=False)["z_mu"].float().cpu()
            )
    z_all = torch.cat(zs)
    return (z_all.pow(2).mean(0) + eps).sqrt().to(dev)
