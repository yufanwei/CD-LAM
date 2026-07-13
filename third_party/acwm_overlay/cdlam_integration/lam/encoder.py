"""LAM encoder wrapper for the baseline same-z finetune.

Loads `external.lam.model.LAM` from the supplied 400k checkpoint, exposes:
    forward(pair: (B, 2, H, W, 3) uint8) -> z: (B, 32) float

For E_new we apply selective freezing per `config.yaml.model.trainable`.
For E_old we freeze everything and run in eval/no_grad/bf16.
"""

from __future__ import annotations

import os

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn


REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))


def _patch_lam_attention():
    """Replace LAM's naive scaled_dot_product_attention with F.scaled_dot_product_attention.

    The bundled implementation in external/lam/modules/blocks.py builds the full (L, S) attn
    bias and softmax matrix in fp32, and stores both for autograd — at 240x320 with batch 64
    × 24 blocks × 3 streams that comes to >25 GB just for attn weights. F.SDPA uses the
    flash/memory-efficient backend on H100 and avoids materializing the L×S matrix.

    Patch is applied once globally; both E_old and E_new use it. The numerics match the
    naive impl up to bf16 rounding (which is fine for both inference and our 32-D distill).
    """
    import torch.nn.functional as F_nn

    sys.path.insert(0, str(REPO))
    from external.lam.modules.blocks import SelfAttention

    if getattr(SelfAttention, "_sdpa_patched", False):
        return

    def patched_attn(self, query, key, value, is_causal: bool = False):
        return F_nn.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )

    SelfAttention.scaled_dot_product_attention = patched_attn
    SelfAttention._sdpa_patched = True


def build_lam(ckpt_path: str, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(REPO))
    _patch_lam_attention()
    from external.lam.model import LAM

    model = LAM(
        image_channels=3,
        lam_model_dim=1024,
        lam_latent_dim=32,
        lam_patch_size=16,
        lam_enc_blocks=24,
        lam_dec_blocks=24,
        lam_num_heads=16,
        ckpt_path=ckpt_path,
    )
    return model.to(device)


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def configure_trainable(
    lam: nn.Module, trainable_cfg: dict
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Apply the `trainable` config to lam (the LightningModule wrapper).

    Returns (head_params, block_params) for parameter-group LR splitting.
    """
    # Freeze everything first.
    freeze_all(lam)

    head_params: List[nn.Parameter] = []
    block_params: List[nn.Parameter] = []

    inner = lam.lam  # LatentActionModel

    # Encoder.out (final linear of the encoder transformer)
    if trainable_cfg.get("encoder_out", False):
        for p in inner.encoder.out.parameters():
            p.requires_grad = True
            head_params.append(p)

    # fc: 1024 -> 64 (mu/var)
    if trainable_cfg.get("fc", False):
        for p in inner.fc.parameters():
            p.requires_grad = True
            head_params.append(p)

    # action_prompt
    if trainable_cfg.get("action_prompt", False):
        inner.action_prompt.requires_grad = True
        head_params.append(inner.action_prompt)

    # last N transformer blocks
    n_last = int(trainable_cfg.get("last_n_blocks", 0))
    if n_last > 0:
        n_total = len(inner.encoder.transformer_blocks)
        for blk in inner.encoder.transformer_blocks[n_total - n_last :]:
            for p in blk.parameters():
                p.requires_grad = True
                block_params.append(p)

    return head_params, block_params


class LAMEncoderForFinetune(nn.Module):
    """Wraps LAM(LightningModule).lam and exposes pair-encoding.

    Encodes (o_i, o_j) -> z_mu (B, 32). Inputs are float32 in [0, 1] (B, 2, 3, H, W).
    """

    def __init__(self, lam: nn.Module):
        super().__init__()
        self.lam = lam
        # Note: nn.Module sees lam.lam (LatentActionModel) when descending; both ok.

    def forward(self, pair_b3hw: torch.Tensor) -> torch.Tensor:
        """pair_b3hw: (B, 2, 3, H, W) float in [0, 1]. Returns z_mu (B, 32)."""
        # Module expects (B, T, H, W, C) per LatentActionModel.encode.
        x = pair_b3hw.permute(0, 1, 3, 4, 2).contiguous()  # (B, 2, H, W, 3)
        out = self.lam.lam.encode(x)
        return out["z_mu"]


class TripleEncoderWrapper(nn.Module):
    """Calls a single LAM encoder THREE times inside one DDP forward call.

    DDP attaches AllReduce hooks once per `forward(...)`; if the inner module is invoked
    multiple times *inside* one outer forward, autograd still sees one graph. This pattern
    avoids the "marked variable ready twice" / static_graph errors caused by 3 separate
    `ddp_model(...)` calls per training step.

    Returns a dict {z_real, z_fake, z_ss}, each (B, 32) float.
    """

    def __init__(self, encoder: LAMEncoderForFinetune):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, pair_real: torch.Tensor, pair_fake: torch.Tensor, pair_ss: torch.Tensor
    ):
        z_real = self.encoder(pair_real)
        z_fake = self.encoder(pair_fake)
        z_ss = self.encoder(pair_ss)
        return z_real, z_fake, z_ss


def pair_uint8_to_float(pair_uint8: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W, 3) uint8 -> (B, 2, 3, H, W) float in [0, 1]."""
    if pair_uint8.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {pair_uint8.dtype}")
    x = pair_uint8.float() / 255.0
    return x.permute(0, 1, 4, 2, 3).contiguous()
