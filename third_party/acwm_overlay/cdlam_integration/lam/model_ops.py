"""LAM core forward helpers — reuses external/lam build_lam (which loads baseline ckpt
including decoder weights), then provides:

  encode_full(lam_inner, videos, sample=True, use_ckpt=False)
    -> {z_mu, z_var, z_rep, patches}        # patches kept for decoder
       reparametrize when sample=True (training mode)
       use z_mu when sample=False (eval / metric)
       use_ckpt=True wraps each encoder block in torch.utils.checkpoint
         (saves ~50% encoder activation memory at ~30% extra compute).

  decode_full(lam_inner, patches, z_rep, H, W, use_ckpt=False)
    -> recon  (B, T-1, H, W, C) sigmoid output

  forward_full(lam_inner, videos, sample, use_ckpt=False)
    -> {z_mu, z_var, z_rep, recon}

  recon_kl_loss(outputs, gt_future, beta=0.01)
    -> (loss, mse, kl)   exact paper recon: MSE.mean() + 0.01 * KL[sum-dim,mean-batch]

This is the canonical L_gen path for F1. We use the same SpatioTemporalTransformer /
SpatioTransformer / patchify / unpatchify modules from external/lam/modules/blocks
(architecture identical to lam_project), so baseline ckpt loads without modification.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint as _ckpt
from torch import Tensor


def _enc_blocks_forward(encoder, x: Tensor, use_ckpt: bool) -> Tensor:
    """Run encoder.transformer_blocks; optionally with grad checkpointing.

    use_reentrant=True is required because the SDPA-patched SelfAttention does
    not save consistent intermediate tensors across forward + recompute under
    bf16 autocast (a recompute mismatch crashes use_reentrant=True).
    """
    ct = encoder.causal_temporal
    if use_ckpt and torch.is_grad_enabled():
        for block in encoder.transformer_blocks:
            x = _ckpt.checkpoint(block, x, ct, use_reentrant=True)
    else:
        for block in encoder.transformer_blocks:
            x = block(x, ct)
    return x


def _dec_blocks_forward(decoder, x: Tensor, use_ckpt: bool) -> Tensor:
    """Run decoder.transformer_blocks; optionally with grad checkpointing."""
    if use_ckpt and torch.is_grad_enabled():
        for block in decoder.transformer_blocks:
            x = _ckpt.checkpoint(block, x, use_reentrant=True)
    else:
        for block in decoder.transformer_blocks:
            x = block(x)
    return x


def encode_full(lam_inner, videos: Tensor, sample: bool, use_ckpt: bool = False):
    """Returns dict with patches (for decoder), z_mu, z_var, z_rep_flat.

    Mixed precision policy:
      - encoder transformer blocks run in caller's autocast (bf16 typical, big matmul).
      - z_mu / z_var / reparam / z_rep returned in fp32 — `exp(0.5 * z_var)` overflows
        bf16 (max ~6.5e4) when z_var > ~11, which can happen sporadically at B>=64
        and triggers backward NaN. Casting to fp32 here makes reparam numerically
        safe without losing encoder bf16 throughput.
    """
    from external.lam.modules.blocks import patchify
    B, T = videos.shape[:2]
    assert T == 2, f"LAM expects T=2, got T={T}"
    patches = patchify(videos, lam_inner.patch_size)
    action_pad = lam_inner.action_prompt.expand(B, T, -1, -1)
    padded = torch.cat([action_pad, patches], dim=2)
    enc = lam_inner.encoder
    x = enc.ffn(padded)
    x = enc.pos_enc(x)
    x = _enc_blocks_forward(enc, x, use_ckpt)
    h = enc.out(x)
    z = h[:, 1:, 0]
    z = z.reshape(B * (T - 1), lam_inner.model_dim)
    moments = lam_inner.fc(z)
    z_mu, z_var = torch.chunk(moments, 2, dim=1)
    # ---- numerically sensitive ops: cast to fp32 for KL / reparam ----
    z_mu_f = z_mu.float()
    z_var_f = z_var.float()
    if sample:
        # clamp z_var to keep exp stable even in fp32; matches typical VAE practice
        z_var_clamped = torch.clamp(z_var_f, min=-10.0, max=10.0)
        z_rep_flat = z_mu_f + torch.randn_like(z_var_f) * torch.exp(0.5 * z_var_clamped)
    else:
        z_rep_flat = z_mu_f
    z_rep = z_rep_flat.reshape(B, T - 1, 1, lam_inner.latent_dim)
    return {"patches": patches, "z_mu": z_mu_f, "z_var": z_var_f, "z_rep": z_rep,
            "z_rep_flat": z_rep_flat}


def decode_full(lam_inner, patches: Tensor, z_rep: Tensor, H: int, W: int,
                  use_ckpt: bool = False) -> Tensor:
    """patches: (B, T, N, patch_token_dim); z_rep: (B, T-1, 1, latent_dim)."""
    from external.lam.modules.blocks import unpatchify
    video_patches = lam_inner.patch_up(patches[:, :-1])
    action_patches = lam_inner.action_up(z_rep)
    video_action_patches = video_patches + action_patches
    dec = lam_inner.decoder
    x = dec.ffn(video_action_patches)
    x = dec.pos_enc(x)
    x = _dec_blocks_forward(dec, x, use_ckpt)
    x = dec.out(x)
    video_recon = torch.sigmoid(x)
    return unpatchify(video_recon, lam_inner.patch_size, H, W)


def forward_full(lam_inner, videos: Tensor, sample: bool, use_ckpt: bool = False):
    H, W = videos.shape[2:4]
    out = encode_full(lam_inner, videos, sample=sample, use_ckpt=use_ckpt)
    recon = decode_full(lam_inner, out["patches"], out["z_rep"], H, W, use_ckpt=use_ckpt)
    return {"z_mu": out["z_mu"], "z_var": out["z_var"], "z_rep": out["z_rep"],
            "z_rep_flat": out["z_rep_flat"], "recon": recon,
            # forward_full also returns patches so callers can re-decode with a
            # modified z (e.g. zero-z / shuffled-z for L_use).
            "patches": out["patches"]}


def recon_kl_loss(outputs: dict, gt_future: Tensor, beta: float = 0.01):
    """Exact original LAM loss (external/lam_project/lam/model.py:59-61).
    KL inputs are already fp32 (encode_full ensures this); recon may be bf16 but
    fp32-vs-bf16 subtraction promotes to fp32 in PyTorch.
    """
    mse = ((gt_future.float() - outputs["recon"].float()) ** 2).mean()
    z_mu, z_var = outputs["z_mu"].float(), outputs["z_var"].float()
    z_var_clamped = torch.clamp(z_var, min=-10.0, max=10.0)
    kl = -0.5 * torch.sum(1 + z_var - z_mu ** 2 - z_var_clamped.exp(), dim=1).mean()
    return mse + beta * kl, mse, kl
