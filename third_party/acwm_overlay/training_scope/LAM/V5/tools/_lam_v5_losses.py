"""LAM v5 loss extensions — fg-weighted reconstruction + bg consistency.

V4 had 4 reconstruction-related losses (L_rec_full, α·L_rec_inter, L_use_full,
η·L_use_inter) — all in service of "make decoder use z and reconstruct
interaction region". V5 reduces this to 2 losses using preprocessed RoboSAM
fg/bg masks:

  L_rec_fg = (1/Σfg) · Σ fg(x) · ||pred(x) - o_{t+1}(x)||²
  L_rec_bg = (1/Σbg) · Σ bg(x) · ||pred(x) - o_t(x)||²    (bg consistency)

L_rec_fg is the core: only the action-region MSE matters. Decoder cannot copy
o_t in fg (objects + arm move there), so it must use z. This naturally enforces
"z carries action info" — no need for L_use double-insurance.

L_rec_bg is the optional bg-consistency: predicted bg should match o_t (NOT
o_{t+1}), under the camera-fixed assumption. This frees decoder capacity for fg
reconstruction.
"""
from __future__ import annotations

from typing import Tuple

import torch


def fg_weighted_reconstruction(
    pred: torch.Tensor,                      # (B, T-1, H, W, C) decoder output
    gt_next: torch.Tensor,                   # (B, T-1, H, W, C) o_{t+1}
    fg_mask: torch.Tensor,                   # (B, H, W) float in {0,1}
    valid: torch.Tensor,                     # (B,) bool
    *,
    min_pixels: int = 64,
) -> Tuple[torch.Tensor, dict]:
    """L_rec_fg = mean over valid rows of (Σ fg · MSE_pix) / clamp(Σfg, min_pix).

    Per-row MSE averaged inside fg mask; rows without valid mask contribute 0
    (and are excluded from the row average).
    """
    pred_f = pred.float()
    gt_f = gt_next.float()
    sq = (pred_f - gt_f) ** 2                                 # (B, T-1, H, W, C)
    if sq.dim() == 5:
        sq = sq.mean(dim=1)                                   # avg time axis: (B, H, W, C)
    sq_pix = sq.mean(dim=-1)                                  # (B, H, W)

    M = fg_mask.float()                                        # (B, H, W)
    area = M.sum(dim=(-1, -2)).clamp_min(float(min_pixels))   # (B,)
    per_row = (sq_pix * M).sum(dim=(-1, -2)) / area           # (B,)

    valid_f = valid.float()
    n_valid = valid_f.sum().clamp_min(1.0)
    L = (per_row * valid_f).sum() / n_valid

    stats = {
        "L_rec_fg": float(L.detach().item()),
        "n_fg_rows": int(valid.sum().item()),
        "fg_area_mean": float(M.sum(dim=(-1, -2)).mean().item()),
    }
    return L, stats


def bg_consistency_reconstruction(
    pred: torch.Tensor,                      # (B, T-1, H, W, C) decoder output
    gt_prev: torch.Tensor,                   # (B, H, W, C) o_t — NOT o_{t+1}
    bg_mask: torch.Tensor,                   # (B, H, W) float in {0,1}
    valid: torch.Tensor,                     # (B,) bool
    *,
    min_pixels: int = 64,
) -> Tuple[torch.Tensor, dict]:
    """L_rec_bg = mean over valid rows of (Σ bg · ||pred - o_t||²_pix) / clamp(Σbg, min_pix).

    Note: target is ``o_t`` (the previous frame), not ``o_{t+1}``. Under the
    camera-fixed assumption, the predicted bg should equal the previous frame's
    bg — decoder doesn't waste capacity on bg.
    """
    pred_f = pred.float()
    if pred_f.dim() == 5:
        pred_f = pred_f.mean(dim=1)                            # (B, H, W, C) avg time
    gt_prev_f = gt_prev.float()
    sq = (pred_f - gt_prev_f) ** 2                             # (B, H, W, C)
    sq_pix = sq.mean(dim=-1)                                   # (B, H, W)

    M = bg_mask.float()
    area = M.sum(dim=(-1, -2)).clamp_min(float(min_pixels))
    per_row = (sq_pix * M).sum(dim=(-1, -2)) / area

    valid_f = valid.float()
    n_valid = valid_f.sum().clamp_min(1.0)
    L = (per_row * valid_f).sum() / n_valid

    stats = {
        "L_rec_bg": float(L.detach().item()),
        "n_bg_rows": int(valid.sum().item()),
        "bg_area_mean": float(M.sum(dim=(-1, -2)).mean().item()),
    }
    return L, stats


__all__ = ["fg_weighted_reconstruction", "bg_consistency_reconstruction"]
