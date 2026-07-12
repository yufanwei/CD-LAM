"""LAM v3 — Route A losses (plan §6.6 / §6.4 / §6.7).

Three families:

* ``centered_supcon_loss``  — supervised contrastive on EMA-centered, L2-normalized
  z (plan §6.6). The centering is the key fix for G1's common-cone (raw pairwise
  cosine ~0.62): we never reward the bias direction.

* ``masked_mse_full_inter`` — full-image MSE + interaction-region MSE, weighted
  by ``alpha_inter`` (plan §6.4). The interaction term is averaged ONLY over rows
  with valid mask, then added to the per-step total; rows without mask contribute
  nothing to the inter term but still contribute to the full term.

* ``masked_usage_gap``      — full + interaction usage-gap hinges (plan §6.7).
  Encourages ``MSE(zero_z) > MSE(real_z) + margin`` and same for shuffle, both
  full and interaction.

Returning a small ``stats`` dict alongside each loss keeps trainer logging trivial.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============== EMA centerer ===============================================


class EMACenterer:
    """Tracks EMA of the per-batch mean of z_mu — the "common cone" direction.

    No grad, no parameters. Updated every step from the current batch's z_mu.
    """

    def __init__(self, dim: int, alpha: float = 0.95, device: Optional[str] = None) -> None:
        self.alpha = float(alpha)
        self.center = torch.zeros(dim, dtype=torch.float32, device=device)
        self.initialized = False

    @torch.no_grad()
    def update(self, z_mu: torch.Tensor) -> None:
        cur = z_mu.detach().float().mean(dim=0)
        if not self.initialized:
            self.center = cur.clone()
            self.initialized = True
        else:
            self.center = self.alpha * self.center + (1.0 - self.alpha) * cur

    @torch.no_grad()
    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "center": self.center.detach().cpu(), "initialized": self.initialized}

    def load_state_dict(self, sd: dict) -> None:
        self.alpha = float(sd.get("alpha", self.alpha))
        c = sd.get("center")
        if c is not None:
            self.center = c.to(self.center.device, dtype=self.center.dtype)
        self.initialized = bool(sd.get("initialized", True))


# =============== Centered SupCon (plan §6.6) =================================


def centered_supcon_loss(
    z_mu: torch.Tensor,
    primitives: torch.Tensor,
    center: torch.Tensor,
    valid_pos: torch.Tensor,
    valid_neg: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
    hard_neg_mask: Optional[torch.Tensor] = None,
    hard_neg_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """SupCon over EMA-centered, L2-normalized z (plan §6.6).

    Args:
      z_mu:        ``(B, D)`` raw posterior means (already fp32).
      primitives:  ``(B,)`` long tensor of primitive class ids; rows with id < 0
                   are excluded as anchors and as positives/negatives.
      center:      ``(D,)`` EMA center (no grad).
      valid_pos:   ``(B, B)`` bool — entry (i, j) is True iff j is a valid
                   *positive* for anchor i (typically: same primitive, different
                   episode, high-confidence). Diagonal must be False.
      valid_neg:   ``(B, B)`` bool — entry (i, j) is True iff j is a valid
                   *negative* for anchor i (typically: different primitive;
                   prefer same episode hard-negs but in-batch fallback OK).
                   Diagonal must be False.
      temperature: SupCon temperature (plan §6.8 default 0.07).
      hard_neg_mask: optional ``(B, B)`` bool. When provided alongside
                     ``hard_neg_weight > 1``, negatives in this mask receive an
                     additive ``log(hard_neg_weight)`` logit shift in the
                     softmax denominator — pushes the anchor harder away from
                     these specific negatives. The standard use (v0.1, plan §6.6)
                     is to mark (same-episode, diff-primitive) pairs so that
                     SupCon attacks the episode-shortcut direction. Numerator
                     (positive anchor-positive sim) is **not** shifted.
      hard_neg_weight: scalar ≥ 1.0. Default 1.0 = unweighted (v0 behaviour).

    Returns ``(loss, stats)``. ``loss`` is 0 if no anchor has both ≥1 positive
    and ≥1 negative.
    """
    B, D = z_mu.shape
    if B == 0:
        return torch.zeros((), device=z_mu.device, dtype=torch.float32), {
            "n_anchors": 0,
            "pairwise_cos_centered_mean": float("nan"),
        }

    # Centered, normalized embeddings
    z_c = z_mu.float() - center.detach().to(z_mu.device, dtype=torch.float32)
    u = F.normalize(z_c, dim=1, eps=eps)                                  # (B, D)

    # Pairwise centered cosine similarity (B, B). NB: we deliberately do NOT
    # masked-fill the diagonal here — sim's diagonal stays finite (= 1/τ since
    # u·u = 1). Multiplying it by 0 (pos_mask is False on diag) is well-defined,
    # whereas `inf * 0` would produce NaN and propagate through the sum.
    sim = u @ u.t() / float(temperature)
    diag_mask = torch.eye(B, dtype=torch.bool, device=z_mu.device)

    # An anchor is valid iff it has at least one positive and one negative
    has_pos = valid_pos.any(dim=1)
    has_neg = valid_neg.any(dim=1)
    is_anchor = has_pos & has_neg & (primitives >= 0)
    n_anchors = int(is_anchor.sum().item())
    if n_anchors == 0:
        with torch.no_grad():
            pc = (u @ u.t()).masked_fill(diag_mask, 0.0)
            pc_mean = float(pc.sum().item() / max(1.0, B * (B - 1)))
        return torch.zeros((), device=z_mu.device, dtype=torch.float32), {
            "n_anchors": 0,
            "pairwise_cos_centered_mean": pc_mean,
        }

    # SupCon (Khosla 2020) numerator over positives, denominator over pos∪neg.
    # log-prob of one positive: sim(a, p) - logsumexp_{j ∈ pos∪neg} sim(a, j)
    # then average over positives, then over anchors.
    contrast = (valid_pos | valid_neg) & ~diag_mask                       # (B, B) bool
    sim_shifted = sim
    n_hard_neg = 0
    if hard_neg_mask is not None and float(hard_neg_weight) > 1.0:
        # Additive logit shift on the denominator: w_j * exp(sim_j) is equivalent
        # to exp(sim_j + log(w_j)). Only negatives in `hard_neg_mask` get the
        # shift; positives & non-hard negatives keep their original logit.
        # NB: this is applied to the LSE input only — the per-positive numerator
        # below uses the unshifted `sim`.
        log_w = float(np.log(float(hard_neg_weight)))
        shift = (hard_neg_mask & valid_neg).float() * log_w
        sim_shifted = sim + shift
        n_hard_neg = int((hard_neg_mask & valid_neg).sum().item())
    sim_for_lse = sim_shifted.masked_fill(~contrast, float("-inf"))
    log_denom = torch.logsumexp(sim_for_lse, dim=1)                       # (B,)

    # per-anchor per-positive loss = log_denom - sim_pos. sim's diagonal is
    # finite (1/τ); valid_pos.float() is 0 on the diagonal, so the diag entries
    # contribute 0 cleanly. Non-anchor rows may have log_denom = -inf, but they
    # are dropped by the `is_anchor` mask before mean().
    per_pos_loss = log_denom.unsqueeze(1) - sim                           # (B, B)
    pos_mask_f = valid_pos.float()
    n_pos_per_anchor = pos_mask_f.sum(dim=1).clamp_min(1.0)
    per_anchor_loss = (per_pos_loss * pos_mask_f).sum(dim=1) / n_pos_per_anchor
    loss = per_anchor_loss[is_anchor].mean()

    # diagnostics: centered pairwise cosine mean (off-diagonal)
    with torch.no_grad():
        pc = (u @ u.t()).masked_fill(diag_mask, 0.0)
        pc_mean = float(pc.sum().item() / max(1.0, B * (B - 1)))
    stats = {
        "n_anchors": n_anchors,
        "pairwise_cos_centered_mean": pc_mean,
        "n_pos_avg": float(n_pos_per_anchor[is_anchor].mean().item()) if n_anchors > 0 else 0.0,
        "n_neg_avg": float(valid_neg.float().sum(dim=1)[is_anchor].mean().item()) if n_anchors > 0 else 0.0,
        "n_hard_neg_pairs": n_hard_neg,
        "hard_neg_weight_used": float(hard_neg_weight) if hard_neg_mask is not None else 1.0,
    }
    return loss, stats


def build_supcon_masks_from_meta(
    primitives: torch.Tensor,
    episodes: torch.Tensor,
    confidence: torch.Tensor,
    is_hard_neg_episode: torch.Tensor,
    min_confidence: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(valid_pos, valid_neg, hard_neg_mask)`` (B, B) bool masks for SupCon
    over an in-batch of anchors.

    - **valid_pos**: same primitive, different episode, both ends high-confidence.
    - **valid_neg**: different primitive (the candidate-negative set).
    - **hard_neg_mask** (NEW in v0.1): the SUBSET of ``valid_neg`` that is also
      same-episode — these are the "diff-primitive same-episode" pairs that v0
      analysis (REPORT §6.2) identified as the gap-killer. ``centered_supcon_loss``
      can give them a logit-shift weight (`hard_neg_weight > 1`) to attack
      episode-shortcut directly.

    All inputs are 1D tensors of length B.

    ``is_hard_neg_episode`` is currently informational (kept for future
    sampler-level weighting); the per-pair hard-neg mask is computed inline from
    ``primitives`` and ``episodes`` since that's what the loss actually needs.
    """
    B = primitives.shape[0]
    device = primitives.device
    same_prim = primitives.unsqueeze(0) == primitives.unsqueeze(1)        # (B, B)
    same_ep = episodes.unsqueeze(0) == episodes.unsqueeze(1)              # (B, B)
    valid_id = primitives >= 0
    valid_pair = valid_id.unsqueeze(0) & valid_id.unsqueeze(1)            # both ends labeled
    high_conf = (confidence >= float(min_confidence))
    high_conf_pair = high_conf.unsqueeze(0) & high_conf.unsqueeze(1)

    valid_pos = same_prim & (~same_ep) & valid_pair & high_conf_pair
    diff_prim = (~same_prim) & valid_pair
    valid_neg = diff_prim
    # Hard negatives = the in-batch pairs where the negative shares anchor's
    # episode but differs in primitive. v0 final eval showed
    # `same_p_diff_e ≈ diff_p_same_e` — these pairs need extra weight to break
    # the episode shortcut.
    hard_neg_mask = diff_prim & same_ep
    _ = is_hard_neg_episode  # informational only

    eye = torch.eye(B, dtype=torch.bool, device=device)
    valid_pos = valid_pos & ~eye
    valid_neg = valid_neg & ~eye
    hard_neg_mask = hard_neg_mask & ~eye
    return valid_pos, valid_neg, hard_neg_mask


# =============== Hard-neg triplet auxiliary (v0.4, REPORT §6.5 Option B) ====


def triplet_hardneg_aux_loss(
    z_mu: torch.Tensor,
    center: torch.Tensor,
    valid_pos: torch.Tensor,
    hard_neg_mask: torch.Tensor,
    margin: float = 0.10,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Triplet hinge directly on `cos(a,p) - cos(a,n)` for (anchor, hard-neg) pairs.

    For each anchor row i, find:
        positives p ∈ valid_pos[i, :]   (same prim, diff ep)
        hard-negs n ∈ hard_neg_mask[i, :]   (same ep, diff prim)
    Loss = mean over (i, p, n): ReLU(margin - cos_centered(i, p) + cos_centered(i, n))

    This is exactly the F1_v3 triplet form (which got gap=+0.065) but operating
    on EMA-centered, normalized z. v0/v0.1/v0.2/v0.3 used SupCon InfoNCE which
    optimizes class log-prob; SupCon doesn't natively optimize the
    `cos(a,p) - cos(a,n)` margin, so gap drifts to ~0 in main phase. This aux
    loss puts gap back as a first-order optimization target.

    Returns (loss, stats). loss = 0 if no anchor has both ≥1 pos and ≥1 hard-neg.
    """
    B, D = z_mu.shape
    if B == 0:
        return torch.zeros((), device=z_mu.device, dtype=torch.float32), {
            "n_aux_anchors": 0, "n_aux_triplets": 0, "L_aux_mean_gap": float("nan"),
        }

    z_c = z_mu.float() - center.detach().to(z_mu.device, dtype=torch.float32)
    u = torch.nn.functional.normalize(z_c, dim=1, eps=eps)
    cos = u @ u.t()                                                         # (B, B), centered cos

    # For each anchor, sample at most 1 positive and 1 hard-neg per partner pair.
    # Using broadcasting: build (B, B, B) tensor would be too big. Instead, for
    # each anchor, average cos over valid_pos[i,:] and hard_neg_mask[i,:].
    # The triplet hinge becomes:
    #   ReLU( margin - mean_p cos(i, p) + mean_n cos(i, n) )
    # This is equivalent to applying triplet to (anchor's mean positive, anchor's
    # mean hard-neg) — slightly weaker per-pair signal but linear in B.
    pos_mask_f = valid_pos.float()
    neg_mask_f = hard_neg_mask.float()
    n_pos = pos_mask_f.sum(dim=1)
    n_neg = neg_mask_f.sum(dim=1)

    has_pos_neg = (n_pos > 0) & (n_neg > 0)
    n_aux_anchors = int(has_pos_neg.sum().item())
    if n_aux_anchors == 0:
        return torch.zeros((), device=z_mu.device, dtype=torch.float32), {
            "n_aux_anchors": 0, "n_aux_triplets": 0, "L_aux_mean_gap": float("nan"),
        }

    cos_pos_per_anchor = (cos * pos_mask_f).sum(dim=1) / n_pos.clamp_min(1.0)
    cos_neg_per_anchor = (cos * neg_mask_f).sum(dim=1) / n_neg.clamp_min(1.0)
    gap_per_anchor = cos_pos_per_anchor - cos_neg_per_anchor                # (B,)

    L_per = torch.clamp(float(margin) - gap_per_anchor, min=0.0)            # (B,)
    loss = L_per[has_pos_neg].mean()

    n_aux_triplets = int((pos_mask_f.sum(dim=1) * neg_mask_f.sum(dim=1))[has_pos_neg].sum().item())
    stats = {
        "n_aux_anchors": n_aux_anchors,
        "n_aux_triplets": n_aux_triplets,
        "L_aux_mean_gap": float(gap_per_anchor[has_pos_neg].mean().item()),
        "L_aux_mean_loss": float(loss.detach().item()),
    }
    return loss, stats


# =============== Masked reconstruction (plan §6.4) ===========================


def masked_mse_full_inter(
    pred: torch.Tensor,                  # (B, T-1, H, W, C) recon (sigmoid)
    gt: torch.Tensor,                    # (B, T-1, H, W, C) target
    mask_inter: Optional[torch.Tensor],  # (Bm, H, W) float mask in {0, 1} (frame_j)
    mask_anchor_idx: Optional[torch.Tensor],  # (Bm,) long indices into batch B
    alpha_inter: float = 2.0,
    min_pixels: int = 64,
) -> Tuple[torch.Tensor, dict]:
    """Returns ``L_rec_full + alpha_inter * L_rec_interaction`` and stats.

    ``L_rec_full`` is the standard mean-squared error over the whole batch.
    ``L_rec_interaction`` averages MSE only inside the per-row interaction
    mask, ONLY for rows that have a valid mask (``mask_anchor_idx``); each row
    is normalized by its mask area (``clamp(sum(M), min_pixels)``) before
    averaging across the masked subset.

    Mixed precision: caller is responsible for fp32 inputs (encoder/decoder may
    have run under bf16 autocast). This function does not cast.
    """
    pred_f = pred.float()
    gt_f = gt.float()
    sq = (pred_f - gt_f) ** 2                                             # (B, T-1, H, W, C)
    L_full = sq.mean()

    stats = {
        "L_rec_full": float(L_full.detach().item()),
        "n_inter_rows": 0,
        "L_rec_inter": float("nan"),
    }

    if (
        mask_inter is None
        or mask_anchor_idx is None
        or mask_inter.numel() == 0
        or mask_anchor_idx.numel() == 0
    ):
        return L_full, stats

    # We assume T-1 == 1 (LAM v2/v3 predicts a single future frame). Generalize
    # by averaging across the T-1 axis if it's >1.
    sq_sub = sq.index_select(0, mask_anchor_idx)                          # (Bm, T-1, H, W, C)
    if sq_sub.dim() == 5:
        # average across temporal axis first (T-1 typically = 1, so this is trivial)
        sq_sub = sq_sub.mean(dim=1)                                       # (Bm, H, W, C)
    # average across channels
    sq_pix = sq_sub.mean(dim=-1)                                          # (Bm, H, W)

    M = mask_inter.float()                                                # (Bm, H, W)
    area = M.sum(dim=(-1, -2)).clamp_min(float(min_pixels))               # (Bm,)
    per_row = (sq_pix * M).sum(dim=(-1, -2)) / area                       # (Bm,)
    L_inter = per_row.mean()

    L_total = L_full + float(alpha_inter) * L_inter
    stats["n_inter_rows"] = int(mask_anchor_idx.numel())
    stats["L_rec_inter"] = float(L_inter.detach().item())
    stats["L_rec_inter_weighted"] = float(L_total.detach().item() - L_full.detach().item())
    return L_total, stats


# =============== Masked usage gap (plan §6.7) ================================


def usage_gap_full(
    pred_real: torch.Tensor,             # (B, T-1, H, W, C)
    pred_zero: torch.Tensor,             # (B, T-1, H, W, C)
    pred_shuf: torch.Tensor,             # (B, T-1, H, W, C)
    gt: torch.Tensor,                    # (B, T-1, H, W, C)
    margin_full: float,
) -> Tuple[torch.Tensor, dict]:
    """Full-frame L_use hinge (plan §6.7 first half).

    ``L_use_full = ReLU(margin - (MSE_zero - MSE_real)) + ReLU(margin - (MSE_shuf - MSE_real))``.

    All inputs assumed fp32 already.
    """
    sq_real = (pred_real.float() - gt.float()) ** 2
    sq_zero = (pred_zero.float() - gt.float()) ** 2
    sq_shuf = (pred_shuf.float() - gt.float()) ** 2
    L_real = sq_real.mean()
    L_zero = sq_zero.mean()
    L_shuf = sq_shuf.mean()
    gap_zero = L_zero - L_real
    gap_shuf = L_shuf - L_real
    L_use = (
        torch.clamp(float(margin_full) - gap_zero, min=0.0)
        + torch.clamp(float(margin_full) - gap_shuf, min=0.0)
    )
    stats = {
        "L_use_full": float(L_use.detach().item()),
        "usage_gap_zero_full": float(gap_zero.detach().item()),
        "usage_gap_shuffle_full": float(gap_shuf.detach().item()),
    }
    return L_use, stats


def usage_gap_inter(
    pred_real: torch.Tensor,             # (Bm, T-1, H, W, C)
    pred_zero: torch.Tensor,             # (Bm, T-1, H, W, C)
    pred_shuf: torch.Tensor,             # (Bm, T-1, H, W, C)
    gt: torch.Tensor,                    # (Bm, T-1, H, W, C)
    mask_inter: torch.Tensor,            # (Bm, H, W) float in {0,1}
    margin_inter: float,
    min_pixels: int = 64,
) -> Tuple[torch.Tensor, dict]:
    """Interaction-region L_use hinge (plan §6.7 second half).

    Each row's MSE is averaged inside its own mask only; per-row gaps are then
    averaged over the masked subset and pushed above ``margin_inter``.

    All inputs assumed fp32 already.
    """
    def _avg_in_mask(sq):
        sub = sq.float()
        if sub.dim() == 5:
            sub = sub.mean(dim=1)                                         # (Bm, H, W, C)
        sub_pix = sub.mean(dim=-1)                                        # (Bm, H, W)
        M = mask_inter.float()
        area = M.sum(dim=(-1, -2)).clamp_min(float(min_pixels))
        return ((sub_pix * M).sum(dim=(-1, -2)) / area).mean()

    sq_real = (pred_real.float() - gt.float()) ** 2
    sq_zero = (pred_zero.float() - gt.float()) ** 2
    sq_shuf = (pred_shuf.float() - gt.float()) ** 2
    L_real = _avg_in_mask(sq_real)
    L_zero = _avg_in_mask(sq_zero)
    L_shuf = _avg_in_mask(sq_shuf)
    gap_zero = L_zero - L_real
    gap_shuf = L_shuf - L_real
    L_use = (
        torch.clamp(float(margin_inter) - gap_zero, min=0.0)
        + torch.clamp(float(margin_inter) - gap_shuf, min=0.0)
    )
    stats = {
        "L_use_inter": float(L_use.detach().item()),
        "usage_gap_zero_inter": float(gap_zero.detach().item()),
        "usage_gap_shuffle_inter": float(gap_shuf.detach().item()),
        "n_use_inter_rows": int(mask_inter.shape[0]),
    }
    return L_use, stats
