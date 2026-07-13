"""Pairwise action-representation loss for Stage-1 LAM training.

Three pieces:

* :class:`SigLIPHead` — projection head (2-layer MLP + L2 normalize) with
  learnable scale & bias, following the SigLIP recipe (Zhai et al. 2023). The
  contrastive geometry is shaped on ``h = norm(P(z))`` so that ``z_mu`` itself
  remains the latent the WM consumes; we only use ``P`` during training.

* :func:`siglip_action_loss` — pairwise sigmoid contrastive loss with explicit
  positive/negative/ignore graph and per-pair weights for opposite-primitive
  and same-episode hard-negative emphasis. Pos/neg are normalized
  *separately* (pairwise §17) so that scarce positives are not drowned by abundant
  negatives.

* :func:`build_siglip_graph_from_meta` — constructs the ``(B, B)``
  ``valid_pos``, ``valid_neg``, ``neg_weight`` masks from per-row metadata
  already present in ``pair_index_train.parquet``. Implements the pairwise §6/§7
  invariance rules: invariant to background/episode/dataset; sensitive to
  primitive/opposite/contact; explicit ignore for single-segment same-prim,
  low confidence, camera-dominant, and low-motion pairs.

The loss is gated in the trainer behind ``cfg.trainer.loss.action_loss_kind ==
"siglip"``; ``"supcon"`` keeps the earlier recipe path alive for A/B comparison.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Hard-coded opposite-primitive pairs over the canonical 13. These four pairs
# are the action-direction antonyms with clear visual evidence from the same
# scene. The remaining canonical primitives (pour / wipe / turn / insert /
# press) have no clean opposite within the canonical set, so we do not boost
# any negatives involving them.
OPPOSITE_PAIRS: List[Tuple[str, str]] = [
    ("pick", "place"),
    ("open", "close"),
    ("push", "pull"),
    ("fold", "unfold"),
]


def opposite_pair_id_table(canonical: List[str]) -> Dict[int, int]:
    """Return {prim_id_a: prim_id_b, prim_id_b: prim_id_a, ...} for OPPOSITE_PAIRS.

    Primitives in :data:`OPPOSITE_PAIRS` that are missing from ``canonical`` are
    silently skipped; the caller logs the resolved set.
    """
    name_to_id = {p: i for i, p in enumerate(canonical)}
    table: Dict[int, int] = {}
    for a, b in OPPOSITE_PAIRS:
        if a in name_to_id and b in name_to_id:
            table[name_to_id[a]] = name_to_id[b]
            table[name_to_id[b]] = name_to_id[a]
    return table


# =============== Projection head ============================================


class SigLIPHead(nn.Module):
    """Projection head + learnable scale/bias for pairwise sigmoid contrastive.

    Args:
      in_dim:        dim of incoming z (typically ``latent_dim`` = 32).
      proj_dim:      dim of projection output. Default ``in_dim`` (no
                     dim-change). Increase to widen the contrastive sphere
                     if A2 needs more capacity.
      hidden_dim:    width of the GELU hidden layer. Default ``in_dim``.
      init_log_scale: initial value of ``log(scale)``. SigLIP paper uses
                      ``log(10) ≈ 2.302585``. Higher = sharper similarity
                      distribution at init.
      init_bias:     initial value of ``bias``. SigLIP paper uses ``-10.0``
                      (assumes negatives ≫ positives). For our graph the ratio
                      is ~3-5×, so caller may pass ``-5.0``.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_log_scale: float = 2.302585,
        init_bias: float = -10.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim) if proj_dim is not None else self.in_dim
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.in_dim
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.proj_dim),
        )
        # log_scale and bias are scalar buffers exposed as learnable Parameters.
        self.log_scale = nn.Parameter(torch.tensor(float(init_log_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(h, scale, bias)``. ``h`` is L2-normalized along the last dim."""
        h = self.proj(z)
        h = F.normalize(h, dim=-1, eps=1e-8)
        return h, self.log_scale.exp(), self.bias

    def parameters_as_group(self, lr: float, name: str = "G_siglip_head") -> Dict:
        """Convenience: emit a param group dict for the trainer's optimizer."""
        return {"name": name, "params": list(self.parameters()), "lr": float(lr)}


# =============== Graph builder ==============================================


def build_siglip_graph_from_meta(
    primitives: torch.Tensor,                    # (B,) long; -1 = unlabeled
    episodes: torch.Tensor,                      # (B,) long hash
    confidence: torch.Tensor,                    # (B,) float in [0, 1]
    episode_has_multi_primitive: torch.Tensor,   # (B,) bool
    is_camera_dominant: torch.Tensor,            # (B,) bool
    is_low_motion: torch.Tensor,                 # (B,) bool
    opposite_table: Dict[int, int],
    *,
    min_confidence: float = 0.5,
    w_opposite: float = 2.0,
    w_hard_neg: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Build SigLIP-style ``(valid_pos, valid_neg, neg_weight)`` over a batch.

    Rules (pairwise §6/§7):

    * **Positive**: ``same prim AND diff episode`` (cross-episode generalization
      test) AND both ends high-confidence AND healthy. The healthy filter
      excludes pairs where BOTH ends are camera-dominant or BOTH are low-motion
      (those pairs offer no discriminative action signal).

    * **Negative**: ``diff prim AND multi-prim-safe AND healthy AND both
      labeled``. The multi-prim-safe gate ensures that within-episode
      diff-prim pairs are only used when ``episode_has_multi_primitive`` is
      True on both ends — otherwise we may be confusing two phases of the
      same action.

    * **Ignore (implicit)**: anything not in pos or neg. In particular:

      - ``same prim AND same episode`` → ignore (could be different phases of
        the same action; pairwise §7 single-segment safety).
      - low confidence pair → ignore.
      - both ends camera-dominant or both low-motion → ignore.
      - unlabeled primitive on either end → ignore.

    * **Negative weight**: opposite-primitive pairs (open↔close etc.) get
      ``w_opposite`` boost; same-episode diff-primitive (hard negatives, the
      earlier recipe lever) get ``w_hard_neg`` boost. Both can apply (open vs close in
      the same episode → ``w_opposite * w_hard_neg``).

    Returns ``(valid_pos, valid_neg, neg_weight, stats)`` all on the same
    device as ``primitives``. ``stats`` carries diagnostic counts.
    """
    if primitives.dim() != 1:
        raise ValueError(f"primitives must be 1D, got shape {tuple(primitives.shape)}")
    B = primitives.shape[0]
    device = primitives.device

    # Pairwise meta tensors -- broadcast row vs column
    same_prim = primitives.unsqueeze(0) == primitives.unsqueeze(1)
    same_ep = episodes.unsqueeze(0) == episodes.unsqueeze(1)
    diff_prim = ~same_prim
    diff_ep = ~same_ep

    labeled_i = primitives >= 0
    labeled = labeled_i.unsqueeze(0) & labeled_i.unsqueeze(1)

    high_conf_i = confidence >= float(min_confidence)
    high_conf = high_conf_i.unsqueeze(0) & high_conf_i.unsqueeze(1)

    cam_dom_pair = is_camera_dominant.unsqueeze(0) & is_camera_dominant.unsqueeze(1)
    low_mo_pair = is_low_motion.unsqueeze(0) & is_low_motion.unsqueeze(1)
    healthy = ~cam_dom_pair & ~low_mo_pair

    multi_prim_i = episode_has_multi_primitive
    multi_prim_pair = multi_prim_i.unsqueeze(0) & multi_prim_i.unsqueeze(1)
    # Within-episode diff-prim pairs are only safe when both ends mark the
    # episode as multi-primitive. Cross-episode pairs are always safe in this
    # respect.
    multi_prim_safe = diff_ep | multi_prim_pair

    eye = torch.eye(B, dtype=torch.bool, device=device)
    not_diag = ~eye

    valid_pos = same_prim & diff_ep & high_conf & healthy & labeled & not_diag
    valid_neg = diff_prim & multi_prim_safe & healthy & labeled & not_diag

    # Negative weights start at 1.0 wherever valid_neg is True, else 0.0
    neg_weight = valid_neg.float()

    # opposite-pair boost
    opp_pair = torch.zeros((B, B), dtype=torch.bool, device=device)
    if opposite_table:
        # vectorized lookup: for each row's primitive, find its opposite id
        prim_long = primitives.long()
        opp_target = torch.full_like(prim_long, fill_value=-1)
        for src, tgt in opposite_table.items():
            opp_target = torch.where(prim_long == int(src), torch.full_like(prim_long, int(tgt)), opp_target)
        # opposite pair iff prim_j == opp_target_i AND opp_target_i >= 0
        opp_pair = (
            (primitives.unsqueeze(0) == opp_target.unsqueeze(1))
            & (opp_target.unsqueeze(1) >= 0)
        )
        opp_pair = opp_pair & valid_neg
    if float(w_opposite) > 1.0 and opp_pair.any():
        neg_weight = torch.where(opp_pair, neg_weight * float(w_opposite), neg_weight)

    # same-episode hard-neg boost (the earlier recipe lever, now per-pair multiplicative
    # rather than logit-shift in a softmax denominator)
    hard_neg_pair = valid_neg & same_ep
    if float(w_hard_neg) > 1.0 and hard_neg_pair.any():
        neg_weight = torch.where(hard_neg_pair, neg_weight * float(w_hard_neg), neg_weight)

    stats = {
        "n_pos_pairs": int(valid_pos.sum().item()),
        "n_neg_pairs": int(valid_neg.sum().item()),
        "n_opposite_pairs": int(opp_pair.sum().item()),
        "n_hard_neg_pairs": int(hard_neg_pair.sum().item()),
        "n_unlabeled": int((~labeled_i).sum().item()),
        "n_low_conf": int((~high_conf_i).sum().item()),
        "n_cam_dominant": int(is_camera_dominant.sum().item()),
        "n_low_motion": int(is_low_motion.sum().item()),
    }
    return valid_pos, valid_neg, neg_weight, stats


# =============== SigLIP-style pairwise sigmoid loss =========================


def siglip_action_loss(
    z_mu: torch.Tensor,                    # (B, D)
    valid_pos: torch.Tensor,               # (B, B) bool
    valid_neg: torch.Tensor,               # (B, B) bool
    head: SigLIPHead,
    neg_weight: Optional[torch.Tensor] = None,    # (B, B) float; >=1.0 outside neg, gated by valid_neg
    pos_weight: Optional[torch.Tensor] = None,    # (B, B) float; >=1.0 outside pos, gated by valid_pos
) -> Tuple[torch.Tensor, dict]:
    """Pairwise sigmoid contrastive over projected z.

    For each pair (i, j) in ``valid_pos``, the loss is ``softplus(-s_ij)``,
    where ``s_ij = scale * <h_i, h_j> + bias``. For each pair in ``valid_neg``
    it is ``softplus(s_ij)``. Pairs in neither mask are ignored (pairwise §7 ignore).

    Pos and neg terms are normalized *separately* (pairwise §17 "positive >
    negative"): without separate normalization the loss collapses to a global
    average that's dominated by abundant negatives, defeating the SigLIP
    advantage. Each side is divided by its own (weighted) count.

    Args:
      z_mu:        (B, D) raw posterior means (fp32 expected).
      valid_pos:   (B, B) bool — see :func:`build_siglip_graph_from_meta`.
      valid_neg:   (B, B) bool — same.
      head:        SigLIPHead instance (trainable).
      neg_weight:  (B, B) float, only consulted on entries where
                    ``valid_neg=True``. If None, ones.
      pos_weight:  (B, B) float, only consulted on entries where
                    ``valid_pos=True``. If None, ones.

    Returns ``(loss, stats)``. Returns 0 loss when the graph contains no valid
    positives (rare, but possible at very small batches).
    """
    if z_mu.dim() != 2:
        raise ValueError(f"z_mu must be (B, D), got shape {tuple(z_mu.shape)}")
    h, scale, bias = head(z_mu.float())
    # logits in fp32; this is a tiny op, no autocast benefit
    s = scale * (h @ h.t()) + bias                # (B, B), diagonal is finite

    pw = pos_weight if pos_weight is not None else torch.ones_like(s)
    nw = neg_weight if neg_weight is not None else torch.ones_like(s)

    pos_mask_f = valid_pos.float() * pw
    neg_mask_f = valid_neg.float() * nw

    # softplus(-s) for positives, softplus(s) for negatives (numerically stable)
    pos_term = (F.softplus(-s) * pos_mask_f).sum()
    neg_term = (F.softplus(s) * neg_mask_f).sum()
    n_pos = pos_mask_f.sum().clamp_min(1.0)
    n_neg = neg_mask_f.sum().clamp_min(1.0)

    n_pos_real = float(valid_pos.float().sum().item())
    if n_pos_real == 0:
        # Degenerate batch: no positives. Return zero loss WITHOUT pushing
        # negatives (pairwise design says ignore, not push apart, when we don't know).
        # CRITICAL: the loss MUST keep graph through `head` even though its
        # value is zero, otherwise `head.params.grad` ends up None on this
        # rank while other ranks have non-None grads — `all_reduce_grads`
        # then issues a different collective sequence per rank and NCCL hangs
        # ~600s before timing out. Bug observed 2026-05-08 step ~520 of
        # legacy fixed-run. (`(s * 0).sum()` is mathematically zero but has graph
        # through `s -> h -> head.proj.* / scale / bias`.)
        loss = (s * 0.0).sum()
        stats = _siglip_stats(s, scale, bias, valid_pos, valid_neg, h, n_pos_real)
        stats["L_sig_pos"] = 0.0
        stats["L_sig_neg"] = 0.0
        return loss, stats

    L_pos = pos_term / n_pos
    L_neg = neg_term / n_neg
    loss = L_pos + L_neg

    stats = _siglip_stats(s, scale, bias, valid_pos, valid_neg, h, n_pos_real)
    stats["L_sig_pos"] = float(L_pos.detach().item())
    stats["L_sig_neg"] = float(L_neg.detach().item())
    return loss, stats


@torch.no_grad()
def _siglip_stats(
    s: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor,
    valid_pos: torch.Tensor, valid_neg: torch.Tensor, h: torch.Tensor,
    n_pos_real: float,
) -> dict:
    """Diagnostic stats: scale, bias, mean cos on pos / neg, projected pcos."""
    B = s.shape[0]
    diag = torch.eye(B, dtype=torch.bool, device=s.device)
    cos = (h @ h.t()).masked_fill(diag, 0.0)         # raw cosine in projection space
    pcos_proj = float(cos.sum().item() / max(1.0, B * (B - 1)))
    if valid_pos.any():
        cos_pos = float(cos[valid_pos].mean().item())
    else:
        cos_pos = float("nan")
    if valid_neg.any():
        cos_neg = float(cos[valid_neg].mean().item())
    else:
        cos_neg = float("nan")
    s_pos_mean = float(s[valid_pos].mean().item()) if valid_pos.any() else float("nan")
    s_neg_mean = float(s[valid_neg].mean().item()) if valid_neg.any() else float("nan")
    return {
        "siglip_scale": float(scale.detach().item()),
        "siglip_bias": float(bias.detach().item()),
        "pcos_proj_mean": pcos_proj,
        "cos_proj_pos_mean": cos_pos,
        "cos_proj_neg_mean": cos_neg,
        "s_pos_mean": s_pos_mean,
        "s_neg_mean": s_neg_mean,
        "n_pos_pairs": int(valid_pos.sum().item()),
        "n_neg_pairs": int(valid_neg.sum().item()),
    }
