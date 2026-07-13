"""LAM pairwise+ loss / graph extensions.

Adds three new pieces on top of the pairwise A1 / pairwise module
(``cdlam_integration.lam.contrastive``):

* :func:`build_structured_siglip_graph` — same rules as pairwise plus:
  - cross-dataset positive boost (``w_pos_cross_dataset``)
  - opposite × hard-neg combination mode (``neg_weight_mode={multiplicative, max}``)
  - final ``w_neg_cap`` clamp.
  Returns ``pos_weight`` as a 5th element so callers can boost the per-pair
  positive contribution without touching the loss internals.

* :func:`siglip_action_loss_split` — splits the SigLIP loss into a separately-
  weighted positive term and negative term:

      L = λ_pos · L_pos + λ_neg · L_neg

  Each term is normalized by its own (weighted) mask sum, exactly like
  :func:`siglip_action_loss`. Callers schedule λ_pos and λ_neg separately so
  that the negative push can ramp up early to spread, then ramp down to let
  positive consolidation happen in main phase.

* :func:`hard_positive_reweight` — multiplicatively boosts pos_weight for pairs
  whose current cosine is below ``tau_pos``, gated to high-confidence pairs
  (pairwise mining).

Per-pairwise design (pairwise.md §17), positive cloud consolidation is the next lever after
pairwise already pushed negatives apart. These hooks are designed to be additive on
top of pairwise — leaving everything default reproduces pairwise behaviour.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# re-export so callers import everything from one place
from cdlam_integration.lam.contrastive import (  # noqa: F401
    SigLIPHead,
    OPPOSITE_PAIRS,
    opposite_pair_id_table,
    siglip_action_loss,
    build_siglip_graph_from_meta,
    _siglip_stats,
)


def build_structured_siglip_graph(
    primitives: torch.Tensor,                    # (B,) long; -1 = unlabeled
    episodes: torch.Tensor,                      # (B,) long hash
    confidence: torch.Tensor,                    # (B,) float in [0, 1]
    episode_has_multi_primitive: torch.Tensor,   # (B,) bool
    is_camera_dominant: torch.Tensor,            # (B,) bool
    is_low_motion: torch.Tensor,                 # (B,) bool
    opposite_table: Dict[int, int],
    *,
    datasets: Optional[torch.Tensor] = None,     # (B,) long hash for w_pos_cross_dataset
    min_confidence: float = 0.5,
    w_opposite: float = 2.0,
    w_hard_neg: float = 1.5,
    w_pos_cross_dataset: float = 1.0,            # NEW (pairwise): boost on same-prim cross-ds positives
    neg_weight_mode: str = "multiplicative",     # NEW (pairwise): {"multiplicative", "max"}
    w_neg_cap: float = float("inf"),             # NEW (pairwise): final clamp on neg_weight
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Same rules as pairwise ``build_siglip_graph_from_meta`` plus three pairwise hooks.

    Returns ``(valid_pos, valid_neg, pos_weight, neg_weight, stats)``.
    Default kwargs reproduce pairwise behavior with ``pos_weight = valid_pos.float()``
    (uniform 1.0) and ``neg_weight = multiplicative opposite·hard_neg`` (matches
    pairwise / pairwise A1 multiplicative mode).

    pairwise changes:

    * ``w_pos_cross_dataset > 1.0``: positive pairs whose two endpoints belong
      to *different* datasets get this multiplicative boost in ``pos_weight``.
      Forces the loss to invest more gradient on AgiBot↔Bridge same-prim pairs,
      which pairwise dashboard showed are the leakage@5 ≈ 0.91 weak spot.
    * ``neg_weight_mode = "max"``: instead of ``opp × hard_neg`` (multiplicative,
      which gave weight 3.0 to opposite-AND-same-ep pairs in pairwise), use
      ``max(opp_weight, hard_neg_weight)`` (capped at ``w_opposite``).
      pairwise metrics suggested neg pressure was already saturating; capping makes
      room for positive consolidation in main phase.
    * ``w_neg_cap``: final pointwise clamp on neg_weight. Independent of mode.
    """
    if neg_weight_mode not in {"multiplicative", "max"}:
        raise ValueError(f"neg_weight_mode must be 'multiplicative' or 'max', "
                         f"got {neg_weight_mode!r}")
    if primitives.dim() != 1:
        raise ValueError(f"primitives must be 1D, got shape {tuple(primitives.shape)}")
    B = primitives.shape[0]
    device = primitives.device

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
    multi_prim_safe = diff_ep | multi_prim_pair

    eye = torch.eye(B, dtype=torch.bool, device=device)
    not_diag = ~eye

    valid_pos = same_prim & diff_ep & high_conf & healthy & labeled & not_diag
    valid_neg = diff_prim & multi_prim_safe & healthy & labeled & not_diag

    # ---- pos weight: 1.0 baseline; cross-dataset positives get a boost ----
    pos_weight = valid_pos.float()
    cross_ds_pair = torch.zeros((B, B), dtype=torch.bool, device=device)
    if datasets is not None:
        cross_ds_pair = (datasets.unsqueeze(0) != datasets.unsqueeze(1))
    if datasets is not None and float(w_pos_cross_dataset) > 1.0:
        cross_ds_pos = valid_pos & cross_ds_pair
        if cross_ds_pos.any():
            pos_weight = torch.where(
                cross_ds_pos, pos_weight * float(w_pos_cross_dataset), pos_weight,
            )

    # ---- neg weight: opposite + hard-neg, combined per `neg_weight_mode` ----
    neg_weight = valid_neg.float()
    opp_pair = torch.zeros((B, B), dtype=torch.bool, device=device)
    if opposite_table:
        prim_long = primitives.long()
        opp_target = torch.full_like(prim_long, fill_value=-1)
        for src, tgt in opposite_table.items():
            opp_target = torch.where(
                prim_long == int(src),
                torch.full_like(prim_long, int(tgt)),
                opp_target,
            )
        opp_pair = (
            (primitives.unsqueeze(0) == opp_target.unsqueeze(1))
            & (opp_target.unsqueeze(1) >= 0)
        )
        opp_pair = opp_pair & valid_neg

    hard_neg_pair = valid_neg & same_ep

    if neg_weight_mode == "multiplicative":
        if float(w_opposite) > 1.0 and opp_pair.any():
            neg_weight = torch.where(
                opp_pair, neg_weight * float(w_opposite), neg_weight,
            )
        if float(w_hard_neg) > 1.0 and hard_neg_pair.any():
            neg_weight = torch.where(
                hard_neg_pair, neg_weight * float(w_hard_neg), neg_weight,
            )
    else:  # "max"
        # Replace neg_weight with the maximum of (1.0, w_opposite if opp, w_hard_neg if hn).
        if float(w_opposite) > 1.0 and opp_pair.any():
            neg_weight = torch.where(
                opp_pair,
                torch.maximum(neg_weight, torch.full_like(neg_weight, float(w_opposite))),
                neg_weight,
            )
        if float(w_hard_neg) > 1.0 and hard_neg_pair.any():
            neg_weight = torch.where(
                hard_neg_pair,
                torch.maximum(neg_weight, torch.full_like(neg_weight, float(w_hard_neg))),
                neg_weight,
            )

    # final cap (clamp to w_neg_cap)
    if float(w_neg_cap) < float("inf"):
        neg_weight = torch.minimum(neg_weight, torch.full_like(neg_weight, float(w_neg_cap)))

    stats = {
        "n_pos_pairs": int(valid_pos.sum().item()),
        "n_neg_pairs": int(valid_neg.sum().item()),
        "n_opposite_pairs": int(opp_pair.sum().item()),
        "n_hard_neg_pairs": int(hard_neg_pair.sum().item()),
        "n_cross_ds_pos_pairs": int((valid_pos & cross_ds_pair).sum().item()),
        "n_unlabeled": int((~labeled_i).sum().item()),
        "n_low_conf": int((~high_conf_i).sum().item()),
        "n_cam_dominant": int(is_camera_dominant.sum().item()),
        "n_low_motion": int(is_low_motion.sum().item()),
        "neg_weight_mode": neg_weight_mode,
        "w_neg_cap": float(w_neg_cap) if float(w_neg_cap) < float("inf") else -1.0,
    }
    return valid_pos, valid_neg, pos_weight, neg_weight, stats


def hard_positive_reweight(
    pos_weight: torch.Tensor,                # (B, B) float, gated on valid_pos
    valid_pos: torch.Tensor,                 # (B, B) bool
    cos: torch.Tensor,                       # (B, B) float — current proj-cos (no grad)
    confidence: torch.Tensor,                # (B,) float
    *,
    tau_pos: float = 0.20,
    gamma: float = 2.0,
    high_conf_min: float = 0.8,
    max_weight: float = 3.0,
) -> torch.Tensor:
    """pairwise hard-positive mining.

    For positive pairs that are *currently* too far apart (cos < ``tau_pos``)
    *and* high-confidence (both ends conf ≥ ``high_conf_min``), multiply
    ``pos_weight`` by ``1 + gamma * (tau_pos - cos)``. Capped at ``max_weight``.

    Pairs not in valid_pos / not high-confidence get unchanged weight. The
    boost is one-sided: well-clustered positives (cos ≥ tau_pos) are not
    reweighted up.
    """
    high_conf_i = confidence >= float(high_conf_min)
    high_conf_pair = high_conf_i.unsqueeze(0) & high_conf_i.unsqueeze(1)
    eligible = valid_pos & high_conf_pair
    deficit = (float(tau_pos) - cos.detach()).clamp_min(0.0)
    boost = (1.0 + float(gamma) * deficit).clamp_max(float(max_weight))
    pos_weight = torch.where(eligible, pos_weight * boost, pos_weight)
    return pos_weight


def siglip_action_loss_split(
    z_mu: torch.Tensor,                       # (B, D)
    valid_pos: torch.Tensor,                  # (B, B) bool
    valid_neg: torch.Tensor,                  # (B, B) bool
    head: SigLIPHead,
    *,
    pos_weight: Optional[torch.Tensor] = None,    # (B, B) float, gated on valid_pos
    neg_weight: Optional[torch.Tensor] = None,    # (B, B) float, gated on valid_neg
    lambda_pos: float = 1.0,
    lambda_neg: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Pairwise sigmoid contrastive with separate λ for pos and neg.

    Replaces ``siglip_action_loss`` for pairwise+. The loss is

        L = λ_pos · (Σ softplus(-s_ij) · pos_weight_ij) / Σ pos_weight_ij
          + λ_neg · (Σ softplus(s_ij)  · neg_weight_ij) / Σ neg_weight_ij

    Each side is **separately normalized** by its own weighted mass; that's the
    same per-side normalization as pairwise ``siglip_action_loss``, just with the
    two halves multiplied by independent λ scalars.

    Returns ``(loss, stats)``. When n_pos == 0, returns a graph-bearing zero
    loss (DDP-safe — all ranks issue identical collectives).
    """
    if z_mu.dim() != 2:
        raise ValueError(f"z_mu must be (B, D), got shape {tuple(z_mu.shape)}")
    h, scale, bias = head(z_mu.float())
    s = scale * (h @ h.t()) + bias

    pw = pos_weight if pos_weight is not None else valid_pos.float()
    nw = neg_weight if neg_weight is not None else valid_neg.float()

    pos_term = (F.softplus(-s) * pw).sum()
    neg_term = (F.softplus(s) * nw).sum()
    n_pos_w = pw.sum().clamp_min(1.0)
    n_neg_w = nw.sum().clamp_min(1.0)

    n_pos_real = float(valid_pos.float().sum().item())
    if n_pos_real == 0:
        loss = (s * 0.0).sum()
        stats = _siglip_stats(s, scale, bias, valid_pos, valid_neg, h, n_pos_real)
        stats["L_sig_pos"] = 0.0
        stats["L_sig_neg"] = 0.0
        stats["lambda_pos"] = float(lambda_pos)
        stats["lambda_neg"] = float(lambda_neg)
        return loss, stats

    L_pos = pos_term / n_pos_w
    L_neg = neg_term / n_neg_w
    loss = float(lambda_pos) * L_pos + float(lambda_neg) * L_neg

    stats = _siglip_stats(s, scale, bias, valid_pos, valid_neg, h, n_pos_real)
    stats["L_sig_pos"] = float(L_pos.detach().item())
    stats["L_sig_neg"] = float(L_neg.detach().item())
    stats["lambda_pos"] = float(lambda_pos)
    stats["lambda_neg"] = float(lambda_neg)
    return loss, stats


def build_neg_subgroup_masks(
    valid_neg: torch.Tensor,                  # (B, B) bool
    primitives: torch.Tensor,                 # (B,) long
    episodes: torch.Tensor,                   # (B,) long
    opposite_table: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pairwise — split valid_neg into 3 disjoint subgroups for separate λ scheduling.

    Returns ``(opp_mask, hard_mask, rand_mask)`` all (B, B) bool. Disjoint:
    opp ∩ hard ∩ rand = ∅; opp ∪ hard ∪ rand = valid_neg.

    * **opp_mask**: opposite-primitive pairs (open/close, push/pull, etc.).
      Highest-priority direction signal — prevents push-vs-pull collapse.
    * **hard_mask**: same-episode different-primitive pairs that are NOT in
      opp_mask. Prevents video-shortcut.
    * **rand_mask**: rest of valid_neg (cross-episode generic diff-prim).
      Lightweight separation only.

    Priority: opp > hard > rand. So opp wins over hard if a pair is both
    same-episode AND opposite-primitive.
    """
    B = primitives.shape[0]
    device = primitives.device

    # opposite-primitive pair mask
    opp_pair = torch.zeros((B, B), dtype=torch.bool, device=device)
    if opposite_table:
        prim_long = primitives.long()
        opp_target = torch.full_like(prim_long, fill_value=-1)
        for src, tgt in opposite_table.items():
            opp_target = torch.where(
                prim_long == int(src),
                torch.full_like(prim_long, int(tgt)),
                opp_target,
            )
        opp_pair = (
            (primitives.unsqueeze(0) == opp_target.unsqueeze(1))
            & (opp_target.unsqueeze(1) >= 0)
        )

    same_ep = episodes.unsqueeze(0) == episodes.unsqueeze(1)

    opp_mask = valid_neg & opp_pair
    hard_mask = valid_neg & same_ep & ~opp_mask
    rand_mask = valid_neg & ~opp_mask & ~hard_mask
    return opp_mask, hard_mask, rand_mask


def grouped_action_contrastive_loss(
    z_mu: torch.Tensor,                       # (B, D)
    valid_pos: torch.Tensor,                  # (B, B) bool
    head: SigLIPHead,
    neg_subgroups: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    pos_weight: Optional[torch.Tensor] = None,
    is_xds_pair: Optional[torch.Tensor] = None,    # (B, B) bool, dataset_i != dataset_j
    lambda_pos_same: float = 0.006,
    lambda_pos_xds: float = 0.010,
    lambda_neg_opp: float = 0.012,
    lambda_neg_hard: float = 0.008,
    lambda_neg_rand: float = 0.003,
    neg_cap_default: float = 2.0,
    neg_cap_opposite: float = 3.0,
) -> Tuple[torch.Tensor, dict]:
    """pairwise — structured siglip loss with 5 separate λ knobs.

    L = λ_pos_same · L_pos_same_ds + λ_pos_xds · L_pos_xds
      + λ_opp · L_neg_opposite + λ_hard · L_neg_hard + λ_rand · L_neg_rand

    Each term is its own softplus mean over its mask. Per-pair
    ``pos_weight`` (when provided) and the per-subgroup neg_cap are honored.

    The opposite-primitive subgroup gets ``neg_cap_opposite`` (>= ``neg_cap_default``)
    so that direction signal isn't washed out by the global cap.
    """
    if z_mu.dim() != 2:
        raise ValueError(f"z_mu must be (B, D), got shape {tuple(z_mu.shape)}")
    h, scale, bias = head(z_mu.float())
    s = scale * (h @ h.t()) + bias

    opp_mask, hard_mask, rand_mask = neg_subgroups

    # pos: split into same-ds and cross-ds when is_xds_pair available
    if is_xds_pair is not None:
        pos_xds_mask = valid_pos & is_xds_pair
        pos_same_mask = valid_pos & ~is_xds_pair
    else:
        # all positives go in pos_same group
        pos_xds_mask = torch.zeros_like(valid_pos)
        pos_same_mask = valid_pos

    # apply optional per-pair pos_weight (e.g. hard-positive boost)
    pw = pos_weight if pos_weight is not None else torch.ones_like(s)
    pos_same_w = pos_same_mask.float() * pw
    pos_xds_w = pos_xds_mask.float() * pw

    # per-subgroup neg cap: clamp opp at neg_cap_opposite, others at neg_cap_default
    opp_w = opp_mask.float().clamp_max(float(neg_cap_opposite))
    hard_w = hard_mask.float().clamp_max(float(neg_cap_default))
    rand_w = rand_mask.float().clamp_max(float(neg_cap_default))

    # softplus terms, separately normalized
    L_pos_same = (F.softplus(-s) * pos_same_w).sum() / pos_same_w.sum().clamp_min(1.0)
    L_pos_xds = (F.softplus(-s) * pos_xds_w).sum() / pos_xds_w.sum().clamp_min(1.0)
    L_neg_opp = (F.softplus(s) * opp_w).sum() / opp_w.sum().clamp_min(1.0)
    L_neg_hard = (F.softplus(s) * hard_w).sum() / hard_w.sum().clamp_min(1.0)
    L_neg_rand = (F.softplus(s) * rand_w).sum() / rand_w.sum().clamp_min(1.0)

    n_pos_real = float(valid_pos.float().sum().item())
    if n_pos_real == 0:
        # graph-bearing zero (DDP-safe)
        loss = (s * 0.0).sum()
    else:
        loss = (
            float(lambda_pos_same) * L_pos_same
            + float(lambda_pos_xds) * L_pos_xds
            + float(lambda_neg_opp) * L_neg_opp
            + float(lambda_neg_hard) * L_neg_hard
            + float(lambda_neg_rand) * L_neg_rand
        )

    stats = _siglip_stats(s, scale, bias, valid_pos, opp_mask | hard_mask | rand_mask, h, n_pos_real)
    stats["L_pos_same"] = float(L_pos_same.detach().item())
    stats["L_pos_xds"] = float(L_pos_xds.detach().item())
    stats["L_neg_opp"] = float(L_neg_opp.detach().item())
    stats["L_neg_hard"] = float(L_neg_hard.detach().item())
    stats["L_neg_rand"] = float(L_neg_rand.detach().item())
    stats["lambda_pos_same"] = float(lambda_pos_same)
    stats["lambda_pos_xds"] = float(lambda_pos_xds)
    stats["lambda_neg_opp"] = float(lambda_neg_opp)
    stats["lambda_neg_hard"] = float(lambda_neg_hard)
    stats["lambda_neg_rand"] = float(lambda_neg_rand)
    stats["n_pos_same"] = int(pos_same_mask.sum().item())
    stats["n_pos_xds"] = int(pos_xds_mask.sum().item())
    stats["n_neg_opp"] = int(opp_mask.sum().item())
    stats["n_neg_hard"] = int(hard_mask.sum().item())
    stats["n_neg_rand"] = int(rand_mask.sum().item())
    return loss, stats


def xds_ranking_centroid_loss(
    h: torch.Tensor,                          # (B, P) projected embeddings
    primitives: torch.Tensor,                 # (B,) long
    datasets: torch.Tensor,                   # (B,) long
    *,
    n_primitives: int,
    n_datasets: int = 2,
    min_count_per_cell: int = 4,
    margin: float = 0.05,
) -> Tuple[torch.Tensor, dict]:
    """pairwise — ranking-style xds centroid loss (replaces pairwise raw alignment).

    For each qualified primitive p (≥ min_count_per_cell in each dataset),
    cross-dataset same-prim centroids must be closer than cross-dataset
    different-prim centroids by a margin:

        L_xds_rank = mean_{p, q != p, d_a != d_b}
            relu(margin + cos(c_{p,d_a}, c_{q,d_b}) - cos(c_{p,d_a}, c_{p,d_b}))

    This pulls open_AgiBot ↔ open_Bridge close, but only enough that the
    same-prim pair beats the diff-prim pair by ``margin`` — does not collapse
    the geometry.
    """
    valid_terms = []
    n_qualified_primitives = 0

    # build (P, D, dim) centroid table; -1 row marks invalid
    cent: Dict[Tuple[int, int], torch.Tensor] = {}
    for p in range(n_primitives):
        prim_mask = primitives == p
        for d in range(n_datasets):
            cell_mask = prim_mask & (datasets == d)
            if cell_mask.sum() < min_count_per_cell:
                continue
            c = h[cell_mask].mean(dim=0)
            cent[(p, d)] = F.normalize(c, dim=-1, eps=1e-8)

    # qualified primitives: have all datasets
    qualified = [p for p in range(n_primitives)
                 if all((p, d) in cent for d in range(n_datasets))]
    n_qualified_primitives = len(qualified)

    if not qualified or n_datasets < 2:
        loss = (h * 0.0).sum()
        return loss, {"L_xds_rank": 0.0, "n_xds_qualified": 0, "n_xds_terms": 0}

    for p in qualified:
        for d_a in range(n_datasets):
            for d_b in range(n_datasets):
                if d_a == d_b:
                    continue
                # cross-dataset, same primitive: anchor pair
                anchor_pos = (cent[(p, d_a)] * cent[(p, d_b)]).sum()
                # cross-dataset, different primitive: negative
                for q in qualified:
                    if q == p:
                        continue
                    if (q, d_b) not in cent:
                        continue
                    anchor_neg = (cent[(p, d_a)] * cent[(q, d_b)]).sum()
                    # want anchor_pos > anchor_neg + margin
                    valid_terms.append(F.relu(float(margin) + anchor_neg - anchor_pos))

    if not valid_terms:
        loss = (h * 0.0).sum()
    else:
        loss = torch.stack(valid_terms).mean()
    stats = {
        "L_xds_rank": float(loss.detach().item()),
        "n_xds_qualified": int(n_qualified_primitives),
        "n_xds_terms": int(len(valid_terms)),
    }
    return loss, stats


def primitive_separation_loss(
    h: torch.Tensor,                          # (B, P) projected
    primitives: torch.Tensor,                 # (B,) long
    *,
    n_primitives: int,
    min_count_per_primitive: int = 8,
    m_sep: float = 0.20,
) -> Tuple[torch.Tensor, dict]:
    """pairwise — prototype (primitive centroid) separation loss.

    For each pair of primitives p ≠ q with sufficient samples, penalize
    centroid cosine being above ``m_sep``:

        L = mean_{p ≠ q} relu(cos(c_p, c_q) - m_sep)^2

    Centroids are pooled over all datasets (un-conditional). Pairs with one
    end having < ``min_count_per_primitive`` samples are skipped.
    """
    cent: Dict[int, torch.Tensor] = {}
    for p in range(n_primitives):
        m = primitives == p
        if m.sum() < min_count_per_primitive:
            continue
        c = h[m].mean(dim=0)
        cent[p] = F.normalize(c, dim=-1, eps=1e-8)

    valid_terms = []
    qualified = sorted(cent.keys())
    for i, p in enumerate(qualified):
        for q in qualified[i + 1:]:
            cos_pq = (cent[p] * cent[q]).sum()
            valid_terms.append(F.relu(cos_pq - float(m_sep)) ** 2)

    if not valid_terms:
        loss = (h * 0.0).sum()
    else:
        loss = torch.stack(valid_terms).mean()
    stats = {
        "L_proto_sep": float(loss.detach().item()),
        "n_proto_pairs": int(len(valid_terms)),
        "n_proto_qualified": int(len(qualified)),
    }
    return loss, stats


def conditional_xds_centroid_loss(
    h: torch.Tensor,                       # (B, P) projected (post-head, post-norm) embeddings
    primitives: torch.Tensor,              # (B,) long
    datasets: torch.Tensor,                # (B,) long; 0=agibot 1=bridge etc.
    *,
    n_primitives: int,
    n_datasets: int = 2,
    min_count_per_cell: int = 4,
) -> Tuple[torch.Tensor, dict]:
    """pairwise — conditional cross-dataset centroid alignment.

    For each primitive p with ≥ ``min_count_per_cell`` samples in *both*
    datasets, compute the centroid in each dataset and pull them together:

        L_xds = mean_p [1 - cos(c_{p,AgiBot}, c_{p,Bridge})]

    Conditional on primitive (NOT a global domain-adversarial). Only primitives
    that pass the count gate contribute. Returns 0 (graph-bearing) when no
    primitives qualify on this batch — DDP-safe.
    """
    valid_terms = []
    n_qualified = 0
    n_skipped = 0

    for p in range(n_primitives):
        prim_mask = primitives == p
        if prim_mask.sum() < 2 * min_count_per_cell:
            n_skipped += 1
            continue
        # check both datasets have ≥ min_count_per_cell samples for this prim
        cell_counts = []
        cells = []
        for d in range(n_datasets):
            cell_mask = prim_mask & (datasets == d)
            cnt = int(cell_mask.sum().item())
            cell_counts.append(cnt)
            cells.append(cell_mask)
        if min(cell_counts) < min_count_per_cell:
            n_skipped += 1
            continue
        # both ds have enough; compute centroids
        centroids = []
        for cm in cells:
            c = h[cm].mean(dim=0)
            c = F.normalize(c, dim=-1, eps=1e-8)
            centroids.append(c)
        # pairwise: all cross-ds centroid pairs (typically just 1 pair when n_datasets=2)
        n_qualified += 1
        for i in range(n_datasets):
            for j in range(i + 1, n_datasets):
                cos_ij = (centroids[i] * centroids[j]).sum()
                valid_terms.append(1.0 - cos_ij)

    if not valid_terms:
        # graph-bearing zero (DDP-safe)
        loss = (h * 0.0).sum()
    else:
        loss = torch.stack(valid_terms).mean()

    stats = {
        "L_xds_centroid": float(loss.detach().item()),
        "n_xds_qualified_primitives": int(n_qualified),
        "n_xds_skipped_primitives": int(n_skipped),
    }
    return loss, stats


class MidLayerSigLIPHead(nn.Module):
    """pairwise — projection head for mid-layer (block 6) features.

    The mid-layer is per-token (B, N_tokens, D_block); we pool over tokens
    (mean) before projecting. Same hyperparameters as the final-z SigLIPHead but
    with an independent set of weights so the two heads can specialize.
    """

    def __init__(
        self,
        in_dim: int,                       # block 6 hidden dim (e.g. 384 / 768)
        proj_dim: int = 32,
        hidden_dim: Optional[int] = None,
        init_log_scale: float = 2.302585,
        init_bias: float = -3.0,
        pool: str = "mean",                # "mean" | "cls" if model has cls
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.in_dim
        self.pool = pool
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.proj_dim),
        )
        self.log_scale = nn.Parameter(torch.tensor(float(init_log_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` shape (B, N, D) or (B, D). Pooled then projected + L2 normalized."""
        if x.dim() == 3:
            if self.pool == "mean":
                x = x.mean(dim=1)
            else:
                x = x[:, 0]                # cls token
        h = self.proj(x)
        h = F.normalize(h, dim=-1, eps=1e-8)
        return h, self.log_scale.exp(), self.bias


__all__ = [
    "SigLIPHead", "OPPOSITE_PAIRS", "opposite_pair_id_table",
    "siglip_action_loss", "build_siglip_graph_from_meta",
    "build_structured_siglip_graph", "siglip_action_loss_split",
    "hard_positive_reweight",
    "conditional_xds_centroid_loss", "MidLayerSigLIPHead",
    # pairwise additions
    "build_neg_subgroup_masks", "grouped_action_contrastive_loss",
    "xds_ranking_centroid_loss", "primitive_separation_loss",
]
