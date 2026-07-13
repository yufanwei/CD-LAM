"""LAM masked trainer: fg/bg reconstruction + SigLIP action geometry.

masked keeps the pairwise SigLIP action head but replaces the older reconstruction
stack with RoboSAM fg/bg supervision:

  L_total = L_rec_fg + gamma_bg · L_rec_bg + beta_kl · KL_freebit
          + lambda_sig · L_sig + w_id · L_id

The masked production path adds ``partial_fullmix``:

  - train on the full pair index instead of only mask-eligible rows;
  - masked rows get fg/bg reconstruction;
  - unmasked rows get rho · L_rec_full;
  - L_sig, L_id, and KL are computed on all real rows;
  - L_id can use a margin form to avoid over-compressing identity pairs.

Reuses core/training/pairwise infrastructure:
  - ``cdlam_integration/tools/_cdlam_forward.py`` (encode_full / decode_full / forward_full)
  - ``cdlam_integration/tools/train_lam_action_readout.build_lam`` (init from baseline ckpt)
  - ``cdlam_integration/tools/distributed_helpers.{init_distributed, all_reduce_grads, write_run_state}``
  - ``cdlam_integration/tools/optimizer_helpers.{collect_trainable_parameter_groups, kl_loss_free_bits, _param_group_norms}``
  - ``cdlam_integration`` sampler and centered-action helpers
  - ``cdlam_integration`` SigLIP loss variants and dashboard/eval helpers

Phases:
  warmup      : [0, warmup_steps)         action loss off
  ramp        : [warmup_steps, ramp_end)  lambda_sig ramps to target
  main        : [ramp_end, total_steps)   lambda_sig may cosine-decay

Usage (preflight):
    torchrun --nproc_per_node=4 --master_port=29511 \
      cdlam_integration/tools/train_cdlam.py \
        --config cdlam_integration/configs/stage1.yaml \
        --out outputs/cdlam_train/stage1_preflight \
        --total-steps 200 --warmup-steps 50 --ramp-end 100 \
        --batch-real 16 --batch-id 4 --batch-mask 0 \
        --log-every 25 --ckpt-every 200 --eval-every 200

Usage (main):
    torchrun --nproc_per_node=4 --master_port=29511 \
      cdlam_integration/tools/train_cdlam.py \
        --config cdlam_integration/configs/stage1.yaml \
        --out outputs/cdlam_train/stage1_run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
from cdlam_integration.lam.model_ops import (  # noqa: E402
    encode_full,
    decode_full,
    forward_full,
)
from cdlam_integration.lam.distributed_helpers import (  # noqa: E402
    init_distributed,
    all_reduce_grads,
    write_run_state,
)
from cdlam_integration.lam.optimizer_helpers import (  # noqa: E402
    collect_trainable_parameter_groups,
    kl_loss_free_bits,
    _param_group_norms,
)

from cdlam_integration.lam.data import (  # noqa: E402
    PrimitiveBalancedSampler,
    MaskCache,
    decode_rows_parallel,
    load_interaction_masks_for_rows,
)
from cdlam_integration.lam.masks import (  # noqa: E402
    FgBgMaskCache,
    load_fg_bg_masks_for_rows,
)
from cdlam_integration.lam.reconstruction import (  # noqa: E402
    fg_weighted_reconstruction,
    bg_consistency_reconstruction,
)
from cdlam_integration.lam.latent_losses import (  # noqa: E402
    EMACenterer,
    centered_supcon_loss,
    build_supcon_masks_from_meta,
    triplet_hardneg_aux_loss,
    usage_gap_full,
    usage_gap_inter,
)
from cdlam_integration.lam.contrastive import (  # noqa: E402
    SigLIPHead,
    siglip_action_loss,
    build_siglip_graph_from_meta,
    opposite_pair_id_table,
)
from cdlam_integration.lam.contrastive_extensions import (  # noqa: E402
    build_structured_siglip_graph,
    siglip_action_loss_split,
    hard_positive_reweight,
    conditional_xds_centroid_loss,
    build_neg_subgroup_masks,
    grouped_action_contrastive_loss,
    xds_ranking_centroid_loss,
    primitive_separation_loss,
)
from cdlam_integration.lam.sampling import DatasetBalancedSampler  # noqa: E402


# =============== Init ckpt loader ===========================================


def load_init_ckpt(
    lam_inner: torch.nn.Module, ckpt_path: Path, device: str, log
) -> None:
    """Load a ckpt into ``lam_inner``. Handles two formats:
    * Stage-1/masked format: top-level "model" key, weights prefixed with "lam."
    * LAM_400k (paper released) format: pytorch-lightning "state_dict" key, "lam." prefix
    * Raw format: weights at top level, with or without "lam." prefix
    """
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        model_sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        model_sd = sd["state_dict"]
    else:
        model_sd = sd
    stripped = {}
    for k, v in model_sd.items():
        if k.startswith("lam."):
            stripped[k[4:]] = v
        else:
            stripped[k] = v
    missing, unexpected = lam_inner.load_state_dict(stripped, strict=False)
    step = sd.get("step") if isinstance(sd, dict) else None
    log(
        f"loaded init ckpt {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)} "
        f"(step={step})"
    )


# =============== Run state ===================================================


def make_state(args, world: int, rank: int, local_rank: int) -> dict:
    return {
        "status": "init",
        "phase": "preflight",
        "current_step": 0,
        "target_steps": args.total_steps,
        "latest_checkpoint": None,
        "latest_eval": None,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pid": os.getpid(),
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "pause_reason": None,
        "args": vars(args),
        "config_path": args.config,
        "trainer": "cdlam_stage1",
    }


# =============== Trainer ====================================================


def train(args, cfg) -> None:
    rank, world, local_rank = init_distributed()
    is_main = rank == 0

    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        (out_dir / "configs").mkdir(exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)
        shutil.copy2(args.config, out_dir / "configs" / "routeA_config.yaml.copy")
    if world > 1:
        dist.barrier()

    log_path = out_dir / "logs" / f"rank{rank}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg: str, all_ranks: bool = False) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}][r{rank}] {msg}"
        if is_main or all_ranks:
            print(line, flush=True)
        log_f.write(line + "\n")

    train_metrics_path = out_dir / "train_metrics.jsonl"
    eval_metrics_path = out_dir / "eval_metrics.jsonl"
    run_state_path = out_dir / "run_state.json"
    train_f = open(train_metrics_path, "a", buffering=1) if is_main else None
    eval_f = open(eval_metrics_path, "a", buffering=1) if is_main else None

    state = make_state(args, world, rank, local_rank)
    if is_main:
        write_run_state(run_state_path, state)

    device = f"cuda:{local_rank}" if world > 1 else cfg["trainer"]["device"]
    target_hw = (int(cfg["trainer"]["target_h"]), int(cfg["trainer"]["target_w"]))
    log(
        f"device={device}  pid={os.getpid()}  world={world}  out={out_dir}",
        all_ranks=True,
    )

    # ---- batch / schedule (CLI overrides config)
    batch_cfg = cfg["trainer"]["batch"]
    B_real = int(args.batch_real or batch_cfg["B_real"])
    B_id = int(args.batch_id or batch_cfg["B_id"])
    B_mask = int(
        args.batch_mask if args.batch_mask is not None else batch_cfg["B_mask"]
    )
    # earlier recipe: hard-neg pair injection. n_pairs=8 → last 16 rows of real_idx are
    # 8 (anchor, hardneg) pairs from multi-prim episodes. earlier recipe/earlier recipe/earlier recipe had this
    # implicitly = 0 and the hard-neg weighting code path was structurally
    # inactive (n_hard_neg_pairs=0 in train_metrics).
    B_hardneg_pairs = int(batch_cfg.get("B_hardneg_pairs", 0))
    decode_workers = int(batch_cfg["decode_workers"])
    use_ckpt = (
        False
        if args.no_grad_ckpt
        else True
        if args.grad_ckpt
        else bool(batch_cfg.get("use_grad_ckpt", True))
    )
    log(
        f"B_real={B_real} B_id={B_id} B_mask={B_mask} B_hardneg_pairs={B_hardneg_pairs} use_grad_ckpt={use_ckpt}"
    )

    sched = cfg["trainer"]["schedule"]
    total_steps = int(args.total_steps or sched["total_steps"])
    warmup_steps = int(
        args.warmup_steps if args.warmup_steps is not None else sched["warmup_steps"]
    )
    ramp_end = int(
        args.ramp_end if args.ramp_end is not None else sched["ramp_end_step"]
    )

    cad = cfg["trainer"]["cadence"]
    log_every = int(args.log_every or cad["log_every"])
    ckpt_every = int(args.ckpt_every or cad["ckpt_every"])
    eval_every = int(args.eval_every or cad["eval_every"])

    loss_cfg = cfg["trainer"]["loss"]
    beta_kl_init = float(loss_cfg.get("beta_kl_init", loss_cfg.get("beta_kl", 0.001)))
    beta_kl_target = float(
        loss_cfg.get("beta_kl_target", loss_cfg.get("beta_kl", 0.001))
    )
    beta_kl_ramp_end_step = int(loss_cfg.get("beta_kl_ramp_end_step", 0))
    beta_kl = beta_kl_init
    kl_free_bit = float(loss_cfg.get("kl_free_bit", 0.5))

    # ---- masked config (2026-05-09) — minimalist 4-loss path:
    # L_total = L_rec_fg + γ·L_rec_bg + β·KL + λ·L_sig
    # When masked_reconstruction_enabled is True, the trainer skips L_id, L_use_full, L_use_inter,
    # L_aux_triplet, L_rec_inter (pairwise mask block) and uses fg/bg-weighted
    # reconstruction instead. Sampler is restricted to mask_eligible rows so
    # every batch row has a valid fg+bg mask.
    masked_reconstruction_enabled = bool(loss_cfg.get("masked_reconstruction_enabled", False))
    foreground_reconstruction_weight = float(loss_cfg.get("foreground_reconstruction_weight", 1.0))  # core L_rec_fg coefficient
    background_consistency_weight = float(loss_cfg.get("background_consistency_weight", 0.1))  # γ — bg consistency weight
    min_foreground_pixels = int(loss_cfg.get("min_foreground_pixels", 64))
    min_background_pixels = int(loss_cfg.get("min_background_pixels", 64))
    # ---- masked (2026-05-09 16:30) — partial_fullmix:
    # Diagnostic 16:10 confirmed masked/5.1 train/eval distribution mismatch is the
    # main bottleneck (top1 +38%, xds +183% on mask-eligible val vs full val).

    # unmasked rows get rho·L_rec_full. L_sig+L_id+KL on all rows always.
    partial_full_mix_enabled = bool(loss_cfg.get("partial_full_mix_enabled", False))
    full_frame_reconstruction_weight = float(loss_cfg.get("full_frame_reconstruction_weight", 0.3))  # ρ on unmasked rows
    # ---- Margin L_id: only push z_id down when ratio > r (above threshold).
    # When w_id_margin > 0, L_id := relu(ratio - r)² instead of ratio².
    # Avoids over-compressing identity beyond what's needed.
    w_id_margin = float(loss_cfg.get("w_id_margin", 0.0))  # 0 = use legacy form

    lambda_rel_target = float(
        args.lambda_rel_target
        if args.lambda_rel_target is not None
        else loss_cfg["lambda_rel_target"]
    )
    w_id = float(loss_cfg["w_id"])
    lambda_use = float(loss_cfg["lambda_use"])
    alpha_inter = float(loss_cfg["alpha_inter"])
    eta_inter = float(loss_cfg["eta_inter"])
    margin_use_full = float(loss_cfg["margin_use_full"])
    margin_use_inter = float(loss_cfg["margin_use_inter"])
    supcon_temperature = float(loss_cfg.get("supcon_temperature", 0.07))
    min_label_confidence = float(loss_cfg.get("min_label_confidence", 0.5))
    # ---- hard_neg_weight schedule (earlier recipe): optionally decay weight from
    #      `init` to `target` across `[ramp_end_step, decay_end_step]`. earlier recipe
    #      observation (REPORT §5.5): fixed weight=3.0 helps gap during ramp
    #      (step 500 gap=+0.088) but gap collapses after step 1000 as
    #      class-clustering signal dominates and hard-neg becomes noise.
    #      Hypothesis: decay weight after ramp lets class-cluster take over
    #      cleanly while keeping the episode-disentanglement learned during ramp.
    hard_neg_weight_init = float(
        loss_cfg.get("hard_neg_weight_init", loss_cfg.get("hard_neg_weight", 1.0))
    )
    hard_neg_weight_target = float(
        loss_cfg.get("hard_neg_weight_target", hard_neg_weight_init)
    )
    hard_neg_weight_decay_end_step = int(
        loss_cfg.get("hard_neg_weight_decay_end_step", 0)
    )
    # earlier recipe (REPORT §6.5 Option B): triplet-style auxiliary on hard-neg pairs.
    # SupCon optimizes class log-prob, NOT the cos(a,p)-cos(a,n) margin. baseline
    # used triplet hinge directly and got gap=+0.065. earlier recipe keeps SupCon as main
    # action signal (top1 win) but adds small triplet aux on (anchor, pos, hard-neg)
    # to put gap back as a first-order target.
    # 0 = disabled (earlier recipe/earlier recipe/earlier recipe/earlier recipe behaviour). 0.05-0.20 reasonable.
    lambda_aux_triplet = float(loss_cfg.get("lambda_aux_triplet_target", 0.0))
    aux_triplet_margin = float(loss_cfg.get("aux_triplet_margin", 0.10))
    l_use_every = int(loss_cfg.get("l_use_every", 2))
    grad_clip = float(cfg["trainer"]["lr"].get("grad_clip_max_norm", 1.0))

    # ---- pairwise A1: action_loss_kind switch (siglip / supcon / none) ----
    action_loss_kind = str(loss_cfg.get("action_loss_kind", "supcon")).lower()
    if action_loss_kind not in {"siglip", "supcon", "none"}:
        raise ValueError(
            f"action_loss_kind must be siglip|supcon|none, got {action_loss_kind!r}"
        )
    lambda_sig_target = float(loss_cfg.get("lambda_sig_target", 0.05))
    # ---- pairwise (2026-05-08): lambda_sig cosine decay in main phase ----
    # pairwise A1 ablation showed eff_rank kept dropping (27.07 → 25.24) under a
    # constant lambda_sig in main phase. Decay from `lambda_sig_target` at
    # ramp_end to `lambda_sig_target_final` at `lambda_sig_decay_end_step`
    # (cosine schedule). When end <= ramp_end, behaves as flat (pairwise A1 baseline).
    lambda_sig_target_final = float(
        loss_cfg.get("lambda_sig_target_final", lambda_sig_target)
    )
    lambda_sig_decay_end_step = int(loss_cfg.get("lambda_sig_decay_end_step", 0))
    siglip_proj_dim = int(loss_cfg.get("siglip_proj_dim", 32))
    # ---- pairwise: head capacity + per-head LR (was hard-coded to G_head LR) ----
    siglip_hidden_dim = int(loss_cfg.get("siglip_hidden_dim", siglip_proj_dim))
    siglip_head_lr = float(
        loss_cfg.get("siglip_head_lr", float(cfg["trainer"]["lr"]["G_head"]))
    )
    siglip_init_log_scale = float(loss_cfg.get("siglip_init_log_scale", 2.302585))
    siglip_init_bias = float(loss_cfg.get("siglip_init_bias", -10.0))
    siglip_w_opposite = float(loss_cfg.get("siglip_w_opposite", 2.0))
    siglip_w_hard_neg = float(loss_cfg.get("siglip_w_hard_neg", 1.5))
    siglip_min_confidence = float(
        loss_cfg.get("siglip_min_label_confidence", min_label_confidence)
    )

    # ---- pairwise (2026-05-08) extensions: cross-dataset positive boost +
    # neg combination mode (multiplicative vs max) + neg cap + sampler dataset
    # balance + L_pos / L_neg split schedule. Defaults reproduce pairwise.
    structured_graph_enabled = bool(loss_cfg.get("structured_graph_enabled", False))
    siglip_w_pos_cross_dataset = float(loss_cfg.get("siglip_w_pos_cross_dataset", 1.0))
    siglip_neg_weight_mode = str(
        loss_cfg.get("siglip_neg_weight_mode", "multiplicative")
    )
    siglip_w_neg_cap = float(loss_cfg.get("siglip_w_neg_cap", float("inf")))
    # split-loss path: when True, use siglip_action_loss_split with separate
    # lambda_pos / lambda_neg schedules. Both ramp 0→target then cosine decay
    # to target_final, mirroring lambda_sig schedule but per-side.
    siglip_action_loss_split_enabled = bool(
        loss_cfg.get("siglip_action_loss_split", False)
    )
    siglip_lambda_pos_target = float(
        loss_cfg.get("siglip_lambda_pos_target", lambda_sig_target)
    )
    siglip_lambda_pos_target_final = float(
        loss_cfg.get("siglip_lambda_pos_target_final", siglip_lambda_pos_target)
    )
    siglip_lambda_neg_target = float(
        loss_cfg.get("siglip_lambda_neg_target", lambda_sig_target)
    )
    siglip_lambda_neg_target_final = float(
        loss_cfg.get("siglip_lambda_neg_target_final", siglip_lambda_neg_target)
    )
    # sampler choice (A1 default; A2 = primitive×dataset balanced)
    sampler_dataset_balanced = bool(
        cfg["trainer"]["sampler"].get("dataset_balanced", False)
    )
    sampler_min_xds_per_prim = int(
        cfg["trainer"]["sampler"].get("min_cross_dataset_per_primitive", 1)
    )
    # ---- pairwise: hard-positive mining (high-conf cos<τ_pos boost) ----
    siglip_hard_pos_enabled = bool(loss_cfg.get("siglip_hard_pos_enabled", False))
    siglip_hard_pos_tau = float(loss_cfg.get("siglip_hard_pos_tau", 0.20))
    siglip_hard_pos_gamma = float(loss_cfg.get("siglip_hard_pos_gamma", 2.0))
    siglip_hard_pos_high_conf_min = float(
        loss_cfg.get("siglip_hard_pos_high_conf_min", 0.8)
    )
    siglip_hard_pos_max_weight = float(loss_cfg.get("siglip_hard_pos_max_weight", 3.0))
    # ---- pairwise: conditional cross-dataset centroid loss ----
    siglip_xds_centroid_enabled = bool(
        loss_cfg.get("siglip_xds_centroid_enabled", False)
    )
    siglip_xds_centroid_lambda = float(loss_cfg.get("siglip_xds_centroid_lambda", 0.01))
    siglip_xds_min_count_per_cell = int(
        loss_cfg.get("siglip_xds_min_count_per_cell", 4)
    )
    # ---- pairwise: structured negatives + ranking xds + prototype separation ----
    # When `grouped_action_loss_enabled` = True, the SigLIP loss uses
    # grouped_action_contrastive_loss (5 separate λ for opp/hard/rand neg + same/xds pos)
    # instead of siglip_action_loss_split. Disjoint subgroups: opposite primitive
    # > same-episode hard > random diff-prim. Each subgroup has its own λ schedule
    # AND can have a different neg cap (`siglip_w_neg_cap_opposite` defaults to 3.0
    # to preserve direction signal even when global cap is 2.0).
    grouped_action_loss_enabled = bool(
        loss_cfg.get("grouped_action_loss_enabled", False)
    )
    siglip_w_neg_cap_opposite = float(loss_cfg.get("siglip_w_neg_cap_opposite", 3.0))
    # 5 λ targets + finals (cosine decay over [ramp_end, lambda_sig_decay_end_step])
    siglip_lambda_pos_same_target = float(
        loss_cfg.get("siglip_lambda_pos_same_target", 0.006)
    )
    siglip_lambda_pos_same_target_final = float(
        loss_cfg.get(
            "siglip_lambda_pos_same_target_final", siglip_lambda_pos_same_target
        )
    )
    siglip_lambda_pos_xds_target = float(
        loss_cfg.get("siglip_lambda_pos_xds_target", 0.010)
    )
    siglip_lambda_pos_xds_target_final = float(
        loss_cfg.get("siglip_lambda_pos_xds_target_final", siglip_lambda_pos_xds_target)
    )
    siglip_lambda_neg_opp_target = float(
        loss_cfg.get("siglip_lambda_neg_opp_target", 0.012)
    )
    siglip_lambda_neg_opp_target_final = float(
        loss_cfg.get("siglip_lambda_neg_opp_target_final", siglip_lambda_neg_opp_target)
    )
    siglip_lambda_neg_hard_target = float(
        loss_cfg.get("siglip_lambda_neg_hard_target", 0.008)
    )
    siglip_lambda_neg_hard_target_final = float(
        loss_cfg.get(
            "siglip_lambda_neg_hard_target_final", siglip_lambda_neg_hard_target
        )
    )
    siglip_lambda_neg_rand_target = float(
        loss_cfg.get("siglip_lambda_neg_rand_target", 0.003)
    )
    siglip_lambda_neg_rand_target_final = float(
        loss_cfg.get(
            "siglip_lambda_neg_rand_target_final", siglip_lambda_neg_rand_target
        )
    )
    # ranking xds (replaces pairwise raw centroid pull)
    siglip_xds_rank_enabled = bool(loss_cfg.get("siglip_xds_rank_enabled", False))
    siglip_xds_rank_lambda = float(loss_cfg.get("siglip_xds_rank_lambda", 0.005))
    siglip_xds_rank_margin = float(loss_cfg.get("siglip_xds_rank_margin", 0.05))
    siglip_xds_rank_min_count = int(loss_cfg.get("siglip_xds_rank_min_count", 4))
    # prototype separation
    siglip_proto_sep_enabled = bool(loss_cfg.get("siglip_proto_sep_enabled", False))
    siglip_proto_sep_lambda = float(loss_cfg.get("siglip_proto_sep_lambda", 0.003))
    siglip_proto_sep_margin = float(loss_cfg.get("siglip_proto_sep_margin", 0.20))
    siglip_proto_sep_min_count = int(loss_cfg.get("siglip_proto_sep_min_count", 8))
    log(
        f"steps={total_steps} warmup={warmup_steps} ramp_end={ramp_end} "
        f"lambda_rel_target={lambda_rel_target} lambda_use={lambda_use} w_id={w_id} "
        f"alpha_inter={alpha_inter} eta_inter={eta_inter} "
        f"beta_kl_init={beta_kl_init} beta_kl_target={beta_kl_target} kl_free_bit={kl_free_bit}"
    )
    log(
        f"action_loss_kind={action_loss_kind} "
        f"lambda_sig_target={lambda_sig_target} lambda_sig_target_final={lambda_sig_target_final} "
        f"lambda_sig_decay_end_step={lambda_sig_decay_end_step} "
        f"siglip_proj_dim={siglip_proj_dim} siglip_hidden_dim={siglip_hidden_dim} "
        f"siglip_head_lr={siglip_head_lr:.2e} "
        f"siglip_init_log_scale={siglip_init_log_scale:.4f} siglip_init_bias={siglip_init_bias:.2f} "
        f"siglip_w_opposite={siglip_w_opposite} siglip_w_hard_neg={siglip_w_hard_neg}"
    )

    # ---- baseline metrics (from Stage-1's baseline metrics json — same architecture, comparable)
    m0_path = Path(args.m0_baseline) if args.m0_baseline else None
    m0 = json.loads(m0_path.read_text()) if (m0_path and m0_path.exists()) else None
    if m0:
        log(
            f"baseline metrics z_norm={m0['z_geometry']['z_mu_norm_mean']:.3f} "
            f"eff_rank={m0['z_geometry']['effective_rank']:.2f}"
        )
    stop_cfg = cfg["trainer"]["stop"]
    m0_z_norm = m0["z_geometry"]["z_mu_norm_mean"] if m0 else None
    m0_eff_rank = m0["z_geometry"]["effective_rank"] if m0 else None

    # ---- sampler
    canonical = cfg["trainer"]["sampler"]["canonical_primitives"]
    pair_index_train = Path(cfg["trainer"]["pair_index_train"])
    if sampler_dataset_balanced:
        log(
            f"sampler: DatasetBalancedSampler (primitive×dataset balanced, min_xds_per_prim={sampler_min_xds_per_prim})"
        )
        sampler = DatasetBalancedSampler(
            pair_index_path=pair_index_train,
            canonical_primitives=canonical,
            p_neg_same_episode=cfg["trainer"]["sampler"]["p_negative_same_episode"],
            seed=args.seed + 1009 * rank,
            dataset_balanced=True,
            min_cross_dataset_per_primitive=sampler_min_xds_per_prim,
        )
    else:
        sampler = PrimitiveBalancedSampler(
            pair_index_path=pair_index_train,
            canonical_primitives=canonical,
            p_neg_same_episode=cfg["trainer"]["sampler"]["p_negative_same_episode"],
            seed=args.seed + 1009 * rank,
        )
    log(
        f"sampler real={len(sampler.real):,} id={len(sampler.id):,} "
        f"mask_eligible={len(sampler.mask_idx_in_real):,} "
        f"primitives={sampler.primitives}"
    )

    # primitive name → int id (for SupCon)
    prim_to_id = {p: i for i, p in enumerate(canonical)}

    # ---- mask cache (pairwise path) + fg+bg cache (masked path)
    mask_cache = MaskCache(
        max_entries=int(cfg["trainer"]["sampler"].get("mask_cache_size", 256))
    )
    fgbg_cache = FgBgMaskCache(
        max_entries=int(cfg["trainer"]["sampler"].get("mask_cache_size", 256))
    )

    # ---- masked: when masked_reconstruction_enabled AND NOT partial_full_mix_enabled, restrict sampler.real to mask_eligible.
    # masked partial_fullmix path keeps full pair_index — fg/bg loss only on rows where mask
    # is valid; unmasked rows get rho·L_rec_full instead.
    if masked_reconstruction_enabled and not partial_full_mix_enabled:
        eligible = (
            sampler.real["robosam_mask_training_eligible"].fillna(False).astype(bool)
        )
        if eligible.sum() < 1000:
            raise RuntimeError(
                f"masked_reconstruction_enabled requires sampler.real with mask_eligible rows; got "
                f"only {int(eligible.sum())} eligible rows."
            )
        # rebuild internal indices in-place on the filtered DataFrame
        from collections import defaultdict
        from cdlam_integration.lam.data import _ep_key

        sampler.real = sampler.real[eligible].reset_index(drop=True)
        sampler.by_prim_ep = defaultdict(lambda: defaultdict(list))
        sampler.by_prim = defaultdict(list)
        sampler.by_episode = defaultdict(list)
        for idx, row in enumerate(sampler.real.itertuples(index=False)):
            ep = _ep_key(row)
            sampler.by_prim_ep[row.primitive][ep].append(idx)
            sampler.by_prim[row.primitive].append(idx)
            sampler.by_episode[ep].append((row.primitive, idx))
        sampler.primitives = [
            p
            for p in sampler.canonical
            if p in sampler.by_prim_ep and len(sampler.by_prim_ep[p]) >= 2
        ]
        sampler.hard_neg_episodes = set()
        for ep, rows in sampler.by_episode.items():
            if len({p for (p, _) in rows}) >= 2:
                sampler.hard_neg_episodes.add(ep)
        # sampler.mask_idx_in_real / sampler.mask are now the same set as sampler.real
        sampler.mask = sampler.real.copy()
        sampler.mask_idx_in_real = np.arange(len(sampler.mask))
        if hasattr(sampler, "by_prim_ds"):
            # rebuild DatasetBalancedSampler additional indices
            sampler.by_prim_ds = defaultdict(lambda: defaultdict(list))
            sampler.by_prim_ds_ep = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
            for idx, row in enumerate(sampler.real.itertuples(index=False)):
                ds = row.dataset
                ep = _ep_key(row)
                sampler.by_prim_ds[row.primitive][ds].append(idx)
                sampler.by_prim_ds_ep[row.primitive][ds][ep].append(idx)
            sampler.cross_dataset_primitives = sorted(
                [
                    p
                    for p, by_ds in sampler.by_prim_ds.items()
                    if len(by_ds) >= 2 and all(len(v) > 0 for v in by_ds.values())
                ]
            )
        log(
            f"masked_reconstruction_enabled: sampler.real restricted to mask_eligible — "
            f"{len(sampler.real):,} rows, {len(sampler.primitives)} primitives, "
            f"{len(sampler.hard_neg_episodes)} hard-neg episodes"
        )

    # ---- build trainable LAM
    log("building trainable training LAM (init from baseline → optionally overlay --init-ckpt) ...")
    lam_train = build_lam("CD_LAM", device=device)
    lam_inner = lam_train.lam
    init_ckpt_path = args.init_ckpt or cfg["trainer"].get("init_ckpt")
    if init_ckpt_path:
        load_init_ckpt(lam_inner, Path(init_ckpt_path), device, log)
    lam_inner.train()

    # ---- pairwise A1: SigLIP head (only when action_loss_kind == "siglip") ----
    siglip_head: Optional[SigLIPHead] = None
    opp_table_ids: Dict[int, int] = {}
    if action_loss_kind == "siglip":
        siglip_head = SigLIPHead(
            in_dim=int(lam_inner.latent_dim),
            proj_dim=siglip_proj_dim,
            hidden_dim=siglip_hidden_dim,
            init_log_scale=siglip_init_log_scale,
            init_bias=siglip_init_bias,
        ).to(device)
        siglip_head.train()
        opp_table_ids = opposite_pair_id_table(canonical)
        head_n_params = sum(p.numel() for p in siglip_head.parameters())
        log(
            f"SigLIPHead: in={lam_inner.latent_dim} proj={siglip_proj_dim} hidden={siglip_hidden_dim} "
            f"lr={siglip_head_lr:.2e} params={head_n_params:,} opposite_table_size={len(opp_table_ids)}"
        )

    # ---- param groups (5-way LAM partition; siglip head as 6th group)
    param_groups = collect_trainable_parameter_groups(lam_inner, cfg["trainer"]["lr"])
    if siglip_head is not None:
        # pairwise: dedicated head LR (was hard-coded to G_head LR in pairwise A1).
        # Higher head LR lets bias/scale/proj layers adapt faster than the
        # huge 700M-param LAM backbone.
        param_groups.append(
            {
                "name": "G_siglip_head",
                "params": list(siglip_head.parameters()),
                "lr": siglip_head_lr,
            }
        )
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    n_total = sum(p.numel() for p in lam_inner.parameters()) + (
        sum(p.numel() for p in siglip_head.parameters())
        if siglip_head is not None
        else 0
    )
    log(f"trainable {n_train:,} / {n_total:,} = {n_train / n_total * 100:.1f}%")
    for g in param_groups:
        log(
            f"  {g['name']:>16}: lr={g['lr']:.2e}  n_params={sum(p.numel() for p in g['params']):,}"
        )

    # ---- optimizer
    optim = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["trainer"]["lr"]["weight_decay"]),
        eps=float(cfg["trainer"]["lr"]["eps"]),
    )
    last_param_snapshots = {
        g["name"]: [p.detach().clone() for p in g["params"]] for g in param_groups
    }
    init_param_norms = {
        g["name"]: _param_group_norms(g["params"])[0] for g in param_groups
    }
    _ = (
        last_param_snapshots,
        init_param_norms,
    )  # logging closures will rebuild as needed

    # ---- EMA center for SupCon
    centerer = EMACenterer(
        dim=int(lam_inner.latent_dim),
        alpha=float(loss_cfg.get("center_ema_alpha", 0.95)),
        device=device,
    )

    # ---- L_id EMA scale
    s_ema: Optional[torch.Tensor] = None
    s_ema_alpha = 0.95

    rec_history: deque = deque(maxlen=200)

    state["status"] = "warmup" if warmup_steps > 0 else "ramp"
    state["phase"] = (
        "warmup" if warmup_steps > 0 else "ramp" if total_steps > 1 else "main"
    )
    if is_main:
        write_run_state(run_state_path, state)

    # ---- prefetch helper (overlap decode with previous step's compute)
    def _build_one_batch():
        # In masked_reconstruction_enabled: B_mask=0 always (fg/bg loaded per real row instead).
        # B_id is honored even in masked_reconstruction_enabled (masked adds L_id back; masked had id_p50=0.564
        # collapse w/o L_id). Set B_id=0 in yaml to disable.
        eff_B_id = B_id
        eff_B_mask = 0 if masked_reconstruction_enabled else B_mask
        ri, ii, _tri_unused, mi, m = sampler.sample(
            B_real,
            eff_B_id,
            0,
            eff_B_mask,
            B_hardneg_pairs=B_hardneg_pairs,
        )
        rr = sampler.real.iloc[ri]
        ir = (
            sampler.id.iloc[ii]
            if len(sampler.id) > 0 and eff_B_id > 0
            else sampler.real.iloc[[]]
        )
        mr = sampler.real.iloc[mi] if mi else sampler.real.iloc[[]]
        rp, rv = decode_rows_parallel(rr, target_hw, decode_workers)
        if len(ir) > 0:
            ip, iv = decode_rows_parallel(ir, target_hw, decode_workers)
        else:
            ip = np.zeros((0, 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
            iv = np.zeros((0,), dtype=bool)
        if len(mr) > 0:
            mp, mv = decode_rows_parallel(mr, target_hw, decode_workers)
            mask_arr, mask_valid = load_interaction_masks_for_rows(
                mr, mask_cache, target_hw
            )
        else:
            mp = np.zeros((0, 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
            mv = np.zeros((0,), dtype=bool)
            mask_arr = np.zeros((0, target_hw[0], target_hw[1]), dtype=np.float32)
            mask_valid = np.zeros((0,), dtype=bool)
        # masked: load fg + bg masks for the REAL block (sampler.real is already
        # filtered to mask_eligible in masked_reconstruction_enabled init).
        if masked_reconstruction_enabled:
            fg_real, bg_real, fgbg_valid_real = load_fg_bg_masks_for_rows(
                rr,
                fgbg_cache,
                target_hw,
            )
        else:
            fg_real = np.zeros((len(rr), target_hw[0], target_hw[1]), dtype=np.float32)
            bg_real = np.zeros((len(rr), target_hw[0], target_hw[1]), dtype=np.float32)
            fgbg_valid_real = np.zeros((len(rr),), dtype=bool)
        return (
            ri,
            ii,
            mi,
            m,
            rp,
            rv,
            ip,
            iv,
            mp,
            mv,
            mask_arr,
            mask_valid,
            fg_real,
            bg_real,
            fgbg_valid_real,
        )

    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    prefetch_fut = prefetch_pool.submit(_build_one_batch)

    # ===== training loop =====
    step = 0
    pause_reason: Optional[str] = None
    t_loop_start = time.time()
    pre_norms: Dict[str, list] = {}
    try:
        while step < total_steps:
            t_step_start = time.time()
            t_data_start = time.time()

            (
                real_idx,
                id_idx,
                mask_idx,
                meta,
                real_pairs,
                real_valid,
                id_pairs,
                id_valid,
                mask_pairs,
                mask_valid_decode,
                mask_arr,
                mask_valid_load,
                fg_real,
                bg_real,
                fgbg_valid_real,
            ) = prefetch_fut.result()
            prefetch_fut = prefetch_pool.submit(_build_one_batch)

            # filter invalid decodes
            if not real_valid.all():
                real_pairs = real_pairs[real_valid]
                real_idx = [real_idx[i] for i, v in enumerate(real_valid) if v]
                if masked_reconstruction_enabled:
                    fg_real = fg_real[real_valid]
                    bg_real = bg_real[real_valid]
                    fgbg_valid_real = fgbg_valid_real[real_valid]
            if len(id_pairs) > 0 and not id_valid.all():
                id_pairs = id_pairs[id_valid]
                id_idx = [id_idx[i] for i, v in enumerate(id_valid) if v]
            if len(mask_pairs) > 0:
                # require BOTH frame decode AND mask load
                m_keep = mask_valid_decode & mask_valid_load
                mask_pairs = mask_pairs[m_keep]
                mask_arr = mask_arr[m_keep]
                mask_idx = [mask_idx[i] for i, v in enumerate(m_keep) if v]

            if len(real_pairs) == 0:
                log(f"step {step}: 0 valid real after decode, skipping")
                step += 1
                continue

            t_data = time.time() - t_data_start

            v_real = torch.from_numpy(real_pairs).float().to(device) / 255.0
            v_id = (
                torch.from_numpy(id_pairs).float().to(device) / 255.0
                if len(id_pairs)
                else None
            )
            v_mask = (
                torch.from_numpy(mask_pairs).float().to(device) / 255.0
                if len(mask_pairs)
                else None
            )
            mask_t = (
                torch.from_numpy(mask_arr).float().to(device)
                if len(mask_pairs)
                else None
            )
            # masked fg/bg masks for the REAL block
            if masked_reconstruction_enabled:
                fg_t = torch.from_numpy(fg_real).float().to(device)
                bg_t = torch.from_numpy(bg_real).float().to(device)
                fgbg_v_t = torch.from_numpy(fgbg_valid_real).bool().to(device)
            else:
                fg_t = None
                bg_t = None
                fgbg_v_t = None

            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

            t_compute_start = time.time()

            # NOTE: encoder is the dominant memory sink (24 self-attn blocks,
            # activation peak ≈ B × H × W × D × n_blocks). Concatenating real
            # and mask into a single forward pass OOMs at production batch on
            # H100 80GB even with grad ckpt; we keep them as TWO encoder passes.
            # The small block (mask, B_m) is cheap. Decoder shuffle/zero passes
            # are concat'd later (much smaller memory budget) for the L_use win.
            B_r = v_real.shape[0]
            B_m = v_mask.shape[0] if v_mask is not None else 0

            # ---- forward real (encoder + decoder + reparam)
            with amp_ctx:
                out_real = forward_full(
                    lam_inner, v_real, sample=True, use_ckpt=use_ckpt
                )
            recon_real = out_real["recon"].float()
            gt_real = v_real[:, 1:].float()
            mse_real_full = ((gt_real - recon_real) ** 2).mean()
            kl_loss, kl_stats = kl_loss_free_bits(
                out_real["z_mu"], out_real["z_var"], free_bit=kl_free_bit
            )
            # current beta_kl by schedule
            if beta_kl_ramp_end_step > 0 and step < beta_kl_ramp_end_step:
                _frac = step / max(1, beta_kl_ramp_end_step)
                beta_kl = beta_kl_init + (beta_kl_target - beta_kl_init) * _frac
            else:
                beta_kl = beta_kl_target
            z_mu_real = out_real["z_mu"].float()
            patches_real = out_real["patches"]
            z_rep_real = out_real["z_rep"]

            # update EMA center BEFORE SupCon so this step uses up-to-date c
            centerer.update(z_mu_real)

            # ---- forward mask block (encoder + decoder; needed for L_rec_inter and L_use_inter)
            recon_mask: Optional[torch.Tensor] = None
            gt_mask: Optional[torch.Tensor] = None
            patches_mask = None
            z_rep_mask = None
            L_rec_inter = torch.zeros((), device=device, dtype=torch.float32)
            n_inter_rows = 0
            if B_m > 0:
                with amp_ctx:
                    out_mask = forward_full(
                        lam_inner, v_mask, sample=True, use_ckpt=use_ckpt
                    )
                recon_mask = out_mask["recon"].float()
                gt_mask = v_mask[:, 1:].float()
                patches_mask = out_mask["patches"]
                z_rep_mask = out_mask["z_rep"]
                if mask_t is not None and mask_t.shape[0] == B_m:
                    # per-row interaction MSE: sum_x M(x) * mse(x) / clamp(sum M, min_pixels)
                    sq_pix = ((recon_mask - gt_mask) ** 2).mean(
                        dim=(1, -1)
                    )  # (Bm, H, W)
                    M = mask_t.float()
                    min_pix = float(loss_cfg.get("mask_min_pixels", 64))
                    area = M.sum(dim=(-1, -2)).clamp_min(min_pix)
                    L_rec_inter = ((sq_pix * M).sum(dim=(-1, -2)) / area).mean()
                    n_inter_rows = int(B_m)

            L_rec_full = mse_real_full
            # ---- masked: fg-weighted reconstruction (replaces L_rec_full + α·L_rec_inter)
            L_rec_fg_t = torch.zeros((), device=device, dtype=torch.float32)
            L_rec_bg_t = torch.zeros((), device=device, dtype=torch.float32)
            reconstruction_stats: Dict = {}
            if masked_reconstruction_enabled and fg_t is not None:
                # gt_real shape: (B, T-1, H, W, C); recon_real same
                # fg_t shape: (B, H, W); fgbg_v_t shape: (B,)
                L_rec_fg_t, fg_stats = fg_weighted_reconstruction(
                    pred=recon_real,
                    gt_next=gt_real,
                    fg_mask=fg_t,
                    valid=fgbg_v_t,
                    min_pixels=min_foreground_pixels,
                )
                # bg consistency: target is the PREVIOUS frame v_real[:, 0]
                gt_prev = v_real[:, 0].float()  # (B, H, W, C)
                L_rec_bg_t, bg_stats = bg_consistency_reconstruction(
                    pred=recon_real,
                    gt_prev=gt_prev,
                    bg_mask=bg_t,
                    valid=fgbg_v_t,
                    min_pixels=min_background_pixels,
                )
                reconstruction_stats.update(fg_stats)
                reconstruction_stats.update(bg_stats)

            # ---- masked partial_fullmix: rho·L_rec_full on rows where mask is invalid.
            # When fgbg_v_t has False entries (rows without mask), apply L_rec_full to
            # those rows as a soft signal — keeps decoder learning on unmasked data.
            L_rec_full_unmasked_t = torch.zeros((), device=device, dtype=torch.float32)
            if partial_full_mix_enabled and fgbg_v_t is not None and fg_t is not None:
                unmasked = (~fgbg_v_t).float()  # (B,)
                n_unmasked = unmasked.sum().clamp_min(1.0)
                # per-row mean MSE over (T-1, H, W, C)
                sq_full = ((recon_real - gt_real) ** 2).float()
                # collapse all non-batch dims
                per_row = sq_full.reshape(sq_full.shape[0], -1).mean(dim=1)  # (B,)
                L_rec_full_unmasked_t = (per_row * unmasked).sum() / n_unmasked
            if masked_reconstruction_enabled:
                L_gen = (
                    foreground_reconstruction_weight * L_rec_fg_t
                    + background_consistency_weight * L_rec_bg_t
                    + full_frame_reconstruction_weight * L_rec_full_unmasked_t
                    + beta_kl * kl_loss
                )
            else:
                L_gen = L_rec_full + alpha_inter * L_rec_inter + beta_kl * kl_loss

            # ---- L_id (no reparam, no decoder).
            # masked had B_id=0 → L_id off → id_p50=0.564 collapse @ step 500.
            # masked honors B_id even in masked_reconstruction_enabled (id forward computed if B_id>0).
            if v_id is not None and v_id.shape[0] > 0:
                with amp_ctx:
                    out_id = encode_full(
                        lam_inner, v_id, sample=False, use_ckpt=use_ckpt
                    )
                with torch.no_grad():
                    z_real_norm_sq = z_mu_real.detach().pow(2).sum(dim=1).mean()
                    if s_ema is None:
                        s_ema = z_real_norm_sq.clone()
                    else:
                        s_ema = s_ema_alpha * s_ema + (1 - s_ema_alpha) * z_real_norm_sq
                # Legacy form: ratio_sq = ||z_id||² / E[||z_real||²], penalize quadratically.
                # Margin form (w_id_margin > 0): only penalize when ratio_norm > w_id_margin
                # (the radius r). This avoids over-compressing identity beyond a healthy
                # band and is the form recommended in masked (avoid pushing id_p50 below
                # ~0.05 unnecessarily — see REPORT §6.5b).
                ratio_sq = out_id["z_mu"].float().pow(2).sum(dim=1).mean() / (
                    s_ema + 1e-8
                )
                if w_id_margin > 0.0:
                    ratio_norm = torch.sqrt(ratio_sq.clamp_min(1e-12))
                    L_id = torch.relu(ratio_norm - w_id_margin).pow(2)
                else:
                    L_id = ratio_sq
            else:
                L_id = torch.zeros((), device=device, dtype=torch.float32)

            # ---- L_use (zero z + shuffled z) on real every l_use_every steps; mask every step
            # masked_reconstruction_enabled: L_use disabled (fg-weighted recon already enforces z-use)
            do_l_use = (step % l_use_every == 0) and not masked_reconstruction_enabled
            L_use_full_val = float("nan")
            L_use_inter_val = float("nan")
            usage_gap_zero_full = float("nan")
            usage_gap_shuffle_full = float("nan")
            usage_gap_zero_inter = float("nan")
            usage_gap_shuffle_inter = float("nan")
            if do_l_use:

                def _no_fixed(rng_size: int) -> torch.Tensor:
                    if rng_size < 2:
                        return torch.zeros(rng_size, dtype=torch.long, device=device)
                    p = torch.randperm(rng_size, device=device)
                    while torch.any(p == torch.arange(rng_size, device=device)):
                        p = torch.randperm(rng_size, device=device)
                    return p

                # ---- real block: zero + shuffle decode (L_use_full)
                perm_r = _no_fixed(B_r)
                z_rep_real_zero = torch.zeros_like(z_rep_real)
                z_rep_real_shuf = z_rep_real.index_select(0, perm_r).detach()
                with amp_ctx:
                    H, W = v_real.shape[2:4]
                    recon_real_zero = decode_full(
                        lam_inner,
                        patches_real,
                        z_rep_real_zero,
                        H,
                        W,
                        use_ckpt=use_ckpt,
                    )
                    recon_real_shuf = decode_full(
                        lam_inner,
                        patches_real,
                        z_rep_real_shuf,
                        H,
                        W,
                        use_ckpt=use_ckpt,
                    )
                L_use_full_t, use_real_stats = usage_gap_full(
                    pred_real=recon_real,
                    pred_zero=recon_real_zero,
                    pred_shuf=recon_real_shuf,
                    gt=gt_real,
                    margin_full=margin_use_full,
                )
                L_use_full_val = use_real_stats["L_use_full"]
                usage_gap_zero_full = use_real_stats["usage_gap_zero_full"]
                usage_gap_shuffle_full = use_real_stats["usage_gap_shuffle_full"]
                # free real-side decoder outputs we no longer need
                del recon_real_zero, recon_real_shuf, z_rep_real_zero, z_rep_real_shuf

                # ---- mask block: zero + shuffle decode (L_use_inter)
                if (
                    B_m > 0
                    and recon_mask is not None
                    and mask_t is not None
                    and mask_t.shape[0] > 0
                ):
                    perm_m = _no_fixed(B_m)
                    z_rep_mask_zero = torch.zeros_like(z_rep_mask)
                    z_rep_mask_shuf = z_rep_mask.index_select(0, perm_m).detach()
                    with amp_ctx:
                        Hm, Wm = v_mask.shape[2:4]
                        recon_mask_zero = decode_full(
                            lam_inner,
                            patches_mask,
                            z_rep_mask_zero,
                            Hm,
                            Wm,
                            use_ckpt=use_ckpt,
                        )
                        recon_mask_shuf = decode_full(
                            lam_inner,
                            patches_mask,
                            z_rep_mask_shuf,
                            Hm,
                            Wm,
                            use_ckpt=use_ckpt,
                        )
                    L_use_inter_t, use_mask_stats = usage_gap_inter(
                        pred_real=recon_mask,
                        pred_zero=recon_mask_zero,
                        pred_shuf=recon_mask_shuf,
                        gt=gt_mask,
                        mask_inter=mask_t,
                        margin_inter=margin_use_inter,
                        min_pixels=int(loss_cfg.get("mask_min_pixels", 64)),
                    )
                    L_use_inter_val = use_mask_stats["L_use_inter"]
                    usage_gap_zero_inter = use_mask_stats["usage_gap_zero_inter"]
                    usage_gap_shuffle_inter = use_mask_stats["usage_gap_shuffle_inter"]
                else:
                    L_use_inter_t = torch.zeros((), device=device, dtype=torch.float32)
                L_use_total = L_use_full_t + eta_inter * L_use_inter_t
            else:
                L_use_total = torch.zeros((), device=device, dtype=torch.float32)

            # ---- L_rel: action representation loss (SigLIP / SupCon / none)
            # Common per-row meta (used by both SigLIP and SupCon paths).
            real_rows = sampler.real.iloc[real_idx]
            prim_ids = torch.tensor(
                [prim_to_id.get(p, -1) for p in real_rows.primitive.tolist()],
                device=device,
                dtype=torch.long,
            )
            # episode hash (collisions are extremely unlikely at our batch size)
            ep_keys = [
                f"{r.dataset}|{r.episode_id}" for r in real_rows.itertuples(index=False)
            ]
            ep_hash = torch.tensor(
                [hash(k) % (2**31 - 1) for k in ep_keys],
                device=device,
                dtype=torch.long,
            )
            confidence = torch.tensor(
                pd.to_numeric(real_rows.get("label_confidence", 1.0), errors="coerce")
                .fillna(0.0)
                .to_numpy(),
                device=device,
                dtype=torch.float32,
            )
            # default values for back-compat logging fields (set by each path below)
            cur_hard_neg_weight = 1.0
            supcon_stats: Dict = {
                "n_anchors": 0,
                "pairwise_cos_centered_mean": float("nan"),
                "n_pos_avg": 0.0,
                "n_neg_avg": 0.0,
                "n_hard_neg_pairs": 0,
                "hard_neg_weight_used": 1.0,
            }

            if action_loss_kind == "siglip":
                # ---- pairwise A1: SigLIP graph + pairwise sigmoid loss (pairwise.md §15)
                ehmp = torch.tensor(
                    real_rows["episode_has_multi_primitive"]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy(),
                    device=device,
                    dtype=torch.bool,
                )
                icd = torch.tensor(
                    real_rows["is_camera_dominant"]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy(),
                    device=device,
                    dtype=torch.bool,
                )
                ilm = torch.tensor(
                    real_rows["is_low_motion"].fillna(False).astype(bool).to_numpy(),
                    device=device,
                    dtype=torch.bool,
                )
                # pairwise: dataset hash (used for w_pos_cross_dataset boost)
                ds_arr = real_rows["dataset"].astype(str).to_numpy()
                # quick deterministic int hash so torch.eq can broadcast
                _ds_to_id = {"agibot": 0, "bridge": 1}
                ds_hash = torch.tensor(
                    [_ds_to_id.get(d, 2) for d in ds_arr],
                    device=device,
                    dtype=torch.long,
                )

                if structured_graph_enabled:
                    (
                        valid_pos_g,
                        valid_neg_g,
                        pos_weight_g,
                        neg_weight_g,
                        graph_stats,
                    ) = build_structured_siglip_graph(
                        primitives=prim_ids,
                        episodes=ep_hash,
                        confidence=confidence,
                        episode_has_multi_primitive=ehmp,
                        is_camera_dominant=icd,
                        is_low_motion=ilm,
                        opposite_table=opp_table_ids,
                        datasets=ds_hash,
                        min_confidence=siglip_min_confidence,
                        w_opposite=siglip_w_opposite,
                        w_hard_neg=siglip_w_hard_neg,
                        w_pos_cross_dataset=siglip_w_pos_cross_dataset,
                        neg_weight_mode=siglip_neg_weight_mode,
                        w_neg_cap=siglip_w_neg_cap,
                    )
                else:
                    valid_pos_g, valid_neg_g, neg_weight_g, graph_stats = (
                        build_siglip_graph_from_meta(
                            primitives=prim_ids,
                            episodes=ep_hash,
                            confidence=confidence,
                            episode_has_multi_primitive=ehmp,
                            is_camera_dominant=icd,
                            is_low_motion=ilm,
                            opposite_table=opp_table_ids,
                            min_confidence=siglip_min_confidence,
                            w_opposite=siglip_w_opposite,
                            w_hard_neg=siglip_w_hard_neg,
                        )
                    )
                    pos_weight_g = None  # falls back to valid_pos.float() in loss

                # pairwise+: pos/neg split λ schedule. Both warmup→ramp→cosine decay
                # but with their own targets. When NOT enabled, lam_rel ramps as
                # before and effective λ_pos = λ_neg = lam_rel via siglip_action_loss.
                def _ramp_decay(target: float, target_final: float) -> float:
                    if step < warmup_steps:
                        return 0.0
                    if step < ramp_end:
                        return (
                            target
                            * (step - warmup_steps)
                            / max(1, ramp_end - warmup_steps)
                        )
                    if lambda_sig_decay_end_step > ramp_end:
                        if step >= lambda_sig_decay_end_step:
                            return target_final
                        _frac = (step - ramp_end) / max(
                            1, lambda_sig_decay_end_step - ramp_end
                        )
                        _cos = 0.5 * (1.0 + float(np.cos(np.pi * _frac)))
                        return target_final + (target - target_final) * _cos
                    return target

                lam_pos_active = _ramp_decay(
                    siglip_lambda_pos_target, siglip_lambda_pos_target_final
                )
                lam_neg_active = _ramp_decay(
                    siglip_lambda_neg_target, siglip_lambda_neg_target_final
                )

                # pairwise: hard-positive mining — reweight pos_weight up for
                # high-conf same-prim positives whose current proj-cos is below
                # tau_pos. Detached cos so the reweight has no gradient.
                n_hard_pos_boosted = 0
                if siglip_hard_pos_enabled:
                    with torch.no_grad():
                        h_det, _, _ = siglip_head(z_mu_real.float())
                        cos_det = (h_det @ h_det.t()).detach()
                    base_pw = (
                        pos_weight_g
                        if pos_weight_g is not None
                        else valid_pos_g.float()
                    )
                    new_pw = hard_positive_reweight(
                        pos_weight=base_pw,
                        valid_pos=valid_pos_g,
                        cos=cos_det,
                        confidence=confidence,
                        tau_pos=siglip_hard_pos_tau,
                        gamma=siglip_hard_pos_gamma,
                        high_conf_min=siglip_hard_pos_high_conf_min,
                        max_weight=siglip_hard_pos_max_weight,
                    )
                    n_hard_pos_boosted = int(((new_pw - base_pw) > 0.01).sum().item())
                    pos_weight_g = new_pw

                if grouped_action_loss_enabled:
                    # pairwise: structured negatives (opp/hard/rand) + 5 λ schedules
                    opp_m, hard_m, rand_m = build_neg_subgroup_masks(
                        valid_neg_g,
                        prim_ids,
                        ep_hash,
                        opp_table_ids,
                    )
                    is_xds_pair = ds_hash.unsqueeze(0) != ds_hash.unsqueeze(1)
                    lam_pos_same_a = _ramp_decay(
                        siglip_lambda_pos_same_target,
                        siglip_lambda_pos_same_target_final,
                    )
                    lam_pos_xds_a = _ramp_decay(
                        siglip_lambda_pos_xds_target, siglip_lambda_pos_xds_target_final
                    )
                    lam_neg_opp_a = _ramp_decay(
                        siglip_lambda_neg_opp_target, siglip_lambda_neg_opp_target_final
                    )
                    lam_neg_hard_a = _ramp_decay(
                        siglip_lambda_neg_hard_target,
                        siglip_lambda_neg_hard_target_final,
                    )
                    lam_neg_rand_a = _ramp_decay(
                        siglip_lambda_neg_rand_target,
                        siglip_lambda_neg_rand_target_final,
                    )
                    L_rel, sig_stats = grouped_action_contrastive_loss(
                        z_mu=z_mu_real,
                        valid_pos=valid_pos_g,
                        head=siglip_head,
                        neg_subgroups=(opp_m, hard_m, rand_m),
                        pos_weight=pos_weight_g,
                        is_xds_pair=is_xds_pair,
                        lambda_pos_same=lam_pos_same_a,
                        lambda_pos_xds=lam_pos_xds_a,
                        lambda_neg_opp=lam_neg_opp_a,
                        lambda_neg_hard=lam_neg_hard_a,
                        lambda_neg_rand=lam_neg_rand_a,
                        neg_cap_default=siglip_w_neg_cap,
                        neg_cap_opposite=siglip_w_neg_cap_opposite,
                    )
                    lambda_rel_active = 1.0
                elif siglip_action_loss_split_enabled:
                    L_rel, sig_stats = siglip_action_loss_split(
                        z_mu=z_mu_real,
                        valid_pos=valid_pos_g,
                        valid_neg=valid_neg_g,
                        head=siglip_head,
                        pos_weight=pos_weight_g,
                        neg_weight=neg_weight_g,
                        lambda_pos=lam_pos_active,
                        lambda_neg=lam_neg_active,
                    )
                    # split path bakes lambda_pos / lambda_neg into the loss; we
                    # don't want the outer lam_rel multiplier to double-scale,
                    # so set lambda_rel_active = 1 below.
                    lambda_rel_active = 1.0
                else:
                    L_rel, sig_stats = siglip_action_loss(
                        z_mu=z_mu_real,
                        valid_pos=valid_pos_g,
                        valid_neg=valid_neg_g,
                        head=siglip_head,
                        neg_weight=neg_weight_g,
                        pos_weight=pos_weight_g,  # None when pairwise graph
                    )
                    # pairwise: outer cosine decay lambda_sig from `target` (at ramp_end)
                    # to `target_final` (at decay_end).
                    if lambda_sig_decay_end_step > ramp_end and step >= ramp_end:
                        if step >= lambda_sig_decay_end_step:
                            lambda_rel_active = lambda_sig_target_final
                        else:
                            _frac = (step - ramp_end) / max(
                                1, lambda_sig_decay_end_step - ramp_end
                            )
                            _cos = 0.5 * (1.0 + float(np.cos(np.pi * _frac)))
                            lambda_rel_active = (
                                lambda_sig_target_final
                                + (lambda_sig_target - lambda_sig_target_final) * _cos
                            )
                    else:
                        lambda_rel_active = lambda_sig_target

                # merge into supcon_stats for unified rec dict
                supcon_stats.update(graph_stats)
                supcon_stats.update(sig_stats)
                supcon_stats["n_anchors"] = graph_stats.get("n_pos_pairs", 0)
                supcon_stats["n_hard_neg_pairs"] = graph_stats.get(
                    "n_hard_neg_pairs", 0
                )
                supcon_stats["hard_neg_weight_used"] = float(siglip_w_hard_neg)
                supcon_stats["lam_pos_active"] = float(lam_pos_active)
                supcon_stats["lam_neg_active"] = float(lam_neg_active)
                supcon_stats["n_cross_ds_pos_pairs"] = graph_stats.get(
                    "n_cross_ds_pos_pairs", 0
                )
                supcon_stats["n_hard_pos_boosted"] = int(n_hard_pos_boosted)
                cur_hard_neg_weight = float(siglip_w_hard_neg)

            elif action_loss_kind == "supcon":
                # ---- earlier recipe / earlier recipe path: EMA-centered SupCon ----
                is_hard_neg_ep = torch.zeros_like(prim_ids, dtype=torch.bool)
                valid_pos, valid_neg, hard_neg_mask = build_supcon_masks_from_meta(
                    primitives=prim_ids,
                    episodes=ep_hash,
                    confidence=confidence,
                    is_hard_neg_episode=is_hard_neg_ep,
                    min_confidence=min_label_confidence,
                )
                # Per-step hard_neg_weight: cosine decay from init→target between
                # ramp_end and decay_end. Before ramp_end: init. After decay_end: target.
                # If decay_end <= ramp_end, behaves as fixed at init (earlier recipe/earlier recipe behaviour).
                if hard_neg_weight_decay_end_step > ramp_end and step >= ramp_end:
                    if step >= hard_neg_weight_decay_end_step:
                        cur_hard_neg_weight = hard_neg_weight_target
                    else:
                        _frac = (step - ramp_end) / max(
                            1, hard_neg_weight_decay_end_step - ramp_end
                        )
                        _cos = 0.5 * (1.0 + float(np.cos(np.pi * _frac)))
                        cur_hard_neg_weight = (
                            hard_neg_weight_target
                            + (hard_neg_weight_init - hard_neg_weight_target) * _cos
                        )
                else:
                    cur_hard_neg_weight = hard_neg_weight_init
                L_rel, sc_stats = centered_supcon_loss(
                    z_mu=z_mu_real,
                    primitives=prim_ids,
                    center=centerer.center,
                    valid_pos=valid_pos,
                    valid_neg=valid_neg,
                    temperature=supcon_temperature,
                    hard_neg_mask=hard_neg_mask,
                    hard_neg_weight=cur_hard_neg_weight,
                )
                supcon_stats.update(sc_stats)
                lambda_rel_active = lambda_rel_target

            else:  # action_loss_kind == "none"
                L_rel = torch.zeros((), device=device, dtype=torch.float32)
                lambda_rel_active = 0.0

            # ---- lambda_rel schedule (uses target appropriate to action_loss_kind)
            # When SigLIP split / structured path is on, all λs already include warmup
            # ramp + cosine decay (per `_ramp_decay`), so the outer ramp must be
            # disabled (lambda_rel_active set to 1.0 above).
            if action_loss_kind == "siglip" and (
                siglip_action_loss_split_enabled or grouped_action_loss_enabled
            ):
                lam_rel = 1.0
            elif step < warmup_steps:
                lam_rel = 0.0
            elif step < ramp_end:
                lam_rel = (
                    lambda_rel_active
                    * (step - warmup_steps)
                    / max(1, ramp_end - warmup_steps)
                )
            else:
                lam_rel = lambda_rel_active

            # ---- earlier recipe triplet aux on hard-neg (REPORT §6.5 Option B): direct
            # gap-as-target signal complementing SupCon's class log-prob signal.
            # Activates only after warmup (so init phase isn't biased) and only
            # if hard-neg pairs exist in batch (B_hardneg_pairs > 0).
            L_aux_triplet = torch.zeros((), device=device, dtype=torch.float32)
            aux_stats = {
                "n_aux_anchors": 0,
                "n_aux_triplets": 0,
                "L_aux_mean_gap": float("nan"),
                "L_aux_mean_loss": float("nan"),
            }
            if lambda_aux_triplet > 0.0 and step >= warmup_steps and not masked_reconstruction_enabled:
                L_aux_triplet, aux_stats = triplet_hardneg_aux_loss(
                    z_mu=z_mu_real,
                    center=centerer.center,
                    valid_pos=valid_pos,
                    hard_neg_mask=hard_neg_mask,
                    margin=aux_triplet_margin,
                )
            # Aux ramps with same schedule as L_rel
            lam_aux = (
                (lam_rel / lambda_rel_active) * lambda_aux_triplet
                if lambda_rel_active > 0
                else 0.0
            )

            # ---- pairwise: conditional cross-dataset centroid loss (SigLIP only) ----
            L_xds_centroid = torch.zeros((), device=device, dtype=torch.float32)
            xds_stats = {
                "L_xds_centroid": 0.0,
                "n_xds_qualified_primitives": 0,
                "n_xds_skipped_primitives": 0,
            }
            # ---- pairwise: ranking xds + prototype separation ----
            L_xds_rank = torch.zeros((), device=device, dtype=torch.float32)
            xds_rank_stats = {"L_xds_rank": 0.0, "n_xds_qualified": 0, "n_xds_terms": 0}
            L_proto_sep = torch.zeros((), device=device, dtype=torch.float32)
            proto_sep_stats = {
                "L_proto_sep": 0.0,
                "n_proto_pairs": 0,
                "n_proto_qualified": 0,
            }
            if (
                action_loss_kind == "siglip"
                and siglip_head is not None
                and step >= warmup_steps
            ):
                # Re-project z_mu through head (cheap, 8k params)
                h_xds, _, _ = siglip_head(z_mu_real.float())
                if siglip_xds_centroid_enabled:
                    L_xds_centroid, xds_stats = conditional_xds_centroid_loss(
                        h=h_xds,
                        primitives=prim_ids,
                        datasets=ds_hash,
                        n_primitives=len(canonical),
                        n_datasets=2,
                        min_count_per_cell=siglip_xds_min_count_per_cell,
                    )
                if siglip_xds_rank_enabled:
                    L_xds_rank, xds_rank_stats = xds_ranking_centroid_loss(
                        h=h_xds,
                        primitives=prim_ids,
                        datasets=ds_hash,
                        n_primitives=len(canonical),
                        n_datasets=2,
                        min_count_per_cell=siglip_xds_rank_min_count,
                        margin=siglip_xds_rank_margin,
                    )
                if siglip_proto_sep_enabled:
                    L_proto_sep, proto_sep_stats = primitive_separation_loss(
                        h=h_xds,
                        primitives=prim_ids,
                        n_primitives=len(canonical),
                        min_count_per_primitive=siglip_proto_sep_min_count,
                        m_sep=siglip_proto_sep_margin,
                    )

            L_total = (
                L_gen
                + lam_rel * L_rel
                + lam_aux * L_aux_triplet
                + siglip_xds_centroid_lambda * L_xds_centroid
                + siglip_xds_rank_lambda * L_xds_rank
                + siglip_proto_sep_lambda * L_proto_sep
                + w_id * L_id
                + lambda_use * L_use_total
            )

            # ---- backward + DDP + clip + step
            L_total.backward()
            trainable_params = [p for g in param_groups for p in g["params"]]
            # Belt-and-braces: every trainable param must have a tensor `.grad`
            # (not None) before all_reduce_grads. NCCL collective signatures
            # must match across ranks — if rank A has a non-None grad on some
            # param P and rank B has None, rank A issues an extra all_reduce
            # for P that rank B never matches → NCCL desync → ~600s timeout.
            # Bug observed 2026-05-08 legacy fixed-run fixed-run hung at step ~551
            # despite the n_pos==0 graph fix in siglip_action_loss; root cause
            # was likely some other batch-conditional path not touching head /
            # mask params on certain ranks. Filling in zeros makes all_reduce
            # see identical signatures regardless of which loss paths fired.
            for p in trainable_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            all_reduce_grads(trainable_params, world)
            grad_norm_total = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=grad_clip
            )
            if not torch.isfinite(grad_norm_total):
                if is_main:
                    log(f"step {step}: grad_norm_total non-finite, skipping optim.step")
                optim.zero_grad(set_to_none=True)
            else:
                if step % log_every == 0:
                    pre_norms = {
                        g["name"]: [p.detach().clone() for p in g["params"]]
                        for g in param_groups
                    }
                optim.step()

            t_compute = time.time() - t_compute_start

            # ---- logging
            rec_history.append(float(mse_real_full.item()))
            step_t = time.time() - t_step_start
            samples_per_step = (
                v_real.shape[0]
                + (v_id.shape[0] if v_id is not None else 0)
                + (v_mask.shape[0] if v_mask is not None else 0)
            ) * world
            samples_per_sec = samples_per_step / max(step_t, 1e-6)

            if step % log_every == 0 or step == total_steps - 1:
                update_ratios: Dict[str, float] = {}
                grad_norms: Dict[str, float] = {}
                for g in param_groups:
                    pn, gn = _param_group_norms(g["params"])
                    grad_norms[g["name"]] = gn
                    if pre_norms.get(g["name"]):
                        delta_sq, ref_sq = 0.0, 0.0
                        for p, ppre in zip(g["params"], pre_norms[g["name"]]):
                            d = (p.detach() - ppre).float()
                            delta_sq += d.pow(2).sum().item()
                            ref_sq += ppre.float().pow(2).sum().item()
                        update_ratios[g["name"]] = float(
                            (delta_sq**0.5) / (ref_sq**0.5 + 1e-8)
                        )
                    else:
                        update_ratios[g["name"]] = 0.0

                with torch.no_grad():
                    z_norm_mean = float(z_mu_real.norm(dim=1).mean().item())
                    z_norm_std = float(z_mu_real.norm(dim=1).std().item())
                    pairwise_cos_raw = float(
                        F.cosine_similarity(
                            z_mu_real.unsqueeze(0), z_mu_real.unsqueeze(1), dim=-1
                        )
                        .fill_diagonal_(0)
                        .sum()
                        .item()
                        / max(1, z_mu_real.shape[0] * (z_mu_real.shape[0] - 1))
                    )

                rec = {
                    "step": step,
                    "phase": state["phase"],
                    "wallclock_s": round(time.time() - t_loop_start, 1),
                    "step_s": round(step_t, 3),
                    "data_time_s": round(t_data, 3),
                    "compute_time_s": round(t_compute, 3),
                    "samples_per_sec_total": round(samples_per_sec, 1),
                    "samples_per_sec_per_gpu": round(
                        samples_per_sec / max(world, 1), 1
                    ),
                    "L_rec_full": float(mse_real_full.item()),
                    "L_rec_inter": float(L_rec_inter.item())
                    if isinstance(L_rec_inter, torch.Tensor)
                    else float(L_rec_inter),
                    "n_inter_rows": int(n_inter_rows),
                    # ---- masked reconstruction stats ----
                    "masked_reconstruction_enabled": bool(masked_reconstruction_enabled),
                    "L_rec_fg": float(L_rec_fg_t.item())
                    if isinstance(L_rec_fg_t, torch.Tensor)
                    else 0.0,
                    "L_rec_bg": float(L_rec_bg_t.item())
                    if isinstance(L_rec_bg_t, torch.Tensor)
                    else 0.0,
                    "masked_foreground_rows": int(reconstruction_stats.get("n_fg_rows", 0))
                    if reconstruction_stats
                    else 0,
                    "foreground_area_mean": float(reconstruction_stats.get("fg_area_mean", 0.0))
                    if reconstruction_stats
                    else 0.0,
                    "background_area_mean": float(reconstruction_stats.get("bg_area_mean", 0.0))
                    if reconstruction_stats
                    else 0.0,
                    "foreground_reconstruction_weight": float(foreground_reconstruction_weight),
                    "background_consistency_weight": float(background_consistency_weight),
                    "L_kl": float(kl_loss.item()),
                    "L_gen": float(L_gen.item()),
                    "L_rel": float(L_rel.item()),
                    "L_id": float(L_id.item()),
                    "L_use_full": L_use_full_val,
                    "L_use_inter": L_use_inter_val,
                    "L_total": float(L_total.item()),
                    "lambda_rel": float(lam_rel),
                    "lambda_rel_active_target": float(lambda_rel_active),
                    "lambda_use": float(lambda_use),
                    "alpha_inter": float(alpha_inter),
                    "eta_inter": float(eta_inter),
                    "beta_kl": float(beta_kl),
                    "kl_free_bit": float(kl_free_bit),
                    "kl_total": kl_stats["kl_total"],
                    "kl_per_dim_mean": kl_stats["kl_per_dim_mean"],
                    "active_kl_dims": kl_stats["active_kl_dims"],
                    "z_mu_std_mean": kl_stats["z_mu_std_mean"],
                    "z_norm_mean": z_norm_mean,
                    "z_norm_std": z_norm_std,
                    "pairwise_cos_raw": pairwise_cos_raw,
                    "action_loss_kind": action_loss_kind,
                    "pairwise_cos_centered": supcon_stats.get(
                        "pairwise_cos_centered_mean", float("nan")
                    ),
                    "n_supcon_anchors": supcon_stats.get("n_anchors", 0),
                    "n_supcon_pos_avg": supcon_stats.get("n_pos_avg", 0.0),
                    "n_supcon_neg_avg": supcon_stats.get("n_neg_avg", 0.0),
                    "n_supcon_hard_neg_pairs": supcon_stats.get("n_hard_neg_pairs", 0),
                    "hard_neg_weight": float(cur_hard_neg_weight),
                    "L_aux_triplet": float(L_aux_triplet.item())
                    if isinstance(L_aux_triplet, torch.Tensor)
                    else float(L_aux_triplet),
                    "lam_aux_triplet": float(lam_aux),
                    "L_aux_mean_gap": aux_stats.get("L_aux_mean_gap", float("nan")),
                    "n_aux_anchors": aux_stats.get("n_aux_anchors", 0),
                    "n_aux_triplets": aux_stats.get("n_aux_triplets", 0),
                    # SigLIP-only diagnostics (NaN/0 when action_loss_kind != "siglip")
                    "siglip_scale": supcon_stats.get("siglip_scale", float("nan")),
                    "siglip_bias": supcon_stats.get("siglip_bias", float("nan")),
                    "L_sig_pos": supcon_stats.get("L_sig_pos", float("nan")),
                    "L_sig_neg": supcon_stats.get("L_sig_neg", float("nan")),
                    "pcos_proj_mean": supcon_stats.get("pcos_proj_mean", float("nan")),
                    "cos_proj_pos_mean": supcon_stats.get(
                        "cos_proj_pos_mean", float("nan")
                    ),
                    "cos_proj_neg_mean": supcon_stats.get(
                        "cos_proj_neg_mean", float("nan")
                    ),
                    "s_pos_mean": supcon_stats.get("s_pos_mean", float("nan")),
                    "s_neg_mean": supcon_stats.get("s_neg_mean", float("nan")),
                    "n_pos_pairs": supcon_stats.get("n_pos_pairs", 0),
                    "n_neg_pairs": supcon_stats.get("n_neg_pairs", 0),
                    "n_opposite_pairs": supcon_stats.get("n_opposite_pairs", 0),
                    "usage_gap_zero_full": usage_gap_zero_full,
                    "usage_gap_shuffle_full": usage_gap_shuffle_full,
                    "usage_gap_zero_inter": usage_gap_zero_inter,
                    "usage_gap_shuffle_inter": usage_gap_shuffle_inter,
                    "same_episode_negative_rate": meta.get(
                        "same_episode_negative_rate", float("nan")
                    ),
                    "cross_dataset_positive_rate": meta.get(
                        "cross_dataset_positive_rate", float("nan")
                    ),
                    "primitive_balance_entropy": meta.get(
                        "primitive_balance_entropy", float("nan")
                    ),
                    "actual_dataset_mix_real": meta.get("actual_dataset_mix_real", {}),
                    "grad_norm_total": float(grad_norm_total.item())
                    if torch.isfinite(grad_norm_total)
                    else None,
                    "update_ratio": update_ratios,
                    "grad_norm_per_group": grad_norms,
                }
                if is_main and train_f is not None:
                    train_f.write(json.dumps(rec) + "\n")
                if is_main:
                    log(
                        f"step {step:5d}/{total_steps} [{state['phase']:>6}] "
                        f"L_rec={rec['L_rec_full']:.4f}/{rec['L_rec_inter']:.4f} "
                        f"KL={rec['kl_total']:.2f}/active={rec['active_kl_dims']} "
                        f"L_rel={rec['L_rel']:.4f} L_id={rec['L_id']:.4f} "
                        f"L_use={rec['L_use_full']!s:>6}/{rec['L_use_inter']!s:>6} "
                        f"lam_rel={lam_rel:.4f} pcos_raw={pairwise_cos_raw:.3f} "
                        f"pcos_c={rec['pairwise_cos_centered']:.3f} "
                        f"z|.|={z_norm_mean:.3f}±{z_norm_std:.3f} "
                        f"step={step_t:.2f}s ({rec['samples_per_sec_per_gpu']}/s/gpu, data={rec['data_time_s']:.2f}s)"
                    )

            # ---- phase + state update
            if step < warmup_steps:
                state["phase"] = "warmup"
                state["status"] = "warmup"
            elif step < ramp_end:
                state["phase"] = "ramp"
                state["status"] = "ramp"
            else:
                state["phase"] = "main"
                state["status"] = "main"
            state["current_step"] = step
            state["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            if is_main and step % 50 == 0:
                write_run_state(run_state_path, state)

            # ---- warmup → ramp gate
            if step == warmup_steps and warmup_steps > 0:
                gate_issues = []
                with torch.no_grad():
                    cur_z_norm = float(z_mu_real.detach().norm(dim=1).mean().item())
                if m0_z_norm and cur_z_norm < 0.4 * m0_z_norm:
                    gate_issues.append(
                        f"z_norm {cur_z_norm:.2f} < 0.4 × baseline ({0.4 * m0_z_norm:.2f}) — z collapsed"
                    )
                if kl_stats["kl_total"] < 1.0:
                    gate_issues.append(
                        f"KL_total {kl_stats['kl_total']:.3f} < 1.0 — posterior collapsed"
                    )
                if not np.isnan(usage_gap_zero_full) and usage_gap_zero_full < 0.0005:
                    gate_issues.append(
                        f"usage_gap_zero_full {usage_gap_zero_full:.4f} < 0.0005 — decoder bypassing z"
                    )
                if gate_issues:
                    log(
                        f"WARMUP HEALTH GATE FAIL @ step {warmup_steps}: {gate_issues}",
                        all_ranks=True,
                    )
                    if is_main:
                        state["status"] = "paused"
                        state["pause_reason"] = (
                            "warmup health gate failed: " + "; ".join(gate_issues)
                        )
                        write_run_state(run_state_path, state)
                    pause_reason = "warmup health gate"
                    break
                else:
                    log(
                        f"WARMUP HEALTH GATE PASS @ step {warmup_steps}: "
                        f"z_norm={cur_z_norm:.2f} kl={kl_stats['kl_total']:.2f} "
                        f"usage_gap_zero_full={usage_gap_zero_full}",
                        all_ranks=True,
                    )

            # ---- ckpt + eval (rank0 only)
            stop_now = False
            if (step > 0 and step % ckpt_every == 0) or step == total_steps - 1:
                if is_main:
                    ckpt_path = out_dir / "checkpoints" / f"step_{step:06d}.pt"
                    ckpt_blob = {
                        "step": step,
                        "model": {
                            f"lam.{k}": v for k, v in lam_inner.state_dict().items()
                        },
                        "optimizer": optim.state_dict(),
                        "centerer": centerer.state_dict(),
                        "config_path": args.config,
                        "args": vars(args),
                        "trainer": (
                            "cdlam_stage1_siglip"
                            if action_loss_kind == "siglip"
                            else "cdlam_stage1_pairwise"
                        ),
                        "action_loss_kind": action_loss_kind,
                    }
                    if siglip_head is not None:
                        ckpt_blob["siglip_head"] = siglip_head.state_dict()
                    torch.save(ckpt_blob, ckpt_path)
                    state["latest_checkpoint"] = str(ckpt_path)
                    log(f"step {step}: saved {ckpt_path}")
                    write_run_state(run_state_path, state)
                if world > 1:
                    dist.barrier()

            if (
                (step > 0 and step % eval_every == 0) or step == total_steps - 1
            ) and not args.skip_eval:
                if is_main:
                    eval_path = out_dir / "eval" / f"step_{step:06d}.json"
                    eval_path.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_path_str = state.get("latest_checkpoint")
                    if ckpt_path_str is None:
                        log(f"step {step}: no checkpoint yet, skipping eval")
                    else:
                        cmd = [
                            sys.executable,
                            str(REPO / "cdlam_integration/lam/eval.py"),
                            "--pair-index",
                            cad["eval_split_parquet"],
                            "--out",
                            str(eval_path),
                            "--checkpoint",
                            str(ckpt_path_str),
                            "--n-pairs-real",
                            str(int(cad["eval_n_pairs_real"])),
                            "--n-pairs-id",
                            str(int(cad["eval_n_pairs_id"])),
                            "--n-per-primitive",
                            str(int(cad["eval_n_per_primitive"])),
                            "--n-recon-tile",
                            str(int(cad.get("eval_n_recon_tile", 8))),
                            "--seed",
                            str(args.seed),
                        ]
                        if cad.get("baseline_checkpoint"):
                            cmd += [
                                "--baseline-checkpoint",
                                str(cad["baseline_checkpoint"]),
                            ]
                        log(f"step {step}: running eval -> {eval_path}")
                        sub_env = {**os.environ}
                        if world > 1:
                            sub_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
                            for k in (
                                "RANK",
                                "LOCAL_RANK",
                                "WORLD_SIZE",
                                "MASTER_ADDR",
                                "MASTER_PORT",
                            ):
                                sub_env.pop(k, None)
                        try:
                            t_eval_start = time.time()
                            res = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=900,
                                env=sub_env,
                            )
                            t_eval = time.time() - t_eval_start
                            if res.returncode != 0:
                                log(
                                    f"step {step}: eval FAILED rc={res.returncode}\n"
                                    f"{res.stdout[-1500:]}\n{res.stderr[-1500:]}"
                                )
                            else:
                                ev = json.loads(eval_path.read_text())
                                ev["step"] = step
                                ev["wallclock_s"] = round(time.time() - t_loop_start, 1)
                                ev["eval_time_s"] = round(t_eval, 1)
                                eval_f.write(json.dumps(ev, default=str) + "\n")
                                state["latest_eval"] = str(eval_path)

                                issues: List[str] = []
                                geom = ev.get("trained", {}).get("z_geometry", {})
                                ret = ev.get("trained", {}).get("retrieval", {})
                                idr = ev.get("trained", {}).get(
                                    "identity_ratio_vs_median", {}
                                )
                                ug = ev.get("trained", {}).get("usage_gap", {})
                                if (
                                    idr.get("p50")
                                    and idr["p50"] > stop_cfg["identity_ratio_max"]
                                ):
                                    issues.append(
                                        f"identity_p50={idr['p50']:.3f} > {stop_cfg['identity_ratio_max']}"
                                    )
                                if m0_eff_rank and geom.get("effective_rank"):
                                    drop = (
                                        m0_eff_rank - geom["effective_rank"]
                                    ) / m0_eff_rank
                                    if drop > stop_cfg["eff_rank_drop_frac_max"]:
                                        issues.append(
                                            f"eff_rank_drop={drop:.3f} > {stop_cfg['eff_rank_drop_frac_max']}"
                                        )
                                if ug.get("zero_p50") is not None and ug[
                                    "zero_p50"
                                ] <= stop_cfg.get("usage_gap_zero_min", 0.0):
                                    issues.append(
                                        f"usage_gap_zero_p50={ug['zero_p50']:.4f} ≤ floor"
                                    )
                                if ug.get("shuffle_p50") is not None and ug[
                                    "shuffle_p50"
                                ] <= stop_cfg.get("usage_gap_shuffle_min", 0.0):
                                    issues.append(
                                        f"usage_gap_shuffle_p50={ug['shuffle_p50']:.4f} ≤ floor"
                                    )
                                if ret.get("top1") is not None and ret[
                                    "top1"
                                ] < stop_cfg.get("top1_min", 0.0):
                                    issues.append(
                                        f"top1={ret['top1']:.3f} < {stop_cfg['top1_min']}"
                                    )
                                # Stage-1 specifically watches raw pairwise cosine (common-cone)
                                pcos_raw = geom.get("pairwise_cos_mean", float("nan"))
                                if (
                                    pcos_raw is not None
                                    and not np.isnan(pcos_raw)
                                    and pcos_raw
                                    > stop_cfg.get("pairwise_cos_raw_max", 1.0)
                                ):
                                    issues.append(
                                        f"pairwise_cos_raw={pcos_raw:.3f} > {stop_cfg['pairwise_cos_raw_max']}"
                                    )

                                log(
                                    f"step {step}: eval | "
                                    f"top1={ret.get('top1')} gap={ret.get('gap')} id_p50={idr.get('p50')} "
                                    f"z_norm={geom.get('z_mu_norm_mean')} eff_rank={geom.get('effective_rank')} "
                                    f"pcos_raw={pcos_raw} "
                                    f"u_gap_z={ug.get('zero_p50')} u_gap_s={ug.get('shuffle_p50')} "
                                    f"eval_time={t_eval:.1f}s"
                                )
                                if issues:
                                    log(f"step {step}: STOP CONDITION HIT: {issues}")
                                    state["status"] = "paused"
                                    state["pause_reason"] = "; ".join(issues)
                                    pause_reason = state["pause_reason"]
                                    write_run_state(run_state_path, state)
                                    stop_now = True
                        except subprocess.TimeoutExpired:
                            log(f"step {step}: eval TIMEOUT after 900s")
                        except Exception as exc:
                            log(
                                f"step {step}: eval EXCEPTION {exc}\n{traceback.format_exc()}"
                            )
                if world > 1:
                    dist.barrier()
                if world > 1:
                    stop_tensor = torch.tensor(
                        [1 if stop_now else 0], device=device, dtype=torch.int
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop_now = bool(stop_tensor.item())
                if stop_now:
                    break

            step += 1

        if pause_reason is None:
            state["status"] = "completed"
        log(f"finished. status={state['status']} step={step}")

    except KeyboardInterrupt:
        log("KeyboardInterrupt — saving state as 'interrupted'")
        state["status"] = "interrupted"
        state["pause_reason"] = "KeyboardInterrupt"
    except Exception as exc:
        log(
            f"EXCEPTION at step {step}: {exc}\n{traceback.format_exc()}", all_ranks=True
        )
        state["status"] = "failed"
        state["pause_reason"] = f"{type(exc).__name__}: {exc}"
        if is_main:
            try:
                ckpt_path = out_dir / "checkpoints" / f"crash_step_{step:06d}.pt"
                crash_blob = {
                    "step": step,
                    "model": {f"lam.{k}": v for k, v in lam_inner.state_dict().items()},
                    "trainer": (
                        "cdlam_stage1_siglip"
                        if action_loss_kind == "siglip"
                        else "cdlam_stage1_pairwise"
                    ),
                    "action_loss_kind": action_loss_kind,
                }
                if siglip_head is not None:
                    crash_blob["siglip_head"] = siglip_head.state_dict()
                torch.save(crash_blob, ckpt_path)
                state["latest_checkpoint"] = str(ckpt_path)
            except Exception:
                pass
    finally:
        state["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        state["current_step"] = step
        if is_main:
            write_run_state(run_state_path, state)
            if train_f is not None:
                train_f.close()
            if eval_f is not None:
                eval_f.close()
        log_f.close()
        try:
            prefetch_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if world > 1 and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--init-ckpt",
        default=None,
        help="path to trainer-format ckpt (model keys 'lam.<...>') used as init; "
        "overrides config trainer.init_ckpt",
    )
    ap.add_argument(
        "--m0-baseline",
        default=None,
        help="path to baseline metrics metrics json for warmup gate / stop conditions",
    )
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--ramp-end", type=int, default=None)
    ap.add_argument("--lambda-rel-target", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--batch-real", type=int, default=None)
    ap.add_argument("--batch-id", type=int, default=None)
    ap.add_argument("--batch-mask", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip the in-process eval subprocess (smoke / debug)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
