#!/usr/bin/env python3
"""cdlam_stage1 trainer
====================================

Mainline as of 2026-05-06 (replacing earlier trainer frozen-decoder line):

  - encoder + decoder + patch_up + action_up ALL trainable (joint)
  - From baseline camera-clean ckpt as init
  - Loss = L_gen + w_gap·L_gap + w_id·L_id + w_use·L_use
        (no L_norm, no L_trust, no SupCon, no fake-cam train, no inverse dynamics)
  - L_use is HARD requirement: decoder must use z (real_z reconstructs better
    than zero_z and shuffled_z by margin_use).
  - KL free-bits opt-in (KL_dim_eff = max(KL_dim, free_bit)).
  - Param groups: G_head (action_prompt + encoder.out + fc), G_late (block 19-23),
    G_mid (block 8-18), G_early (block 0-7), G_decoder (decoder + patch_up + action_up).
  - update_ratio = ||Δθ_g|| / (||θ_g|| + eps) logged per group.
  - Throughput stats: data_time / compute_time / samples_per_sec_gpu.
  - 4-card DDP via torchrun --nproc_per_node=4, manual all_reduce of grads
    (consistent with earlier trainer trainer; our forward goes through custom encode/decode).
  - run_state.json + train_metrics.jsonl + eval_metrics.jsonl + checkpoints/ + logs/rank*.log

Phases:
  preflight   : 100-300 step verify
  warmup      : [0, warmup_steps)         L = L_gen + w_id·L_id + w_use·L_use   (w_gap=0)
  ramp        : [warmup_steps, ramp_end)  w_gap 0 → target linear
  main        : [ramp_end, total_steps)   w_gap = target

For Stage-1_no_gap control, just set --w-gap-target 0.0 (no schedule needed).

Usage (preflight):
    torchrun --nproc_per_node=4 --master_port=29501 \\
      cdlam_integration/tools/optimizer_helpers.py \\
        --config configs/stage1_recipe.yaml \\
        --out outputs/cdlam_train/Stage-1_preflight \\
        --m0-baseline outputs/cdlam_train/earlier trainer_all_encoder_gen_gap_guard/baseline_baseline.json \\
        --total-steps 200 --warmup-steps 50 --ramp-end 100 \\
        --batch-real 16 --batch-id 4 --triplets 8 \\
        --log-every 25 --ckpt-every 200 --eval-every 200

Usage (main):
    torchrun --nproc_per_node=4 --master_port=29501 \\
      cdlam_integration/tools/optimizer_helpers.py \\
        --config configs/stage1_recipe.yaml \\
        --out outputs/cdlam_train/cdlam_stage1 \\
        --m0-baseline outputs/cdlam_train/earlier trainer_all_encoder_gen_gap_guard/baseline_baseline.json
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
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

# Reuse earlier trainer's sampler + decode + DDP helpers.
from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
from cdlam_integration.lam.model_ops import (  # noqa: E402
    encode_full,
    decode_full,
    forward_full,
)
from cdlam_integration.lam.distributed_helpers import (  # noqa: E402
    TrainingBatchSampler,
    decode_rows_parallel,
    init_distributed,
    all_reduce_grads,
    write_run_state,
)


# =============== Param groups (Stage-1-specific) ================================


def collect_trainable_parameter_groups(lam_inner: torch.nn.Module, lr_cfg: dict) -> List[dict]:
    """Five-group learning-rate partition:

        G_head        : action_prompt + encoder.out + fc
        G_late        : encoder.transformer_blocks[19..23]
        G_mid         : encoder.transformer_blocks[8..18]
        G_early       : encoder.transformer_blocks[0..7] + encoder.ffn + encoder.pos_enc
        G_decoder     : decoder + patch_up + action_up

    All trainable; no freeze.

    LR ordering: head ≥ late_encoder > mid/early > decoder
    (decoder LR should not exceed late_encoder initially).
    """
    groups: Dict[str, List[torch.nn.Parameter]] = defaultdict(list)

    # encoder blocks
    blocks = lam_inner.encoder.transformer_blocks
    for i, blk in enumerate(blocks):
        for p in blk.parameters():
            if 0 <= i <= 7:
                groups["G_early"].append(p)
            elif 8 <= i <= 18:
                groups["G_mid"].append(p)
            elif 19 <= i <= 23:
                groups["G_late"].append(p)
    # encoder.ffn / pos_enc — go with G_early
    if hasattr(lam_inner.encoder, "ffn") and lam_inner.encoder.ffn is not None:
        for p in lam_inner.encoder.ffn.parameters():
            groups["G_early"].append(p)
    if hasattr(lam_inner.encoder, "pos_enc"):
        for p in lam_inner.encoder.pos_enc.parameters():
            groups["G_early"].append(p)
    # encoder.out + fc + action_prompt -> G_head
    if hasattr(lam_inner.encoder, "out") and lam_inner.encoder.out is not None:
        for p in lam_inner.encoder.out.parameters():
            groups["G_head"].append(p)
    for p in lam_inner.fc.parameters():
        groups["G_head"].append(p)
    groups["G_head"].append(lam_inner.action_prompt)
    # decoder + patch_up + action_up -> G_decoder
    for p in lam_inner.decoder.parameters():
        groups["G_decoder"].append(p)
    for p in lam_inner.patch_up.parameters():
        groups["G_decoder"].append(p)
    for p in lam_inner.action_up.parameters():
        groups["G_decoder"].append(p)

    # build optimizer groups, set requires_grad
    out_groups = []
    for name in ("G_head", "G_late", "G_mid", "G_early", "G_decoder"):
        params = groups.get(name, [])
        if not params:
            continue
        if name not in lr_cfg:
            raise ValueError(f"lr config missing group '{name}'")
        lr = float(lr_cfg[name])
        if lr <= 0:
            for p in params:
                p.requires_grad = False
            continue
        for p in params:
            p.requires_grad = True
        out_groups.append({"params": params, "lr": lr, "name": name})

    # Sanity: any param not assigned ⇒ frozen (defensive).
    assigned = set()
    for g in out_groups:
        for p in g["params"]:
            assigned.add(id(p))
    for p in lam_inner.parameters():
        if id(p) not in assigned:
            p.requires_grad = False

    return out_groups


def _param_group_norms(group_params: List[torch.nn.Parameter]) -> Tuple[float, float]:
    """Returns (||θ||, ||grad||) for a parameter group."""
    pnorm_sq = 0.0
    gnorm_sq = 0.0
    for p in group_params:
        pnorm_sq += p.detach().float().pow(2).sum().item()
        if p.grad is not None:
            gnorm_sq += p.grad.detach().float().pow(2).sum().item()
    return float(pnorm_sq**0.5), float(gnorm_sq**0.5)


# =============== KL with free-bits =========================================


def kl_loss_free_bits(
    z_mu: torch.Tensor, z_var: torch.Tensor, free_bit: float = 0.0
) -> Tuple[torch.Tensor, dict]:
    """KL(N(z_mu, exp(z_var)) || N(0, I)) summed over latent dim, mean over batch.
    With optional free-bits floor: KL_per_dim_eff = max(KL_per_dim, free_bit).

    Mixed-precision: forces fp32 for the exp(z_var) and z_mu^2 computations even
    when caller is inside a bf16 autocast region. Both are numerically sensitive.

    Returns: (loss, stats).
    """
    z_mu_f = z_mu.float()
    z_var_f = z_var.float()
    # clamp z_var to avoid exp overflow even in fp32 (paranoid bound, matches
    # typical VAE training; values outside [-10, 10] are unphysical for log-var)
    z_var_clamped = torch.clamp(z_var_f, min=-10.0, max=10.0)
    kl_per = -0.5 * (1 + z_var_f - z_mu_f.pow(2) - z_var_clamped.exp())  # (B, D)  fp32
    kl_per_dim_mean_batch = kl_per.mean(dim=0)  # (D,)
    if free_bit > 0:
        kl_per_dim_eff = torch.maximum(
            kl_per_dim_mean_batch, torch.full_like(kl_per_dim_mean_batch, free_bit)
        )
        loss = kl_per_dim_eff.sum()
    else:
        loss = kl_per.sum(dim=1).mean()
    stats = {
        "kl_total": float(kl_per.sum(dim=1).mean().item()),
        "kl_per_dim_mean": float(kl_per_dim_mean_batch.mean().item()),
        "active_kl_dims": int((kl_per_dim_mean_batch > 0.01).sum().item()),
        "z_mu_std_mean": float(z_mu_f.detach().std(dim=0).mean().item()),
    }
    return loss, stats


# =============== Run state =================================================


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


def train(args, cfg):
    rank, world, local_rank = init_distributed()
    is_main = rank == 0

    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        (out_dir / "configs").mkdir(exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)
        shutil.copy2(args.config, out_dir / "configs" / "stage1_config.yaml.copy")
    if world > 1:
        dist.barrier()

    log_path = out_dir / "logs" / f"rank{rank}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg: str, all_ranks: bool = False):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}][r{rank}] {msg}"
        if is_main or all_ranks:
            print(line, flush=True)
        log_f.write(line + "\n")

    train_metrics_path = out_dir / "train_metrics.jsonl"
    eval_metrics_path = out_dir / "eval_metrics.jsonl"
    run_state_path = out_dir / "run_state.json"
    if is_main:
        train_f = open(train_metrics_path, "a", buffering=1)
        eval_f = open(eval_metrics_path, "a", buffering=1)
    else:
        train_f = None
        eval_f = None

    state = make_state(args, world, rank, local_rank)
    if is_main:
        write_run_state(run_state_path, state)

    if world > 1:
        device = f"cuda:{local_rank}"
    else:
        device = cfg["trainer"]["device"]
    target_hw = (int(cfg["trainer"]["target_h"]), int(cfg["trainer"]["target_w"]))
    log(
        f"device={device}  pid={os.getpid()}  world={world}  out={out_dir}",
        all_ranks=True,
    )

    # ----- batch / schedule (CLI overrides config)
    B_real = int(args.batch_real or cfg["trainer"]["batch"]["B_real"])
    B_id = int(args.batch_id or cfg["trainer"]["batch"]["B_id"])
    N_triplet = int(args.triplets or cfg["trainer"]["batch"]["N_triplet"])
    decode_workers = int(cfg["trainer"]["batch"]["decode_workers"])

    total_steps = int(args.total_steps or cfg["trainer"]["schedule"]["total_steps"])
    warmup_steps = int(
        args.warmup_steps
        if args.warmup_steps is not None
        else cfg["trainer"]["schedule"]["warmup_steps"]
    )
    ramp_end = int(
        args.ramp_end
        if args.ramp_end is not None
        else cfg["trainer"]["schedule"]["ramp_end_step"]
    )
    log_every = int(args.log_every or cfg["trainer"]["cadence"]["log_every"])
    ckpt_every = int(args.ckpt_every or cfg["trainer"]["cadence"]["ckpt_every"])
    eval_every = int(args.eval_every or cfg["trainer"]["cadence"]["eval_every"])
    l_use_every = int(cfg["trainer"]["loss"].get("l_use_every", 2))
    grad_clip = float(cfg["trainer"]["lr"].get("grad_clip_max_norm", 1.0))

    # use_grad_ckpt: trade compute for memory. CLI override > config.
    if args.no_grad_ckpt:
        use_ckpt = False
    elif args.grad_ckpt:
        use_ckpt = True
    else:
        use_ckpt = bool(cfg["trainer"]["batch"].get("use_grad_ckpt", True))
    log(f"use_grad_ckpt={use_ckpt}")

    loss_cfg = cfg["trainer"]["loss"]
    # ---- beta_kl schedule (anti posterior-collapse) -----------------------
    # Old behavior: constant beta_kl = `beta_kl` (default 0.01). Caused collapse
    # in Stage-1 earlier recipe at step 500 (KL 77 → 0.98, z_norm 6.4 → 1.1, usage_gap → 0).
    # New: linear ramp from `beta_kl_init` (e.g. 1e-4) at step 0 to `beta_kl_target`
    #      (e.g. 1e-3) at `beta_kl_ramp_end_step` (typically same as warmup_steps).
    # Backwards-compat: if config still uses single `beta_kl` and not the schedule
    # keys, treat as constant (init=target=beta_kl).
    beta_kl_init = float(loss_cfg.get("beta_kl_init", loss_cfg.get("beta_kl", 0.01)))
    beta_kl_target = float(
        loss_cfg.get("beta_kl_target", loss_cfg.get("beta_kl", 0.01))
    )
    beta_kl_ramp_end_step = int(loss_cfg.get("beta_kl_ramp_end_step", 0))
    # Default exposed value for logging
    beta_kl = beta_kl_init
    kl_free_bit = float(loss_cfg.get("kl_free_bit", 0.0))
    w_gap_target = float(
        args.w_gap_target if args.w_gap_target is not None else loss_cfg["w_gap_target"]
    )
    w_id = float(loss_cfg["w_id"])
    w_use = float(loss_cfg["w_use"])
    margin = float(loss_cfg["margin"])
    margin_use_init = float(loss_cfg["margin_use_init"])
    margin_use_frac = float(loss_cfg.get("margin_use_frac_of_lrec", 0.05))

    log(
        f"steps={total_steps} warmup={warmup_steps} ramp_end={ramp_end} "
        f"w_gap_target={w_gap_target} w_id={w_id} w_use={w_use} "
        f"beta_kl_init={beta_kl_init} beta_kl_target={beta_kl_target} "
        f"beta_kl_ramp_end_step={beta_kl_ramp_end_step} kl_free_bit={kl_free_bit}"
    )
    log(
        f"B_real={B_real} B_id={B_id} N_triplet={N_triplet} "
        f"margin={margin} margin_use_init={margin_use_init} l_use_every={l_use_every}"
    )

    # ----- baseline metrics (for stop-condition floors)
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

    # ----- sampler
    sampler = TrainingBatchSampler(
        pair_index_path=Path(cfg["trainer"]["pair_dir"]) / "pair_index_train.parquet",
        canonical_primitives=cfg["trainer"]["triplet"]["canonical_primitives"],
        p_neg_same_ep=cfg["trainer"]["triplet"]["p_negative_same_episode"],
        seed=args.seed + 1009 * rank,
    )
    log(
        f"sampler real={len(sampler.real):,} id={len(sampler.id):,} "
        f"primitives={len(sampler.primitives)}"
    )

    # ----- build trainable LAM (init from baseline)
    log("building trainable Stage-1 LAM (init from baseline ckpt) ...")
    lam_train = build_lam("CD_LAM", device=device)
    lam_inner = lam_train.lam
    lam_inner.train()

    # The decoder is trainable in Stage 1; the base model is used only as init.

    # ----- param groups
    param_groups = collect_trainable_parameter_groups(lam_inner, cfg["trainer"]["lr"])
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    n_total = sum(p.numel() for p in lam_inner.parameters())
    log(f"trainable {n_train:,} / {n_total:,} = {n_train / n_total * 100:.1f}%")
    for g in param_groups:
        log(
            f"  {g['name']:>10}: lr={g['lr']:.2e}  n_params={sum(p.numel() for p in g['params']):,}"
        )

    # ----- optimizer
    optim = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["trainer"]["lr"]["weight_decay"]),
        eps=float(cfg["trainer"]["lr"]["eps"]),
    )

    # ----- L_id EMA scale
    s_ema = None
    s_ema_alpha = 0.95

    # ----- L_rec EMA for adaptive margin_use
    lrec_ema = None
    lrec_ema_alpha = 0.95

    rec_history = deque(maxlen=200)

    state["status"] = "warmup" if warmup_steps > 0 else "ramp"
    state["phase"] = (
        "warmup" if warmup_steps > 0 else "ramp" if total_steps > 1 else "main"
    )
    if is_main:
        write_run_state(run_state_path, state)

    # ===== Prefetch helper — overlaps cv2 decode with previous-step's forward+backward.
    # Builds the next batch (sampling + decoding all 3 of real/id/pn) in a background
    # thread; main thread just waits on the future at the start of the next step.
    def _build_one_batch():
        ri, ii, tri, m = sampler.sample(B_real, B_id, N_triplet)
        rr = sampler.real.iloc[ri]
        ir = sampler.id.iloc[ii]
        pn_idx = sorted({i for (_, p, n) in tri for i in (p, n)})
        pnr = sampler.real.iloc[pn_idx]
        rp, rv = decode_rows_parallel(rr, target_hw, decode_workers)
        ip, iv = decode_rows_parallel(ir, target_hw, decode_workers)
        pp, pv = decode_rows_parallel(pnr, target_hw, decode_workers)
        return ri, ii, tri, m, pn_idx, rp, rv, ip, iv, pp, pv

    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    prefetch_fut = prefetch_pool.submit(_build_one_batch)

    # ===== training loop =====
    step = 0
    pause_reason: Optional[str] = None
    t_loop_start = time.time()
    try:
        while step < total_steps:
            t_step_start = time.time()
            t_data_start = time.time()

            # ---- wait for prefetched batch (overlaps with previous step's compute)
            (
                real_idx,
                id_idx,
                triplets,
                meta,
                pos_neg_idx,
                real_pairs,
                real_valid,
                id_pairs,
                id_valid,
                pn_pairs,
                pn_valid,
            ) = prefetch_fut.result()

            # immediately kick off next batch decode (overlaps with this step's compute)
            prefetch_fut = prefetch_pool.submit(_build_one_batch)

            if not real_valid.all() or not id_valid.all() or not pn_valid.all():
                real_pairs = real_pairs[real_valid]
                id_pairs = id_pairs[id_valid]
                pn_pairs = pn_pairs[pn_valid]
                pn_keep = set(pos_neg_idx[i] for i, v in enumerate(pn_valid) if v)
                triplets = [
                    (a, p, n) for (a, p, n) in triplets if p in pn_keep and n in pn_keep
                ]
                if len(real_pairs) == 0:
                    log(f"step {step}: 0 valid real after decode, skipping")
                    step += 1
                    continue

            t_data = time.time() - t_data_start

            v_real = torch.from_numpy(real_pairs).float().to(device) / 255.0
            v_id = torch.from_numpy(id_pairs).float().to(device) / 255.0
            v_pn = torch.from_numpy(pn_pairs).float().to(device) / 255.0
            pn_idx_to_local = {
                orig_idx: local
                for local, orig_idx in enumerate(pos_neg_idx)
                if pn_valid[local]
            }

            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

            t_compute_start = time.time()

            # ---- forward real (encoder + decoder + reparam)
            # Mixed precision: encoder/decoder inside bf16 autocast (heavy matmul),
            # all loss computations forced fp32 (KL, MSE, reparam, cosine, norms).
            with amp_ctx:
                out_real = forward_full(
                    lam_inner, v_real, sample=True, use_ckpt=use_ckpt
                )
            # cast loss inputs to fp32 (recon may still be bf16; gt is fp32 already)
            mse_loss = ((v_real[:, 1:].float() - out_real["recon"].float()) ** 2).mean()
            kl_loss, kl_stats = kl_loss_free_bits(
                out_real["z_mu"], out_real["z_var"], free_bit=kl_free_bit
            )
            # current beta_kl by schedule
            if beta_kl_ramp_end_step > 0 and step < beta_kl_ramp_end_step:
                _frac = step / max(1, beta_kl_ramp_end_step)
                beta_kl = beta_kl_init + (beta_kl_target - beta_kl_init) * _frac
            else:
                beta_kl = beta_kl_target
            L_gen = mse_loss + beta_kl * kl_loss
            z_mu_real = out_real["z_mu"].float()

            # ---- L_id (no reparam, no decoder; identity pairs)
            with amp_ctx:
                out_id = encode_full(lam_inner, v_id, sample=False, use_ckpt=use_ckpt)
            with torch.no_grad():
                z_real_norm_sq = z_mu_real.detach().pow(2).sum(dim=1).mean()
                if s_ema is None:
                    s_ema = z_real_norm_sq.clone()
                else:
                    s_ema = s_ema_alpha * s_ema + (1 - s_ema_alpha) * z_real_norm_sq
            L_id = (out_id["z_mu"].float().pow(2).sum(dim=1).mean()) / (s_ema + 1e-8)

            # ---- L_use (zero z + shuffled z) — every l_use_every steps
            do_l_use = step % l_use_every == 0
            if do_l_use:
                # Reuse encoder outputs from out_real (saves one encoder pass).
                # Decode three ways: real_z, zero_z, shuffled_z.
                # All decoders share v_real[:, 0] (frame_i) as past frame, and the same
                # patches/action_pad output from encoder, so we only re-run decoder.
                B = v_real.shape[0]
                if B >= 2:
                    perm = torch.randperm(B, device=device)
                    while torch.any(perm == torch.arange(B, device=device)):
                        perm = torch.randperm(B, device=device)
                else:
                    perm = torch.zeros(B, dtype=torch.long, device=device)

                z_rep_real = out_real["z_rep"]  # (B, T-1, 1, D)
                z_rep_zero = torch.zeros_like(z_rep_real)
                z_rep_shuf = z_rep_real.index_select(0, perm).detach()
                # patches + decoder
                with amp_ctx:
                    H, W = v_real.shape[2:4]
                    recon_zero = decode_full(
                        lam_inner,
                        out_real["patches"],
                        z_rep_zero,
                        H,
                        W,
                        use_ckpt=use_ckpt,
                    )
                    recon_shuf = decode_full(
                        lam_inner,
                        out_real["patches"],
                        z_rep_shuf,
                        H,
                        W,
                        use_ckpt=use_ckpt,
                    )
                L_rec_real_v = mse_loss.detach()  # already computed
                # fp32 MSE (recon may be bf16 from decoder; cast both sides)
                L_rec_zero = ((v_real[:, 1:].float() - recon_zero.float()) ** 2).mean()
                L_rec_shuf = ((v_real[:, 1:].float() - recon_shuf.float()) ** 2).mean()
                # adaptive margin_use
                if lrec_ema is None:
                    lrec_ema = float(L_rec_real_v.item())
                else:
                    lrec_ema = lrec_ema_alpha * lrec_ema + (1 - lrec_ema_alpha) * float(
                        L_rec_real_v.item()
                    )
                margin_use = max(margin_use_init, margin_use_frac * lrec_ema)
                # hinge: encourage L_rec_zero / shuf  >  L_rec_real + margin_use
                gap_zero = L_rec_zero - mse_loss
                gap_shuf = L_rec_shuf - mse_loss
                L_use = torch.clamp(margin_use - gap_zero, min=0.0) + torch.clamp(
                    margin_use - gap_shuf, min=0.0
                )
                usage_gap_zero = float(gap_zero.detach().item())
                usage_gap_shuf = float(gap_shuf.detach().item())
                L_rec_zero_val = float(L_rec_zero.item())
                L_rec_shuf_val = float(L_rec_shuf.item())
            else:
                L_use = torch.tensor(0.0, device=device)
                usage_gap_zero = float("nan")
                usage_gap_shuf = float("nan")
                L_rec_zero_val = float("nan")
                L_rec_shuf_val = float("nan")
                margin_use = margin_use_init

            # ---- L_gap (z_mu, no reparam, no decoder; triplet)
            with amp_ctx:
                out_pn = encode_full(lam_inner, v_pn, sample=False, use_ckpt=use_ckpt)
            z_pn = out_pn["z_mu"].float()

            anchor_idx_in_real = []
            for a, p, n in triplets:
                anchor_idx_in_real.append(real_idx.index(a) if a in real_idx else None)
            keep = [
                (i, p, n)
                for i, (a, p, n) in enumerate(triplets)
                if anchor_idx_in_real[i] is not None
                and p in pn_idx_to_local
                and n in pn_idx_to_local
            ]
            if len(keep) == 0:
                L_gap = torch.tensor(0.0, device=device)
                same_ep_neg = 0.0
                cross_ds_pos = 0.0
            else:
                a_idx = torch.tensor(
                    [anchor_idx_in_real[i] for i, _, _ in keep],
                    device=device,
                    dtype=torch.long,
                )
                p_local = torch.tensor(
                    [pn_idx_to_local[p] for _, p, _ in keep],
                    device=device,
                    dtype=torch.long,
                )
                n_local = torch.tensor(
                    [pn_idx_to_local[n] for _, _, n in keep],
                    device=device,
                    dtype=torch.long,
                )
                z_a = z_mu_real.index_select(0, a_idx)
                z_p = z_pn.index_select(0, p_local)
                z_n = z_pn.index_select(0, n_local)
                cos_ap = F.cosine_similarity(z_a, z_p, dim=1)
                cos_an = F.cosine_similarity(z_a, z_n, dim=1)
                L_gap = torch.clamp(margin - cos_ap + cos_an, min=0.0).mean()
                same_ep_neg = meta["same_episode_negative_rate"]
                cross_ds_pos = meta["cross_dataset_positive_rate"]

            # ---- w_gap schedule
            if step < warmup_steps:
                w_gap = 0.0
            elif step < ramp_end:
                w_gap = (
                    w_gap_target
                    * (step - warmup_steps)
                    / max(1, ramp_end - warmup_steps)
                )
            else:
                w_gap = w_gap_target

            L_total = L_gen + w_gap * L_gap + w_id * L_id + w_use * L_use

            # ---- backward + DDP + clip + step
            L_total.backward()
            trainable_params = [p for g in param_groups for p in g["params"]]
            all_reduce_grads(trainable_params, world)
            grad_norm_total = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=grad_clip
            )
            if not torch.isfinite(grad_norm_total):
                if is_main:
                    log(
                        f"step {step}: grad_norm_total={grad_norm_total.item()} NON-FINITE, skipping optim.step"
                    )
                optim.zero_grad(set_to_none=True)
            else:
                # snapshot params for update_ratio
                if step % log_every == 0:
                    pre_norms = {
                        g["name"]: [p.detach().clone() for p in g["params"]]
                        for g in param_groups
                    }
                optim.step()

            t_compute = time.time() - t_compute_start

            # ---- logging
            rec_history.append(float(mse_loss.item()))
            step_t = time.time() - t_step_start
            samples_per_step = (B_real + B_id + len(pos_neg_idx)) * world
            samples_per_sec = samples_per_step / max(step_t, 1e-6)

            if step % log_every == 0 or step == total_steps - 1:
                # update_ratio per group
                update_ratios = {}
                grad_norms = {}
                for g in param_groups:
                    pn, gn = _param_group_norms(g["params"])
                    grad_norms[g["name"]] = gn
                    if step == 0 or "pre_norms" not in dir():
                        update_ratios[g["name"]] = 0.0
                    else:
                        # update_ratio = sum_p ||p - p_pre|| / (||p_pre|| + eps)
                        delta_sq = 0.0
                        ref_sq = 0.0
                        for p, ppre in zip(g["params"], pre_norms.get(g["name"], [])):
                            d = (p.detach() - ppre).float()
                            delta_sq += d.pow(2).sum().item()
                            ref_sq += ppre.float().pow(2).sum().item()
                        update_ratios[g["name"]] = float(
                            (delta_sq**0.5) / (ref_sq**0.5 + 1e-8)
                        )

                with torch.no_grad():
                    z_norm_mean = float(z_mu_real.norm(dim=1).mean().item())
                    z_norm_std = float(z_mu_real.norm(dim=1).std().item())

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
                    "L_rec": float(mse_loss.item()),
                    "L_kl": float(kl_loss.item()),
                    "L_gen": float(L_gen.item()),
                    "L_gap": float(L_gap.item())
                    if torch.is_tensor(L_gap)
                    else float(L_gap),
                    "L_id": float(L_id.item()),
                    "L_use": float(L_use.item())
                    if torch.is_tensor(L_use)
                    else float(L_use),
                    "L_total": float(L_total.item()),
                    "w_gap": float(w_gap),
                    "beta_kl": beta_kl,
                    "kl_free_bit": kl_free_bit,
                    "margin_use": float(margin_use),
                    "kl_total": kl_stats["kl_total"],
                    "kl_per_dim_mean": kl_stats["kl_per_dim_mean"],
                    "active_kl_dims": kl_stats["active_kl_dims"],
                    "z_mu_std_mean": kl_stats["z_mu_std_mean"],
                    "z_norm_mean": z_norm_mean,
                    "z_norm_std": z_norm_std,
                    "usage_gap_zero": usage_gap_zero,
                    "usage_gap_shuffle": usage_gap_shuf,
                    "L_rec_zero": L_rec_zero_val,
                    "L_rec_shuffle": L_rec_shuf_val,
                    "valid_triplet_rate": float(meta["valid_triplet_rate"]),
                    "same_episode_negative_rate": same_ep_neg,
                    "cross_dataset_positive_rate": cross_ds_pos,
                    "primitive_balance_entropy": meta["primitive_balance_entropy"],
                    "actual_dataset_mix_real": meta["actual_dataset_mix_real"],
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
                        f"L_rec={rec['L_rec']:.4f} KL={rec['kl_total']:.2f}/active={rec['active_kl_dims']} "
                        f"L_gap={rec['L_gap']:.4f} L_id={rec['L_id']:.4f} L_use={rec['L_use']:.4f} "
                        f"u_gap_z={rec['usage_gap_zero'] if not np.isnan(rec['usage_gap_zero']) else 'na':>6} "
                        f"u_gap_s={rec['usage_gap_shuffle'] if not np.isnan(rec['usage_gap_shuffle']) else 'na':>6} "
                        f"w_gap={w_gap:.4f} z|.|={z_norm_mean:.3f}±{z_norm_std:.3f} "
                        f"step={step_t:.2f}s ({rec['samples_per_sec_per_gpu']}/s/gpu, data={rec['data_time_s']:.2f}s)"
                    )

            # update phase + state
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

            # ---- warmup → ramp gate: refuse to enter L_gap ramp if z latent
            #      already collapsed during warmup. Triggered exactly once at the
            #      first step of ramp (i.e., when step == warmup_steps and we're
            #      about to start applying L_gap pressure).
            if step == warmup_steps and warmup_steps > 0:
                gate_issues = []
                with torch.no_grad():
                    cur_z_norm = float(z_mu_real.detach().norm(dim=1).mean().item())
                # compare against baseline (baseline metrics z_norm)
                if m0_z_norm and cur_z_norm < 0.4 * m0_z_norm:
                    gate_issues.append(
                        f"z_norm {cur_z_norm:.2f} < 0.4 × baseline ({0.4 * m0_z_norm:.2f}) — z collapsed in warmup"
                    )
                # KL too low → posterior collapse
                if kl_stats["kl_total"] < 1.0:
                    gate_issues.append(
                        f"KL_total {kl_stats['kl_total']:.3f} < 1.0 — posterior collapsed"
                    )
                # decoder bypass: usage_gap eval too low
                # (we use the most recent train-time usage_gap as proxy — eval has not run yet here)
                if not np.isnan(usage_gap_zero) and usage_gap_zero < 0.0005:
                    gate_issues.append(
                        f"usage_gap_zero {usage_gap_zero:.4f} < 0.0005 — decoder bypassing z"
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
                        f"usage_gap_zero={usage_gap_zero}",
                        all_ranks=True,
                    )

            # ---- ckpt + eval (rank0 only)
            stop_now = False
            if (step > 0 and step % ckpt_every == 0) or step == total_steps - 1:
                if is_main:
                    ckpt_path = out_dir / "checkpoints" / f"step_{step:06d}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model": {
                                f"lam.{k}": v for k, v in lam_inner.state_dict().items()
                            },
                            "optimizer": optim.state_dict(),
                            "config_path": args.config,
                            "args": vars(args),
                            "trainer": "cdlam_stage1",
                        },
                        ckpt_path,
                    )
                    state["latest_checkpoint"] = str(ckpt_path)
                    log(f"step {step}: saved {ckpt_path}")
                    write_run_state(run_state_path, state)
                if world > 1:
                    dist.barrier()

            if (step > 0 and step % eval_every == 0) or step == total_steps - 1:
                if is_main:
                    eval_path = out_dir / "eval" / f"step_{step:06d}.json"
                    eval_path.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_path = state.get("latest_checkpoint")
                    if ckpt_path is None:
                        log(f"step {step}: no checkpoint yet, skipping eval")
                    else:
                        cad = cfg["trainer"]["cadence"]
                        cmd = [
                            sys.executable,
                            str(REPO / "cdlam_integration/lam/eval.py"),
                            "--pair-index",
                            cad["eval_split_parquet"],
                            "--out",
                            str(eval_path),
                            "--checkpoint",
                            str(ckpt_path),
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
                        reference_checkpoint = cad.get("baseline_checkpoint", None)
                        if reference_checkpoint:
                            cmd += ["--baseline-checkpoint", str(reference_checkpoint)]
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
                                    f"step {step}: eval FAILED rc={res.returncode}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}"
                                )
                            else:
                                ev = json.loads(eval_path.read_text())
                                ev["step"] = step
                                ev["wallclock_s"] = round(time.time() - t_loop_start, 1)
                                ev["eval_time_s"] = round(t_eval, 1)
                                eval_f.write(json.dumps(ev, default=str) + "\n")
                                state["latest_eval"] = str(eval_path)

                                # stop conditions on Stage-1 eval
                                issues = []
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
                                if (
                                    ug.get("zero_p50") is not None
                                    and ug["zero_p50"] <= 0
                                ):
                                    issues.append(
                                        f"usage_gap_zero_p50={ug['zero_p50']:.4f} ≤ 0 — decoder may bypass z"
                                    )
                                if (
                                    ug.get("shuffle_p50") is not None
                                    and ug["shuffle_p50"] <= 0
                                ):
                                    issues.append(
                                        f"usage_gap_shuffle_p50={ug['shuffle_p50']:.4f} ≤ 0"
                                    )

                                log(
                                    f"step {step}: eval | "
                                    f"top1={ret.get('top1')} gap={ret.get('gap')} id_p50={idr.get('p50')} "
                                    f"z_norm={geom.get('z_mu_norm_mean')} eff_rank={geom.get('effective_rank')} "
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
                torch.save(
                    {
                        "step": step,
                        "model": {
                            f"lam.{k}": v for k, v in lam_inner.state_dict().items()
                        },
                        "trainer": "cdlam_stage1",
                    },
                    ckpt_path,
                )
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
    ap.add_argument("--m0-baseline", default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--ramp-end", type=int, default=None)
    ap.add_argument("--w-gap-target", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--batch-real", type=int, default=None)
    ap.add_argument("--batch-id", type=int, default=None)
    ap.add_argument("--triplets", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="disable encoder/decoder grad checkpointing (uses more memory, "
        "but may avoid backward NaN at larger batch sizes)",
    )
    ap.add_argument(
        "--grad-ckpt",
        action="store_true",
        help="force grad ckpt on (default behavior — equivalent to no flag)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
