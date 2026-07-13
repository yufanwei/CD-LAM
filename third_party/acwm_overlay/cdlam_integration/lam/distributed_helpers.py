#!/usr/bin/env python3
"""Distributed helpers for Stage-1 LAM training.

Trainable:
    encoder.transformer_blocks (all 24)
    encoder.out
    fc
    action_prompt

Frozen:
    decoder, patch_up, action_up
    baseline reference LAM (separate copy held on same device)

Loss (per real-batch step):
    L_gen   = MSE(recon, gt) + beta * KL[sum-dim, mean-batch]
    L_id    = mean(||z_id||^2) / s,  s = EMA(mean(||z_reference_real||^2))
    L_gap   = mean(max(0, margin - cos(z_a, z_p) + cos(z_a, z_n)))
    L_norm  = mean((||z_mu|| / (||z_reference|| + eps) - 1)^2)
    L_trust = mean(max(0, d - tau)^2),  d = ||z_mu - sg(z_reference)|| / (||z_reference|| + eps)

    L_total = L_gen + w_gap * L_gap + w_id * L_id + w_norm * L_norm + w_trust * L_trust

w_gap schedule (linear): 0 in [0, warmup); 0..target in [warmup, ramp_end); target after.

Tracking:
    train_metrics.jsonl  (every log_every step)
    eval_metrics.jsonl   (every eval_every step) — runs eval_cdlam_full
    run_state.json       (status, current_step, latest_ckpt, latest_eval, ...)
    checkpoints/step_NNNNNN.pt
    configs/f1_config.yaml.copy

Stop conditions:
    identity_ratio_vs_median p50 > stop.identity_ratio_max
    z_norm ratio outside [stop.z_norm_ratio_lo, stop.z_norm_ratio_hi]
    effective_rank drops > stop.eff_rank_drop_frac_max relative to M0
    L_rec does not drop in stop.l_rec_no_drop_window
    valid_triplet_rate < stop.valid_triplet_min_rate

Usage:
    # dry-run (300 step, smaller batch, separate output dir)
    python cdlam_integration/tools/train_cdlam_f1.py \
        --config cdlam_integration/configs/f1_all_encoder_gen_gap_guard.yaml \
        --out outputs/cdlam_train/F1_dryrun \
        --total-steps 300 --log-every 50 --ckpt-every 300 --eval-every 300 \
        --batch-real 32 --batch-id 4 --triplets 16

    # main F1
    python cdlam_integration/tools/train_cdlam_f1.py \
        --config cdlam_integration/configs/f1_all_encoder_gen_gap_guard.yaml \
        --out outputs/cdlam_train/F1_all_encoder_gen_gap_guard \
        --total-steps 10000
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
from typing import Dict, List, Tuple

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
from cdlam_integration.lam.model_ops import forward_full, encode_full  # noqa: E402


# =============== Sampler ====================================================


class TrainingBatchSampler:
    """Yields per-step batches with three blocks:
    - real:    B_real rows of pair_type=='real', balanced over canonical primitives
                × episodes; used for L_gen + L_norm + L_trust + (anchor for triplet).
    - id:      B_id rows of pair_type=='identity'.
    - triplet: N_triplet (anchor, positive, negative) row triples.
               anchor is one of the real rows (re-uses);
               positive: same primitive, different episode, prefer different dataset;
               negative: different primitive, prefer same episode (or same dataset).
    """

    def __init__(
        self,
        pair_index_path: Path,
        canonical_primitives: List[str],
        p_neg_same_ep: float,
        seed: int = 0,
    ):
        self.df = pd.read_parquet(pair_index_path)
        self.real = self.df[
            (self.df.pair_type == "real") & self.df.eligible_for_lgap
        ].reset_index(drop=True)
        self.id = self.df[self.df.pair_type == "identity"].reset_index(drop=True)
        self.canonical = list(canonical_primitives)
        self.p_neg_same_ep = float(p_neg_same_ep)
        self.rng = np.random.default_rng(seed)

        self.by_prim_ep: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.by_prim: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.real.itertuples(index=False)):
            ep_key = f"{row.dataset}|{row.episode_id}"
            self.by_prim_ep[row.primitive][ep_key].append(idx)
            self.by_prim[row.primitive].append(idx)
        self.primitives = [
            p
            for p in self.canonical
            if p in self.by_prim_ep and len(self.by_prim_ep[p]) >= 2
        ]
        if len(self.primitives) < 2:
            raise RuntimeError(
                f"need >=2 canonical primitives with >=2 episodes; got {self.primitives}"
            )

        # episode -> list of (primitive, idx) for finding same-episode different-primitive negatives
        self.by_episode: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for idx, row in enumerate(self.real.itertuples(index=False)):
            ep_key = f"{row.dataset}|{row.episode_id}"
            self.by_episode[ep_key].append((row.primitive, idx))

    def sample(
        self,
        B_real: int,
        B_id: int,
        N_triplet: int,
    ) -> Tuple[List[int], List[int], List[Tuple[int, int, int]], dict]:
        # ---- real: K primitives x M episodes x N pairs, K*M*N == B_real (best-effort)
        K = min(len(self.primitives), max(2, B_real // 6))
        prims = list(self.rng.choice(self.primitives, size=K, replace=False))
        per_p = max(1, B_real // K)
        real_idx: List[int] = []
        for p in prims:
            ep_list = list(self.by_prim_ep[p].keys())
            n_eps = min(len(ep_list), max(2, per_p // 2))
            chosen_eps = self.rng.choice(ep_list, size=n_eps, replace=False)
            per_e = max(1, per_p // max(1, n_eps))
            for ep in chosen_eps:
                pool = self.by_prim_ep[p][ep]
                n_pick = min(len(pool), per_e)
                pick = self.rng.choice(pool, size=n_pick, replace=False)
                real_idx.extend(pick.tolist())
        # pad / truncate to B_real
        if len(real_idx) > B_real:
            real_idx = list(self.rng.choice(real_idx, size=B_real, replace=False))
        elif len(real_idx) < B_real:
            extras = self.rng.choice(
                len(self.real), size=B_real - len(real_idx), replace=False
            )
            real_idx.extend(extras.tolist())
        real_idx = list(map(int, real_idx))

        # ---- id
        id_idx = list(self.rng.choice(len(self.id), size=B_id, replace=False))
        id_idx = list(map(int, id_idx))

        # ---- triplets: anchor sampled from real_idx
        tri: List[Tuple[int, int, int]] = []
        n_invalid = 0
        for _ in range(N_triplet):
            a = int(self.rng.choice(real_idx))
            a_row = self.real.iloc[a]
            a_prim = a_row.primitive
            a_ds = a_row.dataset
            a_ep = f"{a_row.dataset}|{a_row.episode_id}"

            # positive: same primitive, different episode, prefer different dataset
            other_eps = [ep for ep in self.by_prim_ep[a_prim].keys() if ep != a_ep]
            if not other_eps:
                n_invalid += 1
                continue
            # prefer different dataset
            diff_ds_eps = [ep for ep in other_eps if not ep.startswith(a_ds + "|")]
            if diff_ds_eps and self.rng.random() < 0.6:
                p_ep = str(self.rng.choice(diff_ds_eps))
            else:
                p_ep = str(self.rng.choice(other_eps))
            p_pool = self.by_prim_ep[a_prim][p_ep]
            p = int(self.rng.choice(p_pool))

            # negative: different primitive
            if self.rng.random() < self.p_neg_same_ep and a_ep in self.by_episode:
                # try same episode different primitive
                cand = [
                    idx for (pp, idx) in self.by_episode[a_ep] if pp != a_prim and pp
                ]
                if cand:
                    n = int(self.rng.choice(cand))
                else:
                    other_prims = [pp for pp in self.primitives if pp != a_prim]
                    if not other_prims:
                        n_invalid += 1
                        continue
                    np_prim = str(self.rng.choice(other_prims))
                    n = int(self.rng.choice(self.by_prim[np_prim]))
            else:
                other_prims = [pp for pp in self.primitives if pp != a_prim]
                if not other_prims:
                    n_invalid += 1
                    continue
                np_prim = str(self.rng.choice(other_prims))
                n = int(self.rng.choice(self.by_prim[np_prim]))

            tri.append((a, p, n))

        valid_rate = len(tri) / max(1, N_triplet)
        same_ep_neg = sum(
            1
            for (a, p, n) in tri
            if f"{self.real.iloc[a].dataset}|{self.real.iloc[a].episode_id}"
            == f"{self.real.iloc[n].dataset}|{self.real.iloc[n].episode_id}"
        ) / max(1, len(tri))
        cross_ds_pos = sum(
            1
            for (a, p, n) in tri
            if self.real.iloc[a].dataset != self.real.iloc[p].dataset
        ) / max(1, len(tri))
        prim_counts = pd.Series(
            [self.real.iloc[i].primitive for i in real_idx]
        ).value_counts(normalize=True)
        ent = float(-(prim_counts * np.log(prim_counts + 1e-12)).sum())

        meta = {
            "n_invalid_triplet": n_invalid,
            "valid_triplet_rate": valid_rate,
            "same_episode_negative_rate": same_ep_neg,
            "cross_dataset_positive_rate": cross_ds_pos,
            "primitive_balance_entropy": ent,
            "actual_dataset_mix_real": pd.Series(
                [self.real.iloc[i].dataset for i in real_idx]
            )
            .value_counts(normalize=True)
            .to_dict(),
        }
        return real_idx, id_idx, tri, meta


# =============== Pair decoding ==============================================


def _decode_pair_local(args):
    """Read 2 frames from an mp4 using cv2.grab() to skip non-target frames
    without decoding them (decode is the expensive op). For pair (fi=300, fj=360)
    this turns ~360 decodes into 2 decodes + 358 grab()s, which is much faster.
    """
    video_path, fi, fj, target_hw = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    out = {fi: None, fj: None}
    needed = max(fi, fj)
    i = 0
    try:
        while i <= needed:
            if i in out:
                ok, fr = cap.read()
                if not ok:
                    break
                fr = cv2.resize(
                    fr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA
                )
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                if out[i] is None:
                    out[i] = fr
            else:
                ok = cap.grab()
                if not ok:
                    break
            i += 1
    finally:
        cap.release()
    a = out[fi]
    b = out[fj] if fj != fi else a
    if a is None or b is None:
        return None
    return np.stack([a, b], axis=0).astype(np.uint8)


def decode_rows_parallel(rows: pd.DataFrame, target_hw=(240, 320), workers: int = 16):
    pool = [
        (r.video_path, int(r.frame_i), int(r.frame_j), target_hw)
        for r in rows.itertuples(index=False)
    ]
    out = np.zeros((len(rows), 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
    valid = np.zeros(len(rows), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, arr in enumerate(ex.map(_decode_pair_local, pool)):
            if arr is not None:
                out[idx] = arr
                valid[idx] = True
    return out, valid


# =============== Trainer ====================================================


def collect_param_groups(
    lam_inner: torch.nn.Module, lr_cfg: dict, freeze_blocks_below_idx: int = 0
) -> List[dict]:
    """Layer-wise learning-rate groups.

    If ``freeze_blocks_below_idx > 0``, all encoder.transformer_blocks[i] with
    i < freeze_blocks_below_idx are frozen (requires_grad=False, no param group).
    `encoder.ffn` is also frozen if any early blocks are frozen (it sits before
    block 0 in the encoder forward path; training it while block 0-N are frozen
    would still drift the early features).
    """
    groups: Dict[str, List[torch.nn.Parameter]] = defaultdict(list)

    # encoder blocks (selectively trainable)
    blocks = lam_inner.encoder.transformer_blocks
    for i, blk in enumerate(blocks):
        if i < freeze_blocks_below_idx:
            for p in blk.parameters():
                p.requires_grad = False
            continue
        for p in blk.parameters():
            if 0 <= i <= 7:
                groups["block_0_7"].append(p)
            elif 8 <= i <= 18:
                groups["block_8_18"].append(p)
            elif 19 <= i <= 23:
                groups["block_19_23"].append(p)
    # encoder.out — always trainable (it's the final z projection)
    if hasattr(lam_inner.encoder, "out") and lam_inner.encoder.out is not None:
        for p in lam_inner.encoder.out.parameters():
            groups["encoder_out"].append(p)
    # encoder.ffn (input projection): freeze if any early blocks are frozen
    if hasattr(lam_inner.encoder, "ffn") and lam_inner.encoder.ffn is not None:
        if freeze_blocks_below_idx > 0:
            for p in lam_inner.encoder.ffn.parameters():
                p.requires_grad = False
        else:
            for p in lam_inner.encoder.ffn.parameters():
                groups["block_0_7"].append(p)
    # encoder.pos_enc — usually no learnable params; include defensively
    if hasattr(lam_inner.encoder, "pos_enc"):
        if freeze_blocks_below_idx > 0:
            for p in lam_inner.encoder.pos_enc.parameters():
                p.requires_grad = False
        else:
            for p in lam_inner.encoder.pos_enc.parameters():
                groups["block_0_7"].append(p)
    # fc — always trainable
    for p in lam_inner.fc.parameters():
        groups["fc"].append(p)
    # action_prompt — always trainable
    groups["action_prompt"].append(lam_inner.action_prompt)

    # set requires_grad True for trainables
    trainable_param_ids = set()
    out_groups = []
    for name, params in groups.items():
        if name not in lr_cfg:
            # group exists in code but not in config; skip safely
            continue
        lr = float(lr_cfg[name])
        if lr <= 0:
            # explicit zero LR -> freeze rather than wasting an optimizer slot
            for p in params:
                p.requires_grad = False
            continue
        for p in params:
            p.requires_grad = True
            trainable_param_ids.add(id(p))
        out_groups.append({"params": params, "lr": lr, "name": name})

    # freeze everything else (decoder, patch_up, action_up, plus skipped early blocks)
    for p in lam_inner.parameters():
        if id(p) not in trainable_param_ids:
            p.requires_grad = False

    return out_groups


def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def write_run_state(path: Path, state: dict):
    path.write_text(json.dumps(state, indent=2, default=str))


def init_distributed():
    """Initialize torch.distributed if launched via torchrun. Returns (rank, world, local_rank).

    NCCL ALLREDUCE timeout default ~10min — too short for slow-decode pipelines (e.g.
    ego-centric 1080p mp4 with cold disk cache). Pass `TORCH_NCCL_TIMEOUT_S` env var
    (seconds) to override.
    """
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        if not dist.is_initialized():
            timeout_s = int(os.environ.get("TORCH_NCCL_TIMEOUT_S", "0") or 0)
            kwargs = dict(backend="nccl", rank=rank, world_size=world)
            if timeout_s > 0:
                import datetime

                kwargs["timeout"] = datetime.timedelta(seconds=timeout_s)
            dist.init_process_group(**kwargs)
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def all_reduce_grads(params: list[torch.nn.Parameter], world: int):
    """Average gradients across DDP ranks (manual AVG since not all torch versions support AVG op)."""
    if world <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.mul_(1.0 / world)


def train(args, cfg):
    rank, world, local_rank = init_distributed()
    is_main = rank == 0

    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)
        (out_dir / "configs").mkdir(exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)
        shutil.copy2(args.config, out_dir / "configs" / "f1_config.yaml.copy")
    if world > 1:
        dist.barrier()

    log_path = out_dir / "logs" / f"rank{rank}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg, all_ranks=False):
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

    pid = os.getpid()
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    state = {
        "status": "init",
        "current_step": 0,
        "target_steps": args.total_steps,
        "latest_checkpoint": None,
        "latest_eval": None,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "gpu_ids": gpu_ids,
        "pid": pid,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "pause_reason": None,
        "args": vars(args),
        "config_path": args.config,
    }
    if is_main:
        write_run_state(run_state_path, state)

    # Each rank pinned to its own GPU. Under torchrun + nccl, local_rank maps to cuda:local_rank.
    if world > 1:
        device = f"cuda:{local_rank}"
    else:
        device = cfg["trainer"]["device"]
    target_hw = (int(cfg["trainer"]["target_h"]), int(cfg["trainer"]["target_w"]))
    log(f"device={device}  pid={pid}  gpu_ids={gpu_ids}  world={world}", all_ranks=True)
    log(f"out_dir={out_dir}")

    # ---- batch sizes (CLI overrides config)
    B_real = int(args.batch_real or cfg["trainer"]["batch"]["B_real"])
    B_id = int(args.batch_id or cfg["trainer"]["batch"]["B_id"])
    N_triplet = int(args.triplets or cfg["trainer"]["batch"]["N_triplet"])
    decode_workers = int(cfg["trainer"]["batch"]["decode_workers"])

    # ---- training schedule (CLI overrides)
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

    loss_cfg = cfg["trainer"]["loss"]
    beta_kl = float(loss_cfg["beta_kl"])
    w_gap_target = float(
        args.w_gap_target if args.w_gap_target is not None else loss_cfg["w_gap_target"]
    )
    w_id = float(loss_cfg["w_id"])
    w_norm = float(loss_cfg["w_norm"])
    w_trust = float(loss_cfg["w_trust"])
    margin = float(loss_cfg["margin"])
    tau = float(loss_cfg["tau"])

    log(
        f"steps={total_steps}  warmup={warmup_steps}  ramp_end={ramp_end}  w_gap_target={w_gap_target}"
    )
    log(
        f"B_real={B_real}  B_id={B_id}  N_triplet={N_triplet}  margin={margin}  tau={tau}"
    )
    log(
        f"loss weights: w_id={w_id} w_norm={w_norm} w_trust={w_trust}  beta_kl={beta_kl}"
    )

    # ---- M0 baseline (for stop conditions)
    m0_path = Path(args.m0_baseline) if args.m0_baseline else None
    m0 = json.loads(m0_path.read_text()) if (m0_path and m0_path.exists()) else None
    if m0:
        log(
            f"loaded M0 baseline from {m0_path}: z_norm={m0['z_geometry']['z_mu_norm_mean']:.3f}"
            f"  effective_rank={m0['z_geometry']['effective_rank']:.2f}"
        )
    stop_cfg = cfg["trainer"]["stop"]
    m0_z_norm = m0["z_geometry"]["z_mu_norm_mean"] if m0 else None
    m0_eff_rank = m0["z_geometry"]["effective_rank"] if m0 else None

    # ---- sampler (rank-aware seed so each rank gets different rows per step)
    sampler = TrainingBatchSampler(
        pair_index_path=Path(cfg["trainer"]["pair_dir"]) / "pair_index_train.parquet",
        canonical_primitives=cfg["trainer"]["triplet"]["canonical_primitives"],
        p_neg_same_ep=cfg["trainer"]["triplet"]["p_negative_same_episode"],
        seed=args.seed + 1009 * rank,
    )
    log(
        f"sampler: real={len(sampler.real):,}  identity={len(sampler.id):,}  "
        f"primitives={sampler.primitives}  seed={args.seed + 1009 * rank}"
    )

    # ---- build trainable + frozen reference LAMs
    log("building trainable F1 LAM (init from baseline ckpt) ...")
    lam_train = build_lam("CD_LAM", device=device)
    lam_inner = lam_train.lam
    lam_inner.train()

    log("building frozen baseline reference LAM ...")
    lam_ref = build_lam("CD_LAM", device=device)
    lam_ref.eval()
    freeze_module(lam_ref)

    # ---- freeze decoder + patch_up + action_up
    freeze_module(lam_inner.decoder)
    freeze_module(lam_inner.patch_up)
    freeze_module(lam_inner.action_up)

    # ---- collect param groups for AdamW
    freeze_blocks_below = int(cfg["trainer"].get("freeze_blocks_below_idx", 0))
    param_groups = collect_param_groups(
        lam_inner, cfg["trainer"]["lr"], freeze_blocks_below_idx=freeze_blocks_below
    )
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    n_total = sum(p.numel() for p in lam_inner.parameters())
    log(
        f"trainable params: {n_train:,} / {n_total:,} = {n_train / n_total * 100:.1f}%  "
        f"freeze_blocks_below_idx={freeze_blocks_below}"
    )
    for g in param_groups:
        log(
            f"  {g['name']}: lr={g['lr']:.2e} n_params={sum(p.numel() for p in g['params']):,}"
        )

    optim = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["trainer"]["lr"]["weight_decay"]),
        eps=float(cfg["trainer"]["lr"]["eps"]),
    )

    # ---- EMA scale s for L_id (norm of baseline z on real batch)
    s_ema = None
    s_ema_alpha = 0.95

    rec_history = deque(maxlen=200)  # for L_rec drop check
    state["status"] = (
        "warmup" if warmup_steps > 0 else "ramp" if total_steps > 1 else "smoke"
    )
    write_run_state(run_state_path, state)

    # ---- training loop
    step = 0
    pause_reason = None
    t_loop_start = time.time()
    try:
        while step < total_steps:
            t_step_start = time.time()

            # ---- sample batches
            real_idx, id_idx, triplets, meta = sampler.sample(B_real, B_id, N_triplet)
            real_rows = sampler.real.iloc[real_idx]
            id_rows = sampler.id.iloc[id_idx]

            # decode all unique pairs needed: real + id + triplets (anchor==real already; pos/neg new)
            pos_neg_idx = sorted({i for (_, p, n) in triplets for i in (p, n)})
            pos_neg_rows = sampler.real.iloc[pos_neg_idx]

            real_pairs, real_valid = decode_rows_parallel(
                real_rows, target_hw, decode_workers
            )
            id_pairs, id_valid = decode_rows_parallel(
                id_rows, target_hw, decode_workers
            )
            pn_pairs, pn_valid = decode_rows_parallel(
                pos_neg_rows, target_hw, decode_workers
            )
            if not real_valid.all() or not id_valid.all() or not pn_valid.all():
                # drop invalids
                real_pairs = real_pairs[real_valid]
                id_pairs = id_pairs[id_valid]
                pn_pairs = pn_pairs[pn_valid]
                pn_keep = set(pos_neg_idx[i] for i, v in enumerate(pn_valid) if v)
                triplets = [
                    (a, p, n) for (a, p, n) in triplets if p in pn_keep and n in pn_keep
                ]
                if len(real_pairs) == 0 or len(triplets) == 0:
                    log(f"step {step}: too few valid pairs after decode, skipping")
                    step += 1
                    continue

            # ---- to device
            v_real = torch.from_numpy(real_pairs).float().to(device) / 255.0
            v_id = torch.from_numpy(id_pairs).float().to(device) / 255.0
            v_pn = torch.from_numpy(pn_pairs).float().to(device) / 255.0
            pn_idx_to_local = {
                orig_idx: local
                for local, orig_idx in enumerate(pos_neg_idx)
                if pn_valid[local]
            }

            # ---- forward F1 trainable
            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

            # real with reparam (for L_gen + L_norm + L_trust)
            with amp_ctx:
                out_real = forward_full(lam_inner, v_real, sample=True, use_ckpt=True)
            mse_loss = ((v_real[:, 1:] - out_real["recon"]) ** 2).mean()
            kl_loss = (
                -0.5
                * torch.sum(
                    1
                    + out_real["z_var"]
                    - out_real["z_mu"] ** 2
                    - out_real["z_var"].exp(),
                    dim=1,
                ).mean()
            )
            L_gen = mse_loss + beta_kl * kl_loss

            # ref encoder on same real batch (no grad, no reparam, NO decoder)
            with torch.no_grad(), amp_ctx:
                ref_out = encode_full(lam_ref.lam, v_real, sample=False)
            z_ref = ref_out["z_mu"].float().detach()
            del ref_out  # free patches/z_var/z_rep/z_rep_flat ref tensors
            z_mu_real = out_real["z_mu"].float()

            # L_norm
            n_ref = z_ref.norm(dim=1)
            n_train = z_mu_real.norm(dim=1)
            L_norm = ((n_train / (n_ref + 1e-8) - 1.0) ** 2).mean()

            # L_trust
            d = (z_mu_real - z_ref).norm(dim=1) / (n_ref + 1e-8)
            trust_violation_rate = float((d > tau).float().mean().item())
            L_trust = (torch.clamp(d - tau, min=0.0) ** 2).mean()

            # ---- identity (no reparam, no decoder)
            with amp_ctx:
                out_id = encode_full(lam_inner, v_id, sample=False, use_ckpt=True)
            with torch.no_grad():
                if s_ema is None:
                    s_ema = (n_ref**2).mean().detach().clone()
                else:
                    s_ema = (
                        s_ema_alpha * s_ema
                        + (1 - s_ema_alpha) * (n_ref**2).mean().detach()
                    )
            L_id = (out_id["z_mu"].float().pow(2).sum(dim=1).mean()) / (s_ema + 1e-8)

            # ---- triplet (z_mu, no reparam, no decoder)
            with amp_ctx:
                out_pn = encode_full(lam_inner, v_pn, sample=False, use_ckpt=True)
            z_pn = out_pn["z_mu"].float()

            anchor_idx_in_real = []
            for a, p, n in triplets:
                # find a's position in real_idx (anchor was sampled from real_idx)
                # may have duplicates so take first match
                pos = real_idx.index(a) if a in real_idx else None
                anchor_idx_in_real.append(pos)
            keep = [
                (i, p, n)
                for i, (a, p, n) in enumerate(triplets)
                if anchor_idx_in_real[i] is not None
                and p in pn_idx_to_local
                and n in pn_idx_to_local
            ]
            if len(keep) == 0:
                log(
                    f"step {step}: zero valid triplets after dedup, skipping triplet loss"
                )
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

            L_total = (
                L_gen
                + w_gap * L_gap
                + w_id * L_id
                + w_norm * L_norm
                + w_trust * L_trust
            )

            # ---- backward + step
            L_total.backward()
            # ---- DDP gradient sync (manual, since we don't wrap lam_inner with DDP wrapper —
            #      our forward path is custom encode_full/decode_full, not lam_inner.forward).
            trainable_params = [p for g in param_groups for p in g["params"]]
            all_reduce_grads(trainable_params, world)
            # Tighter clip than training's 5.0 — needed at larger effective batch where
            # bf16 grad accumulation can occasionally hit very large transient norms.
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            # Skip optimizer step if NaN / inf detected (avoid propagating bad weights)
            if not torch.isfinite(grad_norm):
                if is_main:
                    log(
                        f"step {step}: grad_norm={grad_norm.item()} NON-FINITE, skipping optim.step"
                    )
                optim.zero_grad(set_to_none=True)
            else:
                optim.step()

            # ---- logging
            rec_history.append(float(mse_loss.item()))
            step_t = time.time() - t_step_start
            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    z_norm_mean = float(n_train.mean().item())
                    z_norm_std = float(n_train.std().item())
                    z_id_norm = float(out_id["z_mu"].float().norm(dim=1).mean().item())
                rec = {
                    "step": step,
                    "wallclock_s": round(time.time() - t_loop_start, 1),
                    "step_s": round(step_t, 3),
                    "L_rec": float(mse_loss.item()),
                    "L_kl": float(kl_loss.item()),
                    "L_gen": float(L_gen.item()),
                    "L_gap": float(L_gap.item())
                    if torch.is_tensor(L_gap)
                    else float(L_gap),
                    "L_id": float(L_id.item()),
                    "L_norm": float(L_norm.item()),
                    "L_trust": float(L_trust.item()),
                    "L_total": float(L_total.item()),
                    "w_gap": float(w_gap),
                    "beta_kl": beta_kl,
                    "tau": tau,
                    "trust_violation_rate": trust_violation_rate,
                    "z_norm_mean": z_norm_mean,
                    "z_norm_std": z_norm_std,
                    "z_id_norm_mean": z_id_norm,
                    "s_ema": float(s_ema.item()) if s_ema is not None else None,
                    "valid_triplet_rate": float(meta["valid_triplet_rate"]),
                    "same_episode_negative_rate": same_ep_neg,
                    "cross_dataset_positive_rate": cross_ds_pos,
                    "primitive_balance_entropy": meta["primitive_balance_entropy"],
                    "actual_dataset_mix_real": meta["actual_dataset_mix_real"],
                }
                if is_main and train_f is not None:
                    train_f.write(json.dumps(rec) + "\n")
                if is_main:
                    log(
                        f"step {step:5d}/{total_steps}  "
                        f"L_rec={rec['L_rec']:.4f} L_kl={rec['L_kl']:.2f} L_gap={rec['L_gap']:.4f}"
                        f" L_id={rec['L_id']:.4f} L_norm={rec['L_norm']:.4f} L_trust={rec['L_trust']:.4f}"
                        f"  w_gap={w_gap:.4f}  z|.|={z_norm_mean:.3f}±{z_norm_std:.3f}"
                        f"  trust_viol={trust_violation_rate:.2f}  vt={rec['valid_triplet_rate']:.2f}"
                        f"  step={step_t:.2f}s"
                    )

            # ---- update run_state
            state["current_step"] = step
            state["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            if step < warmup_steps:
                state["status"] = "warmup"
            elif step < ramp_end:
                state["status"] = "ramp"
            else:
                state["status"] = "smoke"
            state["latest_train_metrics"] = {
                "step": step,
                "L_rec": float(mse_loss.item()),
                "L_gen": float(L_gen.item()),
                "L_gap": float(L_gap.item())
                if torch.is_tensor(L_gap)
                else float(L_gap),
                "L_id": float(L_id.item()),
                "z_norm_mean": float(n_train.mean().item()),
            }
            if is_main and step % 50 == 0:
                write_run_state(run_state_path, state)

            # ---- ckpt (rank0 only saves)
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
                        },
                        ckpt_path,
                    )
                    state["latest_checkpoint"] = str(ckpt_path)
                    log(f"step {step}: saved {ckpt_path}")
                    write_run_state(run_state_path, state)
                if world > 1:
                    dist.barrier()  # other ranks wait until rank0 finishes ckpt save

            # ---- eval (rank0 only invokes the subprocess; other ranks barrier-wait)
            stop_now = False
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
                            str(REPO / "cdlam_integration/lam/eval_protocol.py"),
                            "--pair-index",
                            cad["eval_split_parquet"],
                            "--out",
                            str(eval_path),
                            "--encoder-mode",
                            "f1",
                            "--ckpt",
                            str(ckpt_path),
                            "--n-pairs-real",
                            str(int(cad["eval_n_pairs_real"])),
                            "--n-pairs-id",
                            str(int(cad["eval_n_pairs_id"])),
                            "--n-per-primitive",
                            str(int(cad["eval_n_per_primitive"])),
                            "--seed",
                            str(args.seed),
                        ]
                        log(f"step {step}: running eval -> {eval_path}")
                        try:
                            # Eval subprocess targets rank0's GPU only. Pin via CUDA_VISIBLE_DEVICES
                            # in env to ensure the subprocess does not see other ranks' GPUs.
                            sub_env = {**os.environ}
                            if world > 1:
                                sub_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
                                sub_env.pop("RANK", None)
                                sub_env.pop("LOCAL_RANK", None)
                                sub_env.pop("WORLD_SIZE", None)
                                sub_env.pop("MASTER_ADDR", None)
                                sub_env.pop("MASTER_PORT", None)
                            res = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=600,
                                env=sub_env,
                            )
                            if res.returncode != 0:
                                log(
                                    f"step {step}: eval FAILED rc={res.returncode}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}"
                                )
                            else:
                                ev = json.loads(eval_path.read_text())
                                ev["step"] = step
                                ev["wallclock_s"] = round(time.time() - t_loop_start, 1)
                                eval_f.write(json.dumps(ev, default=str) + "\n")
                                state["latest_eval"] = str(eval_path)

                                # check stop conditions
                                id_p50 = float(ev["identity_ratio_vs_median"]["p50"])
                                z_norm_mean_ev = float(
                                    ev["z_geometry"]["z_mu_norm_mean"]
                                )
                                eff_rank_ev = float(ev["z_geometry"]["effective_rank"])
                                issues = []
                                if id_p50 > stop_cfg["identity_ratio_max"]:
                                    issues.append(
                                        f"identity_ratio_p50={id_p50:.3f} > {stop_cfg['identity_ratio_max']}"
                                    )
                                if m0_z_norm:
                                    ratio = z_norm_mean_ev / m0_z_norm
                                    if (
                                        ratio < stop_cfg["z_norm_ratio_lo"]
                                        or ratio > stop_cfg["z_norm_ratio_hi"]
                                    ):
                                        issues.append(
                                            f"z_norm_ratio={ratio:.3f} out of [{stop_cfg['z_norm_ratio_lo']}, {stop_cfg['z_norm_ratio_hi']}]"
                                        )
                                if m0_eff_rank:
                                    drop = (m0_eff_rank - eff_rank_ev) / m0_eff_rank
                                    if drop > stop_cfg["eff_rank_drop_frac_max"]:
                                        issues.append(
                                            f"eff_rank_drop={drop:.3f} > {stop_cfg['eff_rank_drop_frac_max']}"
                                        )
                                if rec_history and len(rec_history) > 50:
                                    rec_old = float(np.mean(list(rec_history)[:50]))
                                    rec_new = float(np.mean(list(rec_history)[-50:]))
                                    if (
                                        step > stop_cfg["l_rec_no_drop_window"]
                                        and rec_new > rec_old * 1.05
                                    ):
                                        issues.append(
                                            f"L_rec not dropping: old_avg={rec_old:.4f} new_avg={rec_new:.4f}"
                                        )
                                if (
                                    meta["valid_triplet_rate"]
                                    < stop_cfg["valid_triplet_min_rate"]
                                ):
                                    issues.append(
                                        f"valid_triplet_rate={meta['valid_triplet_rate']:.2f} < {stop_cfg['valid_triplet_min_rate']}"
                                    )

                                log(
                                    f"step {step}: eval | top1={ev['retrieval']['top1']:.3f}"
                                    f" gap={ev['retrieval']['gap']:.4f} same_ep@5={ev['retrieval']['same_episode_share_top5']:.3f}"
                                    f" leakage={ev['retrieval']['dataset_leakage_top5']:.3f}"
                                    f" id_p50={id_p50:.3f} z_norm={z_norm_mean_ev:.3f}"
                                    f" eff_rank={eff_rank_ev:.2f}"
                                )
                                if issues:
                                    log(f"step {step}: STOP CONDITION HIT: {issues}")
                                    state["status"] = "paused"
                                    state["pause_reason"] = "; ".join(issues)
                                    pause_reason = state["pause_reason"]
                                    write_run_state(run_state_path, state)
                                    stop_now = True
                        except subprocess.TimeoutExpired:
                            log(f"step {step}: eval TIMEOUT after 600s")
                        except Exception as exc:
                            log(
                                f"step {step}: eval EXCEPTION {exc}\n{traceback.format_exc()}"
                            )
                # all ranks barrier here so non-main ranks don't race ahead while rank0 does eval
                if world > 1:
                    dist.barrier()
                # broadcast stop_now from rank0 to all ranks (so all ranks break together)
                if world > 1:
                    stop_tensor = torch.tensor(
                        [1 if stop_now else 0], device=device, dtype=torch.int
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop_now = bool(stop_tensor.item())
                if stop_now:
                    break

            step += 1

        # ---- end of loop
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
        # save crash ckpt so user can inspect (rank0 only)
        if is_main:
            try:
                ckpt_path = out_dir / "checkpoints" / f"crash_step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": {
                            f"lam.{k}": v for k, v in lam_inner.state_dict().items()
                        },
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
        "--m0-baseline",
        default=None,
        help="path to M0 baseline JSON; used for stop-condition baselines",
    )
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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
