"""CD-LAM runtime component."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from cdlam_integration.world_model.preprocess import (  # noqa: E402
    decode_video_official,
    decode_window_official,
    official_lam_video_from_wm,
)


def _setup_distributed():
    """Returns (rank, world_size, local_rank). Auto-detects torchrun env."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("LOCAL_RANK", "0")
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size > 1:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    return rank, world_size, local_rank


def _patch_sync_model_states_for_single_rank():
    """The .pt-format checkpoint loader in cosmos_predict2 calls
    `distributed.sync_model_states(model, src=0)` to broadcast weights from rank 0
    to other ranks. With world_size=1 there's nobody to broadcast to, but the call
    still tries to do `_broadcast_coalesced` on CPU tensors and crashes with
    'No backend type associated with device type cpu'. Skip it on single-rank.
    """
    from cosmos_predict2._src.imaginaire.utils import distributed as _dist

    orig = _dist.sync_model_states

    def patched(model, src=0, **kwargs):
        import torch.distributed as td

        ws = td.get_world_size() if td.is_initialized() else 1
        if ws <= 1:
            return  # nothing to sync to
        return orig(model, src=src, **kwargs)

    _dist.sync_model_states = patched


def _decode_video(video: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    """Decode first n_frames via official raw-video protocol.

    Raw frames are center-cropped to 4:3 before resizing to the WM resolution.
    This mirrors `groot_dreams/data/dataset_video.py`, unlike the earlier direct
    resize path.
    """
    return decode_video_official(video, n_frames, wm_hw=(h, w))


def _decode_window(
    video: str, start_frame: int, stop_frame: int, h: int, w: int
) -> np.ndarray:
    """Decode [start_frame, stop_frame) via official raw-video protocol."""
    return decode_window_official(video, start_frame, stop_frame, wm_hw=(h, w))


def _resize_uint8_video(video_np: np.ndarray, h: int, w: int) -> np.ndarray:
    """Derive LAM-resolution video from an already official/cropped WM video."""
    return official_lam_video_from_wm(video_np, lam_hw=(h, w))


def _build_data_batch(
    model, video_np: np.ndarray, action_with_z: torch.Tensor, prompt: str = ""
) -> dict:
    """video_np: (T, H, W, 3) uint8. action_with_z: (B, T_chunk, 384) bf16, last 32D = z*1 (we set placeholder=1)."""
    T, H, W, C = video_np.shape
    img_t = torch.from_numpy(video_np).float() / 255.0  # (T, H, W, 3)
    img_t = img_t.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W) [0,1]

    # WM normalization in `_normalize_video_databatch_inplace` does (x - 0.5) / 0.5 → [-1, 1].
    # We feed [0, 255] uint8 -> WM converts on its own.
    vid_uint8 = (img_t * 255.0).to(torch.uint8)

    data_batch = {
        "dataset_name": "video_data",
        "video": vid_uint8.cuda(),
        "fps": torch.tensor([8], dtype=torch.float).cuda(),
        "padding_mask": torch.zeros(1, 1, H, W).cuda(),
        "num_conditional_frames": 1,
        "action": action_with_z.cuda().to(dtype=torch.bfloat16),
    }

    # T5 (Cosmos-Reason1) text embedding
    if model.text_encoder is not None:
        data_batch["ai_caption"] = [prompt]
        data_batch["t5_text_embeddings"] = (
            model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
        )

    # bf16 cast for floating
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch


def patch_forward_to_use_cached_z(model, z_cached, fix_noise: bool = False):
    """z_cached can be:
    - torch.Tensor (B, T, 32): static, used for every forward (single-video mode)
    - dict-like with key 'current' set to a tensor: swapped per step (multi-video mode)
    """
    """Replace WM forward's inline LAM call by directly inserting our cached z.

    WM forward original:
        lam_video = rearrange(data_batch["lam_video"], "b (p t) h w c -> (b p) t h w c", t=2)
        outputs = self.lam.lam(lam_input)
        latent_action = outputs["z_rep"]  # (B, T_chunk, 32)
        data_batch["action"][:, :, -32:] = data_batch["action"][:, :, -32:] * latent_action

    Our patched version skips the inline LAM and uses z_cached as latent_action directly.

    z_cached: (B, T_chunk, 32) bf16 — our baseline/old LAM z replicated across the chunk.
    fix_noise: if True, freeze epsilon + timestep on first call; reuse in every subsequent call.
    """
    fixed = {"epsilon": None, "timesteps": None}

    def patched_forward(self, data_batch):
        # Inject our z directly. Skip lam.lam(...) entirely.
        # Reproduce the multiplicative blend: action[:, :, -32:] *= z_cached (per-row broadcasted).
        if isinstance(z_cached, dict):
            z_src = z_cached["current"]
        else:
            z_src = z_cached
        z_local = z_src.to(
            device=data_batch["action"].device, dtype=data_batch["action"].dtype
        )
        if z_local.shape[1] != data_batch["action"].shape[1]:
            z_local = z_local.expand(-1, data_batch["action"].shape[1], -1)
        data_batch["action"][:, :, -32:] = data_batch["action"][:, :, -32:] * z_local

        # Now call the rest of original forward but skipping the LAM block.
        # Easiest: copy the post-LAM portion. We re-implement here.
        # Get the input data to noise and denoise + condition
        if (
            self.config.text_encoder_config is not None
            and self.config.text_encoder_config.compute_online
            and "t5_text_embeddings" not in data_batch
        ):
            text_embeddings = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        if fix_noise and fixed["epsilon"] is not None:
            epsilon_B_C_T_H_W = fixed["epsilon"]
            timesteps = fixed["timesteps"]
        else:
            epsilon_B_C_T_H_W = torch.randn(
                x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32
            )
            batch_size = x0_B_C_T_H_W.size()[0]
            t_B = self.rectified_flow.sample_train_time(batch_size).to(
                **self.tensor_kwargs_fp32
            )
            t_B = rearrange(t_B, "b -> b 1")
            x0_B_C_T_H_W_post, condition_post, epsilon_B_C_T_H_W, t_B = (
                self.broadcast_split_for_model_parallelsim(
                    x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
                )
            )
            x0_B_C_T_H_W, condition = x0_B_C_T_H_W_post, condition_post
            timesteps = self.rectified_flow.get_discrete_timestamp(
                t_B, self.tensor_kwargs_fp32
            )
            timesteps = rearrange(timesteps, "b -> b 1")
            if fix_noise:
                fixed["epsilon"] = epsilon_B_C_T_H_W.detach()
                fixed["timesteps"] = timesteps.detach()

        sigmas = self.rectified_flow.get_sigmas(
            timesteps.squeeze(-1), self.tensor_kwargs_fp32
        )
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        time_weights_B = self.rectified_flow.train_time_weight(
            timesteps, self.tensor_kwargs_fp32
        )
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2,
            dim=list(range(1, vt_pred_B_C_T_H_W.dim())),
        )
        per_instance_motion_consistency_loss = torch.mean(
            (
                (vt_pred_B_C_T_H_W[:, 1:] - vt_pred_B_C_T_H_W[:, :-1])
                - (vt_B_C_T_H_W[:, 1:] - vt_B_C_T_H_W[:, :-1])
            )
            ** 2,
            dim=list(range(1, vt_pred_B_C_T_H_W.dim())),
        )
        per_instance_loss = (
            per_instance_loss + per_instance_motion_consistency_loss * 0.1
        )
        loss = torch.mean(time_weights_B * per_instance_loss)
        return {"edm_loss": loss}, loss

    import types

    model.forward = types.MethodType(patched_forward, model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lam-id",
        required=True,
        help="candidate LAM id (registered in lam_registry.yaml + allowed_for_wm_control)",
    )
    ap.add_argument(
        "--ckpt",
        default=os.environ.get(
            "CDLAM_BASE_2B_CKPT",
            str(REPO / "lammodel/checkpoints/CD-LAM/2B_pretrain/iter_000140000/model"),
        ),
        help="path to <iter_dir>/model dir (must contain .metadata). "
        "Default = 2B_pretrain. PR-3 had used 2B_AgiBot_post-train as default which was "
        "wrong for formal C0 calibration semantics.",
    )
    ap.add_argument(
        "--experiment",
        default="dreamdojo_2b_480_640_pretrain",
        help="Hydra experiment name. Default = pretrain (matches ckpt).",
    )
    ap.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    ap.add_argument(
        "--input-video",
        default=None,
        help="single-video mode: required iff --multi-video-dir is not set",
    )
    ap.add_argument(
        "--multi-video-dir",
        default=None,
        help="multi-video mode: directory containing source mp4s. Each step samples one video. "
        "If used together with --z-cache, z is read from parquet by (video_id, frame_i).",
    )
    ap.add_argument(
        "--multi-video-glob",
        default="*/head_color.mp4",
        help="glob inside --multi-video-dir to enumerate videos (default = AgiBot scene/head_color.mp4)",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=10,
        help="cap on # videos sampled in multi-video mode (smoke uses 10).",
    )
    ap.add_argument(
        "--scope",
        default=None,
        choices=["A", "A2", "B", "B_old", "B2", "E", "C", "D"],
        help="trainable scope. PR-7 default = B = embedder + net.blocks[N-4:N] (true last 4). "
        "If None, falls back to PR-3/4 behavior of embedder-only.",
    )
    ap.add_argument(
        "--num-video-frames", type=int, default=13, help="num_action_per_chunk + 1"
    )
    ap.add_argument("--num-action-per-chunk", type=int, default=12)
    ap.add_argument("--action-dim", type=int, default=384)
    ap.add_argument(
        "--resolution",
        default="480,640",
        help="Spatial resolution for WM video. Official raw-video path is 4:3 crop -> 480,640; "
        "use a different value only as an explicit ablation.",
    )
    ap.add_argument(
        "--lam-resolution",
        default="240,320",
        help="Spatial resolution used only for LAM z extraction. Keep WM --resolution full-res; "
        "the upstream LAM/video dataloaders use 240,320 for lam_video.",
    )
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="multi-video mode: stack N videos into a batch per step (B=batch_size).",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear warmup from 0 to --lr over this many steps. Default 0 = no warmup.",
    )
    ap.add_argument(
        "--lr-end-frac",
        type=float,
        default=1.0,
        help="Cosine decay end fraction of --lr. Default 1.0 = no decay (constant). "
        "e.g. 0.1 = cosine decay from --lr to 0.1*--lr over [warmup_steps, total_steps].",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--fix-noise",
        action="store_true",
        help="Fix epsilon AND timestep across all steps so one-batch overfit becomes a "
        "deterministic-target regression. Without this, rectified flow resamples noise "
        "+ sigma each step → loss is a stochastic moving target.",
    )
    ap.add_argument(
        "--save-action-embedder",
        action="store_true",
        help="After training, save `action_embedder_B_D` + `action_embedder_B_3D` "
        "state_dict to <out>/action_embedder.pt for downstream rollout use.",
    )
    ap.add_argument(
        "--ckpt-save-every",
        type=int,
        default=0,
        help="if >0, save full trainable subset every N steps to <out>/ckpt_step{N}.pt",
    )
    ap.add_argument(
        "--init-trainable-from",
        default=None,
        help="Path to a prior ckpt_stepN.pt. After configure_scope, load its "
        "`trainable_state` into model.net (stripping FSDP `_checkpoint_wrapped_module` "
        "prefix). Optimizer state NOT loaded — opt restarts from --lr.",
    )
    ap.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Initial step counter. Useful when resuming: loop runs range(start_step, steps) "
        "so per-rank seeds match the original trajectory and ckpt filenames use global step.",
    )
    ap.add_argument(
        "--z-cache",
        default=None,
        help="Optional path to z_cache parquet. If set, instead of running LAM on the "
        "input video at runtime, lookup (video_id, frame_i) entries from the parquet. "
        "Enables exact reproducibility across runs.",
    )
    ap.add_argument(
        "--train-manifest",
        default=None,
        help="Manifest parquet built by build_wm_lam54_manifest.py. Each row = one "
        "13-frame window with explicit (video_path, start_frame, stop_frame, fps, split). "
        "If set, trainer ignores --input-video / --multi-video-dir and samples rows from "
        "this manifest per step, decoding the [start, stop) window via pyav seek and "
        "computing z online from the LAM encoder (kept alive across steps).",
    )
    ap.add_argument(
        "--manifest-split",
        default="train",
        choices=["train", "heldout"],
        help="which split to use from the manifest (default: train).",
    )
    ap.add_argument(
        "--manifest-audit-all-ranks",
        action="store_true",
        help="manifest mode only: gather sampled row/window ids from every rank "
        "into rank0 train_log.jsonl for sampling-audit. This is intentionally "
        "off by default for old runs; release wrapper enables it.",
    )
    ap.add_argument(
        "--manifest-random-window",
        action="store_true",
        help="manifest mode only: treat each row as a full clean clip with "
        "`clip_nframes`, and sample a random 13-frame window inside it "
        "each time the row is drawn. This matches the upstream VideoDataset "
        "semantics more closely than fixed stride windows.",
    )
    # ---- Stage-2: drive the WM's LAM-z slot from REAL ACTION via trained g_r ----
    ap.add_argument(
        "--z-source",
        default="encode",
        choices=["encode", "gr-action"],
        help="Where the 32-D LAM z (injected into action[:, :, -32:]) comes from. "
        "'encode' (default, UNCHANGED): z = LAM.encode(adjacent frame pairs). "
        "'gr-action' (Stage-2 novelty): z = g_r(REAL_ACTION) un-normalized into "
        "the LAM encoder z space, via cdlam_integration.world_model.gr_zsource. Requires "
        "--gr-ckpt + --gr-robot AND a real-action source aligned to each video "
        "chunk (see --gr-action-cache / the alignment-gap note in main()).",
    )
    ap.add_argument(
        "--gr-ckpt",
        default=os.environ.get(
            "CDLAM_GR_BRIDGE_CKPT",
            str(REPO / "outputs/bridge/decoder.pt"),
        ),
        help="gr-action only: joint ckpt with heads[robot]['g'] (the trained g_r).",
    )
    ap.add_argument(
        "--gr-robot",
        default="agibot_beta_ee_FULL",
        help="gr-action only: which robot head to use (key into ckpt['heads']).",
    )
    ap.add_argument(
        "--gr-action-cache",
        default="outputs/cdlam_data/agibot_beta_ee_cache.npz",
        help="gr-action (LEGACY stub path) only: per-robot .npz used by the joint run; "
        "supplies a_mean/a_std (from ee_delta[train]) and frames to recompute zm/zsd. "
        "Only used when --gr-aligned is NOT set (the old un-aligned stub). With "
        "--gr-aligned, all stats come from the self-contained calibrated g_r* ckpt.",
    )
    # ---- Stage-2 ALIGNED real-action path (replaces the un-aligned stub) ----
    ap.add_argument(
        "--gr-aligned",
        action="store_true",
        help="gr-action only: use the ALIGNED real-action source. Loads agibot episode "
        "windows (frames + matching stride-1 ee_delta) from robot_pairs_10h via "
        "gr_aligned_action, so z_t = g_r*(real_action_t) drives frame_t->frame_{t+1}. "
        "Requires --gr-cal-ckpt (the calibrated g_r* in gr_retrained/). Overrides "
        "the single/multi/manifest video loaders with the aligned-window sampler.",
    )
    ap.add_argument(
        "--gr-cal-ckpt",
        default=os.environ.get(
            "CDLAM_GR_CALIBRATION_CKPT",
            str(REPO / "outputs/bridge/action_encoder.pt"),
        ),
        help="gr-aligned only: the retrained, self-contained calibrated g_r* ckpt "
        "(g_state + action_mean/std + zm/zsd + center + noise_std).",
    )
    ap.add_argument(
        "--gr-aligned-robot",
        default="agibot_beta",
        choices=["agibot_beta"],
        help="gr-aligned only: which agibot robot's episodes to sample from robot_pairs_10h. "
        "Only agibot_beta carries the EE-action h5 fields ee_delta() needs "
        "(agibotworld_alpha's row-group h5 has no EE action stream).",
    )
    ap.add_argument(
        "--gr-aligned-split",
        default="train",
        choices=["train", "test"],
        help="gr-aligned only: robot_pairs_10h split to sample episodes from.",
    )
    ap.add_argument(
        "--gr-aligned-max-episodes",
        type=int,
        default=None,
        help="gr-aligned only: cap on # episodes indexed (None = all). Smoke uses e.g. 40.",
    )
    ap.add_argument(
        "--gr-noise",
        action="store_true",
        help="gr-aligned only: add the calibration residual gaussian (eps~N(0,noise_std^2)) "
        "to z (matches the trainer's z_cal sampling). Default off = deterministic z.",
    )
    ap.add_argument("--out", required=True)
    # ---- condition-dropout + z-usage monitor (ACWM "make it listen"; default OFF = unchanged) ----
    ap.add_argument(
        "--cond-dropout",
        type=float,
        default=0.0,
        help="condition-dropout probability p (AdaWorld ucg_rate). 0 = OFF (default, "
        "behavior unchanged). ~0.10-0.12 forces the WM to read z by zeroing the z "
        "slot on p-fraction of rows each step (legal z=0 null). In this compat "
        "trainer z is fed ONLY via z_holder['current'] (action[-32:] is a 1.0 "
        "placeholder, multiplicative blend) and NO raw [147:169] slot is fed, so "
        "clearing z_holder rows is sufficient (raw_slot=None, no bypass).",
    )
    ap.add_argument(
        "--cond-dropout-seed",
        type=int,
        default=777,
        help="seed for the per-step dropout mask (rank-distinct internally).",
    )
    ap.add_argument(
        "--monitor",
        action="store_true",
        help="enable the z-usage online monitor (action_gap/swap-Δ/future-leak). "
        "Runs a MANDATORY step-0 baseline then re-evaluates every --monitor-every "
        "steps on a fixed eval batch with fix-noise. Writes <out>/z_usage.jsonl.",
    )
    ap.add_argument(
        "--monitor-every",
        type=int,
        default=500,
        help="monitor cadence in steps (default 500).",
    )
    ap.add_argument(
        "--monitor-only",
        action="store_true",
        help="OFFLINE z-usage eval: load (base + --init-trainable-from) ckpt, build a "
        "FIXED eval batch (decoded once, frozen), run action_gap/swap/future probes "
        "with fix-noise pinned INSIDE the patched forward, write <out>/action_gap.json, "
        "and EXIT. No training, no optimizer, single GPU. Used by wm_ft_eval (A).",
    )
    ap.add_argument(
        "--monitor-batch",
        type=int,
        default=8,
        help="eval batch size for --monitor-only (default 8).",
    )
    ap.add_argument(
        "--monitor-repeats",
        type=int,
        default=6,
        help="fix-noise repeats per probe for --monitor-only noise floor (default 6).",
    )
    args = ap.parse_args()

    # --cfg lives at inference (c0_rollout CFG sweep), NOT in this training entrypoint; it is
    # gated by assert_cfg_order(dropout_steps_done>=2000). Documented in launch header + plan §3.
    if args.monitor and not args.fix_noise:
        # action_gap is swamped by rectified-flow noise without fixed epsilon/timestep.
        print(
            "[c0_real][WARN] --monitor without --fix-noise: action_gap will be noise-dominated. "
            "The monitor pins its OWN eval seed, but the trainer's forward fix-noise flag should "
            "be on for a clean step-0 baseline.",
            flush=True,
        )

    if args.gr_aligned and args.z_source != "gr-action":
        sys.exit("[c0_real] --gr-aligned requires --z-source gr-action")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = (int(x) for x in args.resolution.split(","))
    LAM_H, LAM_W = (int(x) for x in args.lam_resolution.split(","))

    rank, world_size, local_rank = _setup_distributed()
    is_rank0 = rank == 0
    if is_rank0:
        print(
            f"[c0_real] rank={rank}/{world_size} local_rank={local_rank}  "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )
    _patch_sync_model_states_for_single_rank()

    # ---- ACL gate
    from cdlam_integration.world_model.registry import assert_allowed_for_wm

    assert_allowed_for_wm(args.lam_id)

    # ---- Stage-2 z-source: z = g_r(real action) instead of z = encode(frame pairs).
    # Default ('encode') leaves ALL behavior below unchanged. 'gr-action' loads g_r + the
    # per-cache norm stats and overrides z_holder['current'] per step (see _apply_gr_z below).
    #
    # TWO sub-paths under --z-source gr-action:
    #   (1) --gr-aligned (NEW, faithful Stage-2): the aligned-window sampler in gr_aligned_action
    #       serves, per step, the 13-frame agibot window AND its matching stride-1 ee_delta(12,20),
    #       so z_t = g_r*(real_action_t) causally drives frame_t->frame_{t+1}. Uses the calibrated
    #       self-contained g_r* (--gr-cal-ckpt). Bypasses the single/multi/manifest video loaders.
    #   (2) legacy stub (no --gr-aligned): draws a (B,T,20) real-action batch from the cache's
    #       ee_delta, NOT aligned to the displayed video. Kept for back-compat / smoke only.
    gr_src = None  # legacy joint-ckpt g_r path
    gr_cal = None  # NEW calibrated g_r* (gr_aligned_action.GrCalibrated)
    gr_aligned_mode = False
    aligned_src = None  # gr_aligned_action.AlignedActionSource
    if args.z_source == "gr-action":
        # Correctness guard: zm/zsd that un-normalize g_r's output are computed against the
        # JOINT LAM's encoder, so the WM's encode-z slot is only interchangeable when the WM
        # runs with that SAME LAM (registry id cdlam_lam -> this ckpt). Warn loudly if not.
        if args.lam_id != "cdlam_lam":
            print(
                f"[c0_real][WARN] --z-source gr-action with --lam-id {args.lam_id!r}: g_r's z is in "
                f"the joint LAM (cdlam_lam) z space; pairing with a different LAM is a "
                f"z-space MISMATCH. Use --lam-id cdlam_lam for a faithful Stage-2 run.",
                flush=True,
            )

        if args.gr_aligned:
            # ---- (1) ALIGNED real-action Stage-2 path ----
            gr_aligned_mode = True
            from cdlam_integration.world_model.gr_aligned_action import (
                load_gr_calibrated,
                build_aligned_source,
            )

            gr_cal = load_gr_calibrated(args.gr_cal_ckpt, device="cuda")
            aligned_src = build_aligned_source(
                robot=args.gr_aligned_robot,
                split=args.gr_aligned_split,
                min_frames=args.num_video_frames + 1,
                max_episodes=args.gr_aligned_max_episodes,
            )
            if aligned_src.n_episodes == 0:
                sys.exit(
                    f"[c0_real] gr-aligned: 0 episodes for {args.gr_aligned_robot}/"
                    f"{args.gr_aligned_split} with >= {args.num_video_frames + 1} frames"
                )
            _gr_aln_rng = np.random.default_rng(args.seed + 1313 + rank)
            if is_rank0:
                print(
                    f"[c0_real] z-source=gr-action ALIGNED: calibrated g_r* robot={gr_cal.robot} "
                    f"from {args.gr_cal_ckpt}; {aligned_src.n_episodes} {args.gr_aligned_robot}/"
                    f"{args.gr_aligned_split} episodes; add_noise={args.gr_noise}",
                    flush=True,
                )
        else:
            # ---- (2) LEGACY un-aligned stub (back-compat) ----
            from cdlam_integration.world_model.gr_zsource import (
                load_gr_zsource,
                build_z_from_action_batch,
            )

            gr_src = load_gr_zsource(
                args.gr_ckpt, args.gr_robot, args.gr_action_cache, device="cuda"
            )
            if is_rank0:
                print(
                    "[c0_real][WARN] z-source=gr-action WITHOUT --gr-aligned: using the LEGACY "
                    "un-aligned stub (cache ee_delta, NOT video-aligned). Pass --gr-aligned for a "
                    "faithful Stage-2 run.",
                    flush=True,
                )
                print(
                    f"[c0_real] loaded g_r robot={args.gr_robot} from {args.gr_ckpt}; "
                    f"a_mean/a_std/zm/zsd from {args.gr_action_cache}",
                    flush=True,
                )
            _gr_cache = np.load(args.gr_action_cache, allow_pickle=True)
            _gr_ee = _gr_cache["ee_delta"].astype(np.float32)  # (N, 20) real action
            _gr_rng = np.random.default_rng(args.seed + 777 + rank)

            def _gr_sample_action(bs_local: int, T_local: int) -> torch.Tensor:
                """STUB real-action source: (bs, T, 20) drawn from the cache (NOT video-aligned)."""
                out = np.empty((bs_local, T_local, _gr_ee.shape[1]), dtype=np.float32)
                for bi in range(bs_local):
                    start = int(_gr_rng.integers(0, max(1, len(_gr_ee) - T_local)))
                    out[bi] = _gr_ee[start : start + T_local]
                return torch.from_numpy(out).cuda()

            def _apply_gr_z(bs_local: int, T_local: int):
                """Override z_holder['current'] with z = g_r(real action), (bs, T, 32)."""
                a_b = _gr_sample_action(bs_local, T_local)  # (bs, T, 20)
                z_b = build_z_from_action_batch(
                    a_b, gr_src.g_r, gr_src.zm, gr_src.zsd, gr_src.a_mean, gr_src.a_std
                )  # (bs, T, 32)
                z_holder["current"] = z_b.to(dtype=torch.bfloat16)

    # ---- decide single-video vs multi-video vs manifest vs gr-aligned mode
    # gr_aligned_mode takes precedence over all video loaders: it owns its own (frames + action)
    # sampler from robot_pairs_10h episode row-groups.
    manifest_mode = (args.train_manifest is not None) and not gr_aligned_mode
    multi_video_mode = (
        (not manifest_mode)
        and (not gr_aligned_mode)
        and (args.multi_video_dir is not None)
    )
    if gr_aligned_mode:
        if args.fix_noise:
            sys.exit(
                "[c0_real] --fix-noise + --gr-aligned is incoherent (gr-aligned samples episodes)"
            )
        # Probe shape by decoding the first episode's window [0, num_video_frames).
        from cdlam_integration.world_model.gr_aligned_action import (
            decode_window_and_action as _aln_decode,
        )

        _ep0 = aligned_src.episodes[0]
        _raw0, _act0 = _aln_decode(aligned_src, _ep0, 0, args.num_video_frames)
        from cdlam_integration.world_model.preprocess import (
            official_wm_video_from_raw as _wm_from_raw,
        )

        video_np = _wm_from_raw(_raw0, wm_hw=(H, W))  # (T, H, W, 3) uint8
        lam_np = _resize_uint8_video(video_np, LAM_H, LAM_W)
        videos_np = [(Path(f"{args.gr_aligned_robot}:ep0"), video_np, lam_np)]
        if is_rank0:
            print(
                f"[c0_real] gr-aligned mode: probe window video={video_np.shape} "
                f"aligned_action={_act0.shape}",
                flush=True,
            )
    elif manifest_mode:
        if args.fix_noise:
            sys.exit(
                "[c0_real] --fix-noise + --train-manifest is incoherent (fix-noise is single-video overfit)"
            )
        if args.input_video is not None or args.multi_video_dir is not None:
            print(
                "[c0_real] WARN: --input-video / --multi-video-dir ignored in manifest mode",
                flush=True,
            )
        import pandas as _pd

        df_manifest = _pd.read_parquet(args.train_manifest)
        df_manifest = df_manifest[
            df_manifest["split"] == args.manifest_split
        ].reset_index(drop=True)
        manifest_rows = df_manifest.to_dict(orient="records")
        if not manifest_rows:
            sys.exit(
                f"[c0_real] manifest {args.train_manifest} has 0 rows for split={args.manifest_split}"
            )
        if is_rank0:
            ds_counts = df_manifest["dataset"].value_counts().to_dict()
            print(
                f"[c0_real] manifest mode: {len(manifest_rows)} '{args.manifest_split}' rows  "
                f"({ds_counts}) from {args.train_manifest}",
                flush=True,
            )
        # Probe shape by decoding the first row's window.
        _r0 = manifest_rows[0]
        video_np = _decode_window(
            _r0["video_path"], int(_r0["start_frame"]), int(_r0["stop_frame"]), H, W
        )
        lam_np = _resize_uint8_video(video_np, LAM_H, LAM_W)
        videos_np = [
            (Path(_r0["video_path"]), video_np, lam_np)
        ]  # for parity with downstream shape uses
    elif multi_video_mode:
        if args.input_video is not None:
            print(
                "[c0_real] WARN: --input-video ignored in multi-video mode", flush=True
            )
        if args.fix_noise:
            sys.exit(
                "[c0_real] --fix-noise + --multi-video-dir is incoherent (fix-noise is only for single-video overfit)"
            )
        # Enumerate videos (cap by --max-videos), pre-decode first chunk_T frames each.
        video_dir = Path(args.multi_video_dir)
        all_videos = sorted(video_dir.glob(args.multi_video_glob))[: args.max_videos]
        if not all_videos:
            sys.exit(f"[c0_real] no videos under {video_dir}/{args.multi_video_glob}")
        print(
            f"[c0_real] multi-video mode: {len(all_videos)} videos, max={args.max_videos}",
            flush=True,
        )
        videos_np = []
        for v in all_videos:
            try:
                arr = _decode_video(v, args.num_video_frames, H, W)
                lam_arr = _resize_uint8_video(arr, LAM_H, LAM_W)
                videos_np.append((v, arr, lam_arr))
            except Exception as e:
                print(f"[c0_real]   skip {v.name}: {e}", flush=True)
        print(f"[c0_real] decoded {len(videos_np)} videos OK", flush=True)
        # Use first video as the "shape probe" for downstream sizes.
        video_np = videos_np[0][1]
    else:
        if args.input_video is None:
            sys.exit(
                "[c0_real] need --input-video OR --multi-video-dir OR --train-manifest"
            )
        video_np = _decode_video(Path(args.input_video), args.num_video_frames, H, W)
        lam_np = _resize_uint8_video(video_np, LAM_H, LAM_W)
        videos_np = [(Path(args.input_video), video_np, lam_np)]
    print(
        f"[c0_real] video shape={video_np.shape} dtype={video_np.dtype}; "
        f"lam_resolution=({LAM_H},{LAM_W})",
        flush=True,
    )

    # ---- compute z for each adjacent pair via LAM (build on-the-fly: simpler than aligning cache)
    from cdlam_integration.lam.encoder import pair_uint8_to_float
    from cdlam_integration.world_model.model_loader import build_encoder

    # Build z for each video: (path, z_tensor (1, T_chunk, 32))
    enc, _ = build_encoder(args.lam_id, device="cuda")

    def _build_z_for(arr: np.ndarray) -> torch.Tensor:
        zs = []
        with torch.no_grad():
            for t in range(args.num_action_per_chunk):
                pair = arr[t : t + 2]
                pair_t = torch.from_numpy(pair[None]).cuda()
                pf = pair_uint8_to_float(pair_t)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z = enc(pf).float()
                zs.append(z)
        return torch.stack(zs, dim=1)

    def _build_z_batch(arrs_list: list, pair_chunk: int = 32) -> torch.Tensor:
        """Vectorized LAM encode: batch ALL (bs * T) pairs into chunks of pair_chunk
        instead of per-pair bs=1 forward. Returns (bs, T, 32) bf16.

        Cost: was bs*T forward calls at bs=1; now ceil(bs*T / pair_chunk) calls.
        For bs=12, T=12 → 144 pairs → 5 chunks at pair_chunk=32 (28× fewer calls).
        Empirically ~3.5× LAM speedup (user-validated config, 2026-05-16).
        """
        bs = len(arrs_list)
        T = args.num_action_per_chunk
        # Stack ALL pairs across all videos: (bs*T, 2, H, W, 3)
        all_pairs_np = np.empty((bs * T, 2) + arrs_list[0].shape[1:], dtype=np.uint8)
        for bi, arr in enumerate(arrs_list):
            for t in range(T):
                all_pairs_np[bi * T + t] = arr[t : t + 2]
        pairs_t = torch.from_numpy(all_pairs_np).cuda(non_blocking=True)
        pf = pair_uint8_to_float(pairs_t)
        zs_list = []
        with torch.no_grad():
            for i in range(0, bs * T, pair_chunk):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_c = enc(pf[i : i + pair_chunk]).float()
                zs_list.append(z_c)
        z_flat = torch.cat(zs_list, dim=0)  # (bs*T, 32)
        return z_flat.reshape(bs, T, 32)

    # ---- Cache T5 emb (constant empty prompt — recomputing every step wastes ~100 ms)
    _t5_cache: dict = {"emb": None}

    def _get_t5_emb_cached(bs):
        if _t5_cache["emb"] is None or _t5_cache["emb"].shape[0] != bs:
            _t5_cache["emb"] = model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [args.prompt] * bs, "images": None},
                input_caption_key="ai_caption",
            )
        return _t5_cache["emb"]

    if gr_aligned_mode:
        # z comes from the calibrated g_r*, NOT the LAM encoder. Build a probe z from the
        # aligned action so patch_forward init has a valid (1,T,32) shape; free the encoder.
        from cdlam_integration.world_model.gr_aligned_action import (
            calibrated_z_from_action as _cal_z,
        )

        _a0_t = torch.from_numpy(_act0).cuda()  # (T-1, 20)
        z_cached = (
            _cal_z(_a0_t, gr_cal, add_noise=False).unsqueeze(0).float()
        )  # (1, T-1, 32)
        videos_with_z = []  # not used in aligned mode
        if is_rank0:
            print(
                f"[c0_real] gr-aligned mode: probe z from g_r* (shape={tuple(z_cached.shape)} "
                f"mean_norm={z_cached.norm(dim=-1).mean().item():.4f}); LAM encoder freed",
                flush=True,
            )
        del enc
        torch.cuda.empty_cache()
    elif manifest_mode:
        # In manifest mode we DON'T pre-build z for the whole dataset (would be ~33k entries).
        # We build z online per step. Keep `enc` alive on GPU across the training loop.
        z_cached = _build_z_for(
            videos_np[0][2]
        )  # probe z to satisfy patch_forward_to_use_cached_z init
        videos_with_z = []  # placeholder; not used in manifest mode
        if is_rank0:
            print(
                f"[c0_real] manifest mode: probe z built (shape={tuple(z_cached.shape)} "
                f"mean_norm={z_cached.norm(dim=-1).mean().item():.4f}); encoder kept alive",
                flush=True,
            )
        # encoder stays on cuda; not deleted.
    else:
        videos_with_z = []
        for vp, arr, lam_arr in videos_np:
            z = _build_z_for(lam_arr)
            videos_with_z.append((vp, arr, z))
        z_cached = videos_with_z[0][
            2
        ]  # for patch_forward init; will be swapped per-step in multi-video mode
        print(
            f"[c0_real] built z for {len(videos_with_z)} videos; z[0] shape={tuple(z_cached.shape)} "
            f"mean_norm={z_cached.norm(dim=-1).mean().item():.4f}",
            flush=True,
        )
        del enc
        torch.cuda.empty_cache()

    # ---- load WM (heavy)
    print(f"[c0_real] loading WM (experiment={args.experiment})...", flush=True)
    t0 = time.time()
    from cosmos_predict2._src.predict2.utils.model_loader import (
        load_model_from_checkpoint,
    )

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.ckpt,
        config_file=args.config_file,
        load_ema_to_reg=True,
        skip_load_model=False,
    )
    print(
        f"[c0_real] WM loaded in {time.time() - t0:.1f}s; net params = "
        f"{sum(p.numel() for p in model.net.parameters()) / 1e9:.3f}B",
        flush=True,
    )

    # ---- monkey-patch forward to use cached z, skip inline LAM
    z_holder = {"current": z_cached}
    patch_forward_to_use_cached_z(model, z_holder, fix_noise=args.fix_noise)
    # DDP wrap is moved AFTER configure_scope, see below.

    # ---- configure trainable scope
    n_total = sum(p.numel() for p in model.parameters())
    if args.scope is None:
        # Legacy PR-3/4 behavior: embedder-only (= scope A).
        for n, p in model.named_parameters():
            p.requires_grad = "action_embedder_B_D" in n or "action_embedder_B_3D" in n
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_params = [p for p in model.parameters() if p.requires_grad]
        print(
            f"[c0_real] (legacy) trainable: embedder-only, {n_train / 1e6:.2f}M",
            flush=True,
        )
    else:
        # Use canonical scope helper from scope_ablation.py.
        # Note: this MUST be called BEFORE patch_forward_to_use_cached_z because
        # configure_scope freezes all params first; patch wraps forward not params.
        from cdlam_integration.world_model.scope import configure_scope

        train_params, n_train = configure_scope(model, args.scope)
        N = len(model.net.blocks)
        if args.scope == "B":
            print(
                f"[c0_real] scope=B (true last 4 = blocks[{N - 4}:{N}]) + embedder, "
                f"{n_train / 1e6:.2f}M / {n_total / 1e9:.3f}B",
                flush=True,
            )
        elif args.scope == "B2":
            print(
                f"[c0_real] scope=B2 (true last 8 = blocks[{N - 8}:{N}]) + embedder, "
                f"{n_train / 1e6:.2f}M / {n_total / 1e9:.3f}B",
                flush=True,
            )
        else:
            print(
                f"[c0_real] scope={args.scope}, trainable {n_train / 1e6:.2f}M / {n_total / 1e9:.3f}B",
                flush=True,
            )

    # ---- optional: overlay trainable subset from a prior ckpt (resume support).
    # Saved keys are post-FSDP-wrap (have `_checkpoint_wrapped_module.` prefix); we strip
    # them so model.net (pre-wrap) accepts them directly. Must run AFTER configure_scope
    # (so trainable param shapes are settled) and BEFORE apply_fsdp (so we load regular
    # tensors, not DTensor shards).
    if args.init_trainable_from is not None:
        saved = torch.load(
            args.init_trainable_from, map_location="cpu", weights_only=False
        )
        trainable_state = saved["trainable_state"]
        cleaned = {
            k.replace("._checkpoint_wrapped_module", ""): v
            for k, v in trainable_state.items()
        }
        missing, unexpected = model.net.load_state_dict(cleaned, strict=False)
        n_loaded = len(cleaned) - len(unexpected)
        if is_rank0:
            print(
                f"[c0_real] init-trainable-from: loaded {n_loaded}/{len(cleaned)} keys "
                f"(prev step={saved.get('step', '?')}) from {args.init_trainable_from}",
                flush=True,
            )
            if len(unexpected) > 0:
                print(
                    f"[c0_real]   WARN unexpected keys ({len(unexpected)}); first 3: {unexpected[:3]}",
                    flush=True,
                )

    # ---- FSDP2 wrap (PyTorch composable.fsdp.fully_shard, in-place — preserves
    # `model.net.disable_context_parallel` etc., unlike DDP which replaces the attribute.

    if world_size > 1:
        from cosmos_predict2._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh

        _shard = int(os.environ.get("FSDP_SHARD_SIZE", str(min(world_size, 8))))
        _shard = max(1, min(_shard, world_size))
        while world_size % _shard != 0:
            _shard -= 1
        _replica = world_size // _shard
        dp_mesh = hsdp_device_mesh(
            replica_group_size=_replica, sharding_group_size=_shard
        )
        if is_rank0:
            print(
                f"[c0_real] HSDP mesh: shard={_shard} replica={_replica} (world={world_size})",
                flush=True,
            )
        model.apply_fsdp(dp_mesh)
        # train_params after fsdp wrap: pull from same model.net (still original module ref + DTensor params)
        train_params = [p for p in model.net.parameters() if p.requires_grad]
        if is_rank0:
            print(
                f"[c0_real] FSDP2 fully_shard applied (world_size={world_size}); "
                f"trainable {sum(p.numel() for p in train_params) / 1e6:.2f}M",
                flush=True,
            )

    # action with placeholder 1 in last 32D (so multiplicative LAM blend = z directly)
    action = torch.zeros(
        1,
        args.num_action_per_chunk,
        args.action_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    action[:, :, -32:] = 1.0  # placeholder; WM forward will multiply by z_cached

    # ---- build a data batch builder. In multi-video mode, called per-step with N random videos
    # stacked into a batch of size args.batch_size.
    rng = np.random.default_rng(args.seed)
    bs = max(1, args.batch_size)
    if args.monitor_only:
        bs = max(1, args.monitor_batch)  # eval batch size for offline action_gap probe

    def _make_batch(vid_idxs):
        """Build a data_batch by stacking videos at given indices. Also swap z_holder['current']
        to a (B, T, 32) tensor so patched forward sees per-row z."""
        if isinstance(vid_idxs, int):
            vid_idxs = [vid_idxs]
        zs = [videos_with_z[i][2] for i in vid_idxs]  # each (1, T, 32)
        arrs = [videos_with_z[i][1] for i in vid_idxs]  # each (T+1, H, W, 3) uint8
        z_stacked = torch.cat(zs, dim=0)  # (B, T, 32)
        z_holder["current"] = z_stacked
        # Action with placeholder 1 in latent slot for each row
        a = torch.zeros(
            len(vid_idxs),
            args.num_action_per_chunk,
            args.action_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        a[:, :, -32:] = 1.0

        # Stack videos: each arr is (T_v+1, H, W, 3) uint8 → tensor (B, 3, T, H, W)
        T_v, H_v, W_v, _ = arrs[0].shape
        vid_t = torch.zeros(
            len(vid_idxs), 3, T_v, H_v, W_v, dtype=torch.uint8, device="cuda"
        )
        for bi, arr in enumerate(arrs):
            t = torch.from_numpy(arr).to("cuda").permute(3, 0, 1, 2)  # (3, T, H, W)
            vid_t[bi] = t
        # Mirror _build_data_batch but for B>1
        H, W = H_v, W_v
        data_batch = {
            "dataset_name": "video_data",
            "video": vid_t,
            "fps": torch.tensor([8] * len(vid_idxs), dtype=torch.float).cuda(),
            "padding_mask": torch.zeros(len(vid_idxs), 1, H, W).cuda(),
            "num_conditional_frames": 1,
            "action": a,
        }
        if model.text_encoder is not None:
            t5 = model.text_encoder.compute_text_embeddings_online(
                data_batch={
                    "ai_caption": [args.prompt] * len(vid_idxs),
                    "images": None,
                },
                input_caption_key="ai_caption",
            )
            data_batch["t5_text_embeddings"] = t5
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        return data_batch

    # ---- gr-aligned step: sample bs agibot episode windows; for each, decode the
    # 13-frame WM video AND compute its matching stride-1 ee_delta(T-1,20); set
    # z_holder['current'] = calibrated g_r*(action). Frames + action share one row-group
    # so z_t causally drives frame_t -> frame_{t+1}.
    def _step_batch_gr_aligned(step_seed: int):
        from cdlam_integration.world_model.gr_aligned_action import (
            decode_window_and_action as _aln_decode,
            calibrated_z_from_action as _cal_z,
        )
        from cdlam_integration.world_model.preprocess import (
            official_wm_video_from_raw as _wm_from_raw,
        )

        step_rng = np.random.default_rng(int(step_seed) + rank * 100003)
        arrs = []  # WM-preprocessed (T, H, W, 3) uint8
        actions = []  # (T-1, 20) aligned ee_delta
        n_ep = aligned_src.n_episodes
        attempt = 0
        while len(arrs) < bs and attempt < bs * 8 + 16:
            attempt += 1
            ei = int(step_rng.integers(0, n_ep))
            ep = aligned_src.episodes[ei]
            max_start = ep.n_frames - args.num_video_frames
            if max_start < 1:
                continue
            sf = int(step_rng.integers(0, max_start))
            try:
                raw, act = _aln_decode(aligned_src, ep, sf, args.num_video_frames)
            except Exception:
                continue
            arrs.append(_wm_from_raw(raw, wm_hw=(H, W)))
            actions.append(act)
        if len(arrs) < bs:
            raise RuntimeError(
                f"gr-aligned: only got {len(arrs)}/{bs} windows after {attempt} attempts"
            )

        # z = calibrated g_r*(aligned action), (bs, T-1, 32)
        a_b = torch.from_numpy(np.stack(actions)).cuda()  # (bs, T-1, 20)
        z_b = _cal_z(a_b, gr_cal, add_noise=args.gr_noise)  # (bs, T-1, 32)
        z_holder["current"] = z_b.to(dtype=torch.bfloat16)

        a = torch.zeros(
            bs,
            args.num_action_per_chunk,
            args.action_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        a[:, :, -32:] = 1.0
        T_v, H_v, W_v, _ = arrs[0].shape
        vid_t = torch.zeros(bs, 3, T_v, H_v, W_v, dtype=torch.uint8, device="cuda")
        for bi, arr in enumerate(arrs):
            vid_t[bi] = torch.from_numpy(arr).to("cuda").permute(3, 0, 1, 2)
        data_batch = {
            "dataset_name": "video_data",
            "video": vid_t,
            "fps": torch.tensor([8] * bs, dtype=torch.float).cuda(),
            "padding_mask": torch.zeros(bs, 1, H_v, W_v).cuda(),
            "num_conditional_frames": 1,
            "action": a,
        }
        if model.text_encoder is not None:
            data_batch["t5_text_embeddings"] = _get_t5_emb_cached(bs)
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        return data_batch

    # In single-video mode, pre-build once for parity with PR-3/4 behavior.
    if gr_aligned_mode:
        if is_rank0:
            print(
                f"[c0_real] gr-aligned mode: {aligned_src.n_episodes} episodes, batch_size={bs}, "
                f"per-step decode + aligned action -> calibrated g_r* z",
                flush=True,
            )
    elif manifest_mode:
        if is_rank0:
            print(
                f"[c0_real] manifest mode: {len(manifest_rows)} rows, batch_size={bs}, "
                f"per-step decode + online LAM z",
                flush=True,
            )
    elif multi_video_mode:
        print(
            f"[c0_real] multi-video mode: {len(videos_with_z)} videos, batch_size={bs}, sampled per step",
            flush=True,
        )
    else:
        data_batch_template = _make_batch([0])
        print(
            f"[c0_real] fixed batch built; video={tuple(data_batch_template['video'].shape)} "
            f"action={tuple(data_batch_template['action'].shape)}",
            flush=True,
        )

    # ---- optimizer + (optional) warmup+cosine LR schedule
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    if args.warmup_steps > 0 or args.lr_end_frac < 1.0:
        import math

        def _lr_lambda(step):
            if args.warmup_steps > 0 and step < args.warmup_steps:
                return (step + 1) / args.warmup_steps  # +1 so step=0 starts at non-zero
            if args.lr_end_frac < 1.0:
                progress = (step - args.warmup_steps) / max(
                    1, args.steps - args.warmup_steps
                )
                progress = min(1.0, max(0.0, progress))
                return args.lr_end_frac + (1.0 - args.lr_end_frac) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=_lr_lambda, last_epoch=args.start_step - 1
        )
        if is_rank0:
            print(
                f"[c0_real] LR schedule: warmup={args.warmup_steps} step, end_frac={args.lr_end_frac} "
                f"(constant unless either set)",
                flush=True,
            )
    else:
        scheduler = None

    # ---- manifest-mode sampling state: one shared deterministic shuffle, sharded by rank.
    # Every rank builds the same permutation, then consumes disjoint offsets:
    # rank 0 gets perm[0], rank 1 gets perm[1], ... (for bs=1), advancing by
    # world_size * bs each step. This keeps the global 4-card stream near no-replacement.
    manifest_epoch = 0
    manifest_perm = np.arange(len(manifest_rows)) if manifest_mode else None
    manifest_global_batch = max(1, world_size * bs)
    manifest_padding_rows = 0
    manifest_epoch_size = 0
    manifest_cursor = 0  # next index into manifest_perm to consume for THIS rank
    if manifest_mode:
        manifest_padding_rows = (-len(manifest_rows)) % manifest_global_batch
        manifest_epoch_size = len(manifest_rows) + manifest_padding_rows
        if manifest_epoch_size < manifest_global_batch:
            sys.exit(
                f"[c0_real] manifest has too few rows ({len(manifest_rows)}) for "
                f"world_size={world_size}, batch_size={bs}"
            )
        manifest_steps_per_epoch = manifest_epoch_size // manifest_global_batch
        if args.start_step > 0:
            manifest_epoch = args.start_step // manifest_steps_per_epoch
            manifest_step_in_epoch = args.start_step % manifest_steps_per_epoch
        else:
            manifest_step_in_epoch = 0

        def _build_manifest_perm(epoch: int) -> np.ndarray:
            perm = np.arange(len(manifest_rows), dtype=np.int64)
            manifest_rng = np.random.default_rng(args.seed + epoch)
            manifest_rng.shuffle(perm)
            if manifest_padding_rows:
                perm = np.concatenate([perm, perm[:manifest_padding_rows]])
            return perm

        manifest_perm = _build_manifest_perm(manifest_epoch)
        manifest_cursor = rank * bs + manifest_step_in_epoch * manifest_global_batch
        if is_rank0 and args.start_step > 0:
            print(
                f"[c0_real] manifest resume cursor: start_step={args.start_step} "
                f"epoch={manifest_epoch} step_in_epoch={manifest_step_in_epoch}/"
                f"{manifest_steps_per_epoch} cursor0={manifest_step_in_epoch * manifest_global_batch}",
                flush=True,
            )
        if is_rank0:
            print(
                f"[c0_real] manifest sampler: rows={len(manifest_rows)} "
                f"global_batch={manifest_global_batch} epoch_size={manifest_epoch_size} "
                f"padding_rows={manifest_padding_rows} steps_per_epoch={manifest_steps_per_epoch}",
                flush=True,
            )
    coverage = np.zeros(len(manifest_rows), dtype=np.int32) if manifest_mode else None
    # Per-step timing + dataset-mix telemetry, written by _step_batch_manifest and read by the
    # training loop. Cleared at start of each step.
    last_step_telemetry: dict = {}

    def _step_batch_manifest(step_seed: int):
        """Take this rank's next shard from the shared manifest permutation."""
        nonlocal manifest_cursor, manifest_perm, manifest_epoch
        if manifest_cursor + bs > manifest_epoch_size:
            manifest_epoch += 1
            manifest_perm = _build_manifest_perm(manifest_epoch)
            manifest_cursor = rank * bs
        base_idxs = manifest_perm[manifest_cursor : manifest_cursor + bs]
        manifest_cursor += manifest_global_batch

        # Decode windows + accumulate decode timing.
        t_decode_start = time.time()
        arrs = []
        sampled_rows = []
        actual_idxs = []
        actual_starts = []
        actual_stops = []
        actual_window_ids = []
        decode_skips = []
        step_rng = np.random.default_rng(int(step_seed) + rank * 100003)
        attempt = 0
        max_attempts = max(bs + 32, bs * 4)

        _decode_workers = int(os.environ.get("WM_DECODE_WORKERS", "1"))

        def _plan_next():
            """CD-LAM runtime component."""
            nonlocal attempt
            if attempt < len(base_idxs):
                i = int(base_idxs[attempt])
            else:
                i = int(step_rng.integers(0, len(manifest_rows)))
            r = manifest_rows[int(i)]
            if args.manifest_random_window:
                clip_nframes = int(
                    r.get("clip_nframes", r.get("n_frames", r["stop_frame"]))
                )
                # The upstream VideoDataset uses randint(0, total - num_frames - 1).
                # keep that distribution, but allow exact-length clips with start=0.
                max_start = max(0, clip_nframes - args.num_video_frames - 1)
                sf = int(step_rng.integers(0, max_start + 1)) if max_start > 0 else 0
                ef = sf + args.num_video_frames
            else:
                sf = int(r["start_frame"])
                ef = int(r["stop_frame"])
            attempt += 1
            return int(i), int(sf), int(ef)

        def _decode_one(plan_item):
            i, sf, ef = plan_item
            r = manifest_rows[int(i)]
            try:
                return ("ok", i, sf, ef, _decode_window(r["video_path"], sf, ef, H, W))
            except Exception as e:
                return ("skip", i, sf, ef, f"{type(e).__name__}: {str(e)[:240]}")

        def _accept(res):
            tag, i, sf, ef, val = res
            r = manifest_rows[int(i)]
            if tag == "skip":
                decode_skips.append(
                    {
                        "row_idx": int(i),
                        "dataset": r.get("dataset", ""),
                        "video_id": r.get("video_id", ""),
                        "video_path": r.get("video_path", ""),
                        "start_frame": int(sf),
                        "stop_frame": int(ef),
                        "error": val,
                    }
                )
                return
            sampled_rows.append(r)
            actual_idxs.append(int(i))
            actual_starts.append(int(sf))
            actual_stops.append(int(ef))
            actual_window_ids.append(
                f"{r.get('dataset', '')}:{r.get('video_id', '')}:{sf}:{ef}"
            )
            arrs.append(val)
            coverage[int(i)] += 1

        while len(arrs) < bs and attempt < max_attempts:
            need = bs - len(arrs)
            plan = []
            while len(plan) < need and attempt < max_attempts:
                plan.append(_plan_next())
            if not plan:
                break
            if _decode_workers > 1 and len(plan) > 1:
                from concurrent.futures import ThreadPoolExecutor as _TPE

                with _TPE(max_workers=min(_decode_workers, len(plan))) as _ex:
                    results = list(_ex.map(_decode_one, plan))
            else:
                results = [_decode_one(p) for p in plan]
            for res in results:
                if len(arrs) >= bs:
                    break
                _accept(res)
        if len(arrs) < bs:
            raise RuntimeError(
                f"manifest decode produced only {len(arrs)}/{bs} samples after "
                f"{attempt} attempts; first skips={decode_skips[:3]}"
            )
        decode_ms = (time.time() - t_decode_start) * 1000

        # LAM forward (online z) timing.
        torch.cuda.synchronize()
        t_lam_start = time.time()
        lam_arrs = [_resize_uint8_video(arr, LAM_H, LAM_W) for arr in arrs]
        # Batched LAM (chunk=32) — was 144 bs=1 forwards, now 5 chunked forwards.
        z_stacked = _build_z_batch(lam_arrs, pair_chunk=32)  # (bs, T, 32)
        torch.cuda.synchronize()
        lam_ms = (time.time() - t_lam_start) * 1000

        z_holder["current"] = z_stacked

        a = torch.zeros(
            bs,
            args.num_action_per_chunk,
            args.action_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        a[:, :, -32:] = 1.0
        T_v, H_v, W_v, _ = arrs[0].shape
        vid_t = torch.zeros(bs, 3, T_v, H_v, W_v, dtype=torch.uint8, device="cuda")
        for bi, arr in enumerate(arrs):
            vid_t[bi] = torch.from_numpy(arr).to("cuda").permute(3, 0, 1, 2)

        manifest_fps = []
        for row in sampled_rows:
            fps = float(row.get("fps", 0.0))
            if not np.isfinite(fps) or fps <= 0.0:
                raise RuntimeError(
                    f"manifest row has invalid fps={row.get('fps')!r}: "
                    f"{row.get('window_id', '<unknown>')}"
                )
            manifest_fps.append(fps)
        data_batch = {
            "dataset_name": "video_data",
            "video": vid_t,
            "fps": torch.tensor(manifest_fps, dtype=torch.float).cuda(),
            "padding_mask": torch.zeros(bs, 1, H_v, W_v).cuda(),
            "num_conditional_frames": 1,
            "action": a,
        }
        if model.text_encoder is not None:
            # Cached: prompt is constant — t5 emb identical across steps; saves ~100ms/step.
            data_batch["t5_text_embeddings"] = _get_t5_emb_cached(bs)
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        # Populate telemetry for the training loop's logger to read this step.
        ds_counts: dict = {}
        for r in sampled_rows:
            ds_counts[r["dataset"]] = ds_counts.get(r["dataset"], 0) + 1
        last_step_telemetry.clear()
        last_step_telemetry.update(
            {
                "decode_ms": round(decode_ms, 1),
                "lam_ms": round(lam_ms, 1),
                "ds_counts": ds_counts,
                "sampled_window_ids": actual_window_ids,
                "manifest_window_ids": [r.get("window_id", "") for r in sampled_rows],
                "row_idxs": actual_idxs,
                "sampled_starts": actual_starts,
                "sampled_stops": actual_stops,
                "decode_skips": decode_skips,
                "decode_skip_count": len(decode_skips),
                "manifest_random_window": bool(args.manifest_random_window),
                "manifest_epoch": int(manifest_epoch),
                "manifest_global_batch": int(manifest_global_batch),
                "manifest_epoch_size": int(manifest_epoch_size),
                "manifest_padding_rows": int(manifest_padding_rows),
            }
        )
        return data_batch

    def _step_batch(step_seed: int):
        if gr_aligned_mode:
            # ALIGNED Stage-2 path: this sampler ALREADY sets z_holder['current'] from the
            # calibrated g_r* on the action that matches the decoded window. No override below.
            return _step_batch_gr_aligned(step_seed)
        if manifest_mode:
            db = _step_batch_manifest(step_seed)
        elif multi_video_mode:
            vid_idxs = list(
                rng.choice(
                    len(videos_with_z),
                    size=min(bs, len(videos_with_z)),
                    replace=(bs > len(videos_with_z)),
                )
            )
            db = _make_batch(vid_idxs)
        else:
            # rebuild with same template (re-init action since forward mutates it)
            db = _make_batch([0])
        # Stage-2 (LEGACY stub): replace the encode-derived z with z = g_r(real action). The
        # loaders above populate z_holder['current'] via the LAM encoder; here we OVERWRITE it so
        # the WM's LAM-z slot is conditioned on real action instead. Default ('encode') skips this.
        # NOTE: only the un-aligned stub uses this; --gr-aligned returns above with aligned z.
        if gr_src is not None:
            T_local = db["action"].shape[1]
            bs_local = db["action"].shape[0]
            _apply_gr_z(bs_local, T_local)
        return db

    # ---- step-0 eval (frozen)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(args.seed)
        db = _step_batch(args.seed)
        out, loss0 = model(db)
    print(f"[c0_real] step-0 loss = {float(loss0):.6e}", flush=True)

    # ---- OFFLINE z-usage eval (--monitor-only): build a fixed eval batch, probe action_gap
    #      across own/zero/shuffle/donor/future with fix-noise pinned, write JSON, EXIT. -------
    if args.monitor_only:
        import json as _json

        # Build a FIXED eval batch ONCE (heldout split via --manifest-split heldout).
        # bs was already set to monitor_batch above, so _step_batch yields the eval-sized batch.
        torch.manual_seed(args.seed + 99991)
        eval_db = _step_batch(args.seed + 99991)
        real_z = z_holder["current"].detach().clone()  # (B,T,32) the real encoded z
        Bz = real_z.shape[0]
        zero_z = torch.zeros_like(real_z)
        donor_z = torch.roll(real_z, shifts=1, dims=0)  # batch-roll donor
        shuf_z = real_z[torch.randperm(Bz)]  # full shuffle across batch
        future_z = torch.roll(real_z, shifts=-1, dims=1)  # time-roll (future leak)

        # Pin fix-noise INSIDE the patched forward by forcing one draw then freezing it.
        # We re-patch forward with fix_noise=True so epsilon/timestep are frozen on first call
        # and reused for every probe -> action_gap is NOT noise-dominated.
        patch_forward_to_use_cached_z(model, z_holder, fix_noise=True)
        # The patched forward MUTATES action[:,:,-32:] *= z IN-PLACE each call. Snapshot the
        # 1.0-placeholder action and RESTORE it before every probe, else z compounds across calls.
        action_template = eval_db["action"].detach().clone()

        def _loss_with(zt):
            eval_db["action"] = (
                action_template.clone()
            )  # reset 1.0 placeholder (no z baked in)
            z_holder["current"] = zt.to(real_z)
            model.eval()
            with torch.no_grad():
                _, loss_value = model(eval_db)
            return float(loss_value)

        # First call freezes the noise (real-z). Then all probes share that frozen noise.
        torch.manual_seed(args.seed)  # pin the single epsilon/timestep draw
        reps = int(args.monitor_repeats)

        def _multi(zt):
            return [_loss_with(zt) for _ in range(reps)]

        real_losses = _multi(real_z)
        zero_losses = _multi(zero_z)
        donor_losses = _multi(donor_z)
        shuf_losses = _multi(shuf_z)
        future_losses = _multi(future_z)
        import numpy as _np

        def _m(x):
            return float(_np.mean(x))

        loss_real = _m(real_losses)
        loss_zero = _m(zero_losses)
        loss_donor = _m(donor_losses)
        loss_shuf = _m(shuf_losses)
        loss_future = _m(future_losses)
        action_gap = loss_zero - loss_real
        zusage_frac = action_gap / max(loss_zero, 1e-9)
        own_minus_shuf = loss_shuf - loss_real
        own_minus_zero = loss_zero - loss_real
        # gate from plan §6.1: own-shuffle >= 0.7*(own-zero) to exclude norm-gaming
        gate_own_shuf = (
            (own_minus_shuf >= 0.7 * own_minus_zero) if own_minus_zero > 0 else False
        )
        floor = float(_np.std(real_losses + zero_losses))
        rec = {
            "ckpt_init_trainable_from": args.init_trainable_from,
            "lam_id": args.lam_id,
            "manifest_split": args.manifest_split,
            "monitor_batch": Bz,
            "monitor_repeats": reps,
            "z_real_mean_norm": float(real_z.norm(dim=-1).mean()),
            "loss_real": loss_real,
            "loss_zero": loss_zero,
            "loss_donor": loss_donor,
            "loss_shuffle": loss_shuf,
            "loss_future": loss_future,
            "action_gap": action_gap,
            "zusage_frac": zusage_frac,
            "swap_gap_loss": loss_donor - loss_real,
            "future_leak_gap": loss_future - loss_real,
            "own_minus_shuffle": own_minus_shuf,
            "own_minus_zero": own_minus_zero,
            "gate_own_shuf_ge_0.7x": bool(gate_own_shuf),
            "noise_floor_std": floor,
            "raw": {
                "real": real_losses,
                "zero": zero_losses,
                "donor": donor_losses,
                "shuffle": shuf_losses,
                "future": future_losses,
            },
        }
        out_json = out_dir / "action_gap.json"
        out_json.write_text(_json.dumps(rec, indent=2))
        print(
            f"[monitor-only] action_gap={action_gap:+.5e} zusage={zusage_frac:+.4f} "
            f"swapΔ={rec['swap_gap_loss']:+.5e} futureLeak={rec['future_leak_gap']:+.5e} "
            f"own-shuf={own_minus_shuf:+.5e} gate={gate_own_shuf} floor={floor:.2e}",
            flush=True,
        )
        print(f"[monitor-only] wrote {out_json}", flush=True)
        return

    # ---- condition-dropout + z-usage monitor setup (ACWM "make it listen") ----
    # Default OFF (cond_dropout==0.0, monitor flag absent) => everything below is inert and the
    # run behaves exactly as before. See monitor_design.md §4.2 hook points.
    dcfg = None
    monitor = None
    monitor_eval_z = None  # fixed real z for the monitor's eval batch
    if args.cond_dropout > 0.0:
        from cdlam_integration.world_model.monitor import DropoutConfig

        # compat trainer feeds z ONLY via z_holder['current'] and NO raw [147:169] slot
        # (action[-32:] is a 1.0 placeholder). So raw_slot=None: clearing z_holder rows is
        # sufficient, there is no raw bypass. (monitor_design.md §2 / appendix A.2 red line.)
        dcfg = DropoutConfig(
            p=args.cond_dropout,
            raw_slot=None,
            seed=args.cond_dropout_seed + rank * 100003,
        )
        if is_rank0:
            print(
                f"[c0_real] condition-dropout ON: p={args.cond_dropout} raw_slot=None "
                f"(z fed via z_holder only; z=0 legal null)",
                flush=True,
            )
    if args.monitor and is_rank0:
        # Monitor runs on rank0 only with a FIXED eval batch (decoded once, frozen) + fix-noise.
        # Loss-space probes only (no decode/encode here) => swap-Δ/future-leak in loss space,
        # action_gap = loss(zero-z) - loss(real-z). step-0 baseline is MANDATORY (raises otherwise).
        from cdlam_integration.world_model.monitor import ZUsageMonitor, MonitorConfig

        torch.manual_seed(args.seed + 99991)
        eval_db = _step_batch(args.seed + 99991)
        monitor_eval_z = (
            z_holder["current"].detach().clone()
        )  # the real z for this fixed batch

        def _loss_with_z(z):
            # z=None => zero-z null. Restore z_holder afterwards so training z is untouched.
            saved = z_holder["current"]
            z_holder["current"] = (
                torch.zeros_like(monitor_eval_z) if z is None else z.to(monitor_eval_z)
            )
            model.eval()
            with torch.no_grad():
                _, loss_value = model(eval_db)
            model.train()
            z_holder["current"] = saved
            return float(loss_value)

        mcfg = MonitorConfig(
            every=args.monitor_every,
            run_swap=True,
            run_future_leak=True,
            run_cycle=False,
            log_path=str(out_dir / "z_usage.jsonl"),
        )
        monitor = ZUsageMonitor(
            mcfg,
            fixed_eval_batch=eval_db,
            fixed_real_z=monitor_eval_z,
            loss_with_z=_loss_with_z,
        )
        base = monitor.run_baseline()  # MANDATORY step-0 baseline
        print(
            f"[c0_real] z-monitor baseline: action_gap={base.action_gap:+.4e} "
            f"zusage={base.zusage_frac:+.3f} floor_std={base.noise_floor_std:.2e}",
            flush=True,
        )

    # ---- training loop
    model.train()
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w") if is_rank0 else None
    if is_rank0:
        log_f.write(
            json.dumps({"step": -1, "loss": float(loss0), "phase": "init"}) + "\n"
        )
    losses = [float(loss0)]
    save_every = getattr(args, "ckpt_save_every", 0) or 0
    t1 = time.time()
    _prof = None
    if os.environ.get("COSMOS_PROFILE", "0") == "1":
        from torch.profiler import (
            profile as _tprofile,
            ProfilerActivity as _PA,
            schedule as _psched,
        )

        _prof = _tprofile(
            activities=[_PA.CPU, _PA.CUDA],
            schedule=_psched(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=False,
            with_stack=False,
        )
        _prof.__enter__()
        if is_rank0:
            print("[c0_real] torch.profiler ON (wait1/warmup1/active3)", flush=True)

    _pf_ex = None
    if os.environ.get("WM_PREFETCH", "0") == "1" and manifest_mode:
        from concurrent.futures import ThreadPoolExecutor as _PFTPE

        _pf_ex = _PFTPE(max_workers=1)
        if is_rank0:
            print("[c0_real] async batch prefetch ON (1-step lookahead)", flush=True)
    _pf_future = None

    def _batch_and_tele(_seed):
        if torch.cuda.is_available():
            with torch.cuda.device(local_rank):
                _db = _step_batch(_seed)
                return _db, dict(last_step_telemetry)
        _db = _step_batch(_seed)
        return _db, dict(last_step_telemetry)

    _ema_on = os.environ.get("WM_EMA", "0") == "1"
    _ema_decay = float(os.environ.get("WM_EMA_DECAY", "0.999"))
    _ema_params = [p.detach().clone() for p in train_params] if _ema_on else None
    if _ema_on and is_rank0:
        print(
            f"[c0_real] EMA ON (decay={_ema_decay}, {len(_ema_params)} tensors)",
            flush=True,
        )
    try:
        for step in range(args.start_step, args.steps):
            torch.manual_seed(args.seed + step + rank * 100003)  # rank-distinct seed
            t_step_start = time.time()
            _tele_cur = None
            if _pf_ex is not None:
                if _pf_future is None:
                    db, _tele_cur = _batch_and_tele(args.seed + step)
                else:
                    db, _tele_cur = _pf_future.result()
                _pf_future = _pf_ex.submit(_batch_and_tele, args.seed + step + 1)
            else:
                db = _step_batch(args.seed + step)
            t_batch_built = time.time()
            # ---- condition-dropout: clear z slot on p-fraction of rows BEFORE forward ----
            # mask_condition_slots clears z_holder['current'] on the hit rows (=> z=0 null under
            # the multiplicative blend). raw_slot=None here (no raw bypass in compat trainer).
            if dcfg is not None:
                from cdlam_integration.world_model.monitor import mask_condition_slots

                mask_condition_slots(db["action"], z_holder, dcfg, step)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_fb_start = time.time()
            out, loss = model(db)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(train_params, max_norm=5.0).item()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if _ema_params is not None:
                with torch.no_grad():
                    for _e, _p in zip(_ema_params, train_params):
                        _e.lerp_(_p.detach(), 1.0 - _ema_decay)
            # ---- z-usage monitor: every --monitor-every steps on the fixed eval batch ----
            if monitor is not None:
                monitor.maybe_run(step)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fb_ms = (time.time() - t_fb_start) * 1000
            if _prof is not None:
                _prof.step()
            if _tele_cur is not None:
                last_step_telemetry.clear()
                last_step_telemetry.update(_tele_cur)
            batch_ms = (t_batch_built - t_step_start) * 1000
            if torch.cuda.is_available():
                mem_stats = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated() / (1024**3),
                        torch.cuda.max_memory_reserved() / (1024**3),
                    ],
                    device="cuda",
                )
                if world_size > 1:
                    torch.distributed.all_reduce(
                        mem_stats, op=torch.distributed.ReduceOp.MAX
                    )
                gpu_alloc_gb = float(mem_stats[0].item())
                gpu_reserved_gb = float(mem_stats[1].item())
                torch.cuda.reset_peak_memory_stats()
            else:
                gpu_alloc_gb = 0.0
                gpu_reserved_gb = 0.0
            loss_val = float(loss)
            if world_size > 1:
                t = torch.tensor([loss_val], device="cuda")
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                loss_val = float(t.item())

            manifest_rank_audit = None
            if manifest_mode and args.manifest_audit_all_ranks:
                local_manifest_audit = {
                    "rank": rank,
                    "row_idxs": last_step_telemetry.get("row_idxs", []),
                    "sampled_window_ids": last_step_telemetry.get(
                        "sampled_window_ids", []
                    ),
                    "manifest_window_ids": last_step_telemetry.get(
                        "manifest_window_ids", []
                    ),
                    "sampled_starts": last_step_telemetry.get("sampled_starts", []),
                    "sampled_stops": last_step_telemetry.get("sampled_stops", []),
                    "ds_counts": last_step_telemetry.get("ds_counts", {}),
                    "decode_skip_count": last_step_telemetry.get(
                        "decode_skip_count", 0
                    ),
                }
                if world_size > 1:
                    gathered = [None for _ in range(world_size)] if is_rank0 else None
                    torch.distributed.gather_object(
                        local_manifest_audit, gathered, dst=0
                    )
                    if is_rank0:
                        manifest_rank_audit = gathered
                else:
                    manifest_rank_audit = [local_manifest_audit]

            if is_rank0:
                rec = {
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": float(gn),
                    "lr": args.lr,
                    "elapsed_sec": round(time.time() - t1, 2),
                    "batch_ms": round(batch_ms, 1),
                    "fb_ms": round(fb_ms, 1),
                    "gpu_alloc_gb": round(gpu_alloc_gb, 2),
                    "gpu_reserved_gb": round(gpu_reserved_gb, 2),
                }
                if manifest_mode:
                    seen_mask = coverage > 0
                    rec["mf_seen_unique"] = int(seen_mask.sum())
                    rec["mf_seen_unique_frac"] = float(seen_mask.mean())
                    rec["mf_max_seen"] = int(coverage.max())
                    rec["mf_mean_seen_among_seen"] = (
                        float(coverage[seen_mask].mean()) if seen_mask.any() else 0.0
                    )
                    # Per-step telemetry from _step_batch_manifest
                    rec["decode_ms"] = last_step_telemetry.get("decode_ms", 0.0)
                    rec["lam_ms"] = last_step_telemetry.get("lam_ms", 0.0)
                    rec["ds_counts"] = last_step_telemetry.get("ds_counts", {})
                    rec["sampled_window_ids"] = last_step_telemetry.get(
                        "sampled_window_ids", []
                    )
                    rec["manifest_window_ids"] = last_step_telemetry.get(
                        "manifest_window_ids", []
                    )
                    rec["row_idxs"] = last_step_telemetry.get("row_idxs", [])
                    rec["sampled_starts"] = last_step_telemetry.get(
                        "sampled_starts", []
                    )
                    rec["sampled_stops"] = last_step_telemetry.get("sampled_stops", [])
                    rec["decode_skip_count"] = last_step_telemetry.get(
                        "decode_skip_count", 0
                    )
                    rec["decode_skips"] = last_step_telemetry.get("decode_skips", [])
                    rec["manifest_random_window"] = last_step_telemetry.get(
                        "manifest_random_window", False
                    )
                    rec["manifest_epoch"] = last_step_telemetry.get("manifest_epoch", 0)
                    rec["manifest_global_batch"] = last_step_telemetry.get(
                        "manifest_global_batch", 0
                    )
                    rec["manifest_epoch_size"] = last_step_telemetry.get(
                        "manifest_epoch_size", 0
                    )
                    rec["manifest_padding_rows"] = last_step_telemetry.get(
                        "manifest_padding_rows", 0
                    )
                    if manifest_rank_audit is not None:
                        global_rows = []
                        global_windows = []
                        global_ds_counts: dict = {}
                        for item in manifest_rank_audit:
                            if item is None:
                                continue
                            global_rows.extend(item.get("row_idxs", []))
                            global_windows.extend(item.get("sampled_window_ids", []))
                            for k, v in item.get("ds_counts", {}).items():
                                global_ds_counts[k] = global_ds_counts.get(k, 0) + int(
                                    v
                                )
                        rec["rank_audit"] = manifest_rank_audit
                        rec["global_row_idxs"] = global_rows
                        rec["global_sampled_window_ids"] = global_windows
                        rec["global_ds_counts"] = global_ds_counts
                        rec["global_duplicate_rows"] = len(global_rows) - len(
                            set(global_rows)
                        )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                losses.append(loss_val)
                if (
                    step < 5
                    or step % max(1, args.steps // 20) == 0
                    or step == args.steps - 1
                ):
                    cov_blurb = ""
                    if manifest_mode:
                        ds = last_step_telemetry.get("ds_counts", {})
                        cov_blurb = (
                            f"  mf_seen={int((coverage > 0).sum())}/{len(coverage)} "
                            f"max={int(coverage.max())} "
                            f"decode={last_step_telemetry.get('decode_ms', 0):.0f}ms "
                            f"lam={last_step_telemetry.get('lam_ms', 0):.0f}ms "
                            f"batch_wait={batch_ms:.0f}ms "
                            f"fb={fb_ms:.0f}ms ds={ds}"
                        )
                    print(
                        f"[c0_real] step {step:4d}/{args.steps} loss={loss_val:.6e} "
                        f"gn={gn:.3f}{cov_blurb}",
                        flush=True,
                    )
            # Periodic save: trigger every `save_every` steps including the FINAL step
            # (previously a `step + 1 < args.steps` guard silently dropped the final ckpt).
            if save_every > 0 and (step + 1) % save_every == 0:
                # FSDP2: params are DTensor (sharded); must call full_tensor() on ALL ranks
                # (it's a collective all-gather), then only rank0 writes to disk.
                net_for_save = model.net
                trainable_state = {}
                ema_state = {}
                _ei = 0
                for n, p in net_for_save.named_parameters():
                    if p.requires_grad:
                        full = p.full_tensor() if hasattr(p, "full_tensor") else p
                        if is_rank0:
                            trainable_state[n] = full.detach().cpu()
                        if _ema_params is not None:
                            _ev = _ema_params[_ei]
                            _efull = (
                                _ev.full_tensor()
                                if hasattr(_ev, "full_tensor")
                                else _ev
                            )
                            if is_rank0:
                                ema_state[n] = _efull.detach().cpu()
                        _ei += 1
                if is_rank0:
                    ckpt_path = out_dir / f"ckpt_step{step + 1}.pt"
                    _ckpt = {
                        "step": step + 1,
                        "trainable_state": trainable_state,
                        "args": vars(args),
                    }
                    if _ema_params is not None:
                        _ckpt["ema_trainable_state"] = ema_state
                    torch.save(_ckpt, str(ckpt_path))
                    print(
                        f"[c0_real]   periodic save -> {ckpt_path}"
                        + (" (+EMA)" if _ema_params is not None else ""),
                        flush=True,
                    )
    except Exception as e:
        if log_f is not None:
            log_f.close()
        print(
            f"[c0_real] (rank {rank}) FAILED at step {step}: {type(e).__name__}: {e}",
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)
    if log_f is not None:
        log_f.close()
    elapsed = time.time() - t1

    if _prof is not None:
        try:
            _prof.__exit__(None, None, None)
        except Exception:
            pass
        if is_rank0:
            print(
                "[c0_real] === torch.profiler key_averages (self_cuda_time_total, top30) ===",
                flush=True,
            )
            print(
                _prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30
                ),
                flush=True,
            )
            _tp = os.environ.get(
                "PROFILE_TRACE",
                str(Path(os.environ.get("TMPDIR", "/tmp")) / "cdlam_wm_trace.json"),
            )
            try:
                _prof.export_chrome_trace(_tp)
                print(f"[c0_real] chrome trace -> {_tp}", flush=True)
            except Exception as _e:
                print(f"[c0_real] trace export failed: {_e}", flush=True)

    # ---- coverage summary (manifest mode only)
    if manifest_mode:
        # Aggregate coverage across ranks via all-reduce (sum of per-rank counts).
        cov_t = torch.from_numpy(coverage).to("cuda")
        if world_size > 1:
            torch.distributed.all_reduce(cov_t, op=torch.distributed.ReduceOp.SUM)
        cov_global = cov_t.cpu().numpy()
        if is_rank0:
            seen_mask = cov_global > 0
            cov_summary = {
                "manifest_path": args.train_manifest,
                "manifest_split": args.manifest_split,
                "n_rows": int(len(cov_global)),
                "manifest_global_batch": int(manifest_global_batch),
                "manifest_epoch_size": int(manifest_epoch_size),
                "manifest_padding_rows": int(manifest_padding_rows),
                "n_rows_seen": int(seen_mask.sum()),
                "frac_rows_seen": float(seen_mask.mean()),
                "max_times_seen": int(cov_global.max()),
                "mean_among_seen": float(cov_global[seen_mask].mean())
                if seen_mask.any()
                else 0.0,
                "p50_among_seen": float(np.percentile(cov_global[seen_mask], 50))
                if seen_mask.any()
                else 0.0,
                "p90_among_seen": float(np.percentile(cov_global[seen_mask], 90))
                if seen_mask.any()
                else 0.0,
            }
            cov_path = out_dir / "coverage_summary.json"
            cov_path.write_text(json.dumps(cov_summary, indent=2))
            np.save(out_dir / "coverage_counts.npy", cov_global)
            print(
                f"[c0_real] manifest coverage: {cov_summary['n_rows_seen']}/{cov_summary['n_rows']} "
                f"({cov_summary['frac_rows_seen'] * 100:.1f}%) rows seen, "
                f"max_seen={cov_summary['max_times_seen']}",
                flush=True,
            )

    # ---- verdict
    init_loss = losses[0]
    final_loss = losses[-1]
    drop = init_loss / max(final_loss, 1e-12)
    verdict = "PASS" if drop >= 2.0 else "WEAK_DROP" if drop >= 1.1 else "FAIL"

    # ---- save trained action_embedder weights
    # FSDP2: params are DTensor; full_tensor() is collective and must be called on ALL ranks.
    if args.save_action_embedder:
        net_for_save = model.net

        def _gather_state(mod):
            out = {}
            for n, p in mod.state_dict().items():
                full = p.full_tensor() if hasattr(p, "full_tensor") else p
                if is_rank0:
                    out[n] = full.detach().cpu()
            return out

        emb_B_D = _gather_state(net_for_save.action_embedder_B_D)
        emb_B_3D = _gather_state(net_for_save.action_embedder_B_3D)
        if is_rank0:
            emb_state = {
                "action_embedder_B_D": emb_B_D,
                "action_embedder_B_3D": emb_B_3D,
                "lam_id": args.lam_id,
                "base_ckpt": args.ckpt,
                "num_action_per_chunk": args.num_action_per_chunk,
                "action_dim": args.action_dim,
                "trained_steps": args.steps,
                "fix_noise": args.fix_noise,
            }
            emb_path = out_dir / "action_embedder.pt"
            torch.save(emb_state, str(emb_path))
            print(f"[c0_real] saved action_embedder state -> {emb_path}", flush=True)

    summary = {
        "lam_id": args.lam_id,
        "experiment": args.experiment,
        "ckpt": args.ckpt,
        "input_video": args.input_video,
        "scope": args.scope,
        "fps_source": "manifest" if manifest_mode else "fixed_8",
        "train_manifest": args.train_manifest,
        "manifest_split": args.manifest_split,
        "manifest_audit_all_ranks": bool(args.manifest_audit_all_ranks),
        "manifest_random_window": bool(args.manifest_random_window),
        "manifest_global_batch": int(manifest_global_batch) if manifest_mode else None,
        "manifest_epoch_size": int(manifest_epoch_size) if manifest_mode else None,
        "manifest_padding_rows": int(manifest_padding_rows) if manifest_mode else None,
        "num_video_frames": args.num_video_frames,
        "num_action_per_chunk": args.num_action_per_chunk,
        "action_dim": args.action_dim,
        "resolution": [H, W],
        "lam_resolution": [LAM_H, LAM_W],
        "n_train_params": int(n_train),
        "steps": args.steps,
        "lr": args.lr,
        "z_cached_shape": list(z_cached.shape),
        "z_cached_mean_norm": float(z_cached.norm(dim=-1).mean().item()),
        "init_loss": init_loss,
        "final_loss": final_loss,
        "min_loss": min(losses),
        "drop_factor": drop,
        "elapsed_sec": round(elapsed, 2),
        "verdict": verdict,
    }
    if is_rank0:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        print(
            f"[c0_real] verdict = {verdict}; init={init_loss:.4e} final={final_loss:.4e} drop={drop:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
