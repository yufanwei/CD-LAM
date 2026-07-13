"""LAM release launcher — masked loss + aligned official crop/scale + pre-decoded cache.

What this does:

1. **Monkey-patches** ``cdlam_integration.lam.data.decode_rows_parallel`` so the
   masked trainer's pair-decode path goes through the **official 4:3 center-crop ->
   480x640 -> 240x320 two-stage resize** chain (defined in
   ``cdlam_integration/tools/_cdlam_data.py``). masked's old single-stage resize is
   replaced byte-for-byte. This is the only data-side change vs masked.

2. If a ``--cache`` npz / memmap is provided, the patched decode does
   **constant-time cache lookup by pair_id**, falling back to fresh decode
   only for missing pair_ids. This is what masked's monkey-patch did — release keeps
   the same shape but the cache itself was produced through the aligned protocol
   (so cache contents are byte-equivalent to aligned's runtime decode).

3. After patching, hands off to the masked trainer (``cdlam_integration/tools/train_cdlam.py``)
   unchanged. The masked trainer's loss assembly (masked partial_fullmix:
   ``ρ·L_rec_full + λ·L_sig + w_id·L_id + β·KL`` with optional fg/bg) runs
   verbatim — this is the masked-tested loss path.

Usage:

  # 4-card DDP, 1000-step, no-mask config (masked partial_fullmix verbatim)
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
    cdlam_integration/tools/mask_adapter.py \\
      --cache outputs/cdlam_data/frame_cache.npz \\
      --config cdlam_integration/configs/stage1.yaml \\
      --out outputs/cdlam_train/stage1_run \\
      --total-steps 1000 --warmup-steps 150 --ramp-end 400 \\
      --batch-real 54 --batch-id 6 --batch-mask 0 \\
      --log-every 50 --ckpt-every 250 --eval-every 250 --seed 0
"""

from __future__ import annotations

import os

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))


# ============================================================================
# Step 1: monkey-patch decode_rows_parallel to use aligned official crop/scale
# ============================================================================


def install_aligned_decode() -> None:
    """Replace training's ``decode_rows_parallel`` with the aligned official-protocol one.

    training's version does ``cv2.resize((W,H), INTER_AREA)`` — a one-stage downsample
    that diverges from cosmos WM's two-stage path on >=4:3 sources (EgoDex
    1080p) by ~3 dB PSNR equivalent. aligned's version does ``center-crop 4:3 ->
    480x640 INTER_LINEAR -> target_hw INTER_LINEAR``, byte-aligned with cosmos
    WM data path (sanity_videosampler_self_test.py: z_cos = 0.99996).

    Same signature: ``(rows: pd.DataFrame, target_hw, workers) -> (uint8[N,2,H,W,3], bool[N])``.
    """
    from cdlam_integration.lam import data as training_data
    from cdlam_integration.lam import preprocess as preprocessing

    training_data.decode_rows_parallel = preprocessing.decode_rows_parallel
    print(
        "[release] patched decode_rows_parallel: training single-stage -> aligned official two-stage",
        flush=True,
    )


def install_mask_loader() -> None:
    """Replace masked's ``load_fg_bg_masks_for_rows`` with aligned's
    ``load_foreground_background_masks``.

    masked's loader indexes ``mask[frame_j]`` directly, which assumes the mask npz
    has the same frame numbering as the source video. cleandata violates this:
    the source.mp4 has ~80-200 frames at 30 fps but the mask npz only has 49
    canonical frames; ``frame_to_mask_idx[frame_j]`` does the remap.

    aligned's loader checks for a ``frame_to_mask_idx_path`` column on the row
    and applies the remap automatically. Same return signature
    ``(fg, bg, valid)`` so the masked trainer is none the wiser.

    Side-effect: also normalises the ``mask_npz_path`` column expected by aligned
    onto the masked-style ``robosam_interaction_mask_path``.
    """
    from cdlam_integration.lam import masks as mask_module
    from cdlam_integration.lam import preprocess as preprocessing

    mask_loader = preprocessing.load_foreground_background_masks

    def patched_load(rows, cache, target_hw=(240, 320)):
        # masked trainer always passes 3 args. aligned supports radius / dilate kwargs
        # but defaults are fine (no radius, no extra dilation).
        return mask_loader(
            rows,
            cache,
            target_hw=target_hw,
            frame_col="frame_j",
            runtime_dilate_px=0,
            mask_temporal_radius=0,
        )

    mask_module.load_fg_bg_masks_for_rows = patched_load
    try:
        from cdlam_integration.lam import train as trainer

        if hasattr(trainer, "load_fg_bg_masks_for_rows"):
            trainer.load_fg_bg_masks_for_rows = patched_load
    except ImportError:
        pass
    print(
        "[release] patched load_fg_bg_masks_for_rows: masked direct-index -> aligned frame_to_mask_idx-aware",
        flush=True,
    )


# ============================================================================
# Step 2: optional cache lookup overlay (no repeat video reads)
# ============================================================================


def install_cache_lookup(cache_path: str) -> None:
    """Wrap the (already aligned-patched) ``decode_rows_parallel`` so that pair_ids
    present in the cache use O(1) memmap lookup; missing pair_ids fall through
    to fresh decode (still via aligned protocol).

    Cache schema (compatible with masked's pre_decode_ego_frames.py output):
      - ``pair_ids``: int64 (N,)
      - ``frame_i``: uint8 (N, H, W, 3)
      - ``frame_j``: uint8 (N, H, W, 3)
      - ``valid``:   bool (N,)

    For larger caches (>10 GB), pass a memmap-backed ``.npz`` produced by
    ``cache builder`` (it uses ``np.savez`` with ``mmap_mode='r'``-loadable
    arrays).
    """
    cache = np.load(cache_path, mmap_mode="r")
    pair_ids = cache["pair_ids"].astype(np.int64)
    frame_i = cache["frame_i"]
    frame_j = cache["frame_j"]
    valid = cache["valid"]
    lookup = {int(pid): i for i, pid in enumerate(pair_ids)}
    print(
        f"[release] loaded cache {cache_path}: {len(pair_ids):,} pairs, "
        f"{int(valid.sum()):,} valid, frame shape {tuple(frame_i.shape[1:])}",
        flush=True,
    )

    from cdlam_integration.lam import data as training_data

    fresh_decode = training_data.decode_rows_parallel  # already aligned official

    def cached_decode(rows: pd.DataFrame, target_hw=(240, 320), workers: int = 16):
        n = len(rows)
        H, W = target_hw
        out = np.zeros((n, 2, H, W, 3), dtype=np.uint8)
        out_valid = np.zeros(n, dtype=bool)
        fallback_idx = []
        for i, r in enumerate(rows.itertuples(index=False)):
            pid = int(r.pair_id)
            idx = lookup.get(pid)
            if idx is not None and bool(valid[idx]):
                out[i, 0] = frame_i[idx]
                out[i, 1] = frame_j[idx]
                out_valid[i] = True
            else:
                fallback_idx.append(i)
        if fallback_idx:
            fb_rows = rows.iloc[fallback_idx]
            fb_arr, fb_v = fresh_decode(fb_rows, target_hw=target_hw, workers=workers)
            for k, i in enumerate(fallback_idx):
                out[i] = fb_arr[k]
                out_valid[i] = bool(fb_v[k])
        return out, out_valid

    training_data.decode_rows_parallel = cached_decode
    try:
        from cdlam_integration.lam import train as trainer

        if hasattr(trainer, "decode_rows_parallel"):
            trainer.decode_rows_parallel = cached_decode
    except ImportError:
        pass
    print(
        "[release] installed cache lookup over aligned decode (fresh-decode fallback for missing pair_ids)",
        flush=True,
    )


# ============================================================================
# Entry point
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cache", default="", help="path to frame_cache.npz (optional)"
    )
    args, remaining = parser.parse_known_args()

    install_aligned_decode()
    install_mask_loader()
    if args.cache:
        install_cache_lookup(args.cache)

    sys.argv = [sys.argv[0]] + remaining
    from cdlam_integration.lam.train import main as train_main

    return train_main()


if __name__ == "__main__":
    sys.exit(main())
