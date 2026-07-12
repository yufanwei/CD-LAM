"""LAM V7 launcher — V5.3 loss + V6.1 official crop/scale + pre-decoded cache.

What this does:

1. **Monkey-patches** ``LAM_V3.tools._lam_v3_data.decode_rows_parallel`` so the
   V5 trainer's pair-decode path goes through the **official 4:3 center-crop ->
   480x640 -> 240x320 two-stage resize** chain (defined in
   ``LAM_V6/tools/_lam_v6_data.py``). V5.3's old single-stage resize is
   replaced byte-for-byte. This is the only data-side change vs V5.3.

2. If a ``--cache`` npz / memmap is provided, the patched decode does
   **constant-time cache lookup by pair_id**, falling back to fresh decode
   only for missing pair_ids. This is what V5.3's monkey-patch did — V7 keeps
   the same shape but the cache itself was produced through the V6.1 protocol
   (so cache contents are byte-equivalent to V6.1's runtime decode).

3. After patching, hands off to the V5 trainer (``LAM_V5/tools/train_lam_v5.py``)
   unchanged. The V5 trainer's loss assembly (V5 partial_fullmix:
   ``ρ·L_rec_full + λ·L_sig + w_id·L_id + β·KL`` with optional fg/bg) runs
   verbatim — this is the V5.3-tested loss path.

Usage:

  # 4-card DDP, 1000-step, no-mask config (V5.3 partial_fullmix verbatim)
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
    LAM_V7/tools/train_v7_cached.py \\
      --cache outputs/lam_v7_data/v7_frame_cache.npz \\
      --config LAM_V7/configs/v7_no_mask.yaml \\
      --out outputs/lam_v7_train/v7_no_mask_1000 \\
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

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(REPO))


# ============================================================================
# Step 1: monkey-patch decode_rows_parallel to use V6.1 official crop/scale
# ============================================================================


def install_v6_1_decode_patch() -> None:
    """Replace V3's ``decode_rows_parallel`` with the V6.1 official-protocol one.

    V3's version does ``cv2.resize((W,H), INTER_AREA)`` — a one-stage downsample
    that diverges from cosmos WM's two-stage path on >=4:3 sources (EgoDex
    1080p) by ~3 dB PSNR equivalent. V6.1's version does ``center-crop 4:3 ->
    480x640 INTER_LINEAR -> target_hw INTER_LINEAR``, byte-aligned with cosmos
    WM data path (sanity_videosampler_self_test.py: z_cos = 0.99996).

    Same signature: ``(rows: pd.DataFrame, target_hw, workers) -> (uint8[N,2,H,W,3], bool[N])``.
    """
    from LAM_V3.tools import _lam_v3_data as v3data
    from LAM_V6.tools import _lam_v6_data as v6data

    v3data.decode_rows_parallel = v6data.decode_rows_parallel
    print(
        "[v7] patched decode_rows_parallel: V3 single-stage -> V6.1 official two-stage",
        flush=True,
    )


def install_v6_1_mask_patch() -> None:
    """Replace V5's ``load_fg_bg_masks_for_rows`` with V6.1's
    ``load_fg_bg_masks_for_v6_rows``.

    V5's loader indexes ``mask[frame_j]`` directly, which assumes the mask npz
    has the same frame numbering as the source video. cleandata violates this:
    the source.mp4 has ~80-200 frames at 30 fps but the mask npz only has 49
    canonical frames; ``frame_to_mask_idx[frame_j]`` does the remap.

    V6.1's loader checks for a ``frame_to_mask_idx_path`` column on the row
    and applies the remap automatically. Same return signature
    ``(fg, bg, valid)`` so the V5 trainer is none the wiser.

    Side-effect: also normalises the ``mask_npz_path`` column expected by V6.1
    onto the V5-style ``robosam_interaction_mask_path``.
    """
    from LAM_V5.tools import _lam_v5_data as v5data
    from LAM_V6.tools import _lam_v6_data as v6data

    v6_loader = v6data.load_fg_bg_masks_for_v6_rows

    def patched_load(rows, cache, target_hw=(240, 320)):
        # V5 trainer always passes 3 args. V6.1 supports radius / dilate kwargs
        # but defaults are fine (no radius, no extra dilation).
        return v6_loader(
            rows,
            cache,
            target_hw=target_hw,
            frame_col="frame_j",
            runtime_dilate_px=0,
            mask_temporal_radius=0,
        )

    v5data.load_fg_bg_masks_for_rows = patched_load
    try:
        from LAM_V5.tools import train_lam_v5 as v5train

        if hasattr(v5train, "load_fg_bg_masks_for_rows"):
            v5train.load_fg_bg_masks_for_rows = patched_load
    except ImportError:
        pass
    print(
        "[v7] patched load_fg_bg_masks_for_rows: V5 direct-index -> V6.1 frame_to_mask_idx-aware",
        flush=True,
    )


# ============================================================================
# Step 2: optional cache lookup overlay (no repeat video reads)
# ============================================================================


def install_cache_lookup(cache_path: str) -> None:
    """Wrap the (already V6.1-patched) ``decode_rows_parallel`` so that pair_ids
    present in the cache use O(1) memmap lookup; missing pair_ids fall through
    to fresh decode (still via V6.1 protocol).

    Cache schema (compatible with V5.3's pre_decode_ego_frames.py output):
      - ``pair_ids``: int64 (N,)
      - ``frame_i``: uint8 (N, H, W, 3)
      - ``frame_j``: uint8 (N, H, W, 3)
      - ``valid``:   bool (N,)

    For larger caches (>10 GB), pass a memmap-backed ``.npz`` produced by
    ``predecode_v7_cache.py`` (it uses ``np.savez`` with ``mmap_mode='r'``-loadable
    arrays).
    """
    cache = np.load(cache_path, mmap_mode="r")
    pair_ids = cache["pair_ids"].astype(np.int64)
    frame_i = cache["frame_i"]
    frame_j = cache["frame_j"]
    valid = cache["valid"]
    lookup = {int(pid): i for i, pid in enumerate(pair_ids)}
    print(
        f"[v7] loaded cache {cache_path}: {len(pair_ids):,} pairs, "
        f"{int(valid.sum()):,} valid, frame shape {tuple(frame_i.shape[1:])}",
        flush=True,
    )

    from LAM_V3.tools import _lam_v3_data as v3data

    fresh_decode = v3data.decode_rows_parallel  # already V6.1 official

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

    v3data.decode_rows_parallel = cached_decode
    try:
        from LAM_V5.tools import train_lam_v5 as v5train

        if hasattr(v5train, "decode_rows_parallel"):
            v5train.decode_rows_parallel = cached_decode
    except ImportError:
        pass
    print(
        "[v7] installed cache lookup over V6.1 decode (fresh-decode fallback for missing pair_ids)",
        flush=True,
    )


# ============================================================================
# Entry point
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cache", default="", help="path to v7_frame_cache.npz (optional)"
    )
    args, remaining = parser.parse_known_args()

    install_v6_1_decode_patch()
    install_v6_1_mask_patch()
    if args.cache:
        install_cache_lookup(args.cache)

    sys.argv = [sys.argv[0]] + remaining
    from LAM_V5.tools.train_lam_v5 import main as train_main

    return train_main()


if __name__ == "__main__":
    sys.exit(main())
