"""Data helpers for LAM aligned no-label view-contrastive training.

The aligned sampler mirrors the official LAM training data pipeline at
``external/lam_project/lam/dataset.py:184-239`` (which produced LAM_400k.ckpt):

  1. Each sample picks one video uniformly from the pool.
  2. A random ``frame_skip = randint(1, 4)`` decides the transition stride.
  3. A random ``start_frame`` is chosen so that the window fits in the video.
  4. ``cap.set(CAP_PROP_POS_FRAMES, start)`` then read ``2*stride`` frames.
  5. Slice ``[::stride]`` to get 2 frames at ``(start, start+stride)``.

That is "Option C" from the 2026-05-13 sampling-protocol audit. We keep the
**two-stage** 4:3 -> 480x640 -> 240x320 resize chain (which matches the WM
data path) so aligned latent action is aligned with what the downstream Cosmos WM
sees at inference. Original LAM training used a single-stage resize, but on
480x640 sources (GR1_robot, AgiBot head_color) the two are equivalent. For
higher-resolution sources (EgoDex 1080x1920) the two-stage chain stays much
closer to z computed by the official WM-side path.

aligned loss is computed on top of the produced pair. Per-row mask validity (for
the fg/bg views) is decided at runtime: if the sampled frames fall inside the
video's mask coverage, mask_valid=True; otherwise that row contributes only to
view-contrastive / null / KL / norm-band losses.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)

from cdlam_integration.lam.masks import FgBgMaskCache  # noqa: E402


OFFICIAL_WM_HW: Tuple[int, int] = (480, 640)
OFFICIAL_LAM_HW: Tuple[int, int] = (240, 320)


def _center_crop_slices(
    shape_hw: Tuple[int, int], target_ratio: float = 640 / 480
) -> Tuple[slice, slice]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid frame shape: {shape_hw}")
    current_ratio = w / h
    if current_ratio > target_ratio:
        target_h = h
        target_w = max(1, int(h * target_ratio))
    elif current_ratio < target_ratio:
        target_h = max(1, int(w / target_ratio))
        target_w = w
    else:
        target_h = h
        target_w = w
    y0 = max(0, (h - target_h) // 2)
    x0 = max(0, (w - target_w) // 2)
    return slice(y0, y0 + target_h), slice(x0, x0 + target_w)


def center_crop_to_official_ratio(arr: np.ndarray) -> np.ndarray:
    """Match upstream ACWM raw-video crop before resizing to 480x640 / 240x320."""
    ys, xs = _center_crop_slices(arr.shape[:2])
    return arr[ys, xs, ...]


def patch_layernorm_to_fp32(model) -> int:
    """Force every nn.LayerNorm in ``model`` to run forward in fp32, even when
    the caller is inside a bf16 autocast context.

    Why: PyTorch's **bf16** autocast does NOT promote LayerNorm to fp32 (unlike
    its fp16 autocast). bf16 shares fp32's exponent range so it was presumed
    safe, but bf16's 7-bit mantissa is much coarser than fp16's 10-bit (and
    fp32's 23-bit). LAM's encoder runs LayerNorm over (B * 600) tokens per
    view × 6 views × 24 blocks each step, and around B>=55 the gamma-gradient
    accumulation in NativeLayerNormBackward starts returning NaN (verified
    with ``torch.autograd.set_detect_anomaly``).

    This patch replaces each LayerNorm's forward with one that locally
    disables autocast and casts input/weight/bias to fp32, then casts the
    output back to the caller's dtype. Cost: an extra cast per LN; ~1-3%
    step-time increase. Benefit: B max climbs above the bf16 NaN threshold.
    """
    import torch  # local import keeps this module importable from CPU contexts
    import torch.nn.functional as F  # noqa: F401

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.forward = _make_fp32_layernorm_forward(module)
            count += 1
    return count


def _make_fp32_layernorm_forward(module):
    import torch
    import torch.nn.functional as F

    def forward(x):
        orig_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            out = F.layer_norm(
                x.float(),
                module.normalized_shape,
                module.weight.float() if module.weight is not None else None,
                module.bias.float() if module.bias is not None else None,
                module.eps,
            )
        return out.to(orig_dtype)

    return forward


def resize_frame_to_official_lam(
    frame_rgb: np.ndarray, target_hw: Tuple[int, int] = OFFICIAL_LAM_HW
) -> np.ndarray:
    """Center-crop to 4:3 then run the official two-stage resize chain.

    Official ``groot_dreams/data/dataset_video.py:79-96`` goes through a 480x640
    intermediate (the WM video resolution) before downsampling to the 240x320
    LAM resolution. A single-step direct resize to 240x320 produces measurably
    different z_mu on 16:9 sources (e.g. EgoDex 1080x1920: z_cos drops to ~0.99
    vs ~1.0000 with the two-stage chain), because the bilinear kernel keeps
    more frequency information when the downsample is split into ~2x steps
    instead of one ~4.5x step. We replicate that here so aligned training, aligned eval,
    the WM gate and any downstream consumer all encode LAM inputs through the
    same byte-aligned chain.
    """
    H, W = target_hw
    cropped = center_crop_to_official_ratio(frame_rgb)
    # Stage 1: -> 480x640 intermediate (matches official WM video resolution).
    if cropped.shape[:2] != OFFICIAL_WM_HW:
        cropped = cv2.resize(
            cropped,
            (OFFICIAL_WM_HW[1], OFFICIAL_WM_HW[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    # Stage 2: -> target (240x320 by default). Skip if caller wanted 480x640.
    if (H, W) == OFFICIAL_WM_HW:
        return cropped.astype(np.uint8, copy=False)
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _resize_mask_nearest(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    if mask.shape == (H, W):
        return mask
    cropped = center_crop_to_official_ratio(mask)
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_NEAREST)


def _decode_pair_official(args):
    """Read two frames and apply the official upstream ACWM LAM resize protocol."""
    video_path, fi, fj, target_hw = args
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    out = {fi: None, fj: None}
    needed = max(fi, fj)
    i = 0
    try:
        while i <= needed:
            if i in out:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                fr = resize_frame_to_official_lam(rgb, target_hw)
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


def decode_rows_parallel(
    rows: pd.DataFrame,
    target_hw: Tuple[int, int] = OFFICIAL_LAM_HW,
    workers: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode pair rows using the official LAM image protocol.

    upstream ACWM uses full-resolution video at 480x640 and computes old-LAM
    z from a 240x320 ``lam_video`` derived after the same 4:3 center crop. aligned
    trains directly on pairs, so this helper applies the crop/resize before the
    frames enter LAM.
    """
    pool = [
        (r.video_path, int(r.frame_i), int(r.frame_j), target_hw)
        for r in rows.itertuples(index=False)
    ]
    out = np.zeros((len(rows), 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
    valid = np.zeros(len(rows), dtype=bool)
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        for i, arr in enumerate(ex.map(_decode_pair_official, pool)):
            if arr is not None:
                out[i] = arr
                valid[i] = True
    return out, valid


def _decode_pair_random_stride(args):
    """Read 2 frames at ``(start, start + stride)`` via fast-seek + sequential read.

    Mirrors ``external/lam_project/lam/dataset.py:196-221``: set ``POS_FRAMES``,
    read ``2*stride`` frames, take ``[::stride]``. Returns RGB uint8 ``(2,H,W,C)``
    after the two-stage official resize chain.
    """
    video_path, start, stride, target_hw = args
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    try:
        frames: list[np.ndarray] = []
        for _ in range(int(2 * stride)):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        if len(frames) < int(2 * stride):
            return None
        pair_raw = [frames[0], frames[int(stride)]]
        pair = [resize_frame_to_official_lam(f, target_hw) for f in pair_raw]
        return np.stack(pair, axis=0).astype(np.uint8)
    finally:
        cap.release()


def decode_rows_random_stride(
    rows: pd.DataFrame,
    target_hw: Tuple[int, int] = OFFICIAL_LAM_HW,
    workers: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel decode for ``VideoPairSampler`` output rows.

    Each row must have columns ``video_path``, ``start`` and ``stride``.
    Returns ``(pairs uint8 (N,2,H,W,C), valid bool (N,))``.
    """
    pool = [
        (r.video_path, int(r.start), int(r.stride), target_hw)
        for r in rows.itertuples(index=False)
    ]
    out = np.zeros((len(rows), 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
    valid = np.zeros(len(rows), dtype=bool)
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        for i, arr in enumerate(ex.map(_decode_pair_random_stride, pool)):
            if arr is not None:
                out[i] = arr
                valid[i] = True
    return out, valid


class VideoPairSampler:
    """Random video + random start + random stride pair sampler (Option C).

    Mirrors ``external/lam_project/lam/dataset.py`` which produced LAM_400k.ckpt.
    Unlike ``IndexedPairSampler`` this sampler does not depend on a pre-built
    label-relation pair index; sampling decisions are made at runtime so the
    model sees random transitions covering the full video set every batch.

    Required parquet columns in ``video_index_path``:
      - ``video_path``, ``dataset``, ``total_frames``
    Optional:
      - ``mask_npz_path`` (for fg/bg view masks)
      - ``episode_id``
    """

    def __init__(
        self,
        video_index_path: Path,
        *,
        stride_range: Tuple[int, int] = (1, 4),
        seed: int = 0,
        require_mask_path: bool = False,
    ) -> None:
        df = pd.read_parquet(video_index_path)
        self.raw_rows = int(len(df))
        for col in ("video_path", "dataset", "total_frames"):
            if col not in df.columns:
                raise KeyError(f"video_index missing required column: {col}")

        if "mask_npz_path" not in df.columns:
            df = df.copy()
            df["mask_npz_path"] = ""
        else:
            df = df.copy()
            df["mask_npz_path"] = df["mask_npz_path"].fillna("").astype(str)
        if "episode_id" not in df.columns:
            df["episode_id"] = df["video_path"].astype(str)

        stride_low, stride_high = int(stride_range[0]), int(stride_range[1])
        if stride_low < 1 or stride_high < stride_low:
            raise ValueError(f"bad stride_range: {stride_range}")
        # Need at least 2*stride_high+1 frames so any sampled stride fits.
        df = df[df["total_frames"] >= (2 * stride_high + 1)]
        if require_mask_path:
            df = df[df["mask_npz_path"].astype(str).str.len() > 0]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(
                f"no usable videos in {video_index_path} "
                f"(stride_high={stride_high}, require_mask_path={require_mask_path})"
            )

        self.df = df
        self.stride_low = stride_low
        self.stride_high = stride_high
        self.rng = np.random.default_rng(seed)
        self.filter_stats = {
            "raw_rows": self.raw_rows,
            "usable_videos": int(len(df)),
            "stride_range": [stride_low, stride_high],
            "require_mask_path": bool(require_mask_path),
        }
        self.by_dataset: Dict[str, np.ndarray] = {
            str(ds): g.index.to_numpy(dtype=np.int64)
            for ds, g in df.groupby(df["dataset"].astype(str), sort=True)
        }
        self.datasets: List[str] = sorted(self.by_dataset)

    def __len__(self) -> int:
        return len(self.df)

    def sample(self, batch_size: int) -> Tuple[pd.DataFrame, dict]:
        """Sample ``batch_size`` (video, start, stride) rows."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        replace = len(self.df) < batch_size
        picks = self.rng.choice(len(self.df), size=batch_size, replace=replace)
        rows = self.df.iloc[picks].copy().reset_index(drop=True)

        # Per-row random stride and start, mirroring lam_project/dataset.py.
        strides = self.rng.integers(
            self.stride_low, self.stride_high + 1, size=batch_size
        )
        starts = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            total = int(rows.at[i, "total_frames"])
            max_start = max(0, total - 2 * int(strides[i]))
            starts[i] = int(self.rng.integers(0, max_start + 1))
        rows["stride"] = strides.astype(np.int64)
        rows["start"] = starts
        rows["frame_i"] = rows["start"].astype(np.int64)
        rows["frame_j"] = (rows["start"] + rows["stride"]).astype(np.int64)

        mix = rows["dataset"].astype(str).value_counts().to_dict()
        meta = {
            "dataset_mix": mix,
            "n_rows": int(len(rows)),
            "stride_counts": {
                int(k): int(v)
                for k, v in pd.Series(strides).value_counts().sort_index().items()
            },
        }
        return rows, meta


class IndexedPairSampler:
    """Random transition sampler for aligned.

    The sampler keeps labels out of the training objective. Optional
    dataset-balanced sampling is allowed because it uses source identity only,
    not action labels.
    """

    def __init__(
        self,
        pair_index_path: Path,
        *,
        seed: int = 0,
        dataset_balanced: bool = True,
        require_real_pair: bool = True,
        require_masked_rows: bool = True,
        require_robosam_m7_pass: bool = True,
        min_mask_coverage: float = 0.0,
        min_bg_mask_coverage: float = 0.0,
    ) -> None:
        df = pd.read_parquet(pair_index_path)
        self.raw_rows = int(len(df))
        required = ["video_path", "frame_i", "frame_j"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"pair_index missing required column: {col}")

        if "pair_id" not in df.columns:
            df = df.copy()
            df["pair_id"] = np.arange(len(df), dtype=np.int64)
        if "dataset" not in df.columns:
            df = df.copy()
            df["dataset"] = "unknown"
        if "episode_id" not in df.columns:
            df = df.copy()
            df["episode_id"] = df["video_path"].astype(str)

        if require_real_pair:
            if "pair_type" in df.columns:
                df = df[df["pair_type"].astype(str) == "real"]
            else:
                df = df[df["frame_i"].astype(int) != df["frame_j"].astype(int)]

        # Do NOT filter on valid_relation_pair. In the core/A1 pair index that
        # column means "usable for the labelled relation graph", so it depends
        # on label-derived pools. aligned's no-label claim would be invalid if the
        # training pool were silently gated by it.

        if require_masked_rows:
            mask_filter = np.ones(len(df), dtype=bool)
            if "has_robosam_mask" in df.columns:
                mask_filter &= (
                    df["has_robosam_mask"].fillna(False).astype(bool).to_numpy()
                )
            if "robosam_interaction_mask_path" in df.columns:
                has_path = (
                    df["robosam_interaction_mask_path"].fillna("").astype(str).str.len()
                    > 0
                )
                mask_filter &= has_path.to_numpy()
            if require_robosam_m7_pass and "robosam_m7_pass" in df.columns:
                mask_filter &= (
                    df["robosam_m7_pass"].fillna(False).astype(bool).to_numpy()
                )
            if "robosam_mask_training_eligible" in df.columns:
                # This is mask-quality eligibility, not label-relation eligibility.
                mask_filter &= (
                    df["robosam_mask_training_eligible"]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy()
                )
            if min_mask_coverage > 0.0 and "mask_coverage" in df.columns:
                mask_filter &= pd.to_numeric(
                    df["mask_coverage"], errors="coerce"
                ).fillna(0.0).to_numpy() >= float(min_mask_coverage)
            if min_bg_mask_coverage > 0.0 and "bg_mask_coverage" in df.columns:
                mask_filter &= pd.to_numeric(
                    df["bg_mask_coverage"], errors="coerce"
                ).fillna(0.0).to_numpy() >= float(min_bg_mask_coverage)
            df = df[mask_filter]

        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"no usable aligned rows in {pair_index_path}")

        self.df = df
        self.filter_stats = {
            "raw_rows": self.raw_rows,
            "usable_rows": int(len(df)),
            "require_real_pair": bool(require_real_pair),
            "require_masked_rows": bool(require_masked_rows),
            "require_robosam_m7_pass": bool(require_robosam_m7_pass),
            "min_mask_coverage": float(min_mask_coverage),
            "min_bg_mask_coverage": float(min_bg_mask_coverage),
        }
        self.rng = np.random.default_rng(seed)
        self.dataset_balanced = bool(dataset_balanced)
        self.by_dataset: Dict[str, np.ndarray] = {
            str(ds): g.index.to_numpy(dtype=np.int64)
            for ds, g in df.groupby(df["dataset"].astype(str), sort=True)
        }
        self.datasets: List[str] = sorted(self.by_dataset)

    def __len__(self) -> int:
        return len(self.df)

    def sample(self, batch_size: int) -> Tuple[pd.DataFrame, dict]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.dataset_balanced and len(self.datasets) > 1:
            picks: List[int] = []
            per_ds = max(1, batch_size // len(self.datasets))
            for ds in self.datasets:
                pool = self.by_dataset[ds]
                n = min(per_ds, batch_size - len(picks))
                if n <= 0:
                    break
                replace = len(pool) < n
                picks.extend(
                    int(x) for x in self.rng.choice(pool, size=n, replace=replace)
                )
            while len(picks) < batch_size:
                ds = str(self.rng.choice(self.datasets))
                pool = self.by_dataset[ds]
                picks.append(int(self.rng.choice(pool)))
            self.rng.shuffle(picks)
        else:
            replace = len(self.df) < batch_size
            picks = [
                int(x)
                for x in self.rng.choice(len(self.df), size=batch_size, replace=replace)
            ]

        rows = self.df.iloc[picks].reset_index(drop=True)
        mix = rows["dataset"].astype(str).value_counts().to_dict()
        meta = {"dataset_mix": mix, "n_rows": int(len(rows))}
        return rows, meta


_F2M_IDX_CACHE: Dict[str, np.ndarray] = {}


def _load_frame_to_mask_idx(path: str) -> np.ndarray | None:
    """LRU-lite cache for the tiny ``frame_to_mask_idx.npy`` (~80 int16 per clip)."""
    if not path:
        return None
    cached = _F2M_IDX_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        arr = np.load(path)
    except Exception:
        return None
    _F2M_IDX_CACHE[path] = arr
    if len(_F2M_IDX_CACHE) > 4096:
        # cheap eviction: drop oldest 25%
        for k in list(_F2M_IDX_CACHE.keys())[: len(_F2M_IDX_CACHE) // 4]:
            _F2M_IDX_CACHE.pop(k, None)
    return arr


def load_foreground_background_masks(
    rows: pd.DataFrame,
    cache: FgBgMaskCache,
    target_hw: Tuple[int, int] = (240, 320),
    *,
    min_fg_pixels: int = 64,
    min_bg_pixels: int = 64,
    frame_col: str = "frame_j",
    runtime_dilate_px: int = 0,
    mask_temporal_radius: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load foreground/background masks for aligned rows at ``frame_col`` (default ``frame_j``).

    Works for both ``IndexedPairSampler`` rows (columns ``robosam_interaction_mask_path``,
    ``frame_j``) and ``VideoPairSampler`` rows (column ``mask_npz_path``, ``frame_j``
    derived as ``start + stride`` at sample time).

    **cleandata path**: when the row has a ``frame_to_mask_idx_path`` column,
    the mask index is remapped via ``frame_to_mask_idx[frame_j]`` so that the
    sampler can pick frames in the full source-mp4 domain (e.g. 80 frames)
    while the mask npz only has 49 SAM-aligned frames. Source-mp4 frame ``t``
    maps to ``f2m_idx[t]`` in mask-frame space.

    **mask_temporal_radius**: optional temporal safety window in mask-frame
    space. Foreground uses an OR over ``[idx-r, idx+r]`` so a slightly stale or
    under-segmented interaction mask does not erase action pixels. Background
    uses an AND over the same window, so only pixels safe across all nearby
    masks remain background-safe.

    **runtime_dilate_px**: optional extra dilation kernel applied to the
    interact mask (and equivalent erosion to bsafe) at load time. Default 0
    (off). This is applied after the optional temporal merge.

    Returns ``fg``, ``bg`` and ``valid``. Missing mask paths or out-of-range
    frames produce ``valid=False`` for that row, in which case the trainer
    falls back to view-contrastive / null / KL losses without fg/bg.
    """
    H, W = target_hw
    n = len(rows)
    fg_out = np.zeros((n, H, W), dtype=np.float32)
    bg_out = np.zeros((n, H, W), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    temporal_radius = max(0, int(mask_temporal_radius))
    dilate_kernel = None
    if runtime_dilate_px and runtime_dilate_px > 0:
        k = int(runtime_dilate_px)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for i, r in enumerate(rows.itertuples(index=False)):
        path = ""
        # VideoPairSampler unified column name.
        if hasattr(r, "mask_npz_path"):
            path = str(getattr(r, "mask_npz_path") or "")
        elif hasattr(r, "robosam_interaction_mask_path"):
            path = str(getattr(r, "robosam_interaction_mask_path") or "")
        elif hasattr(r, "mask_path"):
            path = str(getattr(r, "mask_path") or "")
        if not path:
            continue
        interact_arr, bsafe_arr = cache.get(path)
        if interact_arr is None or bsafe_arr is None:
            continue
        fj = int(getattr(r, frame_col))
        # cleandata path: remap source-frame index → mask-frame index.
        if hasattr(r, "frame_to_mask_idx_path"):
            f2m_path = str(getattr(r, "frame_to_mask_idx_path") or "")
            if f2m_path:
                f2m = _load_frame_to_mask_idx(f2m_path)
                if f2m is None or fj < 0 or fj >= f2m.shape[0]:
                    continue
                fj = int(f2m[fj])
        if fj < 0 or fj >= interact_arr.shape[0]:
            continue
        if temporal_radius > 0:
            lo = max(0, fj - temporal_radius)
            hi = min(int(interact_arr.shape[0]), fj + temporal_radius + 1)
            fg_src = np.any(interact_arr[lo:hi] > 0, axis=0).astype(np.uint8)
            bg_src = np.all(bsafe_arr[lo:hi] > 0, axis=0).astype(np.uint8)
        else:
            fg_src = interact_arr[fj]
            bg_src = bsafe_arr[fj]
        fg = _resize_mask_nearest(fg_src, (H, W)).astype(np.float32)
        bg = _resize_mask_nearest(bg_src, (H, W)).astype(np.float32)
        if dilate_kernel is not None:
            fg = cv2.dilate((fg > 0).astype(np.uint8), dilate_kernel).astype(np.float32)
            bg = cv2.erode((bg > 0).astype(np.uint8), dilate_kernel).astype(np.float32)
        if fg.sum() < float(min_fg_pixels) or bg.sum() < float(min_bg_pixels):
            continue
        fg_out[i] = (fg > 0).astype(np.float32)
        bg_out[i] = (bg > 0).astype(np.float32)
        valid[i] = True
    return fg_out, bg_out, valid


__all__ = [
    "OFFICIAL_LAM_HW",
    "OFFICIAL_WM_HW",
    "IndexedPairSampler",
    "VideoPairSampler",
    "center_crop_to_official_ratio",
    "decode_rows_parallel",
    "decode_rows_random_stride",
    "load_foreground_background_masks",
    "patch_layernorm_to_fp32",
    "resize_frame_to_official_lam",
]
