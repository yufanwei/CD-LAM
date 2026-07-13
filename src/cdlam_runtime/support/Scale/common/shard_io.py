"""Read clips from the unified data-recipe shards and decode embedded MP4 frames.

This module replaces the legacy file-based video path with in-memory decoding
from ``video_mp4`` bytes. Frame rate is derived from timestamps, and verbs are
read from ``step_skills`` or ``step_actions``.

Shard schema (datasets/recipe/<source>/shard-*.parquet):
  episode_id:int64  task_id:int32  num_frames:int32
  video_mp4:bytes
  proprio_h5_raw:bytes
  timestamp:list[int]
  intrinsic_*/extrinsic_*
  step_starts/step_ends:list[int]   step_actions/step_skills:list[str]

The resize path matches the official center-crop and two-stage interpolation
used by WM/LAM training.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np

# Data-recipe root. Keep the default repository-relative and override it when
# shards live on external storage.
SHARD_ROOT = Path(os.environ.get("CDLAM_DATA_RECIPE_ROOT", "data/recipe"))

OFFICIAL_WM_HW = (480, 640)
OFFICIAL_LAM_HW = (240, 320)

_NEEDED_COLS = (
    "episode_id",
    "task_id",
    "num_frames",
    "video_mp4",
    "timestamp",
    "step_starts",
    "step_ends",
    "step_actions",
    "step_skills",
)


# ============================================================================
# Official crop/resize path. Keep byte-aligned with the WM data pipeline.
# ============================================================================
def _center_crop_to_official_ratio(
    arr: np.ndarray, target_ratio: float = 640 / 480
) -> np.ndarray:
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid frame shape: {arr.shape[:2]}")
    cur = w / h
    if cur > target_ratio:
        th, tw = h, max(1, int(h * target_ratio))
    elif cur < target_ratio:
        th, tw = max(1, int(w / target_ratio)), w
    else:
        th, tw = h, w
    y0, x0 = max(0, (h - th) // 2), max(0, (w - tw) // 2)
    return arr[y0 : y0 + th, x0 : x0 + tw, ...]


def resize_frame_to_official_lam(
    frame_rgb: np.ndarray, target_hw=OFFICIAL_LAM_HW
) -> np.ndarray:
    """Center-crop to 4:3, resize through 480x640, then to ``target_hw``."""
    import cv2  # Lazy import keeps non-CV helpers usable in provider environments.

    H, W = target_hw
    cropped = _center_crop_to_official_ratio(frame_rgb)
    if cropped.shape[:2] != OFFICIAL_WM_HW:
        cropped = cv2.resize(
            cropped,
            (OFFICIAL_WM_HW[1], OFFICIAL_WM_HW[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if (H, W) == OFFICIAL_WM_HW:
        return cropped.astype(np.uint8, copy=False)
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


# ============================================================================
# Shard path and row access with projected row-group reads and an LRU cache.
# ============================================================================
def resolve_shard_path(source: str, shard_name: str) -> Path:
    """Return the absolute parquet path for a source and shard name."""
    return SHARD_ROOT / source / shard_name


@lru_cache(maxsize=int(os.environ.get("SHARD_HANDLE_CACHE", "256")))
def _parquet_handle(shard_path: str):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard_path)
    offs, cum = [], 0
    for rg in range(pf.num_row_groups):
        offs.append(cum)
        cum += pf.metadata.row_group(rg).num_rows
    return pf, offs


# Cache full rows to avoid repeated random parquet reads.
_ROW_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_ROW_CACHE_MAX = int(os.environ.get("SHARD_ROW_CACHE", "512"))
_ROW_CACHE_LOCK = threading.Lock()  # Protect the cache during parallel decode.


def read_shard_row(
    shard_path: str | Path, row_index: int, columns: Iterable[str] = _NEEDED_COLS
) -> dict:
    """Read projected columns for one row using row-group lookup and LRU caching."""
    shard_path = str(shard_path)
    key = (shard_path, int(row_index))
    with _ROW_CACHE_LOCK:
        hit = _ROW_CACHE.get(key)
        if hit is not None:
            _ROW_CACHE.move_to_end(key)
            return hit
    # Perform parquet I/O outside the lock so cache misses can run in parallel.
    pf, offs = _parquet_handle(shard_path)
    rg = 0
    for i in range(len(offs) - 1, -1, -1):
        if row_index >= offs[i]:
            rg = i
            break
    tbl = pf.read_row_group(rg, columns=list(columns))
    local = int(row_index) - offs[rg]
    row = tbl.slice(local, 1).to_pylist()[0]
    with _ROW_CACHE_LOCK:
        _ROW_CACHE[key] = row
        if len(_ROW_CACHE) > _ROW_CACHE_MAX:
            _ROW_CACHE.popitem(last=False)
    return row


# ============================================================================
# Frame-rate derivation and verb extraction.
# ============================================================================
def derive_fps(
    timestamp: list[int] | np.ndarray, num_frames: int | None = None
) -> float:
    """Derive native FPS from timestamps, detecting ns, us, ms, or seconds."""
    t = np.asarray(timestamp, dtype=np.float64)
    if t.size < 2:
        return 30.0
    span = float(t[-1] - t[0])
    if span <= 0:
        return 30.0
    # Infer units from the median adjacent timestamp interval.
    step = np.median(np.diff(t))
    if step >= 1e6:  # nanoseconds
        unit = 1e-9
    elif step >= 1e3:  # microseconds
        unit = 1e-6
    elif step >= 1.0:  # milliseconds
        unit = 1e-3
    else:  # seconds
        unit = 1.0
    fps = (t.size - 1) / (span * unit)
    # Reject implausible values.
    return float(fps) if 1.0 <= fps <= 240.0 else 30.0


def clip_verbs_raw(row: dict) -> tuple[list[str], list[str]]:
    """Return ``(step_skills, step_actions)`` for downstream normalization."""
    skills = [s for s in (row.get("step_skills") or []) if s]
    actions = [a for a in (row.get("step_actions") or []) if a]
    return skills, actions


# ============================================================================
# Decode MP4 bytes in memory with PyAV.
# ============================================================================
def decode_frames_from_mp4_bytes(
    mp4_bytes: bytes, frame_idxs: Iterable[int]
) -> dict[int, np.ndarray]:
    """Decode selected frames into native-resolution RGB uint8 arrays.

    Sequential frame counting avoids incorrect PTS-to-index assumptions for
    variable-rate and B-frame videos. Per-stream threading stays disabled
    because clips are already decoded concurrently by the outer executor.
    """
    import av  # pyav 16

    want = sorted(set(int(i) for i in frame_idxs))
    if not want:
        return {}
    need_max = want[-1]
    want_set = set(want)
    out: dict[int, np.ndarray] = {}
    with av.open(io.BytesIO(mp4_bytes)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"  # Avoid nested libav threading races.
        for i, frame in enumerate(container.decode(stream)):
            if i in want_set:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= need_max:
                break
    # Pad a genuinely short clip with its last decoded frame.
    if out and need_max not in out:
        last = out[max(out)]
        for idx in want:
            out.setdefault(idx, last)
    return out


def _as_uint8_rgb_video(frames: np.ndarray) -> np.ndarray:
    """Validate a ``(T,H,W,3)`` video and preserve uint8 interpolation input."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) RGB video, got shape={frames.shape}")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    return np.clip(frames, 0, 255).astype(np.uint8, copy=False)


def _center_crop_video(frames: np.ndarray) -> np.ndarray:
    """Center-crop a video to the 640:480 world-model aspect ratio."""

    frames = _as_uint8_rgb_video(frames)
    h, w = frames.shape[1:3]
    aspect = OFFICIAL_WM_HW[1] / OFFICIAL_WM_HW[0]
    if w / h > aspect:
        target_h, target_w = h, int(h * aspect)
    elif w / h < aspect:
        target_h, target_w = int(w / aspect), w
    else:
        target_h, target_w = h, w
    h0, w0 = (h - target_h) // 2, (w - target_w) // 2
    return np.ascontiguousarray(frames[:, h0 : h0 + target_h, w0 : w0 + target_w])


def _resize_uint8_video(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Apply the release-pinned uint8 bilinear interpolation contract."""

    import torch
    import torch.nn.functional as functional

    frames = _as_uint8_rgb_video(frames)
    height, width = map(int, hw)
    if frames.shape[1:3] == (height, width):
        return np.ascontiguousarray(frames)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    tensor = functional.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def _bundled_official_wm(
    raw_frames: np.ndarray,
    wm_hw: tuple[int, int] = OFFICIAL_WM_HW,
) -> np.ndarray:
    return _resize_uint8_video(_center_crop_video(raw_frames), wm_hw)


def _bundled_official_lam(
    raw_frames: np.ndarray,
    lam_hw: tuple[int, int] = OFFICIAL_LAM_HW,
    wm_hw: tuple[int, int] = OFFICIAL_WM_HW,
) -> np.ndarray:
    return _resize_uint8_video(_bundled_official_wm(raw_frames, wm_hw), lam_hw)


@lru_cache(maxsize=1)
def _official_lam():
    """Return the self-contained, release-pinned raw-to-LAM transform."""

    return _bundled_official_lam


@lru_cache(maxsize=1)
def _official_wm():
    """Return the self-contained, release-pinned raw-to-WM transform."""

    return _bundled_official_wm


def decode_clip_frames(
    shard_path: str | Path,
    row_index: int,
    frame_idxs: Iterable[int],
    target_hw=OFFICIAL_LAM_HW,
    resize: bool = True,
) -> dict[int, np.ndarray]:
    """Decode selected shard frames and optionally apply official LAM resizing."""
    row = read_shard_row(shard_path, row_index)
    frames = decode_frames_from_mp4_bytes(row["video_mp4"], frame_idxs)
    if not resize or not frames:
        return frames
    keys = sorted(frames)
    vid = np.stack([frames[k] for k in keys], axis=0)  # (T,H,W,3) native
    lam = _official_lam()(vid, lam_hw=target_hw)  # Official crop/resize.
    return {k: lam[i] for i, k in enumerate(keys)}


def decode_window_from_shard(
    shard_path: str | Path,
    row_index: int,
    start_frame: int,
    stop_frame: int,
    wm_hw=OFFICIAL_WM_HW,
) -> np.ndarray:
    """Decode a contiguous shard window into official WM uint8 video.

    The result is pixel-aligned with ``decode_window_official`` because both
    paths use ``official_wm_video_from_raw``. Only the raw-frame source differs.
    """
    s, e = int(start_frame), int(stop_frame)
    if e <= s:
        raise ValueError(f"bad window [{s}, {e}) for {shard_path}::{row_index}")
    row = read_shard_row(shard_path, row_index)
    frames = decode_frames_from_mp4_bytes(row["video_mp4"], range(s, e))  # native, dict
    if not frames:
        raise RuntimeError(
            f"no frames decoded from {shard_path}::{row_index} [{s}, {e})"
        )
    raw = np.stack([frames[k] for k in sorted(frames)], axis=0)  # (T,H,W,3) native
    return _official_wm()(raw, wm_hw=tuple(wm_hw))  # (T, wm_h, wm_w, 3) uint8


# ============================================================================
# Contract-compatible replacement for the bundled LAM pair decoder.
# Required columns: shard_path, row_index, frame_i, frame_j.
# Returns: uint8 (N,2,H,W,3), valid bool (N,).
# ============================================================================
def decode_rows_from_shard(rows, target_hw=OFFICIAL_LAM_HW, workers: int = 16):
    from concurrent.futures import ThreadPoolExecutor

    H, W = target_hw
    n = len(rows)
    out = np.zeros((n, 2, H, W, 3), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)

    def _one(args):
        i, sp, ri, fi, fj = args
        try:
            fr = decode_clip_frames(sp, ri, (fi, fj), target_hw, resize=True)
            a = fr.get(int(fi))
            b = fr.get(int(fj), a)
            if a is None or b is None:
                return i, None
            return i, np.stack([a, b], axis=0).astype(np.uint8)
        except Exception:
            return i, None

    pool = [
        (i, r.shard_path, int(r.row_index), int(r.frame_i), int(r.frame_j))
        for i, r in enumerate(rows.itertuples(index=False))
    ]
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        for i, arr in ex.map(_one, pool):
            if arr is not None:
                out[i] = arr
                valid[i] = True
    return out, valid


if __name__ == "__main__":
    # Manual smoke check for the first row of an AgiBot Beta shard.
    import sys

    sp = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(resolve_shard_path("agibot_beta", "shard-00000.parquet"))
    )
    row = read_shard_row(sp, 0)
    fps = derive_fps(row["timestamp"], row["num_frames"])
    skills, actions = clip_verbs_raw(row)
    print(
        f"num_frames={row['num_frames']} fps={fps:.2f} skills={skills[:3]} actions={actions[:1]}"
    )
    fr = decode_clip_frames(sp, 0, [0, row["num_frames"] // 2, row["num_frames"] - 1])
    print({k: v.shape for k, v in fr.items()})
