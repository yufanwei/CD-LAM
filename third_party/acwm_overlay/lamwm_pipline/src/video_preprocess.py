"""upstream ACWM raw-video preprocessing helpers.

Official raw video datasets first center-crop frames to the WM aspect ratio
(640 / 480), resize to the WM resolution, then derive LAM inputs by resizing
that WM video to 240x320. Keep these helpers small and dependency-light so the
LAM benchmark, z-cache builder, WM trainer, and preview runners share one
protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


OFFICIAL_WM_HW: tuple[int, int] = (480, 640)
OFFICIAL_LAM_HW: tuple[int, int] = (240, 320)
OFFICIAL_ASPECT: float = OFFICIAL_WM_HW[1] / OFFICIAL_WM_HW[0]


def _as_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) RGB video, got shape={frames.shape}")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    return np.clip(frames, 0, 255).astype(np.uint8, copy=False)


def center_crop_to_aspect(
    frames: np.ndarray, aspect: float = OFFICIAL_ASPECT
) -> np.ndarray:
    """Center-crop (T,H,W,3) frames to the official 4:3 WM aspect."""
    frames = _as_uint8_rgb(frames)
    h, w = frames.shape[1:3]
    if w / h > aspect:
        target_h = h
        target_w = int(h * aspect)
    elif w / h < aspect:
        target_h = int(w / aspect)
        target_w = w
    else:
        target_h = h
        target_w = w
    h0 = (h - target_h) // 2
    w0 = (w - target_w) // 2
    return np.ascontiguousarray(frames[:, h0 : h0 + target_h, w0 : w0 + target_w])


def resize_uint8_video(frames: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize (T,H,W,3) uint8 video to (T,hw[0],hw[1],3)."""
    frames = _as_uint8_rgb(frames)
    h, w = int(hw[0]), int(hw[1])
    if frames.shape[1] == h and frames.shape[2] == w:
        return np.ascontiguousarray(frames)
    # Keep uint8 interpolation semantics to match the official dataloaders, which
    # pass uint8 tensors directly into F.interpolate before dividing lam_video by 255.
    t = torch.from_numpy(frames).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def official_wm_video_from_raw(
    raw_frames: np.ndarray,
    wm_hw: Tuple[int, int] = OFFICIAL_WM_HW,
) -> np.ndarray:
    """Raw RGB frames -> official WM video: center-crop 4:3, then resize."""
    return resize_uint8_video(center_crop_to_aspect(raw_frames), wm_hw)


def official_lam_video_from_wm(
    wm_frames: np.ndarray,
    lam_hw: Tuple[int, int] = OFFICIAL_LAM_HW,
) -> np.ndarray:
    """Official/cropped WM video -> LAM-resolution video used for z extraction."""
    return resize_uint8_video(wm_frames, lam_hw)


def official_lam_video_from_raw(
    raw_frames: np.ndarray,
    lam_hw: Tuple[int, int] = OFFICIAL_LAM_HW,
    wm_hw: Tuple[int, int] = OFFICIAL_WM_HW,
) -> np.ndarray:
    """Raw RGB frames -> official LAM video via the same WM crop/resize path."""
    wm = official_wm_video_from_raw(raw_frames, wm_hw=wm_hw)
    return official_lam_video_from_wm(wm, lam_hw=lam_hw)


def decode_video_raw(video: Path | str, n_frames: int) -> np.ndarray:
    """Decode the first n_frames without spatial resize/crop."""
    frames = iio.imread(str(video), plugin="pyav")
    if frames.ndim != 4 or frames.shape[0] < n_frames:
        have = frames.shape[0] if frames.ndim == 4 else "bad-shape"
        raise RuntimeError(f"video too short: {video} has {have}, need {n_frames}")
    return _as_uint8_rgb(frames[:n_frames])


def decode_window_raw(
    video: Path | str, start_frame: int, stop_frame: int
) -> np.ndarray:
    """Decode [start_frame, stop_frame) without spatial resize/crop."""
    import av

    n_need = int(stop_frame) - int(start_frame)
    if n_need <= 0:
        raise ValueError(f"bad window [{start_frame}, {stop_frame})")
    try:
        container = av.open(str(video))
        stream = container.streams.video[0]
        avg_rate = float(stream.average_rate or stream.base_rate or 30)
        time_base = stream.time_base
        target_pts = int(start_frame / avg_rate / time_base)
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        out = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_idx = int(round(float(frame.pts * time_base) * avg_rate))
            if frame_idx < start_frame:
                continue
            if frame_idx >= stop_frame:
                break
            out.append(frame.to_ndarray(format="rgb24"))
            if len(out) >= n_need:
                break
        container.close()
        if len(out) < n_need:
            raise RuntimeError(f"av seek path got {len(out)} frames, need {n_need}")
        return _as_uint8_rgb(np.stack(out))
    except Exception as e:
        frames = iio.imread(str(video), plugin="pyav")
        if frames.ndim != 4 or frames.shape[0] < stop_frame:
            have = frames.shape[0] if frames.ndim == 4 else "bad-shape"
            raise RuntimeError(
                f"video {video} too short for [{start_frame}, {stop_frame}): "
                f"have {have}, av-error={e}"
            )
        return _as_uint8_rgb(frames[start_frame:stop_frame])


def decode_video_official(
    video: Path | str,
    n_frames: int,
    wm_hw: Tuple[int, int] = OFFICIAL_WM_HW,
) -> np.ndarray:
    """Decode first n_frames and apply official WM preprocessing."""
    return official_wm_video_from_raw(decode_video_raw(video, n_frames), wm_hw=wm_hw)


def decode_window_official(
    video: Path | str,
    start_frame: int,
    stop_frame: int,
    wm_hw: Tuple[int, int] = OFFICIAL_WM_HW,
) -> np.ndarray:
    """Decode a frame window and apply official WM preprocessing."""
    return official_wm_video_from_raw(
        decode_window_raw(video, start_frame, stop_frame),
        wm_hw=wm_hw,
    )


def lam_pairs_from_official_lam_video(lam_video: np.ndarray) -> np.ndarray:
    """(T,H,W,3) lam video -> adjacent pairs (T-1,2,H,W,3)."""
    lam_video = _as_uint8_rgb(lam_video)
    if lam_video.shape[0] < 2:
        return np.zeros(
            (0, 2, lam_video.shape[1], lam_video.shape[2], 3), dtype=np.uint8
        )
    return np.stack(
        [lam_video[i : i + 2] for i in range(lam_video.shape[0] - 1)], axis=0
    )


def iter_official_adjacent_pairs(
    raw_frames: np.ndarray,
    starts: Iterable[int],
    lam_hw: Tuple[int, int] = OFFICIAL_LAM_HW,
    wm_hw: Tuple[int, int] = OFFICIAL_WM_HW,
) -> Iterable[tuple[np.ndarray, int, int]]:
    """Yield official LAM-resolution pairs for selected adjacent start indices."""
    lam_video = official_lam_video_from_raw(raw_frames, lam_hw=lam_hw, wm_hw=wm_hw)
    for i in starts:
        yield lam_video[i : i + 2], int(i), int(i) + 1
