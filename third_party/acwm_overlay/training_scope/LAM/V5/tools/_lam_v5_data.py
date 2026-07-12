"""LAM v5 data extensions — fg + bg mask loading.

V4 only loaded interact (fg) mask for L_rec_inter. V5 needs both fg AND bg
masks per row to compute:

  L_rec_fg  = fg-weighted MSE between pred and o_{t+1}
  L_rec_bg  = bg-weighted MSE between pred and o_t   (bg consistency)

The mask file (`masks_m7.npz`) has 3 keys: ``interact`` (fg), ``bsafe`` (bg),
``motion``. We load interact + bsafe, both at frame_j (the prediction target).

Note: interact + bsafe ≠ 1 everywhere — there's an "uncertain" middle band where
neither holds. This middle is naturally ignored (zero weight in both losses).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)


def _resize_mask_nearest(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    if mask.shape == (H, W):
        return mask
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


class FgBgMaskCache:
    """LRU cache for masks_m7.npz — exposes both ``interact`` and ``bsafe``.

    Same API as :class:`LAM_V3.tools._lam_v3_data.MaskCache` but returns a
    (interact, bsafe) tuple instead of just interact.
    """

    def __init__(self, max_entries: int = 256) -> None:
        from collections import OrderedDict

        self.max_entries = int(max_entries)
        self._cache: "OrderedDict[str, Tuple[np.ndarray, np.ndarray] | None]" = (
            OrderedDict()
        )

    def get(self, npz_path: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if not npz_path:
            return None, None
        if npz_path in self._cache:
            self._cache.move_to_end(npz_path)
            cached = self._cache[npz_path]
            if cached is None:
                return None, None
            return cached
        try:
            with np.load(npz_path) as z:
                interact = z["interact"]
                bsafe = z["bsafe"]
        except Exception:
            self._cache[npz_path] = None
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return None, None
        if interact.dtype != np.uint8:
            interact = interact.astype(np.uint8)
        if bsafe.dtype != np.uint8:
            bsafe = bsafe.astype(np.uint8)
        self._cache[npz_path] = (interact, bsafe)
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return interact, bsafe

    def clear(self) -> None:
        self._cache.clear()


def load_fg_bg_masks_for_rows(
    rows: pd.DataFrame,
    cache: FgBgMaskCache,
    target_hw: Tuple[int, int] = (240, 320),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row fg + bg masks at ``frame_j``.

    Returns:
      fg_masks: float32 (N, H, W) in {0, 1} — interact mask
      bg_masks: float32 (N, H, W) in {0, 1} — bsafe mask
      valid:   bool (N,) — True iff fg has at least one positive AND bg has at
               least one positive AND mask file loaded ok.
    """
    H, W = target_hw
    n = len(rows)
    fg_out = np.zeros((n, H, W), dtype=np.float32)
    bg_out = np.zeros((n, H, W), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows.itertuples(index=False)):
        path = getattr(r, "robosam_interaction_mask_path", "") or ""
        interact_arr, bsafe_arr = cache.get(str(path))
        if interact_arr is None or bsafe_arr is None:
            continue
        fj = int(r.frame_j)
        if fj < 0 or fj >= interact_arr.shape[0]:
            continue
        fg = _resize_mask_nearest(interact_arr[fj], (H, W))
        bg = _resize_mask_nearest(bsafe_arr[fj], (H, W))
        if fg.sum() <= 0 or bg.sum() <= 0:
            continue
        fg_out[i] = fg.astype(np.float32)
        bg_out[i] = bg.astype(np.float32)
        valid[i] = True
    return fg_out, bg_out, valid


__all__ = ["FgBgMaskCache", "load_fg_bg_masks_for_rows"]
