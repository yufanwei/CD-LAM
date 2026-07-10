"""Foreground Displacement Chamfer Error (FDCE), implemented in NumPy.

This module scores already-computed point tracks.  Mask generation and point
tracking are deliberately outside the metric, so the implementation has no
dependency on a segmentation model or a particular tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FDCEResult:
    """FDCE score and the visibility-filtered scoring counts."""

    score: float
    generated_to_reference: float
    reference_to_generated: float
    generated_tracks: int
    reference_tracks: int
    valid_pairs: int


def _as_tracks(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (T, N, 2), got {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain frame 0 and at least one rollout frame")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one point track")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real numeric array, got dtype {array.dtype}")
    return np.asarray(array, dtype=np.float64)


def _as_visibility(
    name: str,
    value: Optional[ArrayLike],
    tracks: FloatArray,
    *,
    threshold: float,
) -> BoolArray:
    finite_coordinates = np.isfinite(tracks).all(axis=-1)
    if value is None:
        return finite_coordinates
    visibility = np.asarray(value)
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"{name} must have shape {tracks.shape[:2]}, got {visibility.shape}"
        )
    if visibility.dtype == np.bool_:
        visible = visibility.copy()
    else:
        if not np.issubdtype(visibility.dtype, np.number) or np.iscomplexobj(visibility):
            raise TypeError(f"{name} must be boolean or real-valued confidence scores")
        visible = np.isfinite(visibility) & (visibility >= threshold)
    return np.asarray(visible & finite_coordinates, dtype=bool)


def _filter_tracks(
    tracks: FloatArray,
    visibility: BoolArray,
    *,
    min_visible_fraction: float,
) -> tuple[FloatArray, BoolArray]:
    # Frame 0 is required because every trajectory is expressed relative to it.
    keep = visibility[0] & (visibility.mean(axis=0) >= min_visible_fraction)
    return tracks[:, keep], visibility[:, keep]


def pairwise_displacement_costs(
    generated_tracks: ArrayLike,
    reference_tracks: ArrayLike,
    generated_visibility: Optional[ArrayLike] = None,
    reference_visibility: Optional[ArrayLike] = None,
    *,
    visibility_threshold: float = 0.5,
    min_visible_fraction: float = 0.8,
    min_common_frames: int = 1,
) -> FloatArray:
    """Compute paper Eq. A.3 for every generated/reference track pair.

    Tracks have shape ``(T, N, 2)``.  Each trajectory is first translated by
    its own frame-0 point.  Pair cost is then the mean Euclidean displacement
    error over rollout frames where both tracks are visible.  Frame 0 itself is
    not averaged because Eq. A.3 sums rollout steps ``1..H``.

    Low-visibility tracks are discarded before pair construction, matching the
    evaluation protocol.  A pair without ``min_common_frames`` jointly visible
    rollout frames receives ``NaN``.
    """

    generated = _as_tracks("generated_tracks", generated_tracks)
    reference = _as_tracks("reference_tracks", reference_tracks)
    if generated.shape[0] != reference.shape[0]:
        raise ValueError(
            "generated_tracks and reference_tracks must have the same number of frames, "
            f"got {generated.shape[0]} and {reference.shape[0]}"
        )
    if not np.isfinite(visibility_threshold):
        raise ValueError("visibility_threshold must be finite")
    if not np.isfinite(min_visible_fraction) or not 0 <= min_visible_fraction <= 1:
        raise ValueError("min_visible_fraction must lie in [0, 1]")
    if isinstance(min_common_frames, bool) or int(min_common_frames) != min_common_frames:
        raise ValueError("min_common_frames must be a positive integer")
    min_common_frames = int(min_common_frames)
    if min_common_frames < 1:
        raise ValueError("min_common_frames must be a positive integer")
    if min_common_frames > generated.shape[0] - 1:
        raise ValueError(
            "min_common_frames cannot exceed the number of rollout frames "
            f"({generated.shape[0] - 1})"
        )

    generated_vis = _as_visibility(
        "generated_visibility",
        generated_visibility,
        generated,
        threshold=visibility_threshold,
    )
    reference_vis = _as_visibility(
        "reference_visibility",
        reference_visibility,
        reference,
        threshold=visibility_threshold,
    )
    generated, generated_vis = _filter_tracks(
        generated,
        generated_vis,
        min_visible_fraction=min_visible_fraction,
    )
    reference, reference_vis = _filter_tracks(
        reference,
        reference_vis,
        min_visible_fraction=min_visible_fraction,
    )
    if generated.shape[1] == 0:
        raise ValueError("no generated tracks survive the visibility filter")
    if reference.shape[1] == 0:
        raise ValueError("no reference tracks survive the visibility filter")

    generated_delta = generated - generated[0:1]
    reference_delta = reference - reference[0:1]
    distances = np.linalg.norm(
        generated_delta[1:, :, None, :] - reference_delta[1:, None, :, :],
        axis=-1,
    )
    jointly_visible = (
        generated_vis[1:, :, None]
        & reference_vis[1:, None, :]
        & np.isfinite(distances)
    )
    counts = jointly_visible.sum(axis=0)
    sums = np.where(jointly_visible, distances, 0.0).sum(axis=0)
    costs = np.full(counts.shape, np.nan, dtype=np.float64)
    valid = counts >= min_common_frames
    costs[valid] = sums[valid] / counts[valid]
    return costs


def symmetric_chamfer_from_costs(pair_costs: ArrayLike) -> tuple[float, float, float]:
    """Apply the symmetric Chamfer reduction in paper Eq. A.4."""

    costs = np.asarray(pair_costs, dtype=np.float64)
    if costs.ndim != 2 or 0 in costs.shape:
        raise ValueError(f"pair_costs must be a non-empty 2D matrix, got {costs.shape}")
    if np.isinf(costs).any():
        raise ValueError("pair_costs must not contain infinity")

    finite = np.isfinite(costs)
    if not finite.any(axis=1).all():
        bad = np.flatnonzero(~finite.any(axis=1)).tolist()
        raise ValueError(f"generated track(s) have no valid reference comparison: {bad}")
    if not finite.any(axis=0).all():
        bad = np.flatnonzero(~finite.any(axis=0)).tolist()
        raise ValueError(f"reference track(s) have no valid generated comparison: {bad}")

    safe = np.where(finite, costs, np.inf)
    generated_to_reference = float(safe.min(axis=1).mean())
    reference_to_generated = float(safe.min(axis=0).mean())
    score = 0.5 * (generated_to_reference + reference_to_generated)
    return score, generated_to_reference, reference_to_generated


def foreground_displacement_chamfer_error(
    generated_tracks: ArrayLike,
    reference_tracks: ArrayLike,
    generated_visibility: Optional[ArrayLike] = None,
    reference_visibility: Optional[ArrayLike] = None,
    *,
    visibility_threshold: float = 0.5,
    min_visible_fraction: float = 0.8,
    min_common_frames: int = 1,
    return_details: bool = False,
) -> Union[float, FDCEResult]:
    """Compute FDCE from generated and reference foreground point tracks."""

    costs = pairwise_displacement_costs(
        generated_tracks,
        reference_tracks,
        generated_visibility,
        reference_visibility,
        visibility_threshold=visibility_threshold,
        min_visible_fraction=min_visible_fraction,
        min_common_frames=min_common_frames,
    )
    score, generated_to_reference, reference_to_generated = symmetric_chamfer_from_costs(costs)
    if not return_details:
        return score
    return FDCEResult(
        score=score,
        generated_to_reference=generated_to_reference,
        reference_to_generated=reference_to_generated,
        generated_tracks=int(costs.shape[0]),
        reference_tracks=int(costs.shape[1]),
        valid_pairs=int(np.isfinite(costs).sum()),
    )


def symmetric_chamfer_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Symmetric Euclidean Chamfer distance between two point clouds."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.ndim != 2 or second_array.ndim != 2:
        raise ValueError("point clouds must both have shape (N, D)")
    if first_array.shape[0] == 0 or second_array.shape[0] == 0:
        raise ValueError("point clouds must be non-empty")
    if first_array.shape[1] != second_array.shape[1]:
        raise ValueError("point clouds must share the coordinate dimension")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("point clouds must contain only finite coordinates")
    distances = np.linalg.norm(first_array[:, None] - second_array[None, :], axis=-1)
    score, _, _ = symmetric_chamfer_from_costs(distances)
    return score


# Short, conventional alias.
fdce = foreground_displacement_chamfer_error


__all__ = [
    "FDCEResult",
    "fdce",
    "foreground_displacement_chamfer_error",
    "pairwise_displacement_costs",
    "symmetric_chamfer_distance",
    "symmetric_chamfer_from_costs",
]
