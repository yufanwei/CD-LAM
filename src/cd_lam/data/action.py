"""Explicit robot-action transformation contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActionTransform:
    """Identity metadata that must match between bridge training and Stage 3."""

    transform_id: str
    source_stride: int

    def __post_init__(self) -> None:
        if not self.transform_id.strip():
            raise ValueError("transform_id must be non-empty")
        if isinstance(self.source_stride, bool) or self.source_stride < 1:
            raise ValueError("source_stride must be a positive integer")


def _validate_actions(actions: Tensor) -> None:
    if not isinstance(actions, Tensor):
        raise TypeError(f"actions must be a torch.Tensor, got {type(actions).__name__}")
    if actions.ndim < 2:
        raise ValueError(f"actions must have shape (..., T, D), got {tuple(actions.shape)}")
    if not actions.dtype.is_floating_point or actions.dtype.is_complex:
        raise TypeError(f"actions must be real floating point, got {actions.dtype}")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("actions contain NaN or infinity")


def minmax_normalize_absolute(
    actions: Tensor,
    minimum: Tensor,
    maximum: Tensor,
) -> Tensor:
    """Normalize absolute actions to ``[-1, 1]`` per action dimension."""

    _validate_actions(actions)
    for name, value in (("minimum", minimum), ("maximum", maximum)):
        if not isinstance(value, Tensor) or value.shape != (actions.shape[-1],):
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} must have shape ({actions.shape[-1]},), got {shape}")
        if value.device != actions.device:
            raise ValueError(f"{name} and actions must be on the same device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or infinity")
    minimum = minimum.to(actions.dtype)
    span = maximum.to(actions.dtype) - minimum
    if bool((span < 0).any()):
        raise ValueError("maximum must be greater than or equal to minimum")
    active = span > 0
    normalized = torch.zeros_like(actions)
    normalized[..., active] = (
        2.0 * (actions[..., active] - minimum[active]) / span[active] - 1.0
    )
    return normalized


def strided_deltas(actions: Tensor, *, stride: int = 1) -> Tensor:
    """Return ``action[t + stride] - action[t]`` along the penultimate axis."""

    _validate_actions(actions)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if actions.shape[-2] <= stride:
        raise ValueError(
            f"time dimension ({actions.shape[-2]}) must be greater than stride ({stride})"
        )
    return actions[..., stride:, :] - actions[..., :-stride, :]


def minmax_normalized_deltas(
    actions: Tensor,
    minimum: Tensor,
    maximum: Tensor,
    *,
    stride: int = 1,
) -> Tensor:
    """Normalize absolute actions first, then compute strided deltas."""

    return strided_deltas(
        minmax_normalize_absolute(actions, minimum, maximum), stride=stride
    )


def block_anchor_to_adjacent(deltas: Tensor, *, block_size: int = 4) -> Tensor:
    """Convert cumulative block-anchor deltas into adjacent deltas.

    For each block, ``[x1-x0, x2-x0, ...]`` becomes
    ``[x1-x0, x2-x1, ...]``. The conversion operates on the penultimate
    (time/token) axis and preserves every leading batch dimension.
    """

    _validate_actions(deltas)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    tokens = deltas.shape[-2]
    if tokens % block_size:
        raise ValueError(
            f"time dimension ({tokens}) must be divisible by block_size ({block_size})"
        )
    blocks = deltas.reshape(
        *deltas.shape[:-2], tokens // block_size, block_size, deltas.shape[-1]
    )
    adjacent = torch.empty_like(blocks)
    adjacent[..., 0, :] = blocks[..., 0, :]
    adjacent[..., 1:, :] = blocks[..., 1:, :] - blocks[..., :-1, :]
    return adjacent.reshape_as(deltas)


def adjacent_to_block_anchor(deltas: Tensor, *, block_size: int = 4) -> Tensor:
    """Convert adjacent deltas into cumulative deltas within each block."""

    _validate_actions(deltas)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    tokens = deltas.shape[-2]
    if tokens % block_size:
        raise ValueError(
            f"time dimension ({tokens}) must be divisible by block_size ({block_size})"
        )
    blocks = deltas.reshape(
        *deltas.shape[:-2], tokens // block_size, block_size, deltas.shape[-1]
    )
    return torch.cumsum(blocks, dim=-2).reshape_as(deltas)


def normalized_block_anchor_to_raw_adjacent(
    deltas: Tensor,
    minimum: Tensor,
    maximum: Tensor,
    *,
    block_size: int = 4,
) -> Tensor:
    """Recover raw adjacent deltas from normalized block-anchor deltas.

    Min-max normalization is affine, so an adjacent normalized delta is
    multiplied by ``(maximum - minimum) / 2`` to recover raw units. Constant
    dimensions are mapped to zero, matching :func:`minmax_normalize_absolute`.
    """

    _validate_actions(deltas)
    for name, value in (("minimum", minimum), ("maximum", maximum)):
        if not isinstance(value, Tensor) or value.shape != (deltas.shape[-1],):
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} must have shape ({deltas.shape[-1]},), got {shape}")
        if value.device != deltas.device:
            raise ValueError(f"{name} and deltas must be on the same device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or infinity")
    span = maximum.to(deltas.dtype) - minimum.to(deltas.dtype)
    if bool((span < 0).any()):
        raise ValueError("maximum must be greater than or equal to minimum")
    adjacent = block_anchor_to_adjacent(deltas, block_size=block_size)
    return adjacent * (span / 2.0)


__all__ = [
    "ActionTransform",
    "adjacent_to_block_anchor",
    "block_anchor_to_adjacent",
    "minmax_normalize_absolute",
    "minmax_normalized_deltas",
    "normalized_block_anchor_to_raw_adjacent",
    "strided_deltas",
]
