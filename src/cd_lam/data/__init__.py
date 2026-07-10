"""Data contracts shared by CD-LAM training stages."""

from .action import (
    ActionTransform,
    adjacent_to_block_anchor,
    block_anchor_to_adjacent,
    minmax_normalize_absolute,
    minmax_normalized_deltas,
    normalized_block_anchor_to_raw_adjacent,
    strided_deltas,
)
from .manifests import (
    DataContractError,
    PreparationSummary,
    load_episode_records,
    prepare_episode_manifests,
    validate_prepared_manifests,
)

__all__ = [
    "ActionTransform",
    "adjacent_to_block_anchor",
    "block_anchor_to_adjacent",
    "minmax_normalize_absolute",
    "minmax_normalized_deltas",
    "normalized_block_anchor_to_raw_adjacent",
    "strided_deltas",
    "DataContractError",
    "PreparationSummary",
    "load_episode_records",
    "prepare_episode_manifests",
    "validate_prepared_manifests",
]
