#!/usr/bin/env python3
"""Validate and convert the internal AgiBot Stage-3 action contract.

The external ACWM loader emits twelve min-max-normalized action deltas. Within
each four-token block, every delta is measured from the same block anchor.
The legacy CD-LAM bridge was trained on raw-unit, adjacent stride-4 deltas.
This module validates the assets that define that relationship and performs
the exact affine and temporal conversion before the bridge sees an action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ACTION_DIM = 22
ACTION_TOKENS = 12
BLOCK_SIZE = 4
SOURCE_STRIDE = 4
SCHEMA_VERSION = 1
LOADER_REPRESENTATION = "minmax_normalized_block_anchor_cumulative_delta"
BRIDGE_REPRESENTATION = "raw_adjacent_stride_delta"
METADATA_SOURCE_REPOSITORY = "https://github.com/NVIDIA/DreamDojo.git"
METADATA_SOURCE_REVISION = "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"

EXPECTED_LAYOUT = (
    ("left_arm_joint_position", 0, 7),
    ("right_arm_joint_position", 7, 14),
    ("left_effector_position", 14, 15),
    ("right_effector_position", 15, 16),
    ("head_position", 16, 18),
    ("waist_pitch", 18, 19),
    ("waist_lift", 19, 20),
    ("robot_velocity", 20, 22),
)

BRIDGE_REQUIRED_FIELDS = (
    "g_state",
    "action_mean",
    "action_std",
    "zm",
    "zsd",
    "latent_dim",
)


class ActionContractError(ValueError):
    """Raised when Stage-3 action semantics cannot be established safely."""


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ActionContractError(f"missing {label}: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionContractError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActionContractError(f"{label} must contain a JSON object: {path}")
    return value


def _float_vector(value: Any, label: str, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,):
        raise ActionContractError(
            f"{label} has shape {list(result.shape)}, expected [{size}]"
        )
    if not np.isfinite(result).all():
        raise ActionContractError(f"{label} contains non-finite values")
    return result


def _validate_tensor_shape(
    value: torch.Tensor,
    *,
    action_dim: int,
    action_tokens: int,
    block_size: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ActionContractError(f"action must be a torch.Tensor, got {type(value)!r}")
    if value.ndim < 2:
        raise ActionContractError(
            f"action must have at least two dimensions, got shape {tuple(value.shape)}"
        )
    if value.shape[-1] != action_dim:
        raise ActionContractError(
            f"action width is {value.shape[-1]}, expected {action_dim}"
        )
    if value.shape[-2] != action_tokens:
        raise ActionContractError(
            f"action has {value.shape[-2]} tokens, expected {action_tokens}"
        )
    if action_tokens % block_size != 0:
        raise ActionContractError(
            f"action token count {action_tokens} is not divisible by block size {block_size}"
        )
    if not torch.isfinite(value).all().item():
        raise ActionContractError("action contains non-finite values")


def normalized_block_anchor_to_raw_adjacent(
    loader_delta: torch.Tensor,
    action_range: torch.Tensor | np.ndarray | Iterable[float],
    *,
    action_tokens: int = ACTION_TOKENS,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Convert loader deltas to raw adjacent stride deltas.

    For each four-token block the loader stores ``[x1-x0, x2-x0,
    x3-x0, x4-x0]`` after min-max normalization. First differences recover
    adjacent normalized deltas. Multiplication by ``(max-min)/2`` then
    restores raw action units.
    """

    _validate_tensor_shape(
        loader_delta,
        action_dim=ACTION_DIM,
        action_tokens=action_tokens,
        block_size=block_size,
    )

    ranges = torch.as_tensor(
        action_range,
        dtype=torch.float32,
        device=loader_delta.device,
    )
    if tuple(ranges.shape) != (ACTION_DIM,):
        raise ActionContractError(
            f"action range has shape {tuple(ranges.shape)}, expected ({ACTION_DIM},)"
        )
    if not torch.isfinite(ranges).all().item() or not torch.all(ranges > 0).item():
        raise ActionContractError("action range must be finite and strictly positive")

    values = loader_delta.to(dtype=torch.float32)
    shape = values.shape
    blocks = values.reshape(
        *shape[:-2], action_tokens // block_size, block_size, ACTION_DIM
    )
    adjacent = torch.empty_like(blocks)
    adjacent[..., 0, :] = blocks[..., 0, :]
    adjacent[..., 1:, :] = blocks[..., 1:, :] - blocks[..., :-1, :]
    return (adjacent.reshape(shape) * (ranges / 2.0)).contiguous()


def raw_adjacent_to_normalized_block_anchor(
    raw_adjacent: torch.Tensor,
    action_range: torch.Tensor | np.ndarray | Iterable[float],
    *,
    action_tokens: int = ACTION_TOKENS,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Inverse of :func:`normalized_block_anchor_to_raw_adjacent`."""

    _validate_tensor_shape(
        raw_adjacent,
        action_dim=ACTION_DIM,
        action_tokens=action_tokens,
        block_size=block_size,
    )
    ranges = torch.as_tensor(
        action_range,
        dtype=torch.float32,
        device=raw_adjacent.device,
    )
    if tuple(ranges.shape) != (ACTION_DIM,):
        raise ActionContractError(
            f"action range has shape {tuple(ranges.shape)}, expected ({ACTION_DIM},)"
        )
    if not torch.isfinite(ranges).all().item() or not torch.all(ranges > 0).item():
        raise ActionContractError("action range must be finite and strictly positive")

    values = raw_adjacent.to(dtype=torch.float32) * (2.0 / ranges)
    shape = values.shape
    blocks = values.reshape(
        *shape[:-2], action_tokens // block_size, block_size, ACTION_DIM
    )
    return torch.cumsum(blocks, dim=-2).reshape(shape).contiguous()


def _validate_layout(contract: dict[str, Any], modality: dict[str, Any]) -> None:
    expected_json = [
        {"name": name, "start": start, "end": end}
        for name, start, end in EXPECTED_LAYOUT
    ]
    if contract.get("layout") != expected_json:
        raise ActionContractError(
            "contract layout is not the canonical AgiBot 22D layout"
        )

    action_meta = modality.get("action")
    if not isinstance(action_meta, dict):
        raise ActionContractError("AgiBot modality metadata has no action mapping")
    if list(action_meta) != [name for name, _, _ in EXPECTED_LAYOUT]:
        raise ActionContractError(
            "AgiBot action modality order does not match the 22D contract"
        )
    for name, start, end in EXPECTED_LAYOUT:
        row = action_meta.get(name)
        if not isinstance(row, dict):
            raise ActionContractError(
                f"AgiBot modality metadata is missing action.{name}"
            )
        observed = (
            row.get("original_key"),
            row.get("start"),
            row.get("end"),
            row.get("absolute"),
        )
        expected = ("action", start, end, True)
        if observed != expected:
            raise ActionContractError(
                f"action.{name} metadata is {observed}, expected {expected}"
            )


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    _require_file(path, "bridge checkpoint")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ActionContractError(
            f"unable to load bridge checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ActionContractError("bridge checkpoint must contain a dictionary")
    return value


def _validate_bridge(
    bridge: dict[str, Any],
    bridge_digest: str,
    contract: dict[str, Any],
    scale_family: str,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [key for key in BRIDGE_REQUIRED_FIELDS if key not in bridge]
    if missing:
        raise ActionContractError(f"bridge checkpoint is missing fields: {missing}")

    action_mean = _float_vector(bridge["action_mean"], "bridge action_mean", ACTION_DIM)
    action_std = _float_vector(bridge["action_std"], "bridge action_std", ACTION_DIM)
    if np.any(action_std <= 0):
        raise ActionContractError("bridge action_std must be strictly positive")
    _float_vector(bridge["zm"], "bridge zm", 32)
    _float_vector(bridge["zsd"], "bridge zsd", 32)
    if int(bridge["latent_dim"]) != 32:
        raise ActionContractError(
            f"bridge latent_dim is {bridge['latent_dim']}, expected 32"
        )

    state = bridge["g_state"]
    if not isinstance(state, dict):
        raise ActionContractError("bridge g_state must be a dictionary")
    try:
        input_shape = tuple(state["0.weight"].shape)
        output_shape = tuple(state["4.weight"].shape)
    except (KeyError, AttributeError) as exc:
        raise ActionContractError(
            "bridge g_state does not match the expected MLP"
        ) from exc
    if input_shape != (256, ACTION_DIM) or output_shape != (32, 256):
        raise ActionContractError(
            f"bridge MLP shapes are {input_shape} and {output_shape}, expected "
            f"(256, {ACTION_DIM}) and (32, 256)"
        )

    family_hashes = contract.get("legacy_bridge_sha256_by_scale_family")
    if not isinstance(family_hashes, dict) or scale_family not in family_hashes:
        raise ActionContractError(
            f"scale family {scale_family!r} has no pinned source bridge SHA256"
        )
    expected_source_digest = family_hashes[scale_family]

    embedded = bridge.get("cdlam_action_contract")
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise ActionContractError(
                "embedded bridge action contract must be a dictionary"
            )
        required = {
            "contract_id": contract.get("contract_id"),
            "action_dim": ACTION_DIM,
            "action_tokens": ACTION_TOKENS,
            "block_size": BLOCK_SIZE,
            "source_stride": SOURCE_STRIDE,
            "loader_representation": LOADER_REPRESENTATION,
            "representation": BRIDGE_REPRESENTATION,
            "scale_family": scale_family,
            "stats_sha256": contract.get("stats_sha256"),
            "modality_sha256": contract.get("modality_sha256"),
        }
        for key, expected in required.items():
            if embedded.get(key) != expected:
                raise ActionContractError(
                    f"embedded bridge contract {key} is {embedded.get(key)!r}, "
                    f"expected {expected!r}"
                )
    elif bridge.get("format") == "cdlam.action_to_latent_bridge":
        metadata = bridge.get("metadata")
        if not isinstance(metadata, dict):
            raise ActionContractError("sanitized bridge metadata must be a dictionary")
        expected_metadata = {
            "bridge_contract_id": contract.get("contract_id"),
            "data_tier": scale_family,
            "source_sha256": expected_source_digest,
            "action_dim": ACTION_DIM,
            "latent_dim": 32,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ActionContractError(
                    f"sanitized bridge metadata {key} is {metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if bridge.get("action_dim") != ACTION_DIM:
            raise ActionContractError(
                f"sanitized bridge action_dim is {bridge.get('action_dim')!r}, "
                f"expected {ACTION_DIM}"
            )
    else:
        if bridge_digest != expected_source_digest:
            raise ActionContractError(
                f"legacy bridge SHA256 is not pinned for scale family {scale_family!r}"
            )
    return action_mean, action_std


@dataclass(frozen=True)
class Stage3ActionContract:
    """Validated conversion state for one bridge and external metadata set."""

    contract_path: Path
    stats_path: Path
    modality_path: Path
    bridge_path: Path
    contract_id: str
    scale_family: str
    stats_sha256: str
    modality_sha256: str
    bridge_sha256: str
    action_min: np.ndarray
    action_max: np.ndarray
    bridge_action_mean: np.ndarray
    bridge_action_std: np.ndarray

    @property
    def action_range(self) -> np.ndarray:
        return self.action_max - self.action_min

    def loader_to_bridge(self, loader_delta: torch.Tensor) -> torch.Tensor:
        return normalized_block_anchor_to_raw_adjacent(
            loader_delta,
            self.action_range,
            action_tokens=ACTION_TOKENS,
            block_size=BLOCK_SIZE,
        )

    def bridge_to_loader(self, raw_adjacent: torch.Tensor) -> torch.Tensor:
        return raw_adjacent_to_normalized_block_anchor(
            raw_adjacent,
            self.action_range,
            action_tokens=ACTION_TOKENS,
            block_size=BLOCK_SIZE,
        )

    def validate_bridge_input(self, raw_adjacent: torch.Tensor) -> None:
        _validate_tensor_shape(
            raw_adjacent,
            action_dim=ACTION_DIM,
            action_tokens=ACTION_TOKENS,
            block_size=BLOCK_SIZE,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "scale_family": self.scale_family,
            "action_dim": ACTION_DIM,
            "action_tokens": ACTION_TOKENS,
            "block_size": BLOCK_SIZE,
            "source_stride": SOURCE_STRIDE,
            "loader_representation": LOADER_REPRESENTATION,
            "bridge_representation": BRIDGE_REPRESENTATION,
            "metadata_source_repository": METADATA_SOURCE_REPOSITORY,
            "metadata_source_revision": METADATA_SOURCE_REVISION,
            "stats_sha256": self.stats_sha256,
            "modality_sha256": self.modality_sha256,
            "bridge_sha256": self.bridge_sha256,
        }


def load_stage3_action_contract(
    contract_path: str | Path,
    external_root: str | Path,
    bridge_path: str | Path,
    scale_family: str,
) -> Stage3ActionContract:
    contract_path = Path(contract_path).expanduser().resolve()
    external_root = Path(external_root).expanduser().resolve()
    bridge_path = Path(bridge_path).expanduser().resolve()
    contract = _read_json(contract_path, "Stage-3 action contract")
    if not isinstance(scale_family, str) or not scale_family:
        raise ActionContractError("scale_family must be a non-empty string")

    exact_fields = {
        "schema_version": SCHEMA_VERSION,
        "embodiment": "agibot",
        "action_dim": ACTION_DIM,
        "action_tokens": ACTION_TOKENS,
        "block_size": BLOCK_SIZE,
        "source_stride": SOURCE_STRIDE,
        "loader_representation": LOADER_REPRESENTATION,
        "bridge_representation": BRIDGE_REPRESENTATION,
        "metadata_source_repository": METADATA_SOURCE_REPOSITORY,
        "metadata_source_revision": METADATA_SOURCE_REVISION,
    }
    for key, expected in exact_fields.items():
        if contract.get(key) != expected:
            raise ActionContractError(
                f"contract {key} is {contract.get(key)!r}, expected {expected!r}"
            )
    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise ActionContractError("contract_id must be a non-empty string")

    stats_relative = contract.get("stats_relative_path")
    modality_relative = contract.get("modality_relative_path")
    if not isinstance(stats_relative, str) or not isinstance(modality_relative, str):
        raise ActionContractError("contract metadata paths must be strings")
    stats_path = external_root / stats_relative
    modality_path = external_root / modality_relative
    stats = _read_json(stats_path, "AgiBot statistics")
    modality = _read_json(modality_path, "AgiBot modality metadata")

    stats_digest = sha256_file(stats_path)
    modality_digest = sha256_file(modality_path)
    if stats_digest != contract.get("stats_sha256"):
        raise ActionContractError(
            f"AgiBot statistics SHA256 is {stats_digest}, expected {contract.get('stats_sha256')}"
        )
    if modality_digest != contract.get("modality_sha256"):
        raise ActionContractError(
            f"AgiBot modality SHA256 is {modality_digest}, expected "
            f"{contract.get('modality_sha256')}"
        )

    _validate_layout(contract, modality)
    action_stats = stats.get("action")
    if not isinstance(action_stats, dict):
        raise ActionContractError("AgiBot statistics have no action section")
    action_mean = _float_vector(
        action_stats.get("mean"), "AgiBot action mean", ACTION_DIM
    )
    action_std = _float_vector(action_stats.get("std"), "AgiBot action std", ACTION_DIM)
    action_min = _float_vector(action_stats.get("min"), "AgiBot action min", ACTION_DIM)
    action_max = _float_vector(action_stats.get("max"), "AgiBot action max", ACTION_DIM)
    if np.any(action_std < 0):
        raise ActionContractError("AgiBot action std contains negative values")
    if np.any(action_max <= action_min):
        bad = np.flatnonzero(action_max <= action_min).tolist()
        raise ActionContractError(
            f"AgiBot action ranges are not positive at dimensions {bad}"
        )
    del action_mean, action_std

    bridge_digest = sha256_file(bridge_path) if bridge_path.is_file() else ""
    bridge = _load_torch_checkpoint(bridge_path)
    bridge_mean, bridge_std = _validate_bridge(
        bridge,
        bridge_digest,
        contract,
        scale_family,
    )

    return Stage3ActionContract(
        contract_path=contract_path,
        stats_path=stats_path,
        modality_path=modality_path,
        bridge_path=bridge_path,
        contract_id=contract_id,
        scale_family=scale_family,
        stats_sha256=stats_digest,
        modality_sha256=modality_digest,
        bridge_sha256=bridge_digest,
        action_min=action_min,
        action_max=action_max,
        bridge_action_mean=bridge_mean,
        bridge_action_std=bridge_std,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--scale-family", required=True)
    args = parser.parse_args()
    validated = load_stage3_action_contract(
        args.contract,
        args.external_root,
        args.bridge,
        args.scale_family,
    )
    print(json.dumps({"status": "ok", **validated.summary()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
