"""Stable provenance keys and split audits for raw CD-LAM datasets.

The training adapters operate on clips, but train/evaluation isolation must use
the physical recording unit.  AgiBot Alpha stores several action segments under
one physical episode.  EgoDex can cut several clips from one source movie and
records that movie in the HDF5 ``session_name`` attribute.

This module deliberately handles provenance only.  It does not download,
transcode, filter, or pack either dataset.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


AGIBOT_ALPHA_DATASET = "agibot_alpha"
EGODEX_DATASET = "egodex"
_SPLITS = {"train", "eval", "test"}
_EGODEX_PARTS = {"part1", "part2", "part3", "part4", "part5", "extra", "test"}
_AGIBOT_SEGMENT = re.compile(
    r"^(?P<task>[0-9]+)-(?P<episode>[0-9]+)-(?P<segment>[0-9]+)$"
)
_EGODEX_EPISODE = re.compile(
    r"^egodex_(?P<part>part[1-5]|extra|test)_(?P<task>.+)_(?P<index>[0-9]+)$"
)


class RawDataContractError(ValueError):
    """Raised when raw provenance cannot guarantee split isolation."""


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise RawDataContractError(f"{label} must be non-empty")
    return result


def parse_agibot_alpha_segment_id(source_id: Any) -> tuple[str, str, int]:
    """Parse ``<task>-<physical_episode>-<segment>`` from AgiBot Alpha."""

    value = _text(source_id, "AgiBot Alpha source_id")
    match = _AGIBOT_SEGMENT.fullmatch(value)
    if match is None:
        raise RawDataContractError(
            "AgiBot Alpha source_id must be <task>-<episode>-<segment>: "
            f"{value!r}"
        )
    task = str(int(match.group("task")))
    episode = str(int(match.group("episode")))
    segment = int(match.group("segment"))
    return task, episode, segment


def agibot_alpha_physical_episode_key(source_id: Any) -> str:
    """Return the split key shared by every segment of one Alpha episode."""

    task, episode, _ = parse_agibot_alpha_segment_id(source_id)
    return f"agibot_alpha:{task}:{episode}"


def parse_egodex_episode_id(episode_id: Any) -> tuple[str, str, int]:
    """Parse the raw EgoDex identity ``egodex_<part>_<task>_<index>``."""

    value = _text(episode_id, "EgoDex episode_id")
    match = _EGODEX_EPISODE.fullmatch(value)
    if match is None:
        raise RawDataContractError(
            "EgoDex episode_id must be egodex_<part>_<task>_<index>: "
            f"{value!r}"
        )
    return match.group("part"), match.group("task"), int(match.group("index"))


def egodex_physical_session_key(part: Any, session_name: Any) -> str:
    """Return the split key shared by clips cut from one EgoDex source movie."""

    normalized_part = _text(part, "EgoDex part")
    if normalized_part not in _EGODEX_PARTS:
        raise RawDataContractError(f"unsupported EgoDex part: {normalized_part!r}")
    session = _text(session_name, "EgoDex session_name")
    return f"egodex:{normalized_part}:{session}"


def egodex_native_split(part: Any) -> str:
    """Return ``test`` for Apple's native test part, otherwise ``train_pool``."""

    normalized_part = _text(part, "EgoDex part")
    if normalized_part not in _EGODEX_PARTS:
        raise RawDataContractError(f"unsupported EgoDex part: {normalized_part!r}")
    return "test" if normalized_part == "test" else "train_pool"


def audit_raw_split_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit normalized raw provenance records without touching media files.

    AgiBot Alpha records require ``dataset``, ``source_id``, and ``split``.
    EgoDex records require ``dataset``, ``episode_id``, ``session_name``, and
    ``split``.  EgoDex ``part`` may be supplied explicitly but must agree with
    the raw episode ID when present.
    """

    group_splits: dict[str, set[str]] = defaultdict(set)
    dataset_counts: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    row_count = 0

    for row_index, record in enumerate(records):
        row_count += 1
        if not isinstance(record, Mapping):
            errors.append(f"row {row_index}: record must be a mapping")
            continue
        try:
            dataset = _text(record.get("dataset"), f"row {row_index}.dataset")
            split = _text(record.get("split"), f"row {row_index}.split")
            if split not in _SPLITS:
                raise RawDataContractError(
                    f"row {row_index}.split must be train, eval, or test"
                )

            if dataset == AGIBOT_ALPHA_DATASET:
                source_id = record.get("source_id", record.get("segment_id"))
                group_key = agibot_alpha_physical_episode_key(source_id)
                declared = record.get("physical_episode_id")
                if declared is not None and _text(
                    declared, f"row {row_index}.physical_episode_id"
                ) != group_key:
                    raise RawDataContractError(
                        f"row {row_index}.physical_episode_id disagrees with source_id"
                    )
            elif dataset == EGODEX_DATASET:
                inferred_part, _, _ = parse_egodex_episode_id(record.get("episode_id"))
                part = _text(record.get("part", inferred_part), f"row {row_index}.part")
                if part != inferred_part:
                    raise RawDataContractError(
                        f"row {row_index}.part disagrees with episode_id"
                    )
                if egodex_native_split(part) == "test" and split == "train":
                    raise RawDataContractError(
                        f"row {row_index}: native EgoDex test clip cannot enter train"
                    )
                group_key = egodex_physical_session_key(
                    part, record.get("session_name")
                )
            else:
                raise RawDataContractError(
                    f"row {row_index}: unsupported dataset {dataset!r}"
                )
        except RawDataContractError as exc:
            errors.append(str(exc))
            continue

        dataset_counts[dataset] += 1
        group_splits[group_key].add(split)

    leaking_groups = {
        key: sorted(values)
        for key, values in sorted(group_splits.items())
        if len(values) > 1
    }
    errors.extend(
        f"physical group crosses splits: {key} -> {values}"
        for key, values in leaking_groups.items()
    )
    return {
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "errors": errors,
        "groups": len(group_splits),
        "leaking_groups": leaking_groups,
        "records": row_count,
        "status": "pass" if not errors else "fail",
    }


def validate_raw_split_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return an audit summary or raise on any provenance/split violation."""

    summary = audit_raw_split_records(records)
    if summary["errors"]:
        raise RawDataContractError("; ".join(summary["errors"]))
    return summary


__all__ = [
    "AGIBOT_ALPHA_DATASET",
    "EGODEX_DATASET",
    "RawDataContractError",
    "agibot_alpha_physical_episode_key",
    "audit_raw_split_records",
    "egodex_native_split",
    "egodex_physical_session_key",
    "parse_agibot_alpha_segment_id",
    "parse_egodex_episode_id",
    "validate_raw_split_records",
]
