"""Portable reference builders and validators for CD-LAM data contracts.

The public builder operates on JSONL episode metadata. Production adapters may
convert the same records to Parquet or native dataset objects, but must retain
the episode split, source FPS, frame alignment, and action representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ACTION_DIM = 22


class DataContractError(ValueError):
    """Raised when episode metadata violates a staged data contract."""


@dataclass(frozen=True)
class PreparationSummary:
    """Counts and hashes for a prepared reference dataset."""

    episodes: int
    stage1_pairs: int
    stage2_windows: int
    bridge_pairs: int
    stage3_windows: int
    files: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bridge_pairs": self.bridge_pairs,
            "episodes": self.episodes,
            "files": dict(self.files),
            "stage1_pairs": self.stage1_pairs,
            "stage2_windows": self.stage2_windows,
            "stage3_windows": self.stage3_windows,
        }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DataContractError(f"{label} must be finite")
    return number


def _vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataContractError(f"{label} must be a {ACTION_DIM}D sequence")
    result = [
        _finite_number(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != ACTION_DIM:
        raise DataContractError(f"{label} must have {ACTION_DIM} values")
    return result


def _expand_actions(record: Mapping[str, Any], label: str) -> list[list[float]] | None:
    actions = record.get("actions")
    if actions is not None:
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise DataContractError(f"{label}.actions must be a sequence")
        return [
            _vector(row, f"{label}.actions[{index}]")
            for index, row in enumerate(actions)
        ]
    sequence = record.get("action_sequence")
    if sequence is None:
        return None
    if not isinstance(sequence, Mapping):
        raise DataContractError(f"{label}.action_sequence must be a mapping")
    start = _vector(sequence.get("start"), f"{label}.action_sequence.start")
    delta = _vector(sequence.get("delta"), f"{label}.action_sequence.delta")
    steps = sequence.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise DataContractError(f"{label}.action_sequence.steps must be positive")
    return [
        [start[axis] + delta[axis] * step for axis in range(ACTION_DIM)]
        for step in range(steps)
    ]


def load_episode_records(path: Path | str) -> list[dict[str, Any]]:
    """Load and validate JSONL episode metadata."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DataContractError(f"episode metadata does not exist: {source}")
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DataContractError(
                f"invalid JSON at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise DataContractError(f"episode row {line_number} must be a mapping")
        episode_id = str(raw.get("episode_id", "")).strip()
        if not episode_id or episode_id in identities:
            raise DataContractError(f"episode_id is empty or duplicated: {episode_id!r}")
        identities.add(episode_id)
        split = str(raw.get("split", "")).strip()
        if split not in {"train", "eval", "test"}:
            raise DataContractError(f"{episode_id}.split must be train, eval, or test")
        num_frames = raw.get("num_frames")
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 2:
            raise DataContractError(f"{episode_id}.num_frames must be at least 2")
        fps = _finite_number(raw.get("fps"), f"{episode_id}.fps")
        if fps <= 0:
            raise DataContractError(f"{episode_id}.fps must be positive")
        actions = _expand_actions(raw, episode_id)
        if actions is not None and len(actions) != num_frames:
            raise DataContractError(
                f"{episode_id} has {len(actions)} actions for {num_frames} source frames"
            )
        records.append(
            {
                "actions": actions,
                "episode_id": episode_id,
                "fps": fps,
                "num_frames": num_frames,
                "source": str(raw.get("source", "unknown")),
                "split": split,
                "video_ref": str(raw.get("video_ref", raw.get("video_path", ""))),
            }
        )
    if not records:
        raise DataContractError(f"episode metadata is empty: {source}")
    return records


def _starts(length: int, span: int, limit: int) -> list[int]:
    available = length - span + 1
    if available <= 0:
        return []
    if limit <= 0 or available <= limit:
        return list(range(available))
    if limit == 1:
        return [0]
    return sorted(
        {round(index * (available - 1) / (limit - 1)) for index in range(limit)}
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    values = list(rows)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )
    return len(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_episode_manifests(
    source: Path | str,
    output: Path | str,
    *,
    pair_stride: int = 1,
    source_action_stride: int = 4,
    window_frames: int = 13,
    pairs_per_episode: int = 8,
    windows_per_episode: int = 4,
) -> PreparationSummary:
    """Build portable Stage 1/2/bridge/Stage 3 JSONL manifests."""

    for value, label in (
        (pair_stride, "pair_stride"),
        (source_action_stride, "source_action_stride"),
        (window_frames, "window_frames"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DataContractError(f"{label} must be a positive integer")
    if window_frames < 2:
        raise DataContractError("window_frames must be at least 2")

    episodes = load_episode_records(source)
    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage1: list[dict[str, Any]] = []
    stage2: list[dict[str, Any]] = []
    bridge: list[dict[str, Any]] = []
    stage3: list[dict[str, Any]] = []

    source_span = (window_frames - 1) * source_action_stride + 1
    for episode in episodes:
        episode_id = episode["episode_id"]
        frame_count = episode["num_frames"]
        common = {
            "episode_id": episode_id,
            "source": episode["source"],
            "split": episode["split"],
            "video_ref": episode["video_ref"],
        }
        for frame_i in _starts(
            frame_count, pair_stride + 1, pairs_per_episode
        ):
            stage1.append(
                {
                    **common,
                    "frame_i": frame_i,
                    "frame_j": frame_i + pair_stride,
                    "pair_type": "real",
                    "source_fps": episode["fps"],
                }
            )
        identity_frame = frame_count // 2
        stage1.append(
            {
                **common,
                "frame_i": identity_frame,
                "frame_j": identity_frame,
                "pair_type": "identity",
                "source_fps": episode["fps"],
            }
        )

        for start in _starts(frame_count, window_frames, windows_per_episode):
            stage2.append(
                {
                    **common,
                    "clip_nframes": window_frames,
                    "source_fps": episode["fps"],
                    "start_frame": start,
                    "stop_frame": start + window_frames,
                }
            )

        actions = episode["actions"]
        if actions is None:
            continue
        for start in _starts(frame_count, source_span, windows_per_episode):
            transition_indices = [
                [
                    start + offset * source_action_stride,
                    start + (offset + 1) * source_action_stride,
                ]
                for offset in range(window_frames - 1)
            ]
            video_frames = [
                start + offset * source_action_stride
                for offset in range(window_frames)
            ]
            stage3.append(
                {
                    **common,
                    "action_dim": ACTION_DIM,
                    "action_representation": "raw_adjacent_stride_delta",
                    "source_fps": episode["fps"],
                    "source_stride": source_action_stride,
                    "transition_indices": transition_indices,
                    "video_frame_indices": video_frames,
                }
            )
            for frame_i, frame_j in transition_indices:
                delta = [
                    actions[frame_j][axis] - actions[frame_i][axis]
                    for axis in range(ACTION_DIM)
                ]
                bridge.append(
                    {
                        **common,
                        "action_22": delta,
                        "action_representation": "raw_adjacent_stride_delta",
                        "frame_i": frame_i,
                        "frame_j": frame_j,
                        "source_stride": source_action_stride,
                    }
                )

    outputs = {
        "bridge_pairs": output_root / "bridge_pairs.jsonl",
        "stage1_pairs": output_root / "stage1_pairs.jsonl",
        "stage2_windows": output_root / "stage2_windows.jsonl",
        "stage3_windows": output_root / "stage3_windows.jsonl",
    }
    counts = {
        "bridge_pairs": _write_jsonl(outputs["bridge_pairs"], bridge),
        "stage1_pairs": _write_jsonl(outputs["stage1_pairs"], stage1),
        "stage2_windows": _write_jsonl(outputs["stage2_windows"], stage2),
        "stage3_windows": _write_jsonl(outputs["stage3_windows"], stage3),
    }
    summary = PreparationSummary(
        episodes=len(episodes),
        stage1_pairs=counts["stage1_pairs"],
        stage2_windows=counts["stage2_windows"],
        bridge_pairs=counts["bridge_pairs"],
        stage3_windows=counts["stage3_windows"],
        files={name: _sha256(path) for name, path in outputs.items()},
    )
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_prepared_manifests(output_root)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataContractError(f"prepared manifest is missing: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise DataContractError(f"row {line_number} in {path} is not a mapping")
        rows.append(row)
    return rows


def validate_prepared_manifests(root: Path | str) -> dict[str, int]:
    """Validate schemas, dimensions, FPS, splits, and Stage 3 alignment."""

    directory = Path(root).expanduser().resolve()
    rows = {
        name: _read_jsonl(directory / f"{name}.jsonl")
        for name in (
            "stage1_pairs",
            "stage2_windows",
            "bridge_pairs",
            "stage3_windows",
        )
    }
    for name, values in rows.items():
        for index, row in enumerate(values):
            label = f"{name}[{index}]"
            if not str(row.get("episode_id", "")).strip():
                raise DataContractError(f"{label}.episode_id is missing")
            if row.get("split") not in {"train", "eval", "test"}:
                raise DataContractError(f"{label}.split is invalid")
            if name in {"stage1_pairs", "stage2_windows", "stage3_windows"}:
                fps = _finite_number(row.get("source_fps"), f"{label}.source_fps")
                if fps <= 0:
                    raise DataContractError(f"{label}.source_fps must be positive")
    for index, row in enumerate(rows["bridge_pairs"]):
        _vector(row.get("action_22"), f"bridge_pairs[{index}].action_22")
        if row.get("action_representation") != "raw_adjacent_stride_delta":
            raise DataContractError("bridge action representation is incompatible")
    for index, row in enumerate(rows["stage3_windows"]):
        if row.get("action_dim") != ACTION_DIM:
            raise DataContractError(
                f"stage3_windows[{index}].action_dim must be {ACTION_DIM}"
            )
        frames = row.get("video_frame_indices")
        transitions = row.get("transition_indices")
        if not isinstance(frames, list) or not isinstance(transitions, list):
            raise DataContractError(
                f"stage3_windows[{index}] alignment fields are missing"
            )
        if len(transitions) + 1 != len(frames):
            raise DataContractError(
                f"stage3_windows[{index}] transition count is invalid"
            )
        if any(
            list(pair) != [frames[position], frames[position + 1]]
            for position, pair in enumerate(transitions)
        ):
            raise DataContractError(
                f"stage3_windows[{index}] transitions do not match frames"
            )
    return {name: len(values) for name, values in rows.items()}


__all__ = [
    "DataContractError",
    "PreparationSummary",
    "load_episode_records",
    "prepare_episode_manifests",
    "validate_prepared_manifests",
]
