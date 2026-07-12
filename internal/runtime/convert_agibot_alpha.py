#!/usr/bin/env python3
"""Convert AgiBotWorld Alpha raw segments into portable LeRobot datasets.

The converter writes separate ``train``, ``eval``, and ``test`` dataset roots.
The split key is the physical ``<task>-<episode>`` recording, never the action
segment.  Videos are copied by default so the result can be moved to another
machine; symlinks require an explicit CLI option.

The generated training YAML is accepted directly by
``Scale/common/build_alpha_bridge_cache.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = BUNDLE_ROOT / "internal" / "vendor" / "scale_support"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from Scale.common.raw_data_contract import (  # noqa: E402
    agibot_alpha_physical_episode_key,
    parse_agibot_alpha_segment_id,
)


SCHEMA_VERSION = 1
DEFAULT_FPS = 30.0
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MIN_FRAMES = 50
DEFAULT_EVAL_PERCENT = 10
DEFAULT_HELD_OUT_TASKS = 3
VIDEO_KEY = "observation.images.top_head"
DATA_PATH_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_PATTERN = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

STATE_LAYOUT = (
    ("left_arm_joint_position", 0, 7),
    ("right_arm_joint_position", 7, 14),
    ("left_effector_position", 14, 15),
    ("right_effector_position", 15, 16),
    ("head_position", 16, 18),
    ("waist_pitch", 18, 19),
    ("waist_lift", 19, 20),
)
ACTION_LAYOUT = STATE_LAYOUT + (("robot_velocity", 20, 22),)
VALID_SPLITS = ("train", "eval", "test")


class ConversionError(ValueError):
    """Raised when raw input cannot satisfy the conversion contract."""


@dataclass(frozen=True)
class Segment:
    """One raw action segment and its physical-recording provenance."""

    raw_dir: Path
    task_id: int
    source_episode_id: int
    segment_index: int
    skill: str
    action_text: str
    task_name: str
    expected_frames: int | None

    @property
    def source_id(self) -> str:
        return f"{self.task_id}-{self.source_episode_id}-{self.segment_index:03d}"

    @property
    def physical_episode_id(self) -> str:
        return agibot_alpha_physical_episode_key(self.source_id)

    @property
    def video_path(self) -> Path:
        return self.raw_dir / "head_color.mp4"

    @property
    def h5_path(self) -> Path:
        return self.raw_dir / "proprio_stats.h5"


def _md5_int(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.md5(value).hexdigest(), 16)


def held_out_task_ids(task_ids: Iterable[int], count: int) -> set[int]:
    """Choose the deterministic unseen-task holdout used by the Alpha recipe."""

    if isinstance(count, bool) or count < 0:
        raise ConversionError("held-out task count must be non-negative")
    unique = sorted({int(task_id) for task_id in task_ids})
    if count > len(unique):
        raise ConversionError(
            f"held-out task count {count} exceeds available tasks {len(unique)}"
        )
    ranked = sorted(unique, key=lambda task_id: _md5_int("alpha_task", task_id))
    return set(ranked[:count])


def split_for_source_id(
    source_id: str,
    held_out_tasks: set[int],
    eval_percent: int = DEFAULT_EVAL_PERCENT,
) -> tuple[str, str]:
    """Return ``(split, detail)`` using a physical-episode-level key."""

    if isinstance(eval_percent, bool) or not 0 <= eval_percent < 100:
        raise ConversionError("eval percent must be in [0, 100)")
    task_text, episode_text, _ = parse_agibot_alpha_segment_id(source_id)
    task_id = int(task_text)
    if task_id in held_out_tasks:
        return "test", "test_task"
    if _md5_int("alpha_ep", task_text, episode_text) % 100 < eval_percent:
        return "eval", "test_episode"
    return "train", "train"


def _as_matrix(name: str, value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ConversionError(
            f"{name} must have shape (frames, {width}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ConversionError(f"{name} contains non-finite values")
    return array


def construct_state_action(
    state_joint_position: np.ndarray,
    state_effector_position: np.ndarray,
    state_head_position: np.ndarray,
    state_waist_position: np.ndarray,
    action_joint_position: np.ndarray,
    action_effector_position: np.ndarray,
    action_head_position: np.ndarray,
    action_waist_position: np.ndarray,
    action_robot_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct canonical state and action from distinct publisher arrays."""

    arrays = {
        "state/joint/position": _as_matrix(
            "state/joint/position", state_joint_position, 14
        ),
        "state/effector/position": _as_matrix(
            "state/effector/position", state_effector_position, 2
        ),
        "state/head/position": _as_matrix(
            "state/head/position", state_head_position, 2
        ),
        "state/waist/position": _as_matrix(
            "state/waist/position", state_waist_position, 2
        ),
        "action/joint/position": _as_matrix(
            "action/joint/position", action_joint_position, 14
        ),
        "action/effector/position": _as_matrix(
            "action/effector/position", action_effector_position, 2
        ),
        "action/head/position": _as_matrix(
            "action/head/position", action_head_position, 2
        ),
        "action/waist/position": _as_matrix(
            "action/waist/position", action_waist_position, 2
        ),
        "action/robot/velocity": _as_matrix(
            "action/robot/velocity", action_robot_velocity, 2
        ),
    }
    frame_counts = {name: len(value) for name, value in arrays.items()}
    if len(set(frame_counts.values())) != 1:
        raise ConversionError(
            "state/action arrays have different frame counts: "
            + ", ".join(f"{name}={count}" for name, count in frame_counts.items())
        )
    action = np.concatenate(
        (
            arrays["action/joint/position"],
            arrays["action/effector/position"],
            arrays["action/head/position"],
            arrays["action/waist/position"],
            arrays["action/robot/velocity"],
        ),
        axis=1,
    )
    state = np.concatenate(
        (
            arrays["state/joint/position"],
            arrays["state/effector/position"],
            arrays["state/head/position"],
            arrays["state/waist/position"],
        ),
        axis=1,
    )
    return state, action


def relative_timestamp_seconds(timestamp_ns: np.ndarray) -> np.ndarray:
    """Convert strictly increasing nanosecond timestamps to relative seconds."""

    timestamps = np.asarray(timestamp_ns)
    if timestamps.ndim != 1 or not np.issubdtype(timestamps.dtype, np.integer):
        raise ConversionError(
            f"timestamp must be a one-dimensional integer array, got {timestamps.shape} "
            f"with dtype {timestamps.dtype}"
        )
    if timestamps.size == 0:
        raise ConversionError("timestamp must not be empty")
    if timestamps.size > 1 and not bool(np.all(np.diff(timestamps) > 0)):
        raise ConversionError("timestamp must be strictly increasing")
    return (timestamps - timestamps[0]).astype(np.float64) / 1e9


def align_proprio_to_video(
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    video_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Align terminal-state HDF5 records to the decodable video frame count.

    Some official Alpha segments store one terminal proprio record after the
    last camera frame. Dropping exactly that record preserves frame ``i`` to
    action ``i`` alignment. Any larger mismatch is ambiguous and fails closed.
    """

    proprio_frames = len(state)
    if len(action) != proprio_frames or len(timestamp) != proprio_frames:
        raise ConversionError("state, action, and timestamp lengths must match")
    if video_frames <= 0:
        return state, action, timestamp, "video_frame_count_unavailable"
    if video_frames == proprio_frames:
        return state, action, timestamp, "exact"
    if proprio_frames == video_frames + 1:
        return (
            state[:video_frames],
            action[:video_frames],
            timestamp[:video_frames],
            "dropped_terminal_proprio",
        )
    raise ConversionError(
        "video/proprio frame mismatch is not a single terminal-state record: "
        f"video={video_frames}, proprio={proprio_frames}"
    )


def read_proprio(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one raw HDF5 file and return state, action, and timestamps."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "AgiBot conversion requires h5py in the model-integration environment"
        ) from exc
    with h5py.File(path, "r") as handle:
        required = (
            "state/joint/position",
            "state/effector/position",
            "state/head/position",
            "state/waist/position",
            "action/joint/position",
            "action/effector/position",
            "action/head/position",
            "action/waist/position",
            "action/robot/velocity",
            "timestamp",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ConversionError(
                f"required proprio datasets are missing from {path}: {missing}"
            )
        state, action = construct_state_action(
            handle["state/joint/position"][:],
            handle["state/effector/position"][:],
            handle["state/head/position"][:],
            handle["state/waist/position"][:],
            handle["action/joint/position"][:],
            handle["action/effector/position"][:],
            handle["action/head/position"][:],
            handle["action/waist/position"][:],
            handle["action/robot/velocity"][:],
        )
        timestamps = relative_timestamp_seconds(handle["timestamp"][:])
    if len(state) != len(timestamps):
        raise ConversionError(
            f"proprio and timestamp lengths differ: {len(state)} != {len(timestamps)}"
        )
    return state, action, timestamps


def materialize_video(source: Path, destination: Path, mode: str = "copy") -> None:
    """Copy a video by default, or create an explicit source symlink."""

    if mode not in {"copy", "symlink"}:
        raise ConversionError(f"unsupported video mode: {mode!r}")
    if not source.is_file():
        raise ConversionError(f"source video is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ConversionError(f"video destination already exists: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        os.symlink(source.resolve(), destination)


def validate_video_metadata(
    path: Path,
    expected_fps: float,
) -> dict[str, float | int]:
    """Open one video and validate stream metadata without decoding every frame."""

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "video verification requires PyAV; pass --skip-video-check only for "
            "metadata plumbing tests"
        ) from exc
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ConversionError(f"video has no stream: {path}")
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        if int(stream.width) <= 0 or int(stream.height) <= 0:
            raise ConversionError(f"video has invalid dimensions: {path}")
        if (int(stream.height), int(stream.width)) != (480, 640):
            raise ConversionError(
                f"video dimensions for {path} are {stream.height}x{stream.width}; "
                "expected 480x640"
            )
        if rate and abs(rate - expected_fps) > 0.05:
            raise ConversionError(
                f"video FPS mismatch for {path}: {rate:.6f} != {expected_fps:.6f}"
            )
        return {
            "frames": frames,
            "fps": rate,
            "height": int(stream.height),
            "width": int(stream.width),
        }


def parse_clip_list(path: Path) -> set[str]:
    """Read source IDs from text or a recipe-style JSON file."""

    if not path.is_file():
        raise ConversionError(f"clip list is missing: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            data = data.get("clips", data.get("source_ids"))
        if not isinstance(data, list):
            raise ConversionError(
                "JSON clip list must be a list or contain clips/source_ids"
            )
        result: set[str] = set()
        for item in data:
            if isinstance(item, Mapping):
                value = item.get("clip_dir", item.get("source_id"))
            else:
                value = item
            if not isinstance(value, str) or not value.strip():
                raise ConversionError(f"invalid clip-list item: {item!r}")
            result.add(value.strip())
        return result
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def available_task_ids(task_info_dir: Path) -> list[int]:
    """Return task IDs declared by the raw task-info directory."""

    task_ids: list[int] = []
    for path in task_info_dir.glob("task_*.json"):
        try:
            task_ids.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not task_ids:
        raise ConversionError(f"no task_*.json files found under {task_info_dir}")
    return sorted(set(task_ids))


def load_task_table(
    task_info_dir: Path,
    task_ids: Iterable[int],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Load labels only for tasks represented by the selected segments."""

    table: dict[tuple[int, int, int], dict[str, Any]] = {}
    for task_id in sorted(set(task_ids)):
        path = task_info_dir / f"task_{task_id}.json"
        if not path.is_file():
            raise ConversionError(f"task metadata is missing: {path}")
        episodes = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(episodes, list):
            raise ConversionError(f"task metadata must be a list: {path}")
        for episode in episodes:
            episode_id = int(episode["episode_id"])
            task_name = str(episode.get("task_name", "")).strip()
            configs = episode.get("label_info", {}).get("action_config", [])
            for segment_index, config in enumerate(configs):
                start = config.get("start_frame")
                end = config.get("end_frame")
                expected = None
                if start is not None and end is not None:
                    expected = int(end) - int(start)
                table[(task_id, episode_id, segment_index)] = {
                    "action_text": str(config.get("action_text", "")).strip(),
                    "expected_frames": expected,
                    "skill": str(config.get("skill", "")).strip(),
                    "task_name": task_name,
                }
    return table


def discover_segments(
    raw_root: Path,
    source_ids: set[str] | None = None,
) -> list[Segment]:
    """Discover selected raw segments and attach task metadata."""

    train_dir = raw_root / "train"
    task_info_dir = raw_root / "task_info"
    if not train_dir.is_dir() or not task_info_dir.is_dir():
        raise ConversionError(
            f"raw root must contain train/ and task_info/: {raw_root}"
        )
    if source_ids is None:
        candidates = [path for path in train_dir.iterdir() if path.is_dir()]
    else:
        candidates = [train_dir / source_id for source_id in sorted(source_ids)]
    parsed: list[tuple[Path, int, int, int]] = []
    for path in candidates:
        task_text, episode_text, segment_index = parse_agibot_alpha_segment_id(
            path.name
        )
        parsed.append((path, int(task_text), int(episode_text), segment_index))
    table = load_task_table(task_info_dir, (item[1] for item in parsed))
    segments: list[Segment] = []
    for path, task_id, episode_id, segment_index in parsed:
        if not path.is_dir():
            raise ConversionError(f"listed raw segment is missing: {path}")
        for required in (path / "head_color.mp4", path / "proprio_stats.h5"):
            if not required.is_file():
                raise ConversionError(f"raw segment asset is missing: {required}")
        key = (task_id, episode_id, segment_index)
        if key not in table:
            raise ConversionError(
                f"task metadata has no segment {task_id}-{episode_id}-{segment_index:03d}"
            )
        metadata = table[key]
        segments.append(
            Segment(
                raw_dir=path,
                task_id=task_id,
                source_episode_id=episode_id,
                segment_index=segment_index,
                skill=metadata["skill"],
                action_text=metadata["action_text"],
                task_name=metadata["task_name"],
                expected_frames=metadata["expected_frames"],
            )
        )
    segments.sort(
        key=lambda item: (
            item.task_id,
            item.source_episode_id,
            item.segment_index,
        )
    )
    return segments


class RunningStats:
    """Exact first and second moments with bounded quantile samples."""

    def __init__(self, width: int, sample_rows: int) -> None:
        self.width = width
        self.sample_rows = sample_rows
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_square = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)
        self.samples: list[np.ndarray] = []
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if not array.size:
            return
        if not np.isfinite(array).all():
            raise ConversionError("statistics input contains non-finite values")
        self.count += len(array)
        self.total += array.sum(axis=0)
        self.total_square += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))
        remaining = self.sample_rows - self.sample_count
        if remaining <= 0:
            return
        if len(array) > remaining:
            indices = np.linspace(0, len(array) - 1, remaining, dtype=np.int64)
            array = array[indices]
        self.samples.append(array.astype(np.float32, copy=True))
        self.sample_count += len(array)

    def as_dict(self) -> dict[str, list[float]]:
        if not self.count:
            zeros = [0.0] * self.width
            return {name: zeros for name in ("mean", "std", "min", "max", "q01", "q99")}
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        sample = np.concatenate(self.samples, axis=0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "q01": np.quantile(sample, 0.01, axis=0).tolist(),
            "q99": np.quantile(sample, 0.99, axis=0).tolist(),
        }


def modality_metadata() -> dict[str, Any]:
    """Return the self-contained AgiBot state/action layout."""

    def fields(
        original_key: str,
        layout: Sequence[tuple[str, int, int]],
    ) -> dict[str, Any]:
        return {
            name: {
                "original_key": original_key,
                "start": start,
                "end": end,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float64",
                "range": None,
            }
            for name, start, end in layout
        }

    return {
        "state": fields("observation.state", STATE_LAYOUT),
        "action": fields("action", ACTION_LAYOUT),
        "video": {"top_head": {"original_key": VIDEO_KEY}},
    }


def build_info(
    episode_count: int,
    frame_count: int,
    task_count: int,
    chunk_size: int,
    fps: float,
    split: str,
) -> dict[str, Any]:
    """Build the LeRobot metadata consumed by the external ACWM loader."""

    return {
        "codebase_version": "v2.0",
        "robot_type": "AgiBotWorldAlphaRaw",
        "total_episodes": episode_count,
        "total_frames": frame_count,
        "total_tasks": task_count,
        "total_videos": episode_count,
        "total_chunks": math.ceil(episode_count / chunk_size),
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {split: "0:100"},
        "data_path": DATA_PATH_PATTERN,
        "video_path": VIDEO_PATH_PATTERN,
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {"dtype": "object", "shape": [20]},
            "action": {"dtype": "object", "shape": [22]},
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_dataset_path_yaml(path: Path, dataset_roots: Sequence[Path]) -> None:
    """Write paths relative to the YAML location for portable bridge builds."""

    lines = ["dataset_path:"]
    base = path.resolve().parent
    for root in dataset_roots:
        relative = os.path.relpath(root.resolve(), base)
        lines.append(f"  - {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_parquet(
    path: Path,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    task_index: int,
    episode_index: int,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "AgiBot conversion requires pandas and a Parquet engine in the "
            "model-integration environment"
        ) from exc
    frames = len(state)
    frame_index = np.arange(frames, dtype=np.int64)
    table = pd.DataFrame(
        {
            "observation.state": [row.copy() for row in state],
            "action": [row.copy() for row in action],
            "timestamp": timestamp,
            "task_index": np.full(frames, task_index, dtype=np.int64),
            "episode_index": np.full(frames, episode_index, dtype=np.int64),
            "index": frame_index,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("AgiBot conversion requires pyarrow or fastparquet") from exc


class DatasetWriter:
    """Write one split-specific LeRobot dataset root."""

    def __init__(
        self,
        root: Path,
        split: str,
        chunk_size: int,
        fps: float,
        sample_rows: int,
        video_mode: str,
        verify_video: bool,
        min_frames: int,
    ) -> None:
        self.root = root
        self.split = split
        self.chunk_size = chunk_size
        self.fps = fps
        self.video_mode = video_mode
        self.verify_video = verify_video
        self.min_frames = min_frames
        self.episodes: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []
        self.tasks: dict[str, int] = {}
        self.state_stats = RunningStats(20, sample_rows)
        self.action_stats = RunningStats(22, sample_rows)
        scalar_rows = min(sample_rows, 100_000)
        self.timestamp_stats = RunningStats(1, scalar_rows)
        self.task_stats = RunningStats(1, scalar_rows)
        self.episode_stats = RunningStats(1, scalar_rows)
        self.frame_stats = RunningStats(1, scalar_rows)

    def write(self, segment: Segment, split_detail: str) -> None:
        state, action, timestamp = read_proprio(segment.h5_path)
        raw_proprio_frames = len(state)
        video_metadata: dict[str, float | int] = {}
        alignment_policy = "video_frame_count_unavailable"
        if self.verify_video:
            video_metadata = validate_video_metadata(segment.video_path, self.fps)
            state, action, timestamp, alignment_policy = align_proprio_to_video(
                state,
                action,
                timestamp,
                int(video_metadata["frames"]),
            )
        frames = len(state)
        if frames < self.min_frames:
            raise ConversionError(
                f"segment {segment.source_id} has {frames} aligned frames; "
                f"minimum is {self.min_frames}"
            )

        task_text = ": ".join(
            part for part in (segment.skill, segment.task_name) if part
        )
        if not task_text:
            task_text = segment.action_text or segment.source_id
        task_index = self.tasks.setdefault(task_text, len(self.tasks))
        episode_index = len(self.episodes)
        episode_chunk = episode_index // self.chunk_size
        parquet_path = self.root / DATA_PATH_PATTERN.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        video_path = self.root / VIDEO_PATH_PATTERN.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=VIDEO_KEY,
        )
        _write_parquet(
            parquet_path,
            state,
            action,
            timestamp,
            task_index,
            episode_index,
        )
        materialize_video(segment.video_path, video_path, self.video_mode)

        frame_index = np.arange(frames, dtype=np.float64).reshape(-1, 1)
        self.state_stats.update(state)
        self.action_stats.update(action)
        self.timestamp_stats.update(timestamp.reshape(-1, 1))
        self.task_stats.update(np.full((frames, 1), task_index, dtype=np.float64))
        self.episode_stats.update(np.full((frames, 1), episode_index, dtype=np.float64))
        self.frame_stats.update(frame_index)
        episode_row = {
            "episode_index": episode_index,
            "tasks": [task_index],
            "length": frames,
            "source_id": segment.source_id,
            "physical_episode_id": segment.physical_episode_id,
            "task_id": segment.task_id,
            "source_episode_id": segment.source_episode_id,
            "segment_index": segment.segment_index,
            "split": self.split,
        }
        self.episodes.append(episode_row)
        self.provenance.append(
            {
                "dataset": "agibot_alpha",
                "source_id": segment.source_id,
                "physical_episode_id": segment.physical_episode_id,
                "task_id": segment.task_id,
                "source_episode_id": segment.source_episode_id,
                "segment_index": segment.segment_index,
                "split": self.split,
                "split_detail": split_detail,
                "raw_relative_dir": f"train/{segment.source_id}",
                "output_episode_index": episode_index,
                "frames": frames,
                "raw_proprio_frames": raw_proprio_frames,
                "alignment_policy": alignment_policy,
                "expected_frames": segment.expected_frames,
                "video": video_metadata,
            }
        )

    def finish(self) -> dict[str, Any]:
        if not self.episodes:
            return {
                "dataset_root": None,
                "episodes": 0,
                "frames": 0,
                "tasks": 0,
            }
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        _write_json(meta / "modality.json", modality_metadata())
        _write_jsonl(meta / "episodes.jsonl", self.episodes)
        task_rows = [
            {"task_index": index, "task": task}
            for task, index in sorted(self.tasks.items(), key=lambda item: item[1])
        ]
        _write_jsonl(meta / "tasks.jsonl", task_rows)
        stats = {
            "observation.state": self.state_stats.as_dict(),
            "action": self.action_stats.as_dict(),
            "timestamp": self.timestamp_stats.as_dict(),
            "task_index": self.task_stats.as_dict(),
            "episode_index": self.episode_stats.as_dict(),
            "frame_index": self.frame_stats.as_dict(),
            "index": self.frame_stats.as_dict(),
        }
        _write_json(meta / "stats.json", stats)
        frame_count = sum(int(row["length"]) for row in self.episodes)
        _write_json(
            meta / "info.json",
            build_info(
                len(self.episodes),
                frame_count,
                len(self.tasks),
                self.chunk_size,
                self.fps,
                self.split,
            ),
        )
        return {
            "dataset_root": str(self.root),
            "episodes": len(self.episodes),
            "frames": frame_count,
            "tasks": len(self.tasks),
        }


def _safe_output_paths(raw_root: Path, output_root: Path) -> tuple[Path, Path]:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if output_root == Path("/") or output_root == raw_root:
        raise ConversionError(f"unsafe output root: {output_root}")
    if output_root in raw_root.parents or raw_root in output_root.parents:
        raise ConversionError(
            "output root and raw root must not contain one another: "
            f"{output_root}, {raw_root}"
        )
    staging = output_root.with_name(f".{output_root.name}.cdlam-partial")
    return output_root, staging


def convert(args: argparse.Namespace) -> dict[str, Any]:
    """Run a complete, split-safe raw conversion."""

    raw_root, output_root = args.raw_root.resolve(), args.output.resolve()
    output_root, staging = _safe_output_paths(raw_root, output_root)
    if output_root.exists():
        if not args.overwrite:
            raise ConversionError(
                f"output already exists: {output_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    if staging.exists():
        if not args.overwrite:
            raise ConversionError(
                f"stale partial output exists: {staging}; pass --overwrite to replace it"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    source_ids = parse_clip_list(args.clip_list) if args.clip_list else None
    segments = discover_segments(raw_root, source_ids)
    if not segments:
        raise ConversionError("no valid raw segments were selected")
    all_task_ids = available_task_ids(raw_root / "task_info")
    held_tasks = held_out_task_ids(all_task_ids, args.held_out_tasks)
    selected_splits = tuple(args.splits)
    writers = {
        split: DatasetWriter(
            root=staging / split,
            split=split,
            chunk_size=args.chunk_size,
            fps=args.fps,
            sample_rows=args.stats_sample_rows,
            video_mode=args.video_mode,
            verify_video=not args.skip_video_check,
            min_frames=args.min_frames,
        )
        for split in selected_splits
    }
    errors: list[dict[str, str]] = []
    for index, segment in enumerate(segments, 1):
        split, detail = split_for_source_id(
            segment.source_id,
            held_tasks,
            args.eval_percent,
        )
        if split not in writers:
            continue
        try:
            writers[split].write(segment, detail)
        except Exception as exc:
            if args.on_error == "fail":
                raise ConversionError(
                    f"failed to convert {segment.source_id}: {exc}"
                ) from exc
            errors.append({"source_id": segment.source_id, "error": str(exc)})
        if args.log_every and index % args.log_every == 0:
            converted = sum(len(writer.episodes) for writer in writers.values())
            print(
                f"[convert] scanned={index}/{len(segments)} "
                f"converted={converted} errors={len(errors)}",
                flush=True,
            )

    split_summaries = {split: writer.finish() for split, writer in writers.items()}
    provenance = [row for split in selected_splits for row in writers[split].provenance]
    _write_jsonl(staging / "provenance.jsonl", provenance)
    for split, summary in split_summaries.items():
        if summary["dataset_root"]:
            write_dataset_path_yaml(
                staging / f"_dataset_paths_{split}.yaml",
                [staging / split],
            )
    nonempty_roots = [
        staging / split
        for split in selected_splits
        if split_summaries[split]["dataset_root"]
    ]
    write_dataset_path_yaml(staging / "_dataset_paths_all.yaml", nonempty_roots)
    final_split_summaries = {
        split: {
            **values,
            "dataset_root": (
                str(output_root / split) if values["dataset_root"] else None
            ),
        }
        for split, values in split_summaries.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "video_mode": args.video_mode,
        "video_metadata_verified": not args.skip_video_check,
        "held_out_task_ids": sorted(held_tasks),
        "eval_percent": args.eval_percent,
        "selected_splits": list(selected_splits),
        "source_segments": len(segments),
        "converted_segments": len(provenance),
        "errors": errors,
        "splits": final_split_summaries,
        "action_contract": {
            "dimension": 22,
            "layout": [
                {"name": name, "start": start, "end": end}
                for name, start, end in ACTION_LAYOUT
            ],
            "state_policy": "preserve-official-state-arrays",
            "action_policy": "preserve-official-action-arrays",
            "unit_policy": "no-implicit-unit-conversion",
        },
    }
    _write_json(staging / "build_summary.json", summary)
    os.replace(staging, output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    unknown = sorted(set(splits) - set(VALID_SPLITS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported splits: {unknown}")
    if len(set(splits)) != len(splits):
        raise argparse.ArgumentTypeError("split names must be unique")
    return splits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="AgiBot Alpha root containing train/ and task_info/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output root; split-specific LeRobot roots are written below it.",
    )
    parser.add_argument(
        "--clip-list",
        type=Path,
        help="Optional text or recipe JSON source-ID allowlist for a bounded run.",
    )
    parser.add_argument(
        "--splits",
        type=_parse_splits,
        default=VALID_SPLITS,
        help="Comma-separated outputs (default: train,eval,test).",
    )
    parser.add_argument(
        "--video-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Copy videos for portability (default) or explicitly symlink them.",
    )
    parser.add_argument(
        "--skip-video-check",
        action="store_true",
        help="Skip PyAV stream/frame/FPS validation; intended only for plumbing tests.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--held-out-tasks", type=int, default=DEFAULT_HELD_OUT_TASKS)
    parser.add_argument("--eval-percent", type=int, default=DEFAULT_EVAL_PERCENT)
    parser.add_argument("--stats-sample-rows", type=int, default=2_000_000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--on-error",
        choices=("fail", "skip"),
        default="fail",
        help="Fail closed by default; skip is explicit and recorded in build_summary.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and stale partial output.",
    )
    args = parser.parse_args(argv)
    for name in ("fps",):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("chunk_size", "min_frames", "stats_sample_rows"):
        if isinstance(getattr(args, name), bool) or getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if not 0 <= args.eval_percent < 100:
        parser.error("--eval-percent must be in [0, 100)")
    if args.held_out_tasks < 0:
        parser.error("--held-out-tasks must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        convert(parse_args(argv))
    except (ConversionError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
