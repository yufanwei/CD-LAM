#!/usr/bin/env python3
"""Materialize official AgiBotWorld Alpha episodes as action segments.

The official dataset stores one video and one proprioception file per physical
episode. CD-LAM's raw converter consumes one directory per annotated action
segment. This tool validates the publisher layout, applies the same
``[start_frame, end_frame)`` bounds to video and frame-aligned HDF5 datasets,
and writes the exact intermediate tree consumed by that converter.

Selection is always episode-granular. ``--max-episodes`` therefore cannot
split one physical recording across separate runs or downstream data splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
DATASET_ID = "agibot-world/AgiBotWorld-Alpha"
SCHEMA_REFERENCE_REVISION = "128665c9e0244c45d1cbe5c13f5a4706afd24f27"
SCHEMA_REFERENCE_URL = (
    "https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/"
    f"blob/{SCHEMA_REFERENCE_REVISION}/README.md"
)
SAMPLE_ARCHIVE_BYTES = 7_097_989_120
SAMPLE_ARCHIVE_SHA256 = (
    "131c6f99ebe6900e93d56be9f0cbe46f2cff286b8d9102b8d3e01d25f7cebe5e"
)
SOURCE_VIDEO_NAME = "head_color.mp4"
SOURCE_H5_NAME = "proprio_stats.h5"
FPS = 30
HEIGHT = 480
WIDTH = 640

REQUIRED_FRAME_DATASETS: dict[str, tuple[int, ...]] = {
    "state/joint/position": (14,),
    "state/effector/position": (2,),
    "state/head/position": (2,),
    "state/waist/position": (2,),
    "action/joint/position": (14,),
    "action/effector/position": (2,),
    "action/head/position": (2,),
    "action/waist/position": (2,),
    "action/robot/velocity": (2,),
}


class MaterializationError(ValueError):
    """Raised when input cannot satisfy the materialization contract."""


@dataclass(frozen=True)
class SegmentSpec:
    """One action annotation with stable source identity."""

    task_id: int
    episode_id: int
    segment_index: int
    start_frame: int
    end_frame: int
    action_text: str
    skill: str

    @property
    def source_id(self) -> str:
        return f"{self.task_id}-{self.episode_id}-{self.segment_index:03d}"

    @property
    def physical_episode_id(self) -> str:
        return f"agibot_alpha:{self.task_id}:{self.episode_id}"

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class EpisodeSpec:
    """One selected official episode and all of its action segments."""

    task_id: int
    episode_id: int
    metadata: dict[str, Any]
    segments: tuple[SegmentSpec, ...]


def _strict_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaterializationError(f"{name} must be an integer, got {value!r}")
    return value


def _strict_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise MaterializationError(f"{name} must be a string, got {value!r}")
    return value.strip()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_provenance(
    record_path: Path | None, revision_attestation: str | None
) -> dict[str, Any]:
    """Verify a downloader record or explicitly mark unverified local input."""

    if record_path is None:
        return {
            "source_revision": revision_attestation,
            "source_revision_verification": (
                "caller_attested" if revision_attestation else "unverified_local_input"
            ),
            "source_archive_sha256": None,
            "source_record": None,
        }
    record_path = record_path.expanduser().resolve()
    record = _load_json(record_path)
    if not isinstance(record, dict):
        raise MaterializationError("source record must contain a JSON object")
    revision = record.get("revision")
    archive_hash = record.get("archive_sha256")
    archive_value = record.get("archive_path")
    if (
        record.get("dataset") != DATASET_ID
        or record.get("filename") != "sample_dataset.tar"
        or record.get("revision_verified") is not True
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or revision != SCHEMA_REFERENCE_REVISION
        or not isinstance(archive_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", archive_hash) is None
        or archive_hash != SAMPLE_ARCHIVE_SHA256
        or not isinstance(archive_value, str)
        or not archive_value
    ):
        raise MaterializationError(
            "source record is not a verified AgiBot archive record"
        )
    if revision_attestation and revision_attestation != revision:
        raise MaterializationError(
            "--source-revision does not match the verified source record"
        )
    archive = Path(archive_value).expanduser()
    if not archive.is_absolute():
        archive = record_path.parent / archive
    archive = archive.resolve()
    if not archive.is_file():
        raise MaterializationError(f"source-record archive is missing: {archive}")
    expected_bytes = record.get("archive_bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != SAMPLE_ARCHIVE_BYTES
        or archive.stat().st_size != expected_bytes
        or _sha256(archive) != archive_hash
    ):
        raise MaterializationError(
            "source-record archive bytes or SHA-256 do not match"
        )
    return {
        "source_revision": revision,
        "source_revision_verification": "verified_archive_record",
        "source_archive_sha256": archive_hash,
        "source_record": str(record_path),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_episode_selector(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "episode selectors must use <task-id>-<episode-id>"
        )
    try:
        task_id, episode_id = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "episode selectors must contain integer IDs"
        ) from exc
    if task_id < 0 or episode_id < 0:
        raise argparse.ArgumentTypeError("episode selector IDs must be non-negative")
    return task_id, episode_id


def _task_ids(task_info_root: Path, requested: set[int] | None) -> list[int]:
    available: list[int] = []
    for path in task_info_root.glob("task_*.json"):
        try:
            available.append(int(path.stem.removeprefix("task_")))
        except ValueError:
            continue
    available = sorted(set(available))
    if not available:
        raise MaterializationError(
            f"no task_*.json metadata files found under {task_info_root}"
        )
    if requested is None:
        return available
    missing = sorted(requested - set(available))
    if missing:
        raise MaterializationError(f"requested task metadata is missing: {missing}")
    return sorted(requested)


def _segments_from_metadata(
    task_id: int,
    episode_id: int,
    metadata: Mapping[str, Any],
) -> tuple[SegmentSpec, ...]:
    label_info = metadata.get("label_info")
    if not isinstance(label_info, Mapping):
        if "lable_info" in metadata:
            raise MaterializationError(
                f"task {task_id} episode {episode_id} uses misspelled lable_info; "
                "expected the publisher label_info schema"
            )
        raise MaterializationError(
            f"task {task_id} episode {episode_id} has no label_info object"
        )
    configs = label_info.get("action_config")
    if not isinstance(configs, list) or not configs:
        raise MaterializationError(
            f"task {task_id} episode {episode_id} has no action_config segments"
        )
    segments: list[SegmentSpec] = []
    previous_end = 0
    for index, config in enumerate(configs):
        if not isinstance(config, Mapping):
            raise MaterializationError(
                f"action_config[{index}] for {task_id}-{episode_id} must be an object"
            )
        start = _strict_int(
            f"action_config[{index}].start_frame", config.get("start_frame")
        )
        end = _strict_int(f"action_config[{index}].end_frame", config.get("end_frame"))
        if start < 0 or end <= start:
            raise MaterializationError(
                f"invalid frame bounds for {task_id}-{episode_id}-{index:03d}: "
                f"[{start}, {end})"
            )
        if index and start < previous_end:
            raise MaterializationError(
                f"overlapping or unsorted frame bounds for {task_id}-{episode_id}: "
                f"segment {index} starts at {start}, before {previous_end}"
            )
        previous_end = end
        segments.append(
            SegmentSpec(
                task_id=task_id,
                episode_id=episode_id,
                segment_index=index,
                start_frame=start,
                end_frame=end,
                action_text=_strict_text(
                    f"action_config[{index}].action_text",
                    config.get("action_text", ""),
                ),
                skill=_strict_text(
                    f"action_config[{index}].skill", config.get("skill", "")
                ),
            )
        )
    return tuple(segments)


def discover_episodes(
    raw_root: Path,
    requested_tasks: set[int] | None,
    requested_episodes: set[tuple[int, int]] | None,
    max_episodes: int | None,
) -> list[EpisodeSpec]:
    """Discover deterministic episode-granular work from publisher metadata."""

    task_info_root = raw_root / "task_info"
    if not task_info_root.is_dir():
        raise MaterializationError(f"task_info directory is missing: {task_info_root}")
    tasks_from_episode_selector = (
        {task for task, _ in requested_episodes} if requested_episodes else set()
    )
    if requested_tasks is not None:
        unknown = tasks_from_episode_selector - requested_tasks
        if unknown:
            raise MaterializationError(
                f"--episode selects tasks excluded by --task-id: {sorted(unknown)}"
            )
    effective_tasks = requested_tasks or tasks_from_episode_selector or None
    selected: list[EpisodeSpec] = []
    seen_episode_keys: set[tuple[int, int]] = set()
    for task_id in _task_ids(task_info_root, effective_tasks):
        path = task_info_root / f"task_{task_id}.json"
        rows = _load_json(path)
        if not isinstance(rows, list):
            raise MaterializationError(f"task metadata must be a list: {path}")
        for row_index, raw_metadata in enumerate(rows):
            if not isinstance(raw_metadata, Mapping):
                raise MaterializationError(
                    f"{path} row {row_index} must be a JSON object"
                )
            metadata = dict(raw_metadata)
            episode_id = _strict_int(
                f"{path.name}[{row_index}].episode_id", metadata.get("episode_id")
            )
            metadata_task = _strict_int(
                f"{path.name}[{row_index}].task_id", metadata.get("task_id")
            )
            if metadata_task != task_id:
                raise MaterializationError(
                    f"task ID mismatch in {path}: filename={task_id}, "
                    f"metadata={metadata_task}"
                )
            key = (task_id, episode_id)
            if key in seen_episode_keys:
                raise MaterializationError(f"duplicate episode metadata: {key}")
            seen_episode_keys.add(key)
            if requested_episodes is not None and key not in requested_episodes:
                continue
            selected.append(
                EpisodeSpec(
                    task_id=task_id,
                    episode_id=episode_id,
                    metadata=metadata,
                    segments=_segments_from_metadata(task_id, episode_id, metadata),
                )
            )
    selected.sort(key=lambda item: (item.task_id, item.episode_id))
    if requested_episodes is not None:
        discovered = {(item.task_id, item.episode_id) for item in selected}
        missing = sorted(requested_episodes - discovered)
        if missing:
            raise MaterializationError(
                f"requested episode metadata is missing: {missing}"
            )
    if max_episodes is not None:
        selected = selected[:max_episodes]
    if not selected:
        raise MaterializationError("episode selection is empty")
    return selected


def inspect_proprio(path: Path) -> int:
    """Validate the official HDF5 subset needed by the downstream converter."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("AgiBot materialization requires h5py") from exc
    if not path.is_file():
        raise MaterializationError(f"proprioception file is missing: {path}")
    with h5py.File(path, "r") as handle:
        if "timestamp" not in handle:
            raise MaterializationError(f"timestamp dataset is missing: {path}")
        timestamp = np.asarray(handle["timestamp"][:])
        if timestamp.ndim != 1 or not np.issubdtype(timestamp.dtype, np.integer):
            raise MaterializationError(
                f"timestamp must be a one-dimensional integer dataset: {path}"
            )
        if timestamp.size == 0:
            raise MaterializationError(f"timestamp dataset is empty: {path}")
        if timestamp.size > 1 and not bool(np.all(np.diff(timestamp) > 0)):
            raise MaterializationError(
                f"timestamps are not strictly increasing: {path}"
            )
        frames = len(timestamp)
        for name, trailing_shape in REQUIRED_FRAME_DATASETS.items():
            if name not in handle:
                raise MaterializationError(
                    f"required dataset {name} is missing: {path}"
                )
            dataset = handle[name]
            expected = (frames, *trailing_shape)
            if dataset.shape != expected:
                raise MaterializationError(
                    f"{name} has shape {dataset.shape}; expected {expected}: {path}"
                )
            if not np.issubdtype(dataset.dtype, np.number):
                raise MaterializationError(f"{name} must be numeric: {path}")
    return frames


def _copy_attributes(source: Any, destination: Any) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def slice_proprio(
    source_path: Path,
    destination_path: Path,
    start_frame: int,
    end_frame: int,
    episode_frames: int,
) -> None:
    """Slice all frame-aligned datasets and rebase action index datasets."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("AgiBot materialization requires h5py") from exc
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(source_path, "r") as source,
        h5py.File(destination_path, "w") as destination,
    ):
        _copy_attributes(source, destination)

        def copy_item(name: str, item: Any) -> None:
            if isinstance(item, h5py.Group):
                group = destination.require_group(name)
                _copy_attributes(item, group)
                return
            if not isinstance(item, h5py.Dataset):
                raise MaterializationError(f"unsupported HDF5 item: {name}")
            if name.startswith("action/") and name.endswith("/index"):
                if item.ndim != 1 or not np.issubdtype(item.dtype, np.integer):
                    raise MaterializationError(
                        f"{name} must be a one-dimensional integer index dataset"
                    )
                raw = np.asarray(item[:])
                data = raw[(raw >= start_frame) & (raw < end_frame)] - start_frame
            elif item.ndim >= 1 and item.shape[0] == episode_frames:
                data = item[start_frame:end_frame]
            elif item.ndim == 0:
                data = item[()]
            else:
                raise MaterializationError(
                    f"cannot safely slice non-frame-aligned dataset {name} with "
                    f"shape {item.shape}; episode has {episode_frames} frames"
                )
            output = destination.create_dataset(name, data=data, dtype=item.dtype)
            _copy_attributes(item, output)

        source.visititems(copy_item)


class _ClipWriter:
    """Small deterministic H.264 writer for one frame-indexed segment."""

    def __init__(self, path: Path) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("AgiBot video materialization requires PyAV") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=Fraction(FPS, 1))
        self._stream.width = WIDTH
        self._stream.height = HEIGHT
        self._stream.pix_fmt = "yuv420p"
        self._stream.time_base = Fraction(1, FPS)
        self._stream.codec_context.thread_count = 1
        self._stream.options = {
            "bf": "0",
            "crf": "18",
            "g": str(FPS),
            "preset": "medium",
        }
        self.frames = 0
        self._closed = False

    def write(self, frame: Any) -> None:
        converted = frame.reformat(width=WIDTH, height=HEIGHT, format="yuv420p")
        converted.pts = self.frames
        converted.time_base = Fraction(1, FPS)
        for packet in self._stream.encode(converted):
            self._container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._closed = True


def _verify_clip(path: Path, expected_frames: int) -> None:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("AgiBot video verification requires PyAV") from exc
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise MaterializationError(
                f"output clip must have one video stream: {path}"
            )
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (WIDTH, HEIGHT):
            raise MaterializationError(
                f"output clip dimensions are {stream.width}x{stream.height}: {path}"
            )
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        if abs(rate - FPS) > 0.05:
            raise MaterializationError(
                f"output clip FPS is {rate}, expected {FPS}: {path}"
            )
        decoded = sum(1 for _ in container.decode(video=0))
    if decoded != expected_frames:
        raise MaterializationError(
            f"output clip has {decoded} frames, expected {expected_frames}: {path}"
        )


def slice_episode_video(
    source_path: Path,
    output_train_root: Path,
    segments: Sequence[SegmentSpec],
    episode_frames: int,
) -> dict[str, int]:
    """Decode an episode once and write all selected non-overlapping clips."""

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("AgiBot video materialization requires PyAV") from exc
    if not source_path.is_file():
        raise MaterializationError(f"head video is missing: {source_path}")
    writers: dict[str, _ClipWriter] = {}
    frame_counts: dict[str, int] = {segment.source_id: 0 for segment in segments}
    active_index = 0
    active_writer: _ClipWriter | None = None
    decoded_frames = 0
    try:
        with av.open(str(source_path)) as container:
            if len(container.streams.video) != 1:
                raise MaterializationError(
                    f"source must contain exactly one video stream: {source_path}"
                )
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (WIDTH, HEIGHT):
                raise MaterializationError(
                    f"source dimensions are {stream.width}x{stream.height}; "
                    f"expected {WIDTH}x{HEIGHT}: {source_path}"
                )
            rate = float(stream.average_rate) if stream.average_rate else 0.0
            if abs(rate - FPS) > 0.05:
                raise MaterializationError(
                    f"source FPS is {rate}, expected {FPS}: {source_path}"
                )
            for frame_index, frame in enumerate(container.decode(video=0)):
                decoded_frames = frame_index + 1
                while (
                    active_index < len(segments)
                    and frame_index >= segments[active_index].end_frame
                ):
                    if active_writer is not None:
                        active_writer.close()
                        active_writer = None
                    active_index += 1
                if active_index >= len(segments):
                    continue
                segment = segments[active_index]
                if segment.start_frame <= frame_index < segment.end_frame:
                    if active_writer is None:
                        output = (
                            output_train_root / segment.source_id / SOURCE_VIDEO_NAME
                        )
                        active_writer = _ClipWriter(output)
                        writers[segment.source_id] = active_writer
                    active_writer.write(frame)
                    frame_counts[segment.source_id] += 1
                    if frame_index + 1 == segment.end_frame:
                        active_writer.close()
                        active_writer = None
                        active_index += 1
        if active_writer is not None:
            active_writer.close()
            active_writer = None
    finally:
        if active_writer is not None:
            active_writer.close()
        for writer in writers.values():
            writer.close()
    if decoded_frames not in {episode_frames, episode_frames - 1}:
        raise MaterializationError(
            "video/proprio frame mismatch is not a single terminal proprio record: "
            f"video={decoded_frames}, proprio={episode_frames}, source={source_path}"
        )
    for segment in segments:
        expected = max(
            0,
            min(segment.end_frame, decoded_frames) - segment.start_frame,
        )
        if expected <= 0:
            raise MaterializationError(
                f"segment {segment.source_id} has no decodable video frames"
            )
        actual = frame_counts[segment.source_id]
        if actual != expected:
            raise MaterializationError(
                f"segment {segment.source_id} wrote {actual} frames, expected {expected}"
            )
        output = output_train_root / segment.source_id / SOURCE_VIDEO_NAME
        _verify_clip(output, expected)
    return frame_counts


def _source_paths(raw_root: Path, episode: EpisodeSpec) -> tuple[Path, Path]:
    video = (
        raw_root
        / "observations"
        / str(episode.task_id)
        / str(episode.episode_id)
        / "videos"
        / SOURCE_VIDEO_NAME
    )
    proprio = (
        raw_root
        / "proprio_stats"
        / str(episode.task_id)
        / str(episode.episode_id)
        / SOURCE_H5_NAME
    )
    return video, proprio


def _safe_paths(raw_root: Path, output_root: Path) -> tuple[Path, Path, Path]:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if output_root == Path("/") or output_root == raw_root:
        raise MaterializationError("output must be a new non-root path")
    if output_root in raw_root.parents or raw_root in output_root.parents:
        raise MaterializationError(
            f"raw and output roots must not contain one another: {raw_root}, {output_root}"
        )
    staging = output_root.with_name(f".{output_root.name}.staging")
    backup = output_root.with_name(f".{output_root.name}.backup")
    return output_root, staging, backup


def _commit_output(
    staging: Path,
    output_root: Path,
    backup: Path,
    overwrite: bool,
) -> None:
    if output_root.is_symlink() or backup.is_symlink():
        raise MaterializationError("refusing to replace a symlinked output or backup")
    if output_root.exists():
        if not overwrite:
            raise MaterializationError(
                f"output already exists: {output_root}; pass --overwrite to replace it"
            )
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output_root, backup)
        try:
            os.replace(staging, output_root)
        except Exception:
            os.replace(backup, output_root)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output_root)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    """Run a complete, atomic publisher-layout materialization."""

    raw_root = args.raw_root.resolve()
    if not raw_root.is_dir():
        raise MaterializationError(f"raw root is missing: {raw_root}")
    output_root, staging, backup = _safe_paths(raw_root, args.output)
    source_provenance = _source_provenance(args.source_record, args.source_revision)
    if output_root.exists() and not args.overwrite:
        raise MaterializationError(
            f"output already exists: {output_root}; pass --overwrite to replace it"
        )
    if staging.is_symlink() or backup.is_symlink():
        raise MaterializationError("refusing to use a symlinked staging or backup path")
    for partial in (staging, backup):
        if partial.exists():
            if not args.overwrite:
                raise MaterializationError(
                    f"stale partial output exists: {partial}; pass --overwrite to remove it"
                )
            shutil.rmtree(partial)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    requested_tasks = set(args.task_id) if args.task_id else None
    requested_episodes = set(args.episode) if args.episode else None
    episodes = discover_episodes(
        raw_root,
        requested_tasks,
        requested_episodes,
        args.max_episodes,
    )
    staging.mkdir(parents=True)
    provenance: list[dict[str, Any]] = []
    filtered_metadata: dict[int, list[dict[str, Any]]] = {}
    try:
        for episode_index, episode in enumerate(episodes, 1):
            source_video, source_h5 = _source_paths(raw_root, episode)
            episode_frames = inspect_proprio(source_h5)
            for segment in episode.segments:
                if segment.end_frame > episode_frames:
                    raise MaterializationError(
                        f"segment {segment.source_id} ends at {segment.end_frame}, "
                        f"beyond {episode_frames} proprio frames"
                    )
            video_counts = slice_episode_video(
                source_video,
                staging / "train",
                episode.segments,
                episode_frames,
            )
            for segment in episode.segments:
                destination_dir = staging / "train" / segment.source_id
                destination_h5 = destination_dir / SOURCE_H5_NAME
                slice_proprio(
                    source_h5,
                    destination_h5,
                    segment.start_frame,
                    segment.end_frame,
                    episode_frames,
                )
                output_video = destination_dir / SOURCE_VIDEO_NAME
                provenance.append(
                    {
                        "action_text": segment.action_text,
                        "bounds": {
                            "end_frame_exclusive": segment.end_frame,
                            "start_frame_inclusive": segment.start_frame,
                        },
                        "dataset": "agibot_alpha",
                        "h5_sha256": _sha256(destination_h5),
                        "physical_episode_id": segment.physical_episode_id,
                        "proprio_frames": segment.frame_count,
                        "segment_index": segment.segment_index,
                        "skill": segment.skill,
                        "source_episode_id": segment.episode_id,
                        "source_h5": (
                            f"proprio_stats/{segment.task_id}/{segment.episode_id}/"
                            f"{SOURCE_H5_NAME}"
                        ),
                        "source_id": segment.source_id,
                        "source_task_id": segment.task_id,
                        "source_video": (
                            f"observations/{segment.task_id}/{segment.episode_id}/"
                            f"videos/{SOURCE_VIDEO_NAME}"
                        ),
                        "video_frames": video_counts[segment.source_id],
                        "video_sha256": _sha256(output_video),
                    }
                )
            filtered_metadata.setdefault(episode.task_id, []).append(episode.metadata)
            if args.log_every and episode_index % args.log_every == 0:
                print(
                    f"[materialize] episodes={episode_index}/{len(episodes)} "
                    f"segments={len(provenance)}",
                    flush=True,
                )
        for task_id, rows in sorted(filtered_metadata.items()):
            rows.sort(key=lambda item: int(item["episode_id"]))
            _write_json(staging / "task_info" / f"task_{task_id}.json", rows)
        _write_jsonl(staging / "provenance.jsonl", provenance)
        (staging / "source_ids.txt").write_text(
            "".join(f"{row['source_id']}\n" for row in provenance),
            encoding="utf-8",
        )
        summary = {
            "bounds_semantics": "start-inclusive_end-exclusive",
            "dataset_id": DATASET_ID,
            "episode_count": len(episodes),
            "max_episodes": args.max_episodes,
            "schema_reference_revision": SCHEMA_REFERENCE_REVISION,
            "schema_reference_url": SCHEMA_REFERENCE_URL,
            "schema_version": SCHEMA_VERSION,
            "segment_count": len(provenance),
            "selected_episode_ids": [
                f"{episode.task_id}-{episode.episode_id}" for episode in episodes
            ],
            "source_revision_attestation": args.source_revision,
            **source_provenance,
            "split_unit": "physical_episode",
            "video_contract": {
                "filename": SOURCE_VIDEO_NAME,
                "fps": FPS,
                "height": HEIGHT,
                "width": WIDTH,
            },
        }
        _write_json(staging / "materialization.json", summary)
        _commit_output(staging, output_root, backup, args.overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    final_summary = {**summary, "output_root": str(output_root)}
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return final_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help=(
            "Extracted official root containing task_info/, observations/, and "
            "proprio_stats/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New segmented root consumed by convert_agibot_alpha.py.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        help="Select a task ID; repeat to select more than one.",
    )
    parser.add_argument(
        "--episode",
        type=_parse_episode_selector,
        action="append",
        help="Select one complete <task-id>-<episode-id>; repeat as needed.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Deterministic bounded run after sorting complete episodes by IDs.",
    )
    parser.add_argument(
        "--source-revision",
        help=(
            "Optional revision attestation for the local raw files; omitted values are "
            "recorded as null rather than guessed."
        ),
    )
    parser.add_argument(
        "--source-record",
        type=Path,
        help="Verified agibot_download.json record for a pinned archive.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after a successful build.",
    )
    args = parser.parse_args(argv)
    if args.task_id and any(task_id < 0 for task_id in args.task_id):
        parser.error("--task-id must be non-negative")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be a positive integer")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        materialize(parse_args(argv))
    except (
        MaterializationError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
