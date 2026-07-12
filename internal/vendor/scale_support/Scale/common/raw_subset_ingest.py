"""Build a bounded Stage-1/Stage-2 subset directly from raw video clips.

This builder is deliberately smaller than the paper data pipeline.  It accepts
an explicit JSONL clip index, embeds each selected MP4 in one Parquet shard,
and writes train/evaluation manifests that the bundled shard decoders can use.
It does not download datasets, infer annotations, generate SAM3 masks, create
bridge actions, or reproduce the 100h/1000h sampling recipes.

Split isolation is defined by the physical recording unit from
``raw_data_contract``.  AgiBot segments from one physical episode and EgoDex
clips from one Apple session can never cross output splits.  Apple's native
EgoDex test part is retained in provenance but excluded from train/evaluation
shards and manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from Scale.common.raw_data_contract import (
    AGIBOT_ALPHA_DATASET,
    EGODEX_DATASET,
    RawDataContractError,
    agibot_alpha_physical_episode_key,
    egodex_native_split,
    egodex_physical_session_key,
    parse_agibot_alpha_segment_id,
    parse_egodex_episode_id,
    validate_raw_split_records,
)


MASK_POLICY = "full_frame_plumbing_only"
MAX_SUPPORTED_CLIPS = 256


class RawSubsetError(ValueError):
    """Raised when a raw subset cannot satisfy the release data contract."""


@dataclass(frozen=True)
class RawSubsetOptions:
    """Bounded sampling controls for the representative raw subset."""

    pair_stride: int = 1
    pairs_per_clip: int = 8
    window_frames: int = 13
    windows_per_clip: int = 4
    max_clips: int = 32
    max_total_video_bytes: int = 4 * 1024**3

    def validate(self) -> None:
        for name in (
            "pair_stride",
            "pairs_per_clip",
            "window_frames",
            "windows_per_clip",
            "max_clips",
            "max_total_video_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RawSubsetError(f"{name} must be a positive integer")
        if self.window_frames < 2:
            raise RawSubsetError("window_frames must be at least 2")
        if self.max_clips > MAX_SUPPORTED_CLIPS:
            raise RawSubsetError(
                f"max_clips cannot exceed {MAX_SUPPORTED_CLIPS}; "
                "this command is a bounded adapter test, not a full recipe builder"
            )


@dataclass(frozen=True)
class NormalizedClip:
    """Validated clip metadata before video bytes are decoded."""

    dataset: str
    source: str
    split: str
    clip_id: str
    raw_episode_id: str
    physical_group_key: str
    task_id: int
    task_name: str
    video_path: Path
    metadata_path: Path | None
    part: str
    session_name: str
    primitive: str
    primitive_raw: str
    step_starts: tuple[int, ...]
    step_ends: tuple[int, ...]
    step_actions: tuple[str, ...]
    step_skills: tuple[str, ...]


@dataclass(frozen=True)
class VideoInfo:
    """Exact sequential-frame metadata derived from one MP4 container."""

    num_frames: int
    fps: float
    timestamps_us: tuple[int, ...]


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise RawSubsetError(f"{label} must be non-empty")
    return result


def _optional_text(value: Any) -> str:
    return str(value or "").strip()


def _stable_task_id(value: str) -> int:
    return (
        int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(value: Any, root: Path, label: str) -> Path:
    raw = Path(_text(value, label)).expanduser()
    path = raw if raw.is_absolute() else root / raw
    path = path.resolve()
    if not path.is_file():
        raise RawSubsetError(f"{label} does not exist: {path}")
    return path


def _sequence_of_ints(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RawSubsetError(f"{label} must be a sequence of integers")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise RawSubsetError(f"{label}[{index}] must be an integer")
        result.append(item)
    return tuple(result)


def _sequence_of_text(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RawSubsetError(f"{label} must be a sequence of strings")
    return tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RawSubsetError(f"raw clip index does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RawSubsetError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RawSubsetError(f"row {line_number} in {path} must be an object")
        records.append(value)
    if not records:
        raise RawSubsetError(f"raw clip index is empty: {path}")
    return records


def _h5_session_name(path: Path) -> str:
    try:
        import h5py
    except ImportError as exc:
        raise RawSubsetError(
            "h5py is required when EgoDex session_name is read from metadata_h5"
        ) from exc
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("session_name")
    except Exception as exc:  # noqa: BLE001
        raise RawSubsetError(f"cannot read EgoDex metadata HDF5 {path}: {exc}") from exc
    if value is None:
        raise RawSubsetError(
            f"EgoDex metadata has no root session_name attribute: {path}"
        )
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _text(value, f"{path}.attrs['session_name']")


def _normalize_records(
    records: Sequence[Mapping[str, Any]], index_root: Path
) -> list[NormalizedClip]:
    """Normalize provenance and audit physical groups before video decoding."""

    preliminary: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    clip_ids: set[str] = set()
    for row_index, record in enumerate(records):
        dataset = _text(record.get("dataset"), f"row {row_index}.dataset")
        split = _text(record.get("split"), f"row {row_index}.split")
        if split not in {"train", "eval", "test"}:
            raise RawSubsetError(f"row {row_index}.split must be train, eval, or test")
        video_path = _resolve_file(
            record.get("video_path"), index_root, f"row {row_index}.video_path"
        )
        metadata_value = record.get("metadata_h5", record.get("proprio_h5_path"))
        metadata_path = (
            _resolve_file(metadata_value, index_root, f"row {row_index}.metadata_h5")
            if metadata_value
            else None
        )

        if dataset == AGIBOT_ALPHA_DATASET:
            source_id = _text(record.get("source_id"), f"row {row_index}.source_id")
            task, _, _ = parse_agibot_alpha_segment_id(source_id)
            physical_group_key = agibot_alpha_physical_episode_key(source_id)
            task_id = int(task)
            declared_task_id = record.get("task_id")
            if declared_task_id is not None:
                try:
                    declared_task_id = int(declared_task_id)
                except (TypeError, ValueError) as exc:
                    raise RawSubsetError(
                        f"row {row_index}.task_id must be an integer"
                    ) from exc
                if declared_task_id != task_id:
                    raise RawSubsetError(
                        f"row {row_index}.task_id disagrees with source_id"
                    )
            part = ""
            session_name = ""
            raw_episode_id = source_id
            task_name = _optional_text(record.get("task_name")) or task
            default_clip_id = f"agibot_alpha:{source_id}"
            audit_rows.append(
                {"dataset": dataset, "source_id": source_id, "split": split}
            )
        elif dataset == EGODEX_DATASET:
            raw_episode_id = _text(
                record.get("episode_id"), f"row {row_index}.episode_id"
            )
            inferred_part, task_name, _ = parse_egodex_episode_id(raw_episode_id)
            part = _optional_text(record.get("part")) or inferred_part
            if part != inferred_part:
                raise RawSubsetError(f"row {row_index}.part disagrees with episode_id")
            session_name = _optional_text(record.get("session_name"))
            if metadata_path is not None:
                metadata_session = _h5_session_name(metadata_path)
                if session_name and session_name != metadata_session:
                    raise RawSubsetError(
                        f"row {row_index}.session_name disagrees with metadata_h5"
                    )
                session_name = metadata_session
            if not session_name:
                raise RawSubsetError(
                    f"row {row_index}.session_name is required when metadata_h5 is absent"
                )
            if egodex_native_split(part) == "test" and split != "test":
                raise RawSubsetError(
                    f"row {row_index}: native EgoDex test clip must keep split='test'"
                )
            physical_group_key = egodex_physical_session_key(part, session_name)
            declared_task_id = record.get("task_id")
            if declared_task_id is None:
                task_id = _stable_task_id(f"{part}:{task_name}")
            elif isinstance(declared_task_id, bool):
                raise RawSubsetError(f"row {row_index}.task_id must be an integer")
            else:
                try:
                    task_id = int(declared_task_id)
                except (TypeError, ValueError) as exc:
                    raise RawSubsetError(
                        f"row {row_index}.task_id must be an integer"
                    ) from exc
            default_clip_id = f"egodex:{raw_episode_id}"
            audit_rows.append(
                {
                    "dataset": dataset,
                    "episode_id": raw_episode_id,
                    "part": part,
                    "session_name": session_name,
                    "split": split,
                }
            )
        else:
            raise RawSubsetError(f"row {row_index}: unsupported dataset {dataset!r}")

        clip_id = _optional_text(record.get("clip_id")) or default_clip_id
        if clip_id in clip_ids:
            raise RawSubsetError(f"duplicated clip_id: {clip_id!r}")
        clip_ids.add(clip_id)
        step_starts = _sequence_of_ints(
            record.get("step_starts"), f"row {row_index}.step_starts"
        )
        step_ends = _sequence_of_ints(
            record.get("step_ends"), f"row {row_index}.step_ends"
        )
        if len(step_starts) != len(step_ends):
            raise RawSubsetError(
                f"row {row_index}.step_starts and step_ends must have equal lengths"
            )
        preliminary.append(
            {
                "dataset": dataset,
                "source": _optional_text(record.get("source")) or dataset,
                "split": split,
                "clip_id": clip_id,
                "raw_episode_id": raw_episode_id,
                "physical_group_key": physical_group_key,
                "task_id": task_id,
                "task_name": task_name,
                "video_path": video_path,
                "metadata_path": metadata_path,
                "part": part,
                "session_name": session_name,
                "primitive": _optional_text(record.get("primitive")),
                "primitive_raw": _optional_text(record.get("primitive_raw")),
                "step_starts": step_starts,
                "step_ends": step_ends,
                "step_actions": _sequence_of_text(
                    record.get("step_actions"), f"row {row_index}.step_actions"
                ),
                "step_skills": _sequence_of_text(
                    record.get("step_skills"), f"row {row_index}.step_skills"
                ),
            }
        )

    try:
        validate_raw_split_records(audit_rows)
    except RawDataContractError as exc:
        raise RawSubsetError(str(exc)) from exc
    return [NormalizedClip(**record) for record in preliminary]


def inspect_mp4(path: Path) -> VideoInfo:
    """Decode sequential frame metadata and preserve native timing in microseconds."""

    try:
        import av
    except ImportError as exc:
        raise RawSubsetError("PyAV is required to inspect raw MP4 clips") from exc
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise RawSubsetError(f"MP4 has no video stream: {path}")
            stream = container.streams.video[0]
            try:
                nominal_fps = float(stream.average_rate) if stream.average_rate else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                nominal_fps = 0.0
            times: list[float | None] = []
            for frame in container.decode(stream):
                if frame.time is not None:
                    times.append(float(frame.time))
                elif frame.pts is not None and stream.time_base is not None:
                    times.append(float(frame.pts * stream.time_base))
                else:
                    times.append(None)
    except RawSubsetError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RawSubsetError(f"cannot decode MP4 metadata {path}: {exc}") from exc
    if len(times) < 2:
        raise RawSubsetError(f"raw clip must contain at least two frames: {path}")

    known = [value for value in times if value is not None and math.isfinite(value)]
    measured_fps = 0.0
    if len(known) == len(times) and known[-1] > known[0]:
        measured_fps = (len(known) - 1) / (known[-1] - known[0])
    fps = measured_fps if 1.0 <= measured_fps <= 240.0 else nominal_fps
    if not 1.0 <= fps <= 240.0:
        raise RawSubsetError(f"cannot derive a plausible FPS from {path}")

    if len(known) != len(times) or any(
        known[index] <= known[index - 1] for index in range(1, len(known))
    ):
        normalized = [index / fps for index in range(len(times))]
    else:
        start = known[0]
        normalized = [value - start for value in known]
    timestamps_us: list[int] = []
    for index, value in enumerate(normalized):
        timestamp = int(round(value * 1_000_000))
        if index:
            timestamp = max(timestamp, timestamps_us[-1] + 1)
        timestamps_us.append(timestamp)
    return VideoInfo(len(times), float(fps), tuple(timestamps_us))


def _bounded_starts(length: int, span: int, limit: int) -> list[int]:
    available = length - span + 1
    if available <= 0:
        return []
    if available <= limit:
        return list(range(available))
    if limit == 1:
        return [0]
    return sorted(
        {round(index * (available - 1) / (limit - 1)) for index in range(limit)}
    )


def _sample_id(kind: str, *values: Any) -> str:
    payload = "\x1f".join([kind, *(str(value) for value in values)])
    return f"{kind}_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _validate_steps(clip: NormalizedClip, frame_count: int) -> None:
    for label, values in (
        ("step_starts", clip.step_starts),
        ("step_ends", clip.step_ends),
    ):
        for value in values:
            if value < 0 or value >= frame_count:
                raise RawSubsetError(
                    f"{clip.clip_id}.{label} contains out-of-range frame {value}"
                )
    for start, end in zip(clip.step_starts, clip.step_ends):
        if end < start:
            raise RawSubsetError(f"{clip.clip_id} has a step ending before it starts")


def _parquet_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RawSubsetError("pyarrow is required to write the raw subset") from exc
    return pa, pq


def _write_parquet(path: Path, rows: list[dict[str, Any]], schema=None) -> None:
    pa, pq = _parquet_modules()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=1)


def _shard_schema():
    pa, _ = _parquet_modules()
    return pa.schema(
        [
            ("dataset", pa.string()),
            ("source", pa.string()),
            ("episode_id", pa.string()),
            ("physical_group_key", pa.string()),
            ("raw_episode_id", pa.string()),
            ("segment_id", pa.string()),
            ("task_id", pa.int32()),
            ("task_name", pa.string()),
            ("num_frames", pa.int32()),
            ("video_mp4", pa.binary()),
            ("proprio_h5_raw", pa.binary()),
            ("timestamp", pa.list_(pa.int64())),
            ("step_starts", pa.list_(pa.int32())),
            ("step_ends", pa.list_(pa.int32())),
            ("step_actions", pa.list_(pa.string())),
            ("step_skills", pa.list_(pa.string())),
            ("source_fps", pa.float64()),
            ("source_video_path", pa.string()),
            ("source_video_sha256", pa.string()),
            ("source_metadata_path", pa.string()),
            ("source_metadata_sha256", pa.string()),
            ("split", pa.string()),
            ("native_part", pa.string()),
            ("session_name", pa.string()),
        ]
    )


def _stage1_rows(
    clip: NormalizedClip,
    video: VideoInfo,
    shard_path: Path,
    row_index: int,
    options: RawSubsetOptions,
) -> list[dict[str, Any]]:
    common = {
        "dataset": clip.dataset,
        "episode_id": clip.physical_group_key,
        "physical_group_key": clip.physical_group_key,
        "segment_id": clip.clip_id,
        "source_episode_id": clip.raw_episode_id,
        "view_id": "v0",
        "video_path": "",
        "shard_path": str(shard_path),
        "source": clip.source,
        "row_index": row_index,
        "fps": video.fps,
        "source_fps": video.fps,
        "primitive": clip.primitive,
        "primitive_raw": clip.primitive_raw,
        "label_confidence": 0.5 if clip.primitive else 0.0,
        "split": clip.split,
        "robosam_mask_training_eligible": False,
        "robosam_interaction_mask_path": "",
        "frame_to_mask_idx_path": "",
        "mask_policy": MASK_POLICY,
        "paper_equivalent_mask": False,
        "valid_same_episode_hard_negative": False,
        "episode_has_multi_primitive": False,
        "is_camera_dominant": False,
        "is_low_motion": False,
        "high_label_confidence": bool(clip.primitive),
        "total_frames": video.num_frames,
    }
    rows: list[dict[str, Any]] = []
    starts = _bounded_starts(
        video.num_frames, options.pair_stride + 1, options.pairs_per_clip
    )
    for frame_i in starts:
        frame_j = frame_i + options.pair_stride
        rows.append(
            {
                **common,
                "sample_id": _sample_id("pair", clip.clip_id, frame_i, frame_j),
                "frame_i": frame_i,
                "frame_j": frame_j,
                "stride": options.pair_stride,
                "pair_type": "real",
                "eligible_for_lgap": bool(clip.primitive),
                "valid_relation_pair": bool(clip.primitive),
                "is_real_pair": True,
                "is_identity_pair": False,
            }
        )
    identity_frame = video.num_frames // 2
    rows.append(
        {
            **common,
            "sample_id": _sample_id(
                "pair", clip.clip_id, identity_frame, identity_frame
            ),
            "frame_i": identity_frame,
            "frame_j": identity_frame,
            "stride": 0,
            "pair_type": "identity",
            "eligible_for_lgap": False,
            "valid_relation_pair": False,
            "is_real_pair": False,
            "is_identity_pair": True,
        }
    )
    return rows


def _stage2_rows(
    clip: NormalizedClip,
    video: VideoInfo,
    shard_path: Path,
    row_index: int,
    options: RawSubsetOptions,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    encoded_video = f"{shard_path}::{row_index}"
    for start in _bounded_starts(
        video.num_frames, options.window_frames, options.windows_per_clip
    ):
        stop = start + options.window_frames
        window_id = _sample_id("window", clip.clip_id, start, stop)
        rows.append(
            {
                "sample_id": window_id,
                "window_id": window_id,
                "dataset": clip.dataset,
                "source": clip.source,
                "episode_id": clip.physical_group_key,
                "physical_group_key": clip.physical_group_key,
                "source_episode_id": clip.raw_episode_id,
                "source_segment_id": clip.clip_id,
                "video_id": clip.clip_id,
                "video_path": encoded_video,
                "shard_path": str(shard_path),
                "row_index": row_index,
                "start_frame": start,
                "stop_frame": stop,
                "clip_nframes": options.window_frames,
                "fps": video.fps,
                "source_fps": video.fps,
                "duration_s": options.window_frames / video.fps,
                "split": clip.split,
            }
        )
    return rows


def _assert_output_isolation(rows_by_split: Mapping[str, list[dict[str, Any]]]) -> None:
    train = {str(row["physical_group_key"]) for row in rows_by_split["train"]}
    evaluation = {str(row["physical_group_key"]) for row in rows_by_split["eval"]}
    overlap = train & evaluation
    if overlap:
        raise RawSubsetError(
            f"physical groups cross train/eval outputs: {sorted(overlap)}"
        )


def build_raw_subset(
    input_index: Path | str,
    output: Path | str,
    *,
    options: RawSubsetOptions | None = None,
) -> dict[str, Any]:
    """Build one embedded shard plus Stage-1/Stage-2 train/eval manifests."""

    options = options or RawSubsetOptions()
    options.validate()
    index_path = Path(input_index).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise RawSubsetError(f"output already exists: {output_root}")
    records = _load_jsonl(index_path)
    if len(records) > options.max_clips:
        raise RawSubsetError(
            f"input has {len(records)} clips, exceeding max_clips={options.max_clips}"
        )
    clips = _normalize_records(records, index_path.parent)
    included = [clip for clip in clips if clip.split in {"train", "eval"}]
    native_test_excluded = sum(
        clip.dataset == EGODEX_DATASET and clip.part == "test" for clip in clips
    )
    excluded_test = sum(clip.split == "test" for clip in clips)
    if not included:
        raise RawSubsetError("no train/eval clips remain after test exclusion")
    if {clip.split for clip in included} != {"train", "eval"}:
        raise RawSubsetError(
            "the bounded subset must contain both train and eval clips"
        )

    total_video_bytes = sum(clip.video_path.stat().st_size for clip in included)
    if total_video_bytes > options.max_total_video_bytes:
        raise RawSubsetError(
            f"selected video bytes {total_video_bytes} exceed "
            f"max_total_video_bytes={options.max_total_video_bytes}"
        )

    prepared: list[tuple[NormalizedClip, VideoInfo, bytes, str, str]] = []
    for clip in included:
        video = inspect_mp4(clip.video_path)
        if video.num_frames < options.window_frames:
            raise RawSubsetError(
                f"{clip.clip_id} has {video.num_frames} frames; "
                f"Stage 2 requires at least {options.window_frames}"
            )
        _validate_steps(clip, video.num_frames)
        video_bytes = clip.video_path.read_bytes()
        video_sha = hashlib.sha256(video_bytes).hexdigest()
        metadata_sha = _sha256(clip.metadata_path) if clip.metadata_path else ""
        prepared.append((clip, video, video_bytes, video_sha, metadata_sha))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.building-", dir=output_root.parent
        )
    )
    final_shard = output_root / "shards" / "raw-subset-00000.parquet"
    try:
        (temporary / "shards").mkdir()
        (temporary / "stage1").mkdir()
        (temporary / "stage2").mkdir()
        shard_rows: list[dict[str, Any]] = []
        stage1: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
        stage2: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
        provenance: list[dict[str, Any]] = []
        for row_index, (clip, video, video_bytes, video_sha, metadata_sha) in enumerate(
            prepared
        ):
            shard_rows.append(
                {
                    "dataset": clip.dataset,
                    "source": clip.source,
                    "episode_id": clip.physical_group_key,
                    "physical_group_key": clip.physical_group_key,
                    "raw_episode_id": clip.raw_episode_id,
                    "segment_id": clip.clip_id,
                    "task_id": clip.task_id,
                    "task_name": clip.task_name,
                    "num_frames": video.num_frames,
                    "video_mp4": video_bytes,
                    "proprio_h5_raw": b"",
                    "timestamp": list(video.timestamps_us),
                    "step_starts": list(clip.step_starts),
                    "step_ends": list(clip.step_ends),
                    "step_actions": list(clip.step_actions),
                    "step_skills": list(clip.step_skills),
                    "source_fps": video.fps,
                    "source_video_path": str(clip.video_path),
                    "source_video_sha256": video_sha,
                    "source_metadata_path": (
                        str(clip.metadata_path) if clip.metadata_path else ""
                    ),
                    "source_metadata_sha256": metadata_sha,
                    "split": clip.split,
                    "native_part": clip.part,
                    "session_name": clip.session_name,
                }
            )
            stage1[clip.split].extend(
                _stage1_rows(clip, video, final_shard, row_index, options)
            )
            stage2[clip.split].extend(
                _stage2_rows(clip, video, final_shard, row_index, options)
            )
            provenance.append(
                {
                    "clip_id": clip.clip_id,
                    "dataset": clip.dataset,
                    "fps": video.fps,
                    "num_frames": video.num_frames,
                    "part": clip.part,
                    "physical_group_key": clip.physical_group_key,
                    "raw_episode_id": clip.raw_episode_id,
                    "session_name": clip.session_name,
                    "source_metadata_path": (
                        str(clip.metadata_path) if clip.metadata_path else ""
                    ),
                    "source_metadata_sha256": metadata_sha,
                    "source_video_path": str(clip.video_path),
                    "source_video_sha256": video_sha,
                    "split": clip.split,
                    "embedded": True,
                    "exclusion_reason": "",
                }
            )
        for clip in clips:
            if clip.split != "test":
                continue
            excluded_metadata_sha = (
                _sha256(clip.metadata_path) if clip.metadata_path else ""
            )
            provenance.append(
                {
                    "clip_id": clip.clip_id,
                    "dataset": clip.dataset,
                    "fps": None,
                    "num_frames": None,
                    "part": clip.part,
                    "physical_group_key": clip.physical_group_key,
                    "raw_episode_id": clip.raw_episode_id,
                    "session_name": clip.session_name,
                    "source_metadata_path": (
                        str(clip.metadata_path) if clip.metadata_path else ""
                    ),
                    "source_metadata_sha256": excluded_metadata_sha,
                    "source_video_path": str(clip.video_path),
                    "source_video_sha256": _sha256(clip.video_path),
                    "split": clip.split,
                    "embedded": False,
                    "exclusion_reason": "test_split",
                }
            )
        _assert_output_isolation(stage1)
        _assert_output_isolation(stage2)
        for split in ("train", "eval"):
            for pair_id, row in enumerate(stage1[split]):
                row["pair_id"] = pair_id

        temporary_shard = temporary / "shards" / final_shard.name
        _write_parquet(temporary_shard, shard_rows, _shard_schema())
        output_files: dict[str, Path] = {"shard": temporary_shard}
        for split in ("train", "eval"):
            pair_path = temporary / "stage1" / f"lam_pair_{split}.parquet"
            window_path = temporary / "stage2" / f"wm_{split}_manifest.parquet"
            _write_parquet(pair_path, stage1[split])
            _write_parquet(window_path, stage2[split])
            output_files[f"stage1_{split}"] = pair_path
            output_files[f"stage2_{split}"] = window_path
        provenance_path = temporary / "provenance.jsonl"
        provenance_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in provenance),
            encoding="utf-8",
        )
        output_files["provenance"] = provenance_path

        report = {
            "schema_version": 1,
            "scope": "representative_raw_stage1_stage2_subset",
            "paper_recipe_complete": False,
            "bridge_or_stage3_built": False,
            "sam3_masks_built": False,
            "mask_policy": MASK_POLICY,
            "mask_warning": (
                "Full-frame fallback is for data/model plumbing only and is not "
                "paper-equivalent Stage-1 supervision."
            ),
            "input_index": str(index_path),
            "input_index_sha256": _sha256(index_path),
            "options": {
                "max_clips": options.max_clips,
                "max_total_video_bytes": options.max_total_video_bytes,
                "pair_stride": options.pair_stride,
                "pairs_per_clip": options.pairs_per_clip,
                "window_frames": options.window_frames,
                "windows_per_clip": options.windows_per_clip,
            },
            "counts": {
                "input_clips": len(clips),
                "embedded_clips": len(shard_rows),
                "excluded_test_clips": excluded_test,
                "excluded_native_egodex_test_clips": native_test_excluded,
                "stage1_train_pairs": len(stage1["train"]),
                "stage1_eval_pairs": len(stage1["eval"]),
                "stage2_train_windows": len(stage2["train"]),
                "stage2_eval_windows": len(stage2["eval"]),
                "total_video_bytes": total_video_bytes,
            },
            "split_groups": {
                split: sorted({row["physical_group_key"] for row in stage1[split]})
                for split in ("train", "eval")
            },
            "outputs": {
                name: {
                    "path": str(path.relative_to(temporary)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in sorted(output_files.items())
            },
        }
        report_path = temporary / "build_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


__all__ = [
    "MASK_POLICY",
    "RawSubsetError",
    "RawSubsetOptions",
    "VideoInfo",
    "build_raw_subset",
    "inspect_mp4",
]
