#!/usr/bin/env python3
"""Index an extracted official EgoDex part for the CD-LAM raw subset builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_PARTS = ("part1", "part2", "part3", "part4", "part5", "extra", "test")
SPLIT_POLICY = "sha256_ranked_session_v1"


class EgoDexIndexError(ValueError):
    """Raised when an extracted EgoDex tree cannot be indexed safely."""


@dataclass(frozen=True)
class ClipPair:
    """One unambiguous official EgoDex HDF5/video pair."""

    task: str
    index: int
    metadata_path: Path
    video_path: Path


def _video_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".h264.mp4"):
        return name[: -len(".h264.mp4")]
    if name.endswith(".mp4"):
        return name[: -len(".mp4")]
    raise EgoDexIndexError(f"unsupported EgoDex video filename: {path}")


def discover_pairs(root: Path) -> list[ClipPair]:
    """Discover all pairs below ``root`` and reject incomplete or ambiguous data."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise EgoDexIndexError(f"EgoDex root is not a directory: {root}")

    metadata_by_key: dict[tuple[Path, str], list[Path]] = {}
    video_by_key: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(root.rglob("*.hdf5")):
        key = (path.parent.resolve(), path.name[: -len(".hdf5")])
        metadata_by_key.setdefault(key, []).append(path.resolve())
    for path in sorted(root.rglob("*.mp4")):
        key = (path.parent.resolve(), _video_stem(path))
        video_by_key.setdefault(key, []).append(path.resolve())

    if not metadata_by_key and not video_by_key:
        raise EgoDexIndexError(f"no .hdf5 metadata files found below {root}")

    pairs: list[ClipPair] = []
    for key in sorted(
        set(metadata_by_key) | set(video_by_key), key=lambda item: str(item)
    ):
        metadata = metadata_by_key.get(key, [])
        videos = video_by_key.get(key, [])
        label = key[0] / key[1]
        if len(metadata) != 1:
            if not metadata:
                raise EgoDexIndexError(f"video has no matching .hdf5 metadata: {label}")
            raise EgoDexIndexError(
                f"ambiguous EgoDex metadata pair for {label}: "
                + ", ".join(map(str, metadata))
            )
        if len(videos) != 1:
            if not videos:
                raise EgoDexIndexError(f"metadata has no matching MP4 video: {label}")
            raise EgoDexIndexError(
                f"ambiguous EgoDex video pair for {label}: "
                + ", ".join(map(str, videos))
            )

        task = metadata[0].parent.name.strip()
        if not task:
            raise EgoDexIndexError(f"cannot derive task name from {metadata[0]}")
        raw_index = key[1]
        if not raw_index.isdigit():
            raise EgoDexIndexError(
                f"EgoDex clip basename must be a non-negative integer: {metadata[0]}"
            )
        pairs.append(
            ClipPair(
                task=task,
                index=int(raw_index),
                metadata_path=metadata[0],
                video_path=videos[0],
            )
        )

    identities: dict[tuple[str, int], Path] = {}
    for pair in pairs:
        identity = (pair.task, pair.index)
        previous = identities.get(identity)
        if previous is not None:
            raise EgoDexIndexError(
                "duplicate EgoDex task/index identity below the scan root: "
                f"{identity!r} appears in {previous} and {pair.metadata_path}"
            )
        identities[identity] = pair.metadata_path
    return sorted(pairs, key=lambda pair: (pair.task, pair.index))


def read_session_name(path: Path) -> str:
    """Read the physical recording key from an EgoDex HDF5 root attribute."""

    try:
        import h5py
    except ImportError as exc:
        raise EgoDexIndexError(
            "h5py is required to read EgoDex session_name metadata"
        ) from exc
    try:
        with h5py.File(path, "r") as handle:
            value: Any = handle.attrs.get("session_name")
    except Exception as exc:  # noqa: BLE001
        raise EgoDexIndexError(
            f"cannot read EgoDex HDF5 metadata {path}: {exc}"
        ) from exc
    if value is None:
        raise EgoDexIndexError(
            f"EgoDex HDF5 has no root session_name attribute: {path}"
        )
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError as exc:
            raise EgoDexIndexError(
                f"EgoDex session_name must be a scalar string: {path}"
            ) from exc
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EgoDexIndexError(
                f"EgoDex session_name is not valid UTF-8: {path}"
            ) from exc
    if not isinstance(value, str):
        raise EgoDexIndexError(f"EgoDex session_name must be a scalar string: {path}")
    session_name = value.strip()
    if not session_name:
        raise EgoDexIndexError(f"EgoDex session_name is empty: {path}")
    return session_name


def load_primitive_map(path: Path | None) -> dict[str, str]:
    """Load an exact task-name mapping without inferring unknown primitives."""

    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EgoDexIndexError(f"primitive map does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EgoDexIndexError(f"cannot read primitive map {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EgoDexIndexError(
            "primitive map must be a JSON object of task to primitive"
        )
    result: dict[str, str] = {}
    for raw_task, raw_primitive in value.items():
        if not isinstance(raw_task, str) or not raw_task.strip():
            raise EgoDexIndexError("primitive-map task names must be non-empty strings")
        if not isinstance(raw_primitive, str) or not raw_primitive.strip():
            raise EgoDexIndexError(
                f"primitive for task {raw_task!r} must be a non-empty string"
            )
        task = raw_task.strip()
        if task in result:
            raise EgoDexIndexError(
                f"duplicate normalized task in primitive map: {task!r}"
            )
        result[task] = raw_primitive.strip()
    return result


def session_splits(
    sessions: Sequence[str], *, part: str, eval_fraction: float, seed: int
) -> dict[str, str]:
    """Assign deterministic physical-session splits with nonempty train/eval pools."""

    if part not in SUPPORTED_PARTS:
        raise EgoDexIndexError(f"unsupported EgoDex part: {part!r}")
    if not math.isfinite(eval_fraction) or not 0.0 < eval_fraction < 1.0:
        raise EgoDexIndexError(
            "eval fraction must be finite and strictly between 0 and 1"
        )
    unique = sorted({str(session).strip() for session in sessions})
    if not unique or any(not session for session in unique):
        raise EgoDexIndexError("at least one non-empty EgoDex session is required")
    if part == "test":
        return {session: "test" for session in unique}
    if len(unique) == 1:
        raise EgoDexIndexError(
            "non-test EgoDex indexing requires at least two unique session_name "
            "values so both train and eval are nonempty"
        )

    def rank(session: str) -> str:
        payload = f"{SPLIT_POLICY}\0{seed}\0{part}\0{session}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ranked = sorted(unique, key=lambda session: (rank(session), session))
    requested = int(math.floor(len(ranked) * eval_fraction + 0.5))
    eval_count = min(len(ranked) - 1, max(1, requested))
    eval_sessions = set(ranked[:eval_count])
    return {
        session: "eval" if session in eval_sessions else "train" for session in unique
    }


def _relative_path(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.resolve().parent)


def build_rows(
    pairs: Sequence[ClipPair],
    *,
    part: str,
    output: Path,
    eval_fraction: float,
    seed: int,
    primitive_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Read provenance, assign splits, and return builder-compatible rows."""

    session_by_metadata = {
        pair.metadata_path: read_session_name(pair.metadata_path) for pair in pairs
    }
    split_by_session = session_splits(
        list(session_by_metadata.values()),
        part=part,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        session_name = session_by_metadata[pair.metadata_path]
        row: dict[str, Any] = {
            "dataset": "egodex",
            "episode_id": f"egodex_{part}_{pair.task}_{pair.index}",
            "metadata_h5": _relative_path(pair.metadata_path, output),
            "part": part,
            "session_name": session_name,
            "split": split_by_session[session_name],
            "split_eval_fraction": eval_fraction,
            "split_policy": SPLIT_POLICY,
            "split_seed": seed,
            "task_name": pair.task,
            "video_path": _relative_path(pair.video_path, output),
        }
        primitive = primitive_map.get(pair.task)
        if primitive is not None:
            row["primitive"] = primitive
            row["primitive_raw"] = pair.task
        rows.append(row)
    return rows


def select_bounded_rows(
    rows: Sequence[Mapping[str, Any]], *, max_clips: int | None, seed: int
) -> list[dict[str, Any]]:
    """Select deterministic complete session groups without changing splits."""

    materialized = [dict(row) for row in rows]
    if max_clips is None or len(materialized) <= max_clips:
        return materialized
    if max_clips <= 0:
        raise EgoDexIndexError("max_clips must be positive")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in materialized:
        key = (str(row.get("part", "")), str(row.get("session_name", "")))
        grouped.setdefault(key, []).append(row)
    group_splits: dict[tuple[str, str], str] = {}
    for key, group_rows in grouped.items():
        splits = {str(row.get("split", "")) for row in group_rows}
        if len(splits) != 1:
            raise EgoDexIndexError(f"physical session crosses splits: {key!r}")
        group_splits[key] = splits.pop()

    def group_rank(key: tuple[str, str]) -> tuple[str, str, str]:
        part, session = key
        payload = f"egodex_bounded_index_v1\0{seed}\0{part}\0{session}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), part, session

    by_split: dict[str, list[tuple[str, str]]] = {}
    for key, split in group_splits.items():
        by_split.setdefault(split, []).append(key)
    required_splits = [split for split in ("train", "eval") if split in by_split]
    if required_splits and len(required_splits) != 2:
        raise EgoDexIndexError(
            "a bounded non-test index must contain both train and eval clips"
        )
    if len(required_splits) == 2 and max_clips < 2:
        raise EgoDexIndexError(
            "max_clips must be at least 2 to retain train and eval clips"
        )

    selected_groups: set[tuple[str, str]] = set()
    selected_count = 0
    for split in required_splits:
        key = min(
            by_split[split], key=lambda item: (len(grouped[item]), group_rank(item))
        )
        selected_groups.add(key)
        selected_count += len(grouped[key])
    if selected_count > max_clips:
        sizes = {
            split: min(len(grouped[key]) for key in by_split[split])
            for split in required_splits
        }
        raise EgoDexIndexError(
            "max_clips is too small to retain complete train and eval sessions: "
            f"minimum required is {sum(sizes.values())}"
        )

    remaining_groups = sorted(
        (key for key in grouped if key not in selected_groups), key=group_rank
    )
    for key in remaining_groups:
        group_size = len(grouped[key])
        if selected_count + group_size <= max_clips:
            selected_groups.add(key)
            selected_count += group_size
    selected = [row for key in selected_groups for row in grouped[key]]
    return sorted(selected, key=lambda row: str(row.get("episode_id", "")))


def write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool
) -> None:
    """Write a complete index atomically without replacing files by default."""

    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise EgoDexIndexError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Extracted root for one official part"
    )
    parser.add_argument("--part", choices=SUPPORTED_PARTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max-clips",
        type=int,
        help=(
            "Deterministically cap the output while retaining both train and eval; "
            "split labels remain assigned by physical session"
        ),
    )
    parser.add_argument(
        "--primitive-map",
        type=Path,
        help="Optional JSON object mapping exact task names to primitive labels",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pairs = discover_pairs(args.root)
        primitive_map = load_primitive_map(args.primitive_map)
        discovered_rows = build_rows(
            pairs,
            part=args.part,
            output=args.output,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            primitive_map=primitive_map,
        )
        rows = select_bounded_rows(
            discovered_rows, max_clips=args.max_clips, seed=args.seed
        )
        write_jsonl(args.output, rows, overwrite=args.overwrite)
    except (EgoDexIndexError, OSError) as exc:
        print(f"index_egodex: {exc}", file=sys.stderr)
        return 2

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "eval", "test")
    }
    summary = {
        "clips": len(rows),
        "clips_discovered": len(discovered_rows),
        "output": str(args.output.expanduser().resolve()),
        "part": args.part,
        "sessions": len({row["session_name"] for row in rows}),
        "split_eval_fraction": args.eval_fraction,
        "split_policy": SPLIT_POLICY,
        "split_seed": args.seed,
        "split_counts": split_counts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
