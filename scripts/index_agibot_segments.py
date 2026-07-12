#!/usr/bin/env python3
"""Build a split-safe raw-subset index from materialized AgiBot segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "internal" / "vendor" / "scale_support"
SPLIT_POLICY = "cdlam_agibot_physical_episode_v1"
CANONICAL_PRIMITIVES = {
    "pick_place",
    "insert_remove",
    "stack_unstack",
    "scoop_dump",
    "open",
    "close",
    "turn_on",
    "turn_off",
    "wash_rinse",
    "cut",
    "stir",
    "pour",
}


class AgiBotIndexError(ValueError):
    """Raised when materialized segments cannot form a safe bounded index."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgiBotIndexError(f"cannot read {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AgiBotIndexError(f"materialization provenance is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgiBotIndexError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AgiBotIndexError(f"row {line_number} in {path} must be an object")
        rows.append(row)
    if not rows:
        raise AgiBotIndexError(f"materialization provenance is empty: {path}")
    return rows


def load_primitive_map(path: Path | None) -> dict[str, str]:
    """Load an explicit exact skill-to-canonical-primitive mapping."""

    if path is None:
        return {}
    value = _load_json(path.expanduser().resolve())
    if not isinstance(value, dict):
        raise AgiBotIndexError("primitive map must be a JSON object")
    result: dict[str, str] = {}
    for raw_skill, raw_primitive in value.items():
        if not isinstance(raw_skill, str) or not raw_skill.strip():
            raise AgiBotIndexError("primitive-map skills must be non-empty strings")
        if (
            not isinstance(raw_primitive, str)
            or raw_primitive not in CANONICAL_PRIMITIVES
        ):
            raise AgiBotIndexError(
                f"primitive for {raw_skill!r} must be one of {sorted(CANONICAL_PRIMITIVES)}"
            )
        result[raw_skill.strip()] = raw_primitive
    return result


def assign_splits(
    physical_episode_ids: Sequence[str], *, eval_fraction: float, seed: int
) -> dict[str, str]:
    """Assign complete physical episodes with guaranteed nonempty train/eval sets."""

    if not math.isfinite(eval_fraction) or not 0.0 < eval_fraction < 1.0:
        raise AgiBotIndexError("eval fraction must be finite and between zero and one")
    unique = sorted(set(physical_episode_ids))
    if len(unique) < 2:
        raise AgiBotIndexError(
            "at least two physical AgiBot episodes are required for train/eval isolation"
        )

    def rank(identity: str) -> tuple[str, str]:
        payload = f"{SPLIT_POLICY}\0{seed}\0{identity}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), identity

    ranked = sorted(unique, key=rank)
    requested = int(math.floor(len(ranked) * eval_fraction + 0.5))
    eval_count = min(len(ranked) - 1, max(1, requested))
    evaluation = set(ranked[:eval_count])
    return {
        identity: "eval" if identity in evaluation else "train" for identity in unique
    }


def _bounded_groups(
    rows: Sequence[Mapping[str, Any]], *, max_clips: int, seed: int
) -> list[dict[str, Any]]:
    if max_clips < 2:
        raise AgiBotIndexError("max clips must be at least two")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["physical_episode_id"])].append(dict(row))

    def rank(identity: str) -> tuple[str, str]:
        payload = f"cdlam_agibot_bound_v1\0{seed}\0{identity}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), identity

    selected: set[str] = set()
    count = 0
    for split in ("train", "eval"):
        candidates = [
            identity
            for identity, group in grouped.items()
            if {str(row["split"]) for row in group} == {split}
        ]
        if not candidates:
            raise AgiBotIndexError(f"no physical episode was assigned to {split}")
        identity = min(candidates, key=lambda item: (len(grouped[item]), rank(item)))
        selected.add(identity)
        count += len(grouped[identity])
    if count > max_clips:
        raise AgiBotIndexError(
            "max clips is too small to retain complete train and eval physical episodes; "
            f"minimum selected size is {count}"
        )
    for identity in sorted(
        (item for item in grouped if item not in selected), key=rank
    ):
        if count + len(grouped[identity]) <= max_clips:
            selected.add(identity)
            count += len(grouped[identity])
    return sorted(
        (row for identity in selected for row in grouped[identity]),
        key=lambda row: str(row["source_id"]),
    )


def build_rows(
    materialized_root: Path,
    output: Path,
    *,
    eval_fraction: float,
    seed: int,
    max_clips: int,
    primitive_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Validate materialized bytes and return raw-subset-builder rows."""

    root = materialized_root.expanduser().resolve()
    summary = _load_json(root / "materialization.json")
    if not isinstance(summary, dict) or summary.get("dataset_id") != (
        "agibot-world/AgiBotWorld-Alpha"
    ):
        raise AgiBotIndexError("materialization.json is not an AgiBot Alpha record")
    provenance = _load_jsonl(root / "provenance.jsonl")
    physical_ids = [str(row.get("physical_episode_id", "")) for row in provenance]
    if any(not identity for identity in physical_ids):
        raise AgiBotIndexError("materialization row has no physical_episode_id")
    splits = assign_splits(physical_ids, eval_fraction=eval_fraction, seed=seed)

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(provenance):
        source_id = str(source.get("source_id", "")).strip()
        physical_id = str(source.get("physical_episode_id", "")).strip()
        if not source_id:
            raise AgiBotIndexError(f"materialization row {index} has no source_id")
        parts = source_id.split("-")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise AgiBotIndexError(f"invalid AgiBot source_id: {source_id!r}")
        expected_physical = f"agibot_alpha:{int(parts[0])}:{int(parts[1])}"
        if physical_id != expected_physical:
            raise AgiBotIndexError(
                f"source {source_id} physical identity is {physical_id!r}, "
                f"expected {expected_physical!r}"
            )
        video = root / "train" / source_id / "head_color.mp4"
        if not video.is_file():
            raise AgiBotIndexError(f"materialized video is missing: {video}")
        expected_hash = source.get("video_sha256")
        if not isinstance(expected_hash, str) or _sha256(video) != expected_hash:
            raise AgiBotIndexError(f"materialized video hash mismatch: {video}")
        frames = source.get("video_frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
            raise AgiBotIndexError(f"source {source_id} has invalid video_frames")
        skill = str(source.get("skill", "")).strip()
        action_text = str(source.get("action_text", "")).strip()
        rows.append(
            {
                "dataset": "agibot_alpha",
                "source": "official_agibot_alpha_materialization",
                "source_id": source_id,
                "physical_episode_id": physical_id,
                "task_id": int(parts[0]),
                "task_name": action_text or f"AgiBot task {parts[0]}",
                "video_path": os.path.relpath(video, output.resolve().parent),
                "split": splits[physical_id],
                "split_policy": SPLIT_POLICY,
                "split_seed": seed,
                "primitive": primitive_map.get(skill, ""),
                "primitive_raw": skill,
                "step_starts": [0],
                "step_ends": [frames - 1],
                "step_actions": [action_text] if action_text else [],
                "step_skills": [skill] if skill else [],
            }
        )
    return _bounded_groups(rows, max_clips=max_clips, seed=seed)


def write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool
) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise AgiBotIndexError(f"refusing to overwrite existing output: {path}")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clips", type=int, default=32)
    parser.add_argument("--primitive-map", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_clips < 2:
        parser.error("--max-clips must be at least two")
    if args.max_clips > 256:
        parser.error("--max-clips cannot exceed 256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        primitive_map = load_primitive_map(args.primitive_map)
        rows = build_rows(
            args.materialized_root,
            args.output,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            max_clips=args.max_clips,
            primitive_map=primitive_map,
        )
        write_jsonl(args.output, rows, overwrite=args.overwrite)
    except (AgiBotIndexError, OSError) as exc:
        print(f"index_agibot_segments: {exc}", file=os.sys.stderr)
        return 2
    counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "eval")
    }
    print(
        json.dumps(
            {
                "clips": len(rows),
                "output": str(args.output.expanduser().resolve()),
                "physical_episodes": len({row["physical_episode_id"] for row in rows}),
                "split_counts": counts,
                "split_policy": SPLIT_POLICY,
                "split_seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
