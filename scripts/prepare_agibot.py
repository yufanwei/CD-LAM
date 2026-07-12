#!/usr/bin/env python3
"""Run the official AgiBot episode-to-bridge preparation chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class PreparationError(RuntimeError):
    """Raised when the preparation chain cannot preserve its contracts."""


@dataclass(frozen=True)
class PreparationPaths:
    """Resolved locations written by one preparation run."""

    root: Path
    segmented: Path
    lerobot: Path
    stage12_index: Path
    stage12: Path
    bridge: Path


def _paths(output_root: Path) -> PreparationPaths:
    root = output_root.expanduser().resolve()
    if root == Path("/"):
        raise PreparationError("output root must not be the filesystem root")
    return PreparationPaths(
        root=root,
        segmented=root / "segmented",
        lerobot=root / "lerobot",
        stage12_index=root / "stage12_index.jsonl",
        stage12=root / "stage12",
        bridge=root / "bridge" / "alpha_bridge_cache.npz",
    )


def build_commands(
    args: argparse.Namespace,
    repo_root: Path,
    python: Path,
) -> tuple[PreparationPaths, list[list[str]]]:
    """Build the exact fail-closed subprocess commands in execution order."""

    repo_root = repo_root.resolve()
    paths = _paths(args.output_root)
    raw_root = args.raw_root.expanduser().resolve()
    materialize = [
        str(python),
        str(repo_root / "scripts" / "materialize_agibot_alpha.py"),
        "--raw-root",
        str(raw_root),
        "--output",
        str(paths.segmented),
        "--log-every",
        str(args.log_every),
    ]
    for task_id in args.task_id or ():
        materialize.extend(("--task-id", str(task_id)))
    for task_id, episode_id in args.episode or ():
        materialize.extend(("--episode", f"{task_id}-{episode_id}"))
    if args.max_episodes is not None:
        materialize.extend(("--max-episodes", str(args.max_episodes)))
    if args.source_revision:
        materialize.extend(("--source-revision", args.source_revision))
    source_record = args.source_record
    if source_record is None:
        candidate = raw_root.parent / "agibot_download.json"
        if candidate.is_file():
            source_record = candidate
    if source_record is not None:
        materialize.extend(
            ("--source-record", str(source_record.expanduser().resolve()))
        )
    if args.overwrite:
        materialize.append("--overwrite")

    convert = [
        str(python),
        str(repo_root / "internal" / "runtime" / "convert_agibot_alpha.py"),
        "--raw-root",
        str(paths.segmented),
        "--output",
        str(paths.lerobot),
        "--splits",
        "train",
        "--held-out-tasks",
        "0",
        "--eval-percent",
        "0",
        "--min-frames",
        str(args.min_frames),
        "--log-every",
        str(args.log_every),
    ]
    if args.overwrite:
        convert.append("--overwrite")

    index_stage12 = [
        str(python),
        str(repo_root / "scripts" / "index_agibot_segments.py"),
        "--materialized-root",
        str(paths.segmented),
        "--output",
        str(paths.stage12_index),
        "--eval-fraction",
        str(args.stage12_eval_fraction),
        "--seed",
        str(args.seed),
        "--max-clips",
        str(args.stage12_max_clips),
    ]
    if args.primitive_map:
        index_stage12.extend(
            ("--primitive-map", str(args.primitive_map.expanduser().resolve()))
        )

    build_stage12 = [
        str(python),
        str(repo_root / "internal" / "runtime" / "build_raw_subset.py"),
        "--input",
        str(paths.stage12_index),
        "--output",
        str(paths.stage12),
        "--max-clips",
        str(args.stage12_max_clips),
        "--max-total-video-bytes",
        str(args.stage12_max_video_bytes),
    ]

    bridge = [
        str(python),
        str(
            repo_root
            / "internal"
            / "vendor"
            / "scale_support"
            / "Scale"
            / "common"
            / "build_alpha_bridge_cache.py"
        ),
        "--dataset-yaml",
        str(paths.lerobot / "_dataset_paths_train.yaml"),
        "--out",
        str(paths.bridge),
        "--n-episodes",
        "0",
        "--pairs-per-episode",
        str(args.pairs_per_episode),
        "--stride",
        str(args.stride),
        "--eval-frac",
        str(args.bridge_eval_fraction),
        "--min-frames",
        str(args.min_frames),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
    ]
    return paths, [materialize, convert, index_stage12, build_stage12, bridge]


def _bridge_split(physical_episode_id: str, eval_fraction: float) -> str:
    digest = int(
        hashlib.md5(f"alpha_bridge|{physical_episode_id}".encode()).hexdigest(),
        16,
    )
    return "eval" if digest % 1000 < int(eval_fraction * 1000) else "train"


def bridge_split_preflight(
    lerobot_root: Path,
    min_frames: int,
    eval_fraction: float,
) -> dict[str, int]:
    """Predict the bridge split after segment-length filtering."""

    episodes_path = lerobot_root / "train" / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise PreparationError(
            f"converted episode metadata is missing: {episodes_path}"
        )
    physical_ids: set[str] = set()
    eligible_segments = 0
    with episodes_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparationError(
                    f"invalid converted episode metadata at line {line_number}: {exc}"
                ) from exc
            frames = row.get("length")
            physical_id = row.get("physical_episode_id")
            if isinstance(frames, bool) or not isinstance(frames, int):
                raise PreparationError(
                    f"converted episode line {line_number} has invalid length"
                )
            if not isinstance(physical_id, str) or not physical_id.strip():
                raise PreparationError(
                    f"converted episode line {line_number} has no physical_episode_id"
                )
            if frames >= min_frames:
                eligible_segments += 1
                physical_ids.add(physical_id.strip())
    counts = {"train": 0, "eval": 0}
    for physical_id in physical_ids:
        counts[_bridge_split(physical_id, eval_fraction)] += 1
    result = {
        "eligible_segments": eligible_segments,
        "physical_episodes": len(physical_ids),
        "train_physical_episodes": counts["train"],
        "eval_physical_episodes": counts["eval"],
    }
    if eligible_segments == 0 or not counts["train"] or not counts["eval"]:
        raise PreparationError(
            "bridge preparation needs non-empty train and eval sets after "
            "physical-episode-granular hashing; got "
            f"eligible_segments={eligible_segments}, "
            f"physical_episodes={len(physical_ids)}, train={counts['train']}, "
            f"eval={counts['eval']}. Increase --max-episodes or select more "
            "complete physical episodes; segments from one episode are never split."
        )
    return result


def _display_commands(commands: Sequence[Sequence[str]], vendor_root: Path) -> None:
    for index, command in enumerate(commands, 1):
        prefix = ""
        if Path(command[1]).name == "build_alpha_bridge_cache.py":
            prefix = f"PYTHONPATH={shlex.quote(str(vendor_root))} "
        print(f"[{index}/{len(commands)}] {prefix}{shlex.join(command)}")


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.is_symlink():
        raise PreparationError(f"refusing symlinked output root: {path}")
    if path.exists():
        if not overwrite:
            raise PreparationError(
                f"output root already exists: {path}; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _require_disjoint_roots(raw_root: Path, output_root: Path) -> None:
    """Reject either containment direction before any output mutation."""

    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if (
        raw_root == output_root
        or raw_root.is_relative_to(output_root)
        or output_root.is_relative_to(raw_root)
    ):
        raise PreparationError(
            "raw and output roots must be disjoint and must not equal or contain "
            "one another: "
            f"raw={raw_root}, output={output_root}"
        )


def _validate_disjoint(raw_root: Path, output_root: Path) -> None:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if (
        raw_root == output_root
        or raw_root in output_root.parents
        or output_root in raw_root.parents
    ):
        raise PreparationError(
            "raw and output roots must not equal or contain one another: "
            f"{raw_root}, {output_root}"
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    repo_root = args.repo_root.expanduser().resolve()
    python = Path(sys.executable).resolve()
    paths, commands = build_commands(args, repo_root, python)
    vendor_root = repo_root / "internal" / "vendor" / "scale_support"
    _display_commands(commands, vendor_root)
    if args.dry_run:
        return {
            "dry_run": True,
            "output_root": str(paths.root),
            "commands": commands,
        }
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise PreparationError(f"raw root is missing: {args.raw_root}")
    _require_disjoint_roots(raw_root, paths.root)
    if not vendor_root.is_dir():
        raise PreparationError(f"vendored bridge support is missing: {vendor_root}")
    _validate_disjoint(args.raw_root, paths.root)
    _prepare_output(paths.root, args.overwrite)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(vendor_root)
        if not existing_pythonpath
        else f"{vendor_root}{os.pathsep}{existing_pythonpath}"
    )
    try:
        subprocess.run(commands[0], check=True, cwd=repo_root)
        subprocess.run(commands[1], check=True, cwd=repo_root)
        subprocess.run(commands[2], check=True, cwd=repo_root)
        subprocess.run(commands[3], check=True, cwd=repo_root)
        split_summary = bridge_split_preflight(
            paths.lerobot,
            args.min_frames,
            args.bridge_eval_fraction,
        )
        paths.bridge.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            commands[4],
            check=True,
            cwd=repo_root,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        raise PreparationError(
            f"AgiBot preparation subprocess failed with exit code {exc.returncode}: "
            f"{shlex.join(exc.cmd)}"
        ) from exc
    summary: dict[str, object] = {
        "bridge_cache": str(paths.bridge),
        "bridge_split": split_summary,
        "dataset_yaml": str(paths.lerobot / "_dataset_paths_train.yaml"),
        "lerobot_root": str(paths.lerobot),
        "raw_root": str(args.raw_root.expanduser().resolve()),
        "segmented_root": str(paths.segmented),
        "stage12_index": str(paths.stage12_index),
        "stage1_train_index": str(paths.stage12 / "stage1" / "lam_pair_train.parquet"),
        "stage1_eval_index": str(paths.stage12 / "stage1" / "lam_pair_eval.parquet"),
        "stage2_train_manifest": str(
            paths.stage12 / "stage2" / "wm_train_manifest.parquet"
        ),
        "stage2_eval_manifest": str(
            paths.stage12 / "stage2" / "wm_eval_manifest.parquet"
        ),
        "split_policy": (
            "all converted segments enter the bridge source; the bridge cache "
            "then hashes complete physical episodes into train and eval"
        ),
    }
    (paths.root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _episode_selector(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--episode must use <task-id>-<episode-id>")
    try:
        values = tuple(map(int, parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--episode IDs must be integers") from exc
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("--episode IDs must be non-negative")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument("--episode", type=_episode_selector, action="append")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--source-revision")
    parser.add_argument("--source-record", type=Path)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--pairs-per-episode", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--bridge-eval-fraction", type=float, default=0.12)
    parser.add_argument("--stage12-eval-fraction", type=float, default=0.10)
    parser.add_argument("--stage12-max-clips", type=int, default=32)
    parser.add_argument("--stage12-max-video-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--primitive-map", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    for name in ("min_frames", "pairs_per_episode", "stride", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.task_id and any(task_id < 0 for task_id in args.task_id):
        parser.error("--task-id must be non-negative")
    if not 0 < args.bridge_eval_fraction < 1:
        parser.error("--bridge-eval-fraction must be between zero and one")
    if not 0 < args.stage12_eval_fraction < 1:
        parser.error("--stage12-eval-fraction must be between zero and one")
    if args.stage12_max_clips < 2:
        parser.error("--stage12-max-clips must be at least two")
    if args.stage12_max_clips > 256:
        parser.error("--stage12-max-clips cannot exceed 256")
    if args.stage12_max_video_bytes < 1:
        parser.error("--stage12-max-video-bytes must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (PreparationError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
