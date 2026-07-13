#!/usr/bin/env python3
"""Require nearest-neighbor timestamp sampling in the external ACWM source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

SOURCE_RELATIVE_PATH = Path("groot_dreams/utils/video.py")
LEGACY_FRAME_COUNT_CUTOFF = "if len(loaded_frames) >= len(timestamps):"
EXPECTED_BACKEND_AST_SHA256 = (
    "a12072beddc0a39cb16544a9e6cfdc766e30546451ba7f427cfb1a12ced9dbdf"
)
REQUIRED_EXPRESSIONS = {
    "empty decode guard": "if not loaded_frames:",
    "loaded timestamp array": ("loaded_ts = np.asarray(loaded_ts, dtype=np.float64)"),
    "requested timestamp array": (
        "requested_ts = np.asarray(timestamps, dtype=np.float64)"
    ),
    "nearest-neighbor index selection": (
        "indices = np.abs(loaded_ts[:, np.newaxis] - requested_ts[np.newaxis, :]).argmin(axis=0)"
    ),
    "requested frame selection": (
        "frames = np.array([loaded_frames[int(i)] for i in indices])"
    ),
}


def _finite_timestamps(values: Iterable[float], *, label: str) -> tuple[float, ...]:
    timestamps = tuple(float(value) for value in values)
    if not timestamps:
        raise ValueError(f"{label} must not be empty")
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError(f"{label} must contain only finite values")
    return timestamps


def nearest_timestamp_indices(
    loaded_timestamps: Iterable[float],
    requested_timestamps: Iterable[float],
) -> tuple[int, ...]:
    """Select the first closest loaded timestamp for every requested timestamp.

    This pure reference implementation mirrors NumPy ``argmin`` tie-breaking
    and is intentionally independent of video decoders and optional packages.
    """

    loaded = _finite_timestamps(loaded_timestamps, label="loaded_timestamps")
    requested = _finite_timestamps(requested_timestamps, label="requested_timestamps")
    if any(right < left for left, right in zip(loaded, loaded[1:], strict=False)):
        raise ValueError("loaded_timestamps must be nondecreasing")
    return tuple(
        min(range(len(loaded)), key=lambda index: abs(loaded[index] - target))
        for target in requested
    )


def _torchvision_av_branch(text: str) -> tuple[ast.stmt, ...]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError(
            f"ACWM evaluation source is not valid Python: {exc}"
        ) from exc
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_frames_by_timestamps"
    ]
    if len(functions) != 1:
        raise RuntimeError(
            "ACWM evaluation source must define exactly one get_frames_by_timestamps function"
        )
    expected_test = ast.parse('video_backend == "torchvision_av"', mode="eval").body
    branches = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.If)
        and ast.dump(node.test, include_attributes=False)
        == ast.dump(expected_test, include_attributes=False)
    ]
    if len(branches) != 1:
        raise RuntimeError(
            "ACWM evaluation source must contain exactly one torchvision_av branch"
        )
    return tuple(branches[0].body)


def _branch_fingerprint(branch: tuple[ast.stmt, ...]) -> str:
    canonical = "|".join(
        ast.dump(statement, include_attributes=False) for statement in branch
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_source(root: Path) -> Path:
    """Validate the audited torchvision/AV timestamp sampling implementation."""

    source = root / SOURCE_RELATIVE_PATH
    if not source.is_file():
        raise FileNotFoundError(f"external ACWM video source is missing: {source}")
    text = source.read_text(encoding="utf-8")
    branch = _torchvision_av_branch(text)
    if LEGACY_FRAME_COUNT_CUTOFF in text:
        raise RuntimeError(
            "ACWM evaluation source is unsafe: the torchvision_av decoder stops "
            "after the number of requested frames, so strided requests collapse "
            "to consecutive frames. Apply "
            "third_party/patches/acwm_timestamp_nearest_neighbor.patch."
        )
    missing = [
        label
        for label, expression in REQUIRED_EXPRESSIONS.items()
        if expression not in text
    ]
    if missing:
        raise RuntimeError(
            "ACWM evaluation source is not the audited nearest-neighbor "
            f"implementation; missing: {', '.join(missing)}. Apply "
            "third_party/patches/acwm_timestamp_nearest_neighbor.patch and "
            "re-run this validator."
        )
    observed_fingerprint = _branch_fingerprint(branch)
    if observed_fingerprint != EXPECTED_BACKEND_AST_SHA256:
        raise RuntimeError(
            "ACWM evaluation source differs from the audited patched "
            "torchvision_av implementation. Refusing unknown timestamp "
            f"semantics (AST SHA-256 {observed_fingerprint})."
        )
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    args = parser.parse_args()
    source = validate_source(args.external_root.expanduser().resolve())
    print(f"ACWM timestamp nearest-neighbor source: PASS ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
