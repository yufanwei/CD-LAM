"""Portable scoring for protocol-compatible FDCE track bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
import statistics
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import FDCEResult, foreground_displacement_chamfer_error


PROTOCOL_ID = "cdlam-fdce-displacement-v1"
_REQUIRED_KEYS = {"generated_tracks", "reference_tracks"}
_OPTIONAL_KEYS = {"generated_visibility", "reference_visibility"}
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024


class EvaluationError(ValueError):
    """Raised when an evaluation input violates the public protocol."""


def _load_bundle(
    path: Path,
    *,
    expected_frames: int | None,
    max_tracks: int,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise EvaluationError(f"track bundle must be an .npz file: {path}")
    size = path.stat().st_size
    if size > _MAX_ARCHIVE_BYTES:
        raise EvaluationError(
            f"track bundle exceeds {_MAX_ARCHIVE_BYTES} bytes: {path.name}"
        )
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            members = zipped.infolist()
            if len(members) > len(_REQUIRED_KEYS | _OPTIONAL_KEYS):
                raise EvaluationError(f"track bundle has too many members: {path.name}")
            if any(
                member.is_dir()
                or "/" in member.filename
                or "\\" in member.filename
                or not member.filename.endswith(".npy")
                for member in members
            ):
                raise EvaluationError(f"track bundle has unsafe members: {path.name}")
            if sum(member.file_size for member in members) > _MAX_UNCOMPRESSED_BYTES:
                raise EvaluationError(
                    f"track bundle expands beyond {_MAX_UNCOMPRESSED_BYTES} bytes: {path.name}"
                )
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise EvaluationError(f"cannot read track bundle {path.name}: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise EvaluationError(f"track bundle is not an NPZ archive: {path.name}")
    with archive:
        keys = set(archive.files)
        missing = _REQUIRED_KEYS - keys
        extra = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
        if missing or extra:
            raise EvaluationError(
                f"invalid keys in {path.name}: missing={sorted(missing)} extra={sorted(extra)}"
            )
        generated = np.asarray(archive["generated_tracks"])
        reference = np.asarray(archive["reference_tracks"])
        generated_visibility = (
            np.asarray(archive["generated_visibility"])
            if "generated_visibility" in archive
            else None
        )
        reference_visibility = (
            np.asarray(archive["reference_visibility"])
            if "reference_visibility" in archive
            else None
        )
    for name, tracks in (
        ("generated_tracks", generated),
        ("reference_tracks", reference),
    ):
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise EvaluationError(
                f"{path.name}:{name} must have shape (T, N, 2), got {tracks.shape}"
            )
        if tracks.shape[1] > max_tracks:
            raise EvaluationError(
                f"{path.name}:{name} has {tracks.shape[1]} tracks; protocol maximum is {max_tracks}"
            )
    if generated.shape[0] != reference.shape[0]:
        raise EvaluationError(
            f"{path.name} generated/reference frame counts differ: "
            f"{generated.shape[0]} != {reference.shape[0]}"
        )
    if expected_frames is not None and generated.shape[0] != expected_frames:
        raise EvaluationError(
            f"{path.name} has {generated.shape[0]} frames; expected {expected_frames}"
        )
    return {
        "path": path,
        "sha256": digest,
        "generated_tracks": generated,
        "reference_tracks": reference,
        "generated_visibility": generated_visibility,
        "reference_visibility": reference_visibility,
    }


def score_fdce_bundles(
    paths: Sequence[Path],
    *,
    expected_frames: int | None = 49,
    max_tracks: int = 16,
    visibility_threshold: float = 0.5,
    min_visible_fraction: float = 0.8,
    min_common_frames: int = 1,
) -> dict[str, Any]:
    """Score one or more NPZ bundles and return a deterministic report."""

    if not paths:
        raise EvaluationError("at least one track bundle is required")
    if expected_frames is not None and expected_frames < 2:
        raise EvaluationError("expected_frames must be at least two or disabled")
    if max_tracks < 1:
        raise EvaluationError("max_tracks must be positive")
    resolved = [Path(path).expanduser().resolve(strict=True) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise EvaluationError("duplicate track bundle paths are not allowed")

    records: list[dict[str, Any]] = []
    for index, path in enumerate(resolved):
        bundle = _load_bundle(
            path,
            expected_frames=expected_frames,
            max_tracks=max_tracks,
        )
        try:
            details = foreground_displacement_chamfer_error(
                bundle["generated_tracks"],
                bundle["reference_tracks"],
                bundle["generated_visibility"],
                bundle["reference_visibility"],
                visibility_threshold=visibility_threshold,
                min_visible_fraction=min_visible_fraction,
                min_common_frames=min_common_frames,
                return_details=True,
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"cannot score {path.name}: {exc}") from exc
        assert isinstance(details, FDCEResult)
        records.append(
            {
                "sample_index": index,
                "input_name": path.name,
                "input_sha256": bundle["sha256"],
                "frames": int(bundle["generated_tracks"].shape[0]),
                "generated_tracks_input": int(bundle["generated_tracks"].shape[1]),
                "reference_tracks_input": int(bundle["reference_tracks"].shape[1]),
                "fdce": details.score,
                "generated_to_reference": details.generated_to_reference,
                "reference_to_generated": details.reference_to_generated,
                "generated_tracks_scored": details.generated_tracks,
                "reference_tracks_scored": details.reference_tracks,
                "valid_track_pairs": details.valid_pairs,
            }
        )
    scores = [float(record["fdce"]) for record in records]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "parameters": {
            "expected_frames": expected_frames,
            "max_tracks": max_tracks,
            "visibility_threshold": visibility_threshold,
            "min_visible_fraction": min_visible_fraction,
            "min_common_frames": min_common_frames,
            "aggregation": "unweighted_sample_mean_and_median",
        },
        "samples": records,
        "summary": {
            "sample_count": len(records),
            "fdce_mean": statistics.fmean(scores),
            "fdce_median": statistics.median(scores),
            "units": "pixels_at_track_resolution",
            "lower_is_better": True,
        },
    }


def write_report_atomic(report: dict[str, Any], output: Path) -> None:
    """Write a JSON report without exposing a partially written result."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


__all__ = [
    "EvaluationError",
    "PROTOCOL_ID",
    "score_fdce_bundles",
    "write_report_atomic",
]
