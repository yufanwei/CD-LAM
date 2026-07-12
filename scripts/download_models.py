#!/usr/bin/env python3
"""Download CD-LAM release artifacts from Hugging Face."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


ASSET_MANIFEST = "asset_manifest.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "cd-lam-2b-three-entry"
RELEASE_SCOPE = "inference_evaluation"
PUBLISHED_REVISION = "591e22e582e920cbb4fdfac1a45365e81088bd06"
EXPECTED_MAIN_ASSETS = {
    "lam": ("models/lam/model.pt", "stage1", "cdlam.stage1.inference"),
    "pretrain": (
        "models/pretrain/model.pt",
        "stage2",
        "cdlam.stage2.overlay",
    ),
    "posttrain-100h": (
        "models/posttrain-100h/model.pt",
        "stage3",
        "cdlam.stage3.overlay",
    ),
}
EXPECTED_AUXILIARY_FILES = {
    "posttrain-100h-bridge": (
        "models/posttrain-100h/bridge.pt",
        "bridge",
    ),
    "posttrain-100h-action-contract": (
        "models/posttrain-100h/action_contract.json",
        "json_contract",
    ),
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ASSET_STRING_FIELDS = (
    "id",
    "kind",
    "model_scale",
    "data_tier",
    "paper_role",
    "path",
    "checkpoint_format",
)
_AUXILIARY_STRING_FIELDS = (
    "id",
    "kind",
    "main_model_id",
    "path",
    "verification_status",
)


class ArtifactManifestError(ValueError):
    """Raised when a downloaded model snapshot is not release-verifiable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="yufanwei/CD-LAM")
    parser.add_argument(
        "--revision",
        default=os.environ.get("CDLAM_HF_REVISION", PUBLISHED_REVISION),
        help=(
            "Immutable 40-character HF commit; defaults to CDLAM_HF_REVISION "
            "or the published compact release."
        ),
    )
    parser.add_argument("--local-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Repeatable Hugging Face allow pattern; default downloads the release snapshot.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def effective_allow_patterns(patterns: Sequence[str] | None) -> list[str] | None:
    """Include the manifest and a selected model's colocated support files."""

    if patterns is None:
        return None
    result: list[str] = []
    for pattern in patterns:
        if pattern not in result:
            result.append(pattern)
        path = PurePosixPath(pattern)
        if path.name == "model.pt" and not any(token in pattern for token in "*?["):
            sibling_pattern = f"{path.parent.as_posix()}/*"
            if sibling_pattern not in result:
                result.append(sibling_pattern)
    if ASSET_MANIFEST not in result:
        result.append(ASSET_MANIFEST)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactManifestError(f"{label} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {".", ""}:
        raise ArtifactManifestError(f"{label} is not a safe relative path: {value!r}")
    return path


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / ASSET_MANIFEST
    if not path.is_file():
        raise ArtifactManifestError(
            f"{ASSET_MANIFEST} is missing; no released model assets were verified"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"cannot read {ASSET_MANIFEST}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactManifestError(f"{ASSET_MANIFEST} must contain a JSON object")
    return payload


def _validate_release_identity(payload: dict[str, Any]) -> None:
    if payload.get("release_id") != RELEASE_ID:
        raise ArtifactManifestError(f"asset manifest release_id must be {RELEASE_ID!r}")
    if payload.get("distribution_scope") != RELEASE_SCOPE:
        raise ArtifactManifestError(
            f"asset manifest distribution_scope must be {RELEASE_SCOPE!r}"
        )
    commit = payload.get("historical_training_source_commit")
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        raise ArtifactManifestError(
            "historical_training_source_commit must be 40 hex characters"
        )
    runtime = payload.get("public_runtime")
    if not isinstance(runtime, dict):
        raise ArtifactManifestError("asset manifest public_runtime must be an object")
    if runtime.get("repository") != "https://github.com/yufanwei/CD-LAM":
        raise ArtifactManifestError("public_runtime repository is not CD-LAM")
    runtime_revision = runtime.get("revision")
    if (
        not isinstance(runtime_revision, str)
        or _HEX40.fullmatch(runtime_revision) is None
    ):
        raise ArtifactManifestError("public_runtime revision must be a full Git commit")
    base = payload.get("base_model")
    if not isinstance(base, dict):
        raise ArtifactManifestError("asset manifest base_model must be an object")
    if base.get("repository") != "nvidia/DreamDojo":
        raise ArtifactManifestError("base_model repository is not nvidia/DreamDojo")
    revision = base.get("revision")
    if not isinstance(revision, str) or _HEX40.fullmatch(revision) is None:
        raise ArtifactManifestError("base_model revision must be a full HF commit")
    _asset_path(base.get("path"), "base_model.path")
    tree_hash = base.get("tree_manifest_sha256")
    if not isinstance(tree_hash, str) or _HEX64.fullmatch(tree_hash) is None:
        raise ArtifactManifestError(
            "base_model.tree_manifest_sha256 must be 64 lowercase hex characters"
        )
    for field in ("files", "bytes"):
        value = base.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactManifestError(f"base_model.{field} must be positive")
    if payload.get("blocked") not in ([], None):
        raise ArtifactManifestError("compact release must not declare blocked assets")


def validate_downloaded_snapshot(
    root: Path | str,
    requested_patterns: Sequence[str] | None = None,
) -> dict[str, int]:
    """Validate the tensor-exact release manifest and downloaded asset files."""

    directory = Path(root).expanduser().resolve()
    payload = _load_manifest(directory)
    if payload.get("schema_version") != 1:
        raise ArtifactManifestError("asset manifest schema_version must be 1")
    _validate_release_identity(payload)
    if payload.get("verification_status") != "tensor_exact":
        raise ArtifactManifestError(
            "asset manifest verification_status must be 'tensor_exact'"
        )
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ArtifactManifestError(
            "asset manifest must declare at least one released asset"
        )

    normalized: list[tuple[dict[str, Any], PurePosixPath]] = []
    identities: set[str] = set()
    paths: set[str] = set()
    for index, asset in enumerate(assets):
        label = f"assets[{index}]"
        if not isinstance(asset, dict):
            raise ArtifactManifestError(f"{label} must be an object")
        for field in _ASSET_STRING_FIELDS:
            if not isinstance(asset.get(field), str) or not asset[field].strip():
                raise ArtifactManifestError(
                    f"{label}.{field} must be a nonempty string"
                )
        identity = asset["id"]
        if identity in identities:
            raise ArtifactManifestError(f"duplicate asset id: {identity!r}")
        identities.add(identity)
        relative = _asset_path(asset["path"], f"{label}.path")
        if relative.as_posix() in paths:
            raise ArtifactManifestError(
                f"duplicate asset path: {relative.as_posix()!r}"
            )
        paths.add(relative.as_posix())
        for field in ("optimizer_step", "bytes"):
            value = asset.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ArtifactManifestError(
                    f"{label}.{field} must be a positive integer"
                )
        for field in ("release_sha256", "source_sha256"):
            value = asset.get(field)
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                raise ArtifactManifestError(
                    f"{label}.{field} must be 64 lowercase hex characters"
                )
        normalized.append((asset, relative))

    main_identities = set(identities)
    if main_identities != set(EXPECTED_MAIN_ASSETS):
        raise ArtifactManifestError(
            "main asset identities do not match the compact CD-LAM release"
        )
    for asset, relative in normalized:
        expected_path, expected_kind, expected_format = EXPECTED_MAIN_ASSETS[
            asset["id"]
        ]
        if (
            relative.as_posix() != expected_path
            or asset["kind"] != expected_kind
            or asset["checkpoint_format"] != expected_format
        ):
            raise ArtifactManifestError(f"asset contract mismatch for {asset['id']!r}")
    auxiliary_files = payload.get("auxiliary_files", [])
    if not isinstance(auxiliary_files, list):
        raise ArtifactManifestError("asset manifest auxiliary_files must be a list")
    normalized_auxiliary: list[tuple[dict[str, Any], PurePosixPath]] = []
    for index, auxiliary in enumerate(auxiliary_files):
        label = f"auxiliary_files[{index}]"
        if not isinstance(auxiliary, dict):
            raise ArtifactManifestError(f"{label} must be an object")
        for field in _AUXILIARY_STRING_FIELDS:
            if (
                not isinstance(auxiliary.get(field), str)
                or not auxiliary[field].strip()
            ):
                raise ArtifactManifestError(
                    f"{label}.{field} must be a nonempty string"
                )
        identity = auxiliary["id"]
        if identity in identities:
            raise ArtifactManifestError(f"duplicate asset id: {identity!r}")
        identities.add(identity)
        if auxiliary["main_model_id"] not in main_identities:
            raise ArtifactManifestError(
                f"{label}.main_model_id does not reference a released asset"
            )
        relative = _asset_path(auxiliary["path"], f"{label}.path")
        if relative.as_posix() in paths:
            raise ArtifactManifestError(
                f"duplicate asset path: {relative.as_posix()!r}"
            )
        paths.add(relative.as_posix())
        size = auxiliary.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ArtifactManifestError(f"{label}.bytes must be a positive integer")
        for field in ("release_sha256", "source_sha256"):
            value = auxiliary.get(field)
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                raise ArtifactManifestError(
                    f"{label}.{field} must be 64 lowercase hex characters"
                )
        if auxiliary["verification_status"] not in {
            "byte_exact",
            "tensor_exact",
        }:
            raise ArtifactManifestError(
                f"{label}.verification_status must be byte_exact or tensor_exact"
            )
        normalized_auxiliary.append((auxiliary, relative))

    auxiliary_identities = {item[0]["id"] for item in normalized_auxiliary}
    if auxiliary_identities != set(EXPECTED_AUXILIARY_FILES):
        raise ArtifactManifestError(
            "auxiliary identities do not match the compact CD-LAM release"
        )
    for auxiliary, relative in normalized_auxiliary:
        expected_path, expected_kind = EXPECTED_AUXILIARY_FILES[auxiliary["id"]]
        if relative.as_posix() != expected_path or auxiliary["kind"] != expected_kind:
            raise ArtifactManifestError(
                f"auxiliary contract mismatch for {auxiliary['id']!r}"
            )

    selected = normalized
    selected_auxiliary = normalized_auxiliary
    if requested_patterns is not None:
        selected = [
            item
            for item in normalized
            if any(
                fnmatch.fnmatchcase(item[1].as_posix(), pattern)
                for pattern in requested_patterns
            )
        ]
        selected_main_ids = {asset["id"] for asset, _ in selected}
        selected_auxiliary = [
            item
            for item in normalized_auxiliary
            if item[0]["main_model_id"] in selected_main_ids
            or any(
                fnmatch.fnmatchcase(item[1].as_posix(), pattern)
                for pattern in requested_patterns
            )
        ]
        if not selected and not selected_auxiliary:
            raise ArtifactManifestError(
                "allow patterns do not match any released asset path in the manifest"
            )

    for asset, relative in [*selected, *selected_auxiliary]:
        path = directory.joinpath(*relative.parts)
        if not path.is_file():
            raise ArtifactManifestError(
                f"released asset is missing after download: {relative}"
            )
        actual_size = path.stat().st_size
        if actual_size != asset["bytes"]:
            raise ArtifactManifestError(
                f"released asset size mismatch for {relative}: "
                f"expected {asset['bytes']}, got {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != asset["release_sha256"]:
            raise ArtifactManifestError(
                f"released asset SHA-256 mismatch for {relative}: "
                f"expected {asset['release_sha256']}, got {actual_hash}"
            )
    return {
        "assets_declared": len(normalized),
        "assets_verified": len(selected),
        "auxiliary_files_declared": len(normalized_auxiliary),
        "auxiliary_files_verified": len(selected_auxiliary),
    }


def main() -> int:
    args = parse_args()
    if not args.revision:
        raise SystemExit(
            "No Hugging Face revision was provided. Unset the empty "
            "CDLAM_HF_REVISION value to use the published immutable revision, "
            "or set it to a 40-character commit."
        )
    if args.repo_id == "yufanwei/CD-LAM" and _HEX40.fullmatch(args.revision) is None:
        raise SystemExit(
            "The official CD-LAM download requires an immutable 40-character "
            "Hugging Face commit, not a mutable branch name."
        )
    allow_patterns = effective_allow_patterns(args.allow_patterns)
    request = {
        "repo_id": args.repo_id,
        "repo_type": "model",
        "revision": args.revision,
        "local_dir": str(args.local_dir),
        "allow_patterns": allow_patterns,
    }
    if args.dry_run:
        print(json.dumps(request, indent=2, sort_keys=True))
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install CD-LAM with the 'download' extra."
        ) from exc

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    try:
        summary = validate_downloaded_snapshot(path, args.allow_patterns)
    except ArtifactManifestError as exc:
        raise SystemExit(
            f"Downloaded model snapshot is not release-valid: {exc}"
        ) from exc
    print(json.dumps({"path": path, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
