#!/usr/bin/env python3
"""Download CD-LAM release artifacts from Hugging Face."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


ASSET_MANIFEST = "asset_manifest.json"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ASSET_STRING_FIELDS = (
    "id",
    "kind",
    "model_scale",
    "data_tier",
    "paper_role",
    "path",
)


class ArtifactManifestError(ValueError):
    """Raised when a downloaded model snapshot is not release-verifiable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="yufanwei/CD-LAM")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Repeatable Hugging Face allow pattern; default downloads the release snapshot.",
    )
    parser.add_argument("--token", default=None, help="HF token; omit for public artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def effective_allow_patterns(patterns: Sequence[str] | None) -> list[str] | None:
    """Include the release manifest in every selective snapshot request."""

    if patterns is None:
        return None
    result = list(dict.fromkeys(patterns))
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


def validate_downloaded_snapshot(
    root: Path | str,
    requested_patterns: Sequence[str] | None = None,
) -> dict[str, int]:
    """Validate the tensor-exact release manifest and downloaded asset files."""

    directory = Path(root).expanduser().resolve()
    payload = _load_manifest(directory)
    if payload.get("schema_version") != 1:
        raise ArtifactManifestError("asset manifest schema_version must be 1")
    commit = payload.get("source_code_commit")
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        raise ArtifactManifestError(
            "asset manifest source_code_commit must be 40 hex characters"
        )
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

    selected = normalized
    if requested_patterns is not None:
        selected = [
            item
            for item in normalized
            if any(
                fnmatch.fnmatchcase(item[1].as_posix(), pattern)
                for pattern in requested_patterns
            )
        ]
        if not selected:
            raise ArtifactManifestError(
                "allow patterns do not match any released asset path in the manifest"
            )

    for asset, relative in selected:
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
    return {"assets_declared": len(normalized), "assets_verified": len(selected)}


def main() -> int:
    args = parse_args()
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
        token=args.token,
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
