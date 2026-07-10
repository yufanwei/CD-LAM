from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cdlam_download_models", ROOT / "scripts" / "download_models.py"
)
assert SPEC is not None and SPEC.loader is not None
DOWNLOAD_MODELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD_MODELS)


def test_default_download_directory_is_anchored_to_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--dry-run"])
    args = DOWNLOAD_MODELS.parse_args()
    assert args.local_dir == ROOT / "artifacts"


def _write_release(root: Path, *, content: bytes = b"checkpoint") -> Path:
    asset = root / "models" / "2b" / "stage3.pt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_code_commit": "a" * 40,
        "verification_status": "tensor_exact",
        "assets": [
            {
                "id": "stage3-2b-100h",
                "kind": "stage3_acwm",
                "model_scale": "2B",
                "data_tier": "100h",
                "optimizer_step": 3000,
                "paper_role": "candidate",
                "path": "models/2b/stage3.pt",
                "bytes": len(content),
                "release_sha256": digest,
                "source_sha256": "b" * 64,
            }
        ],
        "blocked": [],
    }
    (root / "asset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return asset


def test_selective_download_always_requests_asset_manifest() -> None:
    assert DOWNLOAD_MODELS.effective_allow_patterns(["models/2b/*"]) == [
        "models/2b/*",
        "asset_manifest.json",
    ]
    assert DOWNLOAD_MODELS.effective_allow_patterns(None) is None


def test_download_validator_checks_manifest_asset_size_and_hash(tmp_path) -> None:
    asset = _write_release(tmp_path)
    assert DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path) == {
        "assets_declared": 1,
        "assets_verified": 1,
    }
    assert DOWNLOAD_MODELS.validate_downloaded_snapshot(
        tmp_path, ["models/2b/*"]
    ) == {"assets_declared": 1, "assets_verified": 1}
    asset.write_bytes(b"corrupt")
    with pytest.raises(DOWNLOAD_MODELS.ArtifactManifestError, match="size mismatch"):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path)


def test_download_validator_rejects_checksum_mismatch_and_unsafe_path(tmp_path) -> None:
    asset = _write_release(tmp_path)
    asset.write_bytes(b"checkpoinu")
    with pytest.raises(DOWNLOAD_MODELS.ArtifactManifestError, match="SHA-256 mismatch"):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path)

    manifest_path = tmp_path / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][0]["path"] = "../stage3.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DOWNLOAD_MODELS.ArtifactManifestError, match="not a safe"):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path)


def test_download_validator_fails_closed_for_empty_or_unselected_release(
    tmp_path,
) -> None:
    with pytest.raises(DOWNLOAD_MODELS.ArtifactManifestError, match="is missing"):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path)

    _write_release(tmp_path)
    with pytest.raises(DOWNLOAD_MODELS.ArtifactManifestError, match="do not match"):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path, ["models/14b/*"])
