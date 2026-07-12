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
    monkeypatch.delenv("CDLAM_HF_REVISION", raising=False)
    monkeypatch.setattr(DOWNLOAD_MODELS, "DEFAULT_REVISION", None)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--dry-run"])
    args = DOWNLOAD_MODELS.parse_args()
    assert args.local_dir == ROOT / "artifacts"
    assert args.revision is None


def _write_release(
    root: Path,
    *,
    content: bytes = b"checkpoint",
    include_auxiliary: bool = False,
) -> Path:
    del include_auxiliary
    assets = []
    for identity, kind, checkpoint_format, relative, step, role in (
        (
            "lam",
            "stage1",
            "cdlam.stage1.inference",
            "models/lam/model.pt",
            300,
            "research",
        ),
        (
            "pretrain",
            "stage2",
            "cdlam.stage2.overlay",
            "models/pretrain/model.pt",
            4000,
            "research",
        ),
        (
            "posttrain-100h",
            "stage3",
            "cdlam.stage3.overlay",
            "models/posttrain-100h/model.pt",
            3000,
            "unverified",
        ),
    ):
        value = content if identity == "posttrain-100h" else identity.encode()
        asset = root / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(value)
        assets.append(
            {
                "id": identity,
                "kind": kind,
                "checkpoint_format": checkpoint_format,
                "model_scale": "2B",
                "data_tier": "100h" if identity == "posttrain-100h" else "multi-source",
                "optimizer_step": step,
                "paper_role": role,
                "path": relative,
                "bytes": len(value),
                "release_sha256": hashlib.sha256(value).hexdigest(),
                "source_sha256": "b" * 64,
            }
        )
    bridge_content = b"bridge-checkpoint"
    bridge = root / "models" / "posttrain-100h" / "bridge.pt"
    bridge.write_bytes(bridge_content)
    contract_content = b'{"contract_id":"test"}\n'
    contract = root / "models" / "posttrain-100h" / "action_contract.json"
    contract.write_bytes(contract_content)
    manifest = {
        "schema_version": 1,
        "release_id": "cd-lam-2b-three-entry",
        "distribution_scope": "inference_evaluation",
        "historical_training_source_commit": "a" * 40,
        "public_runtime": {
            "repository": "https://github.com/yufanwei/CD-LAM",
            "revision": "c" * 40,
        },
        "base_model": {
            "repository": "nvidia/DreamDojo",
            "revision": "d" * 40,
            "path": "2B_pretrain/iter_000140000/model",
            "files": 257,
            "bytes": 12925543625,
            "tree_manifest_sha256": "e" * 64,
        },
        "verification_status": "tensor_exact",
        "assets": assets,
        "auxiliary_files": [
            {
                "id": "posttrain-100h-bridge",
                "kind": "bridge",
                "main_model_id": "posttrain-100h",
                "path": "models/posttrain-100h/bridge.pt",
                "bytes": len(bridge_content),
                "release_sha256": hashlib.sha256(bridge_content).hexdigest(),
                "source_sha256": "c" * 64,
                "verification_status": "tensor_exact",
            },
            {
                "id": "posttrain-100h-action-contract",
                "kind": "json_contract",
                "main_model_id": "posttrain-100h",
                "path": "models/posttrain-100h/action_contract.json",
                "bytes": len(contract_content),
                "release_sha256": hashlib.sha256(contract_content).hexdigest(),
                "source_sha256": hashlib.sha256(contract_content).hexdigest(),
                "verification_status": "byte_exact",
            },
        ],
        "blocked": [],
    }
    (root / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "models" / "posttrain-100h" / "model.pt"


def test_selective_download_always_requests_asset_manifest() -> None:
    assert DOWNLOAD_MODELS.effective_allow_patterns(["models/posttrain-100h/*"]) == [
        "models/posttrain-100h/*",
        "asset_manifest.json",
    ]
    assert DOWNLOAD_MODELS.effective_allow_patterns(
        ["models/posttrain-100h/model.pt"]
    ) == [
        "models/posttrain-100h/model.pt",
        "models/posttrain-100h/*",
        "asset_manifest.json",
    ]
    assert DOWNLOAD_MODELS.effective_allow_patterns(None) is None


def test_download_validator_checks_manifest_asset_size_and_hash(tmp_path) -> None:
    asset = _write_release(tmp_path)
    assert DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path) == {
        "assets_declared": 3,
        "assets_verified": 3,
        "auxiliary_files_declared": 2,
        "auxiliary_files_verified": 2,
    }
    assert DOWNLOAD_MODELS.validate_downloaded_snapshot(
        tmp_path, ["models/posttrain-100h/*"]
    ) == {
        "assets_declared": 3,
        "assets_verified": 1,
        "auxiliary_files_declared": 2,
        "auxiliary_files_verified": 2,
    }
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


def test_posttrain_selection_verifies_model_and_auxiliary_files(tmp_path) -> None:
    _write_release(tmp_path, include_auxiliary=True)
    expected = {
        "assets_declared": 3,
        "assets_verified": 1,
        "auxiliary_files_declared": 2,
        "auxiliary_files_verified": 2,
    }
    assert (
        DOWNLOAD_MODELS.validate_downloaded_snapshot(
            tmp_path, ["models/posttrain-100h/*"]
        )
        == expected
    )
    assert (
        DOWNLOAD_MODELS.validate_downloaded_snapshot(
            tmp_path, ["models/posttrain-100h/model.pt"]
        )
        == expected
    )

    contract = tmp_path / "models/posttrain-100h/action_contract.json"
    contract.unlink()
    with pytest.raises(
        DOWNLOAD_MODELS.ArtifactManifestError,
        match="released asset is missing",
    ):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(
            tmp_path, ["models/posttrain-100h/*"]
        )


def test_download_validator_rejects_legacy_or_drifted_release_identity(
    tmp_path,
) -> None:
    _write_release(tmp_path)
    manifest_path = tmp_path / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_id"] = "legacy-eight-entry-release"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DOWNLOAD_MODELS.ArtifactManifestError, match="release_id"):
        DOWNLOAD_MODELS.validate_downloaded_snapshot(tmp_path)
