from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_datasets.py"


def _module():
    spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_dataset_inventory_is_pinned() -> None:
    module = _module()
    inventory = module.plans()
    assert inventory["agibot_alpha"]["official_dataset"].endswith(
        "agibot-world/AgiBotWorld-Alpha"
    )
    assert inventory["agibot_alpha"]["revision"] == (
        "128665c9e0244c45d1cbe5c13f5a4706afd24f27"
    )
    assert inventory["egodex"]["archives"]["part2"]["url"].endswith("/egodex/part2.zip")
    assert inventory["egodex"]["archives"]["test"]["url"].endswith("/egodex/test.zip")


def test_egodex_dry_run_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "raw"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "egodex",
            "--part",
            "test",
            "--output",
            str(output),
            "--accept-license",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["url"].endswith("/egodex/test.zip")
    assert payload["extraction_root"].endswith("/raw/extracted/test")
    assert not output.exists()


def test_agibot_requires_explicit_license_acceptance(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "agibot-sample",
            "--output",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--accept-license" in result.stderr


def test_agibot_local_archive_requires_and_revalidates_source_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    archive = tmp_path / "existing" / module.AGIBOT_SAMPLE
    archive.parent.mkdir()
    archive.write_bytes(b"pinned AgiBot fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "AGIBOT_SAMPLE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(module, "AGIBOT_SAMPLE_SHA256", digest)
    source_record = archive.parent / module.AGIBOT_SOURCE_RECORD
    source_record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": module.AGIBOT_REPO,
                "filename": module.AGIBOT_SAMPLE,
                "revision": module.AGIBOT_REVISION,
                "revision_verified": True,
                "archive_path": module.AGIBOT_SAMPLE,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "reused"
    args = argparse.Namespace(
        accept_license=True,
        dry_run=False,
        extract=False,
        local_archive=archive,
        output=output,
        revision=module.AGIBOT_REVISION,
        source_record=source_record,
    )
    assert module.download_agibot(args) == 0
    reused = json.loads((output / module.AGIBOT_SOURCE_RECORD).read_text())
    assert reused["revision"] == module.AGIBOT_REVISION
    assert reused["archive_sha256"] == digest
    assert reused["revision_verified"] is True

    archive.write_bytes(b"tampered")
    args.output = tmp_path / "rejected"
    with pytest.raises(module.DownloadError, match="SHA-256"):
        module.download_agibot(args)
