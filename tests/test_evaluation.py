from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cd_lam.__main__ import main
from cd_lam.evaluation import EvaluationError, score_fdce_bundles


def _bundle(
    path: Path, *, tracks: int = 1, frames: int = 49, drift: float = 0.0
) -> None:
    reference = np.zeros((frames, tracks, 2), dtype=np.float32)
    generated = reference.copy()
    generated[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None] * drift
    visibility = np.ones((frames, tracks), dtype=bool)
    np.savez(
        path,
        generated_tracks=generated,
        reference_tracks=reference,
        generated_visibility=visibility,
        reference_visibility=visibility,
    )


def test_score_fdce_bundles_reports_per_sample_and_aggregate(tmp_path: Path) -> None:
    identical = tmp_path / "identical.npz"
    moving = tmp_path / "moving.npz"
    _bundle(identical)
    _bundle(moving, drift=1.0)

    report = score_fdce_bundles([identical, moving])

    assert report["protocol_id"] == "cdlam-fdce-displacement-v1"
    assert report["summary"]["sample_count"] == 2
    assert report["samples"][0]["fdce"] == pytest.approx(0.0)
    assert report["samples"][1]["fdce"] == pytest.approx(24.5)
    assert report["summary"]["fdce_mean"] == pytest.approx(12.25)
    assert len(report["samples"][0]["input_sha256"]) == 64
    assert all("/" not in record["input_name"] for record in report["samples"])


def test_score_fdce_rejects_non_protocol_population(tmp_path: Path) -> None:
    too_many = tmp_path / "too-many.npz"
    wrong_frames = tmp_path / "wrong-frames.npz"
    _bundle(too_many, tracks=17)
    _bundle(wrong_frames, frames=13)

    with pytest.raises(EvaluationError, match="protocol maximum"):
        score_fdce_bundles([too_many])
    with pytest.raises(EvaluationError, match="expected 49"):
        score_fdce_bundles([wrong_frames])


def test_score_fdce_cli_writes_atomic_json(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "tracks.npz"
    output = tmp_path / "result" / "fdce.json"
    _bundle(bundle)

    assert main(["score-fdce", "--tracks", str(bundle), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["fdce_mean"] == 0.0
    assert str(output) in capsys.readouterr().out


def test_score_fdce_cli_fails_closed_on_bad_archive(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "bad.npz"
    np.savez(bundle, generated_tracks=np.zeros((49, 1, 2)), unexpected=np.zeros(1))

    assert main(["score-fdce", "--tracks", str(bundle)]) == 2
    assert "invalid keys" in capsys.readouterr().err
