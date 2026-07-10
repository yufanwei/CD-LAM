from __future__ import annotations

import json
from pathlib import Path

import pytest

from cd_lam.data import (
    DataContractError,
    prepare_episode_manifests,
    validate_prepared_manifests,
)

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_portable_fixture_builds_every_stage_and_preserves_test_split(tmp_path) -> None:
    summary = prepare_episode_manifests(
        ROOT / "tests" / "fixtures" / "episodes.jsonl",
        tmp_path,
    )

    assert summary.episodes == 2
    assert summary.stage1_pairs == 18
    assert summary.stage2_windows == 8
    assert summary.bridge_pairs == 24
    assert summary.stage3_windows == 2
    assert validate_prepared_manifests(tmp_path) == {
        "stage1_pairs": 18,
        "stage2_windows": 8,
        "bridge_pairs": 24,
        "stage3_windows": 2,
    }

    stage3 = _jsonl(tmp_path / "stage3_windows.jsonl")
    assert {row["split"] for row in stage3} == {"train", "test"}
    assert all(len(row["video_frame_indices"]) == 13 for row in stage3)
    assert all(len(row["transition_indices"]) == 12 for row in stage3)
    bridge = _jsonl(tmp_path / "bridge_pairs.jsonl")
    assert all(len(row["action_22"]) == 22 for row in bridge)
    assert all(row["source_stride"] == 4 for row in bridge)


def test_episode_metadata_rejects_nonfinite_fps(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        '{"episode_id":"bad","split":"train","num_frames":2,"fps":NaN}\n'
    )
    with pytest.raises(DataContractError, match="fps must be finite"):
        prepare_episode_manifests(source, tmp_path / "out")


def test_stage3_validator_rejects_reordered_transition(tmp_path) -> None:
    prepare_episode_manifests(
        ROOT / "tests" / "fixtures" / "episodes.jsonl", tmp_path
    )
    path = tmp_path / "stage3_windows.jsonl"
    rows = _jsonl(path)
    rows[0]["transition_indices"][0] = [0, 8]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(DataContractError, match="transitions do not match frames"):
        validate_prepared_manifests(tmp_path)
