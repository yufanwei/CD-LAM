from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "index_agibot_segments.py"
SPEC = importlib.util.spec_from_file_location("index_agibot_segments", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INDEX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INDEX
SPEC.loader.exec_module(INDEX)


def _materialized_fixture(root: Path) -> None:
    (root / "materialization.json").parent.mkdir(parents=True)
    (root / "materialization.json").write_text(
        json.dumps({"dataset_id": "agibot-world/AgiBotWorld-Alpha"}),
        encoding="utf-8",
    )
    rows = []
    for task, episode, segments in ((1, 10, 2), (1, 11, 1), (2, 20, 1)):
        for segment in range(segments):
            source_id = f"{task}-{episode}-{segment:03d}"
            video = root / "train" / source_id / "head_color.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(f"video:{source_id}".encode())
            rows.append(
                {
                    "source_id": source_id,
                    "physical_episode_id": f"agibot_alpha:{task}:{episode}",
                    "video_frames": 20,
                    "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "action_text": "Pick object",
                    "skill": "Pick",
                }
            )
    (root / "provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_index_preserves_physical_groups_and_exact_primitive_map(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "materialized"
    output = tmp_path / "index" / "agibot.jsonl"
    _materialized_fixture(materialized)

    first = INDEX.build_rows(
        materialized,
        output,
        eval_fraction=0.34,
        seed=7,
        max_clips=4,
        primitive_map={"Pick": "pick_place"},
    )
    second = INDEX.build_rows(
        materialized,
        output,
        eval_fraction=0.34,
        seed=7,
        max_clips=4,
        primitive_map={"Pick": "pick_place"},
    )

    assert first == second
    assert {row["split"] for row in first} == {"train", "eval"}
    assert {row["primitive"] for row in first} == {"pick_place"}
    by_episode: dict[str, set[str]] = {}
    for row in first:
        by_episode.setdefault(row["physical_episode_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_episode.values())
    selected_ids = {row["source_id"] for row in first}
    assert {"1-10-000", "1-10-001"} <= selected_ids or not (
        {"1-10-000", "1-10-001"} & selected_ids
    )


def test_index_rejects_materialized_video_hash_drift(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized"
    output = tmp_path / "index.jsonl"
    _materialized_fixture(materialized)
    video = materialized / "train" / "1-10-000" / "head_color.mp4"
    video.write_bytes(b"tampered")

    with pytest.raises(INDEX.AgiBotIndexError, match="hash mismatch"):
        INDEX.build_rows(
            materialized,
            output,
            eval_fraction=0.34,
            seed=7,
            max_clips=4,
            primitive_map={},
        )
