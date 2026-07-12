from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "internal" / "vendor" / "scale_support"
sys.path.insert(0, str(VENDOR_ROOT))

from Scale.common.raw_data_contract import (  # noqa: E402
    agibot_alpha_physical_episode_key,
    audit_raw_split_records,
    egodex_physical_session_key,
    validate_raw_split_records,
)


BUILD_PATH = VENDOR_ROOT / "Scale" / "common" / "build_alpha_bridge_cache.py"
SPEC = importlib.util.spec_from_file_location("build_alpha_bridge_cache", BUILD_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_agibot_segments_share_a_physical_episode_key() -> None:
    first = agibot_alpha_physical_episode_key("327-648642-000")
    second = agibot_alpha_physical_episode_key("327-648642-003")
    assert first == second == "agibot_alpha:327:648642"
    assert BUILD._md5_split(first, 0.12) == BUILD._md5_split(second, 0.12)


def test_bridge_builder_derives_physical_keys_from_lerobot_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shard-00"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "source_id": source_id}) + "\n"
            for index, source_id in enumerate(("327-648642-000", "327-648642-003"))
        ),
        encoding="utf-8",
    )

    rows, chunk_size = BUILD._iter_episode_files(root, 1000)

    assert chunk_size == 1000
    assert [row[1] for row in rows] == [
        "agibot_alpha:327:648642",
        "agibot_alpha:327:648642",
    ]
    assert [row[2] for row in rows] == ["327-648642-000", "327-648642-003"]


def test_raw_audit_rejects_agibot_segment_level_split() -> None:
    summary = audit_raw_split_records(
        [
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-000",
                "split": "train",
            },
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-003",
                "split": "eval",
            },
        ]
    )

    assert summary["status"] == "fail"
    assert summary["leaking_groups"] == {"agibot_alpha:327:648642": ["eval", "train"]}


def test_raw_audit_rejects_native_egodex_test_in_train() -> None:
    summary = audit_raw_split_records(
        [
            {
                "dataset": "egodex",
                "episode_id": "egodex_test_arrange_topple_dominoes_0",
                "session_name": "2025-03-04_14-13-51.mov",
                "split": "train",
            }
        ]
    )

    assert summary["status"] == "fail"
    assert "native EgoDex test clip cannot enter train" in summary["errors"][0]


def test_raw_audit_rejects_egodex_session_crossing_splits() -> None:
    session = "2025-03-04_14-13-51.mov"
    summary = audit_raw_split_records(
        [
            {
                "dataset": "egodex",
                "episode_id": "egodex_test_arrange_topple_dominoes_0",
                "session_name": session,
                "split": "test",
            },
            {
                "dataset": "egodex",
                "episode_id": "egodex_test_arrange_topple_dominoes_2",
                "session_name": session,
                "split": "eval",
            },
        ]
    )

    assert summary["status"] == "fail"
    assert summary["leaking_groups"] == {
        egodex_physical_session_key("test", session): ["eval", "test"]
    }


def test_raw_audit_accepts_group_consistent_rows() -> None:
    summary = validate_raw_split_records(
        [
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-000",
                "split": "train",
            },
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-003",
                "split": "train",
            },
            {
                "dataset": "egodex",
                "episode_id": "egodex_part2_basic_pick_place_1094",
                "session_name": "2025-02-01_12-00-00.mov",
                "split": "eval",
            },
        ]
    )

    assert summary["status"] == "pass"
    assert summary["groups"] == 2


def test_bridge_builder_fails_closed_without_source_id(tmp_path: Path) -> None:
    root = tmp_path / "shard-00"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "data_path": "data/{episode_index}.parquet",
                "video_path": "videos/{video_key}/{episode_index}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index": 0}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="has no source_id"):
        BUILD._iter_episode_files(root, 1000)
