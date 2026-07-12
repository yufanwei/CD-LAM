from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "index_egodex.py"
SPEC = importlib.util.spec_from_file_location("cdlam_index_egodex", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INDEXER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INDEXER
SPEC.loader.exec_module(INDEXER)


def _touch_pair(root: Path, task: str, index: int, video_suffix: str = ".mp4"):
    directory = root / "wrapper" / task
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / f"{index}.hdf5"
    video = directory / f"{index}{video_suffix}"
    metadata.touch()
    video.touch()
    return metadata.resolve(), video.resolve()


def test_discovers_recursive_official_pairs_and_h264_name(tmp_path: Path) -> None:
    first = _touch_pair(tmp_path, "basic_pick_place", 7)
    second = _touch_pair(tmp_path, "insert_object", 12, ".h264.mp4")

    pairs = INDEXER.discover_pairs(tmp_path)

    assert [(pair.task, pair.index) for pair in pairs] == [
        ("basic_pick_place", 7),
        ("insert_object", 12),
    ]
    assert (pairs[0].metadata_path, pairs[0].video_path) == first
    assert (pairs[1].metadata_path, pairs[1].video_path) == second


@pytest.mark.parametrize("missing", ["metadata", "video"])
def test_fails_closed_on_missing_pair(tmp_path: Path, missing: str) -> None:
    directory = tmp_path / "task"
    directory.mkdir()
    if missing != "metadata":
        (directory / "0.hdf5").touch()
    if missing != "video":
        (directory / "0.mp4").touch()

    with pytest.raises(INDEXER.EgoDexIndexError, match="no matching"):
        INDEXER.discover_pairs(tmp_path)


def test_fails_closed_when_both_video_encodings_exist(tmp_path: Path) -> None:
    _touch_pair(tmp_path, "task", 0)
    (tmp_path / "wrapper" / "task" / "0.h264.mp4").touch()

    with pytest.raises(INDEXER.EgoDexIndexError, match="ambiguous EgoDex video"):
        INDEXER.discover_pairs(tmp_path)


def test_session_split_is_deterministic_grouped_and_nonempty() -> None:
    sessions = ["session-a.mov", "session-a.mov", "session-b.mov", "session-c.mov"]

    first = INDEXER.session_splits(sessions, part="part2", eval_fraction=0.1, seed=42)
    second = INDEXER.session_splits(
        list(reversed(sessions)), part="part2", eval_fraction=0.1, seed=42
    )

    assert first == second
    assert set(first.values()) == {"train", "eval"}
    assert first["session-a.mov"] in {"train", "eval"}
    assert INDEXER.session_splits(
        sessions, part="test", eval_fraction=0.1, seed=42
    ) == {session: "test" for session in sorted(set(sessions))}


def test_non_test_part_rejects_one_physical_session() -> None:
    with pytest.raises(INDEXER.EgoDexIndexError, match="at least two unique"):
        INDEXER.session_splits(
            ["only-session.mov"], part="part2", eval_fraction=0.1, seed=42
        )
    assert INDEXER.session_splits(
        ["only-session.mov"], part="test", eval_fraction=0.1, seed=42
    ) == {"only-session.mov": "test"}


def test_bounded_selection_is_deterministic_and_keeps_both_splits() -> None:
    rows = [
        {
            "dataset": "egodex",
            "episode_id": f"egodex_part2_task_{index}",
            "part": "part2",
            "session_name": f"session-{index // 2}",
            "split": "eval" if index < 4 else "train",
        }
        for index in range(12)
    ]

    first = INDEXER.select_bounded_rows(rows, max_clips=5, seed=42)
    second = INDEXER.select_bounded_rows(list(reversed(rows)), max_clips=5, seed=42)

    assert first == second
    assert len(first) <= 5
    assert {row["split"] for row in first} == {"train", "eval"}
    selected_sessions = {row["session_name"] for row in first}
    assert {row["episode_id"] for row in first} == {
        row["episode_id"] for row in rows if row["session_name"] in selected_sessions
    }


def test_bounded_selection_rejects_cap_that_drops_a_required_split() -> None:
    rows = [
        {
            "episode_id": "egodex_part2_task_0",
            "part": "part2",
            "session_name": "train-session",
            "split": "train",
        },
        {
            "episode_id": "egodex_part2_task_1",
            "part": "part2",
            "session_name": "eval-session",
            "split": "eval",
        },
    ]

    with pytest.raises(INDEXER.EgoDexIndexError, match="at least 2"):
        INDEXER.select_bounded_rows(rows, max_clips=1, seed=0)


def test_unknown_tasks_stay_unlabeled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    known_metadata, _ = _touch_pair(tmp_path / "raw", "known_task", 0)
    unknown_metadata, _ = _touch_pair(tmp_path / "raw", "unknown_task", 1)
    pairs = INDEXER.discover_pairs(tmp_path / "raw")
    sessions = {
        known_metadata: "known-session.mov",
        unknown_metadata: "unknown-session.mov",
    }
    monkeypatch.setattr(INDEXER, "read_session_name", sessions.__getitem__)

    rows = INDEXER.build_rows(
        pairs,
        part="part2",
        output=tmp_path / "index.jsonl",
        eval_fraction=0.5,
        seed=7,
        primitive_map={"known_task": "pick_place"},
    )

    by_task = {row["task_name"]: row for row in rows}
    assert by_task["known_task"]["primitive"] == "pick_place"
    assert by_task["known_task"]["primitive_raw"] == "known_task"
    assert "primitive" not in by_task["unknown_task"]
    assert "primitive_raw" not in by_task["unknown_task"]
    assert set(row["split"] for row in rows) == {"train", "eval"}
    assert all(row["split_policy"] == "sha256_ranked_session_v1" for row in rows)
    assert all(row["split_seed"] == 7 for row in rows)


def test_hdf5_index_is_accepted_by_raw_subset_normalizer(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    raw = tmp_path / "extracted" / "part2"
    for task, index, session in (
        ("basic_pick_place", 0, "session-a.mov"),
        ("basic_pick_place", 1, "session-b.mov"),
    ):
        metadata, video = _touch_pair(raw, task, index)
        metadata.unlink()
        with h5py.File(metadata, "w") as handle:
            handle.attrs["session_name"] = session
        video.write_bytes(b"video bytes are decoded only after provenance validation")
    primitive_map = tmp_path / "primitives.json"
    primitive_map.write_text(
        json.dumps({"basic_pick_place": "pick_place"}), encoding="utf-8"
    )
    output = tmp_path / "prepared" / "egodex-part2.jsonl"

    exit_code = INDEXER.main(
        [
            "--root",
            str(raw),
            "--part",
            "part2",
            "--output",
            str(output),
            "--eval-fraction",
            "0.1",
            "--seed",
            "42",
            "--primitive-map",
            str(primitive_map),
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert set(row["split"] for row in rows) == {"train", "eval"}
    assert all(row["primitive"] == "pick_place" for row in rows)

    vendor_root = ROOT / "internal" / "vendor" / "scale_support"
    sys.path.insert(0, str(vendor_root))
    from Scale.common.raw_subset_ingest import _normalize_records

    normalized = _normalize_records(rows, output.parent)
    assert len(normalized) == 2
    assert {clip.session_name for clip in normalized} == {
        "session-a.mov",
        "session-b.mov",
    }
