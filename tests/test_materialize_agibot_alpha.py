from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("av")


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load_module(
    "cdlam_materialize_agibot_alpha_candidate",
    ROOT / "scripts" / "materialize_agibot_alpha.py",
)


def test_verified_download_record_binds_revision_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "sample_dataset.tar"
    archive.write_bytes(b"archive fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(MATERIALIZER, "SAMPLE_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(MATERIALIZER, "SAMPLE_ARCHIVE_SHA256", digest)
    record = tmp_path / "agibot_download.json"
    record.write_text(
        json.dumps(
            {
                "dataset": MATERIALIZER.DATASET_ID,
                "filename": "sample_dataset.tar",
                "revision": MATERIALIZER.SCHEMA_REFERENCE_REVISION,
                "revision_verified": True,
                "archive_path": archive.name,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    observed = MATERIALIZER._source_provenance(record, None)

    assert observed["source_revision"] == MATERIALIZER.SCHEMA_REFERENCE_REVISION
    assert observed["source_revision_verification"] == "verified_archive_record"
    assert observed["source_archive_sha256"] == digest

    archive.write_bytes(b"tampered")
    with pytest.raises(MATERIALIZER.MaterializationError, match="do not match"):
        MATERIALIZER._source_provenance(record, None)


def _write_video(path: Path, frames: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = MATERIALIZER._ClipWriter(path)
    for index in range(frames):
        pixels = np.full(
            (MATERIALIZER.HEIGHT, MATERIALIZER.WIDTH, 3),
            fill_value=(index * 17) % 255,
            dtype=np.uint8,
        )
        writer.write(av.VideoFrame.from_ndarray(pixels, format="rgb24"))
    writer.close()


def _write_proprio(
    path: Path,
    frames: int,
    *,
    include_velocity: bool = True,
    nonaligned_extra: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["fixture"] = "official-layout"
        handle.create_dataset(
            "timestamp",
            data=1_000_000_000 + np.arange(frames, dtype=np.int64) * 33_333_333,
        )
        handle.create_dataset(
            "state/joint/position",
            data=np.arange(frames * 14, dtype=np.float64).reshape(frames, 14),
        )
        handle.create_dataset(
            "state/effector/position",
            data=np.arange(frames * 2, dtype=np.float64).reshape(frames, 2),
        )
        handle.create_dataset(
            "state/head/position",
            data=np.arange(frames * 2, dtype=np.float64).reshape(frames, 2),
        )
        handle.create_dataset(
            "state/waist/position",
            data=np.arange(frames * 2, dtype=np.float64).reshape(frames, 2),
        )
        handle.create_dataset(
            "action/joint/position",
            data=np.arange(frames * 14, dtype=np.float64).reshape(frames, 14),
        )
        handle.create_dataset(
            "action/effector/position",
            data=np.linspace(0.0, 1.0, frames * 2).reshape(frames, 2),
        )
        handle.create_dataset(
            "action/head/position",
            data=np.arange(frames * 2, dtype=np.float64).reshape(frames, 2) + 100,
        )
        handle.create_dataset(
            "action/waist/position",
            data=np.arange(frames * 2, dtype=np.float64).reshape(frames, 2) + 200,
        )
        if include_velocity:
            handle.create_dataset(
                "action/robot/velocity",
                data=np.arange(frames * 2, dtype=np.float64).reshape(frames, 2),
            )
        handle.create_dataset(
            "action/joint/index",
            data=np.array([0, 4, 5, frames - 1], dtype=np.int64),
        )
        handle.create_dataset(
            "action/effector/index", data=np.arange(frames, dtype=np.int64)
        )
        handle.create_dataset("metadata/version", data=np.int64(1))
        if nonaligned_extra:
            handle.create_dataset("unsupported/samples", data=np.zeros((3, 4)))


def _metadata(
    task_id: int,
    episode_id: int,
    bounds: list[tuple[int, int]],
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "task_name": "Fixture task",
        "init_scene_text": "Fixture scene",
        "label_info": {
            "action_config": [
                {
                    "start_frame": start,
                    "end_frame": end,
                    "action_text": f"Action {index}",
                    "skill": "Pick" if index % 2 == 0 else "Place",
                }
                for index, (start, end) in enumerate(bounds)
            ]
        },
    }


def _write_task_info(
    raw_root: Path,
    task_id: int,
    episodes: list[dict[str, object]],
) -> None:
    path = raw_root / "task_info" / f"task_{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episodes), encoding="utf-8")


def _write_episode_assets(
    raw_root: Path,
    task_id: int,
    episode_id: int,
    *,
    video_frames: int,
    proprio_frames: int,
) -> None:
    _write_video(
        raw_root
        / "observations"
        / str(task_id)
        / str(episode_id)
        / "videos"
        / MATERIALIZER.SOURCE_VIDEO_NAME,
        video_frames,
    )
    _write_proprio(
        raw_root
        / "proprio_stats"
        / str(task_id)
        / str(episode_id)
        / MATERIALIZER.SOURCE_H5_NAME,
        proprio_frames,
    )


def _video_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def test_official_episode_is_materialized_for_existing_converter(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "official"
    output_root = tmp_path / "segmented"
    _write_task_info(raw_root, 327, [_metadata(327, 648642, [(0, 5), (5, 12)])])
    _write_task_info(raw_root, 352, [_metadata(352, 648544, [(0, 4)])])
    _write_episode_assets(
        raw_root,
        327,
        648642,
        video_frames=11,
        proprio_frames=12,
    )

    summary = MATERIALIZER.materialize(
        MATERIALIZER.parse_args(
            [
                "--raw-root",
                str(raw_root),
                "--output",
                str(output_root),
                "--max-episodes",
                "1",
                "--source-revision",
                "fixture-revision",
                "--log-every",
                "0",
            ]
        )
    )

    assert summary["episode_count"] == 1
    assert summary["segment_count"] == 2
    assert summary["selected_episode_ids"] == ["327-648642"]
    assert summary["split_unit"] == "physical_episode"
    assert not (output_root / "task_info" / "task_352.json").exists()
    assert (output_root / "source_ids.txt").read_text(encoding="utf-8") == (
        "327-648642-000\n327-648642-001\n"
    )

    first = output_root / "train" / "327-648642-000"
    second = output_root / "train" / "327-648642-001"
    assert _video_frames(first / "head_color.mp4") == 5
    assert _video_frames(second / "head_color.mp4") == 6
    with h5py.File(first / "proprio_stats.h5") as handle:
        assert handle["timestamp"].shape == (5,)
        assert handle["state/joint/position"].shape == (5, 14)
        np.testing.assert_array_equal(handle["action/joint/index"][:], [0, 4])
        np.testing.assert_array_equal(handle["action/effector/index"][:], range(5))
        assert handle["metadata/version"][()] == 1
        assert handle.attrs["fixture"] == "official-layout"
    with h5py.File(second / "proprio_stats.h5") as handle:
        assert handle["timestamp"].shape == (7,)
        assert handle["action/robot/velocity"].shape == (7, 2)
        np.testing.assert_array_equal(handle["action/joint/index"][:], [0, 6])
        np.testing.assert_array_equal(handle["action/effector/index"][:], range(7))

    rows = [
        json.loads(line)
        for line in (output_root / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["physical_episode_id"] for row in rows} == {"agibot_alpha:327:648642"}
    assert [row["proprio_frames"] for row in rows] == [5, 7]
    assert [row["video_frames"] for row in rows] == [5, 6]
    assert rows[1]["bounds"] == {
        "start_frame_inclusive": 5,
        "end_frame_exclusive": 12,
    }
    filtered = json.loads(
        (output_root / "task_info" / "task_327.json").read_text(encoding="utf-8")
    )
    assert [row["episode_id"] for row in filtered] == [648642]


def test_overlapping_or_misspelled_annotations_fail_closed() -> None:
    overlap = _metadata(327, 1, [(0, 5), (4, 8)])
    with pytest.raises(MATERIALIZER.MaterializationError, match="overlapping"):
        MATERIALIZER._segments_from_metadata(327, 1, overlap)

    misspelled = _metadata(327, 1, [(0, 5)])
    misspelled["lable_info"] = misspelled.pop("label_info")
    with pytest.raises(MATERIALIZER.MaterializationError, match="misspelled"):
        MATERIALIZER._segments_from_metadata(327, 1, misspelled)


def test_proprio_schema_and_non_frame_aligned_arrays_fail_closed(
    tmp_path: Path,
) -> None:
    missing_velocity = tmp_path / "missing_velocity.h5"
    _write_proprio(missing_velocity, 8, include_velocity=False)
    with pytest.raises(
        MATERIALIZER.MaterializationError,
        match="action/robot/velocity is missing",
    ):
        MATERIALIZER.inspect_proprio(missing_velocity)

    nonaligned = tmp_path / "nonaligned.h5"
    _write_proprio(nonaligned, 8, nonaligned_extra=True)
    with pytest.raises(
        MATERIALIZER.MaterializationError,
        match="cannot safely slice non-frame-aligned",
    ):
        MATERIALIZER.slice_proprio(nonaligned, tmp_path / "slice.h5", 0, 4, 8)


def test_episode_selector_and_bounded_selection_remain_episode_granular(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "official"
    _write_task_info(
        raw_root,
        327,
        [
            _metadata(327, 20, [(0, 3), (3, 6)]),
            _metadata(327, 10, [(0, 2), (2, 4)]),
        ],
    )

    bounded = MATERIALIZER.discover_episodes(raw_root, None, None, 1)
    selected = MATERIALIZER.discover_episodes(
        raw_root,
        {327},
        {(327, 20)},
        None,
    )

    assert [(item.task_id, item.episode_id) for item in bounded] == [(327, 10)]
    assert [segment.source_id for segment in bounded[0].segments] == [
        "327-10-000",
        "327-10-001",
    ]
    assert [(item.task_id, item.episode_id) for item in selected] == [(327, 20)]


def test_raw_and_output_paths_must_be_disjoint(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with pytest.raises(MATERIALIZER.MaterializationError, match="must not contain"):
        MATERIALIZER._safe_paths(raw_root, raw_root / "segmented")
    with pytest.raises(MATERIALIZER.MaterializationError, match="must not contain"):
        MATERIALIZER._safe_paths(raw_root, tmp_path)
