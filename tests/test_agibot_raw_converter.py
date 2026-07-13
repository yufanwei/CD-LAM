from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "internal" / "vendor" / "scale_support"
sys.path.insert(0, str(VENDOR_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONVERTER = _load_module(
    "cdlam_convert_agibot_alpha",
    ROOT / "internal" / "runtime" / "convert_agibot_alpha.py",
)
BRIDGE_BUILDER = _load_module(
    "cdlam_build_alpha_bridge_cache_for_converter",
    VENDOR_ROOT / "Scale" / "common" / "build_alpha_bridge_cache.py",
)


def test_construct_state_action_preserves_distinct_official_arrays() -> None:
    state_joint = np.arange(28, dtype=np.float64).reshape(2, 14)
    state_effector = np.array([[35.0, 120.0], [45.0, 110.0]])
    state_head = np.array([[0.1, 0.2], [0.3, 0.4]])
    state_waist = np.array([[0.5, 0.6], [0.7, 0.8]])
    action_joint = state_joint + 100.0
    action_effector = np.array([[0.0, 1.0], [1.0, 0.0]])
    action_head = state_head + 10.0
    action_waist = state_waist + 20.0
    action_velocity = np.array([[0.25, -0.5], [0.75, 0.125]])

    state, action = CONVERTER.construct_state_action(
        state_joint,
        state_effector,
        state_head,
        state_waist,
        action_joint,
        action_effector,
        action_head,
        action_waist,
        action_velocity,
    )

    assert state.shape == (2, 20)
    assert action.shape == (2, 22)
    np.testing.assert_array_equal(state[:, :14], state_joint)
    np.testing.assert_array_equal(state[:, 14:16], state_effector)
    np.testing.assert_array_equal(action[:, :14], action_joint)
    np.testing.assert_array_equal(action[:, 14:16], action_effector)
    np.testing.assert_array_equal(action[:, 16:18], action_head)
    np.testing.assert_array_equal(action[:, 18:20], action_waist)
    np.testing.assert_array_equal(action[:, 20:22], action_velocity)


def test_construct_state_action_rejects_misaligned_or_nonfinite_arrays() -> None:
    arrays = [
        np.zeros((3, 14)),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        np.zeros((3, 14)),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
    ]
    arrays[5] = np.zeros((2, 2))
    with pytest.raises(CONVERTER.ConversionError, match="different frame counts"):
        CONVERTER.construct_state_action(*arrays)
    arrays[5] = np.zeros((3, 2))
    arrays[8][0, 0] = np.nan
    with pytest.raises(CONVERTER.ConversionError, match="non-finite"):
        CONVERTER.construct_state_action(*arrays)


def test_read_proprio_uses_action_commands_not_state_proxies(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "proprio_stats.h5"
    frames = 3
    state_effector = np.array(
        [[35.0, 120.0], [40.0, 115.0], [45.0, 110.0]], dtype=np.float64
    )
    action_effector = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    action_velocity = np.array(
        [[0.25, -0.5], [0.75, 0.125], [-0.25, 0.0]], dtype=np.float64
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("state/joint/position", data=np.zeros((frames, 14)))
        handle.create_dataset("state/effector/position", data=state_effector)
        handle.create_dataset("state/head/position", data=np.zeros((frames, 2)))
        handle.create_dataset("state/waist/position", data=np.zeros((frames, 2)))
        handle.create_dataset("action/joint/position", data=np.ones((frames, 14)))
        handle.create_dataset("action/effector/position", data=action_effector)
        handle.create_dataset("action/head/position", data=np.ones((frames, 2)) * 2)
        handle.create_dataset("action/waist/position", data=np.ones((frames, 2)) * 3)
        handle.create_dataset("action/robot/velocity", data=action_velocity)
        handle.create_dataset(
            "timestamp", data=np.arange(frames, dtype=np.int64) * 33_333_333
        )

    state, action, timestamp = CONVERTER.read_proprio(path)

    np.testing.assert_array_equal(state[:, 14:16], state_effector)
    np.testing.assert_array_equal(action[:, 14:16], action_effector)
    np.testing.assert_array_equal(action[:, 20:22], action_velocity)
    np.testing.assert_allclose(timestamp, [0.0, 0.033333333, 0.066666666])


def test_read_proprio_rejects_missing_action_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "proprio_stats.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("timestamp", data=np.arange(3, dtype=np.int64))
    with pytest.raises(CONVERTER.ConversionError, match="datasets are missing"):
        CONVERTER.read_proprio(path)


def test_relative_timestamp_seconds_is_strict_and_nanosecond_based() -> None:
    raw = np.array([1_000_000_000, 1_033_333_333, 1_066_666_666], dtype=np.int64)
    np.testing.assert_allclose(
        CONVERTER.relative_timestamp_seconds(raw),
        [0.0, 0.033333333, 0.066666666],
    )
    with pytest.raises(CONVERTER.ConversionError, match="strictly increasing"):
        CONVERTER.relative_timestamp_seconds(np.array([1, 1], dtype=np.int64))


def test_terminal_proprio_record_is_aligned_without_silent_general_truncation() -> None:
    state = np.arange(60, dtype=np.float64).reshape(3, 20)
    action = np.arange(66, dtype=np.float64).reshape(3, 22)
    timestamp = np.array([0.0, 0.1, 0.2])

    aligned = CONVERTER.align_proprio_to_video(state, action, timestamp, 2)

    assert aligned[0].shape == (2, 20)
    assert aligned[1].shape == (2, 22)
    np.testing.assert_array_equal(aligned[2], timestamp[:2])
    assert aligned[3] == "dropped_terminal_proprio"
    with pytest.raises(CONVERTER.ConversionError, match="not a single terminal"):
        CONVERTER.align_proprio_to_video(state, action, timestamp, 1)


def test_split_assignment_is_shared_by_all_segments_of_physical_episode() -> None:
    held = {366}
    first = CONVERTER.split_for_source_id("327-648642-000", held)
    later = CONVERTER.split_for_source_id("327-648642-003", held)
    unseen_task = CONVERTER.split_for_source_id("366-700001-000", held)

    assert first == later
    assert unseen_task == ("test", "test_task")


def test_modality_metadata_matches_action_contract() -> None:
    metadata = CONVERTER.modality_metadata()

    assert list(metadata["action"]) == [item[0] for item in CONVERTER.ACTION_LAYOUT]
    assert metadata["action"]["left_effector_position"]["start"] == 14
    assert metadata["action"]["robot_velocity"]["end"] == 22
    assert metadata["video"]["top_head"]["original_key"] == (
        "observation.images.top_head"
    )


def test_video_materialization_copies_by_default_and_symlinks_explicitly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"synthetic-video-placeholder")
    copied = tmp_path / "portable" / "episode.mp4"
    linked = tmp_path / "linked" / "episode.mp4"

    CONVERTER.materialize_video(source, copied)
    CONVERTER.materialize_video(source, linked, mode="symlink")

    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_bytes() == source.read_bytes()
    assert linked.is_symlink()
    assert linked.resolve() == source.resolve()


def test_generated_dataset_yaml_is_portable_for_bridge_builder(tmp_path: Path) -> None:
    dataset_root = tmp_path / "converted" / "train"
    dataset_root.mkdir(parents=True)
    yaml_path = tmp_path / "converted" / "_dataset_paths_train.yaml"

    CONVERTER.write_dataset_path_yaml(yaml_path, [dataset_root])

    assert yaml_path.read_text(encoding="utf-8") == "dataset_path:\n  - train\n"
    assert BRIDGE_BUILDER._load_dataset_roots(yaml_path) == [dataset_root.resolve()]


def test_bridge_builder_accepts_converter_provenance_metadata(tmp_path: Path) -> None:
    root = tmp_path / "agibot-alpha" / "train"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": CONVERTER.DATA_PATH_PATTERN,
                "video_path": CONVERTER.VIDEO_PATH_PATTERN,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "source_id": "327-648642-003",
                "physical_episode_id": "agibot_alpha:327:648642",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    episodes, chunk_size = BRIDGE_BUILDER._iter_episode_files(root, 1000)

    assert chunk_size == 1000
    assert episodes[0][1:3] == (
        "agibot_alpha:327:648642",
        "327-648642-003",
    )


def test_bridge_preprocessing_needs_no_model_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Scale.common.shard_io import _bundled_official_lam

    monkeypatch.delenv("CDLAM_ACWM_ROOT", raising=False)
    pair = np.stack(
        [
            np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3),
            np.full((480, 640, 3), 127, dtype=np.uint8),
        ]
    )

    actual = BRIDGE_BUILDER._preprocess_lam_pair(pair)
    expected = _bundled_official_lam(pair, lam_hw=(240, 320))

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (2, 240, 320, 3)
