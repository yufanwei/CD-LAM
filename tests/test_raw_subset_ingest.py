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

from Scale.common.raw_subset_ingest import (  # noqa: E402
    MASK_POLICY,
    RawSubsetError,
    RawSubsetOptions,
    build_raw_subset,
)


def _write_index(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_mp4(path: Path, *, frames: int = 16, fps: int = 20, seed: int = 0) -> None:
    av = pytest.importorskip("av")
    rng = np.random.default_rng(seed)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
            image[:, :4, :] = index
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_raw_subset_rejects_physical_episode_split_leakage(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not decoded because provenance validation fails")
    index = tmp_path / "clips.jsonl"
    _write_index(
        index,
        [
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-000",
                "split": "train",
                "video_path": str(video),
            },
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-003",
                "split": "eval",
                "video_path": str(video),
            },
        ],
    )

    with pytest.raises(RawSubsetError, match="physical group crosses splits"):
        build_raw_subset(index, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_native_egodex_test_must_keep_test_split(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not decoded because provenance validation fails")
    index = tmp_path / "clips.jsonl"
    _write_index(
        index,
        [
            {
                "dataset": "egodex",
                "episode_id": "egodex_test_arrange_topple_dominoes_0",
                "session_name": "2025-03-04_14-13-51.mov",
                "split": "eval",
                "video_path": str(video),
            }
        ],
    )

    with pytest.raises(RawSubsetError, match="must keep split='test'"):
        build_raw_subset(index, tmp_path / "out")


def test_builds_embedded_shard_and_disjoint_stage_manifests(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    train_video = tmp_path / "alpha.mp4"
    eval_video = tmp_path / "egodex-part2.mp4"
    native_test_video = tmp_path / "egodex-test.mp4"
    _write_mp4(train_video, seed=1)
    _write_mp4(eval_video, seed=2)
    _write_mp4(native_test_video, seed=3)
    index = tmp_path / "clips.jsonl"
    _write_index(
        index,
        [
            {
                "dataset": "agibot_alpha",
                "source_id": "327-648642-000",
                "task_name": "place fruit",
                "primitive": "pick_place",
                "split": "train",
                "video_path": train_video.name,
            },
            {
                "dataset": "egodex",
                "episode_id": "egodex_part2_basic_pick_place_1094",
                "session_name": "2025-02-01_12-00-00.mov",
                "split": "eval",
                "video_path": eval_video.name,
            },
            {
                "dataset": "egodex",
                "episode_id": "egodex_test_arrange_topple_dominoes_0",
                "session_name": "2025-03-04_14-13-51.mov",
                "split": "test",
                "video_path": native_test_video.name,
            },
        ],
    )
    output = tmp_path / "prepared"

    report = build_raw_subset(
        index,
        output,
        options=RawSubsetOptions(pairs_per_clip=2, windows_per_clip=2),
    )

    shard_path = output / "shards" / "raw-subset-00000.parquet"
    shard_rows = pq.read_table(shard_path).to_pylist()
    assert len(shard_rows) == 2
    assert [row["split"] for row in shard_rows] == ["train", "eval"]
    assert shard_rows[0]["episode_id"] == "agibot_alpha:327:648642"
    assert shard_rows[1]["episode_id"] == ("egodex:part2:2025-02-01_12-00-00.mov")
    assert all(len(row["video_mp4"]) > 100 for row in shard_rows)
    assert all(len(row["timestamp"]) == 16 for row in shard_rows)
    assert all(row["timestamp"][0] == 0 for row in shard_rows)
    assert all(19.9 < row["source_fps"] < 20.1 for row in shard_rows)

    pair_rows = {}
    window_rows = {}
    for split in ("train", "eval"):
        pair_rows[split] = pq.read_table(
            output / "stage1" / f"lam_pair_{split}.parquet"
        ).to_pylist()
        window_rows[split] = pq.read_table(
            output / "stage2" / f"wm_{split}_manifest.parquet"
        ).to_pylist()
        assert len(pair_rows[split]) == 3
        assert len(window_rows[split]) == 2
        assert all(row["split"] == split for row in pair_rows[split])
        assert all(row["split"] == split for row in window_rows[split])
        assert all(row["mask_policy"] == MASK_POLICY for row in pair_rows[split])
        assert all(not row["paper_equivalent_mask"] for row in pair_rows[split])
        assert all(
            not row["robosam_mask_training_eligible"] for row in pair_rows[split]
        )
        assert all(row["clip_nframes"] == 13 for row in window_rows[split])
        assert all("::" in row["video_path"] for row in window_rows[split])
    train_groups = {row["physical_group_key"] for row in pair_rows["train"]}
    eval_groups = {row["physical_group_key"] for row in pair_rows["eval"]}
    assert not train_groups & eval_groups

    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert len(provenance) == 3
    native_test = [row for row in provenance if row["part"] == "test"]
    assert len(native_test) == 1
    assert native_test[0]["embedded"] is False
    assert native_test[0]["exclusion_reason"] == "test_split"
    assert len(native_test[0]["source_video_sha256"]) == 64
    assert report["paper_recipe_complete"] is False
    assert report["sam3_masks_built"] is False
    assert report["counts"]["excluded_native_egodex_test_clips"] == 1
    assert report["counts"]["embedded_clips"] == 2

    shard_spec = importlib.util.find_spec("Scale.common.shard_io")
    assert shard_spec is not None
    from Scale.common.shard_io import decode_frames_from_mp4_bytes, read_shard_row

    first = read_shard_row(shard_path, 0)
    decoded = decode_frames_from_mp4_bytes(first["video_mp4"], [0, 12])
    assert sorted(decoded) == [0, 12]
    assert decoded[0].shape == (48, 64, 3)
