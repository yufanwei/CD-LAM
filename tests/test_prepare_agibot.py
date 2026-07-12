from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load_module(
    "cdlam_prepare_agibot_candidate",
    ROOT / "scripts" / "prepare_agibot.py",
)


def test_dry_run_builds_complete_data_chain(tmp_path: Path) -> None:
    args = PREPARE.parse_args(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(tmp_path / "prepared"),
            "--max-episodes",
            "8",
            "--source-revision",
            "pinned-revision",
            "--dry-run",
        ]
    )

    paths, commands = PREPARE.build_commands(
        args,
        tmp_path / "repo",
        Path("/usr/bin/python3"),
    )

    assert paths.segmented == (tmp_path / "prepared" / "segmented").resolve()
    assert paths.stage12 == (tmp_path / "prepared" / "stage12").resolve()
    assert len(commands) == 5
    assert commands[0][1].endswith("scripts/materialize_agibot_alpha.py")
    assert commands[1][1].endswith("internal/runtime/convert_agibot_alpha.py")
    assert commands[2][1].endswith("scripts/index_agibot_segments.py")
    assert commands[3][1].endswith("internal/runtime/build_raw_subset.py")
    assert commands[4][1].endswith("Scale/common/build_alpha_bridge_cache.py")
    assert commands[0][-4:] == [
        "--max-episodes",
        "8",
        "--source-revision",
        "pinned-revision",
    ]
    assert commands[1][commands[1].index("--splits") + 1] == "train"
    assert commands[1][commands[1].index("--held-out-tasks") + 1] == "0"
    assert commands[1][commands[1].index("--eval-percent") + 1] == "0"
    assert commands[3][commands[3].index("--max-clips") + 1] == "32"
    assert commands[4][commands[4].index("--n-episodes") + 1] == "0"


def _write_episode_rows(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / "train" / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_bridge_preflight_counts_physical_episodes_not_segments(
    tmp_path: Path,
) -> None:
    train_id = next(
        f"agibot_alpha:327:{episode}"
        for episode in range(1000)
        if PREPARE._bridge_split(f"agibot_alpha:327:{episode}", 0.5) == "train"
    )
    eval_id = next(
        f"agibot_alpha:327:{episode}"
        for episode in range(1000)
        if PREPARE._bridge_split(f"agibot_alpha:327:{episode}", 0.5) == "eval"
    )
    _write_episode_rows(
        tmp_path,
        [
            {"length": 80, "physical_episode_id": train_id},
            {"length": 70, "physical_episode_id": train_id},
            {"length": 60, "physical_episode_id": eval_id},
            {"length": 10, "physical_episode_id": "agibot_alpha:327:short"},
        ],
    )

    result = PREPARE.bridge_split_preflight(tmp_path, 50, 0.5)

    assert result == {
        "eligible_segments": 3,
        "physical_episodes": 2,
        "train_physical_episodes": 1,
        "eval_physical_episodes": 1,
    }


def test_bridge_preflight_explains_small_episode_bound(tmp_path: Path) -> None:
    _write_episode_rows(
        tmp_path,
        [
            {
                "length": 80,
                "physical_episode_id": "agibot_alpha:327:648642",
            },
            {
                "length": 70,
                "physical_episode_id": "agibot_alpha:327:648642",
            },
        ],
    )

    with pytest.raises(
        PREPARE.PreparationError,
        match="Increase --max-episodes.*complete physical episodes",
    ):
        PREPARE.bridge_split_preflight(tmp_path, 50, 0.12)


@pytest.mark.parametrize("direction", ["raw-inside-output", "output-inside-raw"])
def test_raw_and_output_roots_are_rejected_before_mutation(
    tmp_path: Path,
    direction: str,
) -> None:
    if direction == "raw-inside-output":
        output = tmp_path / "work"
        raw = output / "raw"
    else:
        raw = tmp_path / "raw"
        output = raw / "work"
    raw.mkdir(parents=True)
    sentinel = raw / "source-must-survive"
    sentinel.write_text("source", encoding="utf-8")

    with pytest.raises(PREPARE.PreparationError, match="must be disjoint"):
        PREPARE._require_disjoint_roots(raw, output)

    assert sentinel.read_text(encoding="utf-8") == "source"


@pytest.mark.parametrize("raw_inside_output", [True, False])
def test_overlapping_roots_are_rejected_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_inside_output: bool,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "internal" / "vendor" / "scale_support").mkdir(parents=True)
    if raw_inside_output:
        output_root = tmp_path / "prepared"
        raw_root = output_root / "raw"
    else:
        raw_root = tmp_path / "raw"
        output_root = raw_root / "prepared"
    raw_root.mkdir(parents=True)
    calls: list[Path] = []

    def forbidden_prepare(path: Path, overwrite: bool) -> None:
        calls.append(path)
        raise AssertionError("output mutation must not be reached")

    monkeypatch.setattr(PREPARE, "_prepare_output", forbidden_prepare)
    args = PREPARE.parse_args(
        [
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "--repo-root",
            str(repo_root),
            "--overwrite",
        ]
    )

    with pytest.raises(PREPARE.PreparationError, match="must not equal or contain"):
        PREPARE.run(args)
    assert calls == []
