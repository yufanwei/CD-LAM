from __future__ import annotations

from pathlib import Path

from cd_lam.__main__ import main


ROOT = Path(__file__).resolve().parents[1]


def test_doctor_strict_bootstrap_does_not_require_weights(capsys) -> None:
    assert main(["doctor", "--strict"]) == 0
    output = capsys.readouterr().out
    assert "bootstrap strict mode" in output
    assert "CD-LAM doctor: PASS" in output


def test_doctor_strict_checks_only_explicitly_configured_assets(tmp_path, capsys) -> None:
    config = tmp_path / "profile.json"
    config.write_text(
        '{"paths": {"data_root": "data", "lam_init": null, '
        '"robot_action_manifest": "missing.json"}}',
        encoding="utf-8",
    )
    assert main(["doctor", "--config", str(config)]) == 0
    assert "optional configured assets missing" in capsys.readouterr().out

    assert main(["doctor", "--strict", "--config", str(config)]) == 1
    output = capsys.readouterr().out
    assert "robot_action_manifest" in output
    assert "data_root" not in output
    assert "lam_init" not in output


def test_doctor_strict_accepts_public_null_asset_configs(capsys) -> None:
    for name in ("pipeline_100h_2b.yaml", "pipeline_100h_14b.yaml"):
        assert main(
            ["doctor", "--strict", "--config", str(ROOT / "configs" / name)]
        ) == 0
        output = capsys.readouterr().out
        assert "no local full-runtime assets are configured" in output
        assert "CD-LAM doctor: PASS" in output


def test_doctor_strict_enforces_populated_public_asset_keys(tmp_path, capsys) -> None:
    config = tmp_path / "populated.json"
    config.write_text(
        '{"paths": {'
        '"data_root": "data", "artifact_root": "artifacts", '
        '"output_root": "outputs", '
        '"unlabeled_manifest": "missing-unlabeled.json", '
        '"robot_action_manifest": "missing-robot.json", '
        '"base_acwm": "missing-base.pt", '
        '"lam_init": "missing-lam.pt"}}',
        encoding="utf-8",
    )

    assert main(["doctor", "--strict", "--config", str(config)]) == 1
    output = capsys.readouterr().out
    for required in (
        "unlabeled_manifest",
        "robot_action_manifest",
        "base_acwm",
        "lam_init",
    ):
        assert f"[missing] paths.{required}" in output
    for routing_root in ("data_root", "artifact_root", "output_root"):
        assert f"[missing] paths.{routing_root}" not in output


def test_doctor_resolves_profile_paths_from_working_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    (tmp_path / "data").mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "profile.json"
    config.write_text('{"paths": {"data_root": "data"}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["doctor", "--strict", "--config", str(config)]) == 0
    assert str(tmp_path / "data") in capsys.readouterr().out


def test_smoke_cli(capsys) -> None:
    assert main(["smoke"]) == 0
    assert "CD-LAM smoke: PASS" in capsys.readouterr().out


def test_blocked_training_dry_run_returns_nonzero_and_prints_json(capsys) -> None:
    code = main(
        [
            "stage1",
            "--config",
            str(ROOT / "configs" / "pipeline_100h_2b.yaml"),
            "--dry-run",
            "--json",
        ]
    )
    assert code == 2
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["stage"] == "stage1"
    assert payload["ready"] is False
    assert payload["blockers"]
