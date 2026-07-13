from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cd_lam.config import ConfigError, PipelineConfig, StageName, load_pipeline_config
from cd_lam.plans import build_stage_plan


def _write_config(path: Path, *, scope: str = "D") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: typed_test",
                "model_scale: 2B",
                "paths:",
                "  project_root: ..",
                "  data_root: data",
                "  artifact_root: artifacts",
                "  output_root: outputs",
                "stage1:",
                "  optimizer_steps: 1000",
                "  per_gpu_batch_size: 4",
                "stage2:",
                "  optimizer_steps: 2000",
                f"  training_scope: {scope}",
                "bridge_training:",
                "  optimizer_steps: 20",
                "  action_transform_id: minmax_delta",
                "  source_stride: 4",
                "stage3:",
                "  optimizer_steps: 3000",
                "  action_dim: 22",
                "  latent_dim: 32",
                "  action_transform_id: minmax_delta",
                "  source_stride: 4",
                "  working_directory: .",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_config_paths_resolve_from_config_directory_and_cli_override(tmp_path) -> None:
    config_path = _write_config(tmp_path / "configs" / "pipeline.yaml")
    config = load_pipeline_config(config_path)
    assert config.paths.project_root == tmp_path
    assert config.paths.data_root == tmp_path / "data"
    assert config.stage2.training_scope == "D"
    assert config.bridge.action_transform_id == config.stage3.action_transform_id

    override = tmp_path / "elsewhere"
    overridden = load_pipeline_config(config_path, project_root=override)
    assert overridden.paths.project_root == override.resolve()
    assert overridden.paths.data_root == (override / "data").resolve()


def test_config_digest_is_stable(tmp_path) -> None:
    config_path = _write_config(tmp_path / "configs" / "pipeline.yaml")
    assert (
        load_pipeline_config(config_path).digest
        == load_pipeline_config(config_path).digest
    )


def test_exported_path_profile_overrides_yaml_and_digest(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path / "configs" / "pipeline.yaml")
    baseline = load_pipeline_config(config_path)
    data_root = tmp_path / "external-data"
    stage1_lam = tmp_path / "checkpoints" / "stage1.pt"
    monkeypatch.setenv("CDLAM_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CDLAM_STAGE1_LAM", str(stage1_lam))

    configured = load_pipeline_config(config_path)

    assert configured.paths.data_root == data_root
    assert configured.paths.stage1_lam == stage1_lam
    assert configured.digest != baseline.digest


def test_stage2_rejects_nonproduction_scope(tmp_path) -> None:
    config_path = _write_config(tmp_path / "configs" / "pipeline.yaml", scope="B")
    with pytest.raises(ConfigError, match="training_scope must be 'D'"):
        load_pipeline_config(config_path)


def test_checkpoint_step_mismatch_is_an_explicit_plan_blocker(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs")
    mismatched = replace(
        config,
        stage1=replace(
            config.stage1,
            observed_checkpoint_steps=150,
            protocol_steps=1000,
            allow_partial_checkpoint=False,
        ),
    )
    plan = build_stage_plan(mismatched, StageName.STAGE1, synthetic=True)
    assert not plan.ready
    assert "150" in "\n".join(plan.blockers)
    assert "1000" in "\n".join(plan.blockers)


def test_partial_checkpoint_requires_explicit_opt_in(tmp_path) -> None:
    config = PipelineConfig.synthetic(tmp_path / "outputs")
    allowed = replace(
        config,
        stage1=replace(
            config.stage1,
            observed_checkpoint_steps=150,
            protocol_steps=1000,
            allow_partial_checkpoint=True,
        ),
    )
    assert build_stage_plan(allowed, StageName.STAGE1, synthetic=True).ready


def test_runtime_rejects_invalid_cuda_device(tmp_path) -> None:
    with pytest.raises(ConfigError, match="must select CUDA"):
        PipelineConfig.synthetic(tmp_path / "outputs", device="cuda:bad")
