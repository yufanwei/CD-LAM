"""Typed configuration loading for CD-LAM training commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .bridge import ACTION_DIM, LATENT_DIM


PATH_ENVIRONMENT_KEYS = {
    "project_root": "CDLAM_ROOT",
    "data_root": "CDLAM_DATA_ROOT",
    "artifact_root": "CDLAM_ARTIFACT_ROOT",
    "output_root": "CDLAM_OUTPUT_ROOT",
    "unlabeled_manifest": "CDLAM_UNLABELED_MANIFEST",
    "robot_action_manifest": "CDLAM_ROBOT_ACTION_MANIFEST",
    "base_acwm": "CDLAM_BASE_ACWM",
    "lam_init": "CDLAM_LAM_INIT",
    "stage1_lam": "CDLAM_STAGE1_LAM",
    "stage2_acwm": "CDLAM_STAGE2_ACWM",
    "bridge_bundle": "CDLAM_BRIDGE_BUNDLE",
    "stage3_acwm": "CDLAM_STAGE3_ACWM",
}


class ConfigError(ValueError):
    """Raised when a pipeline configuration violates the public schema."""


class StageName(str, Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    BRIDGE = "bridge-train"
    STAGE3 = "stage3"


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    data_root: Path
    artifact_root: Path
    output_root: Path
    unlabeled_manifest: Optional[Path] = None
    robot_action_manifest: Optional[Path] = None
    base_acwm: Optional[Path] = None
    lam_init: Optional[Path] = None
    stage1_lam: Optional[Path] = None
    stage2_acwm: Optional[Path] = None
    bridge_bundle: Optional[Path] = None
    stage3_acwm: Optional[Path] = None

    def with_updates(self, **updates: Optional[Path]) -> "PathConfig":
        return replace(self, **updates)


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    device: str = "cuda"


@dataclass(frozen=True)
class BaseStageConfig:
    name: str
    protocol_steps: int
    batch_size: int
    learning_rate: float
    adapter: Optional[str]
    observed_checkpoint_steps: Optional[int]
    allow_partial_checkpoint: bool
    resume_from: Optional[Path]
    resume_steps: Optional[int]


@dataclass(frozen=True)
class Stage1Config(BaseStageConfig):
    latent_dim: int = LATENT_DIM
    foreground_weight: float = 5.0
    background_weight: float = 1.0
    contrastive_weight: float = 0.05
    calibration_weight: float = 0.001
    free_bits: float = 0.5
    zero_margin: float = 0.05


@dataclass(frozen=True)
class Stage2Config(BaseStageConfig):
    latent_dim: int = LATENT_DIM
    condition: str = "debiased_latent_action"
    training_scope: str = "D"


@dataclass(frozen=True)
class BridgeTrainingConfig(BaseStageConfig):
    action_dim: int = ACTION_DIM
    latent_dim: int = LATENT_DIM
    action_transform_id: Optional[str] = None
    source_stride: Optional[int] = None


@dataclass(frozen=True)
class Stage3Config(BaseStageConfig):
    action_dim: int = ACTION_DIM
    latent_dim: int = LATENT_DIM
    action_transform_id: Optional[str] = None
    source_stride: Optional[int] = None
    working_directory: Optional[Path] = None


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    name: str
    model_scale: str
    paths: PathConfig
    runtime: RuntimeConfig
    stage1: Stage1Config
    stage2: Stage2Config
    bridge: BridgeTrainingConfig
    stage3: Stage3Config
    digest: str
    source: Optional[Path]

    def stage(self, name: StageName) -> BaseStageConfig:
        return {
            StageName.STAGE1: self.stage1,
            StageName.STAGE2: self.stage2,
            StageName.BRIDGE: self.bridge,
            StageName.STAGE3: self.stage3,
        }[name]

    def with_paths(self, **updates: Optional[Path]) -> "PipelineConfig":
        return replace(self, paths=self.paths.with_updates(**updates))

    @classmethod
    def synthetic(
        cls, output_root: Path, *, seed: int = 0, device: str = "cuda"
    ) -> "PipelineConfig":
        root = output_root.resolve().parent
        raw: dict[str, Any] = {
            "schema_version": 1,
            "name": f"cd_lam_synthetic_{device.replace(':', '_')}",
            "model_scale": "synthetic",
            "paths": {
                "project_root": str(root),
                "data_root": str(root / "data"),
                "artifact_root": str(root / "artifacts"),
                "output_root": str(output_root),
            },
            "runtime": {"seed": seed, "device": device},
            "stage1": {"optimizer_steps": 2, "per_gpu_batch_size": 4},
            "stage2": {
                "optimizer_steps": 2,
                "per_gpu_batch_size": 4,
                "training_scope": "D",
            },
            "bridge_training": {
                "optimizer_steps": 2,
                "batch_size": 8,
                "action_transform_id": "synthetic.minmax_delta",
                "source_stride": 1,
            },
            "stage3": {
                "optimizer_steps": 2,
                "action_dim": ACTION_DIM,
                "latent_dim": LATENT_DIM,
                "action_transform_id": "synthetic.minmax_delta",
                "source_stride": 1,
                "working_directory": str(root),
            },
        }
        return _parse_pipeline(raw, source=None, project_root=root)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _integer(
    mapping: Mapping[str, Any], key: str, default: Optional[int], *, minimum: int = 0
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}")
    return value


def _optional_integer(mapping: Mapping[str, Any], key: str) -> Optional[int]:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} must be a non-negative integer or null")
    return value


def _number(mapping: Mapping[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be numeric")
    value = float(value)
    if not (value > 0):
        raise ConfigError(f"{key} must be positive")
    return value


def _optional_string(mapping: Mapping[str, Any], key: str) -> Optional[str]:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string or null")
    return value.strip()


def _boolean(mapping: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be boolean")
    return value


def _resolve_path(
    value: Any, label: str, root: Path, *, optional: bool
) -> Optional[Path]:
    if value is None and optional:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        suffix = " or null" if optional else ""
        raise ConfigError(f"{label} must be a non-empty path{suffix}")
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ConfigError(
            f"{label} contains an unresolved environment variable: {value}"
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _base_stage(
    raw: Mapping[str, Any],
    *,
    label: str,
    default_name: str,
    default_steps: int,
    default_batch: int,
    default_lr: float,
    adapter: Optional[str],
    root: Path,
) -> dict[str, Any]:
    resume = _mapping(raw.get("resume", {}), f"{label}.resume")
    resume_value = resume.get("checkpoint", raw.get("resume_from"))
    return {
        "name": str(raw.get("name", default_name)),
        "protocol_steps": _integer(raw, "optimizer_steps", default_steps, minimum=1),
        "batch_size": _integer(
            raw,
            "per_gpu_batch_size" if "per_gpu_batch_size" in raw else "batch_size",
            default_batch,
            minimum=1,
        ),
        "learning_rate": _number(raw, "learning_rate", default_lr),
        "adapter": _optional_string(raw, "adapter") or adapter,
        "observed_checkpoint_steps": _optional_integer(
            raw, "observed_checkpoint_steps"
        ),
        "allow_partial_checkpoint": _boolean(raw, "allow_partial_checkpoint", False),
        "resume_from": _resolve_path(
            resume_value, f"{label}.resume.checkpoint", root, optional=True
        ),
        "resume_steps": _optional_integer(resume, "steps"),
    }


def _validate_fixed_dimensions(action_dim: int, latent_dim: int, label: str) -> None:
    if action_dim != ACTION_DIM:
        raise ConfigError(f"{label}.action_dim must be {ACTION_DIM}, got {action_dim}")
    if latent_dim != LATENT_DIM:
        raise ConfigError(f"{label}.latent_dim must be {LATENT_DIM}, got {latent_dim}")


def _canonical_digest(raw: Mapping[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply_path_environment(raw_value: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay explicitly exported asset paths onto a pipeline configuration."""

    raw = dict(raw_value)
    paths = dict(_mapping(raw.get("paths", {}), "paths"))
    for path_key, environment_key in PATH_ENVIRONMENT_KEYS.items():
        value = os.environ.get(environment_key)
        if value:
            paths[path_key] = value
    raw["paths"] = paths
    return raw


def _parse_pipeline(
    raw_value: Mapping[str, Any],
    *,
    source: Optional[Path],
    project_root: Optional[Path],
    project_root_override: bool = False,
) -> PipelineConfig:
    raw = _mapping(raw_value, "pipeline")
    schema_version = _integer(raw, "schema_version", None, minimum=1)
    if schema_version != 1:
        raise ConfigError(f"schema_version must be 1, got {schema_version}")
    name = raw.get("name")
    model_scale = raw.get("model_scale")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("name must be a non-empty string")
    if not isinstance(model_scale, str) or not model_scale.strip():
        raise ConfigError("model_scale must be a non-empty string")

    paths_raw = _mapping(raw.get("paths", {}), "paths")
    cwd_root = Path.cwd().resolve() if project_root is None else project_root.resolve()
    configured_root = paths_raw.get("project_root")
    if project_root_override:
        root = cwd_root
    else:
        root = (
            _resolve_path(
                configured_root, "paths.project_root", cwd_root, optional=False
            )
            if configured_root is not None
            else cwd_root
        )
    assert root is not None
    paths = PathConfig(
        project_root=root,
        data_root=_resolve_path(
            paths_raw.get("data_root", "data"), "paths.data_root", root, optional=False
        ),  # type: ignore[arg-type]
        artifact_root=_resolve_path(
            paths_raw.get("artifact_root", "artifacts"),
            "paths.artifact_root",
            root,
            optional=False,
        ),  # type: ignore[arg-type]
        output_root=_resolve_path(
            paths_raw.get("output_root", "outputs"),
            "paths.output_root",
            root,
            optional=False,
        ),  # type: ignore[arg-type]
        unlabeled_manifest=_resolve_path(
            paths_raw.get("unlabeled_manifest"),
            "paths.unlabeled_manifest",
            root,
            optional=True,
        ),
        robot_action_manifest=_resolve_path(
            paths_raw.get("robot_action_manifest"),
            "paths.robot_action_manifest",
            root,
            optional=True,
        ),
        base_acwm=_resolve_path(
            paths_raw.get("base_acwm"), "paths.base_acwm", root, optional=True
        ),
        lam_init=_resolve_path(
            paths_raw.get("lam_init"), "paths.lam_init", root, optional=True
        ),
        stage1_lam=_resolve_path(
            paths_raw.get("stage1_lam", paths_raw.get("debiased_lam")),
            "paths.stage1_lam",
            root,
            optional=True,
        ),
        stage2_acwm=_resolve_path(
            paths_raw.get("stage2_acwm"), "paths.stage2_acwm", root, optional=True
        ),
        bridge_bundle=_resolve_path(
            paths_raw.get("bridge_bundle"), "paths.bridge_bundle", root, optional=True
        ),
        stage3_acwm=_resolve_path(
            paths_raw.get("stage3_acwm"), "paths.stage3_acwm", root, optional=True
        ),
    )

    runtime_raw = _mapping(raw.get("runtime", {}), "runtime")
    runtime = RuntimeConfig(
        seed=_integer(runtime_raw, "seed", 0, minimum=0),
        device=str(runtime_raw.get("device", "cuda")),
    )
    if re.fullmatch(r"cuda(?::[0-9]+)?", runtime.device) is None:
        raise ConfigError("runtime.device must select CUDA")
    adapters = _mapping(raw.get("adapters", {}), "adapters")

    stage1_raw = _mapping(raw.get("stage1", {}), "stage1")
    stage1_base = _base_stage(
        stage1_raw,
        label="stage1",
        default_name="lam_debiased_finetuning",
        default_steps=1000,
        default_batch=32,
        default_lr=1e-4,
        adapter=_optional_string(adapters, "stage1"),
        root=root,
    )
    objectives = _mapping(stage1_raw.get("objectives", {}), "stage1.objectives")
    emb = _mapping(
        objectives.get("embodiment_centric_reconstruction", {}),
        "stage1.objectives.embodiment_centric_reconstruction",
    )
    cal = _mapping(
        objectives.get("latent_space_calibration", {}),
        "stage1.objectives.latent_space_calibration",
    )
    stage1 = Stage1Config(
        **stage1_base,
        latent_dim=_integer(stage1_raw, "latent_dim", LATENT_DIM, minimum=1),
        foreground_weight=_number(emb, "foreground_weight", 5.0),
        background_weight=_number(emb, "background_weight", 1.0),
        contrastive_weight=_number(stage1_raw, "contrastive_weight", 0.05),
        calibration_weight=_number(stage1_raw, "calibration_weight", 0.001),
        free_bits=_number(cal, "free_bits", 0.5),
        zero_margin=_number(cal, "zero_margin", 0.05),
    )
    if stage1.latent_dim != LATENT_DIM:
        raise ConfigError(f"stage1.latent_dim must be {LATENT_DIM}")
    if stage1.foreground_weight <= stage1.background_weight:
        raise ConfigError("stage1 foreground_weight must exceed background_weight")

    stage2_raw = _mapping(raw.get("stage2", {}), "stage2")
    stage2 = Stage2Config(
        **_base_stage(
            stage2_raw,
            label="stage2",
            default_name="acwm_debiased_finetuning",
            default_steps=2000,
            default_batch=12,
            default_lr=1e-5,
            adapter=_optional_string(adapters, "stage2"),
            root=root,
        ),
        latent_dim=_integer(stage2_raw, "latent_dim", LATENT_DIM, minimum=1),
        condition=str(stage2_raw.get("condition", "debiased_latent_action")),
        training_scope=str(stage2_raw.get("training_scope", "D")),
    )
    if stage2.latent_dim != LATENT_DIM:
        raise ConfigError(f"stage2.latent_dim must be {LATENT_DIM}")
    if stage2.training_scope != "D":
        raise ConfigError(
            "stage2.training_scope must be 'D' for production CD-LAM training"
        )

    bridge_raw = _mapping(raw.get("bridge_training", {}), "bridge_training")
    legacy_bridge = _mapping(
        _mapping(raw.get("stage3", {}), "stage3").get("bridge", {}), "stage3.bridge"
    )
    merged_bridge = dict(legacy_bridge)
    merged_bridge.update(bridge_raw)
    bridge_action_dim = _integer(merged_bridge, "action_dim", ACTION_DIM, minimum=1)
    bridge_latent_dim = _integer(merged_bridge, "latent_dim", LATENT_DIM, minimum=1)
    _validate_fixed_dimensions(bridge_action_dim, bridge_latent_dim, "bridge_training")
    bridge = BridgeTrainingConfig(
        **_base_stage(
            merged_bridge,
            label="bridge_training",
            default_name="bridge_training",
            default_steps=1000,
            default_batch=32,
            default_lr=1e-3,
            adapter=_optional_string(adapters, "bridge")
            or _optional_string(adapters, "bridge_train"),
            root=root,
        ),
        action_dim=bridge_action_dim,
        latent_dim=bridge_latent_dim,
        action_transform_id=_optional_string(merged_bridge, "action_transform_id"),
        source_stride=_optional_integer(merged_bridge, "source_stride"),
    )

    stage3_raw = _mapping(raw.get("stage3", {}), "stage3")
    action_dim = _integer(stage3_raw, "action_dim", ACTION_DIM, minimum=1)
    latent_dim = _integer(stage3_raw, "latent_dim", LATENT_DIM, minimum=1)
    _validate_fixed_dimensions(action_dim, latent_dim, "stage3")
    stage3 = Stage3Config(
        **_base_stage(
            stage3_raw,
            label="stage3",
            default_name="robot_action_adaptation",
            default_steps=3000 if model_scale == "2B" else 6000,
            default_batch=4,
            default_lr=2.5e-5,
            adapter=_optional_string(adapters, "stage3"),
            root=root,
        ),
        action_dim=action_dim,
        latent_dim=latent_dim,
        action_transform_id=_optional_string(stage3_raw, "action_transform_id"),
        source_stride=_optional_integer(stage3_raw, "source_stride"),
        working_directory=_resolve_path(
            stage3_raw.get("working_directory"),
            "stage3.working_directory",
            root,
            optional=True,
        ),
    )

    return PipelineConfig(
        schema_version=schema_version,
        name=name.strip(),
        model_scale=model_scale.strip(),
        paths=paths,
        runtime=runtime,
        stage1=stage1,
        stage2=stage2,
        bridge=bridge,
        stage3=stage3,
        digest=_canonical_digest(raw),
        source=source,
    )


def load_pipeline_config(
    path: Path | str, *, project_root: Optional[Path] = None
) -> PipelineConfig:
    """Load a JSON/YAML pipeline with deterministic path resolution."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(
            f"failed to parse configuration {config_path}: {exc}"
        ) from exc
    resolution_root = config_path.parent if project_root is None else project_root
    effective = _apply_path_environment(_mapping(payload, "pipeline"))
    return _parse_pipeline(
        effective,
        source=config_path,
        project_root=resolution_root,
        project_root_override=project_root is not None,
    )


__all__ = [
    "BaseStageConfig",
    "BridgeTrainingConfig",
    "ConfigError",
    "PathConfig",
    "PATH_ENVIRONMENT_KEYS",
    "PipelineConfig",
    "RuntimeConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "StageName",
    "load_pipeline_config",
]
