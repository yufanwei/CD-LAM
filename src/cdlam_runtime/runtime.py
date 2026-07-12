"""Command construction and fail-closed validation for the CD-LAM pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from cdlam_runtime.action_contract import load_stage3_action_contract
from cdlam_runtime.config import ConfigError, RuntimeConfig
from cdlam_runtime.resolve_stage1 import resolve_config
from cdlam_runtime.validate_eval_source import validate_source as validate_eval_source
from cdlam_runtime.validate_stage2_source import (
    validate_source as validate_stage2_source,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
SUPPORT_ROOT = PACKAGE_ROOT / "support"
ENTRY_ROOT = PACKAGE_ROOT / "entries"
STAGES = ("stage1", "bridge", "stage2", "stage3")
LAM_ID = "cdlam_lam"
BASE_LAM_ID = "base_lam"
BRIDGE_LINEAGE = "100h"

# These are upstream Hydra registry keys. They are intentionally internal to
# the adapter and never appear in the public profile or command-line surface.
_PRETRAIN_EXPERIMENT = "cdlam_pretrain"
_POSTTRAIN_EXPERIMENT = "cdlam_posttrain"
_TRAIN_CONFIG = (
    "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
)


class RuntimeError(ConfigError):
    """Raised when a launch contract cannot be satisfied."""


@dataclass(frozen=True)
class LaunchCommand:
    """One subprocess in a stage launch."""

    label: str
    argv: tuple[str, ...]

    def display(self) -> str:
        return shlex.join(self.argv)


def _required_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"missing or empty {label}: {path}")


def _required_directory(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"missing {label}: {path}")


def _required_executable(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        errors.append(f"missing executable {label}: {path}")


def _external_files(config: RuntimeConfig, stage: str) -> list[tuple[str, Path]]:
    acwm = config.path("acwm_root")
    assert acwm is not None
    common = [
        ("ACWM package", acwm / "cosmos_predict2" / "__init__.py"),
        (
            "Cosmos CUDA package",
            acwm / "packages" / "cosmos-cuda" / "cosmos_cuda" / "__init__.py",
        ),
    ]
    per_stage = {
        "stage1": [
            (
                "Stage-1 trainer",
                acwm / "training_scope" / "LAM" / "V5" / "tools" / "train_lam_v5.py",
            ),
            (
                "Stage-1 data adapter",
                acwm / "training_scope" / "LAM" / "V3" / "tools" / "_lam_v3_data.py",
            ),
            (
                "Stage-1 mask adapter",
                acwm / "training_scope" / "LAM" / "V7" / "tools" / "train_v7_cached.py",
            ),
            (
                "Stage-1 transforms",
                acwm / "finetune_4-30" / "scripts" / "transforms.py",
            ),
        ],
        "bridge": [
            (
                "bridge trainer",
                acwm / "New LAM" / "iterations" / "_dist" / "train_a22z_frozen.py",
            ),
            (
                "bridge alias adapter",
                acwm / "training_scope" / "LAM" / "V7" / "tools" / "_lam_alias_shim.py",
            ),
            (
                "bridge encoder",
                acwm
                / "training_scope"
                / "LAM"
                / "V7"
                / "tools"
                / "probe_idm_upper_bound.py",
            ),
            (
                "bridge context",
                acwm / "training_scope" / "LAM" / "V7" / "tools" / "_lam_v7_ctx.py",
            ),
            (
                "bridge utilities",
                acwm
                / "training_scope"
                / "LAM"
                / "V7"
                / "tools"
                / "probe_ee_vs_joint.py",
            ),
        ],
        "stage2": [
            (
                "Stage-2 trainer",
                acwm / "lamwm_pipline" / "tools" / "train_wm_compat_real.py",
            ),
            ("LAM registry loader", acwm / "lamwm_pipline" / "src" / "registry.py"),
            ("LAM model loader", acwm / "lamwm_pipline" / "src" / "lam_loader.py"),
            ("Stage-2 experiment", acwm / "configs" / "2b_480_640_pretrain.yaml"),
        ],
        "stage3": [
            (
                "Stage-3 trainer",
                acwm / "New LAM" / "Post Train" / "train_gbridge_z_posttrain.py",
            ),
            ("Stage-3 experiment", acwm / "configs" / "2b_480_640_agibot.yaml"),
            ("AgiBot statistics", acwm / "shared_meta" / "AgiBot_stats.json"),
            ("AgiBot modality", acwm / "shared_meta" / "AgiBot_modality.json"),
        ],
    }
    return common + per_stage[stage]


def _validate_public_experiment_aliases(acwm: Path, stages: Sequence[str]) -> None:
    aliases: list[str] = []
    if "stage2" in stages:
        aliases.append(_PRETRAIN_EXPERIMENT)
    if "stage3" in stages:
        aliases.append(_POSTTRAIN_EXPERIMENT)
    if not aliases:
        return
    source = acwm / "cosmos_predict2" / "experiments" / "base" / "action.py"
    text = source.read_text(encoding="utf-8")
    missing = [alias for alias in aliases if json.dumps(alias) not in text]
    if missing:
        raise RuntimeError(
            "staged ACWM source does not register the public experiment aliases: "
            + ", ".join(missing)
        )


def stage_checkpoint(config: RuntimeConfig, stage: str) -> Path:
    mapping = {
        "stage1": "stage1_checkpoint",
        "bridge": "bridge_checkpoint",
        "stage2": "stage2_checkpoint",
    }
    try:
        key = mapping[stage]
    except KeyError as exc:
        raise RuntimeError(f"unsupported checkpoint stage: {stage}") from exc
    path = config.path(key)
    assert path is not None
    return path


def output_dir(config: RuntimeConfig, stage: str) -> Path:
    root = config.path("output_root")
    assert root is not None
    return root / stage


def generated_registry(config: RuntimeConfig) -> Path:
    root = config.path("output_root")
    assert root is not None
    return root / ".runtime" / "lam_registry.yaml"


def write_registry(config: RuntimeConfig, checkpoint: Path) -> Path:
    """Write the two-entry LAM registry consumed by Stage 2."""

    base = config.path("base_lam_checkpoint")
    assert base is not None
    document = {
        "lams": {
            BASE_LAM_ID: {
                "ckpt_path": str(base),
                "ckpt_format": "official",
                "output_dim": 32,
                "parent": "none",
                "status": "base",
                "allowed_for_wm_control": True,
            },
            LAM_ID: {
                "ckpt_path": str(checkpoint),
                "ckpt_format": "finetune_ddp",
                "base_ckpt": BASE_LAM_ID,
                "output_dim": 32,
                "parent": BASE_LAM_ID,
                "status": "qualified_for_wm",
                "allowed_for_wm_control": True,
            },
        }
    }
    path = generated_registry(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def runtime_environment(config: RuntimeConfig, lam_checkpoint: Path) -> dict[str, str]:
    """Build the subprocess environment without legacy runtime namespaces."""

    acwm = config.path("acwm_root")
    base_lam = config.path("base_lam_checkpoint")
    base_world = config.path("base_world_checkpoint")
    action_contract = config.path("action_contract")
    hf_home = config.path("hf_home")
    assert acwm and base_lam and base_world and action_contract and hf_home

    env = os.environ.copy()
    env.update(
        {
            "CDLAM_ACWM_ROOT": str(acwm),
            "CDLAM_PYTHON_ROOT": str(acwm.parent),
            "CDLAM_LAM_REGISTRY_DIR": str(
                write_registry(config, lam_checkpoint).parent
            ),
            "CDLAM_STAGE3_ACTION_CONTRACT": str(action_contract),
            "CDLAM_SCALE_FAMILY": BRIDGE_LINEAGE,
            "LAM_400K_LOCAL": str(base_lam),
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "COSMOS_SKIP_DEFAULT_CHECKPOINT_DOWNLOADS": "1",
            "COSMOS_SKIP_14B_EXPERIMENT_CONFIGS": "1",
            "COSMOS_LOCAL_2B_CHECKPOINT": str(base_world),
        }
    )
    path_entries = [
        SOURCE_ROOT,
        SUPPORT_ROOT,
        acwm,
        acwm / "packages" / "cosmos-cuda",
        acwm / "scripts",
        acwm / "training_scope" / "LAM",
        acwm / "finetune_4-30" / "scripts",
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        path_entries.append(Path(existing))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in path_entries)
    return env


def _base_world_errors(config: RuntimeConfig) -> list[str]:
    checkpoint = config.path("base_world_checkpoint")
    assert checkpoint is not None
    errors: list[str] = []
    _required_directory(checkpoint, "2B base world-model checkpoint", errors)
    if checkpoint.is_dir():
        _required_file(checkpoint / ".metadata", "2B checkpoint metadata", errors)
        if not any(checkpoint.glob("*.distcp")):
            errors.append(f"2B checkpoint has no .distcp shards: {checkpoint}")
    return errors


def _hf_errors(config: RuntimeConfig) -> list[str]:
    hf_home = config.path("hf_home")
    assert hf_home is not None
    errors: list[str] = []
    _required_directory(hf_home, "Hugging Face cache", errors)
    if hf_home.is_dir():
        tokenizers = list(
            hf_home.glob(
                "hub/models--nvidia--Cosmos-Predict*/snapshots/*/tokenizer.pth"
            )
        )
        reason_indices = list(
            hf_home.glob(
                "hub/models--nvidia--Cosmos-Reason*/snapshots/*/model.safetensors.index.json"
            )
        )
        if not tokenizers:
            errors.append(f"Cosmos tokenizer is missing under {hf_home / 'hub'}")
        if not reason_indices:
            errors.append(f"Cosmos text encoder is missing under {hf_home / 'hub'}")
    return errors


def _validate_contract_assets(config: RuntimeConfig, acwm: Path) -> None:
    contract_path = config.path("action_contract")
    assert contract_path is not None
    document = json.loads(contract_path.read_text(encoding="utf-8"))
    for path_key, digest_key, label in (
        ("stats_relative_path", "stats_sha256", "AgiBot statistics"),
        ("modality_relative_path", "modality_sha256", "AgiBot modality metadata"),
    ):
        relative = document.get(path_key)
        expected = document.get(digest_key)
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError(
                f"action contract has no valid {path_key} or {digest_key}"
            )
        source = acwm / relative
        if not source.is_file():
            raise RuntimeError(f"{label} are missing: {source}")
        observed = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"{label} SHA256 does not match the action contract")


def _runtime_import_modules(stages: Sequence[str]) -> list[str]:
    """Return the CPU-safe import closure exercised by the runtime doctor."""

    modules = ["torch", "numpy", "yaml"]
    if "stage1" in stages:
        modules.extend(
            [
                "cdlam_runtime.entries.stage1",
                "cdlam_runtime.entries.stage1_eval",
                "external.lam.model",
            ]
        )
    if "stage2" in stages or "stage3" in stages:
        modules.extend(["cosmos_cuda", "cosmos_predict2", "groot_dreams.dataloader"])
    if "stage2" in stages:
        modules.extend(
            [
                "cdlam_runtime.entries.stage2",
                "lamwm_pipline.tools.train_wm_compat_real",
            ]
        )
    if "stage3" in stages:
        modules.append("cdlam_runtime.entries.stage3")
    return modules


def doctor(
    config: RuntimeConfig,
    stage: str,
    *,
    imports: bool = True,
    lam_checkpoint: Path | None = None,
    bridge_checkpoint: Path | None = None,
    stage2_checkpoint: Path | None = None,
) -> list[str]:
    """Return every preflight error for one stage or the full pipeline."""

    if stage not in (*STAGES, "all"):
        raise RuntimeError(f"unsupported doctor stage: {stage}")
    stages = STAGES if stage == "all" else (stage,)
    from_scratch = stage == "all"
    errors: list[str] = []
    python = config.path("python")
    torchrun = config.path("torchrun")
    acwm = config.path("acwm_root")
    assert python and torchrun and acwm
    _required_executable(python, "Python", errors)
    _required_executable(torchrun, "torchrun", errors)
    _required_directory(acwm, "staged ACWM source", errors)

    selected_lam = lam_checkpoint or stage_checkpoint(config, "stage1")
    selected_bridge = bridge_checkpoint or stage_checkpoint(config, "bridge")
    selected_stage2 = stage2_checkpoint or stage_checkpoint(config, "stage2")
    for selected_stage in stages:
        for label, path in _external_files(config, selected_stage):
            _required_file(path, label, errors)
        if selected_stage == "stage1":
            for key, label in (
                ("base_lam_checkpoint", "base LAM checkpoint"),
                ("stage1_recipe", "Stage-1 recipe"),
                ("stage1_train_index", "Stage-1 training index"),
            ):
                path = config.path(key)
                assert path is not None
                _required_file(path, label, errors)
            if not config.optional_value("stage1", "skip_eval", bool, False):
                eval_index = config.path("stage1_eval_index")
                assert eval_index is not None
                _required_file(eval_index, "Stage-1 evaluation index", errors)
        elif selected_stage == "bridge":
            cache = config.path("bridge_cache")
            assert cache is not None
            _required_file(cache, "bridge cache", errors)
            if not from_scratch:
                _required_file(selected_lam, "Stage-1 checkpoint", errors)
        elif selected_stage == "stage2":
            manifest = config.path("stage2_manifest")
            assert manifest is not None
            _required_file(manifest, "Stage-2 manifest", errors)
            if not from_scratch:
                _required_file(selected_lam, "Stage-1 checkpoint", errors)
            errors.extend(_base_world_errors(config))
            errors.extend(_hf_errors(config))
        elif selected_stage == "stage3":
            for key, label in (
                ("stage3_dataset_list", "Stage-3 dataset list"),
                ("action_contract", "Stage-3 action contract"),
            ):
                path = config.path(key)
                assert path is not None
                _required_file(path, label, errors)
            if not from_scratch:
                _required_file(selected_bridge, "bridge checkpoint", errors)
                _required_file(
                    selected_stage2,
                    "Stage-2 initialization checkpoint",
                    errors,
                )
            errors.extend(_base_world_errors(config))
            errors.extend(_hf_errors(config))

    if not errors:
        try:
            _validate_public_experiment_aliases(acwm, stages)
            if "stage1" in stages:
                validate_eval_source(acwm)
            if "stage2" in stages:
                validate_stage2_source(acwm)
            if "stage3" in stages:
                contract = config.path("action_contract")
                assert contract
                if from_scratch:
                    _validate_contract_assets(config, acwm)
                else:
                    load_stage3_action_contract(
                        contract,
                        acwm,
                        selected_bridge,
                        BRIDGE_LINEAGE,
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"source or action contract validation failed: {exc}")

    if imports and not errors:
        env = runtime_environment(config, selected_lam)
        env["CUDA_VISIBLE_DEVICES"] = ""
        modules = _runtime_import_modules(stages)
        scripts: list[Path] = []
        if "bridge" in stages:
            scripts.append(
                acwm / "New LAM" / "iterations" / "_dist" / "train_a22z_frozen.py"
            )
        if "stage3" in stages:
            scripts.append(
                acwm / "New LAM" / "Post Train" / "train_gbridge_z_posttrain.py"
            )
        code_lines = ["import importlib", "import runpy"]
        code_lines.extend(
            f"importlib.import_module({json.dumps(name)})" for name in modules
        )
        code_lines.extend(
            f"runpy.run_path({json.dumps(str(path))})" for path in scripts
        )
        code = "; ".join(code_lines)
        result = subprocess.run(
            [str(python), "-c", code],
            cwd=acwm,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            errors.append(f"runtime import probe failed: {detail}")
    return errors


def _torchrun_prefix(config: RuntimeConfig, port: int) -> list[str]:
    torchrun = config.path("torchrun")
    assert torchrun is not None
    processes = config.positive_int("launch", "processes", 1)
    return [
        str(torchrun),
        "--standalone",
        f"--nproc_per_node={processes}",
        f"--master_port={port}",
    ]


def _stage1_commands(config: RuntimeConfig) -> list[LaunchCommand]:
    recipe = config.path("stage1_recipe")
    train_index = config.path("stage1_train_index")
    eval_index = config.path("stage1_eval_index", required=False)
    assert recipe and train_index
    out = output_dir(config, "stage1")
    resolved = out / "config.resolved.yaml"
    resolve_config(
        recipe,
        train_index,
        resolved,
        eval_pair_index=eval_index,
        overrides=config.table("stage1").get("evaluation", {}),
    )
    steps = config.positive_int("stage1", "steps", 1000)
    skip_eval = config.optional_value("stage1", "skip_eval", bool, False)
    train = _torchrun_prefix(config, 29593) + [
        str(ENTRY_ROOT / "stage1.py"),
        "--config",
        str(resolved),
        "--out",
        str(out),
        "--total-steps",
        str(steps),
    ]
    if skip_eval:
        train.append("--skip-eval")
    python = config.path("python")
    assert python is not None
    validate = [
        str(python),
        "-m",
        "cdlam_runtime.validate_stage1",
        "--root",
        str(out),
        "--expected-steps",
        str(steps),
    ]
    if not skip_eval:
        validate.append("--require-eval")
    if config.optional_value("validation", "smoke", bool, False):
        validate.append("--smoke")
    return [
        LaunchCommand("Stage 1 train", tuple(train)),
        LaunchCommand("Stage 1 validate", tuple(validate)),
    ]


def _bridge_commands(
    config: RuntimeConfig, lam_checkpoint: Path
) -> list[LaunchCommand]:
    python = config.path("python")
    acwm = config.path("acwm_root")
    cache = config.path("bridge_cache")
    assert python and acwm and cache
    out = output_dir(config, "bridge")
    argv = [
        str(python),
        str(acwm / "New LAM" / "iterations" / "_dist" / "train_a22z_frozen.py"),
        "--cache",
        str(cache),
        "--action-key",
        "action_22",
        "--robot",
        "agibot_alpha",
        "--lam-ckpt",
        str(lam_checkpoint),
        "--l-enc",
        "1.0",
        "--l-cyc",
        "3.0",
        "--l-cycz",
        "0.0",
        "--l-dec",
        "0.3",
        "--l-mmd",
        "0.0",
        "--tag",
        "_D_cyc3",
        "--n-pairs",
        str(config.nonnegative_int("bridge", "pairs", 0)),
        "--epochs",
        str(config.positive_int("bridge", "epochs", 300)),
        "--enc-bs",
        str(config.positive_int("bridge", "encoder_batch_size", 64)),
        "--out",
        str(out),
    ]
    z_cache = config.path("bridge_z_cache", required=False)
    if z_cache:
        argv.extend(["--z-cache", str(z_cache)])
    checkpoint = out / "a22z_agibot_alpha_D_cyc3.pt"
    contract = config.path("action_contract")
    assert contract is not None
    bind = [
        str(python),
        "-m",
        "cdlam_runtime.bind_bridge",
        "--checkpoint",
        str(checkpoint),
        "--contract",
        str(contract),
        "--external-root",
        str(acwm),
        "--stage1-checkpoint",
        str(lam_checkpoint),
        "--lineage",
        BRIDGE_LINEAGE,
    ]
    validate = [
        str(python),
        "-m",
        "cdlam_runtime.validate_bridge",
        str(checkpoint),
    ]
    return [
        LaunchCommand("bridge train", tuple(argv)),
        LaunchCommand("bridge bind contract", tuple(bind)),
        LaunchCommand("bridge validate", tuple(validate)),
    ]


def _stage2_commands(config: RuntimeConfig) -> list[LaunchCommand]:
    base = config.path("base_world_checkpoint")
    manifest = config.path("stage2_manifest")
    assert base and manifest
    out = output_dir(config, "stage2")
    steps = config.positive_int("stage2", "steps", 2000)
    save_every = config.positive_int("stage2", "save_every", 20)
    if steps % save_every:
        raise RuntimeError(
            "stage2.save_every must divide stage2.steps so the pipeline can bind "
            "the final checkpoint"
        )
    argv = _torchrun_prefix(config, 29588) + [
        str(ENTRY_ROOT / "stage2.py"),
        "--lam-id",
        LAM_ID,
        "--ckpt",
        str(base),
        "--experiment",
        _PRETRAIN_EXPERIMENT,
        "--config-file",
        _TRAIN_CONFIG,
        "--train-manifest",
        str(manifest),
        "--manifest-split",
        "train",
        "--manifest-random-window",
        "--scope",
        "D",
        "--num-video-frames",
        "13",
        "--num-action-per-chunk",
        "12",
        "--action-dim",
        "384",
        "--resolution",
        "480,640",
        "--lam-resolution",
        "240,320",
        "--cond-dropout",
        str(config.optional_value("stage2", "conditioning_dropout", (int, float), 0.1)),
        "--steps",
        str(steps),
        "--batch-size",
        str(config.positive_int("stage2", "batch_size", 12)),
        "--lr",
        str(config.positive_float("stage2", "learning_rate", 1e-5)),
        "--warmup-steps",
        "0",
        "--manifest-audit-all-ranks",
        "--ckpt-save-every",
        str(save_every),
        "--seed",
        str(config.nonnegative_int("launch", "seed", 42)),
        "--out",
        str(out),
    ]
    python = config.path("python")
    assert python is not None
    validate = [
        str(python),
        "-m",
        "cdlam_runtime.validate_summary",
        "--stage",
        "stage2",
        "--summary",
        str(out / "summary.json"),
        "--expected-steps",
        str(steps),
    ]
    if config.optional_value("validation", "smoke", bool, False):
        validate.append("--smoke")
    return [
        LaunchCommand("Stage 2 train", tuple(argv)),
        LaunchCommand("Stage 2 validate", tuple(validate)),
    ]


def _dataset_paths(path: Path) -> list[Path]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"invalid Stage-3 dataset list {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeError(f"Stage-3 dataset list must contain a mapping: {path}")
    raw = document.get("dataset_path")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"Stage-3 dataset list is empty: {path}")
    result: list[Path] = []
    for value in raw:
        item = Path(str(value)).expanduser()
        if not item.is_absolute():
            item = path.parent / item
        item = item.resolve()
        if not (item / "meta" / "info.json").is_file():
            raise RuntimeError(f"Stage-3 dataset has no meta/info.json: {item}")
        result.append(item)
    return result


def _stage3_commands(
    config: RuntimeConfig, bridge: Path, initialization: Path
) -> list[LaunchCommand]:
    base = config.path("base_world_checkpoint")
    dataset_list = config.path("stage3_dataset_list")
    assert base and dataset_list
    out = output_dir(config, "stage3")
    out.mkdir(parents=True, exist_ok=True)
    paths = _dataset_paths(dataset_list)
    dataset_csv = out / "dataset_paths.txt"
    dataset_csv.write_text(",".join(str(path) for path in paths), encoding="utf-8")
    steps = config.positive_int("stage3", "steps", 3000)
    argv = _torchrun_prefix(config, 29600) + [
        str(ENTRY_ROOT / "stage3.py"),
        "--parallelism",
        config.optional_value("stage3", "parallelism", str, "manual-ddp"),
        "--dataset-path",
        dataset_csv.read_text(encoding="utf-8"),
        "--embodiment",
        "agibot",
        "--base-ckpt",
        str(base),
        "--experiment",
        _POSTTRAIN_EXPERIMENT,
        "--init-trainable-from",
        str(initialization),
        "--gr-bridge-ckpt",
        str(bridge),
        "--scope",
        "D",
        "--latent-slice",
        "352,384",
        "--steps",
        str(steps),
        "--batch-size",
        str(config.positive_int("stage3", "batch_size", 11)),
        "--num-workers",
        "0",
        "--lr",
        str(config.positive_float("stage3", "learning_rate", 2.5e-5)),
        "--warmup-steps",
        str(config.nonnegative_int("stage3", "warmup_steps", 1000)),
        "--save-every",
        str(config.positive_int("stage3", "save_every", 500)),
        "--eval-every",
        str(config.positive_int("stage3", "eval_every", 500)),
        "--free-inline-lam",
        "--out",
        str(out),
    ]
    python = config.path("python")
    assert python is not None
    validate = [
        str(python),
        "-m",
        "cdlam_runtime.validate_summary",
        "--stage",
        "stage3",
        "--summary",
        str(out / "summary.json"),
        "--expected-steps",
        str(steps),
    ]
    if config.optional_value("validation", "smoke", bool, False):
        validate.append("--smoke")
    return [
        LaunchCommand("Stage 3 train", tuple(argv)),
        LaunchCommand("Stage 3 validate", tuple(validate)),
    ]


def commands_for_stage(
    config: RuntimeConfig,
    stage: str,
    *,
    lam_checkpoint: Path | None = None,
    bridge_checkpoint: Path | None = None,
    stage2_checkpoint: Path | None = None,
) -> list[LaunchCommand]:
    """Construct commands for one named stage."""

    if stage == "stage1":
        return _stage1_commands(config)
    if stage == "bridge":
        return _bridge_commands(
            config, lam_checkpoint or stage_checkpoint(config, "stage1")
        )
    if stage == "stage2":
        return _stage2_commands(config)
    if stage == "stage3":
        return _stage3_commands(
            config,
            bridge_checkpoint or stage_checkpoint(config, "bridge"),
            stage2_checkpoint or stage_checkpoint(config, "stage2"),
        )
    raise RuntimeError(f"unsupported stage: {stage}")


def execute(
    commands: Iterable[LaunchCommand],
    *,
    config: RuntimeConfig,
    lam_checkpoint: Path,
    dry_run: bool,
) -> None:
    env = runtime_environment(config, lam_checkpoint)
    gpus = config.optional_value("launch", "gpus", str, "0")
    env["CUDA_VISIBLE_DEVICES"] = gpus
    for command in commands:
        print(f"[{command.label}] {command.display()}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(
            command.argv, cwd=config.workspace, env=env, check=False
        )
        if result.returncode:
            raise RuntimeError(
                f"{command.label} failed with exit code {result.returncode}"
            )


def latest_stage1_checkpoint(config: RuntimeConfig) -> Path:
    state_path = output_dir(config, "stage1") / "run_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checkpoint = Path(state["latest_checkpoint"]).expanduser().resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot resolve Stage-1 output from {state_path}: {exc}"
        ) from exc
    if not checkpoint.is_file():
        raise RuntimeError(f"Stage-1 output checkpoint is missing: {checkpoint}")
    return checkpoint


def expected_stage1_checkpoint(config: RuntimeConfig) -> Path:
    steps = config.positive_int("stage1", "steps", 1000)
    return output_dir(config, "stage1") / "checkpoints" / f"step_{steps - 1:06d}.pt"


def trained_bridge_checkpoint(config: RuntimeConfig) -> Path:
    return output_dir(config, "bridge") / "a22z_agibot_alpha_D_cyc3.pt"


def trained_stage2_checkpoint(config: RuntimeConfig) -> Path:
    steps = config.positive_int("stage2", "steps", 2000)
    return output_dir(config, "stage2") / f"ckpt_step{steps}.pt"


def print_errors(errors: Sequence[str]) -> None:
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
