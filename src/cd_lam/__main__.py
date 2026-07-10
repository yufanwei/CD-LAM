"""Small command-line utilities for the public CD-LAM package."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cd_lam")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-bridge", help="validate a local 22D-to-32D bridge checkpoint"
    )
    validate.add_argument("checkpoint", type=Path)

    doctor = subparsers.add_parser(
        "doctor", help="check the core runtime and optional configured assets"
    )
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero for missing assets explicitly configured by --config",
    )
    doctor.add_argument(
        "--config",
        type=Path,
        help="optional JSON/YAML runtime profile whose configured assets should be checked",
    )

    subparsers.add_parser("smoke", help="run a deterministic CPU core-library smoke test")

    for command, help_text in (
        ("stage1", "run or plan LAM debiased fine-tuning"),
        ("stage2", "run or plan latent-conditioned ACWM fine-tuning"),
        ("bridge-train", "run or plan 22D-to-32D bridge training"),
        ("stage3", "run or plan robot-action ACWM adaptation"),
    ):
        stage = subparsers.add_parser(command, help=help_text)
        stage.add_argument("--config", type=Path, required=True)
        stage.add_argument("--project-root", type=Path)
        stage.add_argument("--dry-run", action="store_true")
        stage.add_argument(
            "--synthetic",
            action="store_true",
            help="run the explicit deterministic CPU integration backend",
        )
        stage.add_argument("--steps", type=int)
        stage.add_argument("--device")
        stage.add_argument("--seed", type=int)
        stage.add_argument("--adapter", help="override module:factory adapter specification")
        stage.add_argument("--resume", type=Path)
        stage.add_argument("--json", action="store_true")

    train_smoke = subparsers.add_parser(
        "train-smoke",
        help="run Stage1, Stage2, bridge, and Stage3 synthetic CPU training in order",
    )
    train_smoke.add_argument("--config", type=Path)
    train_smoke.add_argument(
        "--output-root", type=Path, default=Path("outputs/synthetic_smoke")
    )
    train_smoke.add_argument("--steps", type=int, default=2)
    train_smoke.add_argument("--seed", type=int, default=0)
    train_smoke.add_argument("--json", action="store_true")

    data_prepare = subparsers.add_parser(
        "data-prepare", help="build portable staged manifests from episode JSONL"
    )
    data_prepare.add_argument("--input", type=Path, required=True)
    data_prepare.add_argument("--output", type=Path, required=True)
    data_prepare.add_argument("--pair-stride", type=int, default=1)
    data_prepare.add_argument("--action-stride", type=int, default=4)
    data_prepare.add_argument("--window-frames", type=int, default=13)
    data_prepare.add_argument("--pairs-per-episode", type=int, default=8)
    data_prepare.add_argument("--windows-per-episode", type=int, default=4)

    data_validate = subparsers.add_parser(
        "data-validate", help="validate prepared staged JSONL manifests"
    )
    data_validate.add_argument("--root", type=Path, required=True)
    return parser


def _dependency_report() -> tuple[list[str], list[str]]:
    messages = [f"[ok] python {platform.python_version()}"]
    failures: list[str] = []
    for module_name in ("numpy", "torch"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"[fail] {module_name}: {exc}")
        else:
            messages.append(f"[ok] {module_name} {getattr(module, '__version__', 'unknown')}")
    return messages, failures


def _load_profile(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"runtime config does not exist or is not a file: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "reading a YAML runtime config requires PyYAML; install it or use JSON"
        ) from exc
    return yaml.safe_load(text)


def _iter_configured_assets(profile: Any) -> Iterable[tuple[str, str]]:
    """Yield non-null path-like fields from explicit asset sections.

    Release profiles may leave asset fields null for bootstrap use.  Those are
    intentionally not failures.  We inspect ``assets``/``required_assets``
    sections and common checkpoint/weights keys elsewhere without imposing a
    training-config schema on downstream integrations.
    """

    key_tokens = ("checkpoint", "weights", "model_path", "manifest")
    model_asset_keys = {
        "base_acwm",
        "lam_init",
        "stage2_acwm",
        "stage3_acwm",
        "bridge_bundle",
    }

    def walk(value: Any, key_path: tuple[str, ...], in_assets: bool) -> Iterable[tuple[str, str]]:
        if isinstance(value, dict):
            for key, child in value.items():
                key_string = str(key)
                lowered = key_string.lower()
                child_in_assets = in_assets or lowered in {"assets", "required_assets"}
                yield from walk(child, key_path + (key_string,), child_in_assets)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, key_path + (str(index),), in_assets)
        elif isinstance(value, str) and value.strip() and key_path:
            leaf = key_path[-1].lower()
            if (
                in_assets
                or leaf in model_asset_keys
                or any(token in leaf for token in key_tokens)
            ):
                yield ".".join(key_path), value.strip()

    yield from walk(profile, (), False)


def _iter_existing_storage_roots(profile: Any) -> Iterable[tuple[str, str]]:
    """Yield existing generic storage roots for information only.

    A storage destination is useful diagnostic context, but it is not a model
    or dataset asset and its absence must never fail strict bootstrap checks.
    """

    storage_keys = {"data_root", "artifact_root", "output_root", "cache_root"}

    def walk(value: Any, key_path: tuple[str, ...]) -> Iterable[tuple[str, str]]:
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, key_path + (str(key),))
        elif (
            isinstance(value, str)
            and value.strip()
            and key_path
            and key_path[-1].lower() in storage_keys
        ):
            configured = value.strip()
            if "://" in configured or configured.startswith("hf:"):
                return
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                yield ".".join(key_path), str(path)

    yield from walk(profile, ())


def _check_assets(config: Path) -> tuple[list[str], list[str]]:
    profile = _load_profile(config)
    messages: list[str] = []
    failures: list[str] = []
    for key, existing in _iter_existing_storage_roots(profile):
        messages.append(f"[ok] {key}: {existing}")
    configured_assets = list(_iter_configured_assets(profile))
    for key, configured in configured_assets:
        if "://" in configured or configured.startswith("hf:"):
            messages.append(f"[ok] {key}: remote reference configured")
            continue
        path = Path(configured).expanduser()
        if not path.is_absolute():
            # Public profiles document relative paths from the repository (or
            # caller) working directory, not from the configs/ subdirectory.
            path = Path.cwd() / path
        if path.exists():
            messages.append(f"[ok] {key}: {path}")
        else:
            failures.append(f"[missing] {key}: {path}")
    if not configured_assets:
        messages.append("[ok] no local full-runtime assets are configured")
    return messages, failures


def _doctor(*, strict: bool, config: Optional[Path]) -> int:
    messages, failures = _dependency_report()
    if config is not None:
        try:
            asset_messages, asset_failures = _check_assets(config.resolve())
        except Exception as exc:
            asset_messages, asset_failures = [], [f"[fail] config: {exc}"]
        messages.extend(asset_messages)
        failures.extend(asset_failures)
    elif strict:
        messages.append("[ok] bootstrap strict mode: no runtime profile supplied; assets not required")

    for message in messages + failures:
        print(message)
    if failures and (strict or any(message.startswith("[fail]") for message in failures)):
        print("CD-LAM doctor: FAIL")
        return 1
    if failures:
        print("CD-LAM doctor: PASS (optional configured assets missing; use --strict to enforce)")
    else:
        print("CD-LAM doctor: PASS")
    return 0


def _smoke() -> int:
    import numpy as np
    import torch

    from .bridge import (
        ACTION_DIM,
        LATENT_DIM,
        ActionToLatentBridge,
        build_bridge_mlp,
        prepare_latent_condition,
    )
    from .metrics import fdce
    from .objectives import (
        embodiment_centric_reconstruction_loss,
        free_bits_kl_loss,
        siglip_action_contrastive_loss,
    )

    torch.manual_seed(0)
    checkpoint = {
        "g_state": build_bridge_mlp().state_dict(),
        "action_mean": torch.zeros(ACTION_DIM),
        "action_std": torch.ones(ACTION_DIM),
        "zm": torch.zeros(LATENT_DIM),
        "zsd": torch.ones(LATENT_DIM),
        "latent_dim": LATENT_DIM,
    }
    bridge = ActionToLatentBridge(checkpoint)
    latent = prepare_latent_condition(
        robot_action=torch.zeros(2, ACTION_DIM), bridge=bridge
    )
    if latent.shape != (2, LATENT_DIM) or not bool(torch.isfinite(latent).all()):
        raise RuntimeError("bridge smoke check failed")
    direct_latent = prepare_latent_condition(latent=torch.zeros(2, LATENT_DIM))
    if direct_latent.shape != latent.shape or direct_latent.dtype != torch.float32:
        raise RuntimeError("direct latent conditioning smoke check failed")

    prediction = torch.zeros(1, 1, 2, 2, requires_grad=True)
    reconstruction = embodiment_centric_reconstruction_loss(
        prediction, torch.ones_like(prediction), torch.ones(1, 2, 2)
    )
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    contrastive = siglip_action_contrastive_loss(
        embeddings, torch.tensor([0, 0, 1]), temperature=2.0, bias=0.0
    )
    kl = free_bits_kl_loss(torch.zeros(2, 2), torch.zeros(2, 2), free_bits=0.5)
    (reconstruction + contrastive + kl).backward()
    if prediction.grad is None or embeddings.grad is None:
        raise RuntimeError("objective backward smoke check failed")

    tracks = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
    if fdce(tracks, tracks) != 0.0:
        raise RuntimeError("FDCE smoke check failed")
    print("CD-LAM smoke: PASS")
    return 0


def _print_plan(plan: Any, *, as_json: bool) -> None:
    if as_json:
        print(plan.to_json())
        return
    state = "READY" if plan.ready else "BLOCKED"
    print(f"{plan.stage.value} plan: {state} mode={plan.mode}")
    print(f"  config: {plan.config_name} ({plan.config_digest[:12]})")
    print(f"  steps: {plan.target_steps}  device: {plan.device}")
    print(f"  checkpoint: {plan.output_checkpoint}")
    for blocker in plan.blockers:
        print(f"  blocker: {blocker}")


def _training_command(args: argparse.Namespace) -> int:
    from .config import StageName, load_pipeline_config
    from .plans import build_stage_plan
    from .training.runner import execute_stage

    stage = {
        "stage1": StageName.STAGE1,
        "stage2": StageName.STAGE2,
        "bridge-train": StageName.BRIDGE,
        "stage3": StageName.STAGE3,
    }[args.command]
    config = load_pipeline_config(args.config, project_root=args.project_root)
    plan = build_stage_plan(
        config,
        stage,
        synthetic=args.synthetic,
        target_steps=args.steps,
        device=args.device,
        seed=args.seed,
        adapter=args.adapter,
        resume_from=args.resume,
    )
    if args.dry_run:
        _print_plan(plan, as_json=args.json)
        return 0 if plan.ready else 2
    result = execute_stage(config, plan)
    if args.json:
        print(result.to_json())
    else:
        print(
            f"{stage.value}: PASS steps={result.steps} "
            f"loss={result.initial_loss:.6f}->{result.final_loss:.6f}"
        )
        print(f"checkpoint: {result.checkpoint}")
    return 0


def _train_smoke(args: argparse.Namespace) -> int:
    from .config import PipelineConfig, load_pipeline_config
    from .training.runner import run_synthetic_pipeline

    output_root = args.output_root.expanduser().resolve()
    config = (
        PipelineConfig.synthetic(output_root, seed=args.seed)
        if args.config is None
        else load_pipeline_config(args.config)
    )
    results, summary_path = run_synthetic_pipeline(
        config,
        output_root=output_root,
        target_steps=args.steps,
        seed=args.seed,
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["summary_path"] = str(summary_path)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "train-smoke: PASS "
            + " -> ".join(result.stage.value for result in results)
        )
        print(f"summary: {summary_path}")
    return 0


def _data_prepare(args: argparse.Namespace) -> int:
    from .data import prepare_episode_manifests

    summary = prepare_episode_manifests(
        args.input,
        args.output,
        pair_stride=args.pair_stride,
        source_action_stride=args.action_stride,
        window_frames=args.window_frames,
        pairs_per_episode=args.pairs_per_episode,
        windows_per_episode=args.windows_per_episode,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _data_validate(args: argparse.Namespace) -> int:
    from .data import validate_prepared_manifests

    counts = validate_prepared_manifests(args.root)
    print(json.dumps({"counts": counts, "status": "pass"}, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "validate-bridge":
        from .bridge import load_bridge_checkpoint

        contract = load_bridge_checkpoint(args.checkpoint)
        print(
            "valid CD-LAM bridge: "
            f"action_dim={contract.action_mean.numel()} "
            f"latent_dim={contract.latent_dim} hidden_dim=256"
        )
        return 0
    if args.command == "doctor":
        return _doctor(strict=args.strict, config=args.config)
    if args.command == "smoke":
        return _smoke()
    if args.command in {"stage1", "stage2", "bridge-train", "stage3"}:
        try:
            return _training_command(args)
        except Exception as exc:
            from .adapters import AdapterError
            from .config import ConfigError
            from .plans import PlanError
            from .training.common import StageExecutionError

            if isinstance(exc, (AdapterError, ConfigError, PlanError, StageExecutionError)):
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            raise
    if args.command == "train-smoke":
        try:
            return _train_smoke(args)
        except Exception as exc:
            from .config import ConfigError
            from .plans import PlanError
            from .training.common import StageExecutionError

            if isinstance(exc, (ConfigError, PlanError, StageExecutionError)):
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            raise
    if args.command in {"data-prepare", "data-validate"}:
        try:
            return _data_prepare(args) if args.command == "data-prepare" else _data_validate(args)
        except Exception as exc:
            from .data import DataContractError

            if isinstance(exc, DataContractError):
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            raise
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
