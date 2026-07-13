"""Single routed command-line entry point for real CD-LAM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cdlam_runtime.config import ConfigError, RuntimeConfig
from cdlam_runtime.runtime import (
    STAGES,
    commands_for_stage,
    doctor,
    execute,
    expected_stage1_checkpoint,
    latest_stage1_checkpoint,
    print_errors,
    stage_checkpoint,
    trained_bridge_checkpoint,
    trained_stage2_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.json"),
        help="JSON runtime profile (default: configs/runtime.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="validate paths, sources, and imports"
    )
    doctor_parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    doctor_parser.add_argument("--no-imports", action="store_true")

    for name in (*STAGES, "pipeline"):
        stage_parser = subparsers.add_parser(name, help=f"run {name}")
        stage_parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_one(config: RuntimeConfig, stage: str, args: argparse.Namespace) -> None:
    lam = stage_checkpoint(config, "stage1")
    errors = doctor(config, stage, lam_checkpoint=lam)
    if errors:
        print_errors(errors)
        raise ConfigError(f"{stage} preflight failed with {len(errors)} error(s)")
    execute(
        commands_for_stage(config, stage),
        config=config,
        lam_checkpoint=lam,
        dry_run=args.dry_run,
    )


def _run_pipeline(config: RuntimeConfig, args: argparse.Namespace) -> None:
    source_lam = stage_checkpoint(config, "stage1")

    errors = doctor(config, "stage1")
    if errors:
        print_errors(errors)
        raise ConfigError(f"Stage-1 preflight failed with {len(errors)} error(s)")
    execute(
        commands_for_stage(config, "stage1"),
        config=config,
        lam_checkpoint=source_lam,
        dry_run=args.dry_run,
    )
    trained_lam = (
        expected_stage1_checkpoint(config)
        if args.dry_run
        else latest_stage1_checkpoint(config)
    )

    for stage in ("bridge", "stage2"):
        errors = doctor(config, stage, lam_checkpoint=trained_lam)
        if errors and not args.dry_run:
            print_errors(errors)
            raise ConfigError(f"{stage} preflight failed with {len(errors)} error(s)")
        execute(
            commands_for_stage(config, stage, lam_checkpoint=trained_lam),
            config=config,
            lam_checkpoint=trained_lam,
            dry_run=args.dry_run,
        )

    bridge = trained_bridge_checkpoint(config)
    stage2 = trained_stage2_checkpoint(config)
    if not args.dry_run:
        errors = doctor(
            config,
            "stage3",
            lam_checkpoint=trained_lam,
            bridge_checkpoint=bridge,
            stage2_checkpoint=stage2,
        )
        if errors:
            print_errors(errors)
            raise ConfigError(f"Stage-3 preflight failed with {len(errors)} error(s)")
    execute(
        commands_for_stage(
            config,
            "stage3",
            lam_checkpoint=trained_lam,
            bridge_checkpoint=bridge,
            stage2_checkpoint=stage2,
        ),
        config=config,
        lam_checkpoint=trained_lam,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = RuntimeConfig.load(args.config)
        if args.command == "doctor":
            errors = doctor(config, args.stage, imports=not args.no_imports)
            print_errors(errors)
            print(
                f"doctor stage={args.stage} errors={len(errors)} profile={config.profile_path}"
            )
            return 1 if errors else 0
        if args.command == "pipeline":
            _run_pipeline(config, args)
        else:
            _run_one(config, args.command, args)
    except ConfigError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
