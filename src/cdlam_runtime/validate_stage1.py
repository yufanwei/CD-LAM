#!/usr/bin/env python3
"""Validate a completed Stage-1 run and fail closed on missing evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_run(
    root: Path, expected_steps: int, require_eval: bool, *, smoke: bool = False
) -> list[str]:
    errors: list[str] = []
    state_path = root / "run_state.json"
    metrics_path = root / "train_metrics.jsonl"
    if not state_path.is_file():
        return [f"missing run state: {state_path}"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    status = state.get("status")
    if smoke and expected_steps > 5:
        errors.append("smoke acceptance is limited to at most 5 steps")
    accepted_status = {"completed", "paused"} if smoke else {"completed"}
    if status not in accepted_status:
        errors.append(f"run status is not completed: {state.get('status')!r}")
    accepted_steps = {expected_steps, expected_steps - 1} if smoke else {expected_steps}
    if state.get("current_step") not in accepted_steps:
        errors.append(
            f"current_step={state.get('current_step')!r}, expected {expected_steps}"
        )
    if status == "paused" and not str(state.get("pause_reason") or "").strip():
        errors.append("paused smoke run did not record a stop-condition reason")
    checkpoint = Path(str(state.get("latest_checkpoint") or ""))
    if not checkpoint.is_file():
        errors.append(f"latest checkpoint is missing: {checkpoint}")
    if require_eval:
        evaluation = Path(str(state.get("latest_eval") or ""))
        if not evaluation.is_file():
            errors.append(f"required evaluation is missing: {evaluation}")
        else:
            try:
                payload = json.loads(evaluation.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"evaluation is not valid JSON: {exc}")
            else:
                if not isinstance(payload, dict) or not payload:
                    errors.append("evaluation payload is empty")
    if not metrics_path.is_file():
        errors.append(f"training metrics are missing: {metrics_path}")
    else:
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            errors.append("training metrics are empty")
        for index, row in enumerate(rows):
            for key in ("L_total", "L_gen", "grad_norm_total"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"metrics[{index}].{key} is not finite: {value!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    errors = validate_run(
        args.root.resolve(), args.expected_steps, args.require_eval, smoke=args.smoke
    )
    for error in errors:
        print(f"ERROR {error}")
    print(
        f"stage1_run root={args.root.resolve()} expected_steps={args.expected_steps} "
        f"require_eval={args.require_eval} smoke={args.smoke} errors={len(errors)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
