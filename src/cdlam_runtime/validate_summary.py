#!/usr/bin/env python3
"""Validate internal Stage-2 and Stage-3 completion summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class SummaryError(ValueError):
    """Raised when a training summary cannot be promoted."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryError(f"{label} must be finite")
    return result


def validate_summary(
    path: Path, stage: str, expected_steps: int, *, smoke: bool = False
) -> dict[str, Any]:
    if not path.is_file():
        raise SummaryError(f"training summary is missing: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"invalid training summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise SummaryError("training summary must contain a JSON object")

    observed_steps = summary.get("steps")
    if observed_steps != expected_steps:
        raise SummaryError(
            f"summary steps are {observed_steps!r}, expected {expected_steps}"
        )
    for key in ("init_loss", "final_loss", "min_loss"):
        _finite_number(summary.get(key), key)

    if stage == "stage2":
        if summary.get("scope") != "D":
            raise SummaryError(
                f"Stage 2 scope is {summary.get('scope')!r}, expected 'D'"
            )
        verdict = summary.get("verdict")
        if smoke:
            if expected_steps > 5:
                raise SummaryError("smoke acceptance is limited to at most 5 steps")
            if verdict not in {"PASS", "WEAK_DROP"}:
                raise SummaryError(
                    f"Stage 2 smoke verdict is {verdict!r}, expected PASS or WEAK_DROP"
                )
            if _finite_number(
                summary.get("final_loss"), "final_loss"
            ) >= _finite_number(summary.get("init_loss"), "init_loss"):
                raise SummaryError("Stage 2 smoke loss did not decrease")
        elif verdict != "PASS":
            raise SummaryError(
                f"Stage 2 acceptance verdict is {verdict!r}, expected 'PASS'"
            )
        if summary.get("fps_source") != "manifest":
            raise SummaryError(
                "Stage 2 did not record manifest FPS as its timing source"
            )
        if summary.get("manifest_audit_all_ranks") is not True:
            raise SummaryError("Stage 2 did not record an all-rank manifest audit")
    elif stage == "stage3":
        if summary.get("scope") != "D":
            raise SummaryError(
                f"Stage 3 scope is {summary.get('scope')!r}, expected 'D'"
            )
        history = summary.get("eval_history")
        if not isinstance(history, list) or not history:
            raise SummaryError("Stage 3 summary has no action-perturbation evaluation")
        for index, record in enumerate(history):
            if not isinstance(record, dict) or not isinstance(
                record.get("z_eval"), dict
            ):
                raise SummaryError(f"Stage 3 eval record {index} is invalid")
            for mode in ("own", "zero_z", "shuffle_time_z"):
                _finite_number(
                    record["z_eval"].get(mode), f"eval_history[{index}].{mode}"
                )
        bridge_meta = summary.get("bridge_meta")
        contract = (
            bridge_meta.get("cdlam_action_contract")
            if isinstance(bridge_meta, dict)
            else None
        )
        if not isinstance(contract, dict) or not contract.get("contract_id"):
            raise SummaryError(
                "Stage 3 summary is missing the validated action contract"
            )
    else:
        raise SummaryError(f"unsupported stage: {stage}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage2", "stage3"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    validate_summary(args.summary, args.stage, args.expected_steps, smoke=args.smoke)
    mode = "smoke" if args.smoke else "protocol"
    print(f"{args.stage} summary: PASS mode={mode} ({args.summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
