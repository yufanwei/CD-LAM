#!/usr/bin/env python3
"""Validate the schema and canonical values in the paper-result fixture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "docs" / "results" / "paper_results.json"
TITLE = (
    "Causally Debiased Latent Action Model for Embodied Action Conditioned World Models"
)


def _rows_by_model(table: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in table["rows"]:
        key = (str(row["backbone"]), str(row["model"]))
        if key in rows:
            raise ValueError(f"duplicate backbone/model row: {key}")
        rows[key] = row
    return rows


def _assert_close(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} must be numeric, got {actual!r}")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label}: expected {expected}, got {actual}")


def validate(payload: dict[str, Any]) -> dict[str, int]:
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if payload.get("paper", {}).get("title") != TITLE:
        raise ValueError("paper title does not match the canonical manuscript title")
    if payload.get("status", {}).get("recomputed_by_this_release") is not False:
        raise ValueError(
            "fixture must not claim that source-only validation recomputed metrics"
        )

    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("tables must be a mapping")
    expected_counts = {
        "table_i": 7,
        "table_ii": 4,
        "table_iii": 4,
        "table_iv": 5,
        "table_v": 4,
    }
    for name, count in expected_counts.items():
        rows = tables.get(name, {}).get("rows")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{name} must contain {count} rows")

    audit = tables["table_i"]["rows"]
    _assert_close(audit[0]["dreamdojo_lam"], 0.527, "Table I baseline zero response")
    _assert_close(audit[0]["cd_lam"], 0.043, "Table I CD-LAM zero response")
    _assert_close(audit[-1]["dreamdojo_lam"], 0.151, "Table I baseline leakage")
    _assert_close(audit[-1]["cd_lam"], 0.014, "Table I CD-LAM leakage")

    stage2 = _rows_by_model(tables["table_ii"])
    stage2_fdce = {
        ("2B", "DreamDojo"): (34.00, 42.74),
        ("2B", "CD-LAM"): (19.63, 33.81),
        ("14B", "DreamDojo"): (40.29, 50.27),
        ("14B", "CD-LAM"): (29.87, 33.22),
    }
    if set(stage2) != set(stage2_fdce):
        raise ValueError("Table II backbone/model rows changed")
    for key, (own_fdce, transfer_fdce) in stage2_fdce.items():
        _assert_close(
            stage2[key]["own_latent_rollout"]["fdce"],
            own_fdce,
            f"Table II {key} own FDCE",
        )
        _assert_close(
            stage2[key]["target_latent_transfer"]["fdce"],
            transfer_fdce,
            f"Table II {key} transfer FDCE",
        )

    stage3 = _rows_by_model(tables["table_iii"])
    stage3_expected = {
        ("2B", "DreamDojo"): (12.63, 8.15, 19.85, 10.71, 24.36),
        ("2B", "CD-LAM"): (8.24, 6.75, 20.60, 5.03, 22.55),
        ("14B", "DreamDojo"): (11.11, 8.98, 20.01, 9.36, 24.82),
        ("14B", "CD-LAM"): (7.73, 5.99, 21.01, 2.18, 21.11),
    }
    if set(stage3) != set(stage3_expected):
        raise ValueError("Table III backbone/model rows changed")
    for key, expected in stage3_expected.items():
        row = stage3[key]
        rollout = row["robot_action_rollout"]
        actual = (
            rollout["fdce_mean"],
            rollout["fdce_median"],
            rollout["psnr"],
            row["zero_action_fdce"],
            row["target_action_transfer_fdce"],
        )
        for index, (value, target) in enumerate(zip(actual, expected)):
            _assert_close(value, target, f"Table III {key} field {index}")

    if (
        "not comparable"
        not in tables["table_iii"]["protocol"]["zero_action_comparability"]
    ):
        raise ValueError("Table III must retain the zero-action comparability warning")
    if "not comparable" not in tables["table_v"]["protocol"]["fdce"]:
        raise ValueError("Table V must remain explicitly separate from Table III")

    return expected_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    counts = validate(payload)
    print(
        "paper_results: PASS "
        + " ".join(f"{name}={count}" for name, count in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
