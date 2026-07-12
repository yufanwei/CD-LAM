#!/usr/bin/env python3
"""Fail-closed provenance audit for normalized AgiBot Alpha/EgoDex rows.

This command validates a JSONL provenance index before any transitions,
windows, masks, or caches are sampled.  It does not replace the external raw
dataset converters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = BUNDLE_ROOT / "internal" / "vendor" / "scale_support"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from Scale.common.raw_data_contract import audit_raw_split_records  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL provenance index and reject malformed rows."""

    if not path.is_file():
        raise ValueError(f"provenance JSONL is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"row {line_number} in {path} must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"provenance JSONL is empty: {path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Normalized provenance JSONL; this is produced by a dataset adapter.",
    )
    args = parser.parse_args()
    try:
        records = load_jsonl(args.input.resolve())
        summary = audit_raw_split_records(records)
    except ValueError as exc:
        print(json.dumps({"errors": [str(exc)], "status": "fail"}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
