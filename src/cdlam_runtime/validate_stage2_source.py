#!/usr/bin/env python3
"""Require the external Stage-2 trainer to consume manifest FPS metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

PATCHED_EXPRESSION = "torch.tensor(manifest_fps, dtype=torch.float).cuda()"
SUMMARY_EXPRESSION = '"fps_source": "manifest" if manifest_mode else "fixed_8"'
HARDCODED_EXPRESSION = '"fps": torch.tensor([8] * bs, dtype=torch.float).cuda()'


def validate_source(root: Path) -> Path:
    source = root / "cdlam_integration" / "world_model" / "train.py"
    if not source.is_file():
        raise FileNotFoundError(f"external Stage-2 trainer is missing: {source}")
    text = source.read_text(encoding="utf-8")
    if PATCHED_EXPRESSION not in text:
        detail = (
            "the hardcoded fps=8 expression is still present"
            if HARDCODED_EXPRESSION in text
            else "the required manifest FPS expression is absent"
        )
        raise RuntimeError(f"Stage-2 source is not contract-safe: {detail}.")
    if SUMMARY_EXPRESSION not in text or '"scope": args.scope' not in text:
        raise RuntimeError(
            "Stage-2 source does not record scope and FPS provenance in its summary."
        )
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    args = parser.parse_args()
    source = validate_source(args.external_root.expanduser().resolve())
    print(f"Stage-2 manifest FPS source: PASS ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
