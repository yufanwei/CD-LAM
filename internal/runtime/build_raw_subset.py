#!/usr/bin/env python3
"""Build a bounded raw AgiBot Alpha/EgoDex Stage-1/Stage-2 subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = BUNDLE_ROOT / "internal" / "vendor" / "scale_support"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from Scale.common.raw_subset_ingest import (  # noqa: E402
    RawSubsetError,
    RawSubsetOptions,
    build_raw_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Explicit raw clip JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--pairs-per-clip", type=int, default=8)
    parser.add_argument("--window-frames", type=int, default=13)
    parser.add_argument("--windows-per-clip", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=32)
    parser.add_argument("--max-total-video-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    options = RawSubsetOptions(
        pair_stride=args.pair_stride,
        pairs_per_clip=args.pairs_per_clip,
        window_frames=args.window_frames,
        windows_per_clip=args.windows_per_clip,
        max_clips=args.max_clips,
        max_total_video_bytes=args.max_total_video_bytes,
    )
    try:
        report = build_raw_subset(args.input, args.output, options=options)
    except RawSubsetError as exc:
        print(json.dumps({"errors": [str(exc)], "status": "fail"}, indent=2))
        return 2
    print(json.dumps({"counts": report["counts"], "status": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
