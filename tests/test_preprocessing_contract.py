from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "internal" / "vendor" / "scale_support"
sys.path.insert(0, str(VENDOR_ROOT))

from Scale.common import shard_io  # noqa: E402


def _load_pinned_reference():
    imageio = types.ModuleType("imageio")
    imageio_v3 = types.ModuleType("imageio.v3")
    imageio.v3 = imageio_v3
    sys.modules.setdefault("imageio", imageio)
    sys.modules.setdefault("imageio.v3", imageio_v3)
    path = (
        ROOT
        / "third_party"
        / "acwm_overlay"
        / "cdlam_integration"
        / "world_model"
        / "preprocess.py"
    )
    spec = importlib.util.spec_from_file_location("cdlam_pinned_preprocess", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bundled_preprocessing_is_byte_exact_to_pinned_runtime() -> None:
    reference = _load_pinned_reference()
    rng = np.random.default_rng(7)
    for shape in ((2, 97, 151, 3), (2, 151, 97, 3)):
        frames = rng.integers(0, 256, size=shape, dtype=np.uint8)
        np.testing.assert_array_equal(
            shard_io._bundled_official_wm(frames),
            reference.official_wm_video_from_raw(frames),
        )
        np.testing.assert_array_equal(
            shard_io._bundled_official_lam(frames),
            reference.official_lam_video_from_raw(frames),
        )
