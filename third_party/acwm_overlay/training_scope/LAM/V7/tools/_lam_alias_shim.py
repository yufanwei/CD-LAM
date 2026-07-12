"""Post-reorg import shim: make ``LAM_Vx`` names resolve to ``training_scope.LAM.Vx``.

The LAM toolchain imports ``from LAM_V6.tools import _lam_v6_data`` etc., but the
top-level ``LAM_V*`` symlinks were removed in the 2026-05-27 repo reorg (canonical
location is ``training_scope/LAM/V*``). Rather than recreate those top-level dirs
(explicitly discouraged in reference_paths) this registers in-process module
aliases — no filesystem changes, fully contained to the running process.

Call ``install()`` once at the top of any script that needs the LAM toolchain.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4])))

_LAM_DIR = REPO / "training_scope" / "LAM"


def install() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    # Auto-discover every training_scope/LAM/V* that is a real package
    # (has __init__.py) and alias it as the top-level name LAM_V* the
    # toolchain imports. Covers V2/V3/V4/V5/V6 (the train graph) without
    # hard-coding.
    for vdir in sorted(_LAM_DIR.glob("V*")):
        if not (vdir / "__init__.py").is_file():
            continue
        alias = f"LAM_{vdir.name}"  # e.g. LAM_V4
        real = f"training_scope.LAM.{vdir.name}"
        if alias not in sys.modules:
            sys.modules[alias] = importlib.import_module(real)
        if (vdir / "tools" / "__init__.py").is_file() and (
            alias + ".tools"
        ) not in sys.modules:
            sys.modules[alias + ".tools"] = importlib.import_module(real + ".tools")
