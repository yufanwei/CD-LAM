"""Make the src-layout package importable in an uninstalled checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_cdlam_environment(monkeypatch) -> None:
    """Keep unit tests deterministic when a local runtime profile is active."""

    for key in tuple(os.environ):
        if key.startswith("CDLAM_"):
            monkeypatch.delenv(key, raising=False)
