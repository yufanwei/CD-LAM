from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_results import validate  # noqa: E402


def test_paper_results_fixture() -> None:
    payload = json.loads((ROOT / "docs" / "results" / "paper_results.json").read_text())
    assert validate(payload) == {
        "table_i": 7,
        "table_ii": 4,
        "table_iii": 4,
        "table_iv": 5,
        "table_v": 4,
    }
