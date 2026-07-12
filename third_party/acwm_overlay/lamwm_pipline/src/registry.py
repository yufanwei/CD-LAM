"""Load lamwm_pipline registries from YAML.

Three registries (one file each under lamwm_pipline/configs/):
    lam_registry.yaml   — every LAM ckpt we evaluate
    datasets.yaml       — physical roots and per-dataset metadata
    splits.yaml         — named (role, dataset, subset) tuples

This module returns plain Python dicts. Validation is deliberately light —
the philosophy is "fail loudly when a tool dereferences a missing key" rather
than enforcing a schema upfront. The registries are short and reviewed by hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
CONFIG_DIR = REPO / "lamwm_pipline" / "configs"


def _load(name: str) -> Dict[str, Any]:
    p = CONFIG_DIR / name
    if not p.is_file():
        raise FileNotFoundError(f"registry not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def load_lam_registry() -> Dict[str, Dict[str, Any]]:
    cfg = _load("lam_registry.yaml")
    entries = cfg.get("lams", {})
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, value in entries.items():
        entry = dict(value)
        checkpoint = entry.get("ckpt_path")
        if isinstance(checkpoint, str) and not Path(checkpoint).is_absolute():
            entry["ckpt_path"] = str((REPO / checkpoint).resolve())
        normalized[name] = entry
    return normalized


def load_dataset_registry() -> Dict[str, Any]:
    return _load("datasets.yaml")


def load_split_registry() -> Dict[str, Dict[str, Any]]:
    cfg = _load("splits.yaml")
    return cfg.get("splits", {})


def get_lam_entry(lam_id: str) -> Dict[str, Any]:
    reg = load_lam_registry()
    if lam_id not in reg:
        raise KeyError(f"unknown lam_id={lam_id!r}; known: {sorted(reg.keys())}")
    return reg[lam_id]


def assert_allowed_for_wm(lam_id: str) -> Dict[str, Any]:
    """Raise unless lam is registered AND has allowed_for_wm_control=True.

    Use this from any tool that's about to feed a LAM into WM training. The
    flag is the ACL; LAM-candidate `status` is informational. old_lam_400k is
    `lam_gate_failed` but is M1 control — it MUST set allowed_for_wm_control: true.
    """
    e = get_lam_entry(lam_id)
    if not e.get("allowed_for_wm_control", False):
        raise PermissionError(
            f"LAM {lam_id!r} is not allowed for WM training "
            f"(allowed_for_wm_control={e.get('allowed_for_wm_control')}). "
            f"Add `allowed_for_wm_control: true` to its registry entry, or pick another LAM. "
            f"Current status={e.get('status')!r}."
        )
    return e


def get_split(split_id: str) -> Dict[str, Any]:
    reg = load_split_registry()
    if split_id not in reg:
        raise KeyError(f"unknown split={split_id!r}; known: {sorted(reg.keys())}")
    return reg[split_id]
