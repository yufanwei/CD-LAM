"""Load any LAM ckpt registered in `configs/lam_registry.yaml` and return a
unified pair-encoder.

Two ckpt formats handled:

    official      — the upstream LAM_400k.ckpt. Lightning module state_dict.
                    `external.lam.model.LAM(... ckpt_path=PATH)` already loads it.

    finetune_ddp  — output of CD-LAM Stage 1.
                    The file is `torch.save({"state_dict": sd, ...})`. `sd` keys
                    have leading "module." (DDP) and "encoder." (TripleEncoderWrapper)
                    prefixes. We strip both, then load with strict=False over a base
                    LAM that was already initialized from the parent's official ckpt.
                    Any keys not in the finetune state_dict keep their base values.

Usage:

    enc, lam_module = build_encoder("cdlam_camera_clean_step9800", device="cuda")
    z = enc(pair_b3hw)   # (B, 32) bf16-friendly

`enc` is a `LAMEncoderForFinetune` — same wrapper used by lam_finetune evals,
chosen so that gate metrics here are byte-identical to what eval_ckpt.py reports.
"""

from __future__ import annotations

import os

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))
from cdlam_integration.lam.encoder import (  # noqa: E402
    LAMEncoderForFinetune,
    build_lam,
    freeze_all,
)

from .registry import get_lam_entry  # noqa: E402


def _resolve_base_ckpt(entry: dict) -> str:
    """Walk parent chain until we hit an `official`-format ckpt; return its path."""
    base_id = entry.get("base_ckpt") or entry.get("parent")
    if base_id and base_id != "none":
        base_entry = get_lam_entry(base_id)
        if base_entry.get("ckpt_format") == "official":
            return base_entry["ckpt_path"]
        return _resolve_base_ckpt(base_entry)
    if entry.get("ckpt_format") == "official":
        return entry["ckpt_path"]
    raise ValueError(
        f"could not resolve official base ckpt for entry; "
        f"chain ended without an `official` ckpt_format: {entry}"
    )


def build_encoder(
    lam_id: str, device: str = "cuda"
) -> Tuple[LAMEncoderForFinetune, nn.Module]:
    entry = get_lam_entry(lam_id)
    fmt = entry.get("ckpt_format", "official")

    if fmt == "official":
        lam_module = build_lam(entry["ckpt_path"], torch.device(device)).eval()
        freeze_all(lam_module)
        enc = LAMEncoderForFinetune(lam_module).to(device).eval()
        return enc, lam_module

    if fmt == "finetune_ddp":
        # Init from the resolved official base, then overlay finetune deltas.
        base_path = _resolve_base_ckpt(entry)
        lam_module = build_lam(base_path, torch.device(device)).eval()
        enc = LAMEncoderForFinetune(lam_module).to(device).eval()

        ckpt = torch.load(entry["ckpt_path"], map_location="cpu")
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
        # Saved state_dict keys are "module.encoder.lam.lam.<sub>" (DDP + TripleEncoderWrapper).
        # `enc` (LAMEncoderForFinetune) has state_dict keys "lam.lam.<sub>" — we strip the
        # "module." (DDP) and "encoder." (TripleEncoderWrapper) prefixes to match.
        cleaned = {}
        for k, v in sd.items():
            kk = k
            if kk.startswith("module."):
                kk = kk[len("module.") :]
            if kk.startswith("encoder."):
                kk = kk[len("encoder.") :]
            if kk.startswith("lam.") and not kk.startswith("lam.lam."):
                kk = "lam." + kk
            cleaned[kk] = v
        missing, unexpected = enc.load_state_dict(cleaned, strict=False)
        # We expect MOST keys missing (only the trainable subset is in the finetune ckpt).
        # Unexpected keys mean the ckpt format drifted — flag loudly.
        if len(unexpected) > 0:
            print(
                f"[lam_loader] {lam_id}: WARN unexpected keys = {len(unexpected)}; "
                f"first 3 = {unexpected[:3]}",
                flush=True,
            )
        else:
            n_loaded = len(cleaned) - len(unexpected)
            print(
                f"[lam_loader] {lam_id}: loaded {n_loaded} ckpt params overlaid on base "
                f"({len(missing)} unchanged from base)",
                flush=True,
            )

        freeze_all(lam_module)
        return enc, lam_module

    if fmt == "latent_action_state_dict":
        # Our joint-trained LAM ckpt: torch.save({"lam": sd, "heads": ..., "step": ...})
        # where `sd` is a full LatentActionModel state_dict keyed at the inner level
        # ("action_prompt", "encoder.*", "decoder.*", "fc.*") — i.e. exactly the keys of
        # build_lam(...).lam.lam.state_dict(). The training-side probes load it with
        # cdlam_integration.tools.train_cdlam.load_init_ckpt onto build_lam("CD_LAM_BASE").lam
        # (the LatentActionModel). To reproduce that z here, we init from the resolved
        # official base, then overlay the ckpt's `lam` sub-dict by prefixing each key with
        # "lam.lam." so it lands on LAMEncoderForFinetune (whose keys are "lam.lam.<sub>").
        base_path = _resolve_base_ckpt(entry)
        lam_module = build_lam(base_path, torch.device(device)).eval()
        enc = LAMEncoderForFinetune(lam_module).to(device).eval()

        ckpt = torch.load(entry["ckpt_path"], map_location="cpu")
        state_key = entry.get("state_key", "lam")
        if state_key not in ckpt:
            raise KeyError(
                f"{lam_id}: ckpt has no key {state_key!r}; top-level keys = {list(ckpt.keys())}"
            )
        inner_sd = ckpt[state_key]
        # Prefix inner LatentActionModel keys to the enc namespace.
        cleaned = {f"lam.lam.{k}": v for k, v in inner_sd.items()}
        missing, unexpected = enc.load_state_dict(cleaned, strict=False)
        if len(unexpected) > 0:
            print(
                f"[lam_loader] {lam_id}: WARN unexpected keys = {len(unexpected)}; "
                f"first 3 = {unexpected[:3]}",
                flush=True,
            )
        n_loaded = len(cleaned) - len(unexpected)
        step = ckpt.get("step")
        print(
            f"[lam_loader] {lam_id}: loaded {n_loaded} ckpt params overlaid on base "
            f"({len(missing)} unchanged from base, step={step})",
            flush=True,
        )

        freeze_all(lam_module)
        return enc, lam_module

    raise ValueError(f"unknown ckpt_format={fmt!r} for lam_id={lam_id}")
