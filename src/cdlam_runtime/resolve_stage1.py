#!/usr/bin/env python3
"""Resolve the immutable Stage-1 snapshot with the selected pair index."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _required_mapping(parent: dict, name: str, label: str) -> dict:
    value = parent.pop(name, None)
    if not isinstance(value, dict):
        raise ValueError(f"Stage-1 recipe must define {label}.{name} as a mapping")
    return value


def _normalize_stage1_recipe(config: dict) -> None:
    """Flatten the public Stage-1 recipe into the trainer's semantic keys."""

    trainer = config["trainer"]
    loss = trainer.get("loss")
    cadence = trainer.get("cadence")
    if not isinstance(loss, dict) or not isinstance(cadence, dict):
        raise ValueError("Stage-1 recipe must define trainer.loss and trainer.cadence")

    reconstruction = _required_mapping(
        loss,
        "masked_reconstruction",
        "trainer.loss",
    )
    reconstruction_names = {
        "enabled": "masked_reconstruction_enabled",
        "partial_full_mix": "partial_full_mix_enabled",
        "foreground_weight": "foreground_reconstruction_weight",
        "background_weight": "background_consistency_weight",
        "full_frame_weight": "full_frame_reconstruction_weight",
        "min_foreground_pixels": "min_foreground_pixels",
        "min_background_pixels": "min_background_pixels",
    }
    unknown = sorted(set(reconstruction) - set(reconstruction_names))
    missing = sorted(set(reconstruction_names) - set(reconstruction))
    if unknown or missing:
        raise ValueError(
            "trainer.loss.masked_reconstruction keys do not match the release "
            f"contract: missing={missing}, unknown={unknown}"
        )
    for public_name, compatibility_name in reconstruction_names.items():
        loss[compatibility_name] = reconstruction[public_name]

    extensions = _required_mapping(
        loss,
        "contrastive_extensions",
        "trainer.loss",
    )
    extension_names = {
        "structured_graph": "structured_graph_enabled",
        "split_loss": "siglip_action_loss_split",
        "grouped_action_loss": "grouped_action_loss_enabled",
        "cross_dataset_centroid": "siglip_xds_centroid_enabled",
        "cross_dataset_rank": "siglip_xds_rank_enabled",
        "prototype_separation": "siglip_proto_sep_enabled",
        "hard_positive_mining": "siglip_hard_pos_enabled",
    }
    unknown = sorted(set(extensions) - set(extension_names))
    missing = sorted(set(extension_names) - set(extensions))
    if unknown or missing:
        raise ValueError(
            "trainer.loss.contrastive_extensions keys do not match the release "
            f"contract: missing={missing}, unknown={unknown}"
        )
    for public_name, compatibility_name in extension_names.items():
        loss[compatibility_name] = extensions[public_name]

    if "baseline_checkpoint" not in cadence:
        raise ValueError("trainer.cadence.baseline_checkpoint is required")


def resolve_config(
    template: str | Path,
    pair_index: str | Path,
    out: str | Path,
    *,
    eval_pair_index: str | Path | None = None,
    overrides: dict | None = None,
) -> Path:
    """Resolve data paths and optional evaluation counts into a recipe copy."""

    config = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("trainer"), dict):
        raise ValueError("Stage-1 recipe must define a trainer mapping")
    _normalize_stage1_recipe(config)
    cadence = config["trainer"].get("cadence")
    if not isinstance(cadence, dict):
        raise ValueError("Stage-1 recipe must define trainer.cadence")
    config["trainer"]["pair_index_train"] = str(Path(pair_index).resolve())
    if eval_pair_index:
        cadence["eval_split_parquet"] = str(Path(eval_pair_index).resolve())
    allowed = {
        "pairs_real": "eval_n_pairs_real",
        "pairs_identity": "eval_n_pairs_id",
        "pairs_per_primitive": "eval_n_per_primitive",
        "reconstruction_tiles": "eval_n_recon_tile",
    }
    for public_name, upstream_name in allowed.items():
        value = (overrides or {}).get(public_name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"stage1.evaluation.{public_name} must be a positive integer"
            )
        cadence[upstream_name] = value
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--pair-index", required=True)
    parser.add_argument("--eval-pair-index")
    parser.add_argument("--eval-n-pairs-real", type=int)
    parser.add_argument("--eval-n-pairs-id", type=int)
    parser.add_argument("--eval-n-per-primitive", type=int)
    parser.add_argument("--eval-n-recon-tile", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = resolve_config(
        args.template,
        args.pair_index,
        args.out,
        eval_pair_index=args.eval_pair_index,
        overrides={
            "pairs_real": args.eval_n_pairs_real,
            "pairs_identity": args.eval_n_pairs_id,
            "pairs_per_primitive": args.eval_n_per_primitive,
            "reconstruction_tiles": args.eval_n_recon_tile,
        },
    )
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
