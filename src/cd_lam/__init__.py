"""Public CD-LAM core library.

Imports are lazy so that ``python -m cd_lam doctor`` can diagnose a missing
runtime dependency instead of failing while the package itself is imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "ACTION_DIM",
    "HIDDEN_DIM",
    "LATENT_DIM",
    "ActionToLatentBridge",
    "BridgeCheckpointError",
    "ConfigError",
    "PipelineConfig",
    "SigLIPActionHead",
    "StageAdapter",
    "StageName",
    "StagePlan",
    "ValidatedBridgeCheckpoint",
    "action_contrastive_loss",
    "build_bridge_mlp",
    "build_stage_plan",
    "embodiment_centric_reconstruction_loss",
    "embodiment_reconstruction_loss",
    "free_bits_kl_loss",
    "latent_space_calibration_loss",
    "load_bridge_checkpoint",
    "load_pipeline_config",
    "prepare_latent_condition",
    "relative_zero_transition_loss",
    "siglip_action_contrastive_loss",
    "validate_bridge_checkpoint",
]

_BRIDGE_EXPORTS = {
    "ACTION_DIM",
    "HIDDEN_DIM",
    "LATENT_DIM",
    "ActionToLatentBridge",
    "BridgeCheckpointError",
    "ValidatedBridgeCheckpoint",
    "build_bridge_mlp",
    "load_bridge_checkpoint",
    "prepare_latent_condition",
    "validate_bridge_checkpoint",
}
_OBJECTIVE_EXPORTS = set(__all__) - _BRIDGE_EXPORTS
_CONFIG_EXPORTS = {"ConfigError", "PipelineConfig", "StageName", "load_pipeline_config"}
_PLAN_EXPORTS = {"StagePlan", "build_stage_plan"}
_ADAPTER_EXPORTS = {"StageAdapter"}
_OBJECTIVE_EXPORTS -= _CONFIG_EXPORTS | _PLAN_EXPORTS | _ADAPTER_EXPORTS


def __getattr__(name: str) -> Any:
    if name in _BRIDGE_EXPORTS:
        value = getattr(import_module(".bridge", __name__), name)
    elif name in _OBJECTIVE_EXPORTS:
        value = getattr(import_module(".objectives", __name__), name)
    elif name in _CONFIG_EXPORTS:
        value = getattr(import_module(".config", __name__), name)
    elif name in _PLAN_EXPORTS:
        value = getattr(import_module(".plans", __name__), name)
    elif name in _ADAPTER_EXPORTS:
        value = getattr(import_module(".adapters", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
