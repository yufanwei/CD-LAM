"""Robot-action to CD-LAM latent bridge.

The public checkpoint contract is intentionally small and strict.  A valid
checkpoint contains a three-layer GELU MLP together with the statistics that
define both sides of the mapping::

    z = g((action - action_mean) / action_std) * zsd + zm

Dropping any of the four statistics changes the meaning of the bridge, even if
the MLP weights still load successfully.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn


ACTION_DIM = 22
LATENT_DIM = 32
HIDDEN_DIM = 256

_REQUIRED_FIELDS = (
    "g_state",
    "action_mean",
    "action_std",
    "zm",
    "zsd",
    "latent_dim",
)
_G_STATE_SHAPES = OrderedDict(
    (
        ("0.weight", (HIDDEN_DIM, ACTION_DIM)),
        ("0.bias", (HIDDEN_DIM,)),
        ("2.weight", (HIDDEN_DIM, HIDDEN_DIM)),
        ("2.bias", (HIDDEN_DIM,)),
        ("4.weight", (LATENT_DIM, HIDDEN_DIM)),
        ("4.bias", (LATENT_DIM,)),
    )
)


class BridgeCheckpointError(ValueError):
    """Raised when a bridge checkpoint violates the public contract."""


@dataclass(frozen=True)
class ValidatedBridgeCheckpoint:
    """Validated, CPU-float32 representation of a bridge checkpoint."""

    g_state: Mapping[str, Tensor]
    action_mean: Tensor
    action_std: Tensor
    zm: Tensor
    zsd: Tensor
    latent_dim: int
    metadata: Mapping[str, Any]


def build_bridge_mlp() -> nn.Sequential:
    """Build the exact 22D-to-32D bridge architecture used by CD-LAM."""

    return nn.Sequential(
        nn.Linear(ACTION_DIM, HIDDEN_DIM),
        nn.GELU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.GELU(),
        nn.Linear(HIDDEN_DIM, LATENT_DIM),
    )


def _as_vector(value: Any, *, name: str, length: int) -> Tensor:
    try:
        vector = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BridgeCheckpointError(
            f"checkpoint field {name!r} must be a numeric vector of length {length}"
        ) from exc

    if vector.ndim != 1 or tuple(vector.shape) != (length,):
        raise BridgeCheckpointError(
            f"checkpoint field {name!r} must have shape ({length},), "
            f"got {tuple(vector.shape)}"
        )
    if vector.dtype == torch.bool or not (
        vector.dtype.is_floating_point or vector.dtype.is_complex
    ):
        raise BridgeCheckpointError(
            f"checkpoint field {name!r} must use a floating-point dtype, got {vector.dtype}"
        )
    if vector.dtype.is_complex:
        raise BridgeCheckpointError(
            f"checkpoint field {name!r} must be real-valued, got {vector.dtype}"
        )
    vector = vector.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(vector).all()):
        raise BridgeCheckpointError(f"checkpoint field {name!r} contains NaN or infinity")
    return vector


def _validate_g_state(value: Any) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise BridgeCheckpointError("checkpoint field 'g_state' must be a state-dict mapping")

    expected = set(_G_STATE_SHAPES)
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing keys {missing}")
        if unexpected:
            parts.append(f"unexpected keys {unexpected}")
        raise BridgeCheckpointError("invalid 'g_state': " + "; ".join(parts))

    state: OrderedDict[str, Tensor] = OrderedDict()
    for key, expected_shape in _G_STATE_SHAPES.items():
        tensor = value[key]
        if not isinstance(tensor, Tensor):
            raise BridgeCheckpointError(
                f"checkpoint tensor g_state[{key!r}] must be a torch.Tensor"
            )
        if tuple(tensor.shape) != expected_shape:
            raise BridgeCheckpointError(
                f"checkpoint tensor g_state[{key!r}] must have shape "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )
        if not tensor.dtype.is_floating_point:
            raise BridgeCheckpointError(
                f"checkpoint tensor g_state[{key!r}] must be floating point, got {tensor.dtype}"
            )
        copied = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        if not bool(torch.isfinite(copied).all()):
            raise BridgeCheckpointError(
                f"checkpoint tensor g_state[{key!r}] contains NaN or infinity"
            )
        state[key] = copied
    return state


def validate_bridge_checkpoint(checkpoint: Mapping[str, Any]) -> ValidatedBridgeCheckpoint:
    """Validate and normalize an in-memory bridge checkpoint.

    Extra top-level metadata is retained because released training checkpoints
    may also carry evaluation results or an inverse/readout head.  The six
    required fields and the contents of ``g_state`` are validated strictly.
    No checkpoint tensor is mutated.
    """

    if not isinstance(checkpoint, Mapping):
        raise BridgeCheckpointError(
            f"bridge checkpoint must be a mapping, got {type(checkpoint).__name__}"
        )

    missing = [name for name in _REQUIRED_FIELDS if name not in checkpoint]
    if missing:
        raise BridgeCheckpointError(
            "bridge checkpoint is missing required field(s): " + ", ".join(missing)
        )

    latent_dim = checkpoint["latent_dim"]
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, Integral):
        raise BridgeCheckpointError(
            f"checkpoint field 'latent_dim' must be the integer {LATENT_DIM}, "
            f"got {latent_dim!r}"
        )
    latent_dim = int(latent_dim)
    if latent_dim != LATENT_DIM:
        raise BridgeCheckpointError(
            f"checkpoint field 'latent_dim' must be {LATENT_DIM}, got {latent_dim}"
        )

    g_state = _validate_g_state(checkpoint["g_state"])
    action_mean = _as_vector(checkpoint["action_mean"], name="action_mean", length=ACTION_DIM)
    action_std = _as_vector(checkpoint["action_std"], name="action_std", length=ACTION_DIM)
    zm = _as_vector(checkpoint["zm"], name="zm", length=LATENT_DIM)
    zsd = _as_vector(checkpoint["zsd"], name="zsd", length=LATENT_DIM)

    if bool((action_std <= 0).any()):
        raise BridgeCheckpointError("checkpoint field 'action_std' must be strictly positive")
    if bool((zsd <= 0).any()):
        raise BridgeCheckpointError("checkpoint field 'zsd' must be strictly positive")

    metadata = {key: value for key, value in checkpoint.items() if key not in _REQUIRED_FIELDS}
    return ValidatedBridgeCheckpoint(
        g_state=g_state,
        action_mean=action_mean,
        action_std=action_std,
        zm=zm,
        zsd=zsd,
        latent_dim=latent_dim,
        metadata=metadata,
    )


CheckpointSource = Union[
    Mapping[str, Any],
    ValidatedBridgeCheckpoint,
    str,
    PathLike[str],
]


def load_bridge_checkpoint(source: CheckpointSource) -> ValidatedBridgeCheckpoint:
    """Load and validate a bridge checkpoint from a mapping or local file.

    PyTorch checkpoint files are pickle-based.  Only load files obtained from
    a trusted source.
    """

    if isinstance(source, ValidatedBridgeCheckpoint):
        return source
    if isinstance(source, Mapping):
        return validate_bridge_checkpoint(source)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"bridge checkpoint does not exist or is not a file: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise BridgeCheckpointError(f"failed to load bridge checkpoint {path}: {exc}") from exc
    return validate_bridge_checkpoint(checkpoint)


class ActionToLatentBridge(nn.Module):
    """Exact action-normalize, MLP, latent-denormalize bridge.

    The module accepts a single action vector ``(22,)`` or any batched tensor
    ``(..., 22)`` and always produces float32 latent vectors ``(..., 32)``.
    Like other PyTorch modules, inputs must be on the same device as the module.
    """

    action_dim = ACTION_DIM
    latent_dim = LATENT_DIM

    def __init__(self, checkpoint: CheckpointSource, *, freeze: bool = True) -> None:
        super().__init__()
        contract = load_bridge_checkpoint(checkpoint)

        self.g = build_bridge_mlp()
        self.g.load_state_dict(contract.g_state, strict=True)
        self.register_buffer("action_mean", contract.action_mean.clone())
        self.register_buffer("action_std", contract.action_std.clone())
        self.register_buffer("zm", contract.zm.clone())
        self.register_buffer("zsd", contract.zsd.clone())
        self.metadata = dict(contract.metadata)

        if freeze:
            self.requires_grad_(False)
            self.eval()

    @classmethod
    def from_checkpoint(
        cls, source: CheckpointSource, *, freeze: bool = True
    ) -> "ActionToLatentBridge":
        """Construct a bridge from an in-memory checkpoint or local file."""

        return cls(source, freeze=freeze)

    def normalize_action(self, action: Tensor) -> Tensor:
        """Apply the checkpoint's 22D action normalization."""

        self._validate_action(action)
        return (action.to(dtype=torch.float32) - self.action_mean) / self.action_std

    def forward(self, action: Tensor) -> Tensor:
        """Map ``(..., 22)`` actions to denormalized ``(..., 32)`` latents."""

        normalized_action = self.normalize_action(action)
        normalized_latent = self.g(normalized_action)
        return normalized_latent * self.zsd + self.zm

    def _validate_action(self, action: Tensor) -> None:
        if not isinstance(action, Tensor):
            raise TypeError(f"action must be a torch.Tensor, got {type(action).__name__}")
        if action.ndim < 1 or action.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"action must have shape (..., {ACTION_DIM}), got {tuple(action.shape)}"
            )
        if action.dtype.is_complex:
            raise TypeError(f"action must be real-valued, got dtype {action.dtype}")
        if action.device != self.action_mean.device:
            raise ValueError(
                "action and bridge must be on the same device: "
                f"action is on {action.device}, bridge is on {self.action_mean.device}"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("action contains NaN or infinity")

    def to_checkpoint(self) -> dict[str, Any]:
        """Return a self-contained checkpoint mapping satisfying the contract."""

        checkpoint = dict(self.metadata)
        checkpoint.update(
            {
                "g_state": OrderedDict(
                    (key, value.detach().cpu().clone()) for key, value in self.g.state_dict().items()
                ),
                "action_mean": self.action_mean.detach().cpu().clone(),
                "action_std": self.action_std.detach().cpu().clone(),
                "zm": self.zm.detach().cpu().clone(),
                "zsd": self.zsd.detach().cpu().clone(),
                "latent_dim": LATENT_DIM,
            }
        )
        return checkpoint

    def extra_repr(self) -> str:
        frozen = not any(parameter.requires_grad for parameter in self.g.parameters())
        return f"action_dim={ACTION_DIM}, latent_dim={LATENT_DIM}, frozen={frozen}"


def prepare_latent_condition(
    *,
    latent: Optional[Tensor] = None,
    robot_action: Optional[Tensor] = None,
    bridge: Optional[ActionToLatentBridge] = None,
) -> Tensor:
    """Prepare one model-agnostic ``(..., 32)`` latent condition.

    Exactly one conditioning route must be selected:

    - pass a precomputed ``latent`` with shape ``(..., 32)``; or
    - pass a ``robot_action`` with shape ``(..., 22)`` together with its
      validated :class:`ActionToLatentBridge`.

    The function deliberately does not call an ACWM or assume how a downstream
    model names its conditioning argument.  The returned tensor is finite,
    float32, and remains on the selected input/module device.
    """

    latent_route = latent is not None
    action_route = robot_action is not None or bridge is not None
    if latent_route == action_route:
        raise ValueError(
            "select exactly one conditioning route: either latent, or "
            "robot_action together with bridge"
        )

    if latent_route:
        assert latent is not None
        if not isinstance(latent, Tensor):
            raise TypeError(f"latent must be a torch.Tensor, got {type(latent).__name__}")
        if latent.ndim < 1 or latent.shape[-1] != LATENT_DIM:
            raise ValueError(
                f"latent must have shape (..., {LATENT_DIM}), got {tuple(latent.shape)}"
            )
        if latent.dtype.is_complex:
            raise TypeError(f"latent must be real-valued, got dtype {latent.dtype}")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("latent contains NaN or infinity")
        return latent.to(dtype=torch.float32)

    if robot_action is None or bridge is None:
        raise ValueError("robot_action and bridge must be provided together")
    if not isinstance(bridge, ActionToLatentBridge):
        raise TypeError(
            "bridge must be an ActionToLatentBridge, "
            f"got {type(bridge).__name__}"
        )
    condition = bridge(robot_action)
    if condition.ndim < 1 or condition.shape[-1] != LATENT_DIM:
        raise RuntimeError(
            f"bridge returned shape {tuple(condition.shape)}; expected (..., {LATENT_DIM})"
        )
    if condition.device != bridge.action_mean.device:
        raise RuntimeError("bridge returned a condition on an unexpected device")
    if not bool(torch.isfinite(condition).all()):
        raise ValueError("bridge output contains NaN or infinity")
    return condition.to(dtype=torch.float32)


__all__ = [
    "ACTION_DIM",
    "LATENT_DIM",
    "HIDDEN_DIM",
    "ActionToLatentBridge",
    "BridgeCheckpointError",
    "ValidatedBridgeCheckpoint",
    "build_bridge_mlp",
    "load_bridge_checkpoint",
    "prepare_latent_condition",
    "validate_bridge_checkpoint",
]
