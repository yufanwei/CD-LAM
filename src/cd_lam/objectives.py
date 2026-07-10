"""Reusable training objectives from the CD-LAM formulation."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Scalar = Union[float, Tensor]


def _validate_float_tensor(name: str, value: Tensor, *, min_ndim: int = 1) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimension(s), got {value.ndim}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got dtype {value.dtype}")


def _broadcast_foreground_mask(
    foreground_mask: Tensor,
    prediction: Tensor,
    channel_dim: int,
) -> Tensor:
    if not isinstance(foreground_mask, Tensor):
        raise TypeError(
            "foreground_mask must be a torch.Tensor, "
            f"got {type(foreground_mask).__name__}"
        )
    if foreground_mask.device != prediction.device:
        raise ValueError(
            "foreground_mask and prediction must be on the same device: "
            f"{foreground_mask.device} != {prediction.device}"
        )

    channel_dim = channel_dim % prediction.ndim
    mask = foreground_mask
    if mask.ndim == prediction.ndim - 1:
        expected = prediction.shape[:channel_dim] + prediction.shape[channel_dim + 1 :]
        if tuple(mask.shape) != tuple(expected):
            raise ValueError(
                "foreground_mask with one fewer dimension must match prediction "
                f"with channel_dim={channel_dim} removed: expected {tuple(expected)}, "
                f"got {tuple(mask.shape)}"
            )
        mask = mask.unsqueeze(channel_dim)
    elif mask.ndim != prediction.ndim:
        raise ValueError(
            "foreground_mask must have either the same rank as prediction or one "
            f"fewer dimension, got mask {tuple(mask.shape)} and prediction "
            f"{tuple(prediction.shape)}"
        )

    try:
        broadcast_shape = torch.broadcast_shapes(mask.shape, prediction.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"foreground_mask shape {tuple(mask.shape)} is not broadcastable to "
            f"prediction shape {tuple(prediction.shape)}"
        ) from exc
    if tuple(broadcast_shape) != tuple(prediction.shape):
        raise ValueError(
            f"foreground_mask would broadcast to {tuple(broadcast_shape)}, not the "
            f"prediction shape {tuple(prediction.shape)}"
        )

    mask = mask.to(dtype=prediction.dtype)
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("foreground_mask contains NaN or infinity")
    if bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError("foreground_mask values must lie in [0, 1]")
    return mask


def embodiment_centric_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    foreground_mask: Tensor,
    *,
    foreground_weight: float = 5.0,
    background_weight: float = 1.0,
    channel_dim: int = -3,
    reduction: str = "mean",
) -> Tensor:
    """Embodiment-centric weighted reconstruction loss (paper Eqs. 8--9).

    ``foreground_mask`` can match ``prediction`` or omit its channel dimension.
    Soft masks in ``[0, 1]`` are supported.  The default layout is channel-first
    (``..., C, H, W``); pass ``channel_dim=-1`` for channel-last tensors.
    """

    _validate_float_tensor("prediction", prediction, min_ndim=3)
    _validate_float_tensor("target", target, min_ndim=3)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if channel_dim < -prediction.ndim or channel_dim >= prediction.ndim:
        raise ValueError(
            f"channel_dim={channel_dim} is invalid for a {prediction.ndim}D tensor"
        )
    if background_weight < 0:
        raise ValueError("background_weight must be non-negative")
    if foreground_weight <= background_weight:
        raise ValueError("foreground_weight must be greater than background_weight")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")

    mask = _broadcast_foreground_mask(foreground_mask, prediction, channel_dim)
    weights = background_weight + (foreground_weight - background_weight) * mask
    weighted_squared_error = weights * (prediction - target).square()

    if reduction == "none":
        return weighted_squared_error
    if reduction == "sum":
        return weighted_squared_error.sum()
    return weighted_squared_error.mean()


def _scalar_on(value: Scalar, reference: Tensor, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        if value.device != reference.device:
            raise ValueError(f"{name} and embeddings must be on the same device")
        return value.to(dtype=reference.dtype)
    return reference.new_tensor(float(value))


def siglip_action_contrastive_loss(
    embeddings: Tensor,
    primitive_labels: Optional[Tensor] = None,
    *,
    pair_targets: Optional[Tensor] = None,
    temperature: Scalar = 10.0,
    bias: Scalar = -3.0,
    pair_mask: Optional[Tensor] = None,
    ignore_index: int = -1,
    exclude_self: bool = True,
) -> Tensor:
    """SigLIP-style action-centric pair loss (paper Eq. 10).

    ``primitive_labels`` creates positive pairs for equal labels and negative
    pairs for unequal labels.  Labels equal to ``ignore_index`` do not join any
    pair.  Alternatively, callers may supply a ``(B, B)`` ``pair_targets``
    matrix with values ``+1`` (positive), ``-1`` (negative), and ``0`` (ignore).
    The loss is the mean over all selected ordered pairs, exactly as in Eq. 10.
    """

    _validate_float_tensor("embeddings", embeddings, min_ndim=2)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape (B, D), got {tuple(embeddings.shape)}")
    batch_size, embedding_dim = embeddings.shape
    if batch_size < 1 or embedding_dim < 1:
        raise ValueError("embeddings must have non-empty batch and feature dimensions")
    if (primitive_labels is None) == (pair_targets is None):
        raise ValueError("provide exactly one of primitive_labels or pair_targets")

    if primitive_labels is not None:
        if not isinstance(primitive_labels, Tensor):
            raise TypeError("primitive_labels must be a torch.Tensor")
        if primitive_labels.shape != (batch_size,):
            raise ValueError(
                f"primitive_labels must have shape ({batch_size},), "
                f"got {tuple(primitive_labels.shape)}"
            )
        if primitive_labels.device != embeddings.device:
            raise ValueError("primitive_labels and embeddings must be on the same device")
        valid_label = primitive_labels != ignore_index
        targets = torch.where(
            primitive_labels[:, None] == primitive_labels[None, :],
            torch.ones((batch_size, batch_size), device=embeddings.device),
            -torch.ones((batch_size, batch_size), device=embeddings.device),
        )
        selected = valid_label[:, None] & valid_label[None, :]
    else:
        assert pair_targets is not None
        if not isinstance(pair_targets, Tensor):
            raise TypeError("pair_targets must be a torch.Tensor")
        if pair_targets.shape != (batch_size, batch_size):
            raise ValueError(
                f"pair_targets must have shape ({batch_size}, {batch_size}), "
                f"got {tuple(pair_targets.shape)}"
            )
        if pair_targets.device != embeddings.device:
            raise ValueError("pair_targets and embeddings must be on the same device")
        targets = pair_targets.to(dtype=embeddings.dtype)
        legal = (targets == -1) | (targets == 0) | (targets == 1)
        if not bool(legal.all()):
            raise ValueError("pair_targets values must be -1, 0, or +1")
        selected = targets != 0

    if exclude_self:
        selected = selected & ~torch.eye(
            batch_size, dtype=torch.bool, device=embeddings.device
        )
    if pair_mask is not None:
        if not isinstance(pair_mask, Tensor) or pair_mask.shape != (batch_size, batch_size):
            shape = getattr(pair_mask, "shape", None)
            raise ValueError(
                f"pair_mask must be a tensor of shape ({batch_size}, {batch_size}), got {shape}"
            )
        if pair_mask.device != embeddings.device:
            raise ValueError("pair_mask and embeddings must be on the same device")
        selected = selected & pair_mask.bool()

    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-8)
    tau = _scalar_on(temperature, normalized, name="temperature")
    offset = _scalar_on(bias, normalized, name="bias")
    if not bool(torch.isfinite(tau)) or float(tau.detach()) <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if not bool(torch.isfinite(offset)):
        raise ValueError("bias must be finite")

    logits = tau * (normalized @ normalized.transpose(0, 1)) + offset
    pair_losses = F.softplus(-targets.to(dtype=logits.dtype) * logits)
    if not bool(selected.any()):
        # Preserve the graph for distributed batches that happen to have no
        # labeled pair.
        return logits.sum() * 0.0
    return pair_losses[selected].mean()


class SigLIPActionHead(nn.Module):
    """Projection head with learned temperature and bias for Eq. 10."""

    def __init__(
        self,
        latent_dim: int,
        *,
        projection_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        initial_temperature: float = 10.0,
        initial_bias: float = -3.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        projection_dim = latent_dim if projection_dim is None else projection_dim
        hidden_dim = latent_dim if hidden_dim is None else hidden_dim
        if projection_dim <= 0 or hidden_dim <= 0:
            raise ValueError("projection_dim and hidden_dim must be positive")
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError("initial_temperature must be finite and positive")
        if not math.isfinite(initial_bias):
            raise ValueError("initial_bias must be finite")

        self.latent_dim = int(latent_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(projection_dim)),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, latents: Tensor) -> Tensor:
        _validate_float_tensor("latents", latents, min_ndim=2)
        if latents.ndim != 2 or latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latents must have shape (B, {self.latent_dim}), got {tuple(latents.shape)}"
            )
        return F.normalize(self.projection(latents.float()), dim=-1, eps=1e-8)

    def loss(
        self,
        latents: Tensor,
        primitive_labels: Optional[Tensor] = None,
        *,
        pair_targets: Optional[Tensor] = None,
        pair_mask: Optional[Tensor] = None,
        ignore_index: int = -1,
    ) -> Tensor:
        projected = self(latents)
        return siglip_action_contrastive_loss(
            projected,
            primitive_labels,
            pair_targets=pair_targets,
            temperature=self.temperature,
            bias=self.bias,
            pair_mask=pair_mask,
            ignore_index=ignore_index,
        )


def free_bits_kl_loss(
    mean: Tensor,
    log_variance: Tensor,
    *,
    free_bits: float = 0.5,
    reduction: str = "sum",
) -> Tensor:
    """KL to a unit Gaussian with a per-dimension free-bits floor.

    KL is first averaged over every leading (sample) dimension, then each
    latent dimension is clamped from below by ``free_bits``.  The default sums
    the resulting per-dimension terms, matching the CD-LAM training objective.
    """

    _validate_float_tensor("mean", mean)
    _validate_float_tensor("log_variance", log_variance)
    if mean.shape != log_variance.shape:
        raise ValueError(
            f"mean and log_variance must have identical shapes, got "
            f"{tuple(mean.shape)} and {tuple(log_variance.shape)}"
        )
    if mean.device != log_variance.device:
        raise ValueError("mean and log_variance must be on the same device")
    if mean.shape[-1] < 1:
        raise ValueError("latent dimension must be non-empty")
    if not math.isfinite(free_bits) or free_bits < 0:
        raise ValueError("free_bits must be finite and non-negative")
    if reduction not in {"sum", "mean", "none"}:
        raise ValueError("reduction must be one of 'sum', 'mean', or 'none'")

    mean_f = mean.float()
    log_variance_f = log_variance.float()
    stable_log_variance = log_variance_f.clamp(min=-30.0, max=20.0)
    per_sample_dimension = 0.5 * (
        mean_f.square() + stable_log_variance.exp() - 1.0 - log_variance_f
    )
    sample_dims = tuple(range(per_sample_dimension.ndim - 1))
    per_dimension = (
        per_sample_dimension.mean(dim=sample_dims)
        if sample_dims
        else per_sample_dimension
    )
    effective = per_dimension.clamp_min(float(free_bits))
    if reduction == "none":
        return effective
    if reduction == "mean":
        return effective.mean()
    return effective.sum()


def relative_zero_transition_loss(
    zero_latents: Tensor,
    transition_latents: Optional[Tensor] = None,
    *,
    transition_rms: Optional[Scalar] = None,
    margin: float = 0.05,
    epsilon: float = 1e-8,
) -> Tensor:
    """Relative zero-transition calibration loss (paper Eq. 12).

    The denominator is the stop-gradient RMS norm of ordinary transition
    latents.  Callers with a trainer-maintained running RMS can pass it through
    ``transition_rms``; otherwise it is estimated from ``transition_latents``.
    Exactly one source must be provided.
    """

    _validate_float_tensor("zero_latents", zero_latents)
    if zero_latents.shape[-1] < 1:
        raise ValueError("zero_latents must have a non-empty latent dimension")
    if (transition_latents is None) == (transition_rms is None):
        raise ValueError("provide exactly one of transition_latents or transition_rms")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")

    if transition_latents is not None:
        _validate_float_tensor("transition_latents", transition_latents)
        if transition_latents.shape[-1] != zero_latents.shape[-1]:
            raise ValueError(
                "transition_latents and zero_latents must share the latent dimension, "
                f"got {transition_latents.shape[-1]} and {zero_latents.shape[-1]}"
            )
        if transition_latents.device != zero_latents.device:
            raise ValueError("transition_latents and zero_latents must be on the same device")
        ordinary_norm_sq = transition_latents.float().square().sum(dim=-1)
        scale = ordinary_norm_sq.mean().sqrt().detach()
    else:
        assert transition_rms is not None
        scale = _scalar_on(transition_rms, zero_latents.float(), name="transition_rms")
        scale = scale.detach()

    if not bool(torch.isfinite(scale)) or float(scale) < 0:
        raise ValueError("transition RMS must be finite and non-negative")
    if float(scale) == 0.0 and epsilon == 0.0:
        raise ValueError("transition RMS and epsilon cannot both be zero")

    zero_norm = torch.linalg.vector_norm(zero_latents.float(), dim=-1)
    relative_norm = zero_norm / (scale + float(epsilon))
    return F.relu(relative_norm - float(margin)).square().mean()


def latent_space_calibration_loss(
    mean: Tensor,
    log_variance: Tensor,
    zero_latents: Tensor,
    transition_latents: Optional[Tensor] = None,
    *,
    transition_rms: Optional[Scalar] = None,
    free_bits: float = 0.5,
    zero_margin: float = 0.05,
    epsilon: float = 1e-8,
) -> Tensor:
    """Combined free-bits KL and relative zero-transition calibration (Eq. 11)."""

    return free_bits_kl_loss(mean, log_variance, free_bits=free_bits) + relative_zero_transition_loss(
        zero_latents,
        transition_latents,
        transition_rms=transition_rms,
        margin=zero_margin,
        epsilon=epsilon,
    )


# Compact aliases used in some training integrations.
embodiment_reconstruction_loss = embodiment_centric_reconstruction_loss
action_contrastive_loss = siglip_action_contrastive_loss


__all__ = [
    "SigLIPActionHead",
    "action_contrastive_loss",
    "embodiment_centric_reconstruction_loss",
    "embodiment_reconstruction_loss",
    "free_bits_kl_loss",
    "latent_space_calibration_loss",
    "relative_zero_transition_loss",
    "siglip_action_contrastive_loss",
]
