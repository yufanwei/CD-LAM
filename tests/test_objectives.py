from __future__ import annotations

import math

import pytest
import torch

from cd_lam.objectives import (
    SigLIPActionHead,
    embodiment_centric_reconstruction_loss,
    free_bits_kl_loss,
    latent_space_calibration_loss,
    relative_zero_transition_loss,
    siglip_action_contrastive_loss,
)


def test_embodiment_reconstruction_has_analytical_weighted_value() -> None:
    prediction = torch.zeros(1, 1, 1, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.tensor([[[1.0, 0.0]]])
    loss = embodiment_centric_reconstruction_loss(
        prediction,
        target,
        mask,
        foreground_weight=3.0,
        background_weight=1.0,
    )
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_siglip_equation_on_one_positive_and_one_negative_pair() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pair_targets = torch.zeros(3, 3)
    pair_targets[0, 1] = 1
    pair_targets[0, 2] = -1
    loss = siglip_action_contrastive_loss(
        embeddings,
        pair_targets=pair_targets,
        temperature=2.0,
        bias=0.0,
    )
    expected = 0.5 * (torch.nn.functional.softplus(torch.tensor(-2.0)) + math.log(2.0))
    assert loss.item() == pytest.approx(float(expected))


def test_objectives_smoke_and_backprop() -> None:
    torch.manual_seed(11)
    latents = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    head = SigLIPActionHead(8, projection_dim=6, hidden_dim=10)

    mean = torch.randn(6, 8, requires_grad=True)
    log_variance = torch.randn(6, 8, requires_grad=True) * 0.1
    zero_latents = torch.randn(6, 8, requires_grad=True) * 0.05
    transition_latents = torch.randn(6, 8)

    contrastive = head.loss(latents, labels)
    calibration = latent_space_calibration_loss(
        mean,
        log_variance,
        zero_latents,
        transition_latents,
        free_bits=0.01,
        zero_margin=0.01,
    )
    loss = contrastive + calibration
    assert torch.isfinite(loss)
    loss.backward()
    assert latents.grad is not None and torch.isfinite(latents.grad).all()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert head.log_temperature.grad is not None
    assert all(parameter.grad is not None for parameter in head.projection.parameters())


def test_free_bits_and_relative_zero_calibration_analytical_cases() -> None:
    mean = torch.zeros(3, 2)
    log_variance = torch.zeros_like(mean)
    assert free_bits_kl_loss(mean, log_variance, free_bits=0.5).item() == pytest.approx(1.0)

    zero = torch.tensor([[3.0, 4.0]])
    relative = relative_zero_transition_loss(
        zero,
        transition_rms=10.0,
        margin=0.2,
        epsilon=0.0,
    )
    assert relative.item() == pytest.approx(0.09)


def test_objective_shape_checks_are_explicit() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        embodiment_centric_reconstruction_loss(
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 3, 5, 4),
            torch.zeros(1, 4, 4),
        )
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        siglip_action_contrastive_loss(
            torch.randn(3, 4), torch.tensor([0, 1])
        )
    with pytest.raises(ValueError, match="exactly one"):
        relative_zero_transition_loss(torch.randn(2, 4))
