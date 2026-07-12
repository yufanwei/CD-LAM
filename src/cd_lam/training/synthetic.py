"""Deterministic CPU training backends used only for integration smoke tests."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn

from ..bridge import (
    ACTION_DIM,
    LATENT_DIM,
    ActionToLatentBridge,
    build_bridge_mlp,
    prepare_latent_condition,
)
from ..objectives import (
    SigLIPActionHead,
    embodiment_centric_reconstruction_loss,
    latent_space_calibration_loss,
)
from .common import (
    StageContext,
    StageExecutionError,
    StageResult,
    checkpoint_metadata,
    file_sha256,
    load_resume_checkpoint,
    make_result,
)


SYNTHETIC_ADAPTER_ID = "cd_lam.synthetic_cpu"


class _TinyLAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(128, 64), nn.GELU())
        self.mean = nn.Linear(64, LATENT_DIM)
        self.log_variance = nn.Linear(64, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(64 + LATENT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

    def encode(self, current: Tensor, future: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(
            torch.cat([current.flatten(1), future.flatten(1)], dim=-1)
        )
        return self.mean(hidden), self.log_variance(hidden).clamp(-4.0, 4.0)

    def reconstruct(self, current: Tensor, latent: Tensor) -> Tensor:
        decoded = self.decoder(torch.cat([current.flatten(1), latent], dim=-1))
        return decoded.reshape_as(current)


class _TinyACWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16 + LATENT_DIM, 48),
            nn.GELU(),
            nn.Linear(48, 16),
        )

    def forward(self, current: Tensor, latent: Tensor) -> Tensor:
        return self.network(torch.cat([current, latent], dim=-1))


def _initialize(seed: int) -> torch.Generator:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _require_synthetic(context: StageContext) -> None:
    if context.plan.mode != "synthetic" or context.plan.device != "cpu":
        raise StageExecutionError("the built-in synthetic backend is CPU-only")
    if context.plan.adapter_identity != SYNTHETIC_ADAPTER_ID:
        raise StageExecutionError("synthetic adapter identity changed after planning")


def _upstream_seed(context: StageContext, name: str, fallback: int) -> int:
    path = context.upstream_artifacts.get(name)
    if path is None:
        return fallback
    return int(file_sha256(path)[:8], 16)


def _save(context: StageContext, payload: dict) -> None:
    context.plan.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload["metadata"] = checkpoint_metadata(
        context, completed_steps=context.plan.target_steps
    )
    torch.save(payload, context.plan.output_checkpoint)


def _optimize_fixed_batch(
    *,
    context: StageContext,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[], Tensor],
    start_step: int,
) -> tuple[float, list[float]]:
    with torch.no_grad():
        initial_loss = float(loss_fn().detach())
    losses: list[float] = []
    for _ in range(start_step, context.plan.target_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        if not bool(torch.isfinite(loss)):
            raise StageExecutionError("synthetic training produced a non-finite loss")
        loss.backward()
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        if not gradients or not any(gradient is not None for gradient in gradients):
            raise StageExecutionError("synthetic optimizer step had no gradients")
        optimizer.step()
        with torch.no_grad():
            losses.append(float(loss_fn().detach()))
    return initial_loss, losses


def run_stage1(context: StageContext) -> StageResult:
    """Run real backward/optimizer steps through all three Stage-1 objectives."""

    _require_synthetic(context)
    config = context.config.stage1
    generator = _initialize(context.plan.seed)
    batch = max(4, min(context.plan.batch_size, 8))
    current = torch.randn(batch, 1, 8, 8, generator=generator) * 0.1
    labels = torch.arange(batch) % 3
    mask = torch.zeros(batch, 8, 8)
    mask[:, 2:6, 2:6] = 1.0
    motion = (labels.float() + 1.0).view(batch, 1, 1, 1) * mask[:, None] * 0.03
    future = current + motion

    model = _TinyLAM()
    head = SigLIPActionHead(LATENT_DIM, projection_dim=16, hidden_dim=32)
    optimizer = torch.optim.Adam(
        [*model.parameters(), *head.parameters()], lr=context.plan.learning_rate
    )
    resume, start_step = load_resume_checkpoint(context)
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
        head.load_state_dict(resume["head_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])

    def loss_fn() -> Tensor:
        mean, log_variance = model.encode(current, future)
        zero_mean, _ = model.encode(current, current)
        prediction = model.reconstruct(current, mean)
        reconstruction = embodiment_centric_reconstruction_loss(
            prediction,
            future,
            mask,
            foreground_weight=config.foreground_weight,
            background_weight=config.background_weight,
        )
        contrastive = head.loss(mean, labels)
        calibration = latent_space_calibration_loss(
            mean,
            log_variance,
            zero_mean,
            mean,
            free_bits=config.free_bits,
            zero_margin=config.zero_margin,
        )
        return (
            reconstruction
            + config.contrastive_weight * contrastive
            + config.calibration_weight * calibration
        )

    initial_loss, losses = _optimize_fixed_batch(
        context=context,
        optimizer=optimizer,
        loss_fn=loss_fn,
        start_step=start_step,
    )
    _save(
        context,
        {
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
    )
    return make_result(
        context,
        start_step=start_step,
        initial_loss=initial_loss,
        losses=losses,
        metrics={"batch_size": float(batch)},
    )


def _acwm_batch(
    context: StageContext, *, dependency: str, seed_offset: int
) -> tuple[Tensor, Tensor, Tensor]:
    dependency_seed = _upstream_seed(
        context, dependency, context.plan.seed + seed_offset
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(dependency_seed)
    batch = max(4, min(context.plan.batch_size, 16))
    current = torch.randn(batch, 16, generator=generator) * 0.2
    latent = torch.randn(batch, LATENT_DIM, generator=generator) * 0.2
    true_weight = torch.randn(LATENT_DIM, 16, generator=generator) * 0.05
    target = current + latent @ true_weight
    return current, latent, target


def run_stage2(context: StageContext) -> StageResult:
    """Run a tiny latent-conditioned world-model optimization on CPU."""

    _require_synthetic(context)
    _initialize(context.plan.seed)
    current, latent, target = _acwm_batch(
        context, dependency="stage1_lam", seed_offset=101
    )
    model = _TinyACWM()
    optimizer = torch.optim.Adam(model.parameters(), lr=context.plan.learning_rate)
    resume, start_step = load_resume_checkpoint(context)
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])

    def loss_fn() -> Tensor:
        return torch.nn.functional.mse_loss(model(current, latent), target)

    initial_loss, losses = _optimize_fixed_batch(
        context=context,
        optimizer=optimizer,
        loss_fn=loss_fn,
        start_step=start_step,
    )
    _save(
        context,
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_scope": context.config.stage2.training_scope,
        },
    )
    return make_result(
        context,
        start_step=start_step,
        initial_loss=initial_loss,
        losses=losses,
        metrics={"latent_norm": float(latent.norm(dim=-1).mean())},
    )


def _bridge_batch(
    context: StageContext,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dependency_seed = _upstream_seed(context, "stage1_lam", context.plan.seed + 211)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(dependency_seed)
    batch = max(8, min(context.plan.batch_size, 32))
    actions = torch.randn(batch, ACTION_DIM, generator=generator)
    true_weight = torch.randn(ACTION_DIM, LATENT_DIM, generator=generator) * 0.15
    latents = torch.tanh(actions @ true_weight)
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0, unbiased=False).clamp_min(1e-3)
    latent_mean = latents.mean(dim=0)
    latent_std = latents.std(dim=0, unbiased=False).clamp_min(1e-3)
    normalized_actions = (actions - action_mean) / action_std
    normalized_latents = (latents - latent_mean) / latent_std
    return (
        normalized_actions,
        normalized_latents,
        action_mean,
        action_std,
        latent_mean,
        latent_std,
    )


def run_bridge_training(context: StageContext) -> StageResult:
    """Train the exact public 22D-to-32D bridge on deterministic synthetic pairs."""

    _require_synthetic(context)
    _initialize(context.plan.seed)
    (
        actions,
        latents,
        action_mean,
        action_std,
        latent_mean,
        latent_std,
    ) = _bridge_batch(context)
    bridge_mlp = build_bridge_mlp()
    optimizer = torch.optim.Adam(bridge_mlp.parameters(), lr=context.plan.learning_rate)
    resume, start_step = load_resume_checkpoint(context)
    if resume is not None:
        bridge_mlp.load_state_dict(resume["g_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])

    def loss_fn() -> Tensor:
        return torch.nn.functional.mse_loss(bridge_mlp(actions), latents)

    initial_loss, losses = _optimize_fixed_batch(
        context=context,
        optimizer=optimizer,
        loss_fn=loss_fn,
        start_step=start_step,
    )
    _save(
        context,
        {
            "g_state": bridge_mlp.state_dict(),
            "action_mean": action_mean,
            "action_std": action_std,
            "zm": latent_mean,
            "zsd": latent_std,
            "latent_dim": LATENT_DIM,
            "action_transform_id": context.plan.action_transform_id,
            "source_stride": context.plan.source_stride,
            "optimizer_state": optimizer.state_dict(),
        },
    )
    return make_result(
        context,
        start_step=start_step,
        initial_loss=initial_loss,
        losses=losses,
        metrics={"normalized_target_rms": float(latents.square().mean().sqrt())},
    )


def _standalone_bridge_bundle(seed: int) -> dict:
    _initialize(seed)
    return {
        "g_state": build_bridge_mlp().state_dict(),
        "action_mean": torch.zeros(ACTION_DIM),
        "action_std": torch.ones(ACTION_DIM),
        "zm": torch.zeros(LATENT_DIM),
        "zsd": torch.ones(LATENT_DIM),
        "latent_dim": LATENT_DIM,
    }


def run_stage3(context: StageContext) -> StageResult:
    """Optimize a tiny ACWM through the exact robot-action bridge condition path."""

    _require_synthetic(context)
    generator = _initialize(context.plan.seed)
    model = _TinyACWM()
    stage2_path = context.upstream_artifacts.get("stage2_acwm")
    if stage2_path is not None:
        stage2_blob = torch.load(stage2_path, map_location="cpu", weights_only=False)
        model.load_state_dict(stage2_blob["model_state"], strict=True)
    bridge_path = context.upstream_artifacts.get("bridge_bundle")
    if bridge_path is not None:
        bridge_source = torch.load(bridge_path, map_location="cpu", weights_only=False)
        if bridge_source.get("action_transform_id") != context.plan.action_transform_id:
            raise StageExecutionError("synthetic bridge action_transform_id mismatch")
        if bridge_source.get("source_stride") != context.plan.source_stride:
            raise StageExecutionError("synthetic bridge source_stride mismatch")
    else:
        bridge_source = _standalone_bridge_bundle(context.plan.seed + 307)
    bridge = ActionToLatentBridge(bridge_source)

    batch = max(4, min(context.plan.batch_size, 16))
    current = torch.randn(batch, 16, generator=generator) * 0.2
    actions = torch.randn(batch, ACTION_DIM, generator=generator) * 0.2
    latent = prepare_latent_condition(robot_action=actions, bridge=bridge)
    true_weight = torch.randn(LATENT_DIM, 16, generator=generator) * 0.05
    target = current + latent.detach() @ true_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=context.plan.learning_rate)
    resume, start_step = load_resume_checkpoint(context)
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])

    def loss_fn() -> Tensor:
        condition = prepare_latent_condition(robot_action=actions, bridge=bridge)
        return torch.nn.functional.mse_loss(model(current, condition), target)

    initial_loss, losses = _optimize_fixed_batch(
        context=context,
        optimizer=optimizer,
        loss_fn=loss_fn,
        start_step=start_step,
    )
    _save(
        context,
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "bridge_metadata": {
                "action_transform_id": context.plan.action_transform_id,
                "source_stride": context.plan.source_stride,
            },
        },
    )
    if not math.isfinite(float(latent.norm())):
        raise StageExecutionError("Stage3 bridge produced a non-finite condition")
    return make_result(
        context,
        start_step=start_step,
        initial_loss=initial_loss,
        losses=losses,
        metrics={"condition_norm": float(latent.norm(dim=-1).mean())},
    )


__all__ = [
    "SYNTHETIC_ADAPTER_ID",
    "run_bridge_training",
    "run_stage1",
    "run_stage2",
    "run_stage3",
]
