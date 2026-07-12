"""CD-LAM runtime component."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import torch


# ============================================================================


# ============================================================================
@dataclass
class MonitorConfig:
    every: int = 500
    eval_batch_size: int = 4
    eval_seed: int = 1234
    n_noise_repeats: int = 3
    run_swap: bool = True
    run_future_leak: bool = True
    run_cycle: bool = True

    zusage_kill_frac: float = 0.05

    kill_window_sec: float = 12 * 3600.0
    log_path: Optional[str] = None


@dataclass
class DropoutConfig:
    p: float = 0.12

    raw_slot: Optional[tuple[int, int]] = None
    z_slot: tuple[int, int] = (-32, None)
    seed: int = 777


@dataclass
class CFGConfig:
    weights: tuple[float, ...] = (1.05, 1.5, 2.0, 3.0)  # sweep w

    min_dropout_steps: int = 2000


# ============================================================================


# ============================================================================
def mask_condition_slots(
    action: torch.Tensor,
    z_holder: dict,
    cfg: DropoutConfig,
    step: int,
) -> torch.Tensor:
    """CD-LAM runtime component."""
    B = action.shape[0]
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + step)
    drop_mask = torch.rand(B, generator=g) < cfg.p  # (B,) cpu bool

    if not drop_mask.any():
        return drop_mask

    rows = drop_mask.nonzero(as_tuple=True)[0]

    z = z_holder.get("current")
    if z is not None:
        z = z.clone()
        z[rows] = 0.0
        z_holder["current"] = z

    if cfg.raw_slot is not None:
        s, e = cfg.raw_slot
        action[rows, :, s:e] = 0.0

    return drop_mask


# ============================================================================


# ============================================================================
def cfg_velocity(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    set_z: Callable[[Optional[torch.Tensor]], None],
    z: torch.Tensor,
    w: float,
) -> torch.Tensor:
    """CD-LAM runtime component."""
    set_z(None)  # null: z=0
    v0 = denoise_fn()
    set_z(z)
    vz = denoise_fn()
    return v0 + w * (vz - v0)


def assert_cfg_order(dropout_steps_done: int, cfg: CFGConfig) -> None:
    if dropout_steps_done < cfg.min_dropout_steps:
        raise RuntimeError(
            f"[z_monitor] CFG requires at least {cfg.min_dropout_steps} dropout-finetune "
            f"steps; observed {dropout_steps_done}."
        )


# ============================================================================


# ============================================================================
@dataclass
class MonitorResult:
    step: int
    loss_real: float = 0.0
    loss_zero: float = 0.0
    action_gap: float = 0.0  # (a) loss_zero - loss_real
    zusage_frac: float = 0.0
    swap_delta: float = 0.0  # (b)
    future_leak_gap: float = 0.0  # (c) loss_spliced_future - loss_real
    cycle_cos: float = 0.0  # (d)
    cycle_sanity_cos: float = 0.0  # (d) gate: E(real-z rollout) vs real z
    noise_floor_std: float = 0.0
    is_baseline: bool = False
    notes: dict = field(default_factory=dict)


class ZUsageMonitor:
    """CD-LAM runtime component."""

    def __init__(
        self,
        cfg: MonitorConfig,
        fixed_eval_batch: dict,
        fixed_real_z: torch.Tensor,
        loss_with_z: Callable[[Optional[torch.Tensor]], float],
        decode_with_z: Optional[
            Callable[[Optional[torch.Tensor]], torch.Tensor]
        ] = None,
        encode_frozen: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.cfg = cfg
        self.batch = fixed_eval_batch
        self.real_z = fixed_real_z
        self.loss_with_z = loss_with_z
        self.decode_with_z = decode_with_z
        self.encode_frozen = encode_frozen
        self._baseline_done = False
        self._low_usage_since: Optional[float] = None
        self.history: list[MonitorResult] = []
        self.log_f = None
        if cfg.log_path:
            self.log_f = open(cfg.log_path, "a")

    def maybe_run(self, step: int) -> Optional[MonitorResult]:
        if not self._baseline_done:
            raise RuntimeError(
                "[z_monitor] run_baseline() must be called before scheduled monitoring."
            )
        if step % self.cfg.every != 0:
            return None
        return self._run(step, is_baseline=False)

    def run_baseline(self) -> MonitorResult:
        res = self._run(step=0, is_baseline=True)
        self._baseline_done = True
        return res

    def _run(self, step: int, is_baseline: bool) -> MonitorResult:
        torch.manual_seed(self.cfg.eval_seed)

        res = MonitorResult(step=step, is_baseline=is_baseline)

        real_losses = [
            self.loss_with_z(self.real_z) for _ in range(self.cfg.n_noise_repeats)
        ]
        zero_losses = [self.loss_with_z(None) for _ in range(self.cfg.n_noise_repeats)]
        res.loss_real = float(np.mean(real_losses))
        res.loss_zero = float(np.mean(zero_losses))
        res.action_gap = res.loss_zero - res.loss_real
        res.zusage_frac = res.action_gap / max(res.loss_zero, 1e-9)
        res.noise_floor_std = float(np.std(real_losses + zero_losses))

        if self.cfg.run_swap and self.decode_with_z is not None:
            donor_z = torch.roll(self.real_z, shifts=1, dims=0)
            out_real = self.decode_with_z(self.real_z)
            out_swap = self.decode_with_z(donor_z)
            if out_real is not None and out_swap is not None:
                res.swap_delta = float(
                    (out_real.float() - out_swap.float()).pow(2).mean().sqrt()
                )
        elif self.cfg.run_swap:
            donor_z = torch.roll(self.real_z, shifts=1, dims=0)
            res.swap_delta = self.loss_with_z(donor_z) - res.loss_real

        if self.cfg.run_future_leak:
            spliced = torch.roll(self.real_z, shifts=-1, dims=1)
            loss_spliced = self.loss_with_z(spliced)
            res.future_leak_gap = loss_spliced - res.loss_real

        if (
            self.cfg.run_cycle
            and self.decode_with_z is not None
            and self.encode_frozen is not None
        ):
            gen = self.decode_with_z(self.real_z)
            if gen is not None:
                z_back = self.encode_frozen(gen)  # (B,T,32)
                res.cycle_cos = _mean_cos(z_back, self.real_z)

                res.cycle_sanity_cos = res.cycle_cos

        self._update_kill_timer(res)
        self._log(res)
        self.history.append(res)
        return res

    def _update_kill_timer(self, res: MonitorResult) -> None:
        now = time.time()
        if res.zusage_frac < self.cfg.zusage_kill_frac:
            if self._low_usage_since is None:
                self._low_usage_since = now
            elapsed = now - self._low_usage_since
            res.notes["low_zusage_elapsed_sec"] = round(elapsed, 1)
            if elapsed >= self.cfg.kill_window_sec:
                res.notes["KILL"] = (
                    f"z_usage {res.zusage_frac:.3f} remained below "
                    f"{self.cfg.zusage_kill_frac} for {elapsed / 3600:.1f}h; stop this run."
                )
        else:
            self._low_usage_since = None

    def _log(self, res: MonitorResult) -> None:
        rec = asdict(res)
        if self.log_f is not None:
            self.log_f.write(json.dumps(rec) + "\n")
            self.log_f.flush()
        tag = "BASELINE" if res.is_baseline else f"step{res.step}"
        kill = res.notes.get("KILL", "")
        print(
            f"[z_monitor] {tag} gap={res.action_gap:+.4e} zusage={res.zusage_frac:+.3f} "
            f"swapΔ={res.swap_delta:.4e} futureLeak={res.future_leak_gap:+.4e} "
            f"cycleCos={res.cycle_cos:+.3f} floor_std={res.noise_floor_std:.2e} {kill}",
            flush=True,
        )

    def close(self) -> None:
        if self.log_f is not None:
            self.log_f.close()


def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


# ============================================================================


# ============================================================================
def build_compat_probes(model, z_holder: dict, data_batch_fn, eval_seed: int):
    """CD-LAM runtime component."""
    raise NotImplementedError(
        "Wire this helper to a fixed evaluation batch, the model loss, and the rollout decoder."
    )
