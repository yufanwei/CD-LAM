#!/usr/bin/env python
"""CD-LAM runtime component."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT))

from cdlam_integration.stage3.support import (  # noqa: E402
    SLICE_BY_EMBODIMENT,
    _avg_scalar,
    _json_default,
    _load_trainable_state,
    _move_batch,
    _patch_sync_model_states_for_single_rank,
    _save_trainable_ckpt,
    _setup_distributed,
)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def _build_mlp(di: int, do: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(di, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, do),
    )


class FrozenBridge:
    """CD-LAM runtime component."""

    def __init__(self, ckpt_path: Path, device: torch.device):
        blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        g_state = blob["g_state"]
        in_dim = g_state["0.weight"].shape[1]
        out_dim = g_state["4.weight"].shape[0]
        self.in_dim = int(in_dim)
        self.latent_dim = int(out_dim)
        self.g = _build_mlp(in_dim, out_dim)
        self.g.load_state_dict(g_state, strict=True)
        self.g.to(device=device, dtype=torch.float32).eval()
        for p in self.g.parameters():
            p.requires_grad_(False)
        am = np.asarray(blob["action_mean"], dtype=np.float32)
        asd = np.asarray(blob["action_std"], dtype=np.float32)
        asd = np.where(asd < 1e-6, 1.0, asd)
        self.action_mean = torch.tensor(am, device=device, dtype=torch.float32)
        self.action_std = torch.tensor(asd, device=device, dtype=torch.float32)
        self.zm = torch.tensor(
            np.asarray(blob["zm"], dtype=np.float32), device=device, dtype=torch.float32
        )
        self.zsd = torch.tensor(
            np.asarray(blob["zsd"], dtype=np.float32),
            device=device,
            dtype=torch.float32,
        )
        self.meta = {
            "ckpt": str(ckpt_path),
            "arch": blob.get("arch"),
            "in_dim": self.in_dim,
            "latent_dim": self.latent_dim,
            "lam_ckpt": blob.get("lam_ckpt"),
            "robot": blob.get("robot"),
            "lambdas": blob.get("lambdas"),
            "eval": blob.get("eval"),
        }

    @torch.no_grad()
    def z_from_action(self, a: torch.Tensor) -> torch.Tensor:
        """CD-LAM runtime component."""
        a = a.to(dtype=torch.float32)
        an = (a - self.action_mean) / self.action_std
        zn = self.g(an)
        return zn * self.zsd + self.zm


def _inject_z(
    action: torch.Tensor, bridge: FrozenBridge, src_slice, latent_slice
) -> dict:
    """CD-LAM runtime component."""
    s0, s1 = src_slice
    l0, l1 = latent_slice
    a_src = action[..., s0:s1]
    z = bridge.z_from_action(a_src)  # (B,T,32) float32
    src_abs = float(a_src.abs().sum().item())
    action.zero_()
    action[..., l0:l1] = 1.0
    z_bf = z.to(dtype=action.dtype)
    return {
        "z": z_bf,
        "src_abs_sum": src_abs,
        "z_abs_mean": float(z.abs().mean().item()),
        "z_std": float(z.std().item()),
    }


def _average_gradients(params: list[torch.nn.Parameter], world_size: int) -> None:
    if world_size <= 1:
        return
    for p in params:
        if p.grad is None:
            continue
        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
        p.grad.div_(world_size)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def _run_z_eval(
    model,
    eval_batch,
    bridge,
    src_slice,
    latent_slice,
    z_holder,
    seed: int,
    world_size: int,
    device: torch.device,
) -> dict:
    """CD-LAM runtime component."""
    l0, l1 = latent_slice
    result = {}
    was_training = model.training
    model.eval()
    base_action = eval_batch["action"].detach()
    with torch.no_grad():
        a_src = base_action[..., src_slice[0] : src_slice[1]]
        z_own = bridge.z_from_action(a_src).to(dtype=base_action.dtype)
        modes = {
            "own": z_own,
            "zero_z": torch.zeros_like(z_own),
            "shuffle_time_z": z_own[
                :, torch.randperm(z_own.shape[1], device=z_own.device), :
            ],
        }
        for name, zz in modes.items():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            act = torch.zeros_like(base_action)
            act[..., l0:l1] = 1.0
            z_holder["current"] = zz
            b = {k: v for k, v in eval_batch.items()}
            b["action"] = act
            b["num_conditional_frames"] = 1
            _, loss = model(b)
            result[name] = _avg_scalar(float(loss.detach().item()), device, world_size)
    if was_training:
        model.train()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "CDLAM_AGIBOT_DATASET_PATH", str(ROOT / "datasets/agibot")
        ),
    )
    ap.add_argument(
        "--embodiment", choices=sorted(SLICE_BY_EMBODIMENT), default="agibot"
    )
    ap.add_argument(
        "--base-ckpt",
        default=os.environ.get(
            "CDLAM_BASE_2B_CKPT",
            str(ROOT / "lammodel/checkpoints/CD-LAM/2B_pretrain/iter_000140000/model"),
        ),
    )
    ap.add_argument("--experiment", default="dreamdojo_2b_480_640_agibot_local_full")
    ap.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    ap.add_argument(
        "--init-trainable-from",
        required=True,
        help="release pretrained WM, e.g. outputs/pretrain/ckpt_step4000.pt",
    )
    ap.add_argument(
        "--scope", choices=["A", "A2", "B", "B_old", "B2", "E", "C", "D"], default="D"
    )
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=11)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.99)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--disable-z-eval",
        action="store_true",
        help="Disable inline z perturbation eval; useful when an external eval watcher owns GPU eval.",
    )
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", required=True)

    ap.add_argument(
        "--gr-bridge-ckpt",
        default=str(
            ROOT / "outputs/bridge/a22z_agibot_beta_D_cyc3.pt"
        ),
        help="Frozen a22z D_cyc3 bridge; the default AgiBot input is action_22 at [147:169].",
    )
    ap.add_argument(
        "--latent-slice", default="352,384", help="World-model latent-z slot"
    )
    ap.add_argument(
        "--extra-save-steps",
        default="13334",
        help="Comma-separated extra checkpoint steps.",
    )
    ap.add_argument(
        "--free-inline-lam",
        action="store_true",
        help="Release the inline LAM after injection to reduce memory use.",
    )
    ap.add_argument(
        "--parallelism",
        choices=["fsdp", "manual-ddp"],
        default=os.environ.get("GBRIDGE_PARALLELISM", "fsdp"),
        help="Multi-GPU mode: fsdp shards the model; manual-ddp uses full replicas and explicit gradient reduction.",
    )
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank = _setup_distributed()
    is_rank0 = rank == 0
    device = torch.device("cuda", local_rank)
    _patch_sync_model_states_for_single_rank()

    src_slice = SLICE_BY_EMBODIMENT[args.embodiment]  # agibot -> (147,169)
    latent_slice = tuple(int(x) for x in args.latent_slice.split(","))
    extra_saves = {int(x) for x in args.extra_save_steps.split(",") if x.strip()}

    if is_rank0:
        print(
            f"[gbridge_pt] rank={rank}/{world_size} local={local_rank} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"src_slice={src_slice} latent_slice={latent_slice}",
            flush=True,
        )

    torch.manual_seed(args.seed + rank * 100003)
    torch.cuda.manual_seed_all(args.seed + rank * 100003)

    from cosmos_predict2._src.predict2.utils.model_loader import (
        load_model_from_checkpoint,
    )
    from groot_dreams.dataloader import MultiVideoActionDataset
    from cdlam_integration.world_model.scope import configure_scope
    from cdlam_integration.world_model.train import patch_forward_to_use_cached_z

    if is_rank0:
        print(
            f"[gbridge_pt] loading WM base experiment={args.experiment} base={args.base_ckpt}",
            flush=True,
        )
    t0 = time.time()
    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.base_ckpt,
        config_file=args.config_file,
        load_ema_to_reg=True,
        skip_load_model=False,
    )
    if is_rank0:
        n_net = sum(p.numel() for p in model.net.parameters())
        print(
            f"[gbridge_pt] WM loaded {time.time() - t0:.1f}s net_params={n_net / 1e9:.3f}B",
            flush=True,
        )
    if getattr(model, "lam", None) is not None:
        model.lam.to(device)

    train_params, n_train = configure_scope(model, args.scope)

    overlay_summary = _load_trainable_state(
        model, Path(args.init_trainable_from), is_rank0
    )

    if world_size > 1 and args.parallelism == "fsdp":
        from cosmos_predict2._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh

        dp_mesh = hsdp_device_mesh(replica_group_size=1, sharding_group_size=world_size)
        model.apply_fsdp(dp_mesh)
        train_params = [p for p in model.net.parameters() if p.requires_grad]
    elif world_size > 1 and args.parallelism == "manual-ddp" and is_rank0:
        print(
            "[gbridge_pt] parallelism=manual-ddp: no FSDP wrapping; gradients all-reduced after backward",
            flush=True,
        )
    n_train_after = sum(p.numel() for p in train_params if p.requires_grad)
    if is_rank0:
        print(
            f"[gbridge_pt] scope={args.scope} trainable={n_train_after / 1e9:.3f}B "
            f"({n_train_after:,}) world_size={world_size}",
            flush=True,
        )

    bridge = FrozenBridge(Path(args.gr_bridge_ckpt), device)
    if is_rank0:
        print(
            f"[gbridge_pt] bridge loaded: {json.dumps(bridge.meta, default=_json_default)}",
            flush=True,
        )
        assert bridge.latent_dim == (latent_slice[1] - latent_slice[0]), (
            f"latent dim {bridge.latent_dim} != slot {latent_slice}"
        )
        assert bridge.in_dim == (src_slice[1] - src_slice[0]), (
            f"bridge in_dim {bridge.in_dim} != action slot {src_slice}"
        )
    z_holder = {"current": None}
    patch_forward_to_use_cached_z(model, z_holder, fix_noise=False)
    if args.free_inline_lam and getattr(model, "lam", None) is not None:
        model.lam = None
        gc.collect()
        torch.cuda.empty_cache()
        if is_rank0:
            print(
                "[gbridge_pt] freed inline LAM (patched forward bypasses it)",
                flush=True,
            )

    dataset = MultiVideoActionDataset(
        dataset_path=args.dataset_path,
        num_frames=13,
        data_split="train",
        single_base_index=False,
        deterministic_uniform_sampling=False,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    iterator = iter(loader)

    opt = torch.optim.AdamW(
        train_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    log_f = open(out_dir / "train_log.jsonl", "w") if is_rank0 else None
    summary_path = out_dir / "summary.json"
    losses: list[float] = []
    eval_history: list[dict] = []
    last_inj: dict = {}

    try:
        model.train()
        target_step = args.step_offset + args.steps
        for local_step in range(1, args.steps + 1):
            step = args.step_offset + local_step
            sampler.set_epoch(step // max(1, len(loader)))
            t_step = time.time()
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.set_epoch(step)
                iterator = iter(loader)
                batch = next(iterator)
            t_data = time.time()
            batch = _move_batch(batch, device)

            raw_action = batch["action"].detach().clone()
            src_abs = float(
                raw_action[..., src_slice[0] : src_slice[1]].abs().sum().item()
            )
            if src_abs <= 0.0:
                raise RuntimeError(
                    f"empty agibot action slot {src_slice} at step {step}"
                )

            inj = _inject_z(batch["action"], bridge, src_slice, latent_slice)
            z_holder["current"] = inj["z"]
            last_inj = {k: inj[k] for k in ("src_abs_sum", "z_abs_mean", "z_std")}

            lr_now = args.lr
            if args.warmup_steps > 0:
                lr_now = args.lr * min(1.0, step / args.warmup_steps)
                for group in opt.param_groups:
                    group["lr"] = lr_now

            torch.manual_seed(args.seed + step + rank * 100003)
            torch.cuda.manual_seed_all(args.seed + step + rank * 100003)
            _, loss = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if world_size > 1 and args.parallelism == "manual-ddp":
                _average_gradients(train_params, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params, args.grad_clip
            ).item()
            opt.step()
            torch.cuda.synchronize()

            loss_avg = _avg_scalar(float(loss.detach().item()), device, world_size)
            grad_avg = _avg_scalar(float(grad_norm), device, world_size)
            losses.append(loss_avg)
            rec = {
                "step": step,
                "local_step": local_step,
                "target_step": target_step,
                "loss": loss_avg,
                "grad_norm": grad_avg,
                "lr": lr_now,
                "data_ms": round((t_data - t_step) * 1000, 1),
                "step_ms": round((time.time() - t_step) * 1000, 1),
                "gpu_reserved_gb": round(
                    torch.cuda.max_memory_reserved() / (1024**3), 2
                ),
                "inject": last_inj,
            }
            torch.cuda.reset_peak_memory_stats()

            if (not args.disable_z_eval) and (
                local_step == 1
                or (args.eval_every > 0 and step % args.eval_every == 0)
                or local_step == args.steps
            ):
                try:
                    eval_batch = _move_batch(next(iter(loader)), device)
                    z_eval = _run_z_eval(
                        model,
                        eval_batch,
                        bridge,
                        src_slice,
                        latent_slice,
                        z_holder,
                        args.seed + 900000 + step,
                        world_size,
                        device,
                    )
                    z_holder["current"] = inj["z"]
                    rec["z_eval"] = z_eval
                    eval_history.append({"step": step, "z_eval": z_eval})
                except Exception as ee:
                    rec["z_eval_error"] = f"{type(ee).__name__}: {ee}"

            if is_rank0:
                if log_f is not None:
                    log_f.write(json.dumps(rec, default=_json_default) + "\n")
                    log_f.flush()
                if (
                    local_step <= 3
                    or step % args.log_every == 0
                    or local_step == args.steps
                ):
                    msg = (
                        f"[gbridge_pt] step {step:5d}/{target_step} loss={loss_avg:.6e} "
                        f"gn={grad_avg:.3f} z_mean={last_inj.get('z_abs_mean', 0):.4f} "
                        f"data={rec['data_ms']:.0f}ms step={rec['step_ms']:.0f}ms "
                        f"mem={rec['gpu_reserved_gb']:.1f}GB"
                    )
                    print(msg, flush=True)
                    if "z_eval" in rec:
                        print(
                            f"[gbridge_pt] z_eval@{step}: {json.dumps(rec['z_eval'])}",
                            flush=True,
                        )

            if (args.save_every > 0 and step % args.save_every == 0) or (
                step in extra_saves
            ):
                _save_trainable_ckpt(model, out_dir, step, args, is_rank0)

        if log_f is not None:
            log_f.close()
        if is_rank0:
            summary = {
                "kind": "gbridge_z_posttrain",
                "dataset_path": args.dataset_path,
                "embodiment": args.embodiment,
                "src_slice": list(src_slice),
                "latent_slice": list(latent_slice),
                "base_ckpt": args.base_ckpt,
                "experiment": args.experiment,
                "init_trainable_from": args.init_trainable_from,
                "overlay": overlay_summary,
                "gr_bridge_ckpt": args.gr_bridge_ckpt,
                "bridge_meta": bridge.meta,
                "scope": args.scope,
                "world_size": world_size,
                "batch_size_per_rank": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "steps": args.steps,
                "target_step": target_step,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "adam_betas": [args.adam_beta1, args.adam_beta2],
                "warmup_steps": args.warmup_steps,
                "trainable_params": int(n_train_after),
                "init_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None,
                "min_loss": min(losses) if losses else None,
                "eval_history": eval_history,
                "last_inject": last_inj,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, default=_json_default)
            )
            print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    except Exception as exc:
        if log_f is not None:
            log_f.close()
        if is_rank0:
            (out_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "init_trainable_from": args.init_trainable_from,
                        "gr_bridge_ckpt": args.gr_bridge_ckpt,
                        "last_inject": last_inj,
                    },
                    indent=2,
                    default=_json_default,
                )
            )
            print(f"[gbridge_pt] FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
