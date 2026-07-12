#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


SLICE_BY_EMBODIMENT = {
    "gr1": (0, 29),
    "g1": (58, 101),
    "yam": (101, 147),
    "agibot": (147, 169),
    "droid": (169, 197),
}


def _setup_distributed() -> tuple[int, int, int]:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29541")

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return rank, world_size, local_rank


def _patch_sync_model_states_for_single_rank() -> None:
    from cosmos_predict2._src.imaginaire.utils import distributed as _dist

    orig = _dist.sync_model_states

    def patched(model, src=0, **kwargs):
        import torch.distributed as td

        ws = td.get_world_size() if td.is_initialized() else 1
        if ws <= 1:
            return
        return orig(model, src=src, **kwargs)

    _dist.sync_model_states = patched


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj).__name__)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key == "video":
                out[key] = value.to(device=device, non_blocking=True)
            elif key == "action":
                out[key] = value.to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
            elif key == "lam_video":
                out[key] = value.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            elif torch.is_floating_point(value):
                out[key] = value.to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
            else:
                out[key] = value.to(device=device, non_blocking=True)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    out["num_conditional_frames"] = 1
    return out


def _action_slice_stats(action: torch.Tensor, active_slice: tuple[int, int]) -> dict:
    action = action.detach()
    s0, s1 = active_slice
    pre = action[..., :s0].abs().max().item() if s0 > 0 else 0.0
    active = action[..., s0:s1].abs()
    post = action[..., s1:].abs().max().item() if s1 < action.shape[-1] else 0.0
    latent = action[..., 352:384].abs().max().item()
    return {
        "shape": list(action.shape),
        "active_slice": [s0, s1],
        "pre_max_abs": float(pre),
        "active_sum_abs": float(active.sum().item()),
        "active_max_abs": float(active.max().item()),
        "post_max_abs": float(post),
        "latent_max_abs": float(latent),
        "finite": bool(torch.isfinite(action).all().item()),
    }


def _assert_action_slice(action: torch.Tensor, active_slice: tuple[int, int]) -> None:
    stats = _action_slice_stats(action, active_slice)
    if stats["shape"][-2:] != [12, 384]:
        raise RuntimeError(f"unexpected action shape: {stats['shape']}")
    if stats["pre_max_abs"] != 0.0 or stats["post_max_abs"] != 0.0:
        raise RuntimeError(f"action leaked outside active slice: {stats}")
    if stats["active_sum_abs"] <= 0.0:
        raise RuntimeError(f"empty active action slice: {stats}")
    if not stats["finite"]:
        raise RuntimeError(f"non-finite action tensor: {stats}")


def _make_eval_action(action: torch.Tensor, mode: str) -> torch.Tensor:
    a = action.clone()
    if mode == "own":
        return a
    if mode == "zero":
        return torch.zeros_like(a)
    if mode == "shuffle_time":
        perm = torch.randperm(a.shape[1], device=a.device)
        return a[:, perm, :]
    if mode == "delay_plus1":
        return torch.cat([a[:, :1, :], a[:, :-1, :]], dim=1)
    if mode == "delay_minus1":
        return torch.cat([a[:, 1:, :], a[:, -1:, :]], dim=1)
    if mode == "joint_only":
        b = torch.zeros_like(a)
        b[..., 147:161] = a[..., 147:161]
        return b
    if mode == "nonjoint_only":
        b = torch.zeros_like(a)
        b[..., 161:169] = a[..., 161:169]
        return b
    raise ValueError(mode)


def _clone_batch_with_action(batch: dict, action: torch.Tensor) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    out["action"] = action
    return out


def _avg_scalar(value: float, device: torch.device, world_size: int) -> float:
    if world_size <= 1:
        return float(value)
    t = torch.tensor([value], dtype=torch.float32, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
    return float(t.item())


def _run_eval(
    model,
    batch: dict,
    seed: int,
    modes: list[str],
    world_size: int,
    device: torch.device,
) -> dict:
    result = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        base_action = batch["action"].detach()
        for mode in modes:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            eval_batch = _clone_batch_with_action(
                batch, _make_eval_action(base_action, mode)
            )
            _, loss = model(eval_batch)
            result[mode] = _avg_scalar(float(loss.detach().item()), device, world_size)
    if was_training:
        model.train()
    return result


def _load_trainable_state(model, ckpt_path: Path, is_rank0: bool) -> dict:
    saved = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_state = saved.get("trainable_state", saved)
    cleaned = {
        key.replace("._checkpoint_wrapped_module", "").removeprefix("net."): value
        for key, value in raw_state.items()
    }
    missing, unexpected = model.net.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(unexpected)
    summary = {
        "ckpt": str(ckpt_path),
        "ckpt_step": saved.get("step"),
        "keys_in_ckpt": len(cleaned),
        "loaded_keys": loaded,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "first_missing": list(missing[:8]),
        "first_unexpected": list(unexpected[:8]),
    }
    if is_rank0:
        print(f"[robot_pt] loaded trainable overlay: {json.dumps(summary)}", flush=True)
    del saved, raw_state, cleaned
    gc.collect()
    return summary


def _clean_state_key(key: str) -> str:
    key = key.replace("._checkpoint_wrapped_module", "")
    if key.startswith("module."):
        key = key.removeprefix("module.")
    return key.removeprefix("net.")


def _load_action_embedder_state(
    model,
    ckpt_path: Path,
    mode: str,
    active_slice: tuple[int, int],
    is_rank0: bool,
) -> dict:
    saved = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_state = saved.get("trainable_state", saved)
    cleaned = {_clean_state_key(key): value for key, value in raw_state.items()}

    prefixes = ("action_embedder_B_D.", "action_embedder_B_3D.")
    summary = {
        "ckpt": str(ckpt_path),
        "mode": mode,
        "active_slice": list(active_slice),
        "loaded_keys": [],
        "missing_keys": [],
    }

    if mode == "full":
        subset = {
            key: value for key, value in cleaned.items() if key.startswith(prefixes)
        }
        missing, unexpected = model.net.load_state_dict(subset, strict=False)
        summary.update(
            {
                "keys_in_ckpt": len(subset),
                "loaded_keys": sorted(subset),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "first_missing": list(missing[:8]),
                "first_unexpected": list(unexpected[:8]),
            }
        )
    elif mode == "active-fc1":
        start, end = active_slice
        cols = []
        for chunk in range(4):
            offset = chunk * 384
            cols.extend(range(offset + start, offset + end))
        cols_t = torch.as_tensor(cols, dtype=torch.long)
        loaded = []
        missing = []
        with torch.no_grad():
            for module_name in ("action_embedder_B_D", "action_embedder_B_3D"):
                src_key = f"{module_name}.fc1.weight"
                src = cleaned.get(src_key)
                if src is None:
                    missing.append(src_key)
                    continue
                dst = getattr(model.net, module_name).fc1.weight
                dst[:, cols_t.to(dst.device)].copy_(
                    src[:, cols_t].to(device=dst.device, dtype=dst.dtype)
                )
                loaded.append(src_key)
        summary.update(
            {
                "keys_in_ckpt": len(cleaned),
                "loaded_keys": loaded,
                "missing_keys": missing,
                "copied_columns": len(cols),
            }
        )
    else:
        raise ValueError(f"unknown action embedder init mode: {mode}")

    if is_rank0:
        print(
            f"[robot_pt] loaded action embedder init: {json.dumps(summary)}", flush=True
        )
    del saved, raw_state, cleaned
    gc.collect()
    return summary


def _save_trainable_ckpt(
    model, out_dir: Path, step: int, args: argparse.Namespace, is_rank0: bool
) -> None:
    trainable_state = {}
    for name, param in model.net.named_parameters():
        if not param.requires_grad:
            continue
        full = param.full_tensor() if hasattr(param, "full_tensor") else param
        if is_rank0:
            trainable_state[name] = full.detach().cpu()
    if is_rank0:
        ckpt_path = out_dir / f"ckpt_step{step}.pt"
        torch.save(
            {
                "step": int(step),
                "trainable_state": trainable_state,
                "args": vars(args),
            },
            str(ckpt_path),
        )
        print(f"[robot_pt] saved {ckpt_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument(
        "--embodiment", choices=sorted(SLICE_BY_EMBODIMENT), default="agibot"
    )
    ap.add_argument(
        "--base-ckpt",
        default=os.environ.get(
            "CDLAM_BASE_2B_CKPT",
            str(REPO / "lammodel/checkpoints/CD-LAM/2B_pretrain/iter_000140000/model"),
        ),
    )
    ap.add_argument("--experiment", default="dreamdojo_2b_480_640_agibot_local_full")
    ap.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    ap.add_argument("--init-trainable-from", default="")
    ap.add_argument("--init-action-embedder-from", default="")
    ap.add_argument(
        "--init-action-embedder-mode", choices=["full", "active-fc1"], default="full"
    )
    ap.add_argument(
        "--scope", choices=["A", "A2", "B", "B_old", "B2", "E", "C", "D"], default="D"
    )
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank = _setup_distributed()
    is_rank0 = rank == 0
    device = torch.device("cuda", local_rank)
    _patch_sync_model_states_for_single_rank()

    if is_rank0:
        print(
            f"[robot_pt] rank={rank}/{world_size} local_rank={local_rank} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )

    torch.manual_seed(args.seed + rank * 100003)
    torch.cuda.manual_seed_all(args.seed + rank * 100003)

    from cosmos_predict2._src.predict2.utils.model_loader import (
        load_model_from_checkpoint,
    )
    from groot_dreams.dataloader import MultiVideoActionDataset
    from lamwm_pipline.tools.scope_ablation import configure_scope

    if is_rank0:
        print(
            f"[robot_pt] loading WM base: experiment={args.experiment} base={args.base_ckpt}",
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
            f"[robot_pt] WM loaded in {time.time() - t0:.1f}s; net_params={n_net / 1e9:.3f}B",
            flush=True,
        )
    if getattr(model, "lam", None) is not None:
        model.lam.to(device)
        if is_rank0:
            print(f"[robot_pt] moved inline LAM to {device}", flush=True)

    train_params, n_train = configure_scope(model, args.scope)
    if args.init_trainable_from:
        overlay_summary = _load_trainable_state(
            model, Path(args.init_trainable_from), is_rank0
        )
    else:
        overlay_summary = {
            "skipped": True,
            "reason": "no --init-trainable-from provided; using --base-ckpt weights directly",
        }
        if is_rank0:
            print(
                f"[robot_pt] no trainable overlay provided: {json.dumps(overlay_summary)}",
                flush=True,
            )

    if args.init_action_embedder_from:
        action_embedder_summary = _load_action_embedder_state(
            model,
            Path(args.init_action_embedder_from),
            args.init_action_embedder_mode,
            SLICE_BY_EMBODIMENT[args.embodiment],
            is_rank0,
        )
    else:
        action_embedder_summary = {
            "skipped": True,
            "reason": "no --init-action-embedder-from provided",
        }

    if world_size > 1:
        from cosmos_predict2._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh

        dp_mesh = hsdp_device_mesh(replica_group_size=1, sharding_group_size=world_size)
        model.apply_fsdp(dp_mesh)
        train_params = [p for p in model.net.parameters() if p.requires_grad]
    n_train_after = sum(p.numel() for p in train_params if p.requires_grad)
    if is_rank0:
        print(
            f"[robot_pt] scope={args.scope}; trainable={n_train_after / 1e9:.3f}B "
            f"({n_train_after:,} params); world_size={world_size}",
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
    active_slice = SLICE_BY_EMBODIMENT[args.embodiment]

    opt = torch.optim.AdamW(
        train_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    log_f = open(out_dir / "train_log.jsonl", "w") if is_rank0 else None
    summary_path = out_dir / "summary.json"
    modes = [
        "own",
        "zero",
        "shuffle_time",
        "delay_plus1",
        "delay_minus1",
        "joint_only",
        "nonjoint_only",
    ]
    losses: list[float] = []
    eval_history: list[dict] = []
    last_slice_stats: dict = {}

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
            _assert_action_slice(batch["action"], active_slice)
            last_slice_stats = _action_slice_stats(batch["action"], active_slice)
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
                "gpu_alloc_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
                "gpu_reserved_gb": round(
                    torch.cuda.max_memory_reserved() / (1024**3), 2
                ),
                "action_slice": last_slice_stats,
            }
            torch.cuda.reset_peak_memory_stats()

            if (
                local_step == 1
                or (args.eval_every > 0 and step % args.eval_every == 0)
                or local_step == args.steps
            ):
                eval_batch = _move_batch(next(iter(loader)), device)
                _assert_action_slice(eval_batch["action"], active_slice)
                eval_losses = _run_eval(
                    model,
                    eval_batch,
                    args.seed + 900000 + step,
                    modes,
                    world_size,
                    device,
                )
                rec["eval_losses"] = eval_losses
                eval_history.append({"step": step, "losses": eval_losses})

            if is_rank0:
                if log_f is not None:
                    log_f.write(json.dumps(rec, default=_json_default) + "\n")
                    log_f.flush()
                if (
                    local_step <= 3
                    or step % args.log_every == 0
                    or local_step == args.steps
                ):
                    print(
                        f"[robot_pt] step {step:5d}/{target_step} local={local_step}/{args.steps} loss={loss_avg:.6e} "
                        f"gn={grad_avg:.3f} data={rec['data_ms']:.0f}ms step={rec['step_ms']:.0f}ms "
                        f"mem={rec['gpu_reserved_gb']:.1f}GB",
                        flush=True,
                    )
                    if "eval_losses" in rec:
                        print(
                            f"[robot_pt] eval@{step}: {json.dumps(rec['eval_losses'])}",
                            flush=True,
                        )

            if args.save_every > 0 and step % args.save_every == 0:
                _save_trainable_ckpt(model, out_dir, step, args, is_rank0)

        if log_f is not None:
            log_f.close()
        if is_rank0:
            summary = {
                "dataset_path": args.dataset_path,
                "embodiment": args.embodiment,
                "active_slice": list(active_slice),
                "base_ckpt": args.base_ckpt,
                "experiment": args.experiment,
                "init_trainable_from": args.init_trainable_from,
                "overlay": overlay_summary,
                "init_action_embedder_from": args.init_action_embedder_from,
                "action_embedder_init": action_embedder_summary,
                "scope": args.scope,
                "world_size": world_size,
                "batch_size_per_rank": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "steps": args.steps,
                "step_offset": args.step_offset,
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
                "last_action_slice": last_slice_stats,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, default=_json_default)
            )
            print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    except Exception as exc:
        if log_f is not None:
            log_f.close()
        if is_rank0:
            failure = {
                "error": f"{type(exc).__name__}: {exc}",
                "dataset_path": args.dataset_path,
                "init_trainable_from": args.init_trainable_from,
                "scope": args.scope,
                "world_size": world_size,
                "batch_size_per_rank": args.batch_size,
                "last_action_slice": last_slice_stats,
            }
            (out_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, default=_json_default)
            )
            print(f"[robot_pt] FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
