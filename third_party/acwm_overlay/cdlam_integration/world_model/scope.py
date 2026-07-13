"""CD-LAM runtime component."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from cdlam_integration.world_model.preprocess import (  # noqa: E402
    decode_video_official,
    official_lam_video_from_wm,
)


def _setup():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("LOCAL_RANK", "0")
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=0, world_size=1
        )
    from cosmos_predict2._src.imaginaire.utils import distributed as _dist

    orig = _dist.sync_model_states

    def patched(model, src=0, **kwargs):
        import torch.distributed as td

        if (td.get_world_size() if td.is_initialized() else 1) <= 1:
            return
        return orig(model, src=src, **kwargs)

    _dist.sync_model_states = patched


def _decode_video(video: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    return decode_video_official(video, n_frames, wm_hw=(h, w))


def _build_z(lam_id: str, video_np: np.ndarray, T: int) -> torch.Tensor:
    from cdlam_integration.lam.encoder import pair_uint8_to_float
    from cdlam_integration.world_model.model_loader import build_encoder

    enc, _ = build_encoder(lam_id, device="cuda")
    zs = []
    with torch.no_grad():
        for t in range(T):
            pair = video_np[t : t + 2]
            pair_t = torch.from_numpy(pair[None]).cuda()
            pf = pair_uint8_to_float(pair_t)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = enc(pf).float()
            zs.append(z)
    return torch.stack(zs, dim=1)


def patch_forward(model, z_cached):
    import types

    fixed = {"epsilon": None, "timesteps": None}

    def patched_forward(self, data_batch):
        z_local = z_cached.to(
            device=data_batch["action"].device, dtype=data_batch["action"].dtype
        )
        if z_local.shape[1] != data_batch["action"].shape[1]:
            z_local = z_local.expand(-1, data_batch["action"].shape[1], -1)
        data_batch["action"][:, :, -32:] = data_batch["action"][:, :, -32:] * z_local

        if (
            self.config.text_encoder_config is not None
            and self.config.text_encoder_config.compute_online
            and "t5_text_embeddings" not in data_batch
        ):
            txt = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )
            data_batch["t5_text_embeddings"] = txt

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        if fixed["epsilon"] is not None:
            epsilon = fixed["epsilon"]
            timesteps = fixed["timesteps"]
        else:
            epsilon = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
            t_B = self.rectified_flow.sample_train_time(x0_B_C_T_H_W.size(0)).to(
                **self.tensor_kwargs_fp32
            )
            t_B = rearrange(t_B, "b -> b 1")
            x0_B_C_T_H_W, condition, epsilon, t_B = (
                self.broadcast_split_for_model_parallelsim(
                    x0_B_C_T_H_W, condition, epsilon, t_B
                )
            )
            timesteps = self.rectified_flow.get_discrete_timestamp(
                t_B, self.tensor_kwargs_fp32
            )
            timesteps = rearrange(timesteps, "b -> b 1")
            fixed["epsilon"] = epsilon.detach()
            fixed["timesteps"] = timesteps.detach()
        sigmas = self.rectified_flow.get_sigmas(
            timesteps.squeeze(-1), self.tensor_kwargs_fp32
        )
        sigmas = rearrange(sigmas, "b -> b 1")
        xt, vt = self.rectified_flow.get_interpolation(epsilon, x0_B_C_T_H_W, sigmas)
        vt_pred = self.denoise(
            noise=epsilon,
            xt_B_C_T_H_W=xt.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )
        time_w = self.rectified_flow.train_time_weight(
            timesteps, self.tensor_kwargs_fp32
        )
        per_inst = ((vt_pred - vt) ** 2).mean(dim=list(range(1, vt_pred.dim())))
        per_inst_mc = (
            ((vt_pred[:, 1:] - vt_pred[:, :-1]) - (vt[:, 1:] - vt[:, :-1])) ** 2
        ).mean(dim=list(range(1, vt_pred.dim())))
        per_inst = per_inst + per_inst_mc * 0.1
        loss = (time_w * per_inst).mean()
        return {"edm_loss": loss}, loss

    model.forward = types.MethodType(patched_forward, model)


def configure_scope(model, scope: str, lora_rank: int = 16):
    """Returns trainable params list. Side-effect: sets requires_grad on params.

    PR-6.5 fix: net.blocks has 28 blocks (0..27), not 24. Old `B = blocks 20-23`
    was NOT actually the last 4. We now compute indices from len(model.net.blocks).
    Aliases:
        B           = embedder + true last 4 blocks (24..27 for 28-block ckpt)
        B_old       = embedder + blocks 20..23 (PR-6 erroneous "last 4"; kept for
                       retroactive comparison)
        B2          = embedder + true last 8 blocks (20..27 for 28-block ckpt)
    """
    for p in model.parameters():
        p.requires_grad = False
    train: list = []
    N = len(model.net.blocks)

    def add_module(mod):
        for p in mod.parameters():
            p.requires_grad = True
            train.append(p)

    def add_blocks(idxs):
        for bi in idxs:
            add_module(model.net.blocks[bi])

    if scope == "A":
        add_module(model.net.action_embedder_B_D)
        add_module(model.net.action_embedder_B_3D)
    elif scope == "A2":
        add_module(model.net.action_embedder_B_D)
        add_module(model.net.action_embedder_B_3D)
        add_module(model.net.t_embedder)
    elif scope == "B":
        # PR-6.5: TRUE last 4
        add_module(model.net.action_embedder_B_D)
        add_module(model.net.action_embedder_B_3D)
        add_blocks(list(range(N - 4, N)))
    elif scope == "B_old":
        # PR-6 erroneous "last 4" = blocks 20..23 (indices, not the true tail).
        add_module(model.net.action_embedder_B_D)
        add_module(model.net.action_embedder_B_3D)
        add_blocks([20, 21, 22, 23])
    elif scope == "B2":
        # PR-6.5: TRUE last 8
        add_module(model.net.action_embedder_B_D)
        add_module(model.net.action_embedder_B_3D)
        add_blocks(list(range(N - 8, N)))
    elif scope == "E":
        # first 4 blocks (vs B's last 4) — does early matter more?
        add_module(model.net.action_embedder_B_D)
        add_module(model.net.action_embedder_B_3D)
        add_blocks([0, 1, 2, 3])
    elif scope == "C":
        # LoRA on q/k/v + output_proj across all 24 blocks
        from peft import LoraConfig, get_peft_model

        # Cosmos2 attn modules name path looks like net.blocks.X.self_attn.{q_proj,k_proj,v_proj,output_proj}
        target = []
        for n, m in model.net.named_modules():
            if any(
                s in n
                for s in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.output_proj",
                    "cross_attn.q_proj",
                    "cross_attn.k_proj",
                    "cross_attn.v_proj",
                    "cross_attn.output_proj",
                ]
            ):
                target.append(n)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=target,
            init_lora_weights=True,
        )
        # Wrap net into peft model
        model.net = get_peft_model(model.net, lora_cfg)
        for p in model.net.parameters():
            if p.requires_grad:
                train.append(p)
        # Also unfreeze action_embedder
        seen = {id(p) for p in train}
        for n, m in model.net.named_modules():
            if "action_embedder" in n:
                for p in m.parameters():
                    p.requires_grad = True
                    if id(p) not in seen:
                        train.append(p)
                        seen.add(id(p))
    elif scope == "D":
        for p in model.net.parameters():
            p.requires_grad = True
            train.append(p)
    else:
        raise ValueError(f"unknown scope: {scope}")
    n_train = sum(p.numel() for p in train if p.requires_grad)
    return train, n_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scope", required=True, choices=["A", "A2", "B", "B_old", "B2", "E", "C", "D"]
    )
    ap.add_argument("--lam-id", required=True)
    ap.add_argument(
        "--ckpt",
        default=os.environ.get(
            "CDLAM_BASE_2B_CHECKPOINT",
            str(
                REPO
                / "lammodel/checkpoints/CD-LAM/2B_pretrain/iter_000140000/model_ema_bf16.pt"
            ),
        ),
    )
    ap.add_argument("--experiment", default="dreamdojo_2b_480_640_pretrain")
    ap.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    ap.add_argument("--input-video", required=True)
    ap.add_argument("--num-action-per-chunk", type=int, default=12)
    ap.add_argument("--action-dim", type=int, default=384)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", default="480,640")
    ap.add_argument("--lam-resolution", default="240,320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = (int(x) for x in args.resolution.split(","))
    LAM_H, LAM_W = (int(x) for x in args.lam_resolution.split(","))

    _setup()
    video_np = _decode_video(
        Path(args.input_video), args.num_action_per_chunk + 1, H, W
    )
    lam_video_np = official_lam_video_from_wm(video_np, lam_hw=(LAM_H, LAM_W))
    z_cached = _build_z(args.lam_id, lam_video_np, args.num_action_per_chunk)

    print(f"[scope={args.scope}] loading WM...", flush=True)
    t0 = time.time()
    from cosmos_predict2._src.predict2.utils.model_loader import (
        load_model_from_checkpoint,
    )

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.ckpt,
        config_file=args.config_file,
        load_ema_to_reg=True,
        skip_load_model=False,
    )
    print(f"[scope={args.scope}] loaded in {time.time() - t0:.1f}s", flush=True)

    patch_forward(model, z_cached)
    train_params, n_train = configure_scope(model, args.scope)
    print(f"[scope={args.scope}] trainable params: {n_train / 1e6:.2f}M", flush=True)

    img_t = torch.from_numpy(video_np).float() / 255.0
    img_t = img_t.permute(3, 0, 1, 2).unsqueeze(0)
    vid_uint8 = (img_t * 255.0).to(torch.uint8)
    action = torch.zeros(
        1,
        args.num_action_per_chunk,
        args.action_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    action[:, :, -32:] = 1.0
    base_batch = {
        "dataset_name": "video_data",
        "video": vid_uint8.cuda(),
        "fps": torch.tensor([10], dtype=torch.float).cuda(),
        "padding_mask": torch.zeros(1, 1, H, W).cuda(),
        "num_conditional_frames": 1,
        "action": action,
    }
    if model.text_encoder is not None:
        base_batch["t5_text_embeddings"] = (
            model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [args.prompt], "images": None},
                input_caption_key="ai_caption",
            )
        )
    for k, v in base_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            base_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)

    # ---- step-0 init eval (no update)
    model.eval()
    with torch.no_grad():
        db = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in base_batch.items()
        }
        torch.manual_seed(args.seed)
        _, init_loss = model(db)
    init_loss = float(init_loss)

    # ---- training loop
    model.train()
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")
    log_f.write(json.dumps({"step": -1, "loss": init_loss}) + "\n")
    losses = [init_loss]
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    t1 = time.time()
    try:
        for step in range(args.steps):
            t_step0 = time.time()
            db = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in base_batch.items()
            }
            torch.manual_seed(args.seed + step)
            _, loss = model(db)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(train_params, max_norm=5.0).item()
            opt.step()
            torch.cuda.synchronize()
            step_t = time.time() - t_step0
            step_times.append(step_t)
            losses.append(float(loss))
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss),
                        "grad_norm": float(gn),
                        "step_sec": step_t,
                    }
                )
                + "\n"
            )
            if (
                step < 5
                or step % max(1, args.steps // 10) == 0
                or step == args.steps - 1
            ):
                print(
                    f"[scope={args.scope}] step {step:4d} loss={float(loss):.4e} step_sec={step_t:.2f} gn={gn:.3f}",
                    flush=True,
                )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[scope={args.scope}] OOM at step {step}: {e}", flush=True)
        log_f.close()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "scope": args.scope,
                    "lam_id": args.lam_id,
                    "trainable_params_M": n_train / 1e6,
                    "n_steps_completed": step,
                    "init_loss": init_loss,
                    "final_loss": losses[-1] if losses else None,
                    "min_loss": min(losses) if losses else None,
                    "drop_factor": (init_loss / losses[-1])
                    if losses and losses[-1] > 0
                    else None,
                    "step_sec_mean": float(np.mean(step_times)) if step_times else None,
                    "peak_gpu_gb": peak_mem,
                    "verdict": "OOM",
                },
                indent=2,
            )
        )
        sys.exit(2)
    log_f.close()
    elapsed = time.time() - t1
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    final_loss = losses[-1]
    drop = init_loss / max(final_loss, 1e-12)
    summary = {
        "scope": args.scope,
        "lam_id": args.lam_id,
        "trainable_params_M": n_train / 1e6,
        "n_steps": args.steps,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "min_loss": min(losses),
        "drop_factor": drop,
        "step_sec_mean": float(np.mean(step_times)),
        "step_sec_p90": float(np.percentile(step_times, 90)),
        "elapsed_sec": round(elapsed, 2),
        "peak_gpu_gb": peak_mem,
        "verdict": "PASS" if drop >= 2.0 else "WEAK_DROP" if drop >= 1.1 else "FAIL",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
