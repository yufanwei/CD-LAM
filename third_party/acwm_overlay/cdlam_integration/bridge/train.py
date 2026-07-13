"""CD-LAM runtime component."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[2])))
H, W = 240, 320
ACTION_DIM = 22
LATENT_DIM = 32


# ============================================================================


# ============================================================================
def mlp(di, do):
    return nn.Sequential(
        nn.Linear(di, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, do),
    )


# ============================================================================
# metrics
# ============================================================================
def r2(p, g, dims=None):
    """CD-LAM runtime component."""
    if dims is not None:
        p, g = p[:, dims], g[:, dims]
    sst = ((g - g.mean(0)) ** 2).sum(0)
    alive = sst > 1e-6
    if alive.sum() == 0:
        return float("nan")
    ssr = ((g - p) ** 2).sum(0)
    return float((1 - ssr[alive] / sst[alive]).mean())


def mmd2_rbf_np(X, Y, gamma=None, max_n=2000, seed=0):
    """CD-LAM runtime component."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    Z = np.concatenate([X, Y], 0)
    if gamma is None:
        idx = rng.choice(len(Z), min(len(Z), 1000), replace=False)
        Zs = Z[idx]
        d2 = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (med + 1e-12)

    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    nx, ny = len(X), len(Y)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1))
    sxy = Kxy.mean()
    return float(sxx + syy - 2 * sxy), float(gamma)


def mmd2_rbf_torch(X, Y):
    """CD-LAM runtime component."""
    with torch.no_grad():
        d2y = torch.cdist(Y, Y) ** 2
        med = d2y[d2y > 0].median().clamp_min(1e-12)
    base = 1.0 / med
    gammas = [base * c for c in (0.25, 0.5, 1.0, 2.0, 4.0)]

    def kxy(A, B):
        d2 = torch.cdist(A, B) ** 2
        out = 0.0
        for g in gammas:
            out = out + torch.exp(-g * d2)
        return out

    Kxx, Kyy, Kxy = kxy(X, X), kxy(Y, Y), kxy(X, Y)
    n, m = X.shape[0], Y.shape[0]
    sxx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1))
    syy = (Kyy.sum() - torch.diagonal(Kyy).sum()) / (m * (m - 1))
    sxy = Kxy.mean()
    return sxx + syy - 2 * sxy


# ============================================================================


# ============================================================================
def train_gh(
    An,
    Zn,
    live,
    tr,
    te,
    dev,
    epochs,
    lr=1e-3,
    wd=1e-5,
    batch=2048,
    l_enc=1.0,
    l_cyc=1.0,
    l_cycz=0.0,
    l_dec=0.3,
    l_mmd=0.0,
    mmd_batch=512,
    log_every=50,
    verbose=True,
):
    """CD-LAM runtime component."""
    nlive = len(live)
    at = torch.as_tensor(An, dtype=torch.float32, device=dev)
    zt = torch.as_tensor(Zn, dtype=torch.float32, device=dev)
    al = at[:, torch.as_tensor(live, device=dev)]
    tri = torch.as_tensor(tr, device=dev)
    tei = torch.as_tensor(te, device=dev)

    g = mlp(An.shape[1], Zn.shape[1]).to(dev)
    h = mlp(Zn.shape[1], nlive).to(dev)
    opt = torch.optim.Adam(
        list(g.parameters()) + list(h.parameters()), lr, weight_decay=wd
    )

    def h_pad(out_live):
        """CD-LAM runtime component."""
        full = torch.zeros(out_live.shape[0], An.shape[1], device=out_live.device)
        full[:, torch.as_tensor(live, device=out_live.device)] = out_live
        return full

    trend = []
    for ep in range(epochs):
        g.train()
        h.train()
        perm = tri[torch.randperm(len(tri), device=dev)]
        for s in range(0, len(perm), batch):
            i = perm[s : s + batch]
            a_i, z_i, al_i = at[i], zt[i], al[i]
            z_hat = g(a_i)  # a->z
            L_enc = F.mse_loss(z_hat, z_i)
            L_cyc = F.mse_loss(h(z_hat), al_i)  # a->z->a
            L_dec = F.mse_loss(h(z_i), al_i)
            L = l_enc * L_enc + l_cyc * L_cyc + l_dec * L_dec
            if l_cycz > 0:
                L = L + l_cycz * F.mse_loss(g(h_pad(h(z_i))), z_i)  # z->a->z
            if l_mmd > 0:
                j = i[:mmd_batch]
                L = L + l_mmd * mmd2_rbf_torch(g(at[j]), zt[j])
            opt.zero_grad()
            L.backward()
            opt.step()
        if (ep + 1) % log_every == 0 or ep == 0:
            g.eval()
            h.eval()
            with torch.no_grad():
                zhat_te = g(at[tei])
                cyc_te = h(zhat_te).cpu().numpy()  # a->z->a
                dec_te = h(zt[tei]).cpu().numpy()  # z->a
                zhat_te = zhat_te.cpu().numpy()
            al_te = al[tei].cpu().numpy()
            row = {
                "ep": ep + 1,
                "a2z_R2": round(r2(zhat_te, Zn[te]), 4),  # a->z
                "cyc_R2": round(r2(cyc_te, al_te), 4),
                "z2a_R2": round(r2(dec_te, al_te), 4),
            }
            trend.append(row)
            if verbose:
                print(
                    f"[a22z] ep{ep + 1:4d}  a->z R2={row['a2z_R2']:.3f}  "
                    f"cycle(a->z->a) R2={row['cyc_R2']:.3f}  "
                    f"z->a R2={row['z2a_R2']:.3f}",
                    flush=True,
                )
    g.eval()
    h.eval()
    return g, h, trend


# ============================================================================


# ============================================================================
def decode_with_z(decode_full, lam, patches, z, latent_dim):
    B = z.shape[0]
    z_rep = z.reshape(B, 1, 1, latent_dim)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rec = decode_full(lam, patches, z_rep, H, W, use_ckpt=False)
    return rec[:, 0].float().clamp(0, 1).cpu().numpy()


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return mse, float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lam-ckpt",
        default=str(REPO / "outputs/stage1/checkpoints/latest.pt"),
    )
    ap.add_argument(
        "--parquet",
        default=str(REPO / "data/prepared/stage1/train.parquet"),
    )
    ap.add_argument(
        "--n-pairs",
        type=int,
        default=12000,
        help="Number of real pairs to sample while preserving episode-disjoint splits; 0 uses all pairs.",
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument(
        "--enc-bs",
        type=int,
        default=64,
        help="Batch size for LAM latent encoding.",
    )
    ap.add_argument("--decode-workers", type=int, default=24)
    ap.add_argument("--l-enc", type=float, default=1.0)
    ap.add_argument("--l-cyc", type=float, default=1.0)
    ap.add_argument("--l-cycz", type=float, default=0.0)
    ap.add_argument("--l-dec", type=float, default=0.3)
    ap.add_argument("--l-mmd", type=float, default=0.0)
    ap.add_argument(
        "--n-gen",
        type=int,
        default=24,
        help="Number of high-motion evaluation pairs used for generation PSNR.",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Output filename suffix, for example _encfree.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Run the A, B, and C objective configurations in one process.",
    )
    ap.add_argument(
        "--frontier",
        action="store_true",
        help="Run the D, E, and F frontier configurations.",
    )
    ap.add_argument(
        "--z-cache",
        default="",
        help="Optional frozen-z NPZ cache that skips video decoding and latent encoding.",
    )
    ap.add_argument(
        "--cache",
        default="",
        help="Robot NPZ cache containing frames, actions, split, and episode_id.",
    )
    ap.add_argument(
        "--action-key",
        default="ee_delta",
        help="Action-array key in the NPZ cache.",
    )
    ap.add_argument(
        "--robot",
        default="agibot_beta",
        help="Robot name used in output filenames.",
    )
    ap.add_argument(
        "--grip-dims",
        default="",
        help="Comma-separated gripper dimensions; empty selects them automatically.",
    )
    ap.add_argument(
        "--full-action",
        action="store_true",
        help="Decode every action dimension and report per-dimension metrics.",
    )
    ap.add_argument("--out", default=str(REPO / "outputs/bridge"))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small real-data GPU smoke configuration.",
    )
    args = ap.parse_args()

    import pandas as pd  # noqa: E402
    from cdlam_integration.bridge.trunk import encode_trunks  # noqa: E402
    from cdlam_integration.lam.model_loader import build_lam  # noqa: E402
    from cdlam_integration.lam.model_ops import decode_full, encode_full  # noqa: E402
    from cdlam_integration.lam.preprocess import decode_rows_parallel  # noqa: E402
    from cdlam_integration.lam.train import load_init_ckpt  # noqa: E402

    if args.smoke:
        args.n_pairs = 400
        args.epochs = 40
        args.enc_bs = 16
    dev = "cuda"
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    if not args.cache:
        df = pd.read_parquet(args.parquet)
        real = df[df.is_real_pair].reset_index(drop=True)
        if args.n_pairs and args.n_pairs < len(real):
            tr_pool = real[real.split == "train"]
            te_pool = real[real.split == "eval"]
            n_te = max(1, int(round(args.n_pairs * len(te_pool) / len(real))))
            n_tr = args.n_pairs - n_te
            real = pd.concat(
                [
                    tr_pool.sample(n=min(n_tr, len(tr_pool)), random_state=0),
                    te_pool.sample(n=min(n_te, len(te_pool)), random_state=0),
                ]
            ).reset_index(drop=True)
        print(
            f"[a22z] parquet rows={len(df)} real={int(df.is_real_pair.sum())} "
            f"-> use {len(real)} (train={int((real.split == 'train').sum())} "
            f"eval={int((real.split == 'eval').sum())})",
            flush=True,
        )

    lam = build_lam("CD_LAM_BASE", device=dev).lam
    load_init_ckpt(
        lam, Path(args.lam_ckpt), dev, lambda m: print(f"[a22z] {m}", flush=True)
    )
    lam.eval().requires_grad_(False)
    latent_dim = int(lam.latent_dim)
    print(f"[a22z] FROZEN LAM loaded latent_dim={latent_dim}", flush=True)

    cache_path = Path(args.z_cache) if args.z_cache else None
    gen_frames = None
    cached_order = None
    if cache_path and cache_path.exists():
        cd = np.load(cache_path, allow_pickle=True)
        A = cd["A"].astype(np.float32)
        Z = cd["Z"].astype(np.float32)
        ep = cd["ep"].astype(str)
        split = cd["split"].astype(str)
        gen_frames = cd["gen_frames"]
        cached_order = cd["order"]
        print(
            f"[a22z] loaded z-cache {cache_path}: Z{Z.shape} gen_frames{gen_frames.shape} "
            "(video decoding and latent encoding skipped)",
            flush=True,
        )
    else:
        if args.cache:
            cd = np.load(args.cache, allow_pickle=True)
            A = cd[args.action_key].astype(np.float32)
            frames = cd["frames"]
            ep = cd["episode_id"].astype(str)
            split = cd["split"].astype(str)
            if args.n_pairs and args.n_pairs < len(frames):
                rng = np.random.default_rng(0)
                tri = np.where(split == "train")[0]
                tei = np.where(split == "eval")[0]
                n_te = max(1, int(round(args.n_pairs * len(tei) / len(frames))))
                n_tr = args.n_pairs - n_te
                sel = np.sort(
                    np.concatenate(
                        [
                            rng.choice(tri, min(n_tr, len(tri)), replace=False),
                            rng.choice(tei, min(n_te, len(tei)), replace=False),
                        ]
                    )
                )
                frames = frames[sel]
                A = A[sel]
                ep = ep[sel]
                split = split[sel]
            print(
                f"[a22z] npz cache {args.cache} key={args.action_key} dim={A.shape[1]}: use {len(frames)} "
                f"(train={int((split == 'train').sum())} eval={int((split == 'eval').sum())})",
                flush=True,
            )
        else:
            A = np.stack(real["action_22"].values).astype(np.float32)
            ep = real["episode_id"].astype(str).values
            split = real["split"].values
            print(
                f"[a22z] decoding {len(real)} frame pairs (workers={args.decode_workers}) ...",
                flush=True,
            )
            frames, ok = decode_rows_parallel(
                real, target_hw=(H, W), workers=args.decode_workers
            )
            ok_idx = np.where(ok)[0]
            frames = frames[ok_idx]
            A = A[ok_idx]
            ep = ep[ok_idx]
            split = split[ok_idx]
            print(
                f"[a22z] decoded ok={len(ok_idx)}/{len(real)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        Z = encode_trunks(lam, frames, dev, bs=args.enc_bs)[0].astype(
            np.float32
        )  # (N,32) z_mu
        print(
            f"[a22z] encoded frozen z {Z.shape} ({time.time() - t0:.0f}s)", flush=True
        )

    tr = np.where(split == "train")[0]
    te = np.where(split == "eval")[0]
    overlap = len(set(ep[tr]) & set(ep[te]))
    assert overlap == 0, f"split is not episode-disjoint (overlap={overlap})"

    a_std = A[tr].std(0)
    print(
        f"[a22z] {args.robot} action[{args.action_key if args.cache else 'action_22'}] dim={A.shape[1]} "
        f"per-dim std: {np.round(a_std, 4).tolist()}",
        flush=True,
    )
    live = np.where(a_std > 1e-2)[0]
    gen_dims = live.copy()
    if args.full_action:
        live = np.arange(A.shape[1])
        print(
            f"[a22z] --full-action: decoding all {A.shape[1]} dimensions",
            flush=True,
        )
    if args.grip_dims:
        grip_set = set(int(x) for x in args.grip_dims.split(",") if x.strip() != "")
    else:
        grip_set = (
            {14, 15} if A.shape[1] == 22 else ({9, 19} if A.shape[1] == 20 else set())
        )
    grip_pos = [i for i, d in enumerate(live) if d in grip_set]
    arm_pos = [i for i, d in enumerate(live) if d not in grip_set]
    print(
        f"[a22z] {args.robot} train={len(tr)} eval={len(te)} live={live.tolist()} "
        f"grip={sorted(grip_set)} (arm_pos={len(arm_pos)} grip_pos={len(grip_pos)})",
        flush=True,
    )

    am = A[tr].mean(0)
    asd = A[tr].std(0)
    asd[asd < 1e-6] = 1.0
    An = ((A - am) / asd).astype(np.float32)
    zm = Z[tr].mean(0)
    zsd = Z[tr].std(0)
    zsd[zsd < 1e-6] = 1.0
    Zn = ((Z - zm) / zsd).astype(np.float32)

    al = An[:, live]
    zm_t = torch.as_tensor(zm, device=dev)
    zsd_t = torch.as_tensor(zsd, device=dev)

    amag = np.linalg.norm(An[te][:, gen_dims], axis=1)
    order = te[np.argsort(-amag)][: args.n_gen]
    if gen_frames is None:
        gen_frames = frames[order]
        if cache_path:
            np.savez(
                cache_path,
                A=A,
                Z=Z,
                ep=ep,
                split=split,
                gen_frames=gen_frames,
                order=order,
            )
            print(f"[a22z] saved z-cache -> {cache_path}", flush=True)
    elif cached_order is not None:
        assert np.array_equal(order, cached_order), (
            "sample order is not reproducible; z-cache mismatch"
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e_gen = encode_full(
            lam,
            torch.from_numpy(gen_frames).to(dev).float() / 255.0,
            sample=False,
            use_ckpt=False,
        )
    patches_gen = e_gen["patches"]
    z_real_gen = e_gen["z_mu"].float()
    o_t1_true = gen_frames[:, 1].astype(np.float32) / 255.0
    rec_real = decode_with_z(decode_full, lam, patches_gen, z_real_gen, latent_dim)
    rec_zero = decode_with_z(
        decode_full, lam, patches_gen, torch.zeros_like(z_real_gen), latent_dim
    )
    psnr_real = round(psnr(rec_real, o_t1_true)[1], 3)
    psnr_zero = round(psnr(rec_zero, o_t1_true)[1], 3)

    def run_one(tag, l_enc, l_cyc, l_cycz, l_dec, l_mmd):
        """CD-LAM runtime component."""
        lam_str = f"enc{l_enc}/cyc{l_cyc}/cycz{l_cycz}/dec{l_dec}/mmd{l_mmd}"
        print(f"\n[a22z] ===== CONFIG {tag}  λ={lam_str} =====", flush=True)
        g, h, trend = train_gh(
            An,
            Zn,
            live,
            tr,
            te,
            dev,
            args.epochs,
            lr=args.lr,
            batch=args.batch,
            l_enc=l_enc,
            l_cyc=l_cyc,
            l_cycz=l_cycz,
            l_dec=l_dec,
            l_mmd=l_mmd,
            log_every=10 if args.smoke else 50,
        )
        with torch.no_grad():
            at = torch.as_tensor(An, device=dev)
            zt = torch.as_tensor(Zn, device=dev)
            zhat = g(at).cpu().numpy()  # a->z
            cyc = h(g(at)).cpu().numpy()  # a->z->a
            dec = h(zt).cpu().numpy()  # z->a
        eval_res = {
            "cycle_a2z2a_R2": round(r2(cyc[te], al[te]), 4),
            "cycle_arm_R2": round(r2(cyc[te], al[te], arm_pos), 4) if arm_pos else None,
            "cycle_grip_R2": round(r2(cyc[te], al[te], grip_pos), 4)
            if grip_pos
            else None,
            "z2a_persample_R2": round(r2(dec[te], al[te]), 4),
            "z2a_dist_MMD2": round(mmd2_rbf_np(dec[te], al[te])[0], 5),
            "z2a_std_ratio_mean": round(
                float((dec[te].std(0) / (al[te].std(0) + 1e-9)).mean()), 3
            ),
            "a2z_R2": round(r2(zhat[te], Zn[te]), 4),
            "a2z_MMD2": round(mmd2_rbf_np(zhat[te], Zn[te])[0], 5),
        }

        NM22 = {
            **{i: f"joint{i}" for i in range(14)},
            14: "gripper_left",
            15: "gripper_right",
            16: "head0",
            17: "head1",
            18: "waist0",
            19: "waist_lift",
            20: "velocity0",
            21: "velocity1",
        }
        live_arr = np.array(live)
        am_l = am[live_arr]
        asd_l = asd[live_arr]
        cyc_raw = cyc * asd_l + am_l
        dec_raw = dec * asd_l + am_l
        per_dim = []
        for j, d in enumerate(live):
            t = A[te, d]
            tabs = float(np.mean(np.abs(t)))
            tstd = float(t.std())
            cmae = float(np.mean(np.abs(cyc_raw[te, j] - t)))
            crmse = float(np.sqrt(np.mean((cyc_raw[te, j] - t) ** 2)))
            dmae = float(np.mean(np.abs(dec_raw[te, j] - t)))
            has_info = tstd > 1e-4
            gn = al[te, j]
            sst = float(((gn - gn.mean()) ** 2).sum())
            cr2 = (
                round(1 - float(((gn - cyc[te, j]) ** 2).sum()) / sst, 3)
                if (has_info and sst > 1e-9)
                else None
            )
            per_dim.append(
                {
                    "dim": int(d),
                    "name": NM22.get(int(d), f"d{d}"),
                    "tgt_absmean": round(tabs, 6),
                    "tgt_std": round(tstd, 6),
                    "cyc_MAE": round(cmae, 6),
                    "cyc_RMSE": round(crmse, 6),
                    "cyc_R2": cr2,
                    "dec_MAE": round(dmae, 6),
                    "rel_err_absmean": (round(cmae / tabs, 3) if tabs > 1e-6 else None),
                    "rel_err_std": (round(crmse / tstd, 3) if has_info else None),
                    "info": "informative" if has_info else "constant_or_uninformative",
                }
            )
        eval_res["per_dim"] = per_dim
        print(
            f"[a22z] {tag} EVAL {json.dumps({k: v for k, v in eval_res.items() if k != 'per_dim'}, ensure_ascii=False)}",
            flush=True,
        )
        print(
            f"[a22z] {tag} per-dimension errors in raw-delta units:",
            flush=True,
        )
        print(
            "  dim name      | target_std | cyc_MAE | dec_MAE | error/std | cyc_R2 | status",
            flush=True,
        )
        for pd_ in per_dim:
            print(
                f"  {pd_['dim']:2d} {pd_['name']:5s}| {pd_['tgt_std']:.5f} | {pd_['cyc_MAE']:.5f} | "
                f"{pd_['dec_MAE']:.5f} | {str(pd_['rel_err_std']):>6} | {str(pd_['cyc_R2']):>6} | {pd_['info']}",
                flush=True,
            )

        with torch.no_grad():
            zn_g = g(torch.as_tensor(An[order], device=dev))
        rec_ga = decode_with_z(
            decode_full, lam, patches_gen, (zn_g * zsd_t + zm_t).float(), latent_dim
        )
        gen = {
            "real_z": psnr_real,
            "g_a": round(psnr(rec_ga, o_t1_true)[1], 3),
            "zero_z": psnr_zero,
            "n": len(order),
        }
        print(
            f"[a22z] {tag} GEN PSNR real={gen['real_z']} g(a)={gen['g_a']} zero={gen['zero_z']}",
            flush=True,
        )
        lambdas = {
            "enc": l_enc,
            "cyc": l_cyc,
            "cycz": l_cycz,
            "dec": l_dec,
            "mmd": l_mmd,
        }
        stem = f"a22z_{args.robot}{tag}"
        save_path = out / f"{stem}.pt"
        torch.save(
            {
                "g_state": g.state_dict(),
                "h_state": h.state_dict(),
                "arch": {
                    "g": f"mlp(22,{latent_dim})",
                    "h": f"mlp({latent_dim},{len(live)})",
                },
                "live_dims": live.tolist(),
                "latent_dim": latent_dim,
                "action_mean": am,
                "action_std": asd,
                "zm": zm,
                "zsd": zsd,
                "lam_ckpt": args.lam_ckpt,
                "lambdas": lambdas,
                "eval": eval_res,
                "gen_psnr": gen,
            },
            save_path,
        )
        report = {
            "tag": tag,
            "n_train": int(len(tr)),
            "n_eval": int(len(te)),
            "live_dims": live.tolist(),
            "lambdas": lambdas,
            "eval": eval_res,
            "gen_psnr": gen,
            "trend": trend,
            "robot": args.robot,
            "action_key": (args.action_key if args.cache else "action_22"),
            "action_dim": int(An.shape[1]),
            "save_path": str(save_path),
        }
        json.dump(
            report,
            open(out / f"{stem}_report.json", "w"),
            indent=2,
            ensure_ascii=False,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
        print(
            f"[a22z] {tag} saved -> {save_path}  ({time.time() - t0:.0f}s)", flush=True
        )
        return report

    if args.sweep or args.frontier:
        if args.frontier:
            configs = [
                ("_D_cyc3", 1.0, 3.0, 0.0, 0.3, 0.0),
                ("_E_anchcyc", 1.0, 2.0, 0.0, 0.2, 0.3),
                ("_F_lean", 0.5, 2.0, 0.0, 0.1, 0.5),
            ]
        else:
            configs = [
                ("_A_anchor", 1.0, 1.0, 0.0, 0.3, 0.0),
                ("_B_transport", 0.1, 1.0, 0.0, 0.0, 1.0),
                ("_C_mid", 0.3, 1.0, 0.0, 0.1, 0.5),
            ]
        reports = [run_one(*c) for c in configs]
        combo = {
            r["tag"]: {
                "eval": r["eval"],
                "gen_psnr": r["gen_psnr"],
                "lambdas": r["lambdas"],
            }
            for r in reports
        }
        combo_name = (
            f"a22z_{args.robot}_{'FRONTIER' if args.frontier else 'SWEEP'}.json"
        )
        json.dump(
            combo,
            open(out / combo_name, "w"),
            indent=2,
            ensure_ascii=False,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
        print("\n[a22z] ===== SWEEP SUMMARY =====", flush=True)
        for r in reports:
            ev = r["eval"]
            print(
                f"  {r['tag']:14s} cycle={ev['cycle_a2z2a_R2']:.3f} "
                f"(arm={ev['cycle_arm_R2']} grip={ev['cycle_grip_R2']}) | "
                f"z->a MMD={ev['z2a_dist_MMD2']:.4f} std={ev['z2a_std_ratio_mean']} "
                f"persamp_R2={ev['z2a_persample_R2']:.3f} | "
                f"a->z R2={ev['a2z_R2']:.3f} MMD={ev['a2z_MMD2']:.4f} | "
                f"gen g(a)={r['gen_psnr']['g_a']} (real={r['gen_psnr']['real_z']} "
                f"zero={r['gen_psnr']['zero_z']})",
                flush=True,
            )
        print(
            f"[a22z] SWEEP done ({time.time() - t0:.0f}s) -> {out}/{combo_name}",
            flush=True,
        )
    else:
        run_one(args.tag, args.l_enc, args.l_cyc, args.l_cycz, args.l_dec, args.l_mmd)


if __name__ == "__main__":
    main()
