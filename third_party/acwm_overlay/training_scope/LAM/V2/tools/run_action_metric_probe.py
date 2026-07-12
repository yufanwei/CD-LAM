#!/usr/bin/env python3
"""Phase 5 — pre-training probe.

Runs the encoders specified by --encoders (e.g. CD_LAM_BASE, CD_LAM_V1,
DINOv2_diff, RAFT_pooled) on a balanced sample of pairs from
pair_index_scored.parquet and computes the metrics required by NOTE §8:

  - cross-episode primitive top-1 / top-5 retrieval
  - action-vs-video gap = mean cos(same_prim_diff_ep) - mean cos(diff_prim_same_ep)
  - same-episode kNN share (top-5)
  - similarity histograms by relation class
  - cross-dataset retrieval (query=A, candidate=B for each (A,B))
  - dataset leakage: top-1 nearest-neighbor dataset accuracy

Outputs (under --out):
  per_encoder_metrics.json
  similarity_histograms/<encoder>.png
  cross_dataset_retrieval.json
  layerwise_probe_report.md  (compact text summary)

Decoding uses cv2 sequential mp4 reads with a thread pool.
Encoding uses GPU batches (512 default for LAM/DINO, 64 for RAFT).

Usage:
    python LAM_V2/tools/run_action_metric_probe.py \\
        --pair-index outputs/lam_v2_data/pair_index_scored.parquet \\
        --out outputs/lam_v2_probe \\
        --encoders CD_LAM_BASE CD_LAM_V1 DINOv2_diff RAFT_pooled \\
        --n-per-primitive 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO = Path(os.environ.get("CDLAM_ACWM_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(REPO))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)


def sample_balanced(
    df: pd.DataFrame, n_per_primitive: int, split: str = "train", seed: int = 0
) -> pd.DataFrame:
    """Sample up to N pairs per (dataset, primitive), preferring high sample_weight & high motion."""
    sub = df[df["split"] == split].copy()
    sub = sub[sub["primitive_label"] != ""]
    sub = sub[~sub["motion_score"].isna()]
    sub["score"] = sub["sample_weight"].fillna(0.0) + 0.5 * sub["motion_score"].fillna(
        0.0
    )
    rng = np.random.default_rng(seed)
    chunks = []
    for (ds, prim), g in sub.groupby(["dataset", "primitive_label"]):
        n = min(n_per_primitive, len(g))
        if n == 0:
            continue
        # Top-k weighted by score, then random shuffle within top-2k.
        top = g.nlargest(min(2 * n, len(g)), "score")
        idx = rng.choice(len(top), size=n, replace=False)
        chunks.append(top.iloc[idx])
    return (
        pd.concat(chunks, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def decode_pair(
    video_path: str, fi: int, fj: int, target_hw=(240, 320)
) -> np.ndarray | None:
    """Sequential read up to max(fi,fj). Returns (2, H, W, 3) uint8 RGB or None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    out = [None, None]
    needed = {fi: 0, fj: 1}  # may collide if fi==fj
    if fi == fj:
        needed = {fi: 0}
    max_idx = max(fi, fj)
    i = 0
    try:
        while i <= max_idx:
            ok, fr = cap.read()
            if not ok:
                break
            if i in needed:
                fr = cv2.resize(
                    fr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA
                )
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                slot = needed[i]
                out[slot] = fr
                if fi == fj:
                    out[1] = fr
            i += 1
    finally:
        cap.release()
    if out[0] is None or out[1] is None:
        return None
    return np.stack(out, 0).astype(np.uint8)


def decode_pairs_parallel(
    rows: pd.DataFrame, target_hw=(240, 320), workers: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pairs_uint8 [N,2,H,W,3], valid_mask [N] bool)."""
    out = np.zeros((len(rows), 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
    valid = np.zeros(len(rows), dtype=bool)
    args = [
        (i, r.video_path, int(r.frame_i), int(r.frame_j))
        for i, r in enumerate(rows.itertuples(index=False))
    ]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(decode_pair, vp, fi, fj, target_hw): idx
            for (idx, vp, fi, fj) in args
        }
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                arr = fut.result()
            except Exception:
                arr = None
            if arr is not None:
                out[idx] = arr
                valid[idx] = True
            done += 1
            if done % 1000 == 0:
                print(f"  decoded {done}/{len(rows)}", flush=True)
    return out, valid


# ----------- metrics -------------------------------------------------------


def cos_matrix(z: np.ndarray) -> np.ndarray:
    n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    return n @ n.T


def topk_primitive_purity(
    sim: np.ndarray, primitives: np.ndarray, episodes: np.ndarray, ks=(1, 5)
) -> dict[int, float]:
    """For each query, look up top-k nearest pairs from a DIFFERENT episode.
    Return primitive purity = fraction whose top-k are same primitive."""
    np.fill_diagonal(sim, -np.inf)
    out = {k: 0.0 for k in ks}
    n = len(sim)
    for i in range(n):
        # mask out same-episode neighbors
        valid = episodes != episodes[i]
        s = sim[i].copy()
        s[~valid] = -np.inf
        order = np.argsort(-s)
        for k in ks:
            top = order[:k]
            same = (primitives[top] == primitives[i]).mean()
            out[k] += same
    for k in ks:
        out[k] /= n
    return out


def action_vs_video_gap(
    sim: np.ndarray, primitives: np.ndarray, episodes: np.ndarray
) -> dict[str, float]:
    """mean cos(same_prim, diff_ep) - mean cos(diff_prim, same_ep)."""
    n = len(sim)
    same_p_diff_e = []
    diff_p_same_e = []
    for i in range(n):
        for j in range(i + 1, n):
            sp = primitives[i] == primitives[j]
            se = episodes[i] == episodes[j]
            if sp and not se:
                same_p_diff_e.append(sim[i, j])
            elif not sp and se:
                diff_p_same_e.append(sim[i, j])
    a = float(np.mean(same_p_diff_e)) if same_p_diff_e else float("nan")
    b = float(np.mean(diff_p_same_e)) if diff_p_same_e else float("nan")
    return {
        "mean_cos_same_primitive_diff_episode": a,
        "mean_cos_diff_primitive_same_episode": b,
        "gap": a - b,
        "n_same_p_diff_e": len(same_p_diff_e),
        "n_diff_p_same_e": len(diff_p_same_e),
    }


def same_episode_share_topk(sim: np.ndarray, episodes: np.ndarray, k: int = 5) -> float:
    np.fill_diagonal(sim, -np.inf)
    n = len(sim)
    share = 0.0
    for i in range(n):
        order = np.argsort(-sim[i])[:k]
        share += (episodes[order] == episodes[i]).mean()
    return share / n


def cross_dataset_retrieval(
    sim: np.ndarray, primitives: np.ndarray, datasets: np.ndarray, k: int = 5
) -> dict[str, float]:
    """For (A,B) with A!=B, query rows from A, restrict candidates to B, top-k primitive purity."""
    uniq = sorted(set(datasets.tolist()))
    out: dict[str, float] = {}
    for A in uniq:
        for B in uniq:
            if A == B:
                continue
            q_idx = np.where(datasets == A)[0]
            c_mask = datasets == B
            if len(q_idx) == 0 or c_mask.sum() == 0:
                continue
            score = 0.0
            count = 0
            for i in q_idx:
                s = sim[i].copy()
                s[~c_mask] = -np.inf
                s[i] = -np.inf
                order = np.argsort(-s)[:k]
                if order.size == 0:
                    continue
                score += (primitives[order] == primitives[i]).mean()
                count += 1
            out[f"{A}->{B}_top{k}"] = score / max(count, 1)
    return out


def similarity_histogram(
    sim: np.ndarray, primitives: np.ndarray, episodes: np.ndarray, save_path: Path
):
    n = len(sim)
    same_p_diff_e = []
    diff_p_same_e = []
    diff_p_diff_e = []
    for i in range(n):
        for j in range(i + 1, n):
            sp = primitives[i] == primitives[j]
            se = episodes[i] == episodes[j]
            v = sim[i, j]
            if sp and not se:
                same_p_diff_e.append(v)
            elif not sp and se:
                diff_p_same_e.append(v)
            elif not sp and not se:
                diff_p_diff_e.append(v)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(-1, 1, 51)
    plt.hist(
        same_p_diff_e,
        bins=bins,
        alpha=0.55,
        label=f"same-prim, diff-ep (n={len(same_p_diff_e)})",
    )
    plt.hist(
        diff_p_same_e,
        bins=bins,
        alpha=0.55,
        label=f"diff-prim, same-ep (n={len(diff_p_same_e)})",
    )
    plt.hist(
        diff_p_diff_e,
        bins=bins,
        alpha=0.30,
        label=f"diff-prim, diff-ep (n={len(diff_p_diff_e)})",
    )
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.title(save_path.stem)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def dataset_leakage_topk(sim: np.ndarray, datasets: np.ndarray, k: int = 5) -> float:
    np.fill_diagonal(sim, -np.inf)
    n = len(sim)
    share = 0.0
    for i in range(n):
        order = np.argsort(-sim[i])[:k]
        share += (datasets[order] == datasets[i]).mean()
    return share / n


# ----------- main ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--encoders",
        nargs="+",
        default=["CD_LAM_BASE", "CD_LAM_V1", "DINOv2_diff", "RAFT_pooled"],
    )
    ap.add_argument("--n-per-primitive", type=int, default=200)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decode-workers", type=int, default=16)
    ap.add_argument("--encode-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "similarity_histograms").mkdir(parents=True, exist_ok=True)

    print(f"[probe] loading {args.pair_index} ...")
    df = pd.read_parquet(args.pair_index)
    sample = sample_balanced(df, args.n_per_primitive, split=args.split, seed=args.seed)
    print(
        f"[probe] sampled {len(sample)} pairs from split={args.split} "
        f"({sample.dataset.value_counts().to_dict()})"
    )
    print(
        f"[probe] primitive distribution: {sample.primitive_label.value_counts().head(15).to_dict()}"
    )

    # Decode all pairs once.
    print("[probe] decoding pairs (240x320 RGB) ...")
    pairs, valid = decode_pairs_parallel(
        sample, target_hw=(240, 320), workers=args.decode_workers
    )
    sample = sample.loc[valid].reset_index(drop=True)
    pairs = pairs[valid]
    print(f"[probe] {len(sample)} valid pairs after decode  (dropped {(~valid).sum()})")

    primitives = sample["primitive_label"].astype(str).values
    episodes = sample["episode_id"].astype(str).values
    datasets = sample["dataset"].astype(str).values

    from experiments.exp_lam_benchmark.encoders import load_encoders  # noqa: E402

    encoders = load_encoders(tuple(args.encoders), device=args.device)
    print(f"[probe] loaded encoders: {list(encoders.keys())}")

    metrics = {
        "sample": {
            "n_pairs": len(sample),
            "n_primitives": int(sample["primitive_label"].nunique()),
            "n_episodes": int(sample["episode_id"].nunique()),
            "datasets": sample["dataset"].value_counts().to_dict(),
            "primitive_counts": sample["primitive_label"].value_counts().to_dict(),
        },
        "encoders": {},
    }

    cross_ds_results: dict[str, dict] = {}

    for ename, enc in encoders.items():
        print(f"\n[probe] encoding with {ename} ...")
        try:
            z = enc.encode(pairs, batch=args.encode_batch)
        except TypeError:
            z = enc.encode(pairs)
        z = np.asarray(z, dtype=np.float32)
        print(f"[probe] {ename} -> z shape {z.shape}")
        sim = cos_matrix(z)

        topk = topk_primitive_purity(sim.copy(), primitives, episodes, ks=(1, 5))
        gap = action_vs_video_gap(sim.copy(), primitives, episodes)
        se_share = same_episode_share_topk(sim.copy(), episodes, k=5)
        ds_topk = dataset_leakage_topk(sim.copy(), datasets, k=5)
        cross = cross_dataset_retrieval(sim.copy(), primitives, datasets, k=5)
        cross_ds_results[ename] = cross

        similarity_histogram(
            sim.copy(),
            primitives,
            episodes,
            out_dir / "similarity_histograms" / f"{ename}.png",
        )

        metrics["encoders"][ename] = {
            "out_dim": int(z.shape[1]),
            "primitive_top1_cross_episode": float(topk[1]),
            "primitive_top5_cross_episode": float(topk[5]),
            "action_vs_video_gap": gap,
            "same_episode_top5_share": float(se_share),
            "dataset_top5_share": float(ds_topk),
        }
        m = metrics["encoders"][ename]
        print(
            f"[probe] {ename}  top1={m['primitive_top1_cross_episode']:.3f}  "
            f"top5={m['primitive_top5_cross_episode']:.3f}  "
            f"gap={m['action_vs_video_gap']['gap']:.3f}  "
            f"same_ep_share@5={m['same_episode_top5_share']:.3f}  "
            f"dataset_share@5={m['dataset_top5_share']:.3f}"
        )

    with (out_dir / "per_encoder_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "cross_dataset_retrieval.json").open("w") as f:
        json.dump(cross_ds_results, f, indent=2)

    # Compact text summary
    md = [
        "# LAM_v2 Phase 5 probe — pre-training",
        "",
        f"Sample: {metrics['sample']['n_pairs']} pairs, "
        f"{metrics['sample']['n_primitives']} primitives, "
        f"{metrics['sample']['n_episodes']} episodes",
        f"Datasets: {metrics['sample']['datasets']}",
        "",
        "| encoder | top1 | top5 | gap | same_ep@5 | dataset@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ename, m in metrics["encoders"].items():
        md.append(
            f"| {ename} | {m['primitive_top1_cross_episode']:.3f} | "
            f"{m['primitive_top5_cross_episode']:.3f} | "
            f"{m['action_vs_video_gap']['gap']:.3f} | "
            f"{m['same_episode_top5_share']:.3f} | "
            f"{m['dataset_top5_share']:.3f} |"
        )
    md.append("")
    md.append("Cross-dataset top-5 primitive purity:")
    md.append("| encoder | A->B | top5 |")
    md.append("|---|---|---:|")
    for ename, cross in cross_ds_results.items():
        for k, v in cross.items():
            md.append(f"| {ename} | {k} | {v:.3f} |")
    (out_dir / "layerwise_probe_report.md").write_text("\n".join(md))
    print(f"\n[probe] wrote summary -> {out_dir / 'layerwise_probe_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
