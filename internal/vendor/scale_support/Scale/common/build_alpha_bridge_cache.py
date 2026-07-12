"""Build an AgiBot Alpha frame/action cache for Stage 2 bridge training.

Output arrays:

* ``frames``: ``(N, 2, 240, 320, 3)`` uint8 frame pairs using the official
  LAM preprocessing path.
* ``action_22``: ``(N, 22)`` float32 raw adjacent-action deltas at source
  stride four.
* ``episode_id``: physical ``agibot_alpha:<task>:<episode>`` keys used to
  prevent segments from one recording crossing the split.
* ``segment_id``: source ``<task>-<episode>-<segment>`` provenance for every
  pair.
* ``split``: deterministic, episode-disjoint ``train`` or ``eval`` labels.

Pairs are distributed uniformly across each episode and include one identity
pair. The action layout is arm 14, gripper 2, head 2, waist 2, and base 2.

The cache is intentionally stored in raw adjacent-delta coordinates. A Stage 3
consumer must convert its normalized block-anchor representation into this
coordinate system before applying a bridge trained from this cache.

Example:
  python build_alpha_bridge_cache.py \
    --dataset-yaml /datasets/agibot-alpha/_dataset_paths_train.yaml \
    --out /outputs/bridge/alpha_bridge_cache.npz \
    --n-episodes 4000 --pairs-per-episode 8
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from Scale.common.raw_data_contract import agibot_alpha_physical_episode_key

LAM_HW = (240, 320)


def _preprocess_lam_pair(pair_raw: np.ndarray) -> np.ndarray:
    """Apply the bundled frame-exact LAM crop/resize path without model code."""
    from Scale.common.shard_io import _bundled_official_lam

    return _bundled_official_lam(pair_raw, lam_hw=LAM_HW)


def _md5_split(episode_id: str, eval_frac: float) -> str:
    """Return a deterministic episode-level train/eval split."""
    h = int(hashlib.md5(f"alpha_bridge|{episode_id}".encode()).hexdigest(), 16)
    return "eval" if (h % 1000) < int(eval_frac * 1000) else "train"


def _iter_episode_files(root: Path, chunks_size: int):
    """Enumerate segment metadata and data paths from one LeRobot shard."""
    import json
    info = json.load(open(root / "meta" / "info.json"))
    data_tpl = info["data_path"]          # data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    vid_tpl = info["video_path"]          # videos/chunk-.../{video_key}/episode_....mp4
    vid_key = "observation.images.top_head"
    eps = []
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ei = int(e["episode_index"])
            ec = ei // chunks_size
            sid = str(e.get("source_id", "")).strip()
            if not sid:
                raise ValueError(
                    f"{root}/meta/episodes.jsonl episode {ei} has no source_id; "
                    "physical split isolation cannot be established"
                )
            physical_episode_id = str(e.get("physical_episode_id", "")).strip()
            derived_episode_id = agibot_alpha_physical_episode_key(sid)
            if physical_episode_id and physical_episode_id != derived_episode_id:
                raise ValueError(
                    f"source_id {sid!r} disagrees with physical_episode_id "
                    f"{physical_episode_id!r}"
                )
            physical_episode_id = derived_episode_id
            pq = root / data_tpl.format(episode_chunk=ec, episode_index=ei)
            mp4 = root / vid_tpl.format(episode_chunk=ec, video_key=vid_key, episode_index=ei)
            eps.append((ei, physical_episode_id, sid, pq, mp4))
    return eps, int(info.get("chunks_size", chunks_size))


def _read_actions(parquet_path: Path) -> np.ndarray:
    """Read absolute episode actions as a ``(T, 22)`` float64 array."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(parquet_path), columns=["action"])
    rows = tbl.column("action").to_pylist()
    return np.asarray(rows, dtype=np.float64)   # (T, 22)


def _load_dataset_roots(dataset_yaml: Path) -> list[Path]:
    """Load dataset roots, resolving relative entries beside the YAML file."""
    import yaml

    dataset_yaml = dataset_yaml.resolve()
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dataset_path"), list):
        raise ValueError(f"{dataset_yaml} must contain a dataset_path list")
    roots: list[Path] = []
    for index, value in enumerate(payload["dataset_path"]):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{dataset_yaml} dataset_path[{index}] must be a non-empty string"
            )
        root = Path(value).expanduser()
        if not root.is_absolute():
            root = dataset_yaml.parent / root
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"dataset root is missing: {root}")
        roots.append(root)
    if not roots:
        raise ValueError(f"{dataset_yaml} dataset_path list is empty")
    return roots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-yaml", required=True, help="Path to _dataset_paths_train.yaml")
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument("--n-episodes", type=int, default=4000, help="Total episodes sampled across shards")
    ap.add_argument("--pairs-per-episode", type=int, default=10,
                    help="Pairs per episode, spread uniformly, plus one identity pair; 0 uses all starts")
    ap.add_argument("--stride", type=int, default=4,
                    help="Source-frame interval for each action delta; default 4 matches AgiBot sampling")
    ap.add_argument("--eval-frac", type=float, default=0.12, help="Episode-level evaluation fraction")
    ap.add_argument("--min-frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16, help="Decode worker threads")
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from Scale.common.shard_io import decode_frames_from_mp4_bytes

    rng = np.random.default_rng(args.seed)
    shards = _load_dataset_roots(Path(args.dataset_yaml))
    print(f"[build] {len(shards)} shards", flush=True)

    # Sample episodes across all shards.
    all_eps = []
    for root in shards:
        eps, cs = _iter_episode_files(root, 1000)
        for (ei, physical_episode_id, segment_id, pq, mp4) in eps:
            all_eps.append((physical_episode_id, segment_id, pq, mp4))
    print(
        f"[build] total segments={len(all_eps)} "
        f"physical_episodes={len({item[0] for item in all_eps})}; "
        f"requested={args.n_episodes}",
        flush=True,
    )
    if args.n_episodes and args.n_episodes < len(all_eps):
        idx = rng.choice(len(all_eps), args.n_episodes, replace=False)
        all_eps = [all_eps[i] for i in sorted(idx)]

    def _one_episode(item):
        physical_episode_id, segment_id, pq, mp4 = item
        try:
            act = _read_actions(pq)                  # (T,22)
            T = act.shape[0]
            if T < args.min_frames:
                return None
            split = _md5_split(physical_episode_id, args.eval_frac)
            # Spread fixed-stride pairs uniformly across the episode.
            stride = min(args.stride, max(1, T - 1))
            n_avail = max(1, T - stride)
            k = args.pairs_per_episode if args.pairs_per_episode > 0 else n_avail
            k = min(k, n_avail)
            starts = np.unique(np.linspace(0, n_avail - 1, k).astype(int))
            pairs = [(int(s), int(s + stride)) for s in starts]
            pairs.append((T // 2, T // 2))           # Identity anchor.
            need = sorted({f for p in pairs for f in p})
            mp4_bytes = Path(mp4).read_bytes()
            raw = decode_frames_from_mp4_bytes(mp4_bytes, need)   # {idx: (H,W,3) native}
            if not raw:
                return None
            out_frames, out_act = [], []
            for (fi, fj) in pairs:
                if fi not in raw or fj not in raw:
                    continue
                pair_raw = np.stack([raw[fi], raw[fj]], axis=0)          # (2,H,W,3) native
                pair_lam = _preprocess_lam_pair(pair_raw)
                out_frames.append(pair_lam.astype(np.uint8))
                out_act.append((act[fj] - act[fi]).astype(np.float32))
            if not out_frames:
                return None
            return (
                physical_episode_id,
                segment_id,
                split,
                np.stack(out_frames),
                np.stack(out_act),
            )
        except Exception as e:
            return ("__err__", str(e))

    frames_all, act_all = [], []
    ep_all, segment_all, split_all = [], [], []
    n_err = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_one_episode, all_eps):
            done += 1
            if res is None:
                continue
            if res[0] == "__err__":
                n_err += 1
                continue
            physical_episode_id, segment_id, split, fr, ac = res
            frames_all.append(fr)
            act_all.append(ac)
            ep_all.extend([physical_episode_id] * len(fr))
            segment_all.extend([segment_id] * len(fr))
            split_all.extend([split] * len(fr))
            if done % 500 == 0:
                npairs = sum(len(f) for f in frames_all)
                print(f"[build] episodes {done}/{len(all_eps)}  pairs={npairs}  err={n_err}", flush=True)

    if not frames_all:
        raise RuntimeError(
            "no valid bridge pairs were built; inspect the selected videos, "
            "actions, minimum-frame threshold, and decode errors"
        )
    frames = np.concatenate(frames_all, axis=0)
    action_22 = np.concatenate(act_all, axis=0).astype(np.float32)
    episode_id = np.asarray(ep_all, dtype=object)
    segment_id = np.asarray(segment_all, dtype=object)
    split = np.asarray(split_all, dtype=object)
    n_tr = int((split == "train").sum())
    n_te = int((split == "eval").sum())
    if n_tr == 0 or n_te == 0:
        raise RuntimeError(
            "bridge cache requires non-empty episode-disjoint train and eval "
            f"splits, got train={n_tr}, eval={n_te}; select more physical episodes"
        )
    print(
        f"[build] frames{frames.shape} action_22{action_22.shape} "
        f"train={n_tr} eval={n_te} physical_episodes={len(set(ep_all))} "
        f"segments={len(set(segment_all))} err={n_err}",
        flush=True,
    )
    # Report per-dimension training standard deviations for inspection.
    tr = np.where(split == "train")[0]
    print(
        "[build] action_22 train per-dim std: "
        f"{np.round(action_22[tr].std(0), 4).tolist()}",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        frames=frames,
        action_22=action_22,
        episode_id=episode_id,
        segment_id=segment_id,
        split=split,
    )
    print(f"[build] saved -> {out}  ({frames.nbytes/1e9:.2f} GB frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
