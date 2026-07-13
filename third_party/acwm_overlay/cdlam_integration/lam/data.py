"""LAM training — Stage-1 data layer.

Two pieces:

* :class:`PrimitiveBalancedSampler` — primitive-balanced sampler over Stage-1 ``pair_index_train.parquet``
  (columns produced by the Stage-1 data indexer). Compared to the base sampler,
  the differences are:

  - Anchor / triplet pools come from the *valid_relation_pair* subset; same-episode hard
    negatives are taken from rows where *valid_same_episode_hard_negative* is true.
  - A separate **mask-eligible block** is sampled at every step (``B_mask`` rows)
    from rows where ``robosam_mask_training_eligible`` is true. These rows feed the
    interaction-weighted reconstruction / usage gap losses (plan §6.4 / §6.7).
    Mask-eligible rows are ALSO part of the regular real pool — duplicate decoding
    is fine and keeps the per-step bookkeeping trivial.
  - Returns per-row ``relation_weight`` (already in the parquet) so the trainer can
    weight L_rel.

* :class:`MaskCache` + helpers — load ``masks_m7.npz`` (interact / bsafe / motion uint8
  arrays of shape ``(T, H, W)`` produced by ``robo_sam/src/pipeline.py``) and resize
  M_interact at frame_j to the trainer's target H×W. LRU-cached because most rows in
  a clip share the same npz file.

This module is intentionally small and importable: the trainer wires it to the existing
core forward / decode helpers.
"""
from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

cv2.setNumThreads(0)


# =============== A1 sampler =================================================


def _ep_key(row) -> str:
    return f"{row.dataset}|{row.episode_id}"


class PrimitiveBalancedSampler:
    """Yield per-step batches with four blocks:

    * ``real``      — ``B_real`` anchor rows, primitive-balanced × episode, from
                       *valid_relation_pair* rows (full L_rec + L_rel + L_use anchor).
    * ``id``        — ``B_id`` identity-pair rows (L_id).
    * ``triplet``   — ``N_triplet`` (anchor, pos, neg) triples; positives from
                       different episode (prefer different dataset), negatives
                       from same episode (different primitive) when
                       ``valid_same_episode_hard_negative`` allows, else fallback.
    * ``mask``      — ``B_mask`` rows from *robosam_mask_training_eligible*; used
                       for interaction-weighted L_rec_inter and L_use_inter.

    All four blocks are returned as iloc indices into the parquet (independent
    blocks; the trainer decodes / batches them separately).
    """

    def __init__(
        self,
        pair_index_path: Path,
        canonical_primitives: List[str],
        p_neg_same_episode: float = 0.5,
        seed: int = 0,
    ) -> None:
        df = pd.read_parquet(pair_index_path)
        # filter columns we rely on; missing columns mean caller is on an old parquet
        required = [
            "pair_id", "dataset", "episode_id", "video_path", "frame_i", "frame_j",
            "primitive", "valid_relation_pair", "valid_same_episode_hard_negative",
            "robosam_mask_training_eligible", "robosam_interaction_mask_path",
        ]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"pair_index missing required column: {col}")

        # identity rows: pair_type=='identity' if column exists, else fall back
        # to "frame_i == frame_j" rows. F1's parquet has pair_type; A1's may not.
        if "pair_type" in df.columns:
            self.real = df[df["valid_relation_pair"].fillna(False).astype(bool)].reset_index(drop=True)
            self.id = df[df["pair_type"] == "identity"].reset_index(drop=True)
        else:
            self.real = df[df["valid_relation_pair"].fillna(False).astype(bool)].reset_index(drop=True)
            self.id = df[df["frame_i"] == df["frame_j"]].reset_index(drop=True)

        # mask pool: ALL rows with trainable RoboSAM mask + non-empty path. We
        # deliberately do NOT require `valid_relation_pair` here — the mask block
        # only feeds L_rec_inter and L_use_inter (plan §6.4 / §6.7), neither of
        # which uses primitive labels or relation structure. Restricting to
        # valid_relation_pair drops ~65% of trainable masks (mostly Bridge
        # single-primitive episodes); using the broader pool grows mask coverage
        # 914 → 2,656 on 2026-05-07 A1 build.
        mask_pool_filter = (
            df["robosam_mask_training_eligible"].fillna(False).astype(bool)
            & (df["robosam_interaction_mask_path"].fillna("").astype(str).str.len() > 0)
        )
        self.mask = df[mask_pool_filter].reset_index(drop=True)
        # back-compat name kept (was used by old smoke logs); now points at the
        # broader self.mask, not a subset of self.real
        self.mask_idx_in_real: np.ndarray = np.arange(len(self.mask))

        self.canonical = list(canonical_primitives)
        self.p_neg_same_ep = float(p_neg_same_episode)
        self.rng = np.random.default_rng(seed)

        # primitive → episode → list of iloc into self.real
        self.by_prim_ep: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        # primitive → list
        self.by_prim: Dict[str, List[int]] = defaultdict(list)
        # episode → list of (primitive, iloc)
        self.by_episode: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for idx, row in enumerate(self.real.itertuples(index=False)):
            ep = _ep_key(row)
            self.by_prim_ep[row.primitive][ep].append(idx)
            self.by_prim[row.primitive].append(idx)
            self.by_episode[ep].append((row.primitive, idx))

        # canonical primitives that survived the parquet (need >=2 episodes for triplets)
        self.primitives = [
            p for p in self.canonical
            if p in self.by_prim_ep and len(self.by_prim_ep[p]) >= 2
        ]
        if len(self.primitives) < 2:
            raise RuntimeError(
                f"need >=2 canonical primitives with >=2 episodes; got {self.primitives}"
            )

        # episodes that have hard negatives available
        self.hard_neg_episodes: set = set()
        if "valid_same_episode_hard_negative" in self.real.columns:
            for ep, rows in self.by_episode.items():
                # at least 2 distinct primitives in the episode → eligible hard-neg episode
                if len({p for (p, _) in rows}) >= 2:
                    self.hard_neg_episodes.add(ep)

    # -------- public API ------------------------------------------------------

    def sample(
        self,
        B_real: int,
        B_id: int,
        N_triplet: int,
        B_mask: int,
        B_hardneg_pairs: int = 0,
    ) -> Tuple[List[int], List[int], List[Tuple[int, int, int]], List[int], dict]:
        """Returns ``(real_idx, id_idx, triplets, mask_idx, meta)``.

        ``real_idx`` / ``id_idx`` / ``mask_idx`` are iloc into ``self.real`` / ``self.id`` /
        ``self.mask`` respectively. Each triplet entry is ``(a, p, n)`` of iloc into
        ``self.real``; ``a`` is also a member of ``real_idx`` so the trainer can reuse
        the encoded anchor.

        earlier recipe change (2026-05-08): ``B_hardneg_pairs`` controls explicit hard-neg
        injection. When > 0, the last ``2 * B_hardneg_pairs`` rows of ``real_idx``
        come from hard-neg pairs (same episode, different primitives) so that
        ``hard_neg_mask = (diff_prim & same_ep)`` in the SupCon loss has actual
        non-zero entries. earlier recipe/earlier recipe had this set to 0, so the hard-neg weighting
        path was structurally inactive — see ``progress/2026-05-07.md`` 01:38
        diagnosis. ``B_real`` stays the size guarantee (memory budget); the
        regular primitive-balanced subset shrinks to ``B_real - 2 * B_hardneg_pairs``
        to make room.
        """
        real_idx = self._sample_real(B_real, B_hardneg_pairs=B_hardneg_pairs)
        id_idx = self._sample_id(B_id)
        triplets = self._sample_triplets(real_idx, N_triplet)
        mask_idx = self._sample_mask(B_mask)
        meta = self._meta(real_idx, triplets)
        return real_idx, id_idx, triplets, mask_idx, meta

    # -------- block samplers --------------------------------------------------

    def _sample_hardneg_pairs(self, n_pairs: int) -> List[int]:
        """Sample ``n_pairs`` (anchor_prim row, hard-neg-prim row) PAIRS from the same
        multi-primitive episode. Returns a flat list of length ``2 * n_pairs``,
        ordered as [a0, n0, a1, n1, ...] so the SupCon mask sees both ends as
        same-ep diff-prim.

        Falls back to fewer pairs if not enough multi-prim episodes are available
        (rare with 17,793 hard-neg episodes in A1 train).
        """
        if n_pairs <= 0 or not self.hard_neg_episodes:
            return []
        ep_list = list(self.hard_neg_episodes)
        out: List[int] = []
        attempts = 0
        while len(out) < 2 * n_pairs and attempts < n_pairs * 4:
            attempts += 1
            ep = str(self.rng.choice(ep_list))
            rows = self.by_episode[ep]
            prims_in_ep = sorted({p for (p, _) in rows if p})
            if len(prims_in_ep) < 2:
                continue
            p1, p2 = self.rng.choice(prims_in_ep, size=2, replace=False)
            cand_p1 = [idx for (p, idx) in rows if p == p1]
            cand_p2 = [idx for (p, idx) in rows if p == p2]
            if not cand_p1 or not cand_p2:
                continue
            out.append(int(self.rng.choice(cand_p1)))
            out.append(int(self.rng.choice(cand_p2)))
        return out

    def _sample_real(self, B_real: int, B_hardneg_pairs: int = 0) -> List[int]:
        """Primitive-balanced sample + optional hard-neg pair injection.

        Layout:
          [B_real - 2*B_hardneg_pairs primitive-balanced rows] + [2*B_hardneg_pairs hard-neg rows]

        K primitives × M episodes × N pairs ≈ B_main, with padding.
        """
        n_hardneg_rows = 2 * max(0, int(B_hardneg_pairs))
        B_main = max(0, B_real - n_hardneg_rows)

        K = min(len(self.primitives), max(2, max(B_main, 1) // 6))
        prims = list(self.rng.choice(self.primitives, size=K, replace=False))
        per_p = max(1, B_main // K) if B_main > 0 else 0
        out: List[int] = []
        if B_main > 0:
            for p in prims:
                ep_list = list(self.by_prim_ep[p].keys())
                n_eps = min(len(ep_list), max(2, per_p // 2))
                if n_eps <= 0:
                    continue
                chosen_eps = self.rng.choice(ep_list, size=n_eps, replace=False)
                per_e = max(1, per_p // max(1, n_eps))
                for ep in chosen_eps:
                    pool = self.by_prim_ep[p][ep]
                    n_pick = min(len(pool), per_e)
                    pick = self.rng.choice(pool, size=n_pick, replace=False)
                    out.extend(int(x) for x in pick.tolist())
            if len(out) > B_main:
                out = list(self.rng.choice(out, size=B_main, replace=False))
                out = [int(x) for x in out]
            elif len(out) < B_main:
                extras = self.rng.choice(len(self.real), size=B_main - len(out), replace=False)
                out.extend(int(x) for x in extras.tolist())

        # Append hard-neg pairs at the end. The trainer's SupCon loss sees these
        # as in-batch (anchor_a, anchor_b) pairs where same_ep[a,b]=True and
        # diff_prim[a,b]=True → hard_neg_mask[a,b] = True.
        if n_hardneg_rows > 0:
            hn = self._sample_hardneg_pairs(B_hardneg_pairs)
            if len(hn) < n_hardneg_rows:
                # pad with regular extras to keep B_real exact
                pad = self.rng.choice(len(self.real), size=n_hardneg_rows - len(hn), replace=False)
                hn = hn + [int(x) for x in pad.tolist()]
            out.extend(hn)

        return out

    def _sample_id(self, B_id: int) -> List[int]:
        if len(self.id) == 0 or B_id <= 0:
            return []
        idx = self.rng.choice(len(self.id), size=B_id, replace=len(self.id) < B_id)
        return [int(x) for x in idx.tolist()]

    def _sample_triplets(
        self, real_idx: List[int], N: int,
    ) -> List[Tuple[int, int, int]]:
        triplets: List[Tuple[int, int, int]] = []
        if not real_idx or N <= 0:
            return triplets
        for _ in range(N):
            a = int(self.rng.choice(real_idx))
            a_row = self.real.iloc[a]
            a_prim = a_row.primitive
            a_ds = a_row.dataset
            a_ep = _ep_key(a_row)

            # positive: same primitive, different episode, prefer different dataset
            other_eps = [ep for ep in self.by_prim_ep[a_prim].keys() if ep != a_ep]
            if not other_eps:
                continue
            diff_ds = [ep for ep in other_eps if not ep.startswith(f"{a_ds}|")]
            if diff_ds and self.rng.random() < 0.6:
                p_ep = str(self.rng.choice(diff_ds))
            else:
                p_ep = str(self.rng.choice(other_eps))
            p = int(self.rng.choice(self.by_prim_ep[a_prim][p_ep]))

            # negative: prefer same episode, different primitive (valid_hard_neg)
            n: Optional[int] = None
            if a_ep in self.hard_neg_episodes and self.rng.random() < self.p_neg_same_ep:
                cands = [idx for (pp, idx) in self.by_episode[a_ep] if pp and pp != a_prim]
                if cands:
                    n = int(self.rng.choice(cands))
            if n is None:
                other_prims = [pp for pp in self.primitives if pp != a_prim]
                if not other_prims:
                    continue
                n_prim = str(self.rng.choice(other_prims))
                n = int(self.rng.choice(self.by_prim[n_prim]))

            triplets.append((a, p, n))
        return triplets

    def _sample_mask(self, B_mask: int) -> List[int]:
        if B_mask <= 0 or len(self.mask_idx_in_real) == 0:
            return []
        replace = len(self.mask_idx_in_real) < B_mask
        idx = self.rng.choice(self.mask_idx_in_real, size=B_mask, replace=replace)
        return [int(x) for x in idx.tolist()]

    # -------- diagnostics -----------------------------------------------------

    def _meta(self, real_idx: List[int], triplets: List[Tuple[int, int, int]]) -> dict:
        n_tri = max(1, len(triplets))
        same_ep_neg = sum(
            1
            for (a, _, n) in triplets
            if _ep_key(self.real.iloc[a]) == _ep_key(self.real.iloc[n])
        ) / n_tri
        cross_ds_pos = sum(
            1
            for (a, p, _) in triplets
            if self.real.iloc[a].dataset != self.real.iloc[p].dataset
        ) / n_tri
        prim_counts = pd.Series(
            [self.real.iloc[i].primitive for i in real_idx]
        ).value_counts(normalize=True)
        ent = float(-(prim_counts * np.log(prim_counts + 1e-12)).sum())
        ds_counts = pd.Series(
            [self.real.iloc[i].dataset for i in real_idx]
        ).value_counts(normalize=True)
        return {
            "valid_triplet_rate": len(triplets) / max(1, len(triplets) or 1),
            "same_episode_negative_rate": same_ep_neg,
            "cross_dataset_positive_rate": cross_ds_pos,
            "primitive_balance_entropy": ent,
            "actual_dataset_mix_real": ds_counts.to_dict(),
        }


# =============== Pair decoding (frames) =====================================


def _decode_pair_local(args):
    """Read 2 frames from an mp4 with grab+read (skips decoding non-target frames)."""
    video_path, fi, fj, target_hw = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    out = {fi: None, fj: None}
    needed = max(fi, fj)
    i = 0
    try:
        while i <= needed:
            if i in out:
                ok, fr = cap.read()
                if not ok:
                    break
                fr = cv2.resize(fr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                if out[i] is None:
                    out[i] = fr
            else:
                ok = cap.grab()
                if not ok:
                    break
            i += 1
    finally:
        cap.release()
    a = out[fi]
    b = out[fj] if fj != fi else a
    if a is None or b is None:
        return None
    return np.stack([a, b], axis=0).astype(np.uint8)


def decode_rows_parallel(
    rows: pd.DataFrame, target_hw: Tuple[int, int] = (240, 320), workers: int = 16,
):
    """Decode (frame_i, frame_j) pairs from a DataFrame slice in parallel.

    Returns ``(np.uint8[N,2,H,W,3], np.bool_[N])``; rows where decode failed are
    zero'd and marked invalid.
    """
    pool = [
        (r.video_path, int(r.frame_i), int(r.frame_j), target_hw)
        for r in rows.itertuples(index=False)
    ]
    out = np.zeros((len(rows), 2, target_hw[0], target_hw[1], 3), dtype=np.uint8)
    valid = np.zeros(len(rows), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, arr in enumerate(ex.map(_decode_pair_local, pool)):
            if arr is not None:
                out[i] = arr
                valid[i] = True
    return out, valid


# =============== Mask cache + decode ========================================


class MaskCache:
    """LRU cache for ``masks_m7.npz`` files.

    Each npz holds ``interact``/``bsafe``/``motion`` arrays of shape ``(T, H, W)``
    in uint8. We only need ``interact`` for L_rec_inter / L_use_inter (plan §6.4).
    Many pair_index rows share the same clip / episode (and therefore the same
    npz), so caching avoids repeated disk hits.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = int(max_entries)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get_interact(self, npz_path: str) -> Optional[np.ndarray]:
        if not npz_path:
            return None
        if npz_path in self._cache:
            self._cache.move_to_end(npz_path)
            return self._cache[npz_path]
        # Broad catch: corrupt npz / partially-truncated exports surface as
        # `zipfile.BadZipFile` (which in Python 3.10 inherits Exception, NOT
        # OSError — that bit a real run at 2026-05-07 step 181). KeyError
        # covers missing 'interact' key. We treat every load failure as
        # "no mask available" and let the caller drop the row. Logging the
        # bad path once per cache miss is helpful but not essential.
        try:
            with np.load(npz_path) as z:
                arr = z["interact"]
        except Exception:
            # Negative-cache the path so we don't re-attempt every step.
            # Use a sentinel (None array, but we need to indicate cached-bad).
            # OrderedDict can hold None — caller treats None as missing.
            self._cache[npz_path] = None  # type: ignore[assignment]
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return None
        # store as uint8 to keep cache cheap
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        self._cache[npz_path] = arr
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return arr

    def clear(self) -> None:
        self._cache.clear()


def _resize_mask_nearest(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """NEAREST resize of a (H, W) uint8 mask to target_hw. Preserves binary nature."""
    H, W = target_hw
    if mask.shape == (H, W):
        return mask
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def load_interaction_masks_for_rows(
    rows: pd.DataFrame,
    cache: MaskCache,
    target_hw: Tuple[int, int] = (240, 320),
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row interaction mask at frame_j, resized to ``target_hw``.

    Returns ``(masks, valid)``:

    * ``masks`` — ``np.float32[N, H, W]`` in {0, 1}; rows where the npz failed
      to load or the frame index is OOB get zeros.
    * ``valid`` — ``np.bool_[N]``; True iff the mask was successfully loaded
      and contained at least one positive pixel.
    """
    H, W = target_hw
    n = len(rows)
    out = np.zeros((n, H, W), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows.itertuples(index=False)):
        path = getattr(r, "robosam_interaction_mask_path", "") or ""
        arr = cache.get_interact(str(path))
        if arr is None:
            continue
        fj = int(r.frame_j)
        if fj < 0 or fj >= arr.shape[0]:
            continue
        m = _resize_mask_nearest(arr[fj], (H, W))
        if m.sum() <= 0:
            continue
        out[i] = m.astype(np.float32)
        valid[i] = True
    return out, valid
