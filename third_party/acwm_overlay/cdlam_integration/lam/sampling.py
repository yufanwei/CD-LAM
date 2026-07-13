"""LAM pairwise+ data layer extensions.

:class:`DatasetBalancedSampler` extends :class:`cdlam_integration.lam.data.PrimitiveBalancedSampler` with:

* **primitive × dataset balanced sampling**: instead of just primitive-balanced
  (where AgiBot:Bridge falls into the natural 9:1), each chosen primitive's
  episodes are split between datasets up to ``dataset_balance_quota`` (default
  ``[0.5, 0.5]``).
* **cross-dataset positive quota**: ensure that for each primitive present in
  the batch, both datasets contribute episodes when both have samples for that
  primitive — which guarantees the SigLIP graph contains AgiBot↔Bridge
  same-prim positives.

Default flags reproduce PrimitiveBalancedSampler behavior (back-compat).

Per CD-LAM dashboard finding: leakage@5 ≈ 0.91 across all training/pairwise ckpts is the
sampler's natural-distribution artifact, not a loss bug — fixing it requires
seeing AgiBot↔Bridge same-prim positives in the batch graph.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List


from cdlam_integration.lam.data import PrimitiveBalancedSampler, _ep_key


class DatasetBalancedSampler(PrimitiveBalancedSampler):
    """Dataset-balanced extension of PrimitiveBalancedSampler (pairwise+).

    Adds two flags to :meth:`__init__`:

    * ``dataset_balanced``: when True, ``_sample_real`` enforces dataset balance
      across the chosen primitives. Default False ⇒ same as PrimitiveBalancedSampler.
    * ``min_cross_dataset_per_primitive``: when ≥ 1 and ``dataset_balanced`` is
      True, each chosen primitive contributes at least one anchor from each
      dataset (when available).
    """

    def __init__(
        self,
        pair_index_path: Path,
        canonical_primitives: List[str],
        p_neg_same_episode: float = 0.5,
        seed: int = 0,
        *,
        dataset_balanced: bool = False,
        min_cross_dataset_per_primitive: int = 1,
    ) -> None:
        super().__init__(
            pair_index_path=pair_index_path,
            canonical_primitives=canonical_primitives,
            p_neg_same_episode=p_neg_same_episode,
            seed=seed,
        )
        self.dataset_balanced = bool(dataset_balanced)
        self.min_cross_dataset_per_primitive = int(min_cross_dataset_per_primitive)

        # additional indices: by_prim_ds[prim][dataset] = list of (ep_key, iloc)
        # (needed for fast dataset-balanced sampling)
        self.by_prim_ds: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # by_prim_ds_ep[prim][dataset][ep_key] = list of iloc
        self.by_prim_ds_ep: Dict[str, Dict[str, Dict[str, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for idx, row in enumerate(self.real.itertuples(index=False)):
            ds = row.dataset
            ep = _ep_key(row)
            self.by_prim_ds[row.primitive][ds].append(idx)
            self.by_prim_ds_ep[row.primitive][ds][ep].append(idx)
        # primitives that exist in both datasets — cross-dataset positives are
        # only producible from these
        self.cross_dataset_primitives = sorted([
            p for p, by_ds in self.by_prim_ds.items()
            if len(by_ds) >= 2 and all(len(v) > 0 for v in by_ds.values())
        ])

    # -------- override _sample_real ------------------------------------------

    def _sample_real(self, B_real: int, B_hardneg_pairs: int = 0) -> List[int]:
        """Same signature as PrimitiveBalancedSampler._sample_real but dataset-balanced when
        ``self.dataset_balanced`` is True."""
        if not self.dataset_balanced:
            return super()._sample_real(B_real, B_hardneg_pairs=B_hardneg_pairs)

        n_hardneg_rows = 2 * max(0, int(B_hardneg_pairs))
        B_main = max(0, B_real - n_hardneg_rows)
        if B_main <= 0:
            # delegate hard-neg pairs to parent's helper
            out: List[int] = []
            if n_hardneg_rows > 0:
                out.extend(self._sample_hardneg_pairs(B_hardneg_pairs))
            return out

        # Prefer primitives that have BOTH datasets (cross_dataset_primitives)
        # when min_cross_dataset_per_primitive >= 1; fill the rest with any
        # primitive in self.primitives.
        K = min(len(self.primitives), max(2, B_main // 6))
        if self.min_cross_dataset_per_primitive >= 1 and self.cross_dataset_primitives:
            # take min(K, |cross-ds primitives|) from cross-ds, rest from any
            n_cross = min(K, len(self.cross_dataset_primitives))
            cross_chosen = list(self.rng.choice(
                self.cross_dataset_primitives, size=n_cross, replace=False,
            ))
            remaining = [p for p in self.primitives if p not in cross_chosen]
            n_extra = K - n_cross
            extra: List[str] = []
            if n_extra > 0 and remaining:
                extra = list(self.rng.choice(
                    remaining, size=min(n_extra, len(remaining)), replace=False,
                ))
            prims = cross_chosen + extra
        else:
            prims = list(self.rng.choice(self.primitives, size=K, replace=False))

        per_p = max(1, B_main // max(1, len(prims)))
        out: List[int] = []
        for p in prims:
            ds_pool = self.by_prim_ds[p]
            datasets_with_pool = [d for d, ilocs in ds_pool.items() if len(ilocs) > 0]
            if not datasets_with_pool:
                continue
            # split per_p across datasets as evenly as possible
            n_ds = len(datasets_with_pool)
            per_ds = max(1, per_p // n_ds)
            if p in self.cross_dataset_primitives and self.min_cross_dataset_per_primitive >= 1:
                # guarantee at least min_cross_dataset_per_primitive from each dataset
                per_ds = max(per_ds, self.min_cross_dataset_per_primitive)
            for ds in datasets_with_pool:
                ep_dict = self.by_prim_ds_ep[p][ds]
                ep_list = list(ep_dict.keys())
                if not ep_list:
                    continue
                n_eps = min(len(ep_list), max(2, per_ds // 2))
                chosen_eps = self.rng.choice(ep_list, size=n_eps, replace=False)
                per_e = max(1, per_ds // max(1, n_eps))
                for ep in chosen_eps:
                    pool = ep_dict[ep]
                    n_pick = min(len(pool), per_e)
                    if n_pick <= 0:
                        continue
                    pick = self.rng.choice(pool, size=n_pick, replace=False)
                    out.extend(int(x) for x in pick.tolist())

        # cap / pad to B_main
        if len(out) > B_main:
            out = list(self.rng.choice(out, size=B_main, replace=False))
            out = [int(x) for x in out]
        elif len(out) < B_main:
            extras = self.rng.choice(len(self.real), size=B_main - len(out), replace=False)
            out.extend(int(x) for x in extras.tolist())

        # append hard-neg pairs at the tail (same as parent class)
        if n_hardneg_rows > 0:
            out.extend(self._sample_hardneg_pairs(B_hardneg_pairs))
        return out


__all__ = ["DatasetBalancedSampler"]
