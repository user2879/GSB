"""Parallel scheduling under the three budget constraints.

Constraints enforced here:

1. **global pixel budget** - total pixels that may be classified; once exhausted
   every not-yet-done tile gets ``E_PIXEL_BUDGET_EXHAUSTED`` and stays pending;
2. **per-tile quota** - a tile whose classified pixels exceed the quota aborts
   with ``E_TILE_BUDGET_EXHAUSTED``;
3. **fusion batch limit** - at most N fusion bands recomputed per batch; the
   rest get ``E_FUSION_BUDGET_PENDING`` and are retried first next batch.

Concurrency uses a thread pool sized from config and never exceeds it.  Every
task carries a stable sort key derived from row/column and the stage name, and
results are always collected in that deterministic order, so concurrent and
serial runs are byte-identical.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .config import Config
from .errors import Code, FailureList
from .model import Tile


def task_sort_key(stage: str, row: int, col: int, tile_id: str) -> Tuple[str, int, int, str]:
    return (stage, row, col, tile_id)


@dataclass
class BudgetState:
    pixel_budget: int
    pixels_used: int = 0
    tile_quota_hits: List[str] = field(default_factory=list)
    global_budget_hits: List[str] = field(default_factory=list)
    fusion_pending: List[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.pixel_budget - self.pixels_used)


class Scheduler:
    def __init__(self, config: Config, failures: FailureList):
        self.config = config
        self.failures = failures
        self.budget = BudgetState(config.pixel_budget)

    def _ordered(self, tiles: Dict[str, Tile]) -> List[Tile]:
        return sorted(tiles.values(), key=lambda t: task_sort_key("classify", t.spec.row, t.spec.col, t.tile_id))

    def run_classification(
        self,
        tiles: Dict[str, Tile],
        compute_fn: Callable[[Tile], object],
        cache_hit_fn: Callable[[Tile], bool],
    ) -> Tuple[List[object], List[Tile], List[str], List[str]]:
        """Return (results_in_order, pending_tiles, recomputed_ids, cached_ids)."""
        ordered = self._ordered(tiles)
        results: List[object] = []
        pending: List[Tile] = []
        recomputed: List[str] = []
        cached: List[str] = []

        # Separate cache hits (free, no budget) from work items first, but keep
        # the stable order in the final result assembly.
        work: List[Tile] = []
        for tile in ordered:
            if cache_hit_fn(tile):
                cached.append(tile.tile_id)
            else:
                work.append(tile)

        workers = max(1, min(self.config.max_workers, len(work) or 1))

        def _guarded(tile: Tile):
            # global budget gate
            if self.budget.remaining <= 0:
                return tile, None, Code.E_PIXEL_BUDGET_EXHAUSTED
            try:
                result = compute_fn(tile)
            except Exception as exc:  # per-tile compute isolation
                return tile, None, (Code.E_TILE_COMPUTE, str(exc))
            return tile, result, None

        if workers == 1 or len(work) <= 1:
            iterator = [_guarded(t) for t in work]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                iterator = list(pool.map(_guarded, work))

        # Re-impose deterministic order regardless of pool completion order.
        order_index = {t.tile_id: i for i, t in enumerate(ordered)}
        iterator.sort(key=lambda item: order_index[item[0].tile_id])

        for tile, result, error in iterator:
            if error == Code.E_PIXEL_BUDGET_EXHAUSTED:
                self.budget.global_budget_hits.append(tile.tile_id)
                self.failures.add(Code.E_PIXEL_BUDGET_EXHAUSTED, tile.tile_id,
                                  "global pixel budget exhausted; tile left pending")
                pending.append(tile)
                continue
            if isinstance(error, tuple):
                code, message = error
                self.failures.add(code, tile.tile_id, f"compute failure: {message}")
                pending.append(tile)
                continue
            pixels = getattr(result, "pixels_classified", self.config.tile_quota + 1)
            if pixels > self.config.tile_quota:
                self.budget.tile_quota_hits.append(tile.tile_id)
                self.failures.add(
                    Code.E_TILE_BUDGET_EXHAUSTED,
                    tile.tile_id,
                    f"tile needs {pixels} px > quota {self.config.tile_quota}",
                )
                pending.append(tile)
                continue
            if pixels > self.budget.remaining:
                self.budget.global_budget_hits.append(tile.tile_id)
                self.failures.add(
                    Code.E_PIXEL_BUDGET_EXHAUSTED,
                    tile.tile_id,
                    f"needs {pixels} px but only {self.budget.remaining} remain; pending",
                )
                pending.append(tile)
                continue
            self.budget.pixels_used += pixels
            recomputed.append(tile.tile_id)
            results.append(result)

        return results, pending, recomputed, cached

    def run_fusion_batch(
        self,
        pair_items: List[Tuple[str, object, object]],
        fuse_fn: Callable[[object, object], object],
        cache_hit_fn: Callable[[str], bool],
    ) -> Tuple[List[object], List[str], List[str], List[str]]:
        """Process one fusion batch honoring ``fusion_batch_limit``.

        ``pair_items`` is already in stable pair order. Returns
        (results, pending_pair_keys, recomputed, cached).
        """
        results: List[object] = []
        pending: List[str] = []
        recomputed: List[str] = []
        cached: List[str] = []
        limit = self.config.fusion_batch_limit
        processed = 0

        work: List[Tuple[str, object, object]] = []
        for pair_key, a, b in pair_items:
            if cache_hit_fn(pair_key):
                cached.append(pair_key)
                continue
            work.append((pair_key, a, b))

        for pair_key, a, b in work:
            if processed >= limit:
                pending.append(pair_key)
                self.budget.fusion_pending.append(pair_key)
                self.failures.add(
                    Code.E_FUSION_BUDGET_PENDING,
                    pair_key,
                    f"fusion batch limit {limit} reached; deferred to next batch",
                )
                continue
            results.append(fuse_fn(a, b))
            recomputed.append(pair_key)
            processed += 1
        return results, pending, recomputed, cached
