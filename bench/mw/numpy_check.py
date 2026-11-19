"""Optional independent cross-check of aggregates using numpy.

The production path never depends on numpy; this module recomputes finalized
sums/counts and latency percentiles independently so tests can demonstrate a
second implementation agreeing with the ledger.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .latency import nearest_rank_index


def cross_check_aggregates(replay_views: list[dict[str, Any]],
                           finalized_values: dict[str, list[int]],
                           scale: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view in replay_views:
        key = view["partition"]
        values = np.asarray(finalized_values.get(key, []), dtype=np.int64)
        count = int(values.size)
        raw_sum = int(values.sum(dtype=np.object_)) if count else 0
        fixed = int(raw_sum) * (10 ** scale)
        result[key] = {
            "finalized_count_match": count == view["finalized_count"],
            "sum_raw_match": raw_sum == view["sum_raw"],
            "sum_fixed_match": fixed == view["sum_fixed"],
        }
    return result


def cross_check_percentile(samples: Sequence[int]) -> int:
    if not samples:
        raise ValueError("percentile requires samples")
    arr = np.sort(np.asarray(samples, dtype=np.int64))
    return int(arr[nearest_rank_index(arr.size, 95.0)])
