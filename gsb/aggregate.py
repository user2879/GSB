from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from .canonical import digest


def normalize_samples(samples: Sequence[Any]) -> tuple[list[int], list[int]]:
    groups: list[int] = []
    values: list[int] = []
    for sample in samples:
        if isinstance(sample, dict):
            groups.append(int(sample["group"]))
            values.append(int(sample["value"]))
        else:
            groups.append(int(sample[0]))
            values.append(int(sample[1]))
    return groups, values


def aggregate_samples(samples: Sequence[Any], fixed_scale: int) -> dict[str, Any]:
    groups_raw, values_raw = normalize_samples(samples)
    groups = np.asarray(groups_raw, dtype=np.int64)
    values = np.asarray(values_raw, dtype=np.int64)

    group_summary: dict[str, dict[str, int]] = {}
    if groups.size:
        unique_groups = np.unique(groups)
        for group in unique_groups:
            mask = groups == group
            count = int(np.count_nonzero(mask))
            total = int(values[mask].sum(dtype=object))
            mean_q = (total * int(fixed_scale)) // count
            group_summary[str(int(group))] = {
                "count": count,
                "sum": total,
                "mean_q": int(mean_q),
            }

    total_sum = int(values.sum(dtype=object)) if values.size else 0
    summary = {
        "rows": int(len(samples)),
        "total_sum": total_sum,
        "fixed_scale": int(fixed_scale),
        "groups": dict(sorted(group_summary.items(), key=lambda item: int(item[0]))),
    }
    summary["summary_digest"] = digest(_digest_without_self(summary))
    return summary


def _digest_without_self(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "summary_digest"}


def summary_digest(summary: dict[str, Any]) -> str:
    return digest(_digest_without_self(summary))


def timed_aggregate(samples: Sequence[Any], fixed_scale: int) -> tuple[dict[str, Any], int]:
    start = time.perf_counter_ns()
    summary = aggregate_samples(samples, fixed_scale)
    elapsed = max(1, (time.perf_counter_ns() - start) // 1_000_000)
    return summary, elapsed
