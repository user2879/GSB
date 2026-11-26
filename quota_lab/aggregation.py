from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .fixedpoint import floor_divide


def aggregate_samples(
    samples: Sequence[Sequence[int]],
    *,
    q: int,
    pressure_den: int,
) -> dict[str, Any]:
    array = np.asarray(samples, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("samples must be [group, value] pairs")

    groups = array[:, 0]
    values = array[:, 1]
    total_count = int(values.shape[0])
    total_sum = int(values.sum(dtype=np.int64))
    abs_sum = int(np.abs(values).sum(dtype=np.int64))

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_count = int(unique_groups.shape[0])
    group_sums_array = np.zeros(group_count, dtype=np.int64)
    group_counts_array = np.zeros(group_count, dtype=np.int64)
    np.add.at(group_sums_array, inverse, values)
    np.add.at(group_counts_array, inverse, 1)

    group_sums = {
        str(int(group)): {
            "sum": int(group_sums_array[index]),
            "count": int(group_counts_array[index]),
        }
        for index, group in enumerate(unique_groups)
    }

    weighted = groups.astype(np.int64) * 1009 + values.astype(np.int64) * 9176
    checksum = int(np.mod(weighted.sum(dtype=np.int64), np.int64(1_000_000_007)))

    mean = floor_divide(total_sum * q, total_count)
    pressure_num = 1 + (abs_sum % pressure_den)

    return {
        "row_count": total_count,
        "group_count": group_count,
        "total_sum": total_sum,
        "abs_sum": abs_sum,
        "checksum": checksum,
        "group_sums": group_sums,
        "fixed_point": {
            "q": q,
            "mean_q": mean.quotient,
            "mean_truncation": mean.residual,
            "mean_bound_q": mean.bound_q,
            "pressure_num": pressure_num,
            "pressure_den": pressure_den,
        },
    }


def shard_watermark(demand_q: int, pressure_num: int, pressure_den: int) -> dict[str, Any]:
    division = floor_divide(demand_q * pressure_num, pressure_den)
    return {
        "water_q": division.quotient,
        "truncation": division.residual,
        "trunc_bound_q": division.bound_q,
        "saturated": division.saturated,
    }
