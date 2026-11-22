from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical import round_div
from .codecs import Evidence
from .constants import METRICS


@dataclass(frozen=True)
class Reduction:
    value: int
    raw_value: int
    observations_used: int
    reducer: str
    trim_percent: int
    row_numbers: tuple[int, ...]


def select_replays(rows: list[Evidence], replay_count: int) -> list[Evidence]:
    ordered = sorted(rows, key=lambda item: item.row_no)
    if replay_count <= 0:
        raise ValueError("replay_count must be positive")
    if len(ordered) <= replay_count:
        return ordered
    indices = [index * len(ordered) // replay_count for index in range(replay_count)]
    return [ordered[index] for index in indices]


def reduce_integers(
    values: list[int], reducer: str, trim_percent: int = 20
) -> int:
    if not values:
        raise ValueError("cannot reduce an empty list")
    ordered_np = np.sort(np.asarray(values, dtype=np.int64))
    ordered = [int(value) for value in ordered_np.tolist()]
    count = len(ordered)
    if reducer == "median":
        middle = count // 2
        if count % 2:
            return ordered[middle]
        return round_div(ordered[middle - 1] + ordered[middle], 2)
    if reducer == "trimmed_mean":
        trim = (count * trim_percent) // 100
        retained = ordered[trim : count - trim] if trim else ordered
        if not retained:
            return reduce_integers(ordered, "median", 0)
        return round_div(sum(retained), len(retained))
    raise ValueError(f"unknown reducer: {reducer}")


def reduce_rows(
    rows: list[Evidence],
    replay_count: int,
    reducer: str,
    trim_percent: int,
    metric: str,
) -> Reduction:
    selected = select_replays(rows, replay_count)
    values = [row.metric_value(metric) for row in selected]
    raw_values = [row.metric_raw(metric) for row in selected]
    return Reduction(
        value=reduce_integers(values, reducer, trim_percent),
        raw_value=reduce_integers(raw_values, reducer, trim_percent),
        observations_used=len(selected),
        reducer=reducer,
        trim_percent=trim_percent,
        row_numbers=tuple(row.row_no for row in selected),
    )


def all_metric_reductions(
    rows: list[Evidence],
    replay_count: int,
    reducer: str,
    trim_percent: int,
) -> dict[str, Reduction]:
    return {
        metric: reduce_rows(rows, replay_count, reducer, trim_percent, metric)
        for metric in METRICS
    }
