"""Aggregate normalized records into per-(source, query) evidence.

All latency representative values are kept as integer microseconds.
Percentile/statistic calculations use NumPy for sorting and weighted
aggregation; final representative values are produced with explicit
integer division (floor toward zero for non-negative numerators).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .models import (
    CALIBER_AVG,
    CALIBER_MAX,
    NormalizedRecord,
    SourceDeclaration,
    SourceBatch,
)

_PERCENTILE_RANK = {"p50": 50, "p95": 95, "p99": 99}


@dataclass(frozen=True)
class Evidence:
    source_id: str
    source_type: str
    format_version: int
    query_id: int
    latency_us: int
    rows_scanned: int
    sample_count: int           # number of decoded records
    weight_total: int           # summed call/sample weights
    latency_caliber: str
    sample_period_ms: int
    filled_fields: tuple[str, ...]


def aggregate_batch(batch: SourceBatch) -> dict[int, Evidence]:
    source = batch.source
    by_query: dict[int, list[NormalizedRecord]] = {}
    for record in batch.records:
        by_query.setdefault(record.query_id, []).append(record)

    result: dict[int, Evidence] = {}
    for query_id in sorted(by_query):
        records = by_query[query_id]
        result[query_id] = _aggregate(source, query_id, records)
    return result


def _aggregate(source: SourceDeclaration, query_id: int,
               records: list[NormalizedRecord]) -> Evidence:
    latency = np.array([r.latency_us for r in records], dtype=np.int64)
    weights = np.array(
        [max(r.call_count, 1) for r in records], dtype=np.int64
    )
    representative = _representative(source.latency_caliber, latency, weights)

    rows_total = 0
    for record in records:
        if record.rows_scanned > 0:
            rows_total += record.rows_scanned * max(record.call_count, 1)

    filled = sorted({field for r in records for field in r.filled_fields})
    return Evidence(
        source_id=source.source_id,
        source_type=source.source_type,
        format_version=source.format_version,
        query_id=query_id,
        latency_us=int(representative),
        rows_scanned=int(rows_total),
        sample_count=len(records),
        weight_total=int(weights.sum()),
        latency_caliber=source.latency_caliber,
        sample_period_ms=source.sample_period_ms,
        filled_fields=tuple(filled),
    )


def _representative(caliber: str, latency: np.ndarray, weights: np.ndarray) -> int:
    if caliber == CALIBER_MAX:
        return int(latency.max())
    if caliber == CALIBER_AVG:
        total_weight = int(weights.sum())
        # Explicit floor division on non-negative integers.
        return int(int((latency * weights).sum()) // total_weight)

    percentile = _PERCENTILE_RANK[caliber]
    order = np.argsort(latency, kind="stable")
    ordered = latency[order]
    rank = math.ceil(percentile / 100.0 * len(ordered))  # 1-based nearest rank
    return int(ordered[rank - 1])
