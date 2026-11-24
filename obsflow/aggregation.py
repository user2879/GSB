"""Fixed-point integer sliding-window aggregation (pure, deterministic).

All functions here are side-effect free and independent of batching, so the
same record set yields byte-identical window results regardless of how the
caller splits batches or advances windows.

Quantisation
------------
A record carries an integer ``raw_value`` at ``record_scale``. The session
uses ``session_scale`` Q.  The fixed-point integer value is

    q = truncate(raw_value * Q / record_scale)          (toward zero)

so a single value has strict bound ``|value - q| < 1`` in quantised units and
a window of ``n`` values has strict bound ``|window_sum_error| < n``.

Saturation
----------
Quantised values and sums are clamped to the signed 64-bit range; the
``saturation`` field records whether clamping occurred and at which stage.
The mean is the truncated (toward zero) quotient ``sum_q // count``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .canonical import content_digest
from .models import (
    INT64_MAX,
    INT64_MIN,
    NormalizedRecord,
    WindowParams,
    WindowResult,
)

TRUNCATION = "toward_zero"
SATURATION = "int64_clamp"


def quantize_value(raw_value: int, record_scale: int,
                   session_scale: int) -> tuple[int, bool]:
    """Quantise one integer measurement; return ``(q, saturated)``."""
    numerator = raw_value * session_scale
    # Python's // floors, so use explicit truncation toward zero.
    q = abs(numerator) // record_scale
    if numerator < 0:
        q = -q
    saturated = False
    if q > INT64_MAX:
        q = INT64_MAX
        saturated = True
    elif q < INT64_MIN:
        q = INT64_MIN
        saturated = True
    return q, saturated


def quantize_batch(values: Sequence[int], scales: Sequence[int],
                   session_scale: int) -> tuple[list[int], list[bool]]:
    """Vectorised quantisation with a pure-Python fallback.

    NumPy is used when available for bulk conversion; object dtype keeps full
    arbitrary precision before clamping so semantics match the scalar path.
    """
    try:
        import numpy as np

        arr = np.asarray(values, dtype=object)
        sc = np.asarray(scales, dtype=object)
        numerator = arr * session_scale
        # Truncation toward zero implemented on absolute values.
        q = (np.abs(numerator) // sc) * np.sign(numerator).astype(object)
        qs = [int(x) for x in q]
    except Exception:  # pragma: no cover - fallback safety net
        qs = [
            (abs(v * session_scale) // s) * (1 if v * session_scale >= 0 else -1)
            for v, s in zip(values, scales)
        ]

    saturated_flags: list[bool] = []
    for idx, q in enumerate(qs):
        if q > INT64_MAX:
            qs[idx] = INT64_MAX
            saturated_flags.append(True)
        elif q < INT64_MIN:
            qs[idx] = INT64_MIN
            saturated_flags.append(True)
        else:
            saturated_flags.append(False)
    return qs, saturated_flags


def _saturate(value: int) -> tuple[int, bool]:
    if value > INT64_MAX:
        return INT64_MAX, True
    if value < INT64_MIN:
        return INT64_MIN, True
    return value, False


def ordered_records(
        records: Iterable[NormalizedRecord]) -> list[NormalizedRecord]:
    """Global deterministic ordering shared by every code path."""
    return sorted(records, key=lambda r: (r.seq, r.source_id, r.metric_key,
                                          r.line_no))


def window_slices(total: int, window_size: int,
                  slide_step: int) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` index slices of full windows."""
    if total < window_size:
        return []
    last_start = total - window_size
    starts = range(0, last_start + 1, slide_step)
    return [(start, start + window_size) for start in starts]


def aggregate_window(records: Sequence[NormalizedRecord], metric_key: str,
                     window_index: int, params: WindowParams,
                     window_version: int) -> WindowResult:
    """Aggregate one fully populated window of one metric key."""
    window_size = params.window_size
    if len(records) != window_size:
        raise ValueError(
            f"window {window_index} for {metric_key!r} expects "
            f"{window_size} records, got {len(records)}")

    q_values, sat_quant = quantize_batch(
        [r.raw_value for r in records],
        [r.record_scale for r in records],
        params.scale,
    )

    running = 0
    sum_saturated = False
    for q, saturated in zip(q_values, sat_quant):
        running += q
        running, clamped = _saturate(running)
        sum_saturated = sum_saturated or clamped or saturated

    count = len(records)
    # Truncation toward zero mean.
    mean_q = abs(running) // count
    if running < 0:
        mean_q = -mean_q
    mean_q, mean_saturated = _saturate(mean_q)

    # Strict bounds: |single error| < 1, |sum error| < count (quantised units).
    value_error_bound_q = 1
    sum_error_bound_q = count

    first = records[0]
    last = records[-1]
    input_digest = content_digest([
        [r.source_id, r.seq, r.ts_ms, r.metric_key, r.raw_value,
         r.record_scale, r.source_format] for r in records
    ])
    saturation = "none"
    if sum_saturated:
        saturation = "value_or_sum_clamped"
    elif mean_saturated:
        saturation = "mean_clamped"

    payload = {
        "metric_key": metric_key,
        "window_index": window_index,
        "start_source_id": first.source_id,
        "end_source_id": last.source_id,
        "start_seq": first.seq,
        "end_seq": last.seq,
        "count": count,
        "sum_q": running,
        "mean_q": mean_q,
        "value_error_bound_q": value_error_bound_q,
        "sum_error_bound_q": sum_error_bound_q,
        "scale": params.scale,
        "truncation": TRUNCATION,
        "saturation": saturation,
        "window_version": window_version,
        "status": "ready",
        "input_digest": input_digest,
    }
    return WindowResult(
        metric_key=metric_key,
        window_index=window_index,
        start_source_id=first.source_id,
        end_source_id=last.source_id,
        start_seq=first.seq,
        end_seq=last.seq,
        count=count,
        sum_q=running,
        mean_q=mean_q,
        value_error_bound_q=value_error_bound_q,
        sum_error_bound_q=sum_error_bound_q,
        scale=params.scale,
        truncation=TRUNCATION,
        saturation=saturation,
        window_version=window_version,
        status="ready",
        input_digest=input_digest,
        result_digest=content_digest(payload),
        lease_id=None,
    )


def aggregate_all(
        per_metric: dict[str, list[NormalizedRecord]],
        params: WindowParams,
        versions: dict[tuple[str, int], int] | None = None,
        ) -> list[WindowResult]:
    """Aggregate every full window for every metric key (stable order)."""
    versions = versions or {}
    results: list[WindowResult] = []
    for metric_key in sorted(per_metric):
        ordered = ordered_records(per_metric[metric_key])
        slices = window_slices(len(ordered), params.window_size,
                               params.slide_step)
        for window_index, (start, end) in enumerate(slices):
            version = versions.get((metric_key, window_index), 1)
            results.append(aggregate_window(
                ordered[start:end], metric_key, window_index, params,
                version))
    return results


def group_by_metric(
        records: Iterable[NormalizedRecord]
        ) -> dict[str, list[NormalizedRecord]]:
    grouped: dict[str, list[NormalizedRecord]] = {}
    for record in records:
        grouped.setdefault(record.metric_key, []).append(record)
    return grouped
