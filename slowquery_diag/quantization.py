"""Session fixed-point quantization and explicit error-bound accounting.

Scaling factor
    ``s = 10 ** k`` with integer ``k`` in [0, 3].  One quantization tick
    represents ``s`` base integer units (microseconds for latency, rows
    for scan counts, basis points for expected gain).

Quantization (all arithmetic is integer only)
    ``q(x) = min(SAT, x // s)``  -- truncation is *floor toward zero*;
    inputs are non-negative so floor is unambiguous.
    Saturation clamps to ``SAT = 2**31 - 1`` (overflow never wraps).

Error decomposition per query/metric (raw base units)
    B_total = B_quant + B_caliber + B_period
    B_quant   = 2 * (s - 1)              two compared records, each loses < 1 tick
    B_caliber = latency-algebra caliber radius of the participating sources
    B_period  = 1000 * max period delta  latency only; zero for row counts
All contributions are also converted into fixed-point ticks with explicit
floor rounding, and the dominant (largest raw) segment is annotated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .aggregation import Evidence
from .models import (
    SOURCE_APP,
    SOURCE_DB,
    SOURCE_SLOW,
    SourceBatch,
)

SATURATION = (1 << 31) - 1
ALLOWED_SCALES = (1, 10, 100, 1000)


class Quantizer:
    """Fixed-point quantizer with floor truncation and saturation clamp."""

    def __init__(self, scale_exponent: int) -> None:
        if scale_exponent not in (0, 1, 2, 3):
            raise ValueError("scale_exponent must be one of 0,1,2,3")
        self.scale_exponent = scale_exponent
        self.scale = 10 ** scale_exponent
        self.saturated_count = 0

    def quantize(self, raw_value: int) -> int:
        if raw_value < 0:
            raise ValueError("only non-negative raw values are supported")
        scaled = raw_value // self.scale
        if scaled > SATURATION:
            self.saturated_count += 1
            return SATURATION
        return scaled

    def tick_error_raw(self) -> int:
        """Maximum truncation loss of one record, in raw base units."""
        return self.scale - 1

    def to_ticks(self, raw_value: int) -> int:
        """Floor-convert a raw error bound into whole ticks."""
        return raw_value // self.scale


@dataclass(frozen=True)
class ErrorContribution:
    name: str
    raw_units: int
    ticks: int
    dominant: bool


@dataclass(frozen=True)
class ErrorBound:
    metric: str
    query_id: int
    quant_raw: int
    caliber_raw: int
    period_raw: int
    total_raw: int
    total_ticks: int
    dominant_segment: str
    contributions: tuple[ErrorContribution, ...]


@dataclass(frozen=True)
class SourceAggregateError:
    source_id: str
    query_count: int
    quant_raw: int
    caliber_raw: int
    period_raw: int
    total_raw: int


# ---------------------------------------------------------------------------
# Caliber (latency algebra) difference radii.
# ---------------------------------------------------------------------------

def caliber_radius_latency(
    batches: Iterable[SourceBatch], query_id: int
) -> int:
    """Worst-case pairwise latency-algebra radius in microseconds.

    Sources reporting the same caliber have zero caliber conflict.  When
    calibers differ the radius is derived from the pooled latency spread
    of every record for ``query_id`` (max representative minus min
    representative), which bounds any percentile/avg/max disagreement.
    """

    representatives: list[int] = []
    calibers: set[str] = set()
    for batch in batches:
        calibers.add(batch.source.latency_caliber)
        for record in batch.records:
            if record.query_id == query_id:
                representatives.append(record.latency_us)
    if len(calibers) <= 1 or len(representatives) < 2:
        return 0
    return max(representatives) - min(representatives)


def caliber_radius_rows(evidences: Iterable[Evidence]) -> int:
    """Rows share one sum-only caliber; filled (unreported) values add no
    caliber conflict, so the radius is zero unless mixed reporting types
    are present."""

    types = {e.source_type for e in evidences}
    if SOURCE_DB in types and (SOURCE_APP in types or SOURCE_SLOW in types):
        # DB sums scanned rows directly; app/slow may backfill zero.
        # The bound is the largest adopted row count difference.
        values = [e.rows_scanned for e in evidences]
        return max(values) - min(values)
    return 0


def period_radius_latency(evidences: Iterable[Evidence]) -> int:
    periods = [e.sample_period_ms for e in evidences]
    if len(periods) < 2:
        return 0
    return (max(periods) - min(periods)) * 1000


def build_error_bound(
    quantizer: Quantizer,
    metric: str,
    query_id: int,
    evidences: list[Evidence],
    batches: list[SourceBatch],
) -> ErrorBound:
    quant_raw = 2 * quantizer.tick_error_raw()
    if metric == "latency_us":
        caliber_raw = caliber_radius_latency(batches, query_id)
        period_raw = period_radius_latency(evidences)
    else:
        caliber_raw = caliber_radius_rows(evidences)
        period_raw = 0

    segments = {
        "quantization": quant_raw,
        "caliber": caliber_raw,
        "sampling_period": period_raw,
    }
    total_raw = quant_raw + caliber_raw + period_raw
    dominant = max(
        segments.items(), key=lambda item: (item[1], item[0] == "quantization")
    )[0]
    contributions = tuple(
        ErrorContribution(
            name=name,
            raw_units=value,
            ticks=quantizer.to_ticks(value),
            dominant=(name == dominant),
        )
        for name, value in segments.items()
    )
    return ErrorBound(
        metric=metric,
        query_id=query_id,
        quant_raw=quant_raw,
        caliber_raw=caliber_raw,
        period_raw=period_raw,
        total_raw=total_raw,
        total_ticks=quantizer.to_ticks(total_raw),
        dominant_segment=dominant,
        contributions=contributions,
    )


def indistinguishable(value_a: int, value_b: int, bound_ticks: int) -> bool:
    """Two fixed-point values are within the synthesized decision bound."""
    return abs(value_a - value_b) <= max(bound_ticks, 1)


def aggregate_source_errors(
    source_id: str, bounds: list[ErrorBound]
) -> SourceAggregateError:
    return SourceAggregateError(
        source_id=source_id,
        query_count=len(bounds),
        quant_raw=sum(b.quant_raw for b in bounds),
        caliber_raw=sum(b.caliber_raw for b in bounds),
        period_raw=sum(b.period_raw for b in bounds),
        total_raw=sum(b.total_raw for b in bounds),
    )
