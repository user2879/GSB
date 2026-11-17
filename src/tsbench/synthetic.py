"""Deterministic synthetic series generation and update-plan application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from .errors import SeriesFailure


@dataclass
class SeriesData:
    series_id: str
    timestamps: list[str]
    values: list[float]

    def __len__(self) -> int:
        return len(self.values)


def generate_series(spec: dict, start_date: date) -> SeriesData:
    """Generate one series with a fixed seed; no global RNG state is touched."""
    rng = np.random.default_rng(int(spec["seed"]))
    length = int(spec["length"])
    index = np.arange(length, dtype=np.float64)

    trend_cfg = spec["trend"]
    trend = float(trend_cfg.get("slope", 0.0)) * index + float(
        trend_cfg.get("intercept", 0.0)
    )

    season_cfg = spec["seasonal"]
    period = float(season_cfg.get("period", 1))
    amplitude = float(season_cfg.get("amplitude", 0.0))
    phase = float(season_cfg.get("phase", 0.0))
    seasonal = amplitude * np.sin(2.0 * np.pi * index / period + phase)

    noise_cfg = spec["noise"]
    noise = rng.normal(
        loc=float(noise_cfg.get("mean", 0.0)),
        scale=float(noise_cfg.get("std", 0.0)),
        size=length,
    )

    # Fixed operation order: trend + seasonal first, then noise, then anomalies.
    values = (trend + seasonal) + noise
    for anomaly in spec.get("anomalies", []):
        pos = int(anomaly["index"])
        if 0 <= pos < length:
            values[pos] = values[pos] + float(anomaly["delta"])

    if not np.all(np.isfinite(values)):
        raise SeriesFailure(spec["series_id"], "generated values are not finite")

    timestamps = [
        (start_date + timedelta(days=int(i))).isoformat() for i in range(length)
    ]
    return SeriesData(spec["series_id"], timestamps, [float(v) for v in values])


def validate_series(series: SeriesData) -> None:
    """Strict timestamps/values validation; raises SeriesFailure on problems."""
    if len(series.timestamps) != len(series.values):
        raise SeriesFailure(
            series.series_id,
            "timestamp/value length mismatch",
        )
    if len(series.values) < 2:
        raise SeriesFailure(series.series_id, "series must contain >= 2 points")

    previous = None
    delta = None
    for pos, stamp in enumerate(series.timestamps):
        try:
            current = datetime.strptime(stamp, "%Y-%m-%d").date()
        except (ValueError, TypeError) as exc:
            raise SeriesFailure(
                series.series_id, f"invalid timestamp at index {pos}: {stamp!r}"
            ) from exc
        if previous is not None:
            gap = (current - previous).days
            if gap <= 0:
                raise SeriesFailure(
                    series.series_id,
                    f"timestamp not strictly increasing at index {pos}",
                )
            if delta is None:
                delta = gap
            elif gap != delta:
                raise SeriesFailure(
                    series.series_id,
                    f"timestamp interval break at index {pos} (gap {gap} != {delta})",
                )
        previous = current

    for pos, value in enumerate(series.values):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SeriesFailure(
                series.series_id, f"non-finite value at index {pos}"
            )


def fragment_id_of(index: int, block_size: int) -> str:
    return f"frag-{index // block_size:05d}"


def fragment_ids_for(indices: range | list[int], block_size: int) -> list[str]:
    blocks = sorted({index // block_size for index in indices})
    return [f"frag-{block:05d}" for block in blocks]


def apply_operation(
    series: SeriesData, operation: dict, block_size: int
) -> tuple[list[int], list[str]]:
    """Apply one update operation in place and return changed indices/fragments."""
    op_type = operation.get("type")
    values = operation.get("values")
    if not isinstance(values, list) or not values:
        raise SeriesFailure(
            series.series_id, f"operation {op_type!r} requires a non-empty values list"
        )
    for pos, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SeriesFailure(
                series.series_id,
                f"non-finite update value at offset {pos}",
            )

    if op_type == "append":
        start = len(series.values)
        series.values.extend(float(v) for v in values)
        base = datetime.strptime(series.timestamps[-1], "%Y-%m-%d").date()
        series.timestamps.extend(
            (base + timedelta(days=offset + 1)).isoformat()
            for offset in range(len(values))
        )
        changed = list(range(start, start + len(values)))
    elif op_type == "tail_correction":
        width = len(values)
        if width > len(series.values):
            raise SeriesFailure(
                series.series_id,
                f"tail_correction width {width} exceeds series length {len(series.values)}",
            )
        start = len(series.values) - width
        for offset, value in enumerate(values):
            series.values[start + offset] = float(value)
        changed = list(range(start, len(series.values)))
    elif op_type == "replace_range":
        start, end = operation.get("start"), operation.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise SeriesFailure(
                series.series_id, "replace_range needs integer start/end"
            )
        if start < 0 or end >= len(series.values) or start > end:
            raise SeriesFailure(
                series.series_id,
                f"replace_range [{start}, {end}] out of bounds "
                f"(length {len(series.values)})",
            )
        if end - start + 1 != len(values):
            raise SeriesFailure(
                series.series_id,
                "replace_range values length does not cover the interval",
            )
        for offset, value in enumerate(values):
            series.values[start + offset] = float(value)
        changed = list(range(start, end + 1))
    else:
        raise SeriesFailure(
            series.series_id, f"unknown update operation type {op_type!r}"
        )

    validate_series(series)
    fragments = fragment_ids_for(changed, block_size)
    return changed, fragments
