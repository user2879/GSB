"""Integer fixed-point latency samples and exact nearest-rank aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


QUANTIZATION_ERROR_BOUND_US_HALVES = 3  # 1.5 us expressed as halves


@dataclass(frozen=True)
class LatencySample:
    kind: str
    node_id: str | None
    label: str
    duration_us: int
    monotonic_start_ms: int
    monotonic_end_ms: int
    over_budget: bool = False
    overrun_us: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "node_id": self.node_id,
            "label": self.label,
            "duration_us": self.duration_us,
            "monotonic_start_ms": self.monotonic_start_ms,
            "monotonic_end_ms": self.monotonic_end_ms,
            "over_budget": self.over_budget,
            "overrun_us": self.overrun_us,
        }


def aggregate_stats(durations_us: Iterable[int]) -> dict[str, int | str]:
    samples = sorted(max(0, int(value)) for value in durations_us)
    count = len(samples)
    if count == 0:
        return {
            "sample_count": 0,
            "mean_us": 0,
            "p95_us": 0,
            "max_us": 0,
            "mean_fixed6_ms": 0,
            "p95_fixed6_ms": 0,
            "max_fixed6_ms": 0,
            "quantization_error_bound_us": "1.5",
            "quantization_error_bound_fixed6_ms": 1500,
            "quantization_error_bound_us_halves": QUANTIZATION_ERROR_BOUND_US_HALVES,
            "p95_method": "nearest-rank: ceil(0.95*n)-1",
        }
    total = sum(samples)
    # Round half up without floating point.
    mean_us = (2 * total + count) // (2 * count)
    p95_rank = (95 * count + 99) // 100
    p95_index = min(count - 1, max(0, p95_rank - 1))
    p95_us = samples[p95_index]
    max_us = samples[-1]
    return {
        "sample_count": count,
        "mean_us": mean_us,
        "p95_us": p95_us,
        "max_us": max_us,
        "mean_fixed6_ms": mean_us * 1000,
        "p95_fixed6_ms": p95_us * 1000,
        "max_fixed6_ms": max_us * 1000,
        "quantization_error_bound_us": "1.5",
        "quantization_error_bound_fixed6_ms": 1500,
        "quantization_error_bound_us_halves": QUANTIZATION_ERROR_BOUND_US_HALVES,
        "p95_method": "nearest-rank: ceil(0.95*n)-1",
    }


def build_fetch_aggregates(
    samples: Iterable[LatencySample], node_ids: Iterable[str]
) -> list[dict[str, Any]]:
    sample_list = [sample for sample in samples if sample.kind == "fetch"]
    aggregates: list[dict[str, Any]] = []
    overall = aggregate_stats(sample.duration_us for sample in sample_list)
    aggregates.append(
        {
            "scope": "overall",
            "node_id": None,
            "true_source": "latency_samples rows in the SQLite ledger",
            **overall,
        }
    )
    for node_id in sorted(set(node_ids)):
        stats = aggregate_stats(
            sample.duration_us for sample in sample_list if sample.node_id == node_id
        )
        aggregates.append(
            {
                "scope": "node",
                "node_id": node_id,
                "true_source": "latency_samples rows in the SQLite ledger",
                **stats,
            }
        )
    return aggregates


def evaluate_budget(duration_us: int, budget_ms: int) -> tuple[bool, int]:
    budget_us = max(0, budget_ms) * 1000
    overrun_us = max(0, duration_us - budget_us)
    return overrun_us > 0, overrun_us
