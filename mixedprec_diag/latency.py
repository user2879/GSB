from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass
class QueryRecord:
    query_id: int
    operation: str
    batch_id: str | None
    start_mono_ns: int
    end_mono_ns: int
    duration_us: int
    exceeded_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "operation": self.operation,
            "batch_id": self.batch_id,
            "start_mono_ns": self.start_mono_ns,
            "end_mono_ns": self.end_mono_ns,
            "duration_us": self.duration_us,
            "duration_ms": self.duration_us / 1000.0,
            "exceeded_budget": self.exceeded_budget,
        }


class LatencyRecorder:
    """Records UI query latency using a monotonic nanosecond clock.

    Raw duration is quantized to integer microseconds (0.001 ms). That is the
    fixed-point representation used by all mean/P95/max aggregates. Individual
    raw records remain the truth source in the report.
    """

    clock_basis = "time.perf_counter_ns (monotonic)"
    truth_source = "ui_latency.queries"
    fixed_point_unit = "integer microsecond (0.001 ms)"

    def __init__(
        self,
        budget_ms: float,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.budget_ms = float(budget_ms)
        self.budget_us = int(round(self.budget_ms * 1000.0))
        self.clock_ns = clock_ns
        self._records: list[QueryRecord] = []
        self._next_id = 1
        self._lock = threading.Lock()

    def record(self, operation: str, batch_id: str | None, start_ns: int, end_ns: int) -> QueryRecord:
        if end_ns < start_ns:
            raise ValueError("end_ns must be >= start_ns")
        duration_us = _round_half_away_ns_to_us(end_ns - start_ns)
        with self._lock:
            record = QueryRecord(
                query_id=self._next_id,
                operation=operation,
                batch_id=batch_id,
                start_mono_ns=start_ns,
                end_mono_ns=end_ns,
                duration_us=duration_us,
                exceeded_budget=duration_us > self.budget_us,
            )
            self._next_id += 1
            self._records.append(record)
            return record

    @contextmanager
    def measure(self, operation: str, batch_id: str | None = None) -> Iterator[None]:
        start = self.clock_ns()
        try:
            yield
        finally:
            end = self.clock_ns()
            self.record(operation, batch_id, start, end)

    def records(self) -> list[QueryRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._next_id = 1

    def exceeded(self) -> list[QueryRecord]:
        return [record for record in self.records() if record.exceeded_budget]

    def summary(self) -> dict[str, Any]:
        records = self.records()
        codes = ["E_LATENCY"] if any(record.exceeded_budget for record in records) else []
        return {
            "clock_basis": self.clock_basis,
            "truth_source": self.truth_source,
            "fixed_point": {
                "unit": self.fixed_point_unit,
                "scale": 1000,
                "duration_formula": "round_half_away((end_mono_ns-start_mono_ns)/1000)",
                "raw_quantization_error_upper_ms": 0.0005,
            },
            "budget_ms": self.budget_ms,
            "budget_us": self.budget_us,
            "query_count": len(records),
            "queries": [record.to_dict() for record in records],
            "aggregates": self._aggregates(records),
            "codes": codes,
            "exceeded_query_ids": [record.query_id for record in records if record.exceeded_budget],
        }

    def _aggregates(self, records: list[QueryRecord]) -> dict[str, Any]:
        groups: dict[str, list[QueryRecord]] = {"__all__": records}
        for record in records:
            if record.batch_id is not None:
                groups.setdefault(record.batch_id, []).append(record)
        result: dict[str, Any] = {}
        for key in sorted(groups):
            group = groups[key]
            result[key] = _aggregate_group(group)
        return result


def _aggregate_group(records: list[QueryRecord]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "mean_us": None,
            "p95_us": None,
            "max_us": None,
            "mean_quantization_error_upper_ms": 0.001,
            "p95_quantization_error_upper_ms": 0.0005,
            "max_quantization_error_upper_ms": 0.0005,
        }
    durations = sorted(record.duration_us for record in records)
    mean_us = int(round(sum(durations) / len(durations)))
    rank = max(1, math.ceil(0.95 * len(durations)))
    p95_us = durations[rank - 1]
    max_us = durations[-1]
    return {
        "count": len(records),
        "mean_us": mean_us,
        "p95_us": p95_us,
        "max_us": max_us,
        "mean_ms": mean_us / 1000.0,
        "p95_ms": p95_us / 1000.0,
        "max_ms": max_us / 1000.0,
        "p95_rank_nearest": rank,
        "mean_quantization_error_upper_ms": 0.001,
        "p95_quantization_error_upper_ms": 0.0005,
        "max_quantization_error_upper_ms": 0.0005,
        "exceeded_count": sum(1 for record in records if record.exceeded_budget),
    }


def _round_half_away_ns_to_us(duration_ns: int) -> int:
    quotient, remainder = divmod(duration_ns, 1000)
    if remainder >= 500:
        quotient += 1
    return int(quotient)
