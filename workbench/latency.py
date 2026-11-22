"""Monotonic-clock latency sampling and fixed-point statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from .errors import LatencyError
from .logging_utils import JsonlLogger

CLOCK_BASIS = "local.monotonic_ns//1_000_000"


@dataclass(frozen=True)
class LatencySample:
    seq: int
    kind: str
    service_id: str
    elapsed_ms: int
    budget_ms: int
    over_budget: bool

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "service_id": self.service_id,
            "elapsed_ms": self.elapsed_ms,
            "budget_ms": self.budget_ms,
            "over_budget": self.over_budget,
        }


@dataclass(frozen=True)
class FixedPointStats:
    count: int
    mean_ms_fp: int
    p95_ms: int
    max_ms: int
    fixed_point_scale: int
    quantization_error_bound_ms_fp: int

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_ms_fp": self.mean_ms_fp,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "fixed_point_scale": self.fixed_point_scale,
            "quantization_error_bound_ms_fp": self.quantization_error_bound_ms_fp,
        }


def percentile_nearest_rank(values: list[int], percentile: int) -> int:
    """Nearest-rank P95 with no interpolation; defined for non-empty input."""

    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


def fixed_point_stats(samples_ms: Iterable[int], scale: int) -> FixedPointStats:
    """Order-independent integer statistics.

    Each raw sample is an integer millisecond count. The mean is the only
    quantity that needs division, so its absolute error is bounded by one
    fixed-point unit; P95 (nearest rank) and max are exact integer samples.
    """

    values = [int(value) for value in samples_ms]
    if not values:
        return FixedPointStats(0, 0, 0, 0, scale, 1)
    total = sum(values)
    mean_fp = (total * scale) // len(values)
    return FixedPointStats(
        count=len(values),
        mean_ms_fp=mean_fp,
        p95_ms=percentile_nearest_rank(values, 95),
        max_ms=max(values),
        fixed_point_scale=scale,
        # |floor(S*s/n) - S*s/n| < 1 fixed-point unit (1/s millisecond).
        quantization_error_bound_ms_fp=1,
    )


class LatencyRecorder:
    """Records query/replay latency against the local monotonic clock."""

    def __init__(
        self,
        budget_ms: int,
        fixed_point_scale: int,
        logger: JsonlLogger | None = None,
        *,
        strict: bool = False,
        clock: Callable[[], int] | None = None,
        start_seq: int = 0,
    ) -> None:
        self.budget_ms = budget_ms
        self.scale = fixed_point_scale
        self.logger = logger
        self.strict = strict
        self._clock = clock or (lambda: time.monotonic_ns() // 1_000_000)
        self._seq = start_seq
        self.samples: list[LatencySample] = []

    def measure(self, kind: str, service_id: str = "ALL") -> tuple[int, int]:
        """Return ``(start_ms, end_ms)`` bracketing a caller's work."""

        start = self._clock()
        return start, self._clock()

    def commit(
        self,
        start_ms: int,
        end_ms: int,
        kind: str,
        service_id: str = "ALL",
    ) -> LatencySample:
        if end_ms < start_ms:
            raise ValueError("monotonic clock end precedes start")
        self._seq += 1
        elapsed = end_ms - start_ms
        over = elapsed > self.budget_ms
        sample = LatencySample(
            seq=self._seq,
            kind=kind,
            service_id=service_id,
            elapsed_ms=elapsed,
            budget_ms=self.budget_ms,
            over_budget=over,
        )
        self.samples.append(sample)
        if over:
            details = sample.as_dict()
            if self.logger:
                self.logger.error("latency_budget_exceeded", code="E_LATENCY", **details)
            if self.strict:
                raise LatencyError(
                    f"{kind} latency {elapsed}ms exceeded budget {self.budget_ms}ms",
                    details=details,
                )
        return sample

    def run_measured(self, kind: str, work: Callable[[], object], service_id: str = "ALL"):
        start, _ = self.measure(kind, service_id)
        result = work()
        self.commit(start, self._clock(), kind, service_id)
        return result

    def aggregate(self, kind: str | None = None, service_id: str | None = None) -> FixedPointStats:
        values = [
            sample.elapsed_ms
            for sample in self.samples
            if (kind is None or sample.kind == kind)
            and (service_id is None or sample.service_id == service_id)
        ]
        return fixed_point_stats(values, self.scale)

    def aggregate_by_service(self, kind: str | None = None) -> dict[str, FixedPointStats]:
        ids = sorted({sample.service_id for sample in self.samples})
        return {service_id: self.aggregate(kind, service_id) for service_id in ids}

    def over_budget_samples(self) -> list[LatencySample]:
        return [sample for sample in self.samples if sample.over_budget]
