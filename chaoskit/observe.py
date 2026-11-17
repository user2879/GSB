"""Observation collection and integer fixed-point aggregation.

Clock basis is explicitly ``time.monotonic_ns`` truncated to whole
milliseconds (local monotonic count, never wall clock). The per-request
records persisted in the ledger are the source of truth; summaries below are
derived from those records and carry quantization error bounds.
"""
from __future__ import annotations

import socket
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

REQUEST_TIMEOUT_SECONDS = 5.0
BPS_DENOMINATOR = 10000


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class RequestRecord:
    timestamp_ms: int          # monotonic count at send
    phase: str                 # baseline | active | post
    success: bool
    latency_ms: int            # whole-ms client-side latency
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def call_once(host: str, port: int, phase: str) -> RequestRecord:
    start = mono_ms()
    try:
        with socket.create_connection((host, port), timeout=REQUEST_TIMEOUT_SECONDS) as conn:
            conn.settimeout(REQUEST_TIMEOUT_SECONDS)
            conn.sendall(b"REQUEST\n")
            chunks = []
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        latency = mono_ms() - start
        reply = b"".join(chunks).decode("utf-8").strip()
        success = reply.startswith("OK")
        return RequestRecord(start, phase, success, latency, None if success else reply)
    except OSError as exc:
        latency = mono_ms() - start
        return RequestRecord(start, phase, False, latency, type(exc).__name__)


def collect_phase(host: str, port: int, phase: str, sample_interval_ms: int,
                  sample_count: int) -> List[RequestRecord]:
    records: List[RequestRecord] = []
    for _ in range(max(1, sample_count)):
        records.append(call_once(host, port, phase))
        time.sleep(sample_interval_ms / 1000.0)
    return records


def nearest_rank_quantile(values: List[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot compute quantile of empty sample")
    ordered = sorted(values)
    rank = max(1, int(round(quantile * len(ordered))))
    return ordered[rank - 1]


@dataclass(frozen=True)
class ObservationSummary:
    phase: str
    sample_count: int
    success_count: int
    failure_count: int
    success_rate_bps: Optional[int]
    rate_quant_error_bps: int
    p50_latency_ms: Optional[int]
    p95_latency_ms: Optional[int]
    max_latency_ms: Optional[int]
    latency_quant_error_ms: int
    insufficient: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def aggregate(records: List[RequestRecord], phase: str, min_samples: int) -> ObservationSummary:
    sample_count = len(records)
    success_count = sum(1 for record in records if record.success)
    failure_count = sample_count - success_count
    insufficient = sample_count < min_samples

    if insufficient:
        # Never fabricate derived numbers: keep raw counts but leave metrics null.
        return ObservationSummary(
            phase=phase,
            sample_count=sample_count,
            success_count=success_count,
            failure_count=failure_count,
            success_rate_bps=None,
            rate_quant_error_bps=BPS_DENOMINATOR,
            p50_latency_ms=None,
            p95_latency_ms=None,
            max_latency_ms=None,
            latency_quant_error_ms=1,
            insufficient=True,
        )

    # floor-fixed-point rate; max rounding error in basis points is bounded by
    # ceil(BPS / n).
    rate = (success_count * BPS_DENOMINATOR) // sample_count
    rate_error = (BPS_DENOMINATOR + sample_count - 1) // sample_count
    latencies = [record.latency_ms for record in records]
    return ObservationSummary(
        phase=phase,
        sample_count=sample_count,
        success_count=success_count,
        failure_count=failure_count,
        success_rate_bps=rate,
        rate_quant_error_bps=rate_error,
        p50_latency_ms=nearest_rank_quantile(latencies, 0.50),
        p95_latency_ms=nearest_rank_quantile(latencies, 0.95),
        max_latency_ms=max(latencies),
        latency_quant_error_ms=1,
        insufficient=False,
    )
