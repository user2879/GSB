from __future__ import annotations

import threading

from .errors import BudgetError
from .events import monotonic_ms


class ThroughputBudget:
    """Cumulative construction-byte budget.

    The budget is reserved before any deterministic bytes are generated.  It is
    a session total, not a refillable rate: an over-budget request is therefore
    terminal and is surfaced as E_BUDGET after its queue wait is recorded.
    """

    def __init__(self, limit_bytes: int, initial_used_bytes: int = 0) -> None:
        if limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive")
        if initial_used_bytes < 0 or initial_used_bytes > limit_bytes:
            raise ValueError("initial_used_bytes must be within [0, limit_bytes]")
        self.limit_bytes = limit_bytes
        self.used_bytes = initial_used_bytes
        self.reserved_bytes = 0
        self.peak_inflight_bytes = initial_used_bytes
        self._lock = threading.Lock()

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return max(0, self.limit_bytes - self.used_bytes)

    def reserve(self, amount: int) -> dict[str, int]:
        if amount <= 0:
            raise ValueError("reservation must be positive")
        wait_start = monotonic_ms()
        with self._lock:
            wait_end = monotonic_ms()
            if self.used_bytes + amount > self.limit_bytes:
                raise BudgetError(
                    "全局吞吐预算不足",
                    detail={
                        "limit_bytes": self.limit_bytes,
                        "used_bytes": self.used_bytes,
                        "requested_bytes": amount,
                        "remaining_bytes": max(0, self.limit_bytes - self.used_bytes),
                        "wait_start_monotonic_ms": wait_start,
                        "wait_end_monotonic_ms": wait_end,
                    },
                )
            self.used_bytes += amount
            self.reserved_bytes += amount
            self.peak_inflight_bytes = max(self.peak_inflight_bytes, self.used_bytes)
        return {
            "wait_start_monotonic_ms": wait_start,
            "wait_end_monotonic_ms": wait_end,
            "reserved_bytes": amount,
            "used_bytes_after": self.used_bytes,
        }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit_bytes": self.limit_bytes,
                "used_bytes": self.used_bytes,
                "remaining_bytes": max(0, self.limit_bytes - self.used_bytes),
                "reserved_bytes": self.reserved_bytes,
                "peak_inflight_bytes": self.peak_inflight_bytes,
            }
