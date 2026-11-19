from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


def mono_ms(clock_ns: Callable[[], int] = time.monotonic_ns) -> int:
    return clock_ns() // 1_000_000


@dataclass
class BreakerState:
    state: str = "closed"
    consecutive_failures: int = 0
    open_until_ms: int = 0
    half_open_claimed: bool = False


class CircuitBreakerRegistry:
    """Per-instance circuit breaker with monotonic-clock cooldown."""

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._states: dict[str, BreakerState] = {}
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        return mono_ms(self._clock_ns)

    def _state(self, instance_id: str) -> BreakerState:
        return self._states.setdefault(instance_id, BreakerState())

    def allow(self, instance_id: str) -> tuple[bool, str]:
        now = self.now_ms()
        with self._lock:
            state = self._state(instance_id)
            if state.state == "open":
                if now < state.open_until_ms:
                    return False, "circuit_open"
                if state.half_open_claimed:
                    return False, "circuit_probe_busy"
                state.half_open_claimed = True
                state.state = "half_open"
                return True, "circuit_probe"
            if state.state == "half_open":
                if state.half_open_claimed:
                    return False, "circuit_probe_busy"
                state.half_open_claimed = True
                return True, "circuit_probe"
            return True, "closed"

    def record_success(self, instance_id: str) -> None:
        with self._lock:
            self._states[instance_id] = BreakerState()

    def record_failure(
        self,
        instance_id: str,
        *,
        threshold: int | None,
        cooldown_ms: int,
    ) -> BreakerState:
        now = self.now_ms()
        with self._lock:
            state = self._state(instance_id)
            state.consecutive_failures += 1
            state.half_open_claimed = False
            if threshold is not None and state.consecutive_failures >= threshold:
                state.state = "open"
                state.open_until_ms = now + max(0, cooldown_ms)
            elif state.state == "half_open":
                state.state = "open"
                state.open_until_ms = now + max(0, cooldown_ms)
            return BreakerState(
                state=state.state,
                consecutive_failures=state.consecutive_failures,
                open_until_ms=state.open_until_ms,
                half_open_claimed=state.half_open_claimed,
            )

    def snapshot(self) -> tuple[dict[str, object], ...]:
        now = self.now_ms()
        with self._lock:
            rows = []
            for instance_id in sorted(self._states):
                state = self._states[instance_id]
                remaining = max(0, state.open_until_ms - now)
                rows.append(
                    {
                        "instance_id": instance_id,
                        "state": state.state,
                        "consecutive_failures": state.consecutive_failures,
                        "remaining_cooldown_ms": remaining,
                        "half_open_claimed": state.half_open_claimed,
                    }
                )
            return tuple(rows)

    def restore(self, rows: list[dict[str, object]] | tuple[dict[str, object], ...]) -> None:
        now = self.now_ms()
        with self._lock:
            self._states.clear()
            for row in sorted(rows, key=lambda item: str(item.get("instance_id", ""))):
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str):
                    continue
                state_name = row.get("state")
                if state_name not in {"closed", "open", "half_open"}:
                    state_name = "closed"
                failures = row.get("consecutive_failures", 0)
                if not isinstance(failures, int) or failures < 0:
                    failures = 0
                remaining = row.get("remaining_cooldown_ms", 0)
                if not isinstance(remaining, int) or remaining < 0:
                    remaining = 0
                self._states[instance_id] = BreakerState(
                    state=str(state_name),
                    consecutive_failures=failures,
                    open_until_ms=now + remaining if state_name == "open" else 0,
                    half_open_claimed=False,
                )

    def open_instances(self) -> tuple[dict[str, object], ...]:
        return tuple(
            row for row in self.snapshot() if row["state"] in {"open", "half_open"}
        )
