"""Four-dimensional admission and token-bucket bandwidth scheduling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from .config import GlobalLimits, Route

TOKEN_SCALE = 1_000_000
NS_PER_SECOND = 1_000_000_000

QueueObserver = Callable[[str, int], None]
SleepFn = Callable[[float], None]
ClockNs = Callable[[], int]


def _now_ms(now_ns: int) -> int:
    return now_ns // 1_000_000


class FixedPointTokenBucket:
    """A token bucket whose accounting is integer-only.

    Token balances are stored in millionths of a byte.  Capacity is one second
    of the configured rate, so a fresh bucket permits at most a one-second
    burst while the long-run rate remains exact.
    """

    def __init__(self, rate_bytes_per_second: int, *, initial_units: int | None = None) -> None:
        if isinstance(rate_bytes_per_second, bool) or rate_bytes_per_second <= 0:
            raise ValueError("rate must be a positive integer number of bytes/second")
        self.rate_bytes_per_second = int(rate_bytes_per_second)
        self.capacity_units = self.rate_bytes_per_second * TOKEN_SCALE
        self.units = self.capacity_units if initial_units is None else int(initial_units)
        if not (0 <= self.units <= self.capacity_units):
            raise ValueError("initial_units outside capacity")
        self.last_ns: int | None = None

    def refill(self, now_ns: int) -> None:
        if self.last_ns is None:
            self.last_ns = now_ns
            return
        if now_ns <= self.last_ns:
            self.last_ns = min(self.last_ns, now_ns)
            return
        elapsed = now_ns - self.last_ns
        replenished = (
            elapsed * self.rate_bytes_per_second * TOKEN_SCALE // NS_PER_SECOND
        )
        if replenished:
            self.units = min(self.capacity_units, self.units + replenished)
            self.last_ns = now_ns
        # Keep last_ns unchanged when no whole sub-token was earned; this avoids
        # losing sub-token remainder on short waits.

    def wait_ns_for(self, required_units: int) -> int:
        if required_units <= self.units:
            return 0
        missing = required_units - self.units
        return (
            missing * NS_PER_SECOND + self.rate_bytes_per_second * TOKEN_SCALE - 1
        ) // (self.rate_bytes_per_second * TOKEN_SCALE)

    def state(self) -> dict[str, int]:
        return {
            "rate_bytes_per_second": self.rate_bytes_per_second,
            "capacity_units": self.capacity_units,
            "available_units": self.units,
            "available_bytes_floor": self.units // TOKEN_SCALE,
        }


@dataclass(frozen=True)
class AdmissionSnapshot:
    global_active: int
    global_max: int
    global_bandwidth_kbps: int
    global_used_bytes: int
    global_waiting: int
    global_token_units: int
    global_available_bytes_floor: int
    global_rate_bytes_per_second: int
    routes: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "global": {
                "active_sessions": self.global_active,
                "max_sessions": self.global_max,
                "bandwidth_kbps": self.global_bandwidth_kbps,
                "used_bytes": self.global_used_bytes,
                "waiting": self.global_waiting,
                "token_units": self.global_token_units,
                "available_bytes_floor": self.global_available_bytes_floor,
                "rate_bytes_per_second": self.global_rate_bytes_per_second,
            },
            "routes": self.routes,
        }


class BudgetManager:
    """Coordinates concurrency slots and two-level bandwidth atomically."""

    def __init__(
        self,
        limits: GlobalLimits,
        routes: tuple[Route, ...],
        *,
        clock_ns: ClockNs = time.monotonic_ns,
        sleep_fn: SleepFn = time.sleep,
        usage: dict[str, object] | None = None,
    ) -> None:
        self.limits = limits
        self.routes = {route.id: route for route in routes}
        self.clock_ns = clock_ns
        self.sleep_fn = sleep_fn
        self._lock = threading.Lock()
        self._global_active = 0
        self._route_active = {route.id: 0 for route in routes}
        self._global_waiting = 0
        self._route_waiting = {route.id: 0 for route in routes}
        self._global_used_bytes = 0
        self._route_used_bytes = {route.id: 0 for route in routes}

        global_units = None
        route_units: dict[str, int] = {}
        if usage:
            raw_global = usage.get("global", {})
            if isinstance(raw_global, dict):
                self._global_used_bytes = int(raw_global.get("used_bytes", 0))
                if raw_global.get("token_units") is not None:
                    global_units = int(raw_global["token_units"])
            raw_routes = usage.get("routes", {})
            if isinstance(raw_routes, dict):
                for route_id, state in raw_routes.items():
                    if route_id in self.routes and isinstance(state, dict):
                        self._route_used_bytes[route_id] = int(
                            state.get("used_bytes", 0)
                        )
                        if state.get("token_units") is not None:
                            route_units[route_id] = int(state["token_units"])

        self._global_bucket = FixedPointTokenBucket(
            limits.bandwidth_kbps * 1024, initial_units=global_units
        )
        self._route_buckets = {
            route.id: FixedPointTokenBucket(
                route.bandwidth_kbps * 1024,
                initial_units=route_units.get(route.id),
            )
            for route in routes
        }

    def _route(self, route_id: str) -> Route:
        try:
            return self.routes[route_id]
        except KeyError as exc:
            raise KeyError(f"unknown route: {route_id}") from exc

    @staticmethod
    def _notify(observer: QueueObserver | None, event: str, now_ns: int) -> None:
        if observer is not None:
            observer(event, _now_ms(now_ns))

    def acquire_session(
        self,
        route_id: str,
        *,
        observer: QueueObserver | None = None,
    ) -> None:
        route = self._route(route_id)
        started_wait = False
        try:
            while True:
                with self._lock:
                    global_ok = self._global_active < self.limits.max_sessions
                    route_ok = self._route_active[route_id] < route.max_sessions
                    if global_ok and route_ok:
                        self._global_active += 1
                        self._route_active[route_id] += 1
                        if started_wait:
                            self._notify(observer, "end", self.clock_ns())
                        return
                    self._global_waiting += 1
                    self._route_waiting[route_id] += 1
                if not started_wait:
                    started_wait = True
                    self._notify(observer, "start", self.clock_ns())
                self.sleep_fn(0.002)
                with self._lock:
                    self._global_waiting -= 1
                    self._route_waiting[route_id] -= 1
        except BaseException:
            if started_wait:
                self._notify(observer, "end", self.clock_ns())
            raise

    def release_session(self, route_id: str) -> None:
        self._route(route_id)
        with self._lock:
            if self._global_active <= 0 or self._route_active[route_id] <= 0:
                raise RuntimeError("session release without matching acquire")
            self._global_active -= 1
            self._route_active[route_id] -= 1

    def acquire_bandwidth(
        self,
        route_id: str,
        byte_count: int,
        *,
        observer: QueueObserver | None = None,
    ) -> None:
        """Atomically consume from both buckets or wait without reservations."""

        self._route(route_id)
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        required = byte_count * TOKEN_SCALE
        if required > self._global_bucket.capacity_units:
            raise ValueError("byte_count exceeds global token bucket capacity")
        if required > self._route_buckets[route_id].capacity_units:
            raise ValueError("byte_count exceeds route token bucket capacity")
        started_wait = False
        try:
            while True:
                now = self.clock_ns()
                with self._lock:
                    self._global_bucket.refill(now)
                    route_bucket = self._route_buckets[route_id]
                    route_bucket.refill(now)
                    if (
                        self._global_bucket.units >= required
                        and route_bucket.units >= required
                    ):
                        self._global_bucket.units -= required
                        route_bucket.units -= required
                        self._global_used_bytes += byte_count
                        self._route_used_bytes[route_id] += byte_count
                        if started_wait:
                            self._notify(observer, "end", now)
                        return
                    wait_ns = max(
                        self._global_bucket.wait_ns_for(required),
                        route_bucket.wait_ns_for(required),
                    )
                if not started_wait:
                    started_wait = True
                    self._notify(observer, "start", now)
                self.sleep_fn(max(0.001, wait_ns / NS_PER_SECOND))
        except BaseException:
            if started_wait:
                self._notify(observer, "end", self.clock_ns())
            raise

    def max_bandwidth_chunk(self, route_id: str) -> int:
        route = self._route(route_id)
        return max(
            1,
            min(
                self.limits.bandwidth_kbps * 1024,
                route.bandwidth_kbps * 1024,
            ),
        )

    def snapshot(self) -> AdmissionSnapshot:
        now = self.clock_ns()
        with self._lock:
            self._global_bucket.refill(now)
            for bucket in self._route_buckets.values():
                bucket.refill(now)
            route_state: dict[str, dict[str, int]] = {}
            for route_id, route in self.routes.items():
                bucket = self._route_buckets[route_id]
                route_state[route_id] = {
                    "active_sessions": self._route_active[route_id],
                    "max_sessions": route.max_sessions,
                    "waiting": self._route_waiting[route_id],
                    "bandwidth_kbps": route.bandwidth_kbps,
                    "used_bytes": self._route_used_bytes[route_id],
                    **bucket.state(),
                }
            return AdmissionSnapshot(
                global_active=self._global_active,
                global_max=self.limits.max_sessions,
                global_bandwidth_kbps=self.limits.bandwidth_kbps,
                global_used_bytes=self._global_used_bytes,
                global_waiting=self._global_waiting,
                global_token_units=self._global_bucket.units,
                global_available_bytes_floor=self._global_bucket.units // TOKEN_SCALE,
                global_rate_bytes_per_second=self._global_bucket.rate_bytes_per_second,
                routes=route_state,
            )

    def budget_usage(self) -> dict[str, object]:
        now = self.clock_ns()
        with self._lock:
            self._global_bucket.refill(now)
            for bucket in self._route_buckets.values():
                bucket.refill(now)
            return {
                "global": {
                    "used_bytes": self._global_used_bytes,
                    "token_units": self._global_bucket.units,
                },
                "routes": {
                    route_id: {
                        "used_bytes": self._route_used_bytes[route_id],
                        "token_units": self._route_buckets[route_id].units,
                    }
                    for route_id in sorted(self.routes)
                },
            }


class ManualClock:
    """Virtual monotonic clock for deterministic scheduler tests."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now = int(start_ns)
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        with self._lock:
            return self._now

    def advance_ns(self, delta: int) -> None:
        with self._lock:
            self._now += int(delta)

    def advance_ms(self, delta: int) -> None:
        self.advance_ns(int(delta) * 1_000_000)

    def sleep(self, seconds: float) -> None:
        self.advance_ns(max(1, int(seconds * NS_PER_SECOND)))
