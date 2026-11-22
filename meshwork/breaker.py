from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .model import BreakerConfig, InstanceSpec


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class Permit:
    instance_id: str
    generation: int
    trial: bool


@dataclass(frozen=True)
class Decision:
    allowed: bool
    state: BreakerState
    remaining_cooldown_ms: int
    reason: str | None = None
    permit: Permit | None = None


@dataclass
class Transition:
    at_mono_ms: int
    instance_id: str
    previous: str
    current: str
    reason: str


class CircuitBreaker:
    """Per-instance circuit breaker with a real monotonic cooldown clock."""

    def __init__(
        self,
        instance: InstanceSpec,
        config: BreakerConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.instance = instance
        self.config = config
        self._clock = clock
        self._lock = threading.Lock()
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.open_until_wall_ms = 0
        self.half_open_dispatched = 0
        self.half_open_successes = 0
        self.generation = 0
        self.transitions: list[Transition] = []

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _transition(self, current: BreakerState, reason: str) -> None:
        if current == self.state:
            return
        previous = self.state
        self.state = current
        self.transitions.append(
            Transition(self._now_ms(), self.instance.id, previous.value, current.value, reason)
        )

    def _cooldown_remaining_locked(self) -> int:
        if self.state != BreakerState.OPEN:
            return 0
        remaining = self.open_until - self._clock()
        return max(0, int(remaining * 1000 + 0.999999))

    def cooldown_remaining_ms(self) -> int:
        with self._lock:
            return self._cooldown_remaining_locked()

    def before_attempt(self) -> Decision:
        with self._lock:
            now = self._clock()
            if self.state == BreakerState.OPEN:
                remaining = self._cooldown_remaining_locked()
                if remaining > 0:
                    return Decision(
                        False,
                        self.state,
                        remaining,
                        reason="cooldown_active",
                    )
                self.generation += 1
                self.half_open_dispatched = 0
                self.half_open_successes = 0
                self._transition(BreakerState.HALF_OPEN, "cooldown_elapsed")
            if self.state == BreakerState.HALF_OPEN:
                if self.half_open_dispatched >= self.config.half_open_probes:
                    return Decision(
                        False,
                        self.state,
                        0,
                        reason="half_open_probes_exhausted",
                    )
                self.half_open_dispatched += 1
                return Decision(
                    True,
                    self.state,
                    0,
                    reason="half_open_probe",
                    permit=Permit(self.instance.id, self.generation, True),
                )
            return Decision(
                True,
                self.state,
                0,
                reason=None,
                permit=Permit(self.instance.id, self.generation, False),
            )

    def record_result(self, permit: Permit, success: bool) -> BreakerState:
        with self._lock:
            if permit.generation != self.generation:
                return self.state
            if permit.trial:
                if self.state != BreakerState.HALF_OPEN:
                    return self.state
                if success:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.config.half_open_probes:
                        self.consecutive_failures = 0
                        self.half_open_dispatched = 0
                        self.half_open_successes = 0
                        self.open_until = 0.0
                        self.open_until_wall_ms = 0
                        self._transition(BreakerState.CLOSED, "half_open_probe_success")
                else:
                    self._open_locked("half_open_probe_failure")
                return self.state
            if self.state != BreakerState.CLOSED:
                return self.state
            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.config.failure_threshold:
                    self._open_locked("failure_threshold_reached")
            return self.state

    def _open_locked(self, reason: str) -> None:
        self.generation += 1
        self.open_until = self._clock() + self.config.cooldown_ms / 1000.0
        self.open_until_wall_ms = int(time.time() * 1000) + self.config.cooldown_ms
        self.half_open_dispatched = 0
        self.half_open_successes = 0
        self._transition(BreakerState.OPEN, reason)

    def force_open(self, reason: str = "restored_open") -> None:
        with self._lock:
            self._open_locked(reason)

    def serialize(self) -> dict[str, object]:
        with self._lock:
            return {
                "instance_id": self.instance.id,
                "state": self.state.value,
                "consecutive_failures": self.consecutive_failures,
                "cooldown_remaining_ms": self._cooldown_remaining_locked(),
                "open_until_wall_ms": self.open_until_wall_ms,
                "half_open_dispatched": self.half_open_dispatched,
                "half_open_successes": self.half_open_successes,
                "generation": self.generation,
            }

    def restore(self, snapshot: dict[str, object], now_wall_ms: int) -> None:
        with self._lock:
            default_state = BreakerState.CLOSED
            try:
                state = BreakerState(str(snapshot.get("state", "CLOSED")))
            except ValueError:
                state = default_state
            self.state = state
            self.consecutive_failures = int(snapshot.get("consecutive_failures", 0))
            self.generation = int(snapshot.get("generation", 0))
            self.half_open_dispatched = int(snapshot.get("half_open_dispatched", 0))
            self.half_open_successes = int(snapshot.get("half_open_successes", 0))
            open_until_wall_ms = int(snapshot.get("open_until_wall_ms", 0))
            if self.state == BreakerState.OPEN:
                remaining_ms = max(0, open_until_wall_ms - now_wall_ms)
                self.open_until = self._clock() + remaining_ms / 1000.0
                self.open_until_wall_ms = now_wall_ms + remaining_ms
            else:
                self.open_until = 0.0
                self.open_until_wall_ms = 0

    def view(self) -> dict[str, object]:
        with self._lock:
            return {
                "instance_id": self.instance.id,
                "state": self.state.value,
                "consecutive_failures": self.consecutive_failures,
                "failure_threshold": self.config.failure_threshold,
                "cooldown_remaining_ms": self._cooldown_remaining_locked(),
                "half_open_dispatched": self.half_open_dispatched,
                "half_open_successes": self.half_open_successes,
                "half_open_probes": self.config.half_open_probes,
            }


class BreakerManager:
    def __init__(
        self,
        spec_instances: list[InstanceSpec],
        config: BreakerConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._breakers = {
            instance.id: CircuitBreaker(instance, config, clock)
            for instance in sorted(spec_instances, key=lambda item: (item.service_id, item.order, item.id))
        }

    def get(self, instance_id: str) -> CircuitBreaker:
        return self._breakers[instance_id]

    def ordered(self) -> list[CircuitBreaker]:
        return [self._breakers[key] for key in sorted(self._breakers)]

    def views(self) -> list[dict[str, object]]:
        return [breaker.view() for breaker in self.ordered()]

    def serialized(self) -> dict[str, dict[str, object]]:
        return {breaker.instance.id: breaker.serialize() for breaker in self.ordered()}

    def restore(self, snapshot: dict[str, object], now_wall_ms: int) -> None:
        for instance_id in sorted(self._breakers):
            raw = snapshot.get(instance_id)
            if isinstance(raw, dict):
                self._breakers[instance_id].restore(raw, now_wall_ms)

    def transitions(self) -> list[Transition]:
        result: list[Transition] = []
        for breaker in self.ordered():
            result.extend(breaker.transitions)
        return sorted(result, key=lambda item: (item.at_mono_ms, item.instance_id, item.current))

