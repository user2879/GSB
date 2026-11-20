from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .rules import InstanceRule
from .util import mono_ms, utc_now_iso


CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class CircuitRuntime:
    instance: str
    service: str
    host: str
    port: int
    state: str = CLOSED
    consecutive_failures: int = 0
    failure_threshold: int = 1
    cooldown_ms: int = 1
    opened_at_mono_ms: float | None = None
    cooldown_until_mono_ms: float | None = None
    probe_in_flight: bool = False
    updated_at_utc: str = field(default_factory=utc_now_iso)

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class Admission:
    allowed: bool
    state: str
    probe: bool
    reason: str
    checked_at_mono_ms: float


class CircuitManager:
    def __init__(
        self,
        instances: Iterable[InstanceRule],
        *,
        on_state: Callable[[CircuitRuntime], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        restored_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._states: dict[str, CircuitRuntime] = {}
        self._on_state = on_state
        self._on_event = on_event
        for instance in instances:
            state = CircuitRuntime(
                instance=instance.name,
                service=instance.service,
                host=instance.host,
                port=instance.port,
                failure_threshold=instance.failure_threshold,
                cooldown_ms=instance.cooldown_ms,
            )
            restored = (restored_states or {}).get(state.key)
            if restored:
                for field_name in (
                    "state",
                    "consecutive_failures",
                    "opened_at_mono_ms",
                    "cooldown_until_mono_ms",
                    "probe_in_flight",
                ):
                    if field_name in restored and restored[field_name] is not None:
                        setattr(state, field_name, restored[field_name])
                if state.state not in {CLOSED, OPEN, HALF_OPEN}:
                    state.state = CLOSED
                # A half-open probe belongs to a dead process; make it admissible
                # again only after the cooldown has actually elapsed.
                if state.state == HALF_OPEN:
                    state.probe_in_flight = False
            self._states[state.key] = state

    def states(self) -> tuple[CircuitRuntime, ...]:
        with self._lock:
            return tuple(
                self._states[key]
                for key in sorted(self._states, key=lambda item: self._states[item].port)
            )

    def _persist(self, state: CircuitRuntime) -> None:
        state.updated_at_utc = utc_now_iso()
        if self._on_state:
            self._on_state(state)

    def _event(self, state: CircuitRuntime, event: str, reason: str) -> None:
        if self._on_event:
            self._on_event(
                {
                    "instance": state.instance,
                    "service": state.service,
                    "host": state.host,
                    "port": state.port,
                    "event": event,
                    "reason": reason,
                    "state": state.state,
                    "opened_at_mono_ms": state.opened_at_mono_ms,
                    "cooldown_until_mono_ms": state.cooldown_until_mono_ms,
                    "created_at_utc": utc_now_iso(),
                }
            )

    def admit(self, instance: InstanceRule, now_ms: float | None = None) -> Admission:
        now = mono_ms() if now_ms is None else now_ms
        with self._lock:
            state = self._states[instance.key]
            if state.state == CLOSED:
                return Admission(True, CLOSED, False, "closed", now)
            if state.state == OPEN:
                if state.cooldown_until_mono_ms is None or now < state.cooldown_until_mono_ms:
                    return Admission(
                        False,
                        OPEN,
                        False,
                        "cooldown window is active",
                        now,
                    )
                state.state = HALF_OPEN
                state.probe_in_flight = False
                self._persist(state)
                self._event(state, "half_open", "cooldown elapsed; one probe admitted")
            if state.state == HALF_OPEN:
                if state.probe_in_flight:
                    return Admission(
                        False,
                        HALF_OPEN,
                        False,
                        "half-open probe already in flight",
                        now,
                    )
                state.probe_in_flight = True
                self._persist(state)
                return Admission(True, HALF_OPEN, True, "half-open probe", now)
            return Admission(False, state.state, False, "unknown circuit state", now)

    def record_success(
        self,
        instance: InstanceRule,
        now_ms: float | None = None,
        *,
        probe: bool = False,
    ) -> None:
        del now_ms
        with self._lock:
            state = self._states[instance.key]
            if state.state == HALF_OPEN:
                if not probe:
                    return
                state.state = CLOSED
                state.consecutive_failures = 0
                state.opened_at_mono_ms = None
                state.cooldown_until_mono_ms = None
                state.probe_in_flight = False
                self._persist(state)
                self._event(state, "closed", "half-open probe succeeded")
            elif state.state == CLOSED:
                state.consecutive_failures = 0
                self._persist(state)
            # An in-flight attempt admitted before OPEN may finish successfully.
            # It must not bypass or shorten the real cooldown window.

    def record_failure(
        self,
        instance: InstanceRule,
        now_ms: float | None = None,
        *,
        probe: bool = False,
    ) -> None:
        now = mono_ms() if now_ms is None else now_ms
        with self._lock:
            state = self._states[instance.key]
            if state.state == HALF_OPEN:
                if not probe:
                    return
                state.probe_in_flight = False
                state.state = OPEN
                state.opened_at_mono_ms = now
                state.cooldown_until_mono_ms = now + state.cooldown_ms
                self._persist(state)
                self._event(state, "reopened", "half-open probe failed")
                return
            if state.state == OPEN:
                # This is a late non-probe outcome from an attempt admitted
                # before the breaker opened. It must not extend or shorten the
                # active cooldown interval.
                return
            state.consecutive_failures += 1
            if state.state == CLOSED and state.consecutive_failures >= state.failure_threshold:
                state.state = OPEN
                state.opened_at_mono_ms = now
                state.cooldown_until_mono_ms = now + state.cooldown_ms
                self._persist(state)
                self._event(
                    state,
                    "opened",
                    f"consecutive failures reached threshold {state.failure_threshold}",
                )
            else:
                self._persist(state)

    def as_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for state in self.states():
                row = asdict(state)
                row["key"] = state.key
                rows.append(row)
            return rows
