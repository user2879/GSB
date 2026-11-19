"""Deterministic error-budget and circuit-breaker state machine."""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .clock import MonotonicClock
from .models import (
    ChainSpec,
    CircuitState,
    Outcome,
    ServiceSpec,
    TraceCall,
    canonical_json,
    sha256_json,
)

STATE_SCHEMA = "srebudget.state.v1"


@dataclass
class WindowEvent:
    timestamp_ms: int
    failed: bool


@dataclass
class ServiceRuntime:
    sample_count: int = 0
    failure_count: int = 0
    budget_total: int = 0
    raw_remaining: int = 0
    budget_remaining: int = 0
    budget_exhausted: bool = False
    consecutive_failures: int = 0
    last_failure_timestamp_ms: int | None = None
    circuit_state: CircuitState = CircuitState.CLOSED
    opened_at_ms: int | None = None
    open_count: int = 0
    half_open_probes_used: int = 0
    events: deque[WindowEvent] = field(default_factory=deque)


@dataclass(frozen=True)
class ServiceResult:
    service: str
    status: str
    actual_outcome: str | None
    consumed: bool
    fast_failed: bool
    trigger_source: str | None
    trigger_reason: str | None
    propagation_depth: int
    budget: dict[str, int | bool]
    circuit: dict[str, int | str | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "status": self.status,
            "actual_outcome": self.actual_outcome,
            "consumed": self.consumed,
            "fast_failed": self.fast_failed,
            "trigger_source": self.trigger_source,
            "trigger_reason": self.trigger_reason,
            "propagation_depth": self.propagation_depth,
            "budget": self.budget,
            "circuit": self.circuit,
        }


@dataclass(frozen=True)
class CallDecision:
    call_index: int
    timestamp_ms: int
    path: tuple[str, ...]
    outcome: str
    duration_ms: int
    trigger_source: str | None
    trigger_reason: str | None
    fast_failed: bool
    fast_fail_chain: tuple[dict[str, Any], ...]
    services: tuple[ServiceResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "timestamp_ms": self.timestamp_ms,
            "path": list(self.path),
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "trigger_source": self.trigger_source,
            "trigger_reason": self.trigger_reason,
            "fast_failed": self.fast_failed,
            "fast_fail_chain": list(self.fast_fail_chain),
            "services": [item.as_dict() for item in self.services],
        }


def _failed(outcome: Outcome) -> bool:
    return outcome is not Outcome.SUCCESS


class DiagnosticEngine:
    """Advance a chain over calls while preserving one shared timeline."""

    def __init__(self, chain: ChainSpec, clock: Any | None = None) -> None:
        self.chain = chain
        self.clock = clock if clock is not None else MonotonicClock()
        self.runtimes: dict[str, ServiceRuntime] = {
            service.id: ServiceRuntime() for service in chain.services
        }
        self.cursor = 0
        self.last_trace_timestamp_ms: int | None = None
        self.last_logical_clock_ms: int = 0
        self._events: list[dict[str, Any]] = []
        self._next_event_index = 1

    # ------------------------------------------------------------------
    # Budget and circuit primitives
    # ------------------------------------------------------------------
    @staticmethod
    def _cutoff(timestamp_ms: int, service: ServiceSpec) -> int:
        return timestamp_ms - service.budget.window_ms

    def _refresh_budget(
        self,
        service: ServiceSpec,
        runtime: ServiceRuntime,
        timestamp_ms: int,
        call_index: int,
    ) -> None:
        cutoff = self._cutoff(timestamp_ms, service)
        while runtime.events and runtime.events[0].timestamp_ms < cutoff:
            event = runtime.events.popleft()
            runtime.sample_count -= 1
            if event.failed:
                runtime.failure_count -= 1

        runtime.budget_total = (
            runtime.sample_count * service.budget.ratio_bp // 10_000
        )
        runtime.raw_remaining = runtime.budget_total - runtime.failure_count
        runtime.budget_remaining = max(0, runtime.raw_remaining)
        previous_exhausted = runtime.budget_exhausted
        runtime.budget_exhausted = (
            runtime.sample_count >= service.budget.min_samples
            and runtime.raw_remaining < 0
        )
        if runtime.budget_exhausted and not previous_exhausted:
            self._record_event(
                call_index=call_index,
                timestamp_ms=timestamp_ms,
                service=service.id,
                kind="budget_exhausted",
                reason="error_budget_negative",
                consecutive_failures=runtime.consecutive_failures,
                cooldown_remaining_ms=self._cooldown_remaining(service, runtime),
                extra={
                    "sample_count": runtime.sample_count,
                    "failure_count": runtime.failure_count,
                    "budget_total": runtime.budget_total,
                    "raw_remaining": runtime.raw_remaining,
                },
            )
        elif previous_exhausted and not runtime.budget_exhausted:
            self._record_event(
                call_index=call_index,
                timestamp_ms=timestamp_ms,
                service=service.id,
                kind="budget_recovered",
                reason="window_or_min_samples_recovered",
                consecutive_failures=runtime.consecutive_failures,
                cooldown_remaining_ms=self._cooldown_remaining(service, runtime),
                extra={
                    "sample_count": runtime.sample_count,
                    "failure_count": runtime.failure_count,
                    "budget_total": runtime.budget_total,
                    "raw_remaining": runtime.raw_remaining,
                },
            )

    def _cooldown_remaining(
        self, service: ServiceSpec, runtime: ServiceRuntime
    ) -> int:
        if runtime.circuit_state is not CircuitState.OPEN:
            return 0
        if runtime.opened_at_ms is None:
            return service.circuit.cooldown_ms
        elapsed = max(0, self.last_logical_clock_ms - runtime.opened_at_ms)
        return max(0, service.circuit.cooldown_ms - elapsed)

    def _record_event(
        self,
        *,
        call_index: int,
        timestamp_ms: int,
        service: str,
        kind: str,
        reason: str,
        consecutive_failures: int,
        cooldown_remaining_ms: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "event_index": self._next_event_index,
            "call_index": call_index,
            "timestamp_ms": timestamp_ms,
            "service": service,
            "kind": kind,
            "reason": reason,
            "consecutive_failures": consecutive_failures,
            "cooldown_remaining_ms": cooldown_remaining_ms,
        }
        if extra:
            event.update(extra)
        self._events.append(event)
        self._next_event_index += 1

    def _transition_circuit(
        self,
        service: ServiceSpec,
        runtime: ServiceRuntime,
        target: CircuitState,
        *,
        call_index: int,
        timestamp_ms: int,
        reason: str,
        now_ms: int,
    ) -> None:
        previous = runtime.circuit_state
        if previous is target:
            return
        runtime.circuit_state = target
        if target is CircuitState.OPEN:
            runtime.opened_at_ms = now_ms
            runtime.open_count += 1
            runtime.half_open_probes_used = 0
        elif target is CircuitState.HALF_OPEN:
            runtime.half_open_probes_used = 0
        elif target is CircuitState.CLOSED:
            runtime.consecutive_failures = 0
            runtime.opened_at_ms = None
            runtime.half_open_probes_used = 0
        self._record_event(
            call_index=call_index,
            timestamp_ms=timestamp_ms,
            service=service.id,
            kind="circuit_transition",
            reason=reason,
            consecutive_failures=runtime.consecutive_failures,
            cooldown_remaining_ms=self._cooldown_remaining(service, runtime),
            extra={"from": previous.value, "to": target.value},
        )

    def _tick(self, call_index: int, timestamp_ms: int, now_ms: int) -> None:
        self.last_logical_clock_ms = max(self.last_logical_clock_ms, now_ms)
        for service in sorted(self.chain.services, key=lambda item: item.id):
            runtime = self.runtimes[service.id]
            self._refresh_budget(service, runtime, timestamp_ms, call_index)
            if runtime.circuit_state is CircuitState.OPEN:
                if runtime.opened_at_ms is None:
                    runtime.opened_at_ms = now_ms
                if now_ms - runtime.opened_at_ms >= service.circuit.cooldown_ms:
                    self._transition_circuit(
                        service,
                        runtime,
                        CircuitState.HALF_OPEN,
                        call_index=call_index,
                        timestamp_ms=timestamp_ms,
                        reason="cooldown_elapsed",
                        now_ms=now_ms,
                    )

    def _gate(
        self,
        service: ServiceSpec,
        runtime: ServiceRuntime,
        *,
        call_index: int,
        timestamp_ms: int,
        now_ms: int,
    ) -> str | None:
        self._refresh_budget(service, runtime, timestamp_ms, call_index)
        if runtime.circuit_state is CircuitState.OPEN:
            if runtime.opened_at_ms is None:
                runtime.opened_at_ms = now_ms
            if now_ms - runtime.opened_at_ms >= service.circuit.cooldown_ms:
                self._transition_circuit(
                    service,
                    runtime,
                    CircuitState.HALF_OPEN,
                    call_index=call_index,
                    timestamp_ms=timestamp_ms,
                    reason="cooldown_elapsed",
                    now_ms=now_ms,
                )
            else:
                return "circuit_open"

        if runtime.circuit_state is CircuitState.HALF_OPEN:
            if runtime.half_open_probes_used >= service.circuit.half_open_probes:
                return "circuit_open"
            if runtime.budget_exhausted:
                return "budget_exhausted"
            runtime.half_open_probes_used += 1
            return None

        if runtime.budget_exhausted:
            return "budget_exhausted"
        return None

    def _consume(
        self,
        service: ServiceSpec,
        runtime: ServiceRuntime,
        outcome: Outcome,
        *,
        call_index: int,
        timestamp_ms: int,
        now_ms: int,
    ) -> str | None:
        """Consume one actual outcome and return a post-consumption trigger."""
        failed = _failed(outcome)
        self._refresh_budget(service, runtime, timestamp_ms, call_index)
        runtime.events.append(WindowEvent(timestamp_ms, failed))
        runtime.sample_count += 1
        if failed:
            runtime.failure_count += 1
            runtime.last_failure_timestamp_ms = timestamp_ms
        budget_before = runtime.budget_exhausted
        self._refresh_budget(service, runtime, timestamp_ms, call_index)

        circuit_transitioned_open = False
        if runtime.circuit_state is CircuitState.CLOSED:
            if failed:
                runtime.consecutive_failures += 1
                if runtime.consecutive_failures >= service.circuit.failure_threshold:
                    self._transition_circuit(
                        service,
                        runtime,
                        CircuitState.OPEN,
                        call_index=call_index,
                        timestamp_ms=timestamp_ms,
                        reason="failure_threshold_reached",
                        now_ms=now_ms,
                    )
                    circuit_transitioned_open = True
            else:
                runtime.consecutive_failures = 0
        elif runtime.circuit_state is CircuitState.HALF_OPEN:
            if failed:
                runtime.consecutive_failures += 1
                self._transition_circuit(
                    service,
                    runtime,
                    CircuitState.OPEN,
                    call_index=call_index,
                    timestamp_ms=timestamp_ms,
                    reason="half_open_probe_failed",
                    now_ms=now_ms,
                )
                circuit_transitioned_open = True
            else:
                self._transition_circuit(
                    service,
                    runtime,
                    CircuitState.CLOSED,
                    call_index=call_index,
                    timestamp_ms=timestamp_ms,
                    reason="half_open_probe_succeeded",
                    now_ms=now_ms,
                )

        # A negative budget has priority when both protections trip together.
        # This is deliberately stable and independent of dictionary order.
        if runtime.budget_exhausted and not budget_before:
            return "budget_exhausted"
        if circuit_transitioned_open:
            return "circuit_open"
        return None

    def _result(
        self,
        service: ServiceSpec,
        runtime: ServiceRuntime,
        *,
        status: str,
        actual_outcome: str | None,
        consumed: bool,
        fast_failed: bool,
        trigger_source: str | None,
        trigger_reason: str | None,
        propagation_depth: int,
    ) -> ServiceResult:
        return ServiceResult(
            service=service.id,
            status=status,
            actual_outcome=actual_outcome,
            consumed=consumed,
            fast_failed=fast_failed,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            propagation_depth=propagation_depth,
            budget={
                "window_samples": runtime.sample_count,
                "failures": runtime.failure_count,
                "budget_total": runtime.budget_total,
                "raw_remaining": runtime.raw_remaining,
                "remaining": runtime.budget_remaining,
                "exhausted": runtime.budget_exhausted,
            },
            circuit={
                "state": runtime.circuit_state.value,
                "consecutive_failures": runtime.consecutive_failures,
                "cooldown_remaining_ms": self._cooldown_remaining(service, runtime),
                "open_count": runtime.open_count,
            },
        )

    # ------------------------------------------------------------------
    # Public advancement API
    # ------------------------------------------------------------------
    def process_call(self, call: TraceCall, *, now_ms: int | None = None) -> CallDecision:
        if call.index != self.cursor + 1:
            raise ValueError(
                f"call index {call.index} cannot follow cursor {self.cursor}"
            )
        if (
            self.last_trace_timestamp_ms is not None
            and call.timestamp_ms < self.last_trace_timestamp_ms
        ):
            raise ValueError("trace timestamps must be monotonic")

        now = self.clock.now_ms() if now_ms is None else int(now_ms)
        self._tick(call.index, call.timestamp_ms, now)

        results: list[ServiceResult] = []
        trigger_source: str | None = None
        trigger_reason: str | None = None
        trigger_position: int | None = None

        for path_position, service_id in enumerate(call.path):
            service = self.chain.get(service_id)
            runtime = self.runtimes[service_id]
            gate_reason = self._gate(
                service,
                runtime,
                call_index=call.index,
                timestamp_ms=call.timestamp_ms,
                now_ms=now,
            )
            if gate_reason is not None:
                trigger_source = service_id
                trigger_reason = gate_reason
                trigger_position = path_position
                results.append(
                    self._result(
                        service,
                        runtime,
                        status="blocked",
                        actual_outcome=None,
                        consumed=False,
                        fast_failed=False,
                        trigger_source=service_id,
                        trigger_reason=gate_reason,
                        propagation_depth=0,
                    )
                )
                break

            post_trigger = self._consume(
                service,
                runtime,
                call.outcome,
                call_index=call.index,
                timestamp_ms=call.timestamp_ms,
                now_ms=now,
            )
            results.append(
                self._result(
                    service,
                    runtime,
                    status="consumed",
                    actual_outcome=call.outcome.value,
                    consumed=True,
                    fast_failed=False,
                    trigger_source=service_id if post_trigger else None,
                    trigger_reason=post_trigger,
                    propagation_depth=0,
                )
            )
            if post_trigger is not None:
                trigger_source = service_id
                trigger_reason = post_trigger
                trigger_position = path_position
                break

        if trigger_position is not None and trigger_source is not None:
            for path_position in range(trigger_position + 1, len(call.path)):
                service_id = call.path[path_position]
                service = self.chain.get(service_id)
                runtime = self.runtimes[service_id]
                depth = path_position - trigger_position
                results.append(
                    self._result(
                        service,
                        runtime,
                        status="fast_failed",
                        actual_outcome=None,
                        consumed=False,
                        fast_failed=True,
                        trigger_source=trigger_source,
                        trigger_reason=trigger_reason,
                        propagation_depth=depth,
                    )
                )

        fast_chain = tuple(
            {
                "service": item.service,
                "trigger_source": item.trigger_source,
                "trigger_reason": item.trigger_reason,
                "propagation_depth": item.propagation_depth,
            }
            for item in results
            if item.fast_failed
        )
        decision = CallDecision(
            call_index=call.index,
            timestamp_ms=call.timestamp_ms,
            path=call.path,
            outcome=call.outcome.value,
            duration_ms=call.duration_ms,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            fast_failed=bool(fast_chain),
            fast_fail_chain=fast_chain,
            services=tuple(results),
        )
        self.cursor = call.index
        self.last_trace_timestamp_ms = call.timestamp_ms
        self.last_logical_clock_ms = max(self.last_logical_clock_ms, now)
        return decision

    def process_calls(self, calls: Any) -> list[CallDecision]:
        return [self.process_call(call) for call in calls]

    # ------------------------------------------------------------------
    # Serialization, replacement, and read-only views
    # ------------------------------------------------------------------
    def export_state(self, *, include_events: bool = True) -> dict[str, Any]:
        services: list[dict[str, Any]] = []
        for service in sorted(self.chain.services, key=lambda item: item.id):
            runtime = self.runtimes[service.id]
            item: dict[str, Any] = {
                "id": service.id,
                "window_events": [
                    [event.timestamp_ms, 1 if event.failed else 0]
                    for event in runtime.events
                ],
                "sample_count": runtime.sample_count,
                "failure_count": runtime.failure_count,
                "budget_total": runtime.budget_total,
                "raw_remaining": runtime.raw_remaining,
                "budget_remaining": runtime.budget_remaining,
                "budget_exhausted": runtime.budget_exhausted,
                "consecutive_failures": runtime.consecutive_failures,
                "last_failure_timestamp_ms": runtime.last_failure_timestamp_ms,
                "circuit_state": runtime.circuit_state.value,
                "opened_at_ms": runtime.opened_at_ms,
                "open_count": runtime.open_count,
                "half_open_probes_used": runtime.half_open_probes_used,
            }
            services.append(item)
        return {
            "schema": STATE_SCHEMA,
            "chain_version": self.chain.version,
            "chain_digest": self.chain.digest,
            "event_cursor": self.cursor,
            "clock_ms": self.last_logical_clock_ms,
            "last_trace_timestamp_ms": self.last_trace_timestamp_ms,
            "next_event_index": self._next_event_index,
            "services": services,
            "events": copy.deepcopy(self._events) if include_events else [],
        }

    def state_digest(self) -> str:
        return sha256_json(self.export_state())

    def state_json(self) -> str:
        return canonical_json(self.export_state())

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("state must be an object")
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"unsupported state schema: {state.get('schema')!r}")
        if state.get("chain_digest") != self.chain.digest:
            raise ValueError("state chain digest does not match current chain")

        by_id = {item["id"]: item for item in state.get("services", [])}
        expected = set(self.chain.service_ids)
        if set(by_id) != expected:
            raise ValueError("state service set does not match current chain")

        restored: dict[str, ServiceRuntime] = {}
        for service in self.chain.services:
            item = by_id[service.id]
            events = deque(
                WindowEvent(int(timestamp), bool(failed))
                for timestamp, failed in item.get("window_events", [])
            )
            runtime = ServiceRuntime(
                events=events,
                consecutive_failures=int(item.get("consecutive_failures", 0)),
                last_failure_timestamp_ms=item.get("last_failure_timestamp_ms"),
                circuit_state=CircuitState(item.get("circuit_state", "closed")),
                opened_at_ms=item.get("opened_at_ms"),
                open_count=int(item.get("open_count", 0)),
                half_open_probes_used=int(item.get("half_open_probes_used", 0)),
            )
            runtime.sample_count = sum(1 for event in events)
            runtime.failure_count = sum(1 for event in events if event.failed)
            runtime.budget_total = (
                runtime.sample_count * service.budget.ratio_bp // 10_000
            )
            runtime.raw_remaining = runtime.budget_total - runtime.failure_count
            runtime.budget_remaining = max(0, runtime.raw_remaining)
            runtime.budget_exhausted = (
                runtime.sample_count >= service.budget.min_samples
                and runtime.raw_remaining < 0
            )
            restored[service.id] = runtime

        self.runtimes = restored
        self.cursor = int(state.get("event_cursor", 0))
        self.last_logical_clock_ms = int(state.get("clock_ms", 0))
        self.last_trace_timestamp_ms = state.get("last_trace_timestamp_ms")
        self._events = copy.deepcopy(state.get("events", []))
        self._next_event_index = int(
            state.get(
                "next_event_index",
                max((event.get("event_index", 0) for event in self._events), default=0)
                + 1,
            )
        )

    def replace_state(self, state: dict[str, Any]) -> None:
        self.restore_state(state)

    def reset(self) -> None:
        self.runtimes = {
            service.id: ServiceRuntime() for service in self.chain.services
        }
        self.cursor = 0
        self.last_trace_timestamp_ms = None
        self.last_logical_clock_ms = 0
        self._events = []
        self._next_event_index = 1

    def service_rows(self, now_ms: int | None = None) -> list[dict[str, Any]]:
        now = self.clock.now_ms() if now_ms is None else int(now_ms)
        rows: list[dict[str, Any]] = []
        for service in sorted(self.chain.services, key=lambda item: item.id):
            runtime = self.runtimes[service.id]
            state = runtime.circuit_state.value
            cooldown = 0
            if runtime.circuit_state is CircuitState.OPEN:
                opened = runtime.opened_at_ms
                elapsed = 0 if opened is None else max(0, now - opened)
                cooldown = max(0, service.circuit.cooldown_ms - elapsed)
                if cooldown == 0:
                    state = CircuitState.HALF_OPEN.value
            rows.append(
                {
                    "service": service.id,
                    "upstream": list(sorted(service.upstream)),
                    "timeout_ms": service.timeout_ms,
                    "window_hours": service.budget.window_hours,
                    "ratio_bp": service.budget.ratio_bp,
                    "min_samples": service.budget.min_samples,
                    "window_samples": runtime.sample_count,
                    "failures": runtime.failure_count,
                    "budget_total": runtime.budget_total,
                    "budget_remaining": runtime.budget_remaining,
                    "budget_exhausted": runtime.budget_exhausted,
                    "consecutive_failures": runtime.consecutive_failures,
                    "circuit_state": state,
                    "cooldown_remaining_ms": cooldown,
                    "open_count": runtime.open_count,
                }
            )
        return rows

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._events))

def state_from_json(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("state JSON must be an object")
    return value
