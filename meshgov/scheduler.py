from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .budget import BudgetSnapshot, ConcurrencyGate
from .circuit import CircuitManager, CircuitRuntime
from .ledger import Ledger
from .logging import JsonlLogger
from .rules import RequestSpec, RuleSet
from .selector import Selection, SelectionPlanner
from .transport import AttemptResponse, Transport
from .util import mono_ms, sha256_file, utc_now_iso


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    status: str
    code: str | None
    attempts_used: int


@dataclass(frozen=True)
class SchedulerResult:
    round_id: str
    requests: tuple[RequestResult, ...]
    budget: BudgetSnapshot
    observed_rule_version: str
    rules_changed_during_round: bool
    started_at_utc: str
    ended_at_utc: str


class Scheduler:
    def __init__(
        self,
        rules: RuleSet,
        requests: Iterable[RequestSpec],
        transport: Transport,
        ledger: Ledger,
        logger: JsonlLogger,
        *,
        rules_path: str | Path | None = None,
        crash_after_ms: int | None = None,
    ) -> None:
        self.rules = rules
        self.requests = tuple(requests)
        self.transport = transport
        self.ledger = ledger
        self.logger = logger
        self.rules_path = Path(rules_path) if rules_path else None
        self.crash_after_ms = crash_after_ms
        self.planner = SelectionPlanner(rules, self.requests)

    def _crash_timer(self) -> threading.Timer | None:
        if self.crash_after_ms is None:
            return None

        def terminate() -> None:
            os._exit(137)

        timer = threading.Timer(self.crash_after_ms / 1000.0, terminate)
        timer.daemon = True
        timer.start()
        return timer

    def _observe_rule_version(self) -> tuple[str, bool]:
        observed = self.rules.rules_version
        changed = False
        if self.rules_path is not None:
            try:
                observed_sha = sha256_file(self.rules_path)
                observed = f"sha256:{observed_sha}"
            except OSError:
                changed = True
                return observed, True
            changed = observed != self.rules.rules_version
        return observed, changed

    def run(self, round_id: str, *, input_sha256: str) -> SchedulerResult:
        started_at = utc_now_iso()
        self.ledger.begin_round(
            round_id,
            self.rules,
            input_sha256,
            {
                "selection_algorithm": self.planner.algorithm,
                "request_count": len(self.requests),
            },
        )
        for request in self.requests:
            self.ledger.queue_request(round_id, request, mono_ms())
        self.logger.emit(
            "round_started",
            round_id=round_id,
            rule_version=self.rules.rules_version,
            request_count=len(self.requests),
            global_limit=self.rules.global_concurrency,
            service_quotas={
                service.name: service.concurrency_quota for service in self.rules.services
            },
        )

        gate = ConcurrencyGate(
            self.rules.global_concurrency,
            {service.name: service.concurrency_quota for service in self.rules.services},
            on_change=lambda snapshot: self.ledger.persist_budget(round_id, snapshot),
        )
        restored = self.ledger.current_circuit_states()
        circuits = CircuitManager(
            [instance for service in self.rules.services for instance in service.instances],
            on_state=self.ledger.save_circuit_state,
            on_event=lambda event: self._on_circuit_event(round_id, event),
            restored_states=restored,
        )
        self.ledger.persist_budget(round_id, gate.snapshot())
        crash_timer = self._crash_timer()
        results: list[RequestResult] = []

        worker_count = max(
            1,
            min(
                len(self.requests),
                max(self.rules.global_concurrency * 2, 16),
            ),
        )
        try:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=f"mesh-{round_id}",
            ) as executor:
                futures = {
                    executor.submit(
                        self._execute_request,
                        round_id,
                        request,
                        gate,
                        circuits,
                        mono_ms(),
                    ): request
                    for request in self.requests
                }
                for future in as_completed(futures):
                    results.append(future.result())
        finally:
            gate.close()
            if crash_timer is not None:
                crash_timer.cancel()

        results.sort(key=lambda item: next(
            request.index for request in self.requests if request.request_id == item.request_id
        ))
        observed_version, changed = self._observe_rule_version()
        budget = gate.snapshot()
        self.ledger.persist_budget(round_id, budget)
        summary = {
            "status": "success" if all(item.status == "success" for item in results) else "partial",
            "requests_total": len(results),
            "requests_succeeded": sum(item.status == "success" for item in results),
            "requests_failed": sum(item.status != "success" for item in results),
            "global_peak": budget.global_peak,
            "including_injected_upstream_errors": any(
                item.code == "E_UPSTREAM" for item in results
            ),
        }
        self.ledger.finish_round(
            round_id,
            status="completed",
            observed_rule_version=observed_version,
            rules_changed_during_round=changed,
            summary=summary,
        )
        ended_at = utc_now_iso()
        self.logger.emit(
            "round_completed",
            round_id=round_id,
            rule_version=self.rules.rules_version,
            observed_rule_version=observed_version,
            rules_changed_during_round=changed,
            summary=summary,
            global_peak=budget.global_peak,
            queued_peak=None,
        )
        return SchedulerResult(
            round_id=round_id,
            requests=tuple(results),
            budget=budget,
            observed_rule_version=observed_version,
            rules_changed_during_round=changed,
            started_at_utc=started_at,
            ended_at_utc=ended_at,
        )

    def _on_circuit_event(self, round_id: str, event: dict[str, Any]) -> None:
        self.ledger.add_circuit_event(round_id, event)
        payload = dict(event)
        circuit_event = payload.pop("event")
        self.logger.emit(
            "circuit_state_changed",
            round_id=round_id,
            code="E_CIRCUIT" if event["state"] in {"open", "half_open"} else None,
            circuit_event=circuit_event,
            **payload,
        )

    def _record_attempt(
        self,
        round_id: str,
        request: RequestSpec,
        attempt_no: int,
        selection: Selection,
        *,
        selected: bool,
        queue_ms: float,
        start: float,
        end: float,
        result: AttemptResponse,
        global_active: int | None,
        service_active: int | None,
        circuit_admitted_mono_ms: float | None = None,
    ) -> None:
        self.ledger.attempt_started(
            round_id,
            request,
            attempt_no,
            selection,
            selected=selected,
            queue_ms=queue_ms,
            global_active_at_start=global_active,
            service_active_at_start=service_active,
            permit_start_mono_ms=None,
            circuit_admitted_mono_ms=circuit_admitted_mono_ms or start,
            start_mono_ms=start,
        )
        self.ledger.attempt_finished(
            round_id,
            request.request_id,
            attempt_no,
            end_mono_ms=end,
            result_status="success" if result.success else "failure",
            code=result.code,
            detail=result.detail,
            http_status=result.http_status,
            selected=selected,
        )
        self.logger.emit(
            "attempt_completed",
            round_id=round_id,
            request_id=request.request_id,
            attempt_no=attempt_no,
            service=request.route.service,
            route_prefix=request.route.prefix,
            target_instance=selection.target.name,
            target_port=selection.target.port,
            selection_ordinal=selection.ordinal,
            selected=selected,
            dispatched=selected,
            queue_ms=round(queue_ms, 3),
            start_mono_ms=round(start, 3),
            end_mono_ms=round(end, 3),
            duration_ms=round(end - start, 3),
            result_status="success" if result.success else "failure",
            code=result.code,
            detail=result.detail,
            http_status=result.http_status,
        )

    def _execute_request(
        self,
        round_id: str,
        request: RequestSpec,
        gate: ConcurrencyGate,
        circuits: CircuitManager,
        enqueued_mono_ms: float,
    ) -> RequestResult:
        self.ledger.request_started(round_id, request.request_id)
        self.logger.emit(
            "request_started",
            round_id=round_id,
            request_id=request.request_id,
            request_index=request.index,
            service=request.route.service,
            path=request.path,
            timeout_ms=request.route.timeout_ms,
            retry_budget=request.route.max_retries,
        )
        attempts_used = 0
        last_code: str | None = None
        final_status = "failure"
        for attempt_index in range(request.route.max_retries + 1):
            attempt_no = attempt_index + 1
            selection = self.planner.get(request.request_id, attempt_index)
            admission = circuits.admit(selection.target)
            if not admission.allowed:
                start = mono_ms()
                result = AttemptResponse(
                    False,
                    "E_CIRCUIT",
                    f"fast fail: {admission.reason}",
                    None,
                )
                end = mono_ms()
                self._record_attempt(
                    round_id,
                    request,
                    attempt_no,
                    selection,
                    selected=False,
                    queue_ms=0.0,
                    start=start,
                    end=end,
                    result=result,
                    global_active=None,
                    service_active=None,
                    circuit_admitted_mono_ms=admission.checked_at_mono_ms,
                )
                attempts_used = attempt_no
                last_code = "E_CIRCUIT"
                continue

            queue_start = enqueued_mono_ms if attempt_no == 1 else mono_ms()
            acquired_at = gate.acquire(request.route.service)
            queue_ms = acquired_at - queue_start
            revalidated = circuits.admit(selection.target) if admission.state == "closed" else None
            if revalidated is not None and not revalidated.allowed:
                gate.release(request.route.service)
                start = end = mono_ms()
                result = AttemptResponse(
                    False,
                    "E_CIRCUIT",
                    f"fast fail after queue: {revalidated.reason}",
                    None,
                )
                self._record_attempt(
                    round_id,
                    request,
                    attempt_no,
                    selection,
                    selected=False,
                    queue_ms=queue_ms,
                    start=start,
                    end=end,
                    result=result,
                    global_active=None,
                    service_active=None,
                    circuit_admitted_mono_ms=revalidated.checked_at_mono_ms,
                )
                attempts_used = attempt_no
                last_code = "E_CIRCUIT"
                continue

            budget_snapshot = gate.snapshot()
            global_active = budget_snapshot.global_active
            service_active = budget_snapshot.services[request.route.service]["active"]
            start = mono_ms()
            permit_start = acquired_at
            dispatch_admission = revalidated or admission
            self.ledger.attempt_started(
                round_id,
                request,
                attempt_no,
                selection,
                selected=True,
                queue_ms=queue_ms,
                global_active_at_start=global_active,
                service_active_at_start=service_active,
                permit_start_mono_ms=permit_start,
                circuit_admitted_mono_ms=dispatch_admission.checked_at_mono_ms,
                start_mono_ms=start,
            )
            try:
                try:
                    response = self.transport.request(selection.target, request, attempt_no)
                except Exception as exc:  # transport implementations must not kill a round
                    response = AttemptResponse(
                        False,
                        "E_UPSTREAM",
                        f"transport exception: {exc}",
                        None,
                    )
                end = mono_ms()
                if response.success:
                    circuits.record_success(
                        selection.target,
                        probe=dispatch_admission.probe,
                    )
                else:
                    circuits.record_failure(
                        selection.target,
                        probe=dispatch_admission.probe,
                    )
                self.ledger.attempt_finished(
                    round_id,
                    request.request_id,
                    attempt_no,
                    end_mono_ms=end,
                    result_status="success" if response.success else "failure",
                    code=response.code,
                    detail=response.detail,
                    http_status=response.http_status,
                    selected=True,
                )
                self.logger.emit(
                    "attempt_completed",
                    round_id=round_id,
                    request_id=request.request_id,
                    attempt_no=attempt_no,
                    service=request.route.service,
                    route_prefix=request.route.prefix,
                    target_instance=selection.target.name,
                    target_port=selection.target.port,
                    selection_ordinal=selection.ordinal,
                    selected=True,
                    dispatched=True,
                    queue_ms=round(queue_ms, 3),
                    start_mono_ms=round(start, 3),
                    end_mono_ms=round(end, 3),
                    duration_ms=round(end - start, 3),
                    result_status="success" if response.success else "failure",
                    code=response.code,
                    detail=response.detail,
                    http_status=response.http_status,
                )
            finally:
                released_at = gate.release(request.route.service)
                self.ledger.attempt_permit_released(
                    round_id,
                    request.request_id,
                    attempt_no,
                    released_at,
                )
            attempts_used = attempt_no
            if response.success:
                final_status = "success"
                last_code = None
                break
            last_code = response.code or "E_UPSTREAM"

        finished = mono_ms()
        self.ledger.request_finished(
            round_id,
            request.request_id,
            final_status=final_status,
            final_code=last_code if final_status != "success" else None,
            attempts_used=attempts_used,
            finished_mono_ms=finished,
        )
        self.logger.emit(
            "request_finished",
            round_id=round_id,
            request_id=request.request_id,
            request_index=request.index,
            service=request.route.service,
            final_status=final_status,
            final_code=last_code if final_status != "success" else None,
            attempts_used=attempts_used,
            retry_budget=request.route.max_retries,
        )
        return RequestResult(
            request_id=request.request_id,
            status=final_status,
            code=last_code if final_status != "success" else None,
            attempts_used=attempts_used,
        )
