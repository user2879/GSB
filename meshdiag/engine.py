from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .circuit import CircuitBreakerRegistry
from .config import Workspace, load_requests, load_revision_document
from .errors import MeshError, diag
from .http_client import LocalHttpClient
from .ledger import Ledger
from .models import (
    AttemptTrace,
    NormalizedRevision,
    RequestResult,
    RequestSpec,
    RevisionChange,
    RoundSummary,
    Service,
    diff_revisions,
    instance_fingerprint,
    normalize_revision,
    parse_service_rows,
    validate_deleted_rule_document,
    validate_deleted_rules,
)
from .scheduler import QuotaScheduler, SchedulingStats
from .selection import select_instance
from .snapshot import SnapshotSession, SnapshotStore, rebuild_revision_from_snapshot, result_rows


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PreparedRevision:
    revision: NormalizedRevision
    previous: NormalizedRevision | None
    change: RevisionChange
    instance_fingerprint: str


@dataclass
class RecoveryReport:
    snapshot_id: str
    snapshot_status: str
    consistency: str
    basis: tuple[str, ...]
    revision_version: int
    current_revision_version: int | None
    revision_replaced: bool
    dependency_graph_replaced: bool
    ledger_replaced: bool
    completed_request_ids: tuple[str, ...]
    unknown_request_ids: tuple[str, ...]
    rerun_required: tuple[str, ...]
    round_id: str
    snapshot_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_status": self.snapshot_status,
            "consistency": self.consistency,
            "basis": list(self.basis),
            "revision_version": self.revision_version,
            "current_revision_version": self.current_revision_version,
            "revision_replaced": self.revision_replaced,
            "dependency_graph_replaced": self.dependency_graph_replaced,
            "ledger_replaced": self.ledger_replaced,
            "completed_request_ids": list(self.completed_request_ids),
            "unknown_request_ids": list(self.unknown_request_ids),
            "rerun_required": list(self.rerun_required),
            "round_id": self.round_id,
            "snapshot_path": self.snapshot_path,
        }


class Engine:
    def __init__(
        self,
        workspace: Workspace,
        *,
        ledger: Ledger | None = None,
        breakers: CircuitBreakerRegistry | None = None,
    ) -> None:
        self.workspace = workspace
        self.ledger = ledger or Ledger(workspace.ledger_path)
        self.breakers = breakers or CircuitBreakerRegistry()
        self.snapshots = SnapshotStore(workspace.snapshot_dir)

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def load_workspace_revision(self) -> NormalizedRevision:
        document = load_revision_document(self.workspace.revision_path)
        return normalize_revision(
            document,
            self.workspace.services,
            workspace_id=self.workspace.workspace_id,
            defaults=self.workspace.defaults,
        )

    def _previous_revision(self) -> NormalizedRevision | None:
        previous_docs = self.ledger.current_revision_documents()
        if previous_docs is None:
            return None
        document, defaults, services_rows = previous_docs
        services = parse_service_rows(services_rows)
        return normalize_revision(
            document,
            services,
            workspace_id=self.workspace.workspace_id,
            defaults=defaults,
        )

    def load_active_revision_and_services(self) -> tuple[NormalizedRevision, tuple[Service, ...]]:
        """Return the revision currently represented by the ledger.

        After snapshot recovery this may intentionally differ from the editable
        workspace revision file. The ledger is the active diagnostic view.
        """

        documents = self.ledger.current_revision_documents()
        if documents is None:
            return self.load_workspace_revision(), self.workspace.services
        document, defaults, service_rows = documents
        services = parse_service_rows(service_rows)
        revision = normalize_revision(
            document,
            services,
            workspace_id=self.workspace.workspace_id,
            defaults=defaults,
        )
        return revision, services

    def prepare_revision(
        self,
        revision_document: dict[str, Any] | None = None,
    ) -> PreparedRevision:
        document = (
            revision_document
            if revision_document is not None
            else load_revision_document(self.workspace.revision_path)
        )
        previous = self._previous_revision()
        if previous is not None:
            validate_deleted_rule_document(previous.rules, document)
        revision = (
            normalize_revision(
                document,
                self.workspace.services,
                workspace_id=self.workspace.workspace_id,
                defaults=self.workspace.defaults,
            )
        )
        fingerprint = instance_fingerprint(self.workspace.services)
        self.ledger.assert_monotonic(
            revision.revision_version, revision.digest, fingerprint
        )
        if previous is not None:
            validate_deleted_rules(previous.rules, revision.rules)
        change = diff_revisions(previous, revision)
        return PreparedRevision(
            revision=revision,
            previous=previous,
            change=change,
            instance_fingerprint=fingerprint,
        )

    @staticmethod
    def match_rule(path: str, revision: NormalizedRevision):
        matches = []
        for rule in revision.rules:
            prefix = rule.prefix
            if prefix == "/" or path == prefix or path.startswith(prefix + "/"):
                matches.append(rule)
        matches.sort(key=lambda item: (-len(item.prefix), item.rule_id))
        return matches[0] if matches else None

    def execute_sequence(
        self,
        *,
        revision: NormalizedRevision,
        services: tuple[Service, ...],
        requests: tuple[RequestSpec, ...],
        global_concurrency: int,
        breakers: CircuitBreakerRegistry | None = None,
        snapshot: SnapshotSession | None = None,
        progress: ProgressCallback | None = None,
        crash_after_completed: int | None = None,
    ) -> tuple[tuple[RequestResult, ...], SchedulingStats]:
        registry = breakers or self.breakers
        service_map = {service.service_id: service for service in services}
        rules_by_request = {
            request.request_id: self.match_rule(request.path, revision)
            for request in requests
        }
        scheduler = QuotaScheduler(
            global_concurrency,
            {service.service_id: service.max_concurrency for service in services},
        )
        state_lock = threading.RLock()
        in_progress = {
            request.request_id: {
                "request_id": request.request_id,
                "request_index": request.index,
                "method": request.method,
                "path": request.path,
                "state": "queued",
                "attempts": 0,
                "budget_consumed_ms": 0,
            }
            for request in requests
        }
        budget_state = {
            "global_active": 0,
            "global_limit": global_concurrency,
            "service_active": {service.service_id: 0 for service in services},
            "service_limits": {
                service.service_id: service.max_concurrency for service in services
            },
            "completed_budget_ms": 0,
            "inflight_budget_ms": 0,
        }
        completed_count = 0
        completed_lock = threading.Lock()

        def emit(event: dict[str, Any]) -> None:
            if progress is not None:
                progress(event)

        def checkpoint() -> None:
            if snapshot is None:
                return
            snapshot.update(
                in_progress=[in_progress[key] for key in sorted(in_progress, key=lambda rid: (in_progress[rid]["request_index"], rid))],
                budget=dict(budget_state),
                circuit_breakers=list(registry.snapshot()),
            )

        def quota_changed(service_id: str, active: int, limit: int) -> None:
            with state_lock:
                budget_state["service_active"][service_id] = active
                budget_state["service_limits"][service_id] = limit
                checkpoint()

        def global_changed(active: int, limit: int) -> None:
            with state_lock:
                budget_state["global_active"] = active
                budget_state["global_limit"] = limit
                checkpoint()

        def finish_request(result: RequestResult) -> RequestResult:
            nonlocal completed_count
            with state_lock:
                in_progress.pop(result.request_id, None)
                budget_state["completed_budget_ms"] += result.budget_consumed_ms
                budget_state["inflight_budget_ms"] = max(
                    0, budget_state["inflight_budget_ms"] - result.budget_consumed_ms
                )
                if snapshot is not None:
                    snapshot.complete_request(result)
                    snapshot.update(
                        in_progress=[
                            in_progress[key]
                            for key in sorted(
                                in_progress,
                                key=lambda rid: (
                                    in_progress[rid]["request_index"],
                                    rid,
                                ),
                            )
                        ],
                        budget=dict(budget_state),
                        circuit_breakers=list(registry.snapshot()),
                    )
            with completed_lock:
                completed_count += 1
                current_completed = completed_count
                if (
                    crash_after_completed is not None
                    and current_completed >= crash_after_completed
                ):
                    if snapshot is not None:
                        snapshot.update(status="running")
                    os._exit(96)
            emit(
                {
                    "event": "request_completed",
                    "completed": current_completed,
                    "request_id": result.request_id,
                    "outcome": result.outcome,
                }
            )
            return result

        def worker(request: RequestSpec) -> RequestResult:
            nonlocal completed_count
            rule = rules_by_request[request.request_id]
            wall_start = time.monotonic_ns()
            with state_lock:
                in_progress[request.request_id].update(
                    state="running",
                    rule_id=rule.rule_id if rule else None,
                    service_id=rule.target_service if rule else None,
                )
                checkpoint()
            if rule is None:
                wall_end = time.monotonic_ns()
                result = RequestResult(
                    request_id=request.request_id,
                    request_index=request.index,
                    method=request.method,
                    path=request.path,
                    rule_id=None,
                    service_id=None,
                    outcome="failure",
                    reason="no_route",
                    attempt_count=0,
                    instance_sequence=(),
                    budget_consumed_ms=0,
                    wall_elapsed_ms=max(0, (wall_end - wall_start) // 1_000_000),
                    traces=(),
                    detail="没有匹配的路由前缀",
                )
                return finish_request(result)
            service = service_map[rule.target_service]
            weights = rule.weights()
            traces: list[AttemptTrace] = []
            for attempt_no in range(1, rule.retries + 2):
                instance = select_instance(
                    service, weights, request.request_id, attempt_no
                )
                started = time.monotonic_ns()
                allowed, circuit_reason = registry.allow(instance.instance_id)
                if not allowed:
                    response_outcome = "circuit_open"
                    reason = circuit_reason
                    status_code = None
                    detail = "熔断冷却窗口内快速失败"
                    ended = time.monotonic_ns()
                else:
                    response = LocalHttpClient(rule.timeout_ms).attempt(
                        instance, request.method, request.path, attempt_no
                    )
                    response_outcome = response.outcome
                    reason = response.reason
                    status_code = response.status_code
                    detail = response.detail
                    ended = time.monotonic_ns()
                    if response.outcome == "success":
                        registry.record_success(instance.instance_id)
                    else:
                        registry.record_failure(
                            instance.instance_id,
                            threshold=rule.circuit_threshold,
                            cooldown_ms=rule.cooldown_ms,
                        )
                trace = AttemptTrace(
                    request_id=request.request_id,
                    attempt_no=attempt_no,
                    rule_id=rule.rule_id,
                    service_id=service.service_id,
                    instance_id=instance.instance_id,
                    outcome=response_outcome,
                    reason=reason,
                    started_mono_ms=started // 1_000_000,
                    ended_mono_ms=ended // 1_000_000,
                    status_code=status_code,
                    detail=detail,
                )
                traces.append(trace)
                with state_lock:
                    budget_state["inflight_budget_ms"] += trace.elapsed_ms
                    in_progress[request.request_id].update(
                        attempts=attempt_no,
                        instance_id=instance.instance_id,
                        budget_consumed_ms=sum(item.elapsed_ms for item in traces),
                        last_reason=reason,
                    )
                    if snapshot is not None:
                        snapshot.add_attempt(request.request_id, trace)
                    checkpoint()
                emit(
                    {
                        "event": "attempt",
                        "request_id": request.request_id,
                        "attempt_no": attempt_no,
                        "instance_id": instance.instance_id,
                        "outcome": response_outcome,
                        "reason": reason,
                    }
                )
                if response_outcome == "success":
                    break
            wall_end = time.monotonic_ns()
            final = traces[-1]
            if final.outcome == "success":
                outcome = "success"
            elif final.outcome == "circuit_open":
                outcome = "circuit_open"
            elif final.outcome == "timeout":
                outcome = "timeout"
            else:
                outcome = "failure"
            result = RequestResult(
                request_id=request.request_id,
                request_index=request.index,
                method=request.method,
                path=request.path,
                rule_id=rule.rule_id,
                service_id=service.service_id,
                outcome=outcome,
                reason=final.reason,
                attempt_count=len(traces),
                instance_sequence=tuple(trace.instance_id for trace in traces),
                budget_consumed_ms=sum(trace.elapsed_ms for trace in traces),
                wall_elapsed_ms=max(0, (wall_end - wall_start) // 1_000_000),
                traces=tuple(traces),
                detail=final.detail,
            )
            return finish_request(result)

        if snapshot is not None:
            with state_lock:
                checkpoint()

        def service_for(request: RequestSpec) -> str | None:
            rule = rules_by_request[request.request_id]
            return None if rule is None else rule.target_service

        results = scheduler.run(
            requests,
            service_for,
            worker,
            quota_changed=quota_changed,
            global_changed=global_changed,
        )
        return tuple(results), scheduler.stats

    def run_round(
        self,
        *,
        round_id: str,
        snapshot_id: str | None = None,
        revision_document: dict[str, Any] | None = None,
        requests: tuple[RequestSpec, ...] | None = None,
        progress: ProgressCallback | None = None,
        crash_after_completed: int | None = None,
    ) -> RoundSummary:
        prepared = self.prepare_revision(revision_document)
        request_sequence = requests or load_requests(self.workspace.requests_path)
        actual_snapshot_id = snapshot_id or f"snapshot-{round_id}"
        initial_breakers = self.breakers.snapshot()
        session = self.snapshots.create_for_round(
            snapshot_id=actual_snapshot_id,
            workspace_id=self.workspace.workspace_id,
            revision=prepared.revision,
            services=self.workspace.services,
            defaults=self.workspace.defaults,
            global_concurrency=self.workspace.global_concurrency,
            requests=request_sequence,
            circuit_breaker_rows=initial_breakers,
        )
        if progress is not None:
            progress(
                {
                    "event": "round_started",
                    "round_id": round_id,
                    "revision_version": prepared.revision.revision_version,
                    "total_requests": len(request_sequence),
                    "change": prepared.change.as_dict(),
                }
            )
        results, stats = self.execute_sequence(
            revision=prepared.revision,
            services=self.workspace.services,
            requests=request_sequence,
            global_concurrency=self.workspace.global_concurrency,
            breakers=self.breakers,
            snapshot=session,
            progress=progress,
            crash_after_completed=crash_after_completed,
        )
        summary = RoundSummary(
            round_id=round_id,
            revision_version=prepared.revision.revision_version,
            revision_digest=prepared.revision.digest,
            dependency_graph_version=prepared.revision.dependency_graph.version,
            instance_fingerprint=prepared.instance_fingerprint,
            results=results,
            total_requests=len(results),
            success=sum(result.outcome == "success" for result in results),
            failure=sum(result.outcome == "failure" for result in results),
            timeout=sum(result.outcome == "timeout" for result in results),
            circuit_open=sum(result.outcome == "circuit_open" for result in results),
            unknown=sum(result.unknown for result in results),
            max_global_active=stats.max_global_active,
            max_service_active=dict(sorted(stats.max_service_active.items())),
            change=prepared.change,
            snapshot_path=str(session.path.resolve()),
        )
        self.ledger.commit_round(
            summary,
            revision=prepared.revision,
            services=self.workspace.services,
            defaults=self.workspace.defaults,
            snapshot_path=summary.snapshot_path,
        )
        ledger_rows = self.ledger.round_rows(round_id)
        session.update(
            status="completed",
            ledger_rows=ledger_rows,
            circuit_breakers=list(self.breakers.snapshot()),
            summary=summary.as_dict(),
        )
        if progress is not None:
            progress({"event": "round_completed", "summary": summary.as_dict()})
        return summary

    def recover_snapshot(self, snapshot_path: str | Path) -> RecoveryReport:
        payload = self.snapshots.load(snapshot_path)
        revision, services = rebuild_revision_from_snapshot(payload)
        completed = result_rows(payload)
        completed_ids = {result.request_id for result in completed}
        request_rows = sorted(
            payload.get("requests", []),
            key=lambda row: (int(row.get("index", 0)), str(row.get("id", ""))),
        )
        unknown_results = []
        for row in request_rows:
            request_id = str(row.get("id", ""))
            if not request_id or request_id in completed_ids:
                continue
            unknown_results.append(
                RequestResult(
                    request_id=request_id,
                    request_index=int(row.get("index", 0)),
                    method=str(row.get("method", "GET")),
                    path=str(row.get("path", "/")),
                    rule_id=None,
                    service_id=None,
                    outcome="unknown",
                    reason="snapshot_incomplete",
                    attempt_count=0,
                    instance_sequence=(),
                    budget_consumed_ms=0,
                    wall_elapsed_ms=0,
                    traces=(),
                    unknown=True,
                    rerun_required=True,
                    detail="进程终止时未完整提交，必须重跑",
                )
            )
        all_results = tuple(completed) + tuple(unknown_results)
        current_revision = None
        current_revision_error = None
        try:
            current_revision = self.load_workspace_revision()
        except MeshError as exc:
            current_revision_error = exc
        ledger_state = self.ledger.current_state()
        basis = []
        revision_replaced = False
        dependency_replaced = False
        if current_revision is None:
            revision_replaced = True
            dependency_replaced = True
            basis.append(
                "当前工作区修订不可用，采用快照修订及依赖图"
                + (f": {current_revision_error}" if current_revision_error else "")
            )
        else:
            if (
                current_revision.revision_version != revision.revision_version
                or current_revision.digest != revision.digest
            ):
                revision_replaced = True
                basis.append(
                    "快照修订 "
                    f"r{revision.revision_version}/{revision.digest[:12]} 与当前修订 "
                    f"r{current_revision.revision_version}/{current_revision.digest[:12]} 不一致"
                )
            if (
                current_revision.dependency_graph.version
                != revision.dependency_graph.version
            ):
                dependency_replaced = True
                basis.append("快照依赖图版本与当前依赖图版本不一致")
        if ledger_state is not None and (
            ledger_state.get("revision_version") != revision.revision_version
            or ledger_state.get("revision_digest") != revision.digest
        ):
            revision_replaced = True
            basis.append("账本当前版本与快照修订版本不一致")
        fingerprint = instance_fingerprint(services)
        round_id = self.ledger.replace_with_snapshot(
            snapshot_id=str(payload["snapshot_id"]),
            revision=revision,
            services=services,
            defaults=dict(payload.get("defaults", {})),
            instance_fp=fingerprint,
            results=all_results,
            status="interrupted" if payload.get("status") == "running" else "restored",
            snapshot_path=str(payload.get("_snapshot_file", snapshot_path)),
        )
        self.breakers.restore(payload.get("circuit_breakers", []))
        unknown_ids = tuple(result.request_id for result in unknown_results)
        if payload.get("status") == "running":
            consistency = "INTERRUPTED_RERUN_REQUIRED"
            basis.append(
                f"快照为 running，{len(unknown_ids)} 个未完成请求已标记 unknown"
            )
        elif revision_replaced or dependency_replaced:
            consistency = "SNAPSHOT_REVISION_MISMATCH"
        else:
            consistency = "CONSISTENT"
            basis.append("快照修订、依赖图与当前修订一致；内容来自原子提交记录")
        return RecoveryReport(
            snapshot_id=str(payload["snapshot_id"]),
            snapshot_status=str(payload.get("status")),
            consistency=consistency,
            basis=tuple(basis),
            revision_version=revision.revision_version,
            current_revision_version=(
                current_revision.revision_version if current_revision is not None else None
            ),
            revision_replaced=revision_replaced,
            dependency_graph_replaced=dependency_replaced,
            ledger_replaced=True,
            completed_request_ids=tuple(
                result.request_id for result in sorted(completed, key=lambda item: item.request_index)
            ),
            unknown_request_ids=unknown_ids,
            rerun_required=unknown_ids,
            round_id=round_id,
            snapshot_path=str(payload.get("_snapshot_file", snapshot_path)),
        )
