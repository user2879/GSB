from __future__ import annotations

import os
import signal
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from typing import Any, Iterable

from .diagnostics import compute_semantic_outcome
from .errors import Codes, DiagnosticError
from .ledger import Ledger, monotonic_ms
from .nodepool import NodePool
from .spec import SpecBundle, stable_topological_order
from .tcpio import request_response


class Scheduler:
    def __init__(self, bundle: SpecBundle, ledger: Ledger, pool: NodePool | None = None) -> None:
        self.bundle = bundle
        self.ledger = ledger
        self.pool = pool
        self.wait_started: dict[str, int] = {}
        self.wait_records: list[dict[str, Any]] = []
        self.failures: dict[str, DiagnosticError] = {}

    def _lease_id(self, shard_id: str, capacity_version: int) -> str:
        return f"lease-{shard_id}-v{capacity_version}"

    def _summary_available(self, shard_id: str, capacity_version: int) -> bool:
        row = self.ledger.conn.execute(
            "SELECT status,summary FROM shard_outcomes WHERE shard_id=? AND capacity_version=?",
            (shard_id, capacity_version),
        ).fetchone()
        return row is not None and row["status"] == "COMPLETED" and bool(row["summary"])

    def _stored_status(self, shard_id: str, capacity_version: int) -> str | None:
        row = self.ledger.conn.execute(
            "SELECT status FROM shard_outcomes WHERE shard_id=? AND capacity_version=?",
            (shard_id, capacity_version),
        ).fetchone()
        return str(row["status"]) if row else None

    def freeze_ready_set(self, order: list[str], status: dict[str, str], capacity_version: int) -> list[str]:
        """Freeze this pass's deterministic ready frontier; FROZEN items never consume slots."""

        ready: list[str] = []
        for shard_id in order:
            if status.get(shard_id) != "PENDING":
                continue
            shard = self.bundle.shards_by_id[shard_id]
            if all(status.get(dep) == "COMPLETED" and self._summary_available(dep, capacity_version) for dep in shard.dependencies):
                ready.append(shard_id)
        return ready

    def run(
        self,
        *,
        target_ids: Iterable[str] | None = None,
        rerun_ids: Iterable[str] | None = None,
        crash_before_commit: str | None = None,
        lease_ttl_ms: int = 30_000,
        request_timeout: float = 5.0,
        max_workers: int = 8,
    ) -> dict[str, Any]:
        capacity = self.bundle.active_capacity()
        scale = self.bundle.active_fixed().scale_factor
        self.ledger.initialize_capacity_slots(capacity)
        order = stable_topological_order(self.bundle.shards)
        target = set(target_ids) if target_ids is not None else set(order)
        rerun = set(rerun_ids or [])
        unknown_ids = {
            row["shard_id"]
            for row in self.ledger.conn.execute(
                "SELECT shard_id FROM shard_outcomes WHERE capacity_version=? AND (unknown=1 OR status='UNKNOWN')",
                (capacity.version,),
            ).fetchall()
        }
        rerun.update(unknown_ids & target)
        if rerun:
            self.ledger.mark_rerun(rerun, capacity.version)

        status: dict[str, str] = {}
        for shard_id in order:
            stored_status = self._stored_status(shard_id, capacity.version)
            if shard_id in rerun:
                status[shard_id] = "PENDING"
            elif self._summary_available(shard_id, capacity.version):
                status[shard_id] = "COMPLETED"
            elif stored_status == "FROZEN":
                status[shard_id] = "FROZEN"
            elif shard_id not in target:
                status[shard_id] = "OUT_OF_SCOPE"
            else:
                status[shard_id] = "PENDING"

        resume_frontier = 0
        while resume_frontier < len(order) and status[order[resume_frontier]] == "COMPLETED":
            resume_frontier += 1
        persisted_cursor = self.ledger.get_cursor()
        if persisted_cursor is not None and int(persisted_cursor["capacity_version"]) == capacity.version:
            if int(persisted_cursor["next_index"]) > resume_frontier:
                raise DiagnosticError(
                    Codes.REPLAY,
                    "committed cursor is ahead of committed shard outcomes after recovery",
                    field="cursor.next_index",
                    details={"cursor_next_index": persisted_cursor["next_index"], "outcome_frontier": resume_frontier},
                )

        committed: list[str] = []
        futures: dict[Future[dict[str, Any]], str] = {}
        executor = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(order) or 1)), thread_name_prefix="shard-dispatch")
        fatal: DiagnosticError | None = None
        try:
            idle_loops = 0
            while True:
                self.ledger.expire_due_leases()
                active_future_ids = set(futures.values())
                admitted = False
                ready_frozen = self.freeze_ready_set(order, status, capacity.version)
                for shard_id in ready_frozen:
                    if shard_id not in target or shard_id in active_future_ids:
                        continue
                    shard = self.bundle.shards_by_id[shard_id]
                    limit = capacity.node_limits[shard.node_id]
                    lease_id = self._lease_id(shard_id, capacity.version)
                    try:
                        grant = self.ledger.grant_lease(
                            lease_id=lease_id,
                            shard_id=shard_id,
                            node_id=shard.node_id,
                            capacity_version=capacity.version,
                            resource_units=shard.resource_units,
                            ttl_ms=lease_ttl_ms,
                            budget_units=capacity.budget_units,
                            concurrency_quota=limit.concurrency_quota,
                            lease_limit=limit.lease_limit,
                        )
                    except DiagnosticError as exc:
                        if exc.code != Codes.BUDGET:
                            raise
                        if shard_id not in self.wait_started:
                            self.wait_started[shard_id] = monotonic_ms()
                            self.ledger.record_event(
                                capacity_version=capacity.version,
                                kind="SHARD_QUEUED",
                                basis=exc.message,
                                shard_id=shard_id,
                                node_id=shard.node_id,
                                detail=exc.details or {},
                            )
                        continue
                    if shard_id in self.wait_started:
                        start = self.wait_started.pop(shard_id)
                        end = monotonic_ms()
                        wait_record = {
                            "shard_id": shard_id,
                            "wait_started_ms": start,
                            "wait_ended_ms": end,
                            "wait_duration_ms": max(0, end - start),
                            "reason": "global budget/node quota/lease limit was saturated",
                            "grant_event_seq": grant["event_seq"],
                        }
                        self.wait_records.append(wait_record)
                    status[shard_id] = "RUNNING"
                    admitted = True
                    future = executor.submit(self._dispatch_shard, shard_id, grant, scale, request_timeout)
                    futures[future] = shard_id

                pending_target = [sid for sid in order if sid in target and status[sid] != "COMPLETED"]
                if not pending_target:
                    break
                if futures:
                    idle_loops = 0
                    done, _ = wait(set(futures.keys()), timeout=0.02, return_when=FIRST_COMPLETED)
                    for future in done:
                        shard_id = futures.pop(future)
                        try:
                            outcome = future.result()
                        except DiagnosticError as exc:
                            self.failures[shard_id] = exc
                            status[shard_id] = "FAILED"
                            fatal = exc
                            continue
                        if crash_before_commit == shard_id:
                            self.ledger.begin_crash_partial_outcome(outcome)
                            os.kill(os.getpid(), signal.SIGKILL)
                        status[shard_id] = "COMPLETED"
                        frontier = 0
                        while frontier < len(order) and status.get(order[frontier]) == "COMPLETED":
                            frontier += 1
                        completed_count = sum(1 for sid in order if status.get(sid) == "COMPLETED")
                        cursor = {
                            "topology": order,
                            "next_index": frontier,
                            "completed_count": completed_count,
                            "last_shard_id": shard_id,
                            "target_ids": sorted(target),
                            "monotonic_cursor_ms": monotonic_ms(),
                        }
                        self.ledger.commit_outcome(outcome, outcome["water"], outcome["error"], cursor)
                        committed.append(shard_id)
                elif admitted:
                    idle_loops = 0
                else:
                    idle_loops += 1
                    if idle_loops > 250:
                        not_ready = [
                            sid
                            for sid in order
                            if sid in target and status[sid] == "PENDING"
                            and not all(status.get(dep) == "COMPLETED" for dep in self.bundle.shards_by_id[sid].dependencies)
                        ]
                        if not_ready:
                            raise DiagnosticError(Codes.STATE, f"scheduler stalled with unresolved dependencies: {not_ready}")
                        raise DiagnosticError(Codes.BUDGET, "scheduler stalled: shard demand cannot fit current budget/quota limits")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if fatal is not None:
            raise fatal
        return {
            "capacity_version": capacity.version,
            "scale_factor": scale,
            "topology": order,
            "target_ids": sorted(target),
            "committed": committed,
            "cursor_resume_index": resume_frontier,
            "wait_records": self.wait_records,
            "failures": {key: str(value) for key, value in self.failures.items()},
        }

    def _dispatch_shard(self, shard_id: str, grant: dict[str, Any], scale: int, timeout: float) -> dict[str, Any]:
        if self.pool is None:
            raise DiagnosticError(Codes.STATE, "NodePool is required for real TCP dispatch")
        shard = self.bundle.shards_by_id[shard_id]
        node = self.bundle.nodes_by_id[shard.node_id]
        samples = [asdict(self.bundle.samples_by_id[sample_id]) for sample_id in shard.sample_ids]
        canonical = compute_semantic_outcome(self.bundle, shard_id, capacity_version=grant["capacity_version"], scale_factor=scale)
        response = request_response(
            node.host,
            node.port,
            {
                "type": "shard",
                "node_id": node.id,
                "shard_id": shard_id,
                "lease_id": grant["lease_id"],
                "lease_version": grant["lease_version"],
                "capacity_version": grant["capacity_version"],
                "scale_factor": scale,
                "shard": {"id": shard_id, "sample_ids": list(shard.sample_ids)},
                "samples": samples,
            },
            timeout=timeout,
            handshake_node_id=node.id,
        )
        if not response.get("ok"):
            error = response.get("error", {})
            code = error.get("code", Codes.STATE) if isinstance(error, dict) else Codes.STATE
            raise DiagnosticError(code, f"node rejected shard {shard_id}: {error}", source=node.id)
        if response.get("summary") != canonical["summary"]:
            raise DiagnosticError(
                Codes.DIVERGE,
                "TCP node aggregate differs from deterministic local aggregate",
                field="summary",
                details={"shard_id": shard_id, "node_summary": response.get("summary"), "expected": canonical["summary"]},
            )
        outcome = dict(canonical)
        outcome.update(
            {
                "lease_id": grant["lease_id"],
                "lease_version": int(response.get("lease_version", grant["lease_version"])),
                "duration_ms": float(response.get("duration_ms", 0.0)),
                "node_duration_monotonic_start_ms": response.get("duration_monotonic_start_ms"),
                "node_duration_monotonic_end_ms": response.get("duration_monotonic_end_ms"),
            }
        )
        return outcome

    def renew_lease(self, lease_id: str, *, ttl_ms: int | None = None, timeout: float = 2.0) -> dict[str, Any]:
        lease = self.ledger.get_lease(lease_id)
        if lease is None:
            raise DiagnosticError(Codes.LEASE, f"unknown lease: {lease_id}", field="lease_id")
        node = self.bundle.nodes_by_id[lease["node_id"]]
        expected_version = int(lease["lease_version"])
        response = request_response(
            node.host,
            node.port,
            {
                "type": "renew",
                "node_id": node.id,
                "lease_id": lease_id,
                "lease_version": expected_version,
            },
            timeout=timeout,
            handshake_node_id=node.id,
        )
        if not response.get("ok"):
            error = response.get("error", {})
            code = error.get("code", Codes.LEASE) if isinstance(error, dict) else Codes.LEASE
            raise DiagnosticError(code, error.get("message", "lease renewal rejected") if isinstance(error, dict) else str(error), field="lease_version", details=error if isinstance(error, dict) else None)
        return self.ledger.renew_lease(lease_id, expected_version, ttl_ms=ttl_ms)
