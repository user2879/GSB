from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

from .determinism import build_summary, generate_samples, split_batches
from .errors import DiagnosticError, FieldError, LeaseError
from .impact import invalidation_plan
from .lease import (
    LEASE_ACTIVE,
    LEASE_CANCELLED,
    LEASE_EXPIRED,
    LEASE_FAILED,
    LEASE_REVOKED_ROLLBACK,
    Lease,
    LeaseRegistry,
)
from .ledger import Ledger
from .protocol import require_fields
from .spec import Event, EventStream, Spec, topo_rank_map
from .watermark import (
    budget_candidate_judgement,
    compare_fixed_water,
    global_budget_attribution,
    monotonic_ms,
    node_water_report,
    stable_candidate_sort_key,
)

STATUS_WAITING_DEPENDENCY = "waiting_dependency"
STATUS_QUEUED_CAPACITY = "queued_capacity"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_INVALIDATED = "invalidated_rollback"
STATUS_BLOCKED = "blocked_dependency"
STATUS_FAILED = "failed"


class ExecutionEngine:
    def __init__(self, spec: Spec, ledger: Ledger, nodes: Any | None = None, samples_dir: str = "samples") -> None:
        self.spec = spec
        self.ledger = ledger
        self.nodes = nodes
        self.samples_dir = samples_dir
        self.registry = LeaseRegistry(spec)
        self.topo_rank = topo_rank_map(spec)
        self.engine_lock = threading.RLock()
        self.pool: ThreadPoolExecutor | None = None
        self.futures: dict[str, Future[Any]] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.status: dict[str, str] = {}
        self.wait_started_ms: dict[str, int] = {}
        self.wait_ended_ms: dict[str, int] = {}
        self.priority_overrides: dict[str, int] = {}
        self.quota_versions = {node.node_id: node.quota_version for node in spec.nodes}
        self.dispatch_node: dict[str, str] = {}
        self.dispatch_options: dict[str, dict[str, Any]] = {}
        self.granted_quota_version: dict[str, int] = {}
        self.finalized_terminal: set[str] = set()
        self.worker_threads: dict[str, str] = {}
        self.outputs: list[dict[str, Any]] = []
        self.admission_log: list[dict[str, Any]] = []
        self.last_event_seq = 0

    def run_events(self, events: EventStream | list[Event]) -> dict[str, Any]:
        event_list = list(events.events) if isinstance(events, EventStream) else list(events)
        worker_count = max(4, sum(node.concurrency_quota for node in self.spec.nodes) + 2)
        self.pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="shard-worker")
        try:
            for event in event_list:
                output = self.handle_event(event)
                self.outputs.append(output)
            self.wait_idle()
            self.pump("final")
            summary = self.final_summary()
            return {"ok": True, "summary": summary, "events": self.outputs}
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
                self.pool = None

    def handle_event(self, event: Event) -> dict[str, Any]:
        self.last_event_seq = event.seq
        handler = {
            "dispatch": self._on_dispatch,
            "cancel": self._on_cancel,
            "priority": self._on_priority,
            "quota_upgrade": self._on_quota_upgrade,
            "node_offline": self._on_node_offline,
            "rollback": self._on_rollback,
        }[event.type]
        decision = handler(event)
        with self.ledger.transaction() as ledger:
            ledger.journal_event_locked(
                event.seq,
                self._event_hash(event),
                {"seq": event.seq, "type": event.type, **event.data},
                decision,
                decision.get("impact"),
            )
            ledger.set_progress_locked(event.seq, "event_processed")
        return {"seq": event.seq, "type": event.type, "decision": decision}

    @staticmethod
    def _event_hash(event: Event) -> str:
        from .impact import event_hash

        return event_hash(event)

    def _on_dispatch(self, event: Event) -> dict[str, Any]:
        shard_id = event.data["shard_id"]
        node_id = event.data["node_id"]
        with self.engine_lock:
            self.dispatch_node[shard_id] = node_id
            self.dispatch_options[shard_id] = dict(event.data)
            self.cancel_events[shard_id] = threading.Event()
            self.status[shard_id] = STATUS_WAITING_DEPENDENCY
            self.wait_started_ms[shard_id] = monotonic_ms()
            self.wait_ended_ms.pop(shard_id, None)
            decisions = self.pump(f"event:{event.seq}:dispatch")
        return {"action": "dispatch_registered", "shard_id": shard_id, "node_id": node_id, "admission_decisions": decisions}

    def _on_cancel(self, event: Event) -> dict[str, Any]:
        shard_id = event.data["shard_id"]
        with self.engine_lock:
            cancel_event = self.cancel_events.setdefault(shard_id, threading.Event())
            cancel_event.set()
            lease = self.registry.get_shard_lease(shard_id)
            changed = False
            if lease is not None and lease.state == LEASE_ACTIVE:
                try:
                    if self.nodes is not None:
                        self.nodes.call(lease.node_id, {"type": "lease_release", "lease_id": lease.lease_id}, timeout_ms=500)
                except Exception:
                    pass
                changed = self.registry.transition_locked(lease, LEASE_CANCELLED)
            if self.status.get(shard_id) not in {STATUS_COMPLETED, STATUS_INVALIDATED}:
                self.status[shard_id] = STATUS_CANCELLED
            self._persist_simple_terminal(shard_id, STATUS_CANCELLED, {"code": "E_CANCELLED", "reason": event.data.get("reason", "user_cancel")}, lease)
            decisions = self.pump(f"event:{event.seq}:cancel")
        return {"action": "cancel", "shard_id": shard_id, "lease_transitioned": changed, "admission_decisions": decisions}

    def _on_priority(self, event: Event) -> dict[str, Any]:
        shard_id = event.data["shard_id"]
        priority = int(event.data["priority"])
        with self.engine_lock:
            old = self.priority_overrides.get(shard_id, self.spec.shard_map[shard_id].priority)
            self.priority_overrides[shard_id] = priority
            decisions = self.pump(f"event:{event.seq}:priority")
        return {"action": "priority_adjusted", "shard_id": shard_id, "old_priority": old, "new_priority": priority, "admission_decisions": decisions}

    def _on_quota_upgrade(self, event: Event) -> dict[str, Any]:
        version = int(event.data["quota_version"])
        node_id = event.data.get("node_id")
        targets = [node_id] if node_id else list(self.quota_versions)
        changed: list[dict[str, Any]] = []
        with self.engine_lock:
            for target_node in targets:
                old = self.quota_versions[target_node]
                self.quota_versions[target_node] = version
                if self.nodes is not None:
                    try:
                        self.nodes.call(target_node, {"type": "quota_set", "quota_version": version}, timeout_ms=1000)
                    except Exception:
                        pass
                changed.append({"node_id": target_node, "old_version": old, "new_version": version})
                self._persist_watermarks(event.seq, reason="quota_upgrade")
        return {"action": "quota_upgrade", "targets": changed}

    def _on_node_offline(self, event: Event) -> dict[str, Any]:
        node_id = event.data["node_id"]
        force_kill = bool(event.data.get("force_kill", False))
        with self.engine_lock:
            self.registry.mark_offline(node_id)
            if self.nodes is not None:
                if force_kill:
                    self.nodes.force_kill_node(node_id)
                else:
                    # Drain/default offline keeps the local process available for
                    # in-flight requests but blocks new admissions.
                    pass
            self._persist_watermarks(event.seq, reason="node_offline")
            decisions = self.pump(f"event:{event.seq}:offline")
        return {"action": "node_offline", "node_id": node_id, "force_kill": force_kill, "admission_decisions": decisions}

    def _on_rollback(self, event: Event) -> dict[str, Any]:
        target = int(event.data["target_version"])
        node_id = event.data.get("node_id")
        with self.engine_lock:
            result = self.registry.rollback(
                target,
                node_id,
                node_call=(lambda nid, payload: self.nodes.call(nid, payload, timeout_ms=1000)) if self.nodes is not None else None,
            )
            revoked = result["revoked"]
            expired = result["expired"]
            direct = {lease.shard_id for lease in revoked}
            for shard_id, granted_version in self.granted_quota_version.items():
                assigned_node = self.dispatch_node.get(shard_id)
                if node_id is not None and assigned_node != node_id:
                    continue
                if granted_version > target:
                    direct.add(shard_id)
            impact = invalidation_plan(
                self.spec,
                direct,
                node_id=node_id,
                quota_version=max((lease.quota_version for lease in revoked), default=None),
                target_version=target,
            )
            invalid_set = set(impact["invalid_shards"])
            for shard_id in invalid_set:
                cancel = self.cancel_events.setdefault(shard_id, threading.Event())
                cancel.set()
                if shard_id not in self.finalized_terminal:
                    self.status[shard_id] = STATUS_INVALIDATED
            self._persist_rollback(event.seq, event.data, impact, revoked, expired, invalid_set)
            decisions = self.pump(f"event:{event.seq}:rollback")
        return {
            "action": "rollback",
            "target_version": target,
            "scoped_node": node_id,
            "revoked_lease_ids": [lease.lease_id for lease in revoked],
            "expired_lease_ids": [lease.lease_id for lease in expired],
            "impact": impact,
            "admission_decisions": decisions,
        }

    def priority_for(self, shard_id: str) -> int:
        return self.priority_overrides.get(shard_id, self.spec.shard_map[shard_id].priority)

    def pump(self, trigger: str) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        with self.engine_lock:
            self._reap_expired_internal(persist=True)
            candidates = [
                shard_id
                for shard_id, status in self.status.items()
                if status in {STATUS_WAITING_DEPENDENCY, STATUS_QUEUED_CAPACITY}
            ]
            candidates.sort(key=lambda sid: stable_candidate_sort_key(sid, self.priority_for(sid), self.topo_rank))
            for shard_id in candidates:
                shard = self.spec.shard_map[shard_id]
                node_id = self.dispatch_node.get(shard_id)
                if node_id is None:
                    continue
                unmet = [dep for dep in shard.dependencies if self.status.get(dep) != STATUS_COMPLETED]
                terminal_dep = [
                    dep for dep in shard.dependencies
                    if self.status.get(dep) in {STATUS_CANCELLED, STATUS_INVALIDATED, STATUS_FAILED, STATUS_BLOCKED}
                ]
                if terminal_dep:
                    self.status[shard_id] = STATUS_BLOCKED
                    decision = self._decision(shard_id, node_id, False, ["dependency_terminal"], trigger, unmet)
                    decisions.append(decision)
                    self._persist_simple_terminal(
                        shard_id,
                        STATUS_BLOCKED,
                        {"code": "E_DEPENDENCY", "terminal_dependencies": terminal_dep},
                        None,
                    )
                    continue
                if unmet:
                    self.status[shard_id] = STATUS_WAITING_DEPENDENCY
                    if shard_id not in self.wait_started_ms:
                        self.wait_started_ms[shard_id] = monotonic_ms()
                    decision = self._decision(shard_id, node_id, False, ["dependencies_unmet"], trigger, unmet)
                    decisions.append(decision)
                    continue
                reasons = self.registry.capacity_reasons(shard, node_id)
                if reasons:
                    self.status[shard_id] = STATUS_QUEUED_CAPACITY
                    if shard_id not in self.wait_started_ms:
                        self.wait_started_ms[shard_id] = monotonic_ms()
                    decision = self._decision(shard_id, node_id, False, reasons, trigger, [])
                    decisions.append(decision)
                    continue
                wait_start = self.wait_started_ms.get(shard_id, monotonic_ms())
                wait_end = monotonic_ms()
                self.wait_ended_ms[shard_id] = wait_end
                quota_version = self.quota_versions[node_id]
                lease = self.registry.admit(shard_id, node_id, quota_version)
                self.granted_quota_version[shard_id] = quota_version
                self.status[shard_id] = STATUS_RUNNING
                self.finalized_terminal.discard(shard_id)
                if self.nodes is not None:
                    self.nodes.call(
                        node_id,
                        {
                            "type": "lease_grant",
                            "lease_id": lease.lease_id,
                            "shard_id": shard_id,
                            "lease_version": lease.version,
                            "expires_monotonic_ms": lease.expires_monotonic_ms,
                        },
                        timeout_ms=1000,
                    )
                self._persist_grant(lease, wait_start, wait_end)
                decision = self._decision(shard_id, node_id, True, [], trigger, [])
                decision["lease_id"] = lease.lease_id
                decision["lease_version"] = lease.version
                decision["wait_duration_ms"] = max(0, wait_end - wait_start)
                decisions.append(decision)
                assert self.pool is not None
                future = self.pool.submit(self._run_shard, shard_id, lease)
                self.futures[shard_id] = future
                future.add_done_callback(lambda _future, sid=shard_id: self._future_done(sid))
            self.admission_log.extend(decisions)
        return decisions

    def _decision(self, shard_id: str, node_id: str, admitted: bool, reasons: list[str], trigger: str, unmet: list[str]) -> dict[str, Any]:
        shard = self.spec.shard_map[shard_id]
        used = self.registry.used_resource_units()
        budget = budget_candidate_judgement(
            used_units=used,
            candidate_units=shard.resource_units,
            budget=self.spec.global_budget,
            scale=self.spec.runtime.fixed_point_scale,
        )
        node_report = node_water_report(
            node_id=node_id,
            active_shards=self.registry.active_shard_count_on_node(node_id),
            active_leases=self.registry.active_count_on_node(node_id),
            concurrency_quota=self.spec.node_map[node_id].concurrency_quota,
            max_leases=self.spec.node_map[node_id].max_leases_per_node,
            scale=self.spec.runtime.fixed_point_scale,
        )
        key = stable_candidate_sort_key(shard_id, self.priority_for(shard_id), self.topo_rank)
        return {
            "seq_hint": self.last_event_seq,
            "trigger": trigger,
            "order_key": key,
            "shard_id": shard_id,
            "node_id": node_id,
            "admitted": admitted,
            "reasons": reasons,
            "unmet_dependencies": unmet,
            "judgement_order": "priority_desc,topological_index,shard_id",
            "budget": budget,
            "node_water": node_report,
        }

    def _future_done(self, shard_id: str) -> None:
        try:
            self.pump(f"future:{shard_id}")
        except Exception:
            # Pump errors are persisted by the worker; callback must not raise.
            pass

    def wait_idle(self, timeout_ms: int | None = None) -> None:
        deadline = None if timeout_ms is None else monotonic_ms() + timeout_ms
        while True:
            with self.engine_lock:
                self.pump("wait_idle")
                pending = [future for future in self.futures.values() if not future.done()]
            if not pending:
                with self.engine_lock:
                    self.pump("wait_idle_empty")
                    pending = [future for future in self.futures.values() if not future.done()]
                if not pending:
                    return
            if deadline is not None and monotonic_ms() > deadline:
                raise TimeoutError("engine did not become idle before timeout")
            wait(pending, timeout=0.02)

    def renew_lease(self, shard_id: str, expected_version: int) -> Lease:
        def node_call(node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            if self.nodes is None:
                raise LeaseError("node manager is not available")
            return self.nodes.call(node_id, payload, timeout_ms=1000)

        lease = self.registry.renew(shard_id, expected_version, node_call=node_call)
        with self.ledger.transaction() as ledger:
            ledger.insert_lease_event_locked({
                "event_seq": self.last_event_seq,
                "lease_id": lease.lease_id,
                "shard_id": shard_id,
                "node_id": lease.node_id,
                "kind": "renew",
                "from_version": expected_version,
                "to_version": lease.version,
                "expires_monotonic_ms": lease.expires_monotonic_ms,
            })
        return lease

    def reap_expired(self) -> list[Lease]:
        with self.engine_lock:
            return self._reap_expired_internal(persist=True)

    def _reap_expired_internal(self, *, persist: bool) -> list[Lease]:
        expired = self.registry.reap_expired()
        if expired and persist:
            for lease in expired:
                if lease.shard_id not in self.finalized_terminal:
                    self.status[lease.shard_id] = "expired"
                    self.finalized_terminal.add(lease.shard_id)
            with self.ledger.transaction() as ledger:
                for lease in expired:
                    ledger.insert_lease_event_locked({
                        "event_seq": self.last_event_seq,
                        "lease_id": lease.lease_id,
                        "shard_id": lease.shard_id,
                        "node_id": lease.node_id,
                        "kind": "expire_reclaim",
                        "from_version": lease.version,
                        "to_version": lease.version,
                        "expires_monotonic_ms": lease.expires_monotonic_ms,
                    })
                    self._upsert_terminal_from_engine_locked(ledger, lease.shard_id, "expired", {"code": "E_TIMEOUT", "message": "lease expired by monotonic clock"}, lease)
                self._insert_watermarks_locked(ledger, self.last_event_seq, "expiry")
        return expired

    def _run_shard(self, shard_id: str, lease: Lease) -> None:
        self.worker_threads[shard_id] = threading.current_thread().name
        started_ms = monotonic_ms()
        try:
            from pathlib import Path

            from .determinism import write_sample_artifact

            node = self.spec.node_map[lease.node_id]
            shard = self.spec.shard_map[shard_id]
            options = self.dispatch_options.get(shard_id, {})
            batch_size = int(options.get("batch_size") or self.spec.runtime.batch_size)
            samples = generate_samples(node.seed, shard_id, shard.sample_rows)
            sample_digest = __import__("shard_diag.canonical", fromlist=["sha256_hex"]).sha256_hex(
                __import__("shard_diag.determinism", fromlist=["samples_raw_bytes"]).samples_raw_bytes(samples)
            )
            self._write_sample(node, shard)
            batches = split_batches(samples, batch_size)
            partials: list[dict[str, Any]] = []
            renewed_once = False
            for index, batch in enumerate(batches):
                cancel_event = self.cancel_events.get(shard_id)
                if cancel_event is not None and cancel_event.is_set():
                    raise LeaseError("cancelled before batch", code="E_CANCELLED", object_id=shard_id)
                if not renewed_once and len(batches) >= 2 and index == 1:
                    self.renew_lease(shard_id, lease.version)
                    lease = self.registry.get_shard_lease(shard_id) or lease
                    renewed_once = True
                payload = {
                    "type": "aggregate_batch",
                    "shard_id": shard_id,
                    "quota_version": lease.quota_version,
                    "batch_index": index,
                    "batch_count": len(batches),
                    "rows": batch.tolist(),
                }
                if index == 0 and options.get("processing_delay_ms"):
                    payload["processing_delay_ms"] = int(options["processing_delay_ms"])
                if self.nodes is None:
                    raise DiagnosticError("node manager is unavailable", code="E_PORT")
                response = self.nodes.call(lease.node_id, payload, timeout_ms=max(self.spec.runtime.io_timeout_ms, int(options.get("processing_delay_ms", 0)) + 1000))
                require_fields(response, ["partial", "caliber_version"], shard_id)
                partials.append(response["partial"])
            from .determinism import combine_partials

            combined = combine_partials(partials)
            summary = build_summary(
                node_id=lease.node_id,
                shard_id=shard_id,
                sample_rows=shard.sample_rows,
                quota_version=lease.quota_version,
                partial=combined,
                fixed_point_scale=self.spec.runtime.fixed_point_scale,
            )
            with self.engine_lock:
                current = self.registry.get_shard_lease(shard_id)
                if current is None or current.state != LEASE_ACTIVE or current.version != lease.version and not renewed_once:
                    raise LeaseError("lease is no longer authoritative", object_id=shard_id)
                if current is not None:
                    lease = current
                if lease.state != LEASE_ACTIVE:
                    raise LeaseError("lease is not active", object_id=shard_id)
                used_before = self.registry.used_resource_units()
                before = self._water_snapshot()
                released = self.registry.release(
                    shard_id,
                    expected_version=lease.version,
                    node_call=(lambda nid, payload: self.nodes.call(nid, payload, timeout_ms=1000)) if self.nodes is not None else None,
                )
                if released is None or released.state != "released":
                    raise LeaseError("lease could not be released as authoritative", object_id=lease.lease_id)
                used_after = self.registry.used_resource_units()
                after = self._water_snapshot()
                wait_start = self.wait_started_ms.get(shard_id)
                wait_end = self.wait_ended_ms.get(shard_id)
                outcome = {
                    "shard_id": shard_id,
                    "node_id": lease.node_id,
                    "status": STATUS_COMPLETED,
                    "sample_rows": shard.sample_rows,
                    "participated_rows": summary["participated_rows"],
                    "sample_digest": sample_digest,
                    "aggregate_digest": summary["aggregate_digest"],
                    "aggregate_json": summary,
                    "caliber_version": summary["caliber_version"],
                    "quota_version": lease.quota_version,
                    "lease_id": lease.lease_id,
                    "lease_version": lease.version,
                    "resource_units": shard.resource_units,
                    "used_units_before": used_before,
                    "used_units_after": used_after,
                    "global_water_q": after["global"].aggregate_q,
                    "node_slot_water_q": after["nodes"][lease.node_id]["slot_water_q"],
                    "node_lease_water_q": after["nodes"][lease.node_id]["lease_water_q"],
                    "fixed_scale": self.spec.runtime.fixed_point_scale,
                    "quantization": {"before": before["dict"], "after": after["dict"]},
                    "event_seq": self.last_event_seq,
                    "wait_started_ms": wait_start,
                    "wait_ended_ms": wait_end,
                    "wait_duration_ms": None if wait_start is None or wait_end is None else max(0, wait_end - wait_start),
                    "started_ms": started_ms,
                    "finished_ms": monotonic_ms(),
                }
                with self.ledger.transaction() as ledger:
                    ledger.upsert_outcome_locked(outcome)
                    ledger.insert_lease_event_locked({
                        "event_seq": self.last_event_seq,
                        "lease_id": lease.lease_id,
                        "shard_id": shard_id,
                        "node_id": lease.node_id,
                        "kind": "release_complete",
                        "from_version": lease.version,
                        "to_version": lease.version,
                        "expires_monotonic_ms": lease.expires_monotonic_ms,
                    })
                    self._insert_watermarks_locked(ledger, self.last_event_seq, "shard_completed")
                    ledger.set_progress_locked(self.last_event_seq, "shard_completed")
                self.status[shard_id] = STATUS_COMPLETED
                self.finalized_terminal.add(shard_id)
        except Exception as exc:
            self._handle_worker_failure(shard_id, lease, exc)
        finally:
            try:
                self.pump(f"worker-finally:{shard_id}")
            except Exception:
                pass

    def _write_sample(self, node: Any, shard: Any) -> None:
        from pathlib import Path

        from .determinism import write_sample_artifact

        path = Path(self.samples_dir) / f"{node.node_id}__{shard.shard_id}.json"
        write_sample_artifact(path, node, shard)

    def _handle_worker_failure(self, shard_id: str, lease: Lease, exc: Exception) -> None:
        with self.engine_lock:
            if shard_id in self.finalized_terminal:
                return
            current = self.registry.get_shard_lease(shard_id) or lease
            code = getattr(exc, "code", "E_INTERNAL")
            if isinstance(exc, LeaseError):
                message = str(exc)
            elif isinstance(exc, FieldError):
                code = "E_FIELD"
                message = str(exc)
            elif isinstance(exc, DiagnosticError):
                message = str(exc)
            else:
                message = f"{type(exc).__name__}: {exc}"
            if self.cancel_events.get(shard_id) is not None and self.cancel_events[shard_id].is_set():
                status = STATUS_CANCELLED
                lease_state = LEASE_CANCELLED
            elif current.state == LEASE_REVOKED_ROLLBACK:
                status = STATUS_INVALIDATED
                lease_state = LEASE_REVOKED_ROLLBACK
            elif current.state == LEASE_EXPIRED:
                status = "expired"
                lease_state = LEASE_EXPIRED
            else:
                status = STATUS_FAILED
                lease_state = LEASE_FAILED
                self.registry.fail(shard_id, LEASE_FAILED)
            self.status[shard_id] = status
            error = {"code": code, "message": message, "details": getattr(exc, "details", {})}
            self._persist_simple_terminal(shard_id, status, error, current, lease_state=lease_state)

    def _persist_grant(self, lease: Lease, wait_start: int, wait_end: int) -> None:
        with self.ledger.transaction() as ledger:
            ledger.insert_lease_event_locked({
                "event_seq": self.last_event_seq,
                "lease_id": lease.lease_id,
                "shard_id": lease.shard_id,
                "node_id": lease.node_id,
                "kind": "grant",
                "from_version": None,
                "to_version": lease.version,
                "expires_monotonic_ms": lease.expires_monotonic_ms,
                "detail": {
                    "quota_version": lease.quota_version,
                    "granted_monotonic_ms": lease.granted_monotonic_ms,
                    "wait_started_ms": wait_start,
                    "wait_ended_ms": wait_end,
                },
            })
            self._insert_watermarks_locked(ledger, self.last_event_seq, "grant")

    def _persist_simple_terminal(self, shard_id: str, status: str, error: dict[str, Any], lease: Lease | None, lease_state: str | None = None) -> None:
        if shard_id in self.finalized_terminal and status not in {STATUS_BLOCKED}:
            return
        self.finalized_terminal.add(shard_id)
        shard = self.spec.shard_map.get(shard_id)
        with self.ledger.transaction() as ledger:
            if lease is not None:
                kind = {
                    STATUS_CANCELLED: "cancel",
                    STATUS_INVALIDATED: "rollback_revoke",
                    "expired": "expire_reclaim",
                    STATUS_FAILED: "fail",
                    STATUS_BLOCKED: "dependency_block",
                }.get(status, "terminal")
                ledger.insert_lease_event_locked({
                    "event_seq": self.last_event_seq,
                    "lease_id": lease.lease_id,
                    "shard_id": shard_id,
                    "node_id": lease.node_id,
                    "kind": kind,
                    "from_version": lease.version,
                    "to_version": lease.version,
                    "expires_monotonic_ms": lease.expires_monotonic_ms,
                    "detail": {"error": error, "lease_state": lease_state or lease.state},
                })
            self._upsert_terminal_from_engine_locked(ledger, shard_id, status, error, lease)
            self._insert_watermarks_locked(ledger, self.last_event_seq, status)
            ledger.set_progress_locked(self.last_event_seq, status)

    def _persist_rollback(
        self,
        seq: int,
        event_data: dict[str, Any],
        impact: dict[str, Any],
        revoked: list[Lease],
        expired: list[Lease],
        invalid_set: set[str],
    ) -> None:
        with self.ledger.transaction() as ledger:
            for lease in revoked:
                ledger.insert_lease_event_locked({
                    "event_seq": seq,
                    "lease_id": lease.lease_id,
                    "shard_id": lease.shard_id,
                    "node_id": lease.node_id,
                    "kind": "rollback_revoke",
                    "from_version": lease.version,
                    "to_version": lease.version,
                    "expires_monotonic_ms": lease.expires_monotonic_ms,
                    "detail": {"target_version": event_data["target_version"], "state": "revoked_rollback"},
                })
            for lease in expired:
                ledger.insert_lease_event_locked({
                    "event_seq": seq,
                    "lease_id": lease.lease_id,
                    "shard_id": lease.shard_id,
                    "node_id": lease.node_id,
                    "kind": "expire_reclaim",
                    "from_version": lease.version,
                    "to_version": lease.version,
                    "expires_monotonic_ms": lease.expires_monotonic_ms,
                    "detail": {"adjudicated_in": "rollback", "unique_attribution": "expired_before_rollback"},
                })
            for shard_id in sorted(invalid_set):
                existing = ledger.get_outcome(shard_id)
                error = {"code": "E_ROLLBACK_INVALIDATED", "target_version": event_data["target_version"], "impact": impact}
                if existing is not None:
                    outcome = dict(existing)
                    outcome["status"] = STATUS_INVALIDATED
                    outcome["error"] = error
                    if existing.get("quantization_json"):
                        try:
                            outcome["quantization"] = json.loads(existing["quantization_json"])
                        except json.JSONDecodeError:
                            outcome["quantization"] = None
                    ledger.upsert_outcome_locked(outcome)
                else:
                    lease = self.registry.get_shard_lease(shard_id)
                    self._upsert_terminal_from_engine_locked(ledger, shard_id, STATUS_INVALIDATED, error, lease)
                self.finalized_terminal.add(shard_id)
            self._insert_watermarks_locked(ledger, seq, "rollback")
            ledger.set_progress_locked(seq, "rollback_applied")

    def _upsert_terminal_from_engine_locked(self, ledger: Ledger, shard_id: str, status: str, error: dict[str, Any], lease: Lease | None) -> None:
        shard = self.spec.shard_map.get(shard_id)
        wait_start = self.wait_started_ms.get(shard_id)
        wait_end = self.wait_ended_ms.get(shard_id)
        outcome = {
            "shard_id": shard_id,
            "node_id": lease.node_id if lease else self.dispatch_node.get(shard_id),
            "status": status,
            "sample_rows": shard.sample_rows if shard else None,
            "resource_units": shard.resource_units if shard else None,
            "quota_version": lease.quota_version if lease else None,
            "lease_id": lease.lease_id if lease else None,
            "lease_version": lease.version if lease else None,
            "wait_started_ms": wait_start,
            "wait_ended_ms": wait_end,
            "wait_duration_ms": None if wait_start is None or wait_end is None else max(0, wait_end - wait_start),
            "finished_ms": monotonic_ms(),
            "error": error,
            "event_seq": self.last_event_seq,
        }
        ledger.upsert_outcome_locked(outcome)

    def _water_snapshot(self) -> dict[str, Any]:
        allocations = self.registry.active_allocations()
        attribution = global_budget_attribution(
            [(shard_id, units) for shard_id, units, _node in allocations],
            self.spec.global_budget,
            self.spec.runtime.fixed_point_scale,
        )
        nodes: dict[str, dict[str, Any]] = {}
        for node in self.spec.nodes:
            active = self.registry.active_leases_on_node(node.node_id)
            nodes[node.node_id] = node_water_report(
                node_id=node.node_id,
                active_shards=len(active),
                active_leases=len(active),
                concurrency_quota=node.concurrency_quota,
                max_leases=node.max_leases_per_node,
                scale=self.spec.runtime.fixed_point_scale,
                participant_ids=[lease.shard_id for lease in active],
            )
        comparisons: list[dict[str, Any]] = []
        node_ids = sorted(nodes)
        for index, node_id in enumerate(node_ids):
            for other_id in node_ids[index + 1 :]:
                cmp_slot = compare_fixed_water(nodes[node_id]["slot_water_q"], nodes[other_id]["slot_water_q"])
                comparisons.append({"a": node_id, "b": other_id, "dimension": "slot", **cmp_slot})
        return {"global": attribution, "nodes": nodes, "node_comparisons": comparisons, "dict": {"global": attribution.to_dict(), "nodes": nodes, "comparisons": comparisons}}

    def _insert_watermarks_locked(self, ledger: Ledger, event_seq: int, reason: str) -> None:
        snapshot = self._water_snapshot()
        global_report = snapshot["global"]
        ledger.insert_watermark_locked({
            "event_seq": event_seq,
            "scope_type": "global",
            "scope_id": "$budget",
            "used_units": global_report.used_units,
            "denominator": self.spec.global_budget,
            "scale": self.spec.runtime.fixed_point_scale,
            "water_q": global_report.aggregate_q,
            "report": {"reason": reason, **global_report.to_dict()},
            "active_shards": sum(report["active_shards"] for report in snapshot["nodes"].values()),
            "active_leases": sum(report["active_leases"] for report in snapshot["nodes"].values()),
        })
        for node_id, report in snapshot["nodes"].items():
            ledger.insert_watermark_locked({
                "event_seq": event_seq,
                "scope_type": "node",
                "scope_id": node_id,
                "used_units": None,
                "active_shards": report["active_shards"],
                "active_leases": report["active_leases"],
                "denominator": self.spec.node_map[node_id].concurrency_quota,
                "scale": self.spec.runtime.fixed_point_scale,
                "water_q": report["slot_water_q"],
                "report": {"reason": reason, **report},
            })

    def _persist_watermarks(self, event_seq: int, reason: str) -> None:
        with self.ledger.transaction() as ledger:
            self._insert_watermarks_locked(ledger, event_seq, reason)

    def final_summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for status in self.status.values():
            counts[status] = counts.get(status, 0) + 1
        return {
            "status_counts": counts,
            "dispatched_shards": len(self.dispatch_node),
            "admission_decisions": len(self.admission_log),
            "worker_threads": dict(sorted(self.worker_threads.items())),
            "quota_versions": dict(sorted(self.quota_versions.items())),
            "active_leases": sum(1 for lease in self.registry.leases.values() if lease.state == LEASE_ACTIVE),
        }


def recover_incomplete_shards(ledger: Ledger) -> dict[str, Any]:
    outcomes = {row["shard_id"] for row in ledger.outcomes()}
    dispatched: set[str] = set()
    granted: set[str] = set()
    for row in ledger.journal():
        try:
            event = json.loads(row["event_json"])
        except json.JSONDecodeError:
            continue
        if event.get("type") == "dispatch":
            dispatched.add(event.get("shard_id"))
    for row in ledger.lease_events():
        if row["kind"] == "grant" and row["shard_id"]:
            granted.add(row["shard_id"])
    incomplete = sorted((dispatched | granted) - outcomes)
    unknown: list[str] = []
    with ledger.transaction():
        for shard_id in incomplete:
            ledger.upsert_recovery_locked(shard_id, "unknown", {"action": "rerun_required", "basis": "no_atomic_terminal_shard_outcome"})
            unknown.append(shard_id)
        ledger.set_progress_locked(ledger.progress()["last_event_seq"] if ledger.progress() else 0, "recovered_unknown")
    return {"unknown_shards": unknown, "rerun_required": unknown, "committed_outcomes": sorted(outcomes)}
