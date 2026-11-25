from __future__ import annotations

from typing import Any

from .canonical import monotonic_ms, sha256_hex
from .ledger import Ledger
from .leases import Lease, LeaseManager
from .propagation import VersionedPlan
from .scheduler import ParallelScalingScheduler, chunks_for_allocation
from .spec import Spec


class RoundRunner:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger | None = None,
        plan: VersionedPlan | None = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.plan = plan or VersionedPlan(spec)
        service_replica_quotas = {
            service_id: (
                spec.budget.service_concurrency_quotas[service_id]
                // spec.service(service_id).concurrency_per_replica
            )
            for service_id in spec.service_ids
        }
        self.lease_manager = LeaseManager(
            spec.budget.global_replica_budget,
            service_replica_quotas,
            spec.lease_policy.lease_ttl_ms,
        )
        self.scheduler = ParallelScalingScheduler(
            self.lease_manager, spec.lease_policy
        )
        self.event_offset = 0
        self.last_round_seq: int | None = None

    def run_round(
        self,
        minute: int,
        *,
        round_id: str | None = None,
        batch_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not 0 <= minute < self.spec.minute_count:
            raise ValueError(f"minute {minute} is outside spec range")
        round_id = round_id or f"round-v{self.plan.version}-m{minute:05d}"
        batch_id = batch_id or f"batch-v{self.plan.version}-m{minute:05d}"
        rows = [
            self.plan.allocation_rows[(minute, service_id)]
            for service_id in self.spec.service_ids
        ]
        tasks = []
        for row in rows:
            tasks.extend(
                chunks_for_allocation(
                    round_id=round_id,
                    batch_id=batch_id,
                    minute=minute,
                    service_id=row.service_id,
                    allocated_replicas=row.allocated_replicas,
                    max_task_replicas=self.spec.lease_policy.max_task_replicas,
                )
            )

        started = monotonic_ms()
        results = self.scheduler.run_tasks(tasks)
        completed = monotonic_ms()
        all_events = self.lease_manager.events()
        round_events = all_events[self.event_offset :]
        self.event_offset = len(all_events)

        leases = [
            self.lease_manager.get_lease(result["lease_id"]) for result in results
        ]
        snapshot = self.lease_manager.snapshot()
        consumption: list[dict[str, Any]] = [
            {
                "scope": "global",
                "target": "__global_replica_budget__",
                "consumed": snapshot["global_held"],
                "remaining": snapshot["global_remaining"],
            }
        ]
        for service_id in self.spec.service_ids:
            service_state = snapshot["services"][service_id]
            concurrency_quota = self.spec.budget.service_concurrency_quotas[service_id]
            capacity = self.spec.service(service_id).concurrency_per_replica
            consumption.append(
                {
                    "scope": "service_concurrency",
                    "target": service_id,
                    "consumed": service_state["held"] * capacity,
                    "remaining": concurrency_quota
                    - service_state["held"] * capacity,
                    "replica_held": service_state["held"],
                    "replica_quota": service_state["quota"],
                    "replica_remaining": service_state["remaining"],
                }
            )

        minute_records = []
        for service_id in self.spec.service_ids:
            demand = self.plan.full_result.minutes[service_id][minute]
            allocation = self.plan.allocation_rows[(minute, service_id)]
            minute_records.append(
                {
                    "service_id": service_id,
                    "minute": minute,
                    "record_type": "demand",
                    "value": demand.comparable_dict(),
                }
            )
            minute_records.append(
                {
                    "service_id": service_id,
                    "minute": minute,
                    "record_type": "allocation",
                    "value": allocation.comparable_dict(),
                }
            )

        payload = {
            "round_id": round_id,
            "plan_version": self.plan.version,
            "minute": minute,
            "batch_id": batch_id,
            "spec": self.spec.canonical_dict(),
            "input_digest": sha256_hex(self.spec.canonical_bytes()),
            "selection": self.plan.selection.ledger_basis(),
            "started_monotonic_ms": started,
            "completed_monotonic_ms": completed,
            "scheduler_max_observed_inflight": self.scheduler.max_observed_inflight,
            "minute_records": minute_records,
            "leases": [lease.to_dict() for lease in leases],
            "events": round_events,
            "consumption": consumption,
            "wait_spans": self.scheduler.wait_spans,
            "cursor": {
                "last_minute": minute,
                "minute_count": self.spec.minute_count,
            },
        }
        commit_result = None
        if commit and self.ledger is not None:
            self.ledger.save_plan_version(self.plan)
            commit_result = self.ledger.commit_round(payload)
            self.last_round_seq = commit_result["round_seq"]
        return {
            "payload": payload,
            "commit": commit_result,
            "task_results": results,
            "wait_spans": self.scheduler.wait_spans,
            "leases": leases,
            "events": round_events,
            "quota_snapshot": snapshot,
            "max_observed_inflight": self.scheduler.max_observed_inflight,
        }

    def persist_new_lease_events(self) -> list[dict[str, Any]]:
        if self.ledger is None or self.last_round_seq is None:
            return []
        all_events = self.lease_manager.events()
        new_events = all_events[self.event_offset :]
        self.event_offset = len(all_events)
        if not new_events:
            return []
        with self.ledger._lock:
            conn = self.ledger.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                for event in new_events:
                    lease_id = event.get("lease_id")
                    if lease_id:
                        lease = self.lease_manager.get_lease(lease_id)
                        lease_dict = lease.to_dict()
                        conn.execute(
                            "UPDATE leases SET version=?,status=?,"
                            "expires_monotonic_ms=?,lease_json=? WHERE lease_id=?",
                            (
                                lease.version,
                                lease.status,
                                lease.expires_at_monotonic_ms,
                                __import__("json").dumps(lease_dict, sort_keys=True),
                                lease_id,
                            ),
                        )
                    conn.execute(
                        "INSERT INTO quota_events(round_seq,event_seq,event_type,"
                        "lease_id,service_id,effective,event_json) VALUES(?,?,?,?,?,?,?)",
                        (
                            self.last_round_seq,
                            event.get("event_seq"),
                            event.get("event_type", "event"),
                            event.get("lease_id"),
                            event.get("service_id"),
                            1 if event.get("effective") else 0,
                            __import__("json").dumps(event, sort_keys=True),
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return new_events
