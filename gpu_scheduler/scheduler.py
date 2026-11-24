from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .diagnostics import JsonLogger
from .graph import DependencyGraph
from .hashing import canonical_text, digest_obj
from .ledger import Ledger
from .quant import build_watermarks
from .spec import Spec


VICTIM_RULE = (
    "priority_asc,then_submitted_monotonic_ms_desc,then_completed_shard_count_asc,"
    "then_task_id_asc;local_pool_constraints_before_global_quota"
)


class SchedulerEngine:
    def __init__(
        self,
        spec: Spec,
        workdir: Path | str,
        *,
        executor_manager: Any | None = None,
        logger: JsonLogger | None = None,
        global_gpu_quota: int | None = None,
        fail_shards: set[str] | None = None,
        retry_shards: set[str] | None = None,
        max_wall_seconds: float = 60.0,
        view_writer: Any | None = None,
    ):
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or JsonLogger(self.workdir / "logs" / "scheduler.jsonl")
        self.ledger = Ledger(self.workdir / "ledger.sqlite3")
        self.ledger.initialize_meta(spec)
        self.graph = DependencyGraph.build(spec)
        self.executors = executor_manager
        self.global_gpu_quota = global_gpu_quota or spec.global_gpu_quota
        if self.global_gpu_quota <= 0:
            raise ValueError("global_gpu_quota must be positive")
        self.fail_shards = set(fail_shards or ())
        self.retry_shards = set(retry_shards or ())
        self.max_wall_seconds = max_wall_seconds
        self.view_writer = view_writer

        self.started_ns = time.monotonic_ns()
        self.pass_started_ms = 0
        self.round_index = self.ledger.head_round
        self.sequence = 0
        self.stop_requested = False
        self.finished_reason: str | None = None
        self.lock = threading.RLock()
        self.wake_event = threading.Event()
        self.workers = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(8, min(32, sum(p.gpu_count for p in spec.pools) * 2 + 4)),
            thread_name_prefix="shard-client",
        )
        self.in_flight: dict[concurrent.futures.Future[Any], str] = {}
        self.active: dict[str, dict[str, Any]] = {}
        self.pending_events: list[dict[str, Any]] = []
        self.pending_preemptions: list[dict[str, Any]] = []
        self.pending_suspensions: list[dict[str, Any]] = []
        self.pending_cursors: list[dict[str, Any]] = []
        self.pending_receipts: list[dict[str, Any]] = []
        self.pending_propagations: list[dict[str, Any]] = []
        self.pending_interrupt_attempt_ids: list[int] = []
        self.crash_ready_path: Path | None = None
        self.crash_ready_written = False

        self.shards: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self._fresh_state()
        self.recovery_report = self.recover_interrupted()
        if self.retry_shards:
            self.apply_outcome_changes(self.retry_shards, reason="explicit_retry_after_outcome_change")

    def close(self) -> None:
        self.workers.shutdown(wait=False, cancel_futures=False)
        self.ledger.close()

    def _mono_ms(self) -> int:
        return max(0, (time.monotonic_ns() - self.started_ns) // 1_000_000)

    def _next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def _fresh_state(self) -> None:
        for task in self.spec.tasks:
            self.tasks[task.id] = {
                "task": task,
                "admitted": task.admission_after_running is None,
                "ever_started": False,
                "suspend_requested": False,
                "suspended": False,
                "completed": set(),
                "lease_id": f"lease-{task.id}-{uuid.uuid4().hex[:12]}",
                "lease_expires_wall_ms": int(time.time() * 1000) + 30_000,
                "renew_count": 0,
            }
            for shard_id in task.shard_ids:
                index = int(shard_id.rsplit("#", 1)[1])
                self.shards[shard_id] = {
                    "shard_id": shard_id,
                    "task_id": task.id,
                    "index": index,
                    "status": "pending",
                    "queue_start_ms": None,
                    "queue_end_ms": None,
                    "receipt_id": None,
                    "receipt_digest": None,
                    "dispatch_count": 0,
                    "unknown_reason": None,
                }

    def _event(self, shard_id: str, phase: str, **fields: Any) -> None:
        task_id = shard_id.split("#", 1)[0]
        shard_state = self.shards.get(shard_id, {})
        event = {
            "shard_id": shard_id,
            "task_id": task_id,
            "phase": phase,
            "sequence_no": self._next_sequence(),
            "monotonic_ms": self._mono_ms(),
            "queue_start_ms": shard_state.get("queue_start_ms"),
            "queue_end_ms": shard_state.get("queue_end_ms"),
            "payload": fields.pop("payload", {}),
            **fields,
        }
        self.pending_events.append(event)

    def recover_interrupted(self) -> dict[str, Any]:
        verification = self.ledger.verify_digests()
        meta_spec = self.ledger.get_meta("spec_digest")
        if meta_spec and meta_spec != self.spec.spec_digest:
            from .errors import SnapshotDigestError

            raise SnapshotDigestError(
                "ledger spec digest does not match current declaration; use snapshot whole-replacement restore",
                locator={"field": "spec_digest", "ledger": meta_spec, "current": self.spec.spec_digest},
            )

        latest_phase: dict[str, str] = {}
        for row in self.ledger.conn.execute("SELECT shard_id,phase FROM shard_events ORDER BY id"):
            latest_phase[row["shard_id"]] = row["phase"]

        latest_receipts = self.ledger.latest_receipts()
        unknown: set[str] = set()
        interrupt_ids: list[int] = []
        for attempt in self.ledger.open_attempts():
            interrupt_ids.append(int(attempt["attempt_id"]))
            for shard_id in json.loads(attempt["shard_ids_json"]):
                if shard_id in self.shards and shard_id not in latest_receipts:
                    unknown.add(shard_id)
                    self.shards[shard_id]["unknown_reason"] = "open_attempt_after_forced_exit"

        for shard_id, phase in latest_phase.items():
            if shard_id not in self.shards:
                continue
            if shard_id in latest_receipts:
                continue
            if phase in {"dispatched", "running"}:
                unknown.add(shard_id)
                self.shards[shard_id]["unknown_reason"] = f"last_committed_phase_{phase}_without_receipt"

        for shard_id, row in latest_receipts.items():
            if shard_id not in self.shards:
                continue
            receipt = json.loads(row["canonical_receipt_json"])
            state = self.shards[shard_id]
            state["receipt_id"] = row["receipt_id"]
            state["receipt_digest"] = row["receipt_digest"]
            if receipt.get("status") == "success":
                state["status"] = "completed"
                self.tasks[state["task_id"]]["completed"].add(shard_id)
            else:
                state["status"] = "failed"

        for row in self.ledger.conn.execute("SELECT * FROM cursor_rows"):
            task_id = row["task_id"]
            if task_id in self.tasks:
                completed = set(json.loads(row["completed_shards_json"]))
                self.tasks[task_id]["completed"].update(s for s in completed if s in self.shards)

        recovered_suspensions: list[str] = []
        for row in self.ledger.conn.execute("SELECT * FROM suspension_rows WHERE state IN ('suspend_requested','suspended')"):
            task_id = row["task_id"]
            if task_id not in self.tasks:
                continue
            task_state = self.tasks[task_id]
            task_state["suspend_requested"] = False
            task_state["suspended"] = True
            task_state["lease_id"] = f"recovery-{uuid.uuid4().hex[:12]}"
            task_state["renew_count"] = int(row["renew_count"]) + 1
            task_state["lease_expires_wall_ms"] = int(time.time() * 1000) + 30_000
            recovered_suspensions.append(task_id)
            self.pending_suspensions.append(
                self._suspension_payload(
                    task_id,
                    "recovery_renewed",
                    int(row["occupied_gpu"]),
                    int(row["occupied_memory_mib"]),
                    "forced_exit_recovered_from_sqlite_cursor",
                )
            )

        for shard_id in sorted(unknown):
            state = self.shards[shard_id]
            state["status"] = "unknown"
            self._event(
                shard_id,
                "unknown",
                payload={"reason": state["unknown_reason"], "rerun_required": True},
            )

        report = {
            "ledger_verification": verification,
            "unknown_shards": sorted(unknown),
            "recovered_suspended_tasks": sorted(recovered_suspensions),
            "head_round": self.head_round,
        }
        if unknown or recovered_suspensions:
            self.pending_interrupt_attempt_ids = interrupt_ids
            self._commit_pending_round(recovery=True)
        else:
            # No uncertain attempt remains; still close open attempt markers if any
            # somehow had receipts (defensive path).
            if interrupt_ids:
                self.ledger.conn.execute("BEGIN IMMEDIATE")
                for attempt_id in interrupt_ids:
                    self.ledger.conn.execute(
                        "UPDATE attempts SET state='interrupted',updated_wall_ms=? WHERE attempt_id=?",
                        (int(time.time() * 1000), attempt_id),
                    )
                self.ledger.conn.execute("COMMIT")
        return report

    @property
    def head_round(self) -> int:
        return self.ledger.head_round

    def apply_outcome_changes(self, source_ids: set[str] | list[str], *, reason: str) -> dict[str, Any]:
        sources = {sid for sid in source_ids if sid in self.shards}
        affected = self.graph.descendants(sources)
        statuses = {sid: self.shards[sid]["status"] for sid in affected if sid in self.shards}
        trace = self.graph.propagation_trace(sources, statuses=statuses, details={sid: reason for sid in sources})
        for shard_id in affected:
            if shard_id not in self.shards:
                continue
            state = self.shards[shard_id]
            state["status"] = "pending"
            state["queue_start_ms"] = None
            state["queue_end_ms"] = None
            state["unknown_reason"] = None
            if shard_id in sources:
                state["dispatch_count"] = 0
        record = {
            "trigger_type": "outcome_changed",
            "reason": reason,
            "trigger_shards": sorted(sources),
            "affected": sorted(affected),
            "trace": [step.to_dict() for step in trace],
            "reschedule_set": sorted(affected),
        }
        self.pending_propagations.append(record)
        self.logger.event(
            "WARNING",
            "outcome_change_propagated",
            trigger_shards=sorted(sources),
            affected=sorted(affected),
        )
        return record

    def _incoming_ready(self, shard_id: str) -> tuple[bool, list[str]]:
        deps = sorted(self.graph.incoming.get(shard_id, set()))
        if not deps:
            return True, []
        if any(self.shards.get(dep, {}).get("status") in {"failed", "blocked", "skipped", "unschedulable"} for dep in deps):
            return False, [dep for dep in deps if self.shards.get(dep, {}).get("status") != "completed"]
        return all(self.shards.get(dep, {}).get("status") == "completed" for dep in deps), deps

    def _usage_maps(self, *, removed_tasks: set[str] | None = None) -> tuple[dict[str, int], dict[str, int], dict[str, set[str]]]:
        removed = removed_tasks or set()
        gpu = {pool.id: 0 for pool in self.spec.pools}
        memory = {pool.id: 0 for pool in self.spec.pools}
        tasks_by_pool: dict[str, set[str]] = {pool.id: set() for pool in self.spec.pools}
        for shard_id, assignment in self.active.items():
            if assignment["task_id"] in removed:
                continue
            gpu[assignment["pool_id"]] += assignment["gpu_count"]
            memory[assignment["pool_id"]] += assignment["memory_mib"]
            tasks_by_pool[assignment["pool_id"]].add(assignment["task_id"])
        return gpu, memory, tasks_by_pool

    def _build_watermarks(self, removed_tasks: set[str] | None = None) -> dict[str, Any]:
        gpu, memory, tasks_by_pool = self._usage_maps(removed_tasks=removed_tasks)
        active_task_counts = {pool_id: len(ids) for pool_id, ids in tasks_by_pool.items()}
        return build_watermarks(
            self.spec,
            gpu,
            memory,
            active_task_counts,
            global_gpu_quota=self.global_gpu_quota,
        )

    def _occupancy_rows(self, watermarks: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        global_block = watermarks["global"]["gpu_quota"]
        global_memory = watermarks["global"]["memory_mib"]
        rows.append(
            {
                "scope": "global",
                "pool_id": "__global__",
                "used_gpu": global_block["used"],
                "used_memory_mib": global_memory["used"],
                "gpu_capacity": global_block["capacity"],
                "memory_capacity_mib": global_memory["capacity"],
            }
        )
        for pool in watermarks["pools"]:
            rows.append(
                {
                    "scope": "pool",
                    "pool_id": pool["id"],
                    "used_gpu": pool["gpu_physical"]["used"],
                    "used_memory_mib": pool["memory_mib"]["used"],
                    "gpu_capacity": pool["gpu_physical"]["capacity"],
                    "memory_capacity_mib": pool["memory_mib"]["capacity"],
                }
            )
        return rows

    def _candidate_sort_key(self, shard_id: str) -> tuple[int, int, str, int]:
        task = self.spec.task_by_id[self.shards[shard_id]["task_id"]]
        return (-task.priority, task.submitted_monotonic_ms, task.id, self.shards[shard_id]["index"])

    def _can_fit(self, shard_id: str, *, removed_tasks: set[str] | None = None) -> bool:
        state = self.shards[shard_id]
        task = self.spec.task_by_id[state["task_id"]]
        pool = self.spec.pool_by_id[task.pool]
        gpu_used, mem_used, tasks_by_pool = self._usage_maps(removed_tasks=removed_tasks)
        removed = removed_tasks or set()
        global_used = sum(gpu_used.values())
        task_active = any(tid == task.id for tid in tasks_by_pool[pool.id]) and task.id not in removed
        return (
            global_used + task.gpu_count <= self.global_gpu_quota
            and gpu_used[pool.id] + task.gpu_count <= min(pool.gpu_count, pool.quota_share)
            and mem_used[pool.id] + task.memory_per_shard_mib <= pool.memory_capacity_mib
            and (task_active or len(tasks_by_pool[pool.id]) < pool.max_concurrent_tasks)
        )

    def _victim_key(self, task_id: str) -> tuple[int, int, int, str]:
        task_state = self.tasks[task_id]
        task = task_state["task"]
        return (
            task.priority,
            -task.submitted_monotonic_ms,
            len(task_state["completed"]),
            task.id,
        )

    def _choose_victims(self, candidate_shard: str) -> list[str] | None:
        candidate_task_id = self.shards[candidate_shard]["task_id"]
        candidate_task = self.spec.task_by_id[candidate_task_id]
        active_task_ids = {assignment["task_id"] for assignment in self.active.values()}
        eligible = [
            task_id
            for task_id in active_task_ids
            if self.spec.task_by_id[task_id].priority < candidate_task.priority
        ]
        selected: list[str] = []

        def local_ok() -> bool:
            return self._can_fit(candidate_shard, removed_tasks=set(selected))

        while not local_ok():
            choices = sorted(
                (tid for tid in eligible if tid not in selected and self.spec.task_by_id[tid].pool == candidate_task.pool),
                key=self._victim_key,
            )
            if not choices:
                return None
            selected.append(choices[0])

        def global_ok() -> bool:
            gpu_used, _, _ = self._usage_maps(removed_tasks=set(selected))
            return sum(gpu_used.values()) + candidate_task.gpu_count <= self.global_gpu_quota

        while not global_ok():
            choices = sorted((tid for tid in eligible if tid not in selected), key=self._victim_key)
            if not choices:
                return None
            selected.append(choices[0])
        return selected

    def _suspension_payload(
        self,
        task_id: str,
        event_type: str,
        occupied_gpu: int,
        occupied_memory: int,
        reason: str,
    ) -> dict[str, Any]:
        state = self.tasks[task_id]
        now_wall = int(time.time() * 1000)
        if event_type == "lease_renewed":
            state["renew_count"] += 1
        state["lease_expires_wall_ms"] = now_wall + 30_000
        task = state["task"]
        completed = sorted(state["completed"])
        incomplete = [sid for sid in task.shard_ids if sid not in state["completed"]]
        return {
            "task_id": task_id,
            "pool_id": task.pool,
            "state": state["suspended"] if isinstance(state.get("suspended"), str) else ("suspended" if state["suspended"] else "suspend_requested"),
            "event_type": event_type,
            "completed_shards": completed,
            "cursor": {"next_shard_id": incomplete[0] if incomplete else None, "incomplete_shards": incomplete},
            "occupied_gpu": occupied_gpu,
            "occupied_memory_mib": occupied_memory,
            "reason": reason,
            "lease_id": state["lease_id"],
            "lease_expires_wall_ms": state["lease_expires_wall_ms"],
            "renew_count": state["renew_count"],
            "recovery_path": f"sqlite://{self.workdir / 'ledger.sqlite3'}#suspension_rows/cursor_rows",
            "monotonic_ms": self._mono_ms(),
        }

    def _request_preemption(self, candidate_shard: str, victims: list[str]) -> bool:
        high_task_id = self.shards[candidate_shard]["task_id"]
        now = self._mono_ms()
        victim_set = sorted(victims, key=self._victim_key)
        newly_requested: list[str] = []
        for victim_id in victim_set:
            state = self.tasks[victim_id]
            if state["suspend_requested"]:
                continue
            newly_requested.append(victim_id)
            state["suspend_requested"] = True
            task_usage_gpu = sum(a["gpu_count"] for sid, a in self.active.items() if a["task_id"] == victim_id)
            task_usage_mem = sum(a["memory_mib"] for sid, a in self.active.items() if a["task_id"] == victim_id)
            payload = self._suspension_payload(
                victim_id,
                "preempt_requested",
                task_usage_gpu,
                task_usage_mem,
                f"higher_priority_task_waiting:{high_task_id}",
            )
            payload["state"] = "suspend_requested"
            self.pending_suspensions.append(payload)
            preemption = {
                "high_task_id": high_task_id,
                "victim_task_id": victim_id,
                "victim_set": victim_set,
                "selection_rule": VICTIM_RULE,
                "reason": "quota_memory_or_concurrency_unavailable_for_higher_priority",
                "released_gpu": task_usage_gpu,
                "released_memory_mib": task_usage_mem,
                "monotonic_ms": now,
            }
            self.pending_preemptions.append(preemption)
            self.logger.event("WARNING", "preemption_requested", **preemption)
        if newly_requested:
            source_shards: list[str] = []
            statuses: dict[str, str] = {}
            for victim_id in newly_requested:
                task = self.spec.task_by_id[victim_id]
                completed = self.tasks[victim_id]["completed"]
                incomplete = [sid for sid in task.shard_ids if sid not in completed]
                source_shards.extend(incomplete)
                for sid in incomplete:
                    statuses[sid] = "preempted_at_shard_boundary"
            affected = self.graph.descendants(source_shards)
            trace = self.graph.propagation_trace(
                source_shards,
                statuses=statuses,
                details={sid: "lower_priority_task_preempted" for sid in source_shards},
            )
            self.pending_propagations.append(
                {
                    "trigger_type": "preempted",
                    "reason": f"higher_priority_task_waiting:{high_task_id}",
                    "trigger_shards": sorted(source_shards),
                    "preempted_tasks": sorted(newly_requested),
                    "affected": sorted(affected),
                    "trace": [step.to_dict() for step in trace],
                    "reschedule_set": sorted(affected),
                }
            )
            return True
        return False

    def _admit_tasks(self) -> bool:
        changed = False
        active_or_ever = {tid for tid, state in self.tasks.items() if state["ever_started"] or self.active and any(a["task_id"] == tid for a in self.active.values())}
        for task in sorted(self.spec.tasks, key=lambda t: (t.submitted_monotonic_ms, t.id)):
            state = self.tasks[task.id]
            if not state["admitted"] and task.admission_after_running:
                target = task.admission_after_running
                if self.tasks[target]["ever_started"] or target in active_or_ever:
                    state["admitted"] = True
                    self._event(task.shard_id(0), "admitted", payload={"trigger": task.admission_after_running})
                    changed = True
        return changed

    def _dispatch_shard(self, shard_id: str) -> None:
        state = self.shards[shard_id]
        task = self.spec.task_by_id[state["task_id"]]
        pool = self.spec.pool_by_id[task.pool]
        state["status"] = "running"
        state["queue_end_ms"] = self._mono_ms()
        state["dispatch_count"] += 1
        self.tasks[task.id]["ever_started"] = True
        if self.tasks[task.id]["suspended"]:
            self.tasks[task.id]["suspended"] = False
            payload = self._suspension_payload(task.id, "resumed", 0, 0, "resources_available_resume_from_cursor")
            payload["state"] = "resumed"
            self.pending_suspensions.append(payload)
        seed = int(digest_obj([self.spec.spec_digest, shard_id])[:16], 16)
        attempt_id = self.ledger.begin_attempt(self.head_round + 1, [shard_id])
        compute_ms = task.estimated_duration_ms
        if task.duration_overrides_ms and 0 <= state["index"] < len(task.duration_overrides_ms):
            compute_ms = task.duration_overrides_ms[state["index"]]
        request = {
            "batch_id": f"{self.head_round + 1}-{shard_id}-{state['dispatch_count']}",
            "shard_id": shard_id,
            "attempt_id": attempt_id,
            "seed": seed,
            "compute_ms": compute_ms,
            "gpu_count": task.gpu_count,
            "memory_mib": task.memory_per_shard_mib,
            "integer_units": pool.integer_units_per_gpu * task.gpu_count,
            "force_failure": shard_id in self.fail_shards,
        }
        assignment = {
            "shard_id": shard_id,
            "task_id": task.id,
            "pool_id": pool.id,
            "attempt_id": attempt_id,
            "gpu_count": task.gpu_count,
            "memory_mib": task.memory_per_shard_mib,
            "dispatched_monotonic_ms": self._mono_ms(),
        }
        self.active[shard_id] = assignment
        timeout = max(30.0, min(180.0, compute_ms / 1000.0 + 20.0))
        future = self.workers.submit(
            self.executors.client_execute_shard,
            pool,
            request,
            timeout_seconds=timeout,
        )
        future.add_done_callback(lambda _future: self.wake_event.set())
        self.in_flight[future] = shard_id
        self._event(shard_id, "dispatched", payload={"request": request, "pool_id": pool.id})

    def _mark_failure_propagation(self, shard_id: str) -> None:
        affected = self.graph.descendants([shard_id])
        trace = self.graph.propagation_trace(
            [shard_id],
            statuses={sid: self.shards.get(sid, {}).get("status", "unknown") for sid in affected},
            details={shard_id: "execution_failed"},
        )
        for affected_id in sorted(affected - {shard_id}):
            state = self.shards.get(affected_id)
            if not state or state["status"] == "completed":
                # A completed downstream is stale after an upstream outcome change and
                # is handled by apply_outcome_changes. A fresh failure cannot have
                # allowed a dependent to complete before its dependency.
                continue
            state["status"] = "blocked"
            self._event(affected_id, "blocked", payload={"upstream_failure": shard_id})
        record = {
            "trigger_type": "execution_failed",
            "trigger_shards": [shard_id],
            "affected": sorted(affected),
            "trace": [step.to_dict() for step in trace],
            "reschedule_set": sorted(affected),
        }
        self.pending_propagations.append(record)

    def _process_done_futures(self) -> bool:
        changed = False
        done_futures = [future for future in list(self.in_flight) if future.done()]
        for future in done_futures:
            shard_id = self.in_flight.pop(future)
            assignment = self.active.pop(shard_id, {})
            changed = True
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {
                    "shard_id": shard_id,
                    "status": "failed",
                    "error_code": "E_EXECUTOR_TRANSPORT",
                    "error_message": str(exc),
                    "caliber": {"name": "transport-error", "version": 1},
                    "resource_usage": {
                        "gpu_count": self.spec.task_by_id[shard_id.split("#", 1)[0]].gpu_count,
                        "memory_mib": self.spec.task_by_id[shard_id.split("#", 1)[0]].memory_per_shard_mib,
                    },
                    "elapsed_monotonic_ms": 0,
                    "compute": None,
                }
                receipt["receipt_digest"] = digest_obj({k: v for k, v in receipt.items() if k != "receipt_digest"})
            canonical_receipt = canonical_text(receipt)
            receipt_digest = receipt.get("receipt_digest") or digest_obj(receipt)
            state = self.shards[shard_id]
            state["receipt_digest"] = receipt_digest
            self.pending_receipts.append(
                {
                    "shard_id": shard_id,
                    "canonical_receipt_json": canonical_receipt,
                    "receipt_digest": receipt_digest,
                    "outcome_status": receipt.get("status", "failed"),
                    "resource_usage": receipt.get("resource_usage", {}),
                    "created_monotonic_ms": self._mono_ms(),
                    "attempt_id": assignment.get("attempt_id"),
                }
            )
            if receipt.get("status") == "success":
                state["status"] = "completed"
                self.tasks[state["task_id"]]["completed"].add(shard_id)
                self._event(shard_id, "completed", receipt_digest=receipt_digest, payload={"elapsed_monotonic_ms": receipt.get("elapsed_monotonic_ms")})
            else:
                state["status"] = "failed"
                self._event(shard_id, "failed", receipt_digest=receipt_digest, payload={"error_code": receipt.get("error_code")})
                self._mark_failure_propagation(shard_id)
        return changed

    def _finalize_safe_suspensions(self) -> bool:
        changed = False
        active_task_ids = {assignment["task_id"] for assignment in self.active.values()}
        for task_id, state in sorted(self.tasks.items()):
            if state["suspend_requested"] and task_id not in active_task_ids:
                state["suspend_requested"] = False
                state["suspended"] = True
                task = state["task"]
                used_gpu = sum(
                    cursor.get("occupied_gpu", 0)
                    for cursor in []
                )
                payload = self._suspension_payload(
                    task_id,
                    "safe_boundary_suspended",
                    0,
                    0,
                    "all_in_flight_shards_reached_boundary",
                )
                # Preserve the resources held at preemption request time.
                prior = [p for p in reversed(self.pending_suspensions) if p["task_id"] == task_id]
                if prior:
                    payload["occupied_gpu"] = prior[-1].get("occupied_gpu", 0)
                    payload["occupied_memory_mib"] = prior[-1].get("occupied_memory_mib", 0)
                payload["state"] = "suspended"
                self.pending_suspensions.append(payload)
                self._event(task.shard_id(0), "suspended", payload={"completed_shards": payload["completed_shards"]})
                changed = True
        return changed

    def _schedule(self) -> bool:
        changed = False
        self.pending_cursors = self._cursor_bundles()
        candidate_ids = sorted(
            (sid for sid, state in self.shards.items() if state["status"] in {"pending", "queued", "unknown", "stale"}),
            key=self._candidate_sort_key,
        )
        for shard_id in candidate_ids:
            state = self.shards[shard_id]
            task_state = self.tasks[state["task_id"]]
            if not task_state["admitted"]:
                continue
            if task_state["suspend_requested"]:
                continue
            if state["status"] in {"completed", "failed", "blocked"}:
                continue
            ready, _deps = self._incoming_ready(shard_id)
            if not ready:
                if any(self.shards.get(dep, {}).get("status") in {"failed", "blocked"} for dep in self.graph.incoming.get(shard_id, set())):
                    if state["status"] not in {"blocked"}:
                        state["status"] = "blocked"
                        self._event(shard_id, "blocked", payload={"reason": "dependency_failed_or_blocked"})
                        changed = True
                continue
            if state["queue_start_ms"] is None:
                state["queue_start_ms"] = self._mono_ms()
                state["status"] = "queued"
                self._event(shard_id, "queued")
                changed = True
            if self._can_fit(shard_id):
                self._dispatch_shard(shard_id)
                changed = True
                continue

            victims = self._choose_victims(shard_id)
            if victims:
                changed = self._request_preemption(shard_id, victims) or changed
            elif not self.active:
                state["status"] = "unschedulable"
                self.finished_reason = "unschedulable"
                self._event(
                    shard_id,
                    "unschedulable",
                    payload={"reason": "requested_resources_exceed_quota_or_pool_capacity", "global_gpu_quota": self.global_gpu_quota},
                )
                changed = True
        if changed:
            self.pending_cursors = self._cursor_bundles()
        return changed

    def _cursor_bundles(self) -> list[dict[str, Any]]:
        bundles: list[dict[str, Any]] = []
        for task in sorted(self.spec.tasks, key=lambda t: t.id):
            state = self.tasks[task.id]
            completed = sorted(state["completed"])
            incomplete = [sid for sid in task.shard_ids if sid not in state["completed"]]
            bundles.append(
                {
                    "task_id": task.id,
                    "round_index": self.head_round + 1,
                    "next_shard_id": incomplete[0] if incomplete else None,
                    "completed_shards": completed,
                    "pending_shards": incomplete,
                }
            )
        return bundles

    def _renew_suspended_leases(self) -> bool:
        changed = False
        active_tasks = {a["task_id"] for a in self.active.values()}
        for task_id, state in sorted(self.tasks.items()):
            if state["suspended"] and task_id not in active_tasks:
                payload = self._suspension_payload(task_id, "lease_renewed", 0, 0, "waiting_for_resources_lease_renewal")
                payload["state"] = "suspended"
                self.pending_suspensions.append(payload)
                changed = True
        return changed

    def _commit_pending_round(self, *, recovery: bool = False) -> int:
        self.round_index += 1
        ended = self._mono_ms()
        watermarks = self._build_watermarks()
        bundle = {
            "round_index": self.round_index,
            "started_monotonic_ms": getattr(self, "pass_started_ms", max(0, ended - 1)),
            "ended_monotonic_ms": ended,
            "cursor": {"head_round": self.round_index, "sequence": self.sequence, "recovery": recovery},
            "watermarks": watermarks,
            "events": self.pending_events,
            "occupancy_rows": self._occupancy_rows(watermarks),
            "preemptions": self.pending_preemptions,
            "suspensions": self.pending_suspensions,
            "cursors": self.pending_cursors or self._cursor_bundles(),
            "receipts": self.pending_receipts,
            "propagations": self.pending_propagations,
            "interrupt_attempt_ids": self.pending_interrupt_attempt_ids,
            "commit_attempt_ids": [
                receipt.get("attempt_id")
                for receipt in self.pending_receipts
                if receipt.get("attempt_id") is not None
            ],
        }
        self.ledger.commit_round(bundle)
        if self.view_writer is not None:
            try:
                self.view_writer(self)
            except Exception as exc:
                self.logger.event("ERROR", "ui_view_write_failed", error=str(exc))
        self.pending_events = []
        self.pending_preemptions = []
        self.pending_suspensions = []
        self.pending_cursors = []
        self.pending_receipts = []
        self.pending_propagations = []
        self.pending_interrupt_attempt_ids = []
        return self.round_index

    def state_report(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        running: list[str] = []
        queued: list[str] = []
        unknown: list[str] = []
        for shard_id in sorted(self.shards):
            status = self.shards[shard_id]["status"]
            counts[status] = counts.get(status, 0) + 1
            if status == "running":
                running.append(shard_id)
            if status == "queued":
                queued.append(shard_id)
            if status == "unknown":
                unknown.append(shard_id)
        return {
            "round_index": self.head_round,
            "counts": dict(sorted(counts.items())),
            "running": running,
            "queued": queued,
            "unknown": unknown,
            "active": sorted(self.active),
            "preempted_tasks": sorted(tid for tid, s in self.tasks.items() if s["suspended"] or s["suspend_requested"]),
            "finished_reason": self.finished_reason,
        }

    def _all_terminal(self) -> bool:
        terminal = {"completed", "failed", "blocked", "skipped", "unschedulable"}
        return all(state["status"] in terminal for state in self.shards.values())

    def request_stop(self) -> None:
        self.stop_requested = True
        self.wake_event.set()

    def adjust_quota(self, global_gpu_quota: int) -> dict[str, Any]:
        if not isinstance(global_gpu_quota, int) or isinstance(global_gpu_quota, bool) or global_gpu_quota <= 0:
            raise ValueError("global_gpu_quota must be a positive integer")
        previous = self.global_gpu_quota
        self.global_gpu_quota = global_gpu_quota
        self._event(
            "__quota__#session",
            "quota_adjusted",
            payload={"previous_global_gpu_quota": previous, "global_gpu_quota": global_gpu_quota},
        )
        round_index = self._commit_pending_round()
        return {"ok": True, "round_index": round_index, "global_gpu_quota": global_gpu_quota}

    def run_until_complete(self) -> dict[str, Any]:
        if self.executors is None:
            raise RuntimeError("ExecutorManager is required for real loopback scheduling")
        deadline = time.monotonic() + self.max_wall_seconds
        while time.monotonic() < deadline:
            with self.lock:
                self.pass_started_ms = self._mono_ms()
                self.wake_event.clear()
                changed = False
                changed = self._process_done_futures() or changed
                changed = self._finalize_safe_suspensions() or changed
                changed = self._admit_tasks() or changed
                if not self.stop_requested:
                    changed = self._schedule() or changed
                    changed = self._renew_suspended_leases() or changed
                else:
                    self.finished_reason = self.finished_reason or "signal_stop_at_boundary"
                if changed or self.pending_events or self.pending_propagations:
                    self._commit_pending_round()
                if self.crash_ready_path and self.active and not self.crash_ready_written:
                    self.crash_ready_path.parent.mkdir(parents=True, exist_ok=True)
                    self.crash_ready_path.write_text(
                        canonical_text(
                            {
                                "pid": __import__("os").getpid(),
                                "round_index": self.head_round,
                                "active_shards": sorted(self.active),
                                "wall_ms": int(time.time() * 1000),
                            }
                        ),
                        encoding="utf-8",
                    )
                    self.crash_ready_written = True
                if self._all_terminal():
                    self.finished_reason = self.finished_reason or "all_terminal"
                    return self.state_report()
                if self.stop_requested and not self.in_flight:
                    return self.state_report()

            if self.in_flight:
                concurrent.futures.wait(
                    list(self.in_flight.keys()),
                    timeout=0.05,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            else:
                self.wake_event.wait(timeout=0.05)
        self.finished_reason = "deadline_exceeded"
        raise TimeoutError(f"scheduler did not finish within {self.max_wall_seconds}s: {self.state_report()}")
