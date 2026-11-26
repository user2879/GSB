from __future__ import annotations

import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from .accounting import build_shard_accounting, candidate_error_vector, global_error_report
from .canonical import canonical_dumps
from .errors import QuotaLabError
from .fixedpoint import quantize
from .leases import LeaseRegistry, monotonic_ms, new_lease, ownership_view
from .spec import SessionSpec


class Scheduler:
    def __init__(
        self,
        spec: SessionSpec,
        samples: dict[str, list[list[int]]],
        cluster: Any,
        *,
        ledger: Any | None = None,
        recovered: list[dict[str, Any]] | None = None,
        enable_kill_switch: bool = True,
    ) -> None:
        self.spec = spec
        self.samples = samples
        self.cluster = cluster
        self.ledger = ledger
        self.recovered = recovered or []
        self.enable_kill_switch = enable_kill_switch
        self.attempt_numbers: dict[str, int] = {}
        if ledger is not None:
            for record in ledger.records():
                if record.get("state") != "completed":
                    self.attempt_numbers[str(record["shard_id"])] = int(record["attempt"]) + 1
        self.accounting = {
            shard.shard_id: build_shard_accounting(spec, shard.shard_id, samples[shard.shard_id])
            for shard in spec.shards
        }
        self.leases = LeaseRegistry(spec)
        self.completed: set[str] = set(ledger.completed_shard_ids()) if ledger else set()
        self.running: dict[str, dict[str, Any]] = {}
        self.wait_started: dict[str, int] = {}
        self.wait_records: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.lease_transitions: list[dict[str, Any]] = []
        self.run_errors: list[dict[str, Any]] = []
        self.completion_order: list[str] = []
        self.completed_count = len(self.completed)
        self.peak_active_demand_q = 0
        self.peak_active_shards_by_node: dict[str, int] = {node.node_id: 0 for node in spec.nodes}
        self.peak_active_leases_by_node: dict[str, int] = {node.node_id: 0 for node in spec.nodes}
        self._recorded_block_keys: set[str] = set()

    def run(self) -> dict[str, Any]:
        target = [sid for sid in self.spec.topo_order if sid not in self.completed]
        with ThreadPoolExecutor(max_workers=max(1, self.spec.worker_count)) as pool:
            future_map: dict[Any, str] = {}
            idle_deadline: float | None = None
            while True:
                expired = self.leases.reap_expired()
                for lease in expired:
                    self.lease_transitions.append({"transition": "expired", **lease})

                if not self.run_errors:
                    admissions = self._admit_available()
                    for admission in admissions:
                        sid = admission["shard_id"]
                        self.running[sid] = admission
                        future = pool.submit(self._execute, admission)
                        future_map[future] = sid

                if not future_map:
                    if self.run_errors:
                        break
                    remaining = [sid for sid in target if sid not in self.completed]
                    if not remaining:
                        break
                    if not self.running:
                        self._record_deadlock(remaining)
                        break
                    time.sleep(0.01)
                    continue

                done, _ = wait(list(future_map), timeout=0.02, return_when=FIRST_COMPLETED)
                for future in done:
                    sid = future_map.pop(future)
                    self._handle_finished_future(future, sid)

        return self._build_result(target)

    def _ready_shards(self) -> list[str]:
        ready: list[str] = []
        now = monotonic_ms()
        for sid in self.spec.topo_order:
            if sid in self.completed or sid in self.running:
                continue
            shard = self.spec.shard(sid)
            if all(dep in self.completed for dep in shard.depends_on):
                if sid not in self.wait_started:
                    self.wait_started[sid] = now
                ready.append(sid)
        return ready

    def advance_capacity_version(self, node_id: str, new_version: int) -> dict[str, Any]:
        node = self.spec.node(node_id)
        node_response = None
        if self.cluster is not None:
            node_response = self.cluster.request(
                node_id,
                {
                    "op": "advance_version",
                    "node_id": node_id,
                    "new_capacity_version": new_version,
                },
                timeout_ms=self.spec.socket_timeout_ms,
            )
        voided = self.leases.void_capacity_version(node_id, new_version)
        for lease in voided:
            self.lease_transitions.append({"transition": "version_void", **lease})
        return {
            "node_id": node_id,
            "old_capacity_version": node.capacity_version,
            "new_capacity_version": new_version,
            "node_revoked_leases": (node_response or {}).get("revoked_leases", []),
            "parent_voided_leases": voided,
            "active_lease_count_after": self.leases.active_count(node_id),
        }

    def _evaluate_candidate(self, sid: str, active_ids: set[str], active_demand_q: int) -> dict[str, Any]:
        shard = self.spec.shard(sid)
        node = self.spec.node(shard.node_id)
        node_active_shards = sum(1 for active_sid in self.running if self.spec.shard(active_sid).node_id == node.node_id)
        node_active_leases = self.leases.active_count(node.node_id)
        gap_q = self.spec.budget_q - active_demand_q - shard.resource_units * self.spec.q
        reasons: list[str] = []
        if node_active_shards >= node.concurrency_quota:
            reasons.append("node_concurrency_quota")
        if node_active_leases >= node.lease_limit:
            reasons.append("node_lease_limit")
        if gap_q < 0:
            reasons.append("global_budget_exact_integer")
        vector = candidate_error_vector(self.spec, self.accounting, active_ids, sid)
        budget_indistinguishable = abs(gap_q) <= int(vector["composite_bound_q"])
        return {
            "shard_id": sid,
            "node_id": node.node_id,
            "topological_rank": self.accounting[sid]["topo_rank"],
            "stable_order_key": [self.accounting[sid]["topo_rank"], sid],
            "demand_q": shard.resource_units * self.spec.q,
            "budget_gap_q": gap_q,
            "hard_feasible": not reasons,
            "hard_block_reasons": reasons,
            "budget_within_synthetic_error_bound": budget_indistinguishable,
            "error_vector": vector,
            "watermark_q": vector["candidate_watermark_q"],
            "node_judgement_bound_q": vector["candidate_node_judgement_bound_q"],
        }

    def _admit_available(self) -> list[dict[str, Any]]:
        admissions: list[dict[str, Any]] = []
        while True:
            active_ids = set(self.running)
            active_demand_q = sum(int(self.accounting[sid]["demand_q"]) for sid in active_ids)
            ready = self._ready_shards()
            if not ready:
                return admissions
            evaluations = [self._evaluate_candidate(sid, active_ids, active_demand_q) for sid in ready]
            feasible = [item for item in evaluations if item["hard_feasible"]]
            if not feasible:
                self._record_queue_decision(evaluations)
                return admissions

            selected = feasible[0]
            runner_up = feasible[1] if len(feasible) > 1 else None
            pair_decision = None
            error_affected = bool(selected["budget_within_synthetic_error_bound"])
            if runner_up is not None:
                difference_q = abs(int(selected["watermark_q"]) - int(runner_up["watermark_q"]))
                pair_bound_q = (
                    int(selected["node_judgement_bound_q"])
                    + int(runner_up["node_judgement_bound_q"])
                )
                indistinguishable = difference_q <= pair_bound_q
                error_affected = error_affected or indistinguishable
                pair_decision = {
                    "candidate_a_shard_id": selected["shard_id"],
                    "candidate_b_shard_id": runner_up["shard_id"],
                    "watermark_a_q": selected["watermark_q"],
                    "watermark_b_q": runner_up["watermark_q"],
                    "difference_q": difference_q,
                    "pair_error_bound_q": pair_bound_q,
                    "indistinguishable": indistinguishable,
                    "rule": "if indistinguishable, release lexicographic minimum [topological_rank, shard_id]",
                    "winner_shard_id": selected["shard_id"],
                    "winner_order_key": selected["stable_order_key"],
                    "runner_up_order_key": runner_up["stable_order_key"],
                    "error_vector_a": selected["error_vector"],
                    "error_vector_b": runner_up["error_vector"],
                }

            now = monotonic_ms()
            sid = selected["shard_id"]
            wait_started = self.wait_started.get(sid, now)
            wait_record = {
                "shard_id": sid,
                "node_id": selected["node_id"],
                "wait_started_ms": wait_started,
                "wait_ended_ms": now,
                "wait_ms": max(0, now - wait_started),
                "error_affected": error_affected,
            }
            self.wait_records.append(wait_record)

            lease = new_lease(self.spec, sid)
            self.leases.add(lease)
            admission = {
                "shard_id": sid,
                "lease": lease,
                "wait": wait_record,
                "decision": {
                    "decision_id": f"admit-{len(self.decisions) + 1}",
                    "at_ms": now,
                    "released_shard_id": sid,
                    "error_affected": error_affected,
                    "selection_rule": "stable_topological_rank_then_shard_id",
                    "topological_order": list(self.spec.topo_order),
                    "selected": selected,
                    "runner_up": runner_up,
                    "watermark_pair": pair_decision,
                    "budget": {
                        "global_budget_q": self.spec.budget_q,
                        "active_demand_before_q": active_demand_q,
                        "candidate_demand_q": selected["demand_q"],
                        "remaining_minus_demand_q": selected["budget_gap_q"],
                        "synthetic_error_bound_q": selected["error_vector"]["composite_bound_q"],
                        "indistinguishable": selected["budget_within_synthetic_error_bound"],
                        "hard_integer_release": True,
                        "error_robust_release": selected["budget_gap_q"]
                        > int(selected["error_vector"]["composite_bound_q"]),
                    },
                    "wait": wait_record,
                },
            }
            self.decisions.append(admission["decision"])
            admissions.append(admission)
            self.running[sid] = admission

            self.peak_active_demand_q = max(
                self.peak_active_demand_q,
                active_demand_q + int(selected["demand_q"]),
            )
            node_id = selected["node_id"]
            node_shard_count = sum(
                1 for running_sid in self.running if self.spec.shard(running_sid).node_id == node_id
            )
            self.peak_active_shards_by_node[node_id] = max(
                self.peak_active_shards_by_node[node_id],
                node_shard_count,
            )
            self.peak_active_leases_by_node[node_id] = max(
                self.peak_active_leases_by_node[node_id],
                self.leases.active_count(node_id),
            )

            if self.ledger is not None:
                self.ledger.begin_attempt(
                    shard_id=sid,
                    node_id=node_id,
                    topo_rank=int(self.accounting[sid]["topo_rank"]),
                    attempt=int(self.attempt_numbers.get(sid, 1)),
                    demand_units=self.spec.shard(sid).resource_units,
                    demand_q=int(selected["demand_q"]),
                    wait_started_ms=wait_started,
                    wait_ended_ms=now,
                    started_ms=now,
                    lease=lease,
                    lease_ownership=ownership_view(lease),
                )

    def _record_queue_decision(self, evaluations: list[dict[str, Any]]) -> None:
        signature = tuple((item["shard_id"], tuple(item["hard_block_reasons"])) for item in evaluations[:3])
        key = canonical_dumps(signature)
        if key in self._recorded_block_keys:
            return
        self._recorded_block_keys.add(key)
        now = monotonic_ms()
        self.decisions.append(
            {
                "decision_id": f"queue-{len(self.decisions) + 1}",
                "at_ms": now,
                "queued": True,
                "selection_rule": "stable_topological_rank_then_shard_id",
                "candidates": evaluations,
                "first_candidate": evaluations[0] if evaluations else None,
            }
        )

    def _record_deadlock(self, remaining: list[str]) -> None:
        self.run_errors.append(
            {
                "code": "E_QUOTA_PROGRESS",
                "message": "no runnable worker and no admission; check zero quotas or failed dependencies",
                "remaining": remaining,
            }
        )

    def _execute(self, admission: dict[str, Any]) -> dict[str, Any]:
        sid = admission["shard_id"]
        lease = admission["lease"]
        node_id = lease["node_id"]
        self.cluster.request(
            node_id,
            {
                "op": "lease_grant",
                "node_id": node_id,
                "lease_id": lease["lease_id"],
                "shard_id": sid,
            },
        )
        for _ in range(self.spec.shard(sid).planned_renewals):
            lease = self.leases.renew_optimistic(self.cluster, lease)
        node = self.spec.node(node_id)
        receipt = self.cluster.request(
            node_id,
            {
                "op": "aggregate",
                "node_id": node_id,
                "shard_id": sid,
                "capacity_version": node.capacity_version,
                "samples": self.samples[sid],
            },
        )
        return {"receipt": receipt, "lease": lease}

    def _handle_finished_future(self, future: Any, sid: str) -> None:
        admission = self.running.pop(sid)
        try:
            result = future.result()
        except Exception as exc:
            error = exc.to_dict() if isinstance(exc, QuotaLabError) else {"code": "E_INTERNAL", "message": str(exc)}
            self.run_errors.append({"shard_id": sid, **error})
            terminal = self.leases.terminalize(admission["lease"]["lease_id"], "failed", "execution_exception")
            if terminal:
                self.lease_transitions.append({"transition": "failed", **terminal})
            return

        receipt = result["receipt"]
        lease = result["lease"]
        expected_summary = self.accounting[sid]["summary"]
        if canonical_dumps(receipt["summary"]) != canonical_dumps(expected_summary):
            mismatch = QuotaLabError("node summary does not match deterministic NumPy recomputation", code="E_DIGEST")
            self.run_errors.append({"shard_id": sid, **mismatch.to_dict()})
            return

        finished_ms = monotonic_ms()
        terminal = self.leases.terminalize(lease["lease_id"], "completed", "shard_outcome_committed")
        if terminal:
            self.lease_transitions.append({"transition": "completed", **terminal})
        ownership = ownership_view(lease)
        occupancy = {
            "state": "completed",
            "resource_units": self.spec.shard(sid).resource_units,
            "demand_q": self.accounting[sid]["demand_q"],
            "peak_session_active_demand_q": self.peak_active_demand_q,
            "active_q_after_commit": 0,
        }
        watermark = {
            key: value
            for key, value in self.accounting[sid].items()
            if key not in {"summary"}
        }
        error_attribution = {
            "shard_id": sid,
            "local_trunc_bound_q": self.accounting[sid]["local_trunc_bound_q"],
            "watermark_calculation": self.accounting[sid]["watermark_calculation"],
            "mean_fixed_point": expected_summary["fixed_point"],
            "decision_ref": admission["decision"]["decision_id"],
        }
        if self.ledger is not None:
            self.ledger.commit_outcome(
                shard_id=sid,
                lease=lease,
                lease_ownership=ownership,
                occupancy=occupancy,
                watermark=watermark,
                error_attribution=error_attribution,
                summary=expected_summary,
                receipt={key: value for key, value in receipt.items() if key != "summary"},
                wait_started_ms=admission["wait"]["wait_started_ms"],
                wait_ended_ms=admission["wait"]["wait_ended_ms"],
                finished_ms=finished_ms,
                last_completed_shard=sid,
                completed_count=self.completed_count + 1,
                cursor_payload={
                    "last_completed_shard": sid,
                    "completed_order": self.completion_order + [sid],
                    "at_ms": finished_ms,
                },
            )
        self.completed.add(sid)
        self.completed_count += 1
        self.completion_order.append(sid)

        kill_threshold = self.spec.fault_injection.kill_after_commits
        if self.enable_kill_switch and kill_threshold is not None and self.completed_count >= kill_threshold:
            # Durable commit has returned; SIGKILL models an external force termination.
            os.kill(os.getpid(), signal.SIGKILL)

    def _build_result(self, target: list[str]) -> dict[str, Any]:
        error_report = global_error_report(self.spec, self.accounting)
        ledger_verify = self.ledger.verify_digest() if self.ledger is not None else None
        aggregate_error_bound_q = int(
            error_report["node_watermarks"]["global"]["aggregate_budget_error_bound_q"]
        )
        node_constraints_held = all(
            self.peak_active_shards_by_node[node.node_id] <= node.concurrency_quota
            and self.peak_active_leases_by_node[node.node_id] <= node.lease_limit
            for node in self.spec.nodes
        )
        constraint_check = {
            "peak_active_demand_q": self.peak_active_demand_q,
            "global_budget_q": self.spec.budget_q,
            "budget_never_exceeded_exact_integer": self.peak_active_demand_q <= self.spec.budget_q,
            "aggregate_error_bound_q": aggregate_error_bound_q,
            "error_extended_budget_limit_q": self.spec.budget_q + aggregate_error_bound_q,
            "within_budget_plus_error_bound_limit": self.peak_active_demand_q
            <= self.spec.budget_q + aggregate_error_bound_q,
            "conservative_peak_plus_error_bound_within_budget": self.peak_active_demand_q
            + aggregate_error_bound_q
            <= self.spec.budget_q,
            "node_constraints_held": node_constraints_held,
            "peak_active_shards_by_node": self.peak_active_shards_by_node,
            "peak_active_leases_by_node": self.peak_active_leases_by_node,
            "node_concurrency_quotas": {node.node_id: node.concurrency_quota for node in self.spec.nodes},
            "node_lease_limits": {node.node_id: node.lease_limit for node in self.spec.nodes},
            "quantized_limits_q": {
                "q": self.spec.q,
                "rounding_rule": "integer_limits_are_multiplied_by_q_exactly_before_any_division",
                "global_budget_units": self.spec.global_budget_units,
                "global_budget_q": self.spec.budget_q,
                "peak_active_resource_usage_q": self.peak_active_demand_q,
                "nodes": {
                    node.node_id: {
                        "concurrency_quota_q": quantize(node.concurrency_quota, self.spec.q),
                        "lease_limit_q": quantize(node.lease_limit, self.spec.q),
                        "peak_active_shard_usage_q": self.peak_active_shards_by_node[node.node_id] * self.spec.q,
                        "peak_active_lease_usage_q": self.peak_active_leases_by_node[node.node_id] * self.spec.q,
                    }
                    for node in self.spec.nodes
                },
            },
        }
        return {
            "ok": not self.run_errors and len(self.completed) >= len([s for s in self.spec.shards]),
            "session_id": self.spec.session_id,
            "spec_digest": self.spec.digest(),
            "completed_shard_ids": list(self.completion_order),
            "recovered": self.recovered,
            "wait_records": self.wait_records,
            "decisions": self.decisions,
            "lease_transitions": [
                {k: v for k, v in transition.items() if k not in {"granted_monotonic_ms"}}
                for transition in self.lease_transitions
            ],
            "error_report": error_report,
            "constraint_check": constraint_check,
            "ledger_verify": ledger_verify,
            "run_errors": self.run_errors,
            "target_count": len(target),
        }
