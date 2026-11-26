from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from .clock import monotonic_ms
from .determinism import compute_shard_summary, merge_partial_summaries, partition_ranges
from .diagnostics import water_for_shard
from .errors import DiagnosticError
from .leases import Lease, LeaseManager
from .ledger import Ledger
from .models import ShardSpec, Spec
from .topology import stable_topological_order
from .water import (
    budget_slack_assessment,
    compare_candidate_water,
    global_budget_error_bound,
)


WorkRunner = Callable[["ShardRuntime", Lease | None], dict[str, Any] | None]


@dataclass(slots=True)
class ShardRuntime:
    spec: ShardSpec
    topo_index: int
    status: str = "PENDING"
    wait_started_ms: int | None = None
    wait_ended_ms: int | None = None
    wait_ms: int | None = None
    attempts: int = 0
    reservation_held: bool = False
    lease: Lease | None = None
    result: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    water: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class DispatchScheduler:
    def __init__(
        self,
        spec: Spec,
        *,
        clients: dict[str, Any] | None = None,
        ledger: Ledger | None = None,
        lease_manager: LeaseManager | None = None,
        work_runner: WorkRunner | None = None,
        max_workers: int | None = None,
        processing_delay_ms: int = 0,
    ) -> None:
        self.spec = spec
        self.clients = clients or {}
        self.ledger = ledger
        self.work_runner = work_runner
        self.processing_delay_ms = processing_delay_ms
        self.topo = stable_topological_order(spec.shards)
        self.topo_rank = {shard_id: index for index, shard_id in enumerate(self.topo)}
        self.states = {
            shard.shard_id: ShardRuntime(spec=shard, topo_index=self.topo_rank[shard.shard_id])
            for shard in spec.shards
        }
        node_map = {node.node_id: node for node in spec.nodes}
        self.leases = lease_manager or LeaseManager(
            node_map, self.clients, default_ttl_ms=spec.lease_ttl_ms
        )
        worker_count = max(1, sum(node.concurrency_quota for node in spec.nodes))
        self.executor_workers = max_workers or min(32, worker_count + 4)

        self._lock = threading.RLock()
        self._global_resources = 0
        self._node_shards: dict[str, int] = {node.node_id: 0 for node in spec.nodes}
        self._node_leases: dict[str, int] = {node.node_id: 0 for node in spec.nodes}
        self.peak_global_resources = 0
        self.peak_node_shards = {node.node_id: 0 for node in spec.nodes}
        self.peak_node_leases = {node.node_id: 0 for node in spec.nodes}
        self.water_comparisons: list[dict[str, Any]] = []
        self.budget_assessments: list[dict[str, Any]] = []

    def preload_committed(self) -> "DispatchScheduler":
        if self.ledger is None:
            return self
        for payload in self.ledger.outcomes():
            shard_id = payload["shard_id"]
            if shard_id in self.states:
                state = self.states[shard_id]
                state.status = "SUCCEEDED"
                state.result = payload
        return self

    def mark_succeeded(self, shard_id: str, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            state = self.states[shard_id]
            state.status = "SUCCEEDED"
            state.result = result

    def reset_for_rerun(self, shard_ids: list[str] | set[str] | tuple[str, ...]) -> None:
        with self._lock:
            for shard_id in shard_ids:
                state = self.states[shard_id]
                if state.status != "SUCCEEDED":
                    state.status = "PENDING"
                    state.error = None
                    state.wait_started_ms = None
                    state.wait_ended_ms = None
                    state.wait_ms = None

    def occupancy_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global_resources": self._global_resources,
                "node_shards": dict(self._node_shards),
                "node_leases": dict(self._node_leases),
                "peak_global_resources": self.peak_global_resources,
                "peak_node_shards": dict(self.peak_node_shards),
                "peak_node_leases": dict(self.peak_node_leases),
            }

    def run(self, target_ids: set[str] | None = None) -> dict[str, Any]:
        targets = set(target_ids) if target_ids is not None else set(self.states)
        with ThreadPoolExecutor(
            max_workers=self.executor_workers, thread_name_prefix="shard-dispatch"
        ) as executor:
            in_flight: dict[Any, str] = {}
            while True:
                self._promote_dependencies(targets)
                reserved = self._reserve_ready()
                for shard_id in reserved:
                    future = executor.submit(self._safe_execute, shard_id)
                    in_flight[future] = shard_id

                if not in_flight:
                    blocked = self._finalize_unfinished()
                    if blocked or not reserved:
                        break
                    continue

                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED, timeout=0.05)
                for future in done:
                    shard_id = in_flight.pop(future)
                    future.result()

        return self.summary()

    def _promote_dependencies(self, targets: set[str]) -> None:
        terminal_failure = {"FAILED", "BLOCKED", "DEPENDENCY_FAILED", "REVOKED", "UNKNOWN"}
        with self._lock:
            for shard_id in self.topo:
                if shard_id not in targets:
                    continue
                state = self.states[shard_id]
                if state.status != "PENDING":
                    continue
                dependency_states = [self.states[dep].status for dep in state.spec.dependencies]
                if any(status in terminal_failure for status in dependency_states):
                    state.status = "DEPENDENCY_FAILED"
                    state.error = {
                        "code": "E_DEPENDENCY",
                        "message": "an upstream dependency did not complete",
                    }
                elif all(status == "SUCCEEDED" for status in dependency_states):
                    state.status = "QUEUED"
                    state.wait_started_ms = monotonic_ms()

    def _hard_capacity_available_locked(self, shard: ShardSpec) -> bool:
        node = self.spec.node(shard.node_id)
        return (
            self._global_resources + shard.resource_units <= self.spec.global_resource_budget
            and self._node_shards[node.node_id] < node.concurrency_quota
            and self._node_leases[node.node_id] < node.lease_limit
        )

    def _choose_next_locked(self) -> tuple[str, Any, dict[str, Any]] | None:
        candidates: list[tuple[str, Any]] = []
        for shard_id in self.topo:
            state = self.states[shard_id]
            if state.status != "QUEUED":
                continue
            shard = state.spec
            if not self._hard_capacity_available_locked(shard):
                continue
            rearrangements = self.leases.rearrangements(shard.node_id)
            water = water_for_shard(
                self.spec, shard_id, lease_rearrangements=rearrangements
            )
            if not water.revision_allowed or water.gate == "deny":
                continue
            candidates.append((shard_id, water))

        if not candidates:
            return None

        bound = global_budget_error_bound(
            shard_count=len(self.spec.shards),
            scale_power=self.spec.scale_power,
            jitter_bounds_q=[
                water_for_shard(
                    self.spec, sid, lease_rearrangements=self.leases.rearrangements(
                        self.states[sid].spec.node_id
                    )
                ).jitter_bound_q
                for sid in self.topo
            ],
            lease_bounds_q=[
                water_for_shard(
                    self.spec, sid, lease_rearrangements=self.leases.rearrangements(
                        self.states[sid].spec.node_id
                    )
                ).lease_bound_q
                for sid in self.topo
            ],
        )
        remaining = self.spec.global_resource_budget - self._global_resources
        best_id, best_water = candidates[0]
        for shard_id, water in candidates[1:]:
            comparison = compare_candidate_water(
                best_water,
                water,
                self.topo_rank[best_id],
                self.topo_rank[shard_id],
            )
            comparison_view = comparison.to_dict()
            best_demand = self.states[best_id].spec.resource_units
            candidate_demand = self.states[shard_id].spec.resource_units
            best_budget = budget_slack_assessment(
                remaining_units=remaining,
                demand_units=best_demand,
                scale_power=self.spec.scale_power,
                global_error_bound_q=bound["safe_upper_bound_q"],
            )
            candidate_budget = budget_slack_assessment(
                remaining_units=remaining,
                demand_units=candidate_demand,
                scale_power=self.spec.scale_power,
                global_error_bound_q=bound["safe_upper_bound_q"],
            )
            budget_boundary = bool(
                best_budget["indistinguishable"] or candidate_budget["indistinguishable"]
            )
            stable_winner = (
                best_id
                if (self.topo_rank[best_id], best_id)
                <= (self.topo_rank[shard_id], shard_id)
                else shard_id
            )
            winner = (
                stable_winner
                if comparison.indistinguishable or budget_boundary
                else comparison.winner
            )
            comparison_view["budget_boundary"] = budget_boundary
            comparison_view["winner"] = winner
            self.water_comparisons.append(comparison_view)
            if winner == shard_id:
                best_id, best_water = shard_id, water

        shard = self.states[best_id].spec
        budget = budget_slack_assessment(
            remaining_units=remaining,
            demand_units=shard.resource_units,
            scale_power=self.spec.scale_power,
            global_error_bound_q=bound["safe_upper_bound_q"],
        )
        self.budget_assessments.append(budget)
        decision = {
            "selected_shard_id": best_id,
            "selection_rule": "hard capacity first; significant lower water wins; otherwise stable topology rank then shard_id",
            "water": best_water.to_dict(),
            "global_budget_error": bound,
            "budget_slack": budget,
            "error_affected": bool(best_water.error_affected or budget["indistinguishable"]),
        }
        return best_id, best_water, decision

    def _reserve_ready(self) -> list[str]:
        reserved: list[str] = []
        with self._lock:
            while True:
                choice = self._choose_next_locked()
                if choice is None:
                    break
                shard_id, water, decision = choice
                state = self.states[shard_id]
                shard = state.spec
                if not self._hard_capacity_available_locked(shard):
                    break

                now = monotonic_ms()
                state.status = "RUNNING"
                state.wait_ended_ms = now
                state.wait_ms = (
                    now - state.wait_started_ms if state.wait_started_ms is not None else 0
                )
                state.attempts += 1
                state.reservation_held = True
                state.water = water.to_dict()
                state.decision = decision

                self._global_resources += shard.resource_units
                node_id = shard.node_id
                self._node_shards[node_id] += 1
                self._node_leases[node_id] += 1
                self.peak_global_resources = max(
                    self.peak_global_resources, self._global_resources
                )
                self.peak_node_shards[node_id] = max(
                    self.peak_node_shards[node_id], self._node_shards[node_id]
                )
                self.peak_node_leases[node_id] = max(
                    self.peak_node_leases[node_id], self._node_leases[node_id]
                )
                reserved.append(shard_id)
        return reserved

    def _safe_execute(self, shard_id: str) -> None:
        state = self.states[shard_id]
        lease: Lease | None = None
        try:
            node = self.spec.node(state.spec.node_id)
            lease = self.leases.acquire(
                shard_id,
                node.node_id,
                node.capacity_version,
                ttl_ms=self.spec.lease_ttl_ms,
            )
            state.lease = lease
            if self.ledger is not None:
                self.ledger.record_occupancy(lease, state.spec.resource_units)
            self._execute(state, lease)
        except Exception as exc:
            error = exc.to_dict() if isinstance(exc, DiagnosticError) else {
                "code": "E_DISPATCH",
                "message": f"{type(exc).__name__}: {exc}",
            }
            with self._lock:
                state.status = "UNKNOWN" if isinstance(exc, DiagnosticError) and exc.code in {
                    "E_NODE",
                    "E_CAPACITY",
                } else "FAILED"
                state.error = error
            if lease is not None:
                self.leases.release(lease.lease_id)
                if self.ledger is not None:
                    self.ledger.set_occupancy_status(lease.lease_id, "reclaimed")
        finally:
            with self._lock:
                if state.reservation_held:
                    shard = state.spec
                    self._global_resources -= shard.resource_units
                    self._node_shards[shard.node_id] -= 1
                    self._node_leases[shard.node_id] -= 1
                    state.reservation_held = False

    def _call_node(self, state: ShardRuntime, lease: Lease) -> dict[str, Any]:
        shard = state.spec
        node = self.spec.node(shard.node_id)
        client = self.clients[node.node_id]
        ranges = partition_ranges(shard.sample_rows)
        partials: list[dict[str, Any]] = []
        sample_digest: str | None = None
        last_response: dict[str, Any] = {}
        for offset, row_count in ranges:
            response = client.aggregate(
                {
                    "lease_id": lease.lease_id,
                    "lease_version": lease.lease_version,
                    "shard_id": shard.shard_id,
                    "capacity_version": node.capacity_version,
                    "primary_revision": max(self.spec.revision_chain),
                    "sample_rows": shard.sample_rows,
                    "row_offset": offset,
                    "row_count": row_count,
                    "processing_delay_ms": self.processing_delay_ms,
                }
            )
            if sample_digest is None:
                sample_digest = response["sample_digest"]
            elif sample_digest != response["sample_digest"]:
                raise DiagnosticError("node returned inconsistent sample digest")
            partials.append(response["chunk"])
            last_response = response
        summary = merge_partial_summaries(node, shard, sample_digest or "", partials)
        return {
            "summary": summary.to_dict(),
            "node_response": {
                "capacity_version": last_response.get("capacity_version"),
                "primary_revision": last_response.get("primary_revision"),
                "replayed_revision": last_response.get("replayed_revision"),
                "elapsed_monotonic_ms": last_response.get("elapsed_monotonic_ms"),
                "batch_count": len(ranges),
            },
        }

    def _execute(self, state: ShardRuntime, lease: Lease) -> None:
        work_result: dict[str, Any] = {}
        if self.work_runner is not None:
            returned = self.work_runner(state, lease)
            if isinstance(returned, dict):
                work_result = returned
        else:
            work_result = self._call_node(state, lease)

        shard = state.spec
        node = self.spec.node(shard.node_id)
        summary = work_result.get("summary")
        if summary is None:
            summary = compute_shard_summary(node, shard).to_dict()
        node_response = work_result.get("node_response", {})

        released_lease = self.leases.release(lease.lease_id)
        lease_view = released_lease.to_dict()
        water = water_for_shard(
            self.spec,
            shard.shard_id,
            lease_rearrangements=self.leases.rearrangements(node.node_id),
        ).to_dict()
        decision = dict(state.decision or {})
        payload = {
            "shard_id": shard.shard_id,
            "node_id": node.node_id,
            "capacity_version": node.capacity_version,
            "status": "COMMITTED",
            "summary": summary,
            "lease": lease_view,
            "resource_units": shard.resource_units,
            "resource_occupancy": {
                "resource_units": shard.resource_units,
                "active": False,
                "released": True,
            },
            "water": water,
            "error_attribution": water["error_segments_q"],
            "decision": decision,
            "wait": {
                "wait_started_ms": state.wait_started_ms,
                "wait_ended_ms": state.wait_ended_ms,
                "wait_ms": state.wait_ms,
            },
            "cursor": {
                "shard_id": shard.shard_id,
                "topo_index": state.topo_index,
                "position": state.topo_index + 1,
            },
            "node_response": node_response,
        }
        if self.ledger is not None:
            committed = self.ledger.commit_outcome(payload)
            payload["content_digest"] = committed["content_digest"]

        with self._lock:
            state.status = "SUCCEEDED"
            state.result = payload
            state.error = None

    def _finalize_unfinished(self) -> list[str]:
        finalized: list[str] = []
        with self._lock:
            for shard_id in self.topo:
                state = self.states[shard_id]
                if state.status == "QUEUED":
                    water = water_for_shard(
                        self.spec,
                        shard_id,
                        lease_rearrangements=self.leases.rearrangements(state.spec.node_id),
                    )
                    state.water = water.to_dict()
                    if not water.revision_allowed or water.gate == "deny":
                        state.status = "BLOCKED"
                        state.error = {
                            "code": "E_WATER",
                            "message": "replication revision/delay gate denied dispatch",
                            "water": water.to_dict(),
                        }
                    else:
                        state.status = "FAILED"
                        state.error = {
                            "code": "E_SCHED",
                            "message": "queued shard could not satisfy capacity constraints",
                            "occupancy": self.occupancy_snapshot(),
                        }
                    finalized.append(shard_id)
                elif state.status == "PENDING":
                    state.status = "DEPENDENCY_FAILED"
                    finalized.append(shard_id)
        return finalized

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for state in self.states.values():
            counts[state.status] = counts.get(state.status, 0) + 1
        return {
            "counts": counts,
            "occupancy": self.occupancy_snapshot(),
            "water_comparisons": self.water_comparisons,
            "budget_assessments": self.budget_assessments,
            "shards": {
                shard_id: {
                    "status": state.status,
                    "wait_ms": state.wait_ms,
                    "attempts": state.attempts,
                    "error": state.error,
                }
                for shard_id, state in self.states.items()
            },
        }
