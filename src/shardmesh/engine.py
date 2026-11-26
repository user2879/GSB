"""Parallel shard scheduling, lease lifecycle and transactional outcomes."""

from __future__ import annotations

import concurrent.futures
import queue
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from .capacity import CapacityState
from .cluster import Cluster
from .errors import ShardMeshError
from .graph import transitive_downstream
from .ledger import Ledger
from .leases import Lease, LeaseManager
from .samples import batch_digest
from .spec import Event, Spec
from .timeutil import now_ns
from .water import (
    build_attribution,
    budget_boundary,
    global_aggregation_bound,
    marginal_quantization,
    node_water_view,
    water_boundary,
)


class Engine:
    def __init__(
        self,
        spec: Spec,
        samples: dict[str, Any],
        ledger: Ledger,
        *,
        work_dir: str | Path = "artifacts/runtime",
        cluster: Cluster | None = None,
        manage_cluster: bool = True,
        max_workers: int | None = None,
        delay_overrides: dict[str, int] | None = None,
    ) -> None:
        self.spec = spec
        self.samples = samples.get("batches", samples) if isinstance(samples, dict) else {}
        self.ledger = ledger
        self.work_dir = Path(work_dir)
        self.manage_cluster = manage_cluster
        self.cluster = cluster or Cluster(
            spec.nodes,
            scale=spec.scale,
            log_dir=self.work_dir / "logs",
            timeout_ms=spec.request_timeout_ms,
        )
        self.capacity = CapacityState(spec.global_budget, spec.nodes)
        self.leases = LeaseManager(on_expire=self._on_lease_expired)
        self.current_nodes = dict(spec.node_map)
        self.budget = spec.global_budget
        self.scale = spec.scale
        self.max_renewals = spec.max_renewals
        self.lease_ttl_ms = spec.lease_ttl_ms
        self.delay_overrides = dict(delay_overrides or {})
        self.topo = spec.topology
        self.topo_index = {shard_id: index for index, shard_id in enumerate(self.topo)}
        self.deps_map = spec.deps_map
        self.arrived: set[str] = set()
        self.records: dict[str, dict[str, Any]] = {}
        self.futures: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}
        self.future_to_shard: dict[concurrent.futures.Future[Any], str] = {}
        self.failed: dict[str, ShardMeshError] = {}
        self.attempt: dict[str, int] = {shard.id: 0 for shard in spec.shards}
        self.parameter_revision: dict[str, int] = {shard.id: 1 for shard in spec.shards}
        self.wait_started_ns: dict[str, int] = {}
        self.admission_context: dict[str, dict[str, Any]] = {}
        self.invalidation_plans: list[dict[str, Any]] = []
        self.capacity_revocations: list[dict[str, Any]] = []
        self.wakeups: queue.Queue[str] = queue.Queue()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._max_workers = max_workers or max(4, sum(node.concurrency for node in spec.nodes) + 2)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.leases.start_reaper()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="shardmesh-worker"
            )
            if self.manage_cluster:
                self.cluster.start_all()
            self.load_committed_records()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self.leases.stop_reaper()
        if self.manage_cluster:
            self.cluster.stop_all()
        self._started = False

    def __enter__(self) -> "Engine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def load_committed_records(self) -> list[str]:
        loaded = []
        for record in self.ledger.list_records():
            shard_id = record["shard_id"]
            row_state = record.pop("_row_state", record.get("state"))
            if record.get("outcome") and record.get("current", True) is not False and row_state in {"committed", "current"}:
                self.records[shard_id] = record
                self.arrived.add(shard_id)
                loaded.append(shard_id)
        return loaded

    def _on_lease_expired(self, lease: Lease) -> None:
        self.capacity.release(lease.shard_id, "lease_expired")
        try:
            self.ledger.append_event(
                "lease_expired",
                {"lease_id": lease.lease_id, "shard_id": lease.shard_id, "node_id": lease.node_id},
            )
        except sqlite3.Error:
            pass
        self.wakeups.put("lease_expired")

    def mark_wait_start(self, shard_id: str) -> None:
        self.wait_started_ns.setdefault(shard_id, now_ns())

    def _reset_wait_start(self, shard_id: str) -> None:
        self.wait_started_ns[shard_id] = now_ns()

    def _wait_metrics(self, shard_id: str) -> dict[str, int | bool]:
        start = self.wait_started_ns.get(shard_id, now_ns())
        end = now_ns()
        return {
            "wait_started_ms": start // 1_000_000,
            "wait_ended_ms": end // 1_000_000,
            "wait_ms": max(0, (end - start) // 1_000_000),
            "queued": shard_id in self.wait_started_ns,
        }

    def _dependencies_ready(self, shard_id: str) -> bool:
        for upstream in self.deps_map[shard_id]:
            record = self.records.get(upstream)
            if record is None or not record.get("outcome", {}).get("summary"):
                return False
        return True

    def _candidate_decision(self, shard_id: str) -> dict[str, Any] | None:
        shard = self.spec.shard_map[shard_id]
        check = self.capacity.check(shard)
        if not check["admittable"]:
            return None
        active_count = len(self.capacity.allocations)
        attribution = build_attribution(
            shard_id=shard_id,
            deps=self.deps_map,
            scale=self.scale,
            active_count=active_count,
            global_budget=self.budget,
            node_quota=self.current_nodes[shard.node_id].concurrency,
            node_lease_limit=self.current_nodes[shard.node_id].lease_limit,
        )
        water = node_water_view(
            active_shards=check["active_shards"],
            active_leases=check["active_leases"],
            concurrency=check["concurrency"],
            lease_limit=check["lease_limit"],
            scale=self.scale,
        )
        budget = budget_boundary(
            used_units=check["active_units"],
            candidate_units=shard.units,
            budget=self.budget,
            scale=self.scale,
            bound_q=attribution["synthetic_bound_q"],
        )
        global_quant = marginal_quantization(check["active_units"], shard.units, max(1, self.budget), self.scale)
        concurrency_quant = marginal_quantization(
            check["active_shards"], 1, max(1, check["concurrency"]), self.scale
        )
        lease_quant = marginal_quantization(
            check["active_leases"], 1, max(1, check["lease_limit"]), self.scale
        )
        return {
            "shard": shard,
            "check": check,
            "attribution": attribution,
            "water": water,
            "budget_boundary": budget,
            "global_quantization": global_quant,
            "concurrency_quantization": concurrency_quant,
            "lease_quantization": lease_quant,
            "topology_index": self.topo_index[shard_id],
            "error_affected": bool(budget["ambiguous"]),
        }

    def _choose_winner(self) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for shard_id in self.topo:
            if shard_id not in self.arrived:
                continue
            if shard_id in self.records or shard_id in self.futures or shard_id in self.failed:
                continue
            if not self._dependencies_ready(shard_id):
                continue
            decision = self._candidate_decision(shard_id)
            if decision is not None:
                candidates.append(decision)
        if not candidates:
            return None

        max_priority = max(item["shard"].priority for item in candidates)
        peers = [item for item in candidates if item["shard"].priority == max_priority]
        best_q = min(item["water"]["projected_after_admit_q"] for item in peers)
        best = min(peers, key=lambda item: abs(item["water"]["projected_after_admit_q"] - best_q))
        within_bound_items: list[dict[str, Any]] = [best]
        distinct_boundary = False
        comparisons = []
        for item in peers:
            boundary = water_boundary(
                item["water"]["projected_after_admit_q"],
                best["water"]["projected_after_admit_q"],
                item["attribution"]["synthetic_bound_q"],
                best["attribution"]["synthetic_bound_q"],
            )
            is_distinct = item["shard"].id != best["shard"].id
            boundary["distinct_candidate"] = is_distinct
            boundary["indistinguishable"] = bool(is_distinct and boundary["ambiguous"])
            item["water_boundary"] = boundary
            comparisons.append(
                {
                    "left": item["shard"].id,
                    "right": best["shard"].id,
                    **boundary,
                }
            )
            if boundary["ambiguous"] and item["shard"].id != best["shard"].id:
                within_bound_items.append(item)
                distinct_boundary = True
                item["error_affected"] = True
            if item["budget_boundary"]["ambiguous"]:
                item["error_affected"] = True
        winner = min(within_bound_items, key=lambda item: (item["topology_index"], item["shard"].id))
        winner_budget_affected = bool(winner["budget_boundary"]["ambiguous"])
        winner["error_affected"] = distinct_boundary or winner_budget_affected
        winner["adjudication"] = {
            "rule": "priority_desc_then_error_boundary_then_stable_topology_index_then_shard_id",
            "max_priority": max_priority,
            "best_water_q": best_q,
            "water_comparisons": comparisons,
            "winner": winner["shard"].id,
            "error_affected": winner["error_affected"],
            "distinct_water_boundary": distinct_boundary,
            "budget_boundary": winner["budget_boundary"],
            "tie_break": ["stable_topology_index", "shard_id"],
            "global_aggregation_bound": global_aggregation_bound(
                [item["attribution"]["synthetic_bound_q"] for item in candidates],
                self.budget,
                self.scale,
            ),
        }
        return winner

    def pump(self) -> list[str]:
        admitted: list[str] = []
        if self._executor is None:
            self.start()
        while True:
            decision = self._choose_winner()
            if decision is None:
                return admitted
            shard = decision["shard"]
            self.attempt[shard.id] += 1
            generation = self.attempt[shard.id]
            lease_id = f"lease-{shard.id}-g{generation}"
            allocation = self.capacity.admit(shard, lease_id, now_ns())
            if allocation is None:
                return admitted
            lease = self.leases.issue(
                lease_id=lease_id,
                shard_id=shard.id,
                node_id=shard.node_id,
                ttl_ms=self.lease_ttl_ms,
                capacity_version=self.current_nodes[shard.node_id].capacity_version,
                version=1,
            )
            self.ledger.write_lease_grant(lease.public_dict(), "active")
            wait_metrics = self._wait_metrics(shard.id)
            context = {
                "decision": self._jsonable_decision(decision),
                "wait": wait_metrics,
                "generation": generation,
            }
            self.admission_context[shard.id] = context
            assert self._executor is not None
            future = self._executor.submit(self._execute_job, shard.id, lease, context)
            self.futures[shard.id] = future
            self.future_to_shard[future] = shard.id
            future.add_done_callback(self._future_wake)
            admitted.append(shard.id)

    def _future_wake(self, future: concurrent.futures.Future[dict[str, Any]]) -> None:
        shard_id = self.future_to_shard.get(future, "unknown")
        self.wakeups.put(shard_id)

    def _jsonable_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        shard = decision["shard"]
        return {
            "shard_id": shard.id,
            "check": decision["check"],
            "attribution": decision["attribution"],
            "water": decision["water"],
            "budget_boundary": decision["budget_boundary"],
            "global_quantization": decision["global_quantization"],
            "concurrency_quantization": decision["concurrency_quantization"],
            "lease_quantization": decision["lease_quantization"],
            "topology_index": decision["topology_index"],
            "error_affected": decision["error_affected"],
            "adjudication": decision.get("adjudication"),
        }

    def _execute_job(self, shard_id: str, lease: Lease, context: dict[str, Any]) -> dict[str, Any]:
        shard = self.spec.shard_map[shard_id]
        process = self.cluster.get(shard.node_id)
        node_lease = lease.public_dict()
        process.lease_open(node_lease, self.lease_ttl_ms)
        batch = self.samples[shard_id]
        rpc_start = now_ns()
        response = process.aggregate(
            {
                "shard_id": shard_id,
                "capacity_version": self.current_nodes[shard.node_id].capacity_version,
                "batch": batch,
                "delay_ms": self.delay_overrides.get(shard_id, 0),
            }
        )
        current_capacity_version = self.current_nodes[shard.node_id].capacity_version
        active_lease = self.leases.get(lease.lease_id)
        if active_lease is None:
            raise ShardMeshError(
                "E_LEASE",
                f"lease {lease.lease_id} was revoked or expired before commit",
                category="lease",
                field_name="lease_id",
                details={"shard_id": shard_id, "node_id": shard.node_id},
            )
        if int(response.get("capacity_version", -1)) != current_capacity_version:
            raise ShardMeshError(
                "E_LEASE",
                "refusing to commit aggregate from a stale capacity version",
                category="capacity_version",
                field_name="capacity_version",
                details={
                    "shard_id": shard_id,
                    "node_id": shard.node_id,
                    "response_version": response.get("capacity_version"),
                    "current_version": current_capacity_version,
                },
            )
        parent_rpc_ms = max(0, (now_ns() - rpc_start) // 1_000_000)
        decision = context["decision"]
        record = {
            "shard_id": shard_id,
            "node_id": shard.node_id,
            "state": "committed",
            "current": True,
            "stale": False,
            "topology_index": self.topo_index[shard_id],
            "parameter_revision": self.parameter_revision[shard_id],
            "capacity_version": response["capacity_version"],
            "outcome": {
                "summary": response["summary"],
                "node_elapsed_ms": response["elapsed_ms"],
                "parent_rpc_ms": parent_rpc_ms,
                "sample_digest": batch_digest(batch),
            },
            "lease": active_lease.public_dict(),
            "resource": {
                "units": shard.units,
                "global_budget": self.budget,
                "fixed_units": shard.units * self.scale,
                "budget_fixed": self.budget * self.scale,
                "admission_check": decision["check"],
            },
            "water": decision["water"],
            "quantization": {
                "global": decision["global_quantization"],
                "concurrency": decision["concurrency_quantization"],
                "lease": decision["lease_quantization"],
            },
            "error_attribution": decision["attribution"],
            "adjudication": decision.get("adjudication"),
            "wait": context["wait"],
        }
        committed = self.ledger.commit_outcome(record)
        self.records[shard_id] = committed
        return committed

    def _finish_job_locally(self, shard_id: str) -> None:
        lease = self.leases.get_for_shard(shard_id)
        if lease is not None:
            try:
                self.cluster.get(lease.node_id).lease_release(lease.lease_id)
            except ShardMeshError:
                pass
            self.leases.remove(lease.lease_id)
        self.capacity.release(shard_id, "outcome_settled")

    def _consume_finished(self) -> None:
        done_ids = [shard_id for shard_id, future in list(self.futures.items()) if future.done()]
        for shard_id in done_ids:
            future = self.futures.pop(shard_id)
            self.future_to_shard.pop(future, None)
            try:
                future.result()
                self.failed.pop(shard_id, None)
            except Exception as exc:  # converted to typed failure below
                error = exc if isinstance(exc, ShardMeshError) else ShardMeshError("E_NODE", str(exc), category="worker")
                self.failed[shard_id] = error
                self.ledger.mark_shard_state(
                    shard_id,
                    "failed",
                    {"code": error.code, "message": str(error), "field": error.field_name},
                )
            finally:
                self._finish_job_locally(shard_id)

    def settle(self, idle_polls: int = 1) -> None:
        if self._executor is None:
            self.start()
        polls = 0
        while True:
            admitted = self.pump()
            if admitted:
                polls = 0
            if not self.futures:
                self._consume_finished()
                if self.pump():
                    continue
                polls += 1
                if polls >= max(1, idle_polls):
                    return
                continue
            concurrent.futures.wait(list(self.futures.values()), timeout=0.05)
            try:
                while True:
                    self.wakeups.get_nowait()
            except queue.Empty:
                pass
            self._consume_finished()

    def run_events(self, events: list[Event] | tuple[Event, ...] | None = None) -> list[dict[str, Any]]:
        self.start()
        events = list(events if events is not None else self.spec.events)
        for event in events:
            if event.type == "arrive":
                shard_id = event.params["shard_id"]
                self.arrived.add(shard_id)
                self.mark_wait_start(shard_id)
                self.pump()
                continue
            self.settle()
            plan = self.apply_parameter_event(event)
            if plan is not None:
                self.invalidation_plans.append(plan)
            self.pump()
            self.settle()
        self.settle()
        if self.failed:
            first = next(iter(self.failed.values()))
            raise first
        return [self.records[shard_id] for shard_id in self.topo if shard_id in self.records]

    def run_pending(self) -> list[dict[str, Any]]:
        self.start()
        for shard in self.spec.shards:
            if shard.id not in self.records:
                self.arrived.add(shard.id)
                self.mark_wait_start(shard.id)
        self.settle()
        if self.failed:
            raise next(iter(self.failed.values()))
        return [self.records[shard_id] for shard_id in self.topo if shard_id in self.records]

    def apply_parameter_event(self, event: Event) -> dict[str, Any] | None:
        params = event.params
        direct: list[str] = []
        reason = event.type
        if event.type == "set_budget":
            self.budget = int(params["budget"])
            self.capacity.update_budget(self.budget)
            direct = list(self.spec.shard_map)
        elif event.type == "set_node_quota":
            node_id = params["node_id"]
            quota = int(params["concurrency"])
            self.current_nodes[node_id] = replace(self.current_nodes[node_id], concurrency=quota)
            self.capacity.update_quota(node_id, quota)
            direct = [shard.id for shard in self.spec.shards if shard.node_id == node_id]
        elif event.type == "set_node_lease_limit":
            node_id = params["node_id"]
            limit = int(params["lease_limit"])
            self.current_nodes[node_id] = replace(self.current_nodes[node_id], lease_limit=limit)
            self.capacity.update_lease_limit(node_id, limit)
            direct = [shard.id for shard in self.spec.shards if shard.node_id == node_id]
        elif event.type == "set_scale":
            self.scale = int(params["scale"])
            self.cluster.scale = self.scale
            for node_id in self.cluster.node_specs:
                self.cluster.restart_node(node_id)
            for node_id, node in list(self.current_nodes.items()):
                self.current_nodes[node_id] = replace(node)
            direct = list(self.spec.shard_map)
        elif event.type == "set_max_renewals":
            self.max_renewals = int(params["max_renewals"])
            return {"seq": event.seq, "event_type": event.type, "changed": [], "steps": []}
        elif event.type == "set_lease_ttl":
            self.lease_ttl_ms = int(params["lease_ttl_ms"])
            return {"seq": event.seq, "event_type": event.type, "changed": [], "steps": []}
        elif event.type == "bump_capacity_version":
            node_id = params["node_id"]
            version = int(params["version"])
            revoked = self.leases.revoke_capacity_version(node_id, version)
            for lease in revoked:
                self.capacity.release(lease.shard_id, "capacity_version_bump")
            self.capacity_revocations.append(
                {"event_seq": event.seq, "node_id": node_id, "new_version": version, "leases": [lease.public_dict() for lease in revoked]}
            )
            self.current_nodes[node_id] = replace(self.current_nodes[node_id], capacity_version=version)
            if self.manage_cluster:
                self.cluster.restart_node(node_id, version)
            else:
                self.cluster.node_specs[node_id] = self.current_nodes[node_id]
            direct = [shard.id for shard in self.spec.shards if shard.node_id == node_id]
        else:
            return None

        ordered, steps = transitive_downstream(direct, self.deps_map)
        for shard_id in ordered:
            self.parameter_revision[shard_id] += 1
            self.attempt[shard_id] += 0
            self.records.pop(shard_id, None)
            self.failed.pop(shard_id, None)
            self._reset_wait_start(shard_id)
        stale_state = "stale" if event.type == "bump_capacity_version" else "invalidated"
        self.ledger.flag_records(
            ordered,
            stale_state,
            current=False,
            stale=(event.type == "bump_capacity_version"),
            invalidated_by_event=event.seq,
            invalidation_reason=reason,
        )
        try:
            self.ledger.append_event(
                "invalidation",
                {"seq": event.seq, "type": event.type, "changed": ordered, "steps": steps},
            )
        except Exception:
            pass
        return {
            "seq": event.seq,
            "event_type": event.type,
            "reason": reason,
            "direct": sorted(direct, key=lambda sid: self.topo_index[sid]),
            "changed": ordered,
            "steps": steps,
        }

    def renew_lease_with_node(self, shard_id: str, expected_version: int | None = None) -> dict[str, Any]:
        lease = self.leases.get_for_shard(shard_id)
        if lease is None:
            raise ShardMeshError("E_LEASE", f"no active lease for shard {shard_id}", category="lease", field_name="lease_id")
        expected = lease.version if expected_version is None else expected_version
        response = self.cluster.get(lease.node_id).lease_renew(
            lease.lease_id, expected, self.lease_ttl_ms, self.max_renewals
        )
        updated = self.leases.apply_renewal(
            lease.lease_id,
            int(response["version"]),
            int(response["renew_count"]),
            self.lease_ttl_ms,
        )
        self.ledger.write_lease_grant(updated.public_dict(), "active")
        return updated.public_dict()

    def recover_unknown_shards(self) -> dict[str, Any]:
        reclaimed_leases = self.ledger.reclaim_open_leases("forced_termination")
        reclaimed_capacity = self.capacity.reclaim_all("forced_termination")
        committed = self.ledger.committed_outcome_ids()
        unknown = [shard.id for shard in self.spec.shards if shard.id not in committed]
        for lease in reclaimed_leases:
            shard_id = lease.get("shard_id")
            if shard_id in self.attempt:
                self.attempt[shard_id] += 1
        for shard_id in unknown:
            self.ledger.mark_shard_state(
                shard_id,
                "unknown",
                {"reason": "uncommitted_after_forced_termination", "rerun_required": True},
            )
            self.failed.pop(shard_id, None)
            self.records.pop(shard_id, None)
            self.arrived.discard(shard_id)
        return {
            "reclaimed_leases": reclaimed_leases,
            "reclaimed_capacity": [item.__dict__ for item in reclaimed_capacity],
            "unknown_shards": unknown,
            "quota_reclaimed_first": True,
        }
