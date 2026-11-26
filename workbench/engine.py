from __future__ import annotations

import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .canonical import canonical_json, content_digest
from .errors import LeaseError, NodeGoneError, SchedulingError, WorkbenchError
from .invalidation import rendezvous_node_id, rendezvous_rank
from .ledger import Ledger
from .orchestrator import NodeOrchestrator
from .protocol import call_node, new_request_id
from .samples import generate_samples, merge_batch_summaries, sample_digest
from .snapshot import SnapshotStore
from .spec import Spec
from .timeutil import MonotonicClock, clock_ms
from .water import ErrorAttribution, WaterModel


class WorkbenchEngine:
    def __init__(
        self,
        spec: Spec,
        workdir: str | Path = "artifacts/default",
        *,
        db_path: str | Path | None = None,
        snapshot_path: str | Path | None = None,
        clock: object | None = None,
        batch_size: int | None = None,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.workdir / "workbench.sqlite3"
        self.snapshot_path = (
            Path(snapshot_path) if snapshot_path else self.workdir / "session-snapshot.json"
        )
        self.clock = clock or MonotonicClock()
        self.batch_size = batch_size
        self.parallel = parallel
        self.max_workers = max(1, int(max_workers))
        self.water_model = WaterModel(spec.scale_exponent)
        self.ledger = Ledger(self.db_path)
        self.snapshots = SnapshotStore(self.snapshot_path)
        self.orchestrator = NodeOrchestrator(spec, self.workdir)
        self.topo_index = {shard_id: index for index, shard_id in enumerate(spec.topo_order)}
        self._lock = __import__("threading").RLock()
        self.wait_started: dict[str, int] = {}
        self.decisions: list[dict[str, Any]] = []
        self.water_decisions: list[dict[str, Any]] = []
        self.ledger.initialize_spec(
            spec.digest, canonical_json(spec.raw), self.now_ms()
        )

    def now_ms(self) -> int:
        return clock_ms(self.clock)

    def close(self) -> None:
        self.orchestrator.stop_all()
        self.ledger.close()

    def __enter__(self) -> "WorkbenchEngine":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start_nodes(self) -> None:
        self.orchestrator.start_all()

    def ensure_nodes(self) -> None:
        for node in self.spec.nodes:
            node_id = str(node["node_id"])
            handle = self.orchestrator.handles.get(node_id)
            if handle is None or not handle.alive:
                self.orchestrator.start_node(node_id)

    def version_distance(self, capacity_version: int) -> int:
        versions = [int(item["version"]) for item in self.spec.raw["capacity_chain"]]
        if capacity_version not in versions:
            return 0
        return len(versions) - 1 - versions.index(int(capacity_version))

    def recover_after_termination(self, *, restart_nodes: bool = True) -> list[dict[str, Any]]:
        now = self.now_ms()
        unknown = self.ledger.recover_uncommitted(now)
        if restart_nodes:
            self.ensure_nodes()
            self.restore_active_leases_to_nodes()
        self.snapshots.sync(self.spec, self.ledger)
        return unknown

    def restore_active_leases_to_nodes(self) -> int:
        active = self.ledger.list_leases(active_only=True)
        restored = 0
        for lease in active:
            node = self.spec.node_by_id[lease["node_id"]]
            call_node(
                node,
                {
                    "type": "lease_sync",
                    "request_id": new_request_id("sync"),
                    **lease,
                },
                timeout_ms=int(self.spec.socket_timeout_ms),
                expect_summary=False,
            )
            restored += 1
        return restored

    def select_node(self, shard: dict[str, Any]) -> tuple[str | None, str | None]:
        pinned = shard.get("node_id")
        candidates: list[str]
        if pinned:
            candidates = [str(pinned)]
        else:
            candidates = [str(node["node_id"]) for node in self.spec.nodes]
        counts = self.ledger.active_node_counts()
        ordered_candidates = (
            list(candidates)
            if pinned
            else rendezvous_rank(tuple(candidates), shard["shard_id"])
        )
        for node_id in ordered_candidates:
            node = self.spec.node_by_id[node_id]
            used = counts.get(node_id, {"shards": 0, "leases": 0})
            if int(used["shards"]) + 1 > int(node["concurrency_quota"]):
                continue
            if int(used["leases"]) + 1 > int(node["lease_limit"]):
                continue
            return node_id, None
        if not candidates:
            return None, "no_declared_node"
        node_id = candidates[0]
        node = self.spec.node_by_id[node_id]
        used = counts.get(node_id, {"shards": 0, "leases": 0})
        if int(used["shards"]) + 1 > int(node["concurrency_quota"]):
            return None, "concurrency_quota"
        return None, "lease_limit"

    def admission_block_reason(self, shard_id: str) -> str | None:
        shard = self.spec.shard_by_id[shard_id]
        if shard.get("frozen", False):
            return "frozen"
        demand = int(shard["resource_units"])
        used_resources = self.ledger.active_resource_sum()
        budget = int(self.spec.raw["global_budget_units"])
        if used_resources + demand > budget:
            return "global_budget"
        node_id, reason = self.select_node(shard)
        if node_id is None:
            return reason or "node_capacity"
        return None

    def ready_shard_ids(self) -> list[str]:
        completed = {
            row["shard_id"]: row
            for row in self.ledger.list_shards()
            if row["status"] == "completed" and row.get("summary_digest")
        }
        ready: list[str] = []
        for shard_id in self.spec.topo_order:
            shard = self.spec.shard_by_id[shard_id]
            if shard.get("frozen", False):
                continue
            row = self.ledger.get_shard(shard_id)
            if row and row["status"] == "completed":
                continue
            if row and row["status"] == "in_flight":
                continue
            if all(up in completed for up in shard.get("upstreams", [])):
                ready.append(shard_id)
        priority_order = {
            shard["shard_id"]: (-int(shard.get("priority", 0)), self.topo_index[shard["shard_id"]], shard["shard_id"])
            for shard in self.spec.shards
        }
        return sorted(ready, key=lambda sid: priority_order[sid])

    def adjudicate_ready(self) -> dict[str, Any]:
        ids = self.ready_shard_ids()
        budget = int(self.spec.raw["global_budget_units"])
        used = self.ledger.active_resource_sum()
        remaining = budget - used
        candidates: list[dict[str, Any]] = []
        for shard_id in ids:
            shard = self.spec.shard_by_id[shard_id]
            node_id = self._planned_node_id(shard)
            node = self.spec.node_by_id[node_id]
            version = int(node["capacity_version"])
            distance = self.version_distance(version)
            attribution = ErrorAttribution(
                version_edges=distance,
                version_error_fp=distance,
                dependency_uncertain=0,
                dependency_error_fp=0,
                lease_boundaries=1,
                lease_error_fp=1,
            )
            error = self.water_model.shard_error_record(
                shard_id=shard_id,
                resource_units=int(shard["resource_units"]),
                global_budget=budget,
                attribution=attribution,
            )
            budget_gap = remaining - int(shard["resource_units"])
            budget_ambiguous = self.water_model.budget_gap_is_ambiguous(
                remaining_budget=remaining,
                demand=int(shard["resource_units"]),
                shard_error=error,
            )
            candidates.append(
                {
                    "shard_id": shard_id,
                    "topo_index": self.topo_index[shard_id],
                    "priority": int(shard.get("priority", 0)),
                    "water_fp": error["global_share_water_fp"],
                    "comparison_error_bound_fp": error["synthetic_error_bound_fp"],
                    "budget_gap_raw_units": budget_gap,
                    "budget_error_affected": budget_ambiguous,
                    "error_affected": budget_ambiguous,
                }
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                -int(item["priority"]),
                int(item["topo_index"]),
                str(item["shard_id"]),
            ),
        )
        for index in range(len(ordered) - 1):
            left = ordered[index]
            right = ordered[index + 1]
            bound = max(
                int(left["comparison_error_bound_fp"]),
                int(right["comparison_error_bound_fp"]),
            )
            if self.water_model.water_difference_is_ambiguous(
                int(left["water_fp"]), int(right["water_fp"]), bound
            ):
                left["water_tie_affected"] = True
                right["water_tie_affected"] = True
                stable_winner = min(
                    (left, right),
                    key=lambda item: (int(item["topo_index"]), str(item["shard_id"])),
                )
                stable_winner["stable_tie_winner"] = True
                for candidate in (left, right):
                    candidate["error_affected"] = True
        adjudication = {
            "at_ms": self.now_ms(),
            "remaining_budget_units": remaining,
            "tie_rule": "priority_then_stable_topological_order_then_shard_id",
            "candidates": ordered,
        }
        self.water_decisions.append(adjudication)
        return adjudication

    def admit_shard(self, shard_id: str) -> dict[str, Any] | None:
        with self._lock:
            shard = self.spec.shard_by_id[shard_id]
            if shard.get("frozen", False):
                return None
            existing = self.ledger.get_shard(shard_id)
            if existing and existing["status"] == "in_flight":
                return None
            block_reason = self.admission_block_reason(shard_id)
            now = self.now_ms()
            if block_reason is not None:
                if block_reason != "frozen" and shard_id not in self.wait_started:
                    self.wait_started[shard_id] = now
                self.decisions.append(
                    {
                        "at_ms": now,
                        "shard_id": shard_id,
                        "admitted": False,
                        "reason": block_reason,
                        "topo_index": self.topo_index[shard_id],
                    }
                )
                return None

            node_id, _ = self.select_node(shard)
            if node_id is None:
                return None
            wait_started = self.wait_started.pop(shard_id, now)
            wait_ended = now
            wait_ms = max(0, wait_ended - wait_started)
            lease_id = f"lease-{shard_id}"
            lease = {
                "lease_id": lease_id,
                "shard_id": shard_id,
                "node_id": node_id,
                "granted_at_ms": now,
                "expires_at_ms": now + int(self.spec.lease_ttl_ms),
                "renew_count": 0,
                "lease_version": 1,
                "capacity_version": int(self.spec.node_by_id[node_id]["capacity_version"]),
            }
            scaled = self.water_model.scale_amount(int(shard["resource_units"]))
            dispatch = {
                "shard_id": shard_id,
                "topo_index": self.topo_index[shard_id],
                "node_id": node_id,
                "lease_id": lease_id,
                "sample_rows": int(shard["sample_rows"]),
                "capacity_version": lease["capacity_version"],
                "now_ms": now,
                "wait_started_ms": wait_started,
                "wait_ended_ms": wait_ended,
                "wait_ms": wait_ms,
                "lease": lease,
                "occupancy": {
                    "resource_units": int(shard["resource_units"]),
                    "global_budget": int(self.spec.raw["global_budget_units"]),
                    "scale_exponent": self.spec.scale_exponent,
                    "scaled_resource_units": scaled["scaled"],
                    "saturated": scaled["saturated"],
                },
            }
            self.ledger.upsert_dispatch(dispatch)
            node = self.spec.node_by_id[node_id]
            call_node(
                node,
                {
                    "type": "lease_sync",
                    "request_id": new_request_id("grant"),
                    **lease,
                },
                timeout_ms=int(self.spec.socket_timeout_ms),
                expect_summary=False,
            )

            counts = self.ledger.active_node_counts().get(
                node_id, {"shards": 1, "leases": 1}
            )
            resources = self.ledger.active_node_resources().get(node_id, int(shard["resource_units"]))
            attribution = ErrorAttribution(
                version_edges=self.version_distance(lease["capacity_version"]),
                version_error_fp=self.version_distance(lease["capacity_version"]),
                dependency_uncertain=0,
                dependency_error_fp=0,
                lease_boundaries=1,
                lease_error_fp=1,
            )
            node_water = self.water_model.node_water(
                active_shards=int(counts["shards"]),
                concurrency_quota=int(node["concurrency_quota"]),
                active_leases=int(counts["leases"]),
                lease_limit=int(node["lease_limit"]),
                node_resource_units=resources,
                global_budget=int(self.spec.raw["global_budget_units"]),
                attribution=attribution,
            )
            node_water.update(
                {
                    "scope_type": "node",
                    "scope_id": node_id,
                    "node_id": node_id,
                    "at_ms": now,
                }
            )
            context = {
                "dispatch": dispatch,
                "attribution": attribution.as_dict(),
                "node_water": node_water,
            }
            self.decisions.append(
                {
                    "at_ms": now,
                    "shard_id": shard_id,
                    "admitted": True,
                    "node_id": node_id,
                    "lease_id": lease_id,
                    "wait_ms": wait_ms,
                    "topo_index": self.topo_index[shard_id],
                }
            )
            return context

    def execute_context(self, context: dict[str, Any]) -> dict[str, Any]:
        dispatch = context["dispatch"]
        shard_id = dispatch["shard_id"]
        node_id = dispatch["node_id"]
        node = self.spec.node_by_id[node_id]
        shard = self.spec.shard_by_id[shard_id]
        seed = int(node["seed"])
        rows = int(shard["sample_rows"])
        samples = generate_samples(seed, shard_id, rows)
        full_digest = sample_digest(seed, shard_id, rows)
        batch_size = self.batch_size or rows
        batch_size = max(1, min(int(batch_size), rows))
        batches: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        for batch_index, start in enumerate(range(0, rows, batch_size)):
            batch_samples = samples[start : start + batch_size]
            response = call_node(
                node,
                {
                    "type": "aggregate_batch",
                    "request_id": new_request_id(f"{shard_id}-{batch_index}"),
                    "shard_id": shard_id,
                    "capacity_version": int(dispatch["capacity_version"]),
                    "batch_index": batch_index,
                    "samples": batch_samples,
                },
                timeout_ms=int(self.spec.socket_timeout_ms),
                expect_summary=True,
            )
            batches.append(response["summary"])
            receipts.append(
                {
                    "request_id": response["request_id"],
                    "elapsed_ms_monotonic": response["elapsed_ms_monotonic"],
                }
            )
        fp_exp = int(node.get("fixed_point_exponent", 0))
        summary = merge_batch_summaries(
            batches,
            node_id=node_id,
            shard_id=shard_id,
            sample_rows=rows,
            capacity_version=int(dispatch["capacity_version"]),
            fixed_point_exponent=fp_exp,
            full_sample_digest=full_digest,
        )
        return {"summary": summary, "sample_digest": full_digest, "receipts": receipts}

    def _planned_node_id(self, shard: dict[str, Any]) -> str:
        if shard.get("node_id"):
            return str(shard["node_id"])
        return rendezvous_node_id(
            [str(node["node_id"]) for node in self.spec.nodes], shard["shard_id"]
        )

    def _static_global_water(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        budget = int(self.spec.raw["global_budget_units"])
        for shard in self.spec.shards:
            node_id = self._planned_node_id(shard)
            capacity_version = int(self.spec.node_by_id[node_id]["capacity_version"])
            distance = self.version_distance(capacity_version)
            attribution = ErrorAttribution(
                version_edges=distance,
                version_error_fp=distance,
                dependency_uncertain=0,
                dependency_error_fp=0,
                lease_boundaries=1,
                lease_error_fp=1,
            )
            records.append(
                self.water_model.shard_error_record(
                    shard_id=shard["shard_id"],
                    resource_units=int(shard["resource_units"]),
                    global_budget=budget,
                    attribution=attribution,
                )
            )
        global_water = self.water_model.global_error_summary(records)
        global_water.update(
            {
                "scope_type": "global",
                "scope_id": "global_budget",
                "global_budget_units": int(self.spec.raw["global_budget_units"]),
            }
        )
        return global_water

    def complete_context(
        self, context: dict[str, Any], execution: dict[str, Any]
    ) -> str:
        with self._lock:
            dispatch = context["dispatch"]
            shard_id = dispatch["shard_id"]
            lease_row = self.ledger.get_lease(dispatch["lease_id"])
            if lease_row is None or lease_row["state"] != "active":
                raise LeaseError(
                    "lease is no longer active; late outcome rejected",
                    shard_id=shard_id,
                    lease_id=dispatch["lease_id"],
                    state=None if lease_row is None else lease_row["state"],
                )
            now = self.now_ms()
            summary = execution["summary"]
            shard_error = self.water_model.shard_error_record(
                shard_id=shard_id,
                resource_units=int(dispatch["occupancy"]["resource_units"]),
                global_budget=int(self.spec.raw["global_budget_units"]),
                attribution=ErrorAttribution(**{
                    key: int(value)
                    for key, value in context["attribution"].items()
                    if key != "source_error_fp"
                }),
            )
            shard_water = dict(shard_error)
            shard_water.update({"scope_type": "shard", "scope_id": shard_id})
            global_water = self._static_global_water()
            completed_before = len(
                [s for s in self.ledger.list_shards() if s["status"] == "completed"]
            )
            outcome = {
                "shard_id": shard_id,
                "node_id": dispatch["node_id"],
                "lease_id": dispatch["lease_id"],
                "generation": 1,
                "sample_rows": dispatch["sample_rows"],
                "participating_rows": int(summary["participating_rows"]),
                "capacity_version": dispatch["capacity_version"],
                "summary_digest": summary["summary_digest"],
                "summary": summary,
                "wait_started_ms": dispatch["wait_started_ms"],
                "wait_ended_ms": dispatch["wait_ended_ms"],
                "wait_ms": dispatch["wait_ms"],
                "lease": {
                    "lease_id": lease_row["lease_id"],
                    "granted_at_ms": lease_row["granted_at_ms"],
                    "expires_at_ms": lease_row["expires_at_ms"],
                    "renew_count": lease_row["renew_count"],
                    "lease_version": lease_row["lease_version"],
                    "node_id": lease_row["node_id"],
                    "shard_id": lease_row["shard_id"],
                },
                "scaled_resource_units": dispatch["occupancy"]["scaled_resource_units"],
                "saturated": dispatch["occupancy"]["saturated"],
                "water": {"scale_exponent": self.spec.scale_exponent},
                "error_attribution": shard_error,
                "shard_water": shard_water,
                "water_records": [shard_water, context["node_water"], global_water],
                "cursor": {
                    "completed_count": completed_before + 1,
                    "last_topo_index": self.topo_index[shard_id],
                    "last_event_seq": len(self.spec.events),
                },
                "committed_at_ms": now,
            }
            conclusion_fields = [
                "shard_id",
                "node_id",
                "lease_id",
                "generation",
                "sample_rows",
                "participating_rows",
                "capacity_version",
                "summary_digest",
                "summary",
                "wait_started_ms",
                "wait_ended_ms",
                "wait_ms",
                "lease",
                "scaled_resource_units",
                "saturated",
                "water",
                "error_attribution",
                "shard_water",
            ]
            outcome["conclusion_digest"] = content_digest(
                {key: outcome[key] for key in conclusion_fields}
            )
            digest = self.ledger.commit_shard_outcome(outcome)
            self.snapshots.sync(self.spec, self.ledger)
            return digest

    def run_shard_now(self, shard_id: str) -> str | None:
        context = self.admit_shard(shard_id)
        if context is None:
            return None
        execution = self.execute_context(context)
        return self.complete_context(context, execution)

    def run_until_complete(self, *, recover: bool = True) -> dict[str, Any]:
        self.start_nodes()
        if recover:
            self.recover_after_termination(restart_nodes=False)
        executor = ThreadPoolExecutor(
            max_workers=1 if not self.parallel else self.max_workers,
            thread_name_prefix="shard-dispatch",
        )
        in_flight: dict[Any, dict[str, Any]] = {}
        try:
            idle_passes = 0
            while True:
                self.ledger.reap_expired(self.now_ms())
                active_shard_ids = {
                    context["dispatch"]["shard_id"] for context in in_flight.values()
                }
                admitted_this_pass: set[str] = set()
                adjudication = self.adjudicate_ready()
                for candidate in adjudication["candidates"]:
                    shard_id = str(candidate["shard_id"])
                    if shard_id in active_shard_ids or shard_id in admitted_this_pass:
                        continue
                    context = self.admit_shard(shard_id)
                    if context is None:
                        continue
                    admitted_this_pass.add(shard_id)
                    future = executor.submit(self.execute_context, context)
                    in_flight[future] = context

                unfinished = [
                    sid
                    for sid in self.spec.topo_order
                    if not self.spec.shard_by_id[sid].get("frozen", False)
                    and not (self.ledger.get_shard(sid) or {}).get("status") == "completed"
                ]
                if not in_flight:
                    if not unfinished:
                        break
                    reasons = [
                        {
                            "shard_id": sid,
                            "reason": self.admission_block_reason(sid) or "dependency_not_ready",
                        }
                        for sid in unfinished
                    ]
                    idle_passes += 1
                    if idle_passes > 2:
                        raise SchedulingError(
                            "scheduler cannot admit unfinished shards", blockers=reasons
                        )
                    continue
                idle_passes = 0
                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    context = in_flight.pop(future)
                    shard_id = context["dispatch"]["shard_id"]
                    try:
                        execution = future.result()
                        self.complete_context(context, execution)
                    except Exception as exc:  # noqa: BLE001 - surface first shard failure
                        for pending_future in in_flight:
                            pending_future.cancel()
                        raise WorkbenchError(
                            f"shard {shard_id} failed: {exc}",
                            code=getattr(exc, "code", "E_EXEC"),
                            shard_id=shard_id,
                        ) from exc
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        self.snapshots.sync(self.spec, self.ledger)
        return {
            "completed": [
                row["shard_id"]
                for row in self.ledger.list_shards()
                if row["status"] == "completed"
            ],
            "snapshot_path": str(self.snapshot_path),
            "db_path": str(self.db_path),
        }

    def renew_lease(
        self, lease_id: str, expected_version: int, *, timeout_ms: int | None = None
    ) -> dict[str, Any]:
        with self._lock:
            lease = self.ledger.get_lease(lease_id)
            if lease is None or lease["state"] != "active":
                raise LeaseError("lease is not active", lease_id=lease_id)
            now = self.now_ms()
            expires_at = now + int(self.spec.lease_ttl_ms)
            node = self.spec.node_by_id[lease["node_id"]]
            response = call_node(
                node,
                {
                    "type": "lease_renew",
                    "request_id": new_request_id("renew"),
                    "lease_id": lease_id,
                    "shard_id": lease["shard_id"],
                    "lease_version": int(expected_version),
                    "expires_at_ms": expires_at,
                },
                timeout_ms=int(timeout_ms or self.spec.socket_timeout_ms),
                expect_summary=False,
            )
            new_version = int(response["lease_version"])
            renew_count = int(response["renew_count"])
            updated = self.ledger.renew_lease_row(
                lease_id,
                expected_version=int(expected_version),
                new_version=new_version,
                renew_count=renew_count,
                expires_at_ms=int(response["expires_at_ms"]),
                now_ms=now,
            )
            if not updated:
                raise LeaseError(
                    "ledger lease version mismatch; expiry not extended",
                    lease_id=lease_id,
                    expected_version=expected_version,
                )
            return self.ledger.get_lease(lease_id)  # type: ignore[return-value]

    def release_lease(self, lease_id: str, reason: str = "manual_release") -> bool:
        with self._lock:
            lease = self.ledger.get_lease(lease_id)
            if lease is None:
                return False
            node = self.spec.node_by_id[lease["node_id"]]
            try:
                call_node(
                    node,
                    {
                        "type": "lease_release",
                        "request_id": new_request_id("release"),
                        "lease_id": lease_id,
                    },
                    timeout_ms=int(self.spec.socket_timeout_ms),
                    expect_summary=False,
                )
            except NodeGoneError:
                pass
            return self.ledger.transition_lease(
                lease_id, new_state="released", reap_reason=reason, now_ms=self.now_ms()
            )

    def reap_expired_leases(self) -> list[str]:
        with self._lock:
            return self.ledger.reap_expired(self.now_ms())

    def reap_capacity_generation(self, node_id: str, old_capacity_version: int) -> list[str]:
        with self._lock:
            return self.ledger.reap_version_generation(
                node_id, int(old_capacity_version), self.now_ms()
            )
