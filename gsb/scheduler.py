from __future__ import annotations

import concurrent.futures
import threading
from typing import Any, Iterable

from .clock import monotonic_ms
from .errors import GSBError, LeaseError
from .fixedpoint import (
    adjudicate_candidates,
    budget_boundary_view,
    quantize_units,
    unified_water_report,
)
from .graph import effective_node_id, propagation_basis
from .lease import LeaseManager
from .samples import request_payload_for_shard


class Scheduler:
    def __init__(
        self,
        spec: dict[str, Any],
        cluster: Any,
        ledger: Any,
        *,
        lease_manager: LeaseManager | None = None,
        run_id: str = "run",
    ) -> None:
        self.spec = spec
        self.cluster = cluster
        self.ledger = ledger
        self.run_id = run_id
        ttl = int(spec["scheduling"]["lease_ttl_ms"])
        self.leases = lease_manager or LeaseManager(ttl_ms=ttl)
        self.budget = int(spec["budget_units"])
        self.scale = int(spec["scale_factor"])
        self._lock = threading.RLock()
        self.active_units = 0
        self.active_shards_by_node: dict[str, int] = {
            node["node_id"]: 0 for node in spec["nodes"]
        }
        self.peak_active_units = 0
        self.peak_shards_by_node: dict[str, int] = {
            node["node_id"]: 0 for node in spec["nodes"]
        }
        self.wait_records: dict[str, dict[str, Any]] = {}
        self.decisions: list[dict[str, Any]] = []
        self.rounds: list[dict[str, Any]] = []
        self.outcomes: dict[str, dict[str, Any]] = {}
        self.completed: set[str] = set()
        existing = ledger.list_outcomes() if ledger is not None else []
        self.existing_outcome_ids = {row["shard_id"] for row in existing}
        for row in existing:
            shard_id = row["shard_id"]
            self.outcomes[shard_id] = row
            self.decisions.append(
                {
                    "kind": "hydrated_from_ledger",
                    "shard_id": shard_id,
                    "node_id": row["node_id"],
                    "lease_id": row["lease_id"],
                    "capacity_version": row["capacity_version"],
                    "water_q": row["water_q"],
                    "cursor": row["cursor"],
                }
            )
            if row.get("wait_started_ms") is not None:
                self.wait_records[shard_id] = {
                    "shard_id": shard_id,
                    "wait_started_ms": row.get("wait_started_ms"),
                    "wait_ended_ms": row.get("wait_ended_ms"),
                    "reasons": ["hydrated_from_ledger"],
                }

    def run(self, shard_ids: Iterable[str] | None = None) -> dict[str, Any]:
        all_order = list(self.spec["topo_order"])
        if shard_ids is None:
            targets = [sid for sid in all_order if sid not in self.existing_outcome_ids]
            self.completed.update(self.existing_outcome_ids)
        else:
            target_set = set(shard_ids)
            unknown = target_set - set(all_order)
            if unknown:
                raise GSBError(f"unknown target shards: {sorted(unknown)}", field="shard_ids")
            targets = [sid for sid in all_order if sid in target_set]
            # Old rows for targets will be replaced; all other ledger outcomes
            # satisfy dependencies and remain byte-for-byte untouched.
            self.completed.update(self.existing_outcome_ids - target_set)

        if not targets:
            return self._result(status="nothing_to_do")

        pool_size = self.spec["scheduling"].get("worker_pool_size")
        if pool_size is None:
            pool_size = max(1, sum(max(1, int(node["concurrency"])) for node in self.spec["nodes"]))
        poll_seconds = max(1, int(self.spec["scheduling"]["poll_interval_ms"])) / 1000.0
        max_per_round = int(self.spec["scheduling"]["round_boundary"])

        pending = set(targets)
        futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        round_number = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(pool_size),
            thread_name_prefix="gsb-shard",
        ) as executor:
            while pending or futures:
                done, _ = concurrent.futures.wait(
                    tuple(futures),
                    timeout=poll_seconds,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                ) if futures else (set(), set())

                for future in done:
                    context = futures.pop(future)
                    self._consume_future(future, context)

                expired = self.leases.expire_due()
                for lease in expired:
                    self.decisions.append(
                        {
                            "kind": "lease_expired",
                            "lease_id": lease.lease_id,
                            "shard_id": lease.shard_id,
                            "node_id": lease.node_id,
                            "at_monotonic_ms": monotonic_ms(),
                        }
                    )

                admissions = 0
                round_number += 1
                round_record = {"round": round_number, "admitted": [], "blocked": []}
                ready_ids = self._ready_order(pending)
                if ready_ids:
                    round_record["candidate_adjudication"] = adjudicate_candidates(
                        [
                            {
                                "shard_id": shard_id,
                                "topo_index": self.spec["topo_index"][shard_id],
                                "resource_units": self.spec["shard_by_id"][shard_id]["resource_units"],
                            }
                            for shard_id in ready_ids
                        ],
                        budget_units=self.budget,
                        scale=self.scale,
                    )
                for shard_id in ready_ids:
                    if admissions >= max_per_round:
                        break
                    shard = self.spec["shard_by_id"][shard_id]
                    decision = self._can_admit(shard)
                    if decision["admit"]:
                        context = self._admit(shard, decision, executor)
                        futures[context["future"]] = context
                        pending.remove(shard_id)
                        admissions += 1
                        round_record["admitted"].append(shard_id)
                    else:
                        self._mark_waiting(shard_id, decision["reason"])
                        round_record["blocked"].append(
                            {"shard_id": shard_id, "reason": decision["reason"]}
                        )
                if round_record["admitted"] or round_record["blocked"]:
                    self.rounds.append(round_record)

                if not futures and not admissions and pending:
                    # No task is running and no candidate could be admitted.
                    blockers = [
                        self._can_admit(self.spec["shard_by_id"][sid])
                        for sid in self._ready_order(pending)
                    ]
                    raise GSBError(
                        "scheduler made no progress; likely zero quota or budget smaller than demand",
                        field="scheduling",
                        pending=sorted(pending),
                        blockers=blockers,
                    )

        return self._result(status="completed")

    def _ready_order(self, pending: set[str]) -> list[str]:
        ready = []
        for shard_id in self.spec["topo_order"]:
            if shard_id not in pending:
                continue
            shard = self.spec["shard_by_id"][shard_id]
            if all(dep in self.completed for dep in shard["dependencies"]):
                ready.append(shard_id)
        return ready

    def _mark_waiting(self, shard_id: str, reason: str) -> None:
        now = monotonic_ms()
        record = self.wait_records.setdefault(
            shard_id,
            {"shard_id": shard_id, "wait_started_ms": now, "wait_ended_ms": None, "reasons": []},
        )
        if not record["reasons"] or record["reasons"][-1] != reason:
            record["reasons"].append(reason)

    def _can_admit(self, shard: dict[str, Any]) -> dict[str, Any]:
        shard_id = shard["shard_id"]
        if not all(dep in self.completed for dep in shard["dependencies"]):
            return {"admit": False, "reason": "upstream_dependency_not_committed"}

        node_id = effective_node_id(shard, self.spec)
        node = self.spec["node_by_id"][node_id]
        resource_units = int(shard["resource_units"])
        active_leases = self.leases.active_count_for_node(node_id)
        active_shards = self.active_shards_by_node[node_id]

        if active_leases >= int(node["lease_limit"]):
            return {
                "admit": False,
                "reason": "node_lease_limit",
                "node_id": node_id,
                "active_leases": active_leases,
                "lease_limit": node["lease_limit"],
            }
        if active_shards >= int(node["concurrency"]):
            return {
                "admit": False,
                "reason": "node_concurrency_quota",
                "node_id": node_id,
                "active_shards": active_shards,
                "concurrency": node["concurrency"],
            }
        with self._lock:
            used = self.active_units
        if used + resource_units > self.budget:
            view = budget_boundary_view(
                used_units=used,
                demand_units=resource_units,
                budget_units=self.budget,
                scale=self.scale,
                active_term_count=self._total_active_shards_locked(),
            )
            return {
                "admit": False,
                "reason": "global_budget",
                "node_id": node_id,
                "active_units": used,
                "budget_units": self.budget,
                "boundary": view,
            }

        boundary = budget_boundary_view(
            used_units=used,
            demand_units=resource_units,
            budget_units=self.budget,
            scale=self.scale,
            active_term_count=self._total_active_shards_locked(),
        )
        return {
            "admit": True,
            "node_id": node_id,
            "active_leases_before": active_leases,
            "active_shards_before": active_shards,
            "active_units_before": used,
            "boundary": boundary,
        }

    def _total_active_shards_locked(self) -> int:
        return sum(self.active_shards_by_node.values())

    def _admit(
        self,
        shard: dict[str, Any],
        decision: dict[str, Any],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> dict[str, Any]:
        shard_id = shard["shard_id"]
        node_id = decision["node_id"]
        node = self.spec["node_by_id"][node_id]
        now = monotonic_ms()
        wait = self.wait_records.get(shard_id)
        if wait is None:
            wait = {
                "shard_id": shard_id,
                "wait_started_ms": now,
                "wait_ended_ms": now,
                "reasons": [],
            }
            self.wait_records[shard_id] = wait
        wait["wait_ended_ms"] = now

        lease = self.leases.grant(
            shard_id=shard_id,
            node_id=node_id,
            capacity_version=int(node["capacity_version"]),
            resource_units=int(shard["resource_units"]),
            ttl_ms=int(self.spec["scheduling"]["lease_ttl_ms"]),
        )
        lease_dict = lease.to_dict()
        self.ledger.record_lease_grant(lease_dict)
        self.cluster.grant_lease_on_node(lease_dict)

        with self._lock:
            self.active_units += int(shard["resource_units"])
            self.active_shards_by_node[node_id] += 1
            self.peak_active_units = max(self.peak_active_units, self.active_units)
            self.peak_shards_by_node[node_id] = max(
                self.peak_shards_by_node[node_id],
                self.active_shards_by_node[node_id],
            )

        request = request_payload_for_shard(
            self.spec,
            shard,
            request_id=f"{self.run_id}:{shard_id}:{lease.lease_version}",
            lease={"lease_id": lease.lease_id, "lease_version": lease.lease_version},
        )
        future = executor.submit(
            _execute_shard,
            self.spec,
            shard,
            lease_dict,
            request,
            self.cluster,
            self.ledger,
            dict(wait),
        )
        context = {
            "future": future,
            "shard": shard,
            "lease": lease_dict,
            "node_id": node_id,
            "admitted_monotonic_ms": now,
        }
        self.decisions.append(
            {
                "kind": "admit",
                "shard_id": shard_id,
                "topo_index": self.spec["topo_index"][shard_id],
                "node_id": node_id,
                "lease_id": lease.lease_id,
                "resource_units": shard["resource_units"],
                "water_q": quantize_units(int(shard["resource_units"]), self.scale).q,
                "wait": dict(wait),
                "boundary": decision["boundary"],
                "tie_break_rule": "topo_index_then_shard_id",
            }
        )
        return context

    def _consume_future(
        self,
        future: concurrent.futures.Future[dict[str, Any]],
        context: dict[str, Any],
    ) -> None:
        shard = context["shard"]
        lease_dict = context["lease"]
        node_id = context["node_id"]
        shard_id = shard["shard_id"]
        try:
            outcome = future.result()
            self.outcomes[shard_id] = outcome
            self.completed.add(shard_id)
            self.decisions.append({"kind": "complete", "shard_id": shard_id, "lease_id": lease_dict["lease_id"]})
        except Exception as exc:
            self.decisions.append(
                {
                    "kind": "failed",
                    "shard_id": shard_id,
                    "lease_id": lease_dict["lease_id"],
                    "error_code": getattr(exc, "code", "E_GSB"),
                    "error": str(exc),
                }
            )
            try:
                self.ledger.release_lease(
                    lease_dict,
                    reason="failed",
                    monotonic_ms=monotonic_ms(),
                )
            except Exception:
                pass
            raise
        finally:
            released = self.leases.release_once(
                lease_dict["lease_id"],
                "completed",
                status="completed" if shard_id in self.completed else "failed",
            )
            if released is not None:
                with self._lock:
                    self.active_units = max(0, self.active_units - int(shard["resource_units"]))
                    self.active_shards_by_node[node_id] = max(0, self.active_shards_by_node[node_id] - 1)

    def renew_lease(self, lease_id: str, expected_version: int, *, ttl_ms: int | None = None) -> dict[str, Any]:
        lease = self.leases.get(lease_id)
        response = self.cluster.renew_lease_on_node(
            node_id=lease.node_id,
            lease_id=lease_id,
            expected_version=expected_version,
            ttl_ms=ttl_ms or int(self.spec["scheduling"]["lease_ttl_ms"]),
        )
        node_record = response["lease"]
        renewed = self.leases.renew_locally(
            lease_id,
            expected_version,
            ttl_ms=ttl_ms,
            new_version=int(node_record["lease_version"]),
        )
        return renewed.to_dict()

    def _result(self, *, status: str) -> dict[str, Any]:
        lease_used = {
            node["node_id"]: self.leases.active_count_for_node(node["node_id"])
            for node in self.spec["nodes"]
        }
        water_report = unified_water_report(
            self.spec,
            used_units=self.active_units,
            lease_used=lease_used,
            shard_used=self.active_shards_by_node,
        )
        peak_water_report = unified_water_report(
            self.spec,
            used_units=self.peak_active_units,
            lease_used=self.peak_shards_by_node,
            shard_used=self.peak_shards_by_node,
        )
        return {
            "status": status,
            "run_id": self.run_id,
            "completed": sorted(self.completed),
            "outcomes": list(self.outcomes.values()),
            "decisions": self.decisions,
            "rounds": self.rounds,
            "wait_records": list(self.wait_records.values()),
            "leases": self.leases.all_leases(),
            "final_active_units": self.active_units,
            "water_report": water_report,
            "peak_water_report": peak_water_report,
        }


def _execute_shard(
    spec: dict[str, Any],
    shard: dict[str, Any],
    lease_dict: dict[str, Any],
    request: dict[str, Any],
    cluster: Any,
    ledger: Any,
    wait_record: dict[str, Any],
) -> dict[str, Any]:
    started = monotonic_ms()
    response = cluster.call(
        lease_dict["node_id"],
        request,
        timeout_ms=int(spec["scheduling"]["request_timeout_ms"]),
    )
    if response.get("shard_id") != shard["shard_id"]:
        raise GSBError("node response shard_id mismatch", field="shard_id", remote=response)
    summary = response["summary"]
    if summary.get("summary_digest") != response.get("summary_digest"):
        raise GSBError("node summary digest mismatch", field="summary_digest")

    finished = monotonic_ms()
    water = quantize_units(int(shard["resource_units"]), int(spec["scale_factor"]))
    outcome = {
        "shard_id": shard["shard_id"],
        "topo_index": int(spec["topo_index"][shard["shard_id"]]),
        "lease_id": lease_dict["lease_id"],
        "node_id": lease_dict["node_id"],
        "capacity_version": int(response.get("capacity_version", lease_dict["capacity_version"])),
        "lease_version": int(lease_dict["lease_version"]),
        "status": "committed",
        "summary": summary,
        "summary_digest": summary["summary_digest"],
        "resource_units": int(shard["resource_units"]),
        "water_q": water.q,
        "scale_factor": int(spec["scale_factor"]),
        "wait_started_ms": wait_record.get("wait_started_ms"),
        "wait_ended_ms": wait_record.get("wait_ended_ms"),
        "started_monotonic_ms": started,
        "finished_monotonic_ms": finished,
        "node_elapsed_ms": int(response.get("node_elapsed_ms", 0)),
        "request_digest": None,
        "propagation_basis": propagation_basis(spec, shard["shard_id"]),
    }
    ledger.commit_outcome(outcome, {**lease_dict, "status": "completed"})
    return outcome
