"""Thread-pool scheduler, dependency state, TCP execution, and ledger commits."""

from __future__ import annotations

import os
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .cluster import NodeCluster
from .deterministic import (
    compute_shard_summary_local,
    derivation_basis,
    generate_samples,
    merge_partials,
    sample_digest,
    bind_summary,
    samples_to_wire,
)
from .errors import DiagnosticError, LeaseError
from .fixedpoint import (
    attribute_errors,
    budget_margin_decision,
    global_budget_error,
    quantize_node_water,
    quantize_shard_demand,
    water_gap_decision,
)
from .leases import LeaseManager, monotonic_ms
from .ledger import Ledger
from .protocol import validate_compute_receipt
from .spec import Spec


READY_TERMINAL = {"succeeded"}
BAD_UPSTREAM = {"failed", "cancelled", "frozen", "lease_lost"}
TRANSIENT = {"pending", "unknown", "queued", "leased", "running", "dispatching"}


class Engine:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        cluster: NodeCluster | None = None,
        *,
        batch_size: int | None = None,
        kill_after_terminal_commits: int | None = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.cluster = cluster
        self.batch_size_override = batch_size
        self.kill_after_terminal_commits = kill_after_terminal_commits
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.lease_manager = LeaseManager()
        self.states: dict[str, dict[str, Any]] = {}
        self.futures: dict[Future[None], str] = {}
        self.draining_units: dict[str, int] = {}
        self.decisions: list[dict[str, Any]] = []
        self.renewal_results: list[dict[str, Any]] = []
        self.terminal_commits = 0
        self.peak_used_units = 0
        self.peak_dependency_wait_units = 0
        self.peak_node_active: dict[str, dict[str, int]] = {
            node_id: {"active_shards": 0, "active_leases": 0, "used_units": 0}
            for node_id in spec.nodes
        }
        self._initialize_states()

    def _new_state(self, shard_id: str, status: str = "pending") -> dict[str, Any]:
        shard = self.spec.shards[shard_id]
        return {
            "shard_id": shard_id,
            "status": status,
            "dispatch_count": 0,
            "lease_id": None,
            "node_id": shard["node_id"],
            "wait_started_monotonic_ms": None,
            "wait_ended_monotonic_ms": None,
            "wait_ms": None,
            "block_reasons": [],
            "missing_upstream_ids": [],
            "summary": None,
            "basis": None,
            "batch_receipts": [],
            "renewals": [],
            "error": None,
            "water": None,
        }

    def _recover_if_needed(self) -> dict[str, Any] | None:
        rows = self.ledger.shard_states()
        active = self.ledger.active_leases()
        transient = any(row["status"] in {"leased", "running", "dispatching"} for row in rows.values())
        if not active and not transient:
            return None
        recovery = self.ledger.recover_interrupted()
        return recovery

    def _initialize_states(self) -> None:
        recovery = self._recover_if_needed()
        self.recovery = recovery
        committed = self.ledger.shard_states()
        arrived: set[str] | None = None
        control_status: dict[str, str] = {}
        control_events: list[dict[str, Any]] = []
        for event in self.spec.event_list:
            if event["type"] == "shard_arrived":
                if arrived is None:
                    arrived = set()
                arrived.add(event["shard_id"])
            elif event["type"] in {"cancel", "fail_shard"}:
                shard_id = event["shard_id"]
                if shard_id not in control_status:
                    control_status[shard_id] = "cancelled" if event["type"] == "cancel" else "failed"
                control_events.append(event)

        self.arrived = arrived
        control_upserts: list[dict[str, Any]] = []
        for shard_id in self.spec.topo_order:
            if shard_id in committed and committed[shard_id]["status"] == "succeeded":
                row = committed[shard_id]
                state = self._new_state(shard_id, "succeeded")
                state["summary"] = row.get("summary")
                state["lease_id"] = row.get("lease_id")
                self.states[shard_id] = state
                continue
            if shard_id in control_status:
                status = control_status[shard_id]
                state = self._new_state(shard_id, status)
                state["error"] = f"terminal_control_event:{status}"
                self.states[shard_id] = state
                control_upserts.append(
                    {
                        "shard_id": shard_id,
                        "status": status,
                        "node_id": self.spec.shards[shard_id]["node_id"],
                        "capacity_version": self.spec.nodes[self.spec.shards[shard_id]["node_id"]]["capacity_version"],
                        "resource_units": 0,
                    }
                )
                continue
            if shard_id in committed and committed[shard_id]["status"] == "frozen":
                state = self._new_state(shard_id, "frozen")
                state["missing_upstream_ids"] = []
                self.states[shard_id] = state
                continue
            self.states[shard_id] = self._new_state(shard_id, "pending")

        if control_upserts:
            self.ledger.commit_batch(
                "control_terminal_events",
                {"events": [{"seq": e["seq"], "type": e["type"], "shard_id": e["shard_id"]} for e in control_events]},
                shard_states=control_upserts,
            )

    def run(self) -> dict[str, Any]:
        worker_count = max(1, sum(int(node["concurrency_quota"]) for node in self.spec.nodes.values()))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="shard-worker") as executor:
            while True:
                admissions = self._schedule_pass(executor)
                if admissions:
                    continue
                with self.condition:
                    self._prune_futures_locked()
                    if self.futures:
                        self.condition.wait(timeout=0.02)
                        continue
                    self._finalize_unrunnable_locked()
                    self._prune_futures_locked()
                    if self.futures:
                        continue
                    break
        return self.build_report()

    def _schedule_pass(self, executor: ThreadPoolExecutor) -> list[str]:
        admissions: list[str] = []
        while True:
            with self.condition:
                self._prune_futures_locked()
            admission = self._reserve_next()
            if admission is None:
                break
            shard_id, lease = admission
            try:
                assert self.cluster is not None
                with self.cluster.client(self.states[shard_id]["node_id"]) as client:
                    client.lease_grant(
                        {
                            "lease_id": lease["lease_id"],
                            "shard_id": shard_id,
                            "capacity_version": lease["capacity_version"],
                            "granted_monotonic_ms": lease["granted_monotonic_ms"],
                            "expires_monotonic_ms": lease["expires_monotonic_ms"],
                            "ttl_ms": int(self.spec.raw["lease_ttl_ms"]),
                        }
                    )
            except DiagnosticError as exc:
                with self.condition:
                    self.lease_manager.release(lease["lease_id"], reason="node_grant_failed")
                    state = self.states[shard_id]
                    state["status"] = "failed"
                    state["error"] = exc.to_dict()
                    self.ledger.commit_batch(
                        "shard_failed",
                        {"shard_id": shard_id, "error": exc.to_dict()},
                        shard_states=[self._shard_row_locked(shard_id, status="failed")],
                        lease_states=[{**lease, "status": "released"}],
                    )
                    self.condition.notify_all()
                continue

            with self.condition:
                state = self.states[shard_id]
                state["status"] = "running"
                state["dispatch_count"] += 1
                future = executor.submit(self._run_shard_safely, shard_id, lease["lease_id"])
                self.futures[future] = shard_id
                admissions.append(shard_id)
        return admissions

    def _prune_futures_locked(self) -> None:
        completed = [future for future in self.futures if future.done()]
        for future in completed:
            self.futures.pop(future, None)

    def _reserve_next(self) -> tuple[str, dict[str, Any]] | None:
        with self.condition:
            self._prune_futures_locked()
            expired = self.lease_manager.reconcile_expired()
            for lease in expired:
                shard_id = lease["shard_id"]
                state = self.states.get(shard_id)
                if state and state["status"] not in {"succeeded", "failed", "cancelled"}:
                    state["status"] = "draining"
                    state["error"] = "monotonic lease expiry before terminal commit"
                    self.draining_units[shard_id] = int(lease["resource_units"])

            candidates, water_decision = self._ready_candidates_locked()
            if water_decision is not None:
                self.decisions.append(water_decision)
            if not candidates:
                return None
            winner = candidates[0]
            shard = self.spec.shards[winner]
            node = self.spec.nodes[shard["node_id"]]
            now = monotonic_ms()
            lease = self.lease_manager.grant(
                shard_id=winner,
                node_id=shard["node_id"],
                capacity_version=int(node["capacity_version"]),
                resource_units=int(shard["resource_units"]),
                lease_limit=int(node["lease_limit"]),
                ttl_ms=int(self.spec.raw["lease_ttl_ms"]),
                now_ms=now,
            )
            used = self.lease_manager.used_units() + sum(self.draining_units.values())
            self.peak_used_units = max(self.peak_used_units, used)
            node_active = self.peak_node_active[shard["node_id"]]
            node_active["active_leases"] = self.lease_manager.active_leases_on_node(shard["node_id"])
            node_active["used_units"] = self.lease_manager.used_units_on_node(shard["node_id"])
            node_active["active_shards"] = sum(
                1
                for item in self.states.values()
                if item["node_id"] == shard["node_id"]
                and item["status"] in {"running", "leased", "draining"}
            ) + 1
            node_active["active_leases"] = max(node_active["active_leases"], 1)
            node_active["active_shards"] = max(node_active["active_shards"], 1)

            state = self.states[winner]
            state["status"] = "leased"
            state["lease_id"] = lease["lease_id"]
            if state["wait_started_monotonic_ms"] is not None:
                state["wait_ended_monotonic_ms"] = now
                state["wait_ms"] = max(0, now - int(state["wait_started_monotonic_ms"]))
            self.ledger.commit_batch(
                "lease_granted",
                {"lease": lease, "wait_ms": state["wait_ms"]},
                shard_states=[self._shard_row_locked(winner, status="running")],
                lease_states=[lease],
            )
            return winner, lease

    def _ready_candidates_locked(self) -> tuple[list[str], dict[str, Any] | None]:
        candidates: list[str] = []
        candidate_water: dict[str, int] = {}
        block_snapshot: dict[str, list[str]] = {}
        dependency_wait_units = 0

        for shard_id in self.spec.topo_order:
            state = self.states[shard_id]
            if state["status"] in {"succeeded", "failed", "cancelled", "frozen", "running", "leased"}:
                continue
            if state["dispatch_count"] > 0:
                continue
            if self.arrived is not None and shard_id not in self.arrived:
                continue

            shard = self.spec.shards[shard_id]
            missing: list[str] = []
            unresolved: list[str] = []
            for upstream in shard["depends_on"]:
                upstream_state = self.states[upstream]
                if upstream_state["status"] in BAD_UPSTREAM:
                    missing.append(upstream)
                elif upstream_state["status"] not in READY_TERMINAL or upstream_state.get("summary") is None:
                    unresolved.append(upstream)

            if missing:
                state["status"] = "frozen"
                state["missing_upstream_ids"] = sorted(set(missing))
                state["block_reasons"] = [f"upstream_not_successful:{up}" for up in missing]
                self.ledger.commit_batch(
                    "shard_frozen",
                    {"shard_id": shard_id, "missing_upstream_ids": state["missing_upstream_ids"]},
                    shard_states=[self._shard_row_locked(shard_id, status="frozen")],
                )
                continue
            if unresolved:
                dependency_wait_units += int(shard["resource_units"])
                state["block_reasons"] = [f"waiting_upstream:{up}" for up in unresolved]
                continue

            reasons = self._capacity_block_reasons_locked(shard_id)
            state["block_reasons"] = reasons
            if reasons:
                if state["wait_started_monotonic_ms"] is None:
                    state["wait_started_monotonic_ms"] = monotonic_ms()
                state["status"] = "queued"
                block_snapshot[shard_id] = reasons
                continue

            state["status"] = "pending"
            candidates.append(shard_id)
            candidate_water[shard_id] = int(
                quantize_shard_demand(
                    shard_id=shard_id,
                    resource_units=int(shard["resource_units"]),
                    budget_units=int(self.spec.raw["global_budget_units"]),
                    scale_power=int(self.spec.raw["session_scale"]),
                )["budget_pressure_q"]
            )

        self.peak_dependency_wait_units = max(
            self.peak_dependency_wait_units, dependency_wait_units
        )
        if not candidates:
            return [], None

        candidates.sort(key=lambda sid: (-int(self.spec.shards[sid]["priority"]), self.spec.topo_rank[sid], sid))
        used_units = self.lease_manager.used_units() + sum(self.draining_units.values())
        attribution = attribute_errors(
            candidate_count=len(candidates),
            budget_units=int(self.spec.raw["global_budget_units"]),
            scale_power=int(self.spec.raw["session_scale"]),
            lease_attribution_units=0,
            dependency_wait_units=0,
        )
        winner = candidates[0]
        budget_decision = budget_margin_decision(
            used_units=used_units,
            candidate_units=int(self.spec.shards[winner]["resource_units"]),
            budget_units=int(self.spec.raw["global_budget_units"]),
            attribution=attribution,
        )
        pair_decision = None
        if len(candidates) > 1:
            runner_up = candidates[1]
            pair_decision = water_gap_decision(
                water_a_q=candidate_water[winner],
                water_b_q=candidate_water[runner_up],
                attribution=attribution,
            )
        same_priority_close = False
        if pair_decision and pair_decision["within_synthetic_error_bound"]:
            close_ids = [
                sid
                for sid in candidates
                if int(self.spec.shards[sid]["priority"]) == int(self.spec.shards[winner]["priority"])
                and abs(candidate_water[sid] - candidate_water[winner]) <= int(attribution["synthetic_error_bound_q"])
            ]
            tied_winner = min(close_ids, key=lambda sid: (self.spec.topo_rank[sid], sid))
            same_priority_close = tied_winner == winner
            if tied_winner != winner:
                candidates.remove(tied_winner)
                candidates.insert(0, tied_winner)
                winner = tied_winner
        decision = {
            "at_monotonic_ms": monotonic_ms(),
            "candidate_order_before_tie_rule": candidates,
            "selected": winner,
            "used_units": used_units,
            "attribution": attribution,
            "budget": budget_decision,
            "water_pair": pair_decision,
            "block_reasons": block_snapshot,
            "decision_affected_by_error": bool(
                budget_decision["decision_affected_by_error"]
                or (pair_decision and pair_decision["decision_affected_by_error"])
            ),
            "tie_rule_applied": "priority_then_close_water_set_min_topological_rank_and_shard_id",
            "same_priority_close": same_priority_close,
        }
        for sid in candidates:
            self.states[sid]["water"] = quantize_shard_demand(
                shard_id=sid,
                resource_units=int(self.spec.shards[sid]["resource_units"]),
                budget_units=int(self.spec.raw["global_budget_units"]),
                scale_power=int(self.spec.raw["session_scale"]),
            )
        return candidates, decision

    def _capacity_block_reasons_locked(self, shard_id: str) -> list[str]:
        shard = self.spec.shards[shard_id]
        node = self.spec.nodes[shard["node_id"]]
        reasons: list[str] = []
        occupied_units = self.lease_manager.used_units() + sum(self.draining_units.values())
        if occupied_units + int(shard["resource_units"]) > int(self.spec.raw["global_budget_units"]):
            reasons.append("global_budget")
        active_leases = self.lease_manager.active_leases_on_node(shard["node_id"])
        if active_leases + 1 > int(node["lease_limit"]):
            reasons.append("node_lease_limit")
        active_shards = sum(
            1
            for state in self.states.values()
            if state["node_id"] == shard["node_id"]
            and state["status"] in {"leased", "running", "draining"}
        )
        if active_shards + 1 > int(node["concurrency_quota"]):
            reasons.append("node_concurrency_quota")
        return reasons

    def _shard_row_locked(self, shard_id: str, *, status: str, **overrides: Any) -> dict[str, Any]:
        state = self.states[shard_id]
        shard = self.spec.shards[shard_id]
        node = self.spec.nodes[shard["node_id"]]
        row = {
            "shard_id": shard_id,
            "status": status,
            "lease_id": state.get("lease_id"),
            "node_id": shard["node_id"],
            "capacity_version": int(node["capacity_version"]),
            "resource_units": int(shard["resource_units"]),
            "summary": state.get("summary"),
            "summary_digest": (state.get("summary") or {}).get("summary_digest"),
        }
        row.update(overrides)
        return row

    def _run_shard_safely(self, shard_id: str, lease_id: str) -> None:
        try:
            self._run_shard(shard_id, lease_id)
        except Exception as exc:  # ensure every worker failure reaches ledger
            error = exc if isinstance(exc, DiagnosticError) else DiagnosticError("E_PROTO", repr(exc))
            self._fail_shard(shard_id, lease_id, error)

    def _run_shard(self, shard_id: str, lease_id: str) -> None:
        assert self.cluster is not None
        shard = self.spec.shards[shard_id]
        node = self.spec.nodes[shard["node_id"]]
        batch_size = int(self.batch_size_override or shard.get("batch_rows") or self.spec.raw.get("batch_rows", 16))
        samples = generate_samples(int(node["seed"]), shard_id, int(shard["sample_rows"]))
        digest_of_samples = sample_digest(samples)
        partials: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []

        for batch_index, start in enumerate(range(0, int(shard["sample_rows"]), batch_size)):
            end = min(int(shard["sample_rows"]), start + batch_size)
            wire_samples = samples_to_wire(samples, start, end)
            request_id = f"{shard_id}:{lease_id}:{batch_index}"
            with self.cluster.client(shard["node_id"]) as client:
                receipt = client.compute(
                    {
                        "request_id": request_id,
                        "node_id": shard["node_id"],
                        "shard_id": shard_id,
                        "capacity_version": int(node["capacity_version"]),
                        "seed": int(node["seed"]),
                        "batch_index": batch_index,
                        "batch_start_row": start,
                        "batch_end_row": end,
                        "samples": wire_samples,
                    }
                )
            validate_compute_receipt(
                receipt, expected_node_id=shard["node_id"], expected_shard_id=shard_id
            )
            partials.append(receipt["partial"])
            receipts.append(
                {
                    "request_id": request_id,
                    "batch_index": batch_index,
                    "elapsed_monotonic_ms": receipt["elapsed_monotonic_ms"],
                    "partial_digest": receipt["summary"]["partial_digest"],
                }
            )

        self._apply_renewal_events(shard_id)

        merged = merge_partials(partials)
        summary = bind_summary(
            merged,
            node_id=shard["node_id"],
            shard_id=shard_id,
            sample_rows=int(shard["sample_rows"]),
            capacity_version=int(node["capacity_version"]),
            digest_of_samples=digest_of_samples,
            batch_size=batch_size,
        )
        basis = derivation_basis(
            node_id=shard["node_id"],
            node_seed=int(node["seed"]),
            shard_id=shard_id,
            sample_rows=int(shard["sample_rows"]),
            capacity_version=int(node["capacity_version"]),
            fixed_point_decimals=int(node["fixed_point_decimals"]),
            batch_size=batch_size,
            samples=samples,
            summary=summary,
        )

        current_lease = self.lease_manager.get(lease_id)
        if current_lease is None or current_lease["status"] != "active":
            raise LeaseError(
                "lease is no longer active; computed result is fenced and not committed",
                lease_id=lease_id,
                basis_generation=int((current_lease or {}).get("lease_version", 0)),
            )

        try:
            with self.cluster.client(shard["node_id"]) as client:
                client.lease_release({"lease_id": lease_id, "shard_id": shard_id})
        except DiagnosticError:
            pass

        with self.condition:
            released = self.lease_manager.release(lease_id, reason="shard_succeeded")
            self.draining_units.pop(shard_id, None)
            state = self.states[shard_id]
            state["status"] = "succeeded"
            state["summary"] = summary
            state["basis"] = basis
            state["batch_receipts"] = receipts
            state["wait_ended_monotonic_ms"] = monotonic_ms()
            self.terminal_commits += 1
            row = self._shard_row_locked(shard_id, status="succeeded")
            self.ledger.commit_batch(
                "shard_succeeded",
                {
                    "shard_id": shard_id,
                    "summary_digest": summary["summary_digest"],
                    "sample_digest": digest_of_samples,
                    "batch_size": batch_size,
                },
                shard_states=[row],
                lease_states=[released or {"lease_id": lease_id, "status": "released", "shard_id": shard_id, "node_id": shard["node_id"]}],
            )
            self.condition.notify_all()
            if (
                self.kill_after_terminal_commits is not None
                and self.terminal_commits >= self.kill_after_terminal_commits
            ):
                os.kill(os.getpid(), signal.SIGKILL)

    def _apply_renewal_events(self, shard_id: str) -> None:
        assert self.cluster is not None
        events = [
            event
            for event in self.spec.event_list
            if event["type"] == "renew_lease" and event["shard_id"] == shard_id
        ]
        for event in sorted(events, key=lambda item: item["seq"]):
            with self.condition:
                state = self.states[shard_id]
                lease_id = state["lease_id"]
                expected_version = int(event["expected_version"])
            if not lease_id:
                continue
            try:
                with self.cluster.client(state["node_id"]) as client:
                    response = client.lease_renew(
                        {
                            "lease_id": lease_id,
                            "shard_id": shard_id,
                            "expected_version": expected_version,
                            "max_renewals": int(self.spec.raw["max_renewals"]),
                            "ttl_ms": int(self.spec.raw["lease_ttl_ms"]),
                        }
                    )
                with self.condition:
                    renewed = self.lease_manager.renew_local(
                        lease_id,
                        expected_version,
                        max_renewals=int(self.spec.raw["max_renewals"]),
                        ttl_ms=int(self.spec.raw["lease_ttl_ms"]),
                    )
                    self.ledger.commit_batch(
                        "lease_renewed",
                        {"event_seq": event["seq"], "response": response},
                        lease_states=[renewed],
                    )
                    self.states[shard_id]["renewals"].append(response)
                    self.renewal_results.append(response)
            except DiagnosticError as exc:
                with self.condition:
                    self.states[shard_id]["renewal_error"] = exc.to_dict()
                    self.renewal_results.append(exc.to_dict())
                raise

    def _fail_shard(self, shard_id: str, lease_id: str, error: DiagnosticError) -> None:
        assert self.cluster is not None
        try:
            with self.cluster.client(self.states[shard_id]["node_id"]) as client:
                client.lease_release({"lease_id": lease_id, "shard_id": shard_id})
        except DiagnosticError:
            pass
        with self.condition:
            lease = self.lease_manager.release(lease_id, reason="shard_failed")
            self.draining_units.pop(shard_id, None)
            state = self.states[shard_id]
            state["status"] = "failed"
            state["error"] = error.to_dict()
            self.terminal_commits += 1
            self.ledger.commit_batch(
                "shard_failed",
                {"shard_id": shard_id, "error": error.to_dict()},
                shard_states=[self._shard_row_locked(shard_id, status="failed")],
                lease_states=[lease or {"lease_id": lease_id, "status": "released", "shard_id": shard_id, "node_id": state["node_id"]}],
            )
            self.condition.notify_all()

    def _finalize_unrunnable_locked(self) -> None:
        for shard_id in self.spec.topo_order:
            state = self.states[shard_id]
            if state["status"] in {"succeeded", "failed", "cancelled", "frozen"}:
                continue
            if state["dispatch_count"] > 0:
                continue
            reasons = self._capacity_block_reasons_locked(shard_id)
            missing: list[str] = []
            for upstream in self.spec.shards[shard_id]["depends_on"]:
                if self.states[upstream]["status"] in BAD_UPSTREAM:
                    missing.append(upstream)
            if missing:
                state["status"] = "frozen"
                state["missing_upstream_ids"] = sorted(set(missing))
                self.ledger.commit_batch(
                    "shard_frozen",
                    {"shard_id": shard_id, "missing_upstream_ids": state["missing_upstream_ids"]},
                    shard_states=[self._shard_row_locked(shard_id, status="frozen")],
                )
            elif reasons:
                state["status"] = "blocked"
                state["block_reasons"] = reasons

    def build_water_report(self) -> dict[str, Any]:
        budget = int(self.spec.raw["global_budget_units"])
        power = int(self.spec.raw["session_scale"])
        per_shard = [
            quantize_shard_demand(
                shard_id=shard_id,
                resource_units=int(shard["resource_units"]),
                budget_units=budget,
                scale_power=power,
            )
            for shard_id, shard in sorted(self.spec.shards.items())
        ]
        per_node = []
        for node_id in sorted(self.spec.nodes):
            node = self.spec.nodes[node_id]
            peak = self.peak_node_active[node_id]
            per_node.append(
                quantize_node_water(
                    node_id=node_id,
                    used_units=peak["used_units"],
                    active_shards=int(peak["active_shards"]),
                    active_leases=int(peak["active_leases"]),
                    global_budget_units=budget,
                    concurrency_quota=int(node["concurrency_quota"]),
                    lease_limit=int(node["lease_limit"]),
                    scale_power=power,
                )
            )
        attribution = attribute_errors(
            candidate_count=len(self.spec.shards),
            budget_units=budget,
            scale_power=power,
            lease_attribution_units=0,
            dependency_wait_units=self.peak_dependency_wait_units,
        )
        for node_water in per_node:
            node_water["synthetic_comparison_error_bound_q"] = attribution[
                "synthetic_error_bound_q"
            ]
            node_water["decision_threshold_resource_units"] = attribution[
                "decision_threshold_resource_units"
            ]
        return {
            "per_shard_quantization": per_shard,
            "per_node_water": per_node,
            "global_budget_aggregation": global_budget_error(
                per_shard, shard_count=len(self.spec.shards), scale_power=power
            ),
            "error_attribution": attribution,
            "peak_used_units": self.peak_used_units,
            "peak_dependency_wait_units": self.peak_dependency_wait_units,
            "note": "Exact integer budget and quota checks remain authoritative; error bounds only flag ambiguous water comparisons.",
        }

    def build_report(self) -> dict[str, Any]:
        shard_reports: dict[str, Any] = {}
        for shard_id, state in sorted(self.states.items()):
            shard_reports[shard_id] = {
                "status": state["status"],
                "node_id": state["node_id"],
                "dispatch_count": state["dispatch_count"],
                "wait_started_monotonic_ms": state["wait_started_monotonic_ms"],
                "wait_ended_monotonic_ms": state["wait_ended_monotonic_ms"],
                "wait_ms": state["wait_ms"],
                "block_reasons": state["block_reasons"],
                "missing_upstream_ids": state["missing_upstream_ids"],
                "summary_digest": (state.get("summary") or {}).get("summary_digest"),
                "sample_digest": (state.get("basis") or {}).get("sample_digest"),
                "renewals": state["renewals"],
                "renewal_error": state.get("renewal_error"),
                "error": state["error"],
                "batch_receipts": state["batch_receipts"],
            }
        return {
            "ok": all(state["status"] == "succeeded" for state in self.states.values()),
            "spec_digest": self.spec.spec_digest,
            "recovery": self.recovery,
            "shards": shard_reports,
            "decisions": self.decisions,
            "water_report": self.build_water_report(),
            "ledger_summary": self.ledger.canonical_accounting_summary(),
            "ledger_batch_count": self.ledger.batch_count(),
        }

    def summary_records(self) -> dict[str, dict[str, Any]]:
        return {
            shard_id: {
                "summary": state["summary"],
                "capacity_version": (state.get("summary") or {}).get("capacity_version"),
                "node_id": state["node_id"],
            }
            for shard_id, state in self.states.items()
            if state.get("summary") is not None
        }

    def explain_shard(self, shard_id: str) -> dict[str, Any]:
        state = self.states[shard_id]
        if state.get("basis") is None:
            raise DiagnosticError("E_PROTO", f"no derivation basis for {shard_id}")
        return {"basis": state["basis"], "summary": state["summary"]}


def recompute_invalid_subset_local(
    spec: Spec, invalid_ids: list[str], *, target_capacity_version: int | None = None
) -> dict[str, dict[str, Any]]:
    """Recompute only an invalid subset using the explicit deterministic rule."""

    records: dict[str, dict[str, Any]] = {}
    for shard_id in invalid_ids:
        shard = spec.shards[shard_id]
        node = spec.nodes[shard["node_id"]]
        version = int(target_capacity_version or node["capacity_version"])
        decimals = int(
            spec.capacity_versions[version]["node_calibers"][shard["node_id"]][
                "fixed_point_decimals"
            ]
        )
        batch_size = int(shard.get("batch_rows") or spec.raw.get("batch_rows", 16))
        _, summary, basis = compute_shard_summary_local(
            node_id=shard["node_id"],
            node_seed=int(node["seed"]),
            shard_id=shard_id,
            sample_rows=int(shard["sample_rows"]),
            capacity_version=version,
            fixed_point_decimals=decimals,
            batch_size=batch_size,
        )
        records[shard_id] = {"summary": summary, "basis": basis}
    return records
