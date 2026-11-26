from __future__ import annotations

from collections import defaultdict
from typing import Any

from .errors import GenerationChainError, StateTransitionError
from .generation import audit_generation_chain
from .fixedpoint import (
    attribute_error_segments,
    global_aggregation_bound_fp,
    lease_quantization_error,
    make_scale,
    node_comparison_bound_fp,
    saturation_rule,
    water_level_fp,
    within_integer_bound,
)
from .model import (
    ACTIVE_STATES,
    STATE_DECLARED,
    STATE_GRANTED,
    STATE_QUEUED,
    STATE_RELEASED,
    STATE_RENEWED,
    STATE_REVOKED,
    LeaseRuntime,
    Transition,
    transition_allowed,
)


class DomainState:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.scale = make_scale(int(spec.get("scale", 0)))
        self.factor = self.scale.factor
        self.node_ids = sorted(node["node_id"] for node in spec["nodes"])
        self.node_config = {node["node_id"]: node for node in spec["nodes"]}
        self.version_policies = {
            item["quota_version"]: item for item in spec.get("quota_versions", [])
        }
        self.current_version = int(
            spec.get(
                "current_quota_version",
                max(node["quota_version"] for node in spec["nodes"]),
            )
        )
        self.global_budget = int(spec["global_budget"])
        self.node_quota = {node["node_id"]: int(node["quota"]) for node in spec["nodes"]}
        self.node_max_active = {
            node["node_id"]: int(node["max_active"]) for node in spec["nodes"]
        }
        self.leases: dict[str, LeaseRuntime] = {}
        for raw in spec["leases"]:
            self.leases[raw["lease_id"]] = LeaseRuntime(
                lease_id=raw["lease_id"],
                node_id=raw["node_id"],
                units=int(raw["units"]),
                priority=int(raw["priority"]),
                hold_ms=int(raw["hold_ms"]),
                quota_version=int(raw.get("quota_version", 1)),
                references=list(raw.get("references", [])),
            )
        self.pending: list[str] = []
        self.global_used = 0
        self.node_used: dict[str, int] = defaultdict(int)
        self.node_active: dict[str, int] = defaultdict(int)
        self.decisions: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.invariant_violations: list[dict[str, Any]] = []
        self.stale_generation_events = 0
        self.order_inversions = 0
        self.view_replacements: list[dict[str, Any]] = []
        self.stale_archives: list[dict[str, Any]] = []
        self.check_invariants()

    def check_invariants(self) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        if self.global_used < 0 or self.global_used > self.global_budget:
            violations.append(
                {
                    "scope": "global",
                    "used": self.global_used,
                    "budget": self.global_budget,
                }
            )
        for node_id in self.node_ids:
            if self.node_used[node_id] < 0 or self.node_used[node_id] > self.node_quota[node_id]:
                violations.append(
                    {
                        "scope": "node_quota",
                        "node_id": node_id,
                        "used": self.node_used[node_id],
                        "quota": self.node_quota[node_id],
                    }
                )
            if self.node_active[node_id] < 0 or self.node_active[node_id] > self.node_max_active[node_id]:
                violations.append(
                    {
                        "scope": "node_concurrency",
                        "node_id": node_id,
                        "active": self.node_active[node_id],
                        "max_active": self.node_max_active[node_id],
                    }
                )
        self.invariant_violations.extend(violations)
        return violations

    def transition(
        self,
        lease: LeaseRuntime,
        to_state: str,
        reason: str,
        now_ms: int,
        *,
        event_seq: int | None = None,
        generation_used: int | None = None,
        generation_before: int | None = None,
        generation_after: int | None = None,
    ) -> Transition:
        if not transition_allowed(lease.state, to_state):
            raise StateTransitionError(
                f"illegal transition {lease.state} -> {to_state} for {lease.lease_id}",
                path=f"$.leases[?lease_id={lease.lease_id}].state",
                details={"from_state": lease.state, "to_state": to_state, "reason": reason},
            )
        item = Transition(
            lease_id=lease.lease_id,
            from_state=lease.state,
            to_state=to_state,
            reason=reason,
            monotonic_ms=int(now_ms),
            event_seq=event_seq,
            generation_used=generation_used,
            generation_before=generation_before,
            generation_after=generation_after,
        )
        lease.history.append(item)
        lease.state = to_state
        lease.last_event_seq = event_seq
        return item

    def record_attempt(self, payload: dict[str, Any]) -> None:
        self.attempts.append(payload)

    def capacity_for(self, lease: LeaseRuntime) -> dict[str, Any]:
        global_remaining = self.global_budget - self.global_used
        node_remaining = self.node_quota[lease.node_id] - self.node_used[lease.node_id]
        active = self.node_active[lease.node_id]
        max_active = self.node_max_active[lease.node_id]
        concurrency_remaining = max_active - active
        fits = (
            global_remaining >= lease.units
            and node_remaining >= lease.units
            and concurrency_remaining >= 1
        )
        return {
            "fits": fits,
            "global_used": self.global_used,
            "global_budget": self.global_budget,
            "global_remaining": global_remaining,
            "global_margin": global_remaining - lease.units,
            "node_id": lease.node_id,
            "node_used": self.node_used[lease.node_id],
            "node_quota": self.node_quota[lease.node_id],
            "node_remaining": node_remaining,
            "node_margin": node_remaining - lease.units,
            "node_active": active,
            "node_max_active": max_active,
            "concurrency_remaining": concurrency_remaining,
            "concurrency_margin": concurrency_remaining - 1,
        }

    def quota_snapshot(self, lease: LeaseRuntime) -> dict[str, Any]:
        cap = self.capacity_for(lease)
        return {
            "global_used": self.global_used,
            "global_remaining": self.global_budget - self.global_used,
            "node_used": self.node_used[lease.node_id],
            "node_remaining": self.node_quota[lease.node_id] - self.node_used[lease.node_id],
            "capacity": cap,
        }

    def register_apply(self, lease_id: str, now_ms: int, event_seq: int) -> LeaseRuntime:
        lease = self.leases[lease_id]
        if lease.state == STATE_DECLARED:
            self.transition(
                lease,
                STATE_QUEUED,
                "apply_received_wait_for_capacity",
                now_ms,
                event_seq=event_seq,
            )
        if lease_id not in self.pending and lease.state == STATE_QUEUED:
            self.pending.append(lease_id)
            lease.wait_start_ms = now_ms
            lease.wait_end_ms = None
        lease.arrival_seq = event_seq
        return lease

    def projected_water(self, lease: LeaseRuntime) -> dict[str, Any]:
        node = water_level_fp(
            self.node_used[lease.node_id] + lease.units,
            self.node_quota[lease.node_id],
            self.scale,
        )
        glob = water_level_fp(
            self.global_used + lease.units,
            self.global_budget,
            self.scale,
        )
        return {
            "node_level_fp": node["level_fp"],
            "global_level_fp": glob["level_fp"],
            "water_index_fp": node["level_fp"] + glob["level_fp"],
            "node_report": node,
            "global_report": glob,
        }

    def error_budget(self) -> dict[str, Any]:
        return attribute_error_segments(
            quantization_ulp=4,
            scheduling_order_ulp=self.order_inversions,
            generation_reorder_ulp=self.stale_generation_events,
            scale=self.scale,
        )

    def choose_winner(self) -> tuple[str | None, dict[str, Any] | None]:
        feasible = [
            lease_id
            for lease_id in self.pending
            if self.capacity_for(self.leases[lease_id])["fits"]
        ]
        if not feasible:
            return None, None
        budget = self.error_budget()
        bound = int(budget["total_bound_ulp"])
        scored: list[dict[str, Any]] = []
        for lease_id in feasible:
            lease = self.leases[lease_id]
            water = self.projected_water(lease)
            cap = self.capacity_for(lease)
            margin_q = min(abs(cap["global_margin"]), abs(cap["node_margin"])) * self.factor
            budget_boundary = within_integer_bound(margin_q, bound)
            scored.append(
                {
                    "lease_id": lease_id,
                    "water_index_fp": water["water_index_fp"],
                    "node_level_fp": water["node_level_fp"],
                    "global_level_fp": water["global_level_fp"],
                    "budget_boundary": budget_boundary,
                    "minimum_abs_margin_units": min(
                        abs(cap["global_margin"]), abs(cap["node_margin"])
                    ),
                    "capacity": cap,
                    "water": water,
                }
            )
        min_water = min(item["water_index_fp"] for item in scored)
        indistinguishable = [
            item for item in scored if item["water_index_fp"] - min_water <= bound
        ]
        budget_boundary_items = [item for item in scored if item["budget_boundary"]]
        boundary_set = {item["lease_id"] for item in indistinguishable}
        boundary_affected = len(indistinguishable) > 1 or bool(budget_boundary_items)
        if budget_boundary_items:
            boundary_set.update(item["lease_id"] for item in budget_boundary_items)

        def tie_key(lease_id: str) -> tuple[int, int, str]:
            lease = self.leases[lease_id]
            return (-int(lease.quota_version), -int(lease.priority), lease.lease_id)

        decision_pool = {item["lease_id"] for item in indistinguishable}
        decision_pool.update(item["lease_id"] for item in budget_boundary_items)
        winner = min(decision_pool, key=tie_key)
        strict_water_winner = min(scored, key=lambda item: item["water_index_fp"])["lease_id"]
        earliest_feasible = feasible[0]
        if winner != earliest_feasible:
            self.order_inversions += 1
        decision = {
            "winner": winner,
            "error_affected": boundary_affected,
            "boundary_set": sorted(boundary_set),
            "strict_water_winner": strict_water_winner,
            "tie_rule": "quota_version_desc,priority_desc,lease_id_asc",
            "error_bound": budget,
            "candidates": sorted(scored, key=lambda item: item["lease_id"]),
            "basis": (
                "projected node+global fixed-point water indices within synthetic bound; "
                "winner selected by declared tie rule"
                if boundary_affected
                else "unique minimum projected fixed-point water index"
            ),
        }
        self.decisions.append(decision)
        return winner, decision

    def choose_and_reserve(
        self, lease_id: str, now_ms: int, event_seq: int | None = None
    ) -> dict[str, Any] | None:
        winner, decision = self.choose_winner()
        if winner is None or winner != lease_id:
            return None
        reservation = self.reserve(winner, now_ms, event_seq)
        if reservation is not None:
            reservation["decision"] = decision
        return reservation

    def reserve(
        self, lease_id: str, now_ms: int, event_seq: int | None = None
    ) -> dict[str, Any] | None:
        lease = self.leases[lease_id]
        capacity = self.capacity_for(lease)
        if not capacity["fits"]:
            return None
        self.pending.remove(lease_id)
        self.global_used += lease.units
        self.node_used[lease.node_id] += lease.units
        self.node_active[lease.node_id] += 1
        lease.charged_units = lease.units
        lease.active_slot = True
        lease.wait_end_ms = now_ms
        snapshot = self.quota_snapshot(lease)
        violations = self.check_invariants()
        return {
            "lease_id": lease_id,
            "reserved_at_ms": now_ms,
            "wait_start_ms": lease.wait_start_ms,
            "wait_end_ms": lease.wait_end_ms,
            "wait_ms": lease.wait_ms,
            "quota": snapshot,
            "decision": None,
            "invariant_violations": violations,
        }

    def rollback_reservation(self, lease_id: str, now_ms: int, error_code: str) -> None:
        lease = self.leases[lease_id]
        if lease.charged_units:
            self.global_used = max(0, self.global_used - lease.charged_units)
            self.node_used[lease.node_id] = max(
                0, self.node_used[lease.node_id] - lease.charged_units
            )
            self.node_active[lease.node_id] = max(0, self.node_active[lease.node_id] - 1)
            lease.charged_units = 0
            lease.active_slot = False
        if lease.state == STATE_QUEUED:
            self.transition(
                lease,
                "failed",
                f"node_apply_failed:{error_code}",
                now_ms,
            )
        self.check_invariants()

    def confirm_grant(
        self, lease_id: str, response: dict[str, Any], now_ms: int, event_seq: int
    ) -> Transition:
        lease = self.leases[lease_id]
        before_gen = int(response.get("generation_before", 0))
        after_gen = int(response.get("generation_after", response.get("generation", 1)))
        transition = self.transition(
            lease,
            STATE_GRANTED,
            "node_apply_accepted",
            now_ms,
            event_seq=event_seq,
            generation_used=int(response.get("generation_used", 1)),
            generation_before=before_gen,
            generation_after=after_gen,
        )
        lease.generation_current = after_gen
        lease.granted_monotonic_ms = now_ms
        lease.deadline_monotonic_ms = now_ms + lease.hold_ms
        return transition

    def classify_generation(
        self, lease: LeaseRuntime, request_generation: int | None
    ) -> tuple[int, str]:
        current = lease.generation_current
        requested = current if request_generation is None else int(request_generation)
        if requested > current:
            self.stale_generation_events += 1
            raise GenerationChainError(
                "generation jumps forward across an uncommitted chain edge",
                path=f"$.events[?lease_id={lease.lease_id}].generation",
                details={
                    "lease_id": lease.lease_id,
                    "requested_generation": requested,
                    "current_generation": current,
                },
            )
        if requested < current:
            return requested, "stale_label"
        return requested, "match"

    def _target_state_for_action(self, action: str, reason: str) -> str:
        if action == "renew":
            return STATE_RENEWED
        if action == "revoke" or reason.startswith("quota_upgrade") or reason.startswith("invalidation"):
            return STATE_REVOKED
        if reason.startswith("hold_timeout") or reason.startswith("expire"):
            return "expired"
        return STATE_RELEASED

    def apply_cas_response(
        self,
        lease_id: str,
        action: str,
        reason: str,
        request_generation: int,
        response: dict[str, Any],
        now_ms: int,
        event_seq: int | None,
    ) -> tuple[Transition | None, dict[str, Any]]:
        lease = self.leases[lease_id]
        if not response.get("ok"):
            error_code = str(response.get("error", "E_NODE"))
            snapshot = self.quota_snapshot(lease)
            attempt = {
                "lease_id": lease_id,
                "action": action,
                "reason": reason,
                "state_before": lease.state,
                "state_after": lease.state,
                "generation": lease.generation_current,
                "request_generation": request_generation,
                "error_code": error_code,
                "monotonic_ms": now_ms,
                "event_seq": event_seq,
            }
            self.record_attempt(attempt)
            self.stale_generation_events += 1
            return None, snapshot

        before_gen = int(response.get("generation_before", lease.generation_current))
        after_gen = int(response.get("generation_after", response.get("generation", before_gen + 1)))
        target_state = self._target_state_for_action(action, reason)
        snapshot_before = self.quota_snapshot(lease)
        transition = self.transition(
            lease,
            target_state,
            reason,
            now_ms,
            event_seq=event_seq,
            generation_used=request_generation,
            generation_before=before_gen,
            generation_after=after_gen,
        )
        lease.generation_current = after_gen
        if target_state == STATE_RENEWED:
            lease.deadline_monotonic_ms = now_ms + lease.hold_ms
        quota_after = snapshot_before
        if target_state in {STATE_RELEASED, "expired", STATE_REVOKED}:
            self._release_capacity(lease)
            quota_after = self.quota_snapshot(lease)
        self.check_invariants()
        return transition, quota_after

    def _release_capacity(self, lease: LeaseRuntime) -> None:
        if lease.charged_units:
            self.global_used -= lease.charged_units
            self.node_used[lease.node_id] -= lease.charged_units
        if lease.active_slot:
            self.node_active[lease.node_id] -= 1
        lease.charged_units = 0
        lease.active_slot = False

    def due_leases(self, now_ms: int) -> list[LeaseRuntime]:
        due = [
            lease
            for lease in self.leases.values()
            if lease.state in ACTIVE_STATES
            and lease.deadline_monotonic_ms is not None
            and lease.deadline_monotonic_ms <= now_ms
        ]
        return sorted(due, key=lambda lease: (lease.node_id, lease.lease_id))

    def upgrade_policy(self, version: int, now_ms: int) -> dict[str, Any]:
        policy = self.version_policies.get(version)
        if policy is None:
            raise StateTransitionError(
                f"quota version {version} is not declared",
                path="$.events[?type=upgrade_quota].quota_version",
            )
        old_version = self.current_version
        old_active = sorted(
            [
                lease_id
                for lease_id, lease in self.leases.items()
                if lease.active
                and not lease.stale
                and int(lease.quota_version) < int(version)
            ],
            key=lambda lease_id: (self.leases[lease_id].node_id, lease_id),
        )
        old_pending = sorted(
            [
                lease_id
                for lease_id in self.pending
                if int(self.leases[lease_id].quota_version) < int(version)
            ],
            key=lambda lease_id: (self.leases[lease_id].node_id, lease_id),
        )
        stale_archive = {
            "quota_version": old_version,
            "marked_at_ms": now_ms,
            "active_lease_ids": old_active,
            "pending_lease_ids": old_pending,
            "water_at_replacement": self.water_report(),
            "trajectories": {
                lease_id: self.leases[lease_id].public_dict()
                for lease_id in old_active + old_pending
            },
            "status": "stale_excluded_from_current_summary",
        }
        self.stale_archives.append(stale_archive)
        replacement = {
            "quota_version": version,
            "global_budget": int(policy["global_budget"]),
            "node_quota": {node_id: 0 for node_id in self.node_ids},
            "node_max_active": {node_id: 0 for node_id in self.node_ids},
        }
        nodes_policy = policy.get("nodes", {})
        for node_id, values in nodes_policy.items():
            replacement["node_quota"][node_id] = int(values["quota"])
            replacement["node_max_active"][node_id] = int(values["max_active"])
        self.view_replacements.append(
            {
                "from_version": old_version,
                "to_version": version,
                "at_ms": now_ms,
                "revoke_set": old_active,
                "cancel_set": old_pending,
                "rule": "whole_replacement_no_merge",
            }
        )
        return {
            "old_version": old_version,
            "new_version": version,
            "policy": replacement,
            "active_to_revoke": old_active,
            "pending_to_cancel": old_pending,
            "stale_archive_index": len(self.stale_archives) - 1,
        }

    def commit_policy_replacement(self, replacement: dict[str, Any]) -> None:
        self.current_version = int(replacement["quota_version"])
        self.global_budget = int(replacement["global_budget"])
        self.node_quota = dict(replacement["node_quota"])
        self.node_max_active = dict(replacement["node_max_active"])
        for lease in self.leases.values():
            if lease.quota_version < self.current_version:
                lease.stale = True
        self.pending = [
            lease_id for lease_id in self.pending if not self.leases[lease_id].stale
        ]
        self.check_invariants()

    def water_report(self) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        for node_id in self.node_ids:
            used = self.node_used[node_id]
            quota = self.node_quota[node_id]
            level = water_level_fp(used, quota, self.scale)
            nodes[node_id] = {
                "node_id": node_id,
                "used_units": used,
                "quota_units": quota,
                "remaining_units": quota - used,
                "active_leases": self.node_active[node_id],
                "max_active": self.node_max_active[node_id],
                **level,
            }
        global_level = water_level_fp(self.global_used, self.global_budget, self.scale)
        quantization = {
            lease_id: lease_quantization_error(
                lease.units, self.node_quota[lease.node_id], self.scale
            )
            for lease_id, lease in sorted(self.leases.items())
        }
        return {
            "scale_exponent": self.scale.exponent,
            "factor": self.factor,
            "nodes": nodes,
            "global": {
                "used_units": self.global_used,
                "budget_units": self.global_budget,
                "remaining_units": self.global_budget - self.global_used,
                **global_level,
            },
            "single_lease_quantization": quantization,
            "node_comparison_error_bound_fp": node_comparison_bound_fp(),
            "global_aggregation_error_bound_fp": global_aggregation_bound_fp(len(self.node_ids)),
            "synthetic_error": self.error_budget(),
            "fixed_point_rules": saturation_rule(),
        }

    def lease_view(self, *, current_only: bool = False) -> list[dict[str, Any]]:
        values = []
        for lease_id in sorted(self.leases):
            lease = self.leases[lease_id]
            if current_only and lease.stale:
                continue
            values.append(lease.public_dict())
        return values

    def audit_generation_chains(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for lease_id, lease in sorted(self.leases.items()):
            generations = [
                int(item.generation_after)
                for item in lease.history
                if item.generation_after is not None
            ]
            if generations:
                audit_generation_chain(lease_id, generations)
                result[lease_id] = generations
        return result

    def current_view(self) -> dict[str, Any]:
        return {
            "quota_version": self.current_version,
            "leases": self.lease_view(current_only=True),
            "water": self.water_report(),
            "decisions": self.decisions,
            "generation_audit": self.audit_generation_chains(),
            "invariant_violations": self.invariant_violations,
            "view_replacements": self.view_replacements,
        }

    def stale_view(self) -> dict[str, Any]:
        return {
            "quota_version": "<" + str(self.current_version),
            "archives": self.stale_archives,
            "leases": [lease.public_dict() for lease in self.leases.values() if lease.stale],
            "note": "stale ownership, water and trajectories are excluded from current summary",
        }
