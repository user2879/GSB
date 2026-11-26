from __future__ import annotations

from typing import Any

from .errors import CacheFlowError, GenerationError
from .errors import StaleGenerationError
from .ownership import arbitrate_ownership, assign_lowest_free_slots
from .spec import Bundle
from .statemachine import (
    ACTIVE_STATES,
    GenerationChain,
    next_state,
)


class CapacityError(CacheFlowError):
    code = "E_CAPACITY"


class Engine:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle
        self.quota_version = bundle.quota_version
        self.leases: dict[str, dict[str, Any]] = {}
        self.chains: dict[str, GenerationChain] = {}
        self.transitions: list[dict[str, Any]] = []
        self.holds: dict[str, dict[str, Any]] = {}
        self.decisions: list[dict[str, Any]] = []
        self.reclaims: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._plans: dict[str, dict[str, Any]] = {}

    def remember_event(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))

    def active_holds(self) -> list[dict[str, Any]]:
        return [hold for hold in self.holds.values() if hold["status"] in {"active", "provisional"}]

    def occupied_slots(self, node_id: str, *, include_provisional: bool = True) -> set[int]:
        statuses = {"active"}
        if include_provisional:
            statuses.add("provisional")
        return {
            int(hold["slot_id"])
            for hold in self.holds.values()
            if hold["node_id"] == node_id and hold["status"] in statuses
        }

    def free_slot_count(self, node_id: str, *, include_provisional: bool = False) -> int:
        node = self.bundle.node(node_id)
        statuses = {"active"}
        if include_provisional:
            statuses.add("provisional")
        used = sum(
            1
            for hold in self.holds.values()
            if hold["node_id"] == node_id and hold["status"] in statuses
        )
        return node.capacity_slots - used

    def active_lease_count(self, node_id: str) -> int:
        return sum(
            1
            for lease in self.leases.values()
            if lease["node_id"] == node_id and lease["state"] in ACTIVE_STATES
        )

    def free_slot_map(self) -> dict[str, int]:
        return {node.node_id: self.free_slot_count(node.node_id) for node in self.bundle.nodes}

    def active_lease_count_map(self) -> dict[str, int]:
        return {node.node_id: self.active_lease_count(node.node_id) for node in self.bundle.nodes}

    def _chain(self, lease_id: str) -> GenerationChain:
        if lease_id not in self.chains:
            generation = int(self.leases.get(lease_id, {}).get("generation", 0))
            self.chains[lease_id] = GenerationChain(lease_id, generation)
        return self.chains[lease_id]

    def plan_grant(self, event: dict[str, Any], *, reserve_provisional: bool = True) -> dict[str, Any]:
        seq = int(event["seq"])
        lease_id = str(event["lease_id"])
        artifact_key = str(event["artifact_key"])
        if lease_id in self.leases:
            raise GenerationError("duplicate grant for lease", entry=lease_id, field="lease_id")
        artifact = self.bundle.artifact(artifact_key)
        pinned = event.get("node_id")
        arbitration = arbitrate_ownership(
            artifact=artifact,
            nodes=list(self.bundle.nodes),
            free_slots=self.free_slot_map(),
            active_lease_counts=self.active_lease_count_map(),
            pinned_node_id=pinned,
        )
        owner = arbitration.owner_node_id
        if owner is None:
            raise CapacityError(
                "no node has sufficient free capacity and lease quota",
                entry=lease_id,
                field="node_id",
                details={"artifact_key": artifact_key, "candidates": arbitration.to_dict()["candidates"]},
            )
        node = self.bundle.node(owner)
        occupied = self.occupied_slots(owner, include_provisional=True)
        slots = assign_lowest_free_slots(occupied, node.capacity_slots, len(artifact.shards))
        quota_version = int(event.get("quota_version", self.quota_version))
        hold_rows: list[dict[str, Any]] = []
        for shard, slot_id in zip(artifact.shards, slots):
            hold_id = f"hold:{lease_id}:{shard.index}:1"
            row = {
                "hold_id": hold_id,
                "node_id": owner,
                "slot_id": slot_id,
                "lease_id": lease_id,
                "artifact_key": artifact_key,
                "shard_index": shard.index,
                "status": "provisional" if reserve_provisional else "active",
                "opened_seq": seq,
                "closed_seq": None,
                "reclaim_reason": None,
            }
            hold_rows.append(row)
            if reserve_provisional:
                self.holds[hold_id] = dict(row)
        decision_body = {
            "arbitration": arbitration.to_dict(),
            "assigned_slots": [
                {"shard_index": shard.index, "slot_id": slot_id}
                for shard, slot_id in zip(artifact.shards, slots)
            ],
        }
        plan = {
            "request_id": f"req:{seq}:{lease_id}",
            "lease_id": lease_id,
            "artifact_key": artifact_key,
            "owner_node_id": owner,
            "quota_version": quota_version,
            "generation_after": 1,
            "state_after": "granted",
            "holds": hold_rows,
            "decision": {
                "decision_id": f"decision:{seq}:0:{artifact_key}",
                "event_seq": seq,
                "ordinal": 0,
                "artifact_key": artifact_key,
                "owner_node_id": owner,
                "basis": arbitration.rule,
                "generation": 1,
                "quota_version": quota_version,
                "decision": decision_body,
            },
            "transition": {
                "event_seq": seq,
                "ordinal": 0,
                "lease_id": lease_id,
                "node_id": owner,
                "trigger": "grant",
                "reason": str(event.get("reason", "initial_grant")),
                "state_before": "pending",
                "state_after": "granted",
                "generation_before": 0,
                "generation_after": 1,
                "carried_label": None,
                "quota_version": quota_version,
            },
            "lease": {
                "lease_id": lease_id,
                "node_id": owner,
                "artifact_key": artifact_key,
                "quota_version": quota_version,
                "state": "granted",
                "generation": 1,
                "last_event_seq": seq,
            },
        }
        self._plans[plan["request_id"]] = plan
        return plan

    def confirm_grant(self, event: dict[str, Any], plan: dict[str, Any], response: dict[str, Any] | None = None) -> dict[str, Any]:
        if response is not None:
            if response.get("lease_id") != plan["lease_id"]:
                raise GenerationError("node response lease_id mismatch", field="lease_id")
            if int(response.get("generation", -1)) != 1:
                raise GenerationError("grant response generation must be 1", field="generation")
            if response.get("state") != "granted":
                raise GenerationError("grant response state must be granted", field="state")
        for hold in plan["holds"]:
            active = dict(hold)
            active["status"] = "active"
            self.holds[active["hold_id"]] = active
        chain = self._chain(plan["lease_id"])
        transition = plan["transition"]
        chain.append(**{k: transition[k] for k in (
            "event_seq", "ordinal", "node_id", "trigger", "reason", "state_before",
            "state_after", "generation_before", "generation_after", "carried_label", "quota_version"
        )})
        self.transitions.append(transition)
        self.leases[plan["lease_id"]] = dict(plan["lease"])
        self.decisions.append(plan["decision"])
        self.remember_event(event)
        return {
            "transitions": [transition],
            "leases": [plan["lease"]],
            "holds": [dict(h, status="active") for h in plan["holds"]],
            "decisions": [plan["decision"]],
            "result": response or {"lease_id": plan["lease_id"], "generation": 1, "state": "granted", "planned": True},
        }

    def apply_grant(self, event: dict[str, Any]) -> dict[str, Any]:
        plan = self.plan_grant(event, reserve_provisional=False)
        return self.confirm_grant(event, plan, None)

    def _close_holds(
        self,
        *,
        lease_id: str,
        seq: int,
        status: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for hold in sorted(self.holds.values(), key=lambda item: (item["node_id"], item["shard_index"], item["hold_id"])):
            if hold["lease_id"] == lease_id and hold["status"] in {"active", "provisional"}:
                closed = dict(hold)
                closed["status"] = status
                closed["closed_seq"] = seq
                closed["reclaim_reason"] = reason
                self.holds[hold["hold_id"]] = closed
                rows.append(closed)
        return rows

    def apply_lifecycle(
        self,
        event: dict[str, Any],
        response: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        seq = int(event["seq"])
        lease_id = str(event["lease_id"])
        if lease_id not in self.leases:
            raise GenerationError("unknown lease", entry=lease_id, field="lease_id")
        lease = self.leases[lease_id]
        trigger = str(event["type"])
        state_before = str(lease["state"])
        generation_before = int(lease["generation"])
        state_after = next_state(state_before, trigger)
        generation_after = generation_before + 1
        if response is not None:
            generation_after = int(response.get("generation", generation_after))
            state_after = str(response.get("state", state_after))
        carried_label = event.get("generation")
        transition = {
            "event_seq": seq,
            "ordinal": 0,
            "lease_id": lease_id,
            "node_id": lease["node_id"],
            "trigger": trigger,
            "reason": reason or str(event.get("reason") or f"{trigger}_accepted"),
            "state_before": state_before,
            "state_after": state_after,
            "generation_before": generation_before,
            "generation_after": generation_after,
            "carried_label": None if carried_label is None else int(carried_label),
            "quota_version": int(lease["quota_version"]),
        }
        chain = self._chain(lease_id)
        chain.append(**{k: transition[k] for k in (
            "event_seq", "ordinal", "node_id", "trigger", "reason", "state_before",
            "state_after", "generation_before", "generation_after", "carried_label", "quota_version"
        )})
        self.transitions.append(transition)
        lease["state"] = state_after
        lease["generation"] = generation_after
        lease["last_event_seq"] = seq

        holds: list[dict[str, Any]] = []
        reclaims: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        if trigger in {"release", "expire", "invalidate"}:
            if trigger == "release":
                hold_status = "released"
                reclaim_kind = "release"
                why = "client_release_accepted"
                basis = "release_capacity_reclaimed"
            elif trigger == "expire":
                hold_status = "reclaimed_expiry"
                reclaim_kind = "expiry"
                why = reason or "ttl_due_by_real_monotonic_clock"
                basis = "ttl_expiry_reclaimed"
            else:
                hold_status = "reclaimed_quota"
                reclaim_kind = "quota_invalidation"
                why = reason or "quota_version_superseded"
                basis = "quota_invalidation_reclaimed"
            holds = self._close_holds(lease_id=lease_id, seq=seq, status=hold_status, reason=why)
            reclaim = {
                "reclaim_id": f"reclaim:{seq}:0:{lease_id}",
                "event_seq": seq,
                "ordinal": 0,
                "lease_id": lease_id,
                "node_id": lease["node_id"],
                "reason_kind": reclaim_kind,
                "generation_before": generation_before,
                "generation_after": generation_after,
                "hold_ids": [hold["hold_id"] for hold in holds],
            }
            reclaims.append(reclaim)
            self.reclaims.append(reclaim)
            decision = {
                "decision_id": f"ownership:{seq}:0:{lease_id}",
                "event_seq": seq,
                "ordinal": 0,
                "artifact_key": lease["artifact_key"],
                "owner_node_id": None,
                "basis": basis,
                "generation": generation_after,
                "quota_version": int(lease["quota_version"]),
                "decision": {
                    "owner_node_id": None,
                    "previous_owner_node_id": lease["node_id"],
                    "lease_id": lease_id,
                    "reclaim": reclaim,
                },
            }
            decisions.append(decision)
            self.decisions.append(decision)
        self.remember_event(event)
        return {
            "transitions": [transition],
            "leases": [dict(lease)],
            "holds": holds,
            "decisions": decisions,
            "reclaims": reclaims,
            "result": response
            or {"lease_id": lease_id, "generation": generation_after, "state": state_after, "reaped": trigger == "expire", "planned": True},
        }

    def apply_quota_bump(self, event: dict[str, Any]) -> dict[str, Any]:
        seq = int(event["seq"])
        new_version = int(event["quota_version"])
        if new_version <= self.quota_version:
            raise GenerationError(
                "quota_version must advance monotonically",
                field="quota_version",
                details={"current": self.quota_version, "requested": new_version},
            )
        requested_leases = set(event.get("lease_ids") or [])
        requested_nodes = set(event.get("node_ids") or [])
        candidates = [
            lease
            for lease in self.leases.values()
            if lease["state"] in ACTIVE_STATES and int(lease["quota_version"]) < new_version
            and (not requested_leases or lease["lease_id"] in requested_leases)
            and (not requested_nodes or lease["node_id"] in requested_nodes)
        ]
        candidates.sort(key=lambda lease: (lease["node_id"], lease["last_event_seq"], lease["lease_id"]))
        transitions: list[dict[str, Any]] = []
        leases: list[dict[str, Any]] = []
        all_holds: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        reclaims: list[dict[str, Any]] = []
        for ordinal, lease in enumerate(candidates):
            lease_id = lease["lease_id"]
            before = int(lease["generation"])
            after = before + 1
            transition = {
                "event_seq": seq,
                "ordinal": ordinal,
                "lease_id": lease_id,
                "node_id": lease["node_id"],
                "trigger": "invalidate",
                "reason": f"quota_version_superseded:{lease['quota_version']}->{new_version}",
                "state_before": lease["state"],
                "state_after": "expired",
                "generation_before": before,
                "generation_after": after,
                "carried_label": None,
                "quota_version": int(lease["quota_version"]),
            }
            chain = self._chain(lease_id)
            chain.append(**{k: transition[k] for k in (
                "event_seq", "ordinal", "node_id", "trigger", "reason", "state_before",
                "state_after", "generation_before", "generation_after", "carried_label", "quota_version"
            )})
            self.transitions.append(transition)
            transitions.append(transition)
            lease["state"] = "expired"
            lease["generation"] = after
            lease["last_event_seq"] = seq
            leases.append(dict(lease))
            holds = self._close_holds(
                lease_id=lease_id,
                seq=seq,
                status="reclaimed_quota",
                reason=transition["reason"],
            )
            all_holds.extend(holds)
            reclaim = {
                "reclaim_id": f"reclaim:{seq}:{ordinal}:{lease_id}",
                "event_seq": seq,
                "ordinal": ordinal,
                "lease_id": lease_id,
                "node_id": lease["node_id"],
                "reason_kind": "quota_invalidation",
                "generation_before": before,
                "generation_after": after,
                "hold_ids": [hold["hold_id"] for hold in holds],
            }
            reclaims.append(reclaim)
            self.reclaims.append(reclaim)
            decision = {
                "decision_id": f"ownership:{seq}:{ordinal}:{lease_id}",
                "event_seq": seq,
                "ordinal": ordinal,
                "artifact_key": lease["artifact_key"],
                "owner_node_id": None,
                "basis": "quota_invalidation_reclaimed",
                "generation": after,
                "quota_version": int(lease["quota_version"]),
                "decision": {
                    "owner_node_id": None,
                    "previous_owner_node_id": lease["node_id"],
                    "lease_id": lease_id,
                    "quota_version_from": int(lease["quota_version"]),
                    "quota_version_to": new_version,
                },
            }
            decisions.append(decision)
            self.decisions.append(decision)
        self.quota_version = new_version
        self.remember_event(event)
        return {
            "transitions": transitions,
            "leases": leases,
            "holds": all_holds,
            "decisions": decisions,
            "reclaims": reclaims,
            "result": {"quota_version": new_version, "invalidated_lease_ids": [item["lease_id"] for item in candidates]},
        }

    def apply_spec_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_type = event["type"]
        if event_type == "grant":
            return self.apply_grant(event)
        if event_type in {"renew", "release", "expire"}:
            lease_id = str(event["lease_id"])
            lease = self.leases[lease_id]
            carried = event.get("generation")
            if carried is not None and int(carried) != int(lease["generation"]):
                current = int(lease["generation"])
                if event.get("expect_error") == "E_STALE":
                    return {"expected_error": "E_STALE", "current_generation": current}
                raise StaleGenerationError(
                    "planned generation label does not match current generation",
                    current_generation=current,
                    entry=lease_id,
                    field="generation",
                )
            return self.apply_lifecycle(event)
        if event_type == "quota_bump":
            return self.apply_quota_bump(event)
        return {"result": {"planned": True, "op": event_type}}


def replay_bundle(bundle: Bundle, *, through_seq: int | None = None) -> Engine:
    engine = Engine(bundle)
    for event in bundle.events:
        if through_seq is not None and int(event["seq"]) > through_seq:
            break
        engine.apply_spec_event(event)
    return engine
