from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from .canonical import canonical_text
from .errors import E_VERIFY, VerifyError
from .state_machine import ACTIVE_STATES, transition_record


ReclaimKey = tuple[str, str, int]


class OwnershipAdjudicator:
    def __init__(self, spec: Optional[dict[str, Any]] = None) -> None:
        self.spec = spec or {}
        self.leases: dict[str, dict[str, Any]] = {}
        self.owners: dict[str, dict[str, Any]] = {}
        self.changes: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.stale_claims: list[dict[str, Any]] = []
        self.decisions_by_key: dict[ReclaimKey, str] = {}
        self._change_serial = 0

    def apply_all(self, events: list[dict[str, Any]]) -> "OwnershipAdjudicator":
        for event in sorted(events, key=lambda item: int(item["seq"])):
            self.apply(event)
        self.assert_invariants()
        return self

    def apply(self, event: dict[str, Any]) -> None:
        kind = event["kind"]
        if kind == "lease_granted":
            self._grant(event)
        elif kind == "lease_renewed":
            self._renew(event)
        elif kind == "lease_released":
            self._release(event)
        elif kind == "lease_expired":
            self._expire(event)
        elif kind == "reclaim_proposed":
            self._propose_reclaim(event)
        elif kind in {"quota_epoch_changed", "node_added", "node_removed"}:
            self.changes.append(
                {
                    "change_type": "topology",
                    "event_seq": int(event["seq"]),
                    "kind": kind,
                    "details": {key: value for key, value in event.items() if key not in {"seq", "kind"}},
                }
            )

    def _new_change(self, event: dict[str, Any], change_type: str, **fields: Any) -> None:
        self._change_serial += 1
        change = {
            "change_index": self._change_serial,
            "change_type": change_type,
            "event_seq": int(event["seq"]),
            **fields,
        }
        self.changes.append(change)

    def _grant(self, event: dict[str, Any]) -> None:
        lease_id = event["lease_id"]
        node_id = event["node_id"]
        slot_id = event["slot_id"]
        generation = int(event["generation"])
        if lease_id in self.leases:
            existing = self.leases[lease_id]
            record = transition_record(
                lease_id=lease_id,
                before_state=existing["state"],
                after_state="granted",
                reason="lease_granted",
                generation_before=int(existing["generation"]),
                generation_after=generation,
                event_seq=int(event["seq"]),
            )
            raise VerifyError("grant after terminal state is not permitted", details=record)

        incumbent = self.owners.get(slot_id)
        if incumbent is not None and incumbent["lease_id"] != lease_id:
            raise VerifyError(
                f"slot {slot_id} is simultaneously held by two leases",
                field=f"slots.{slot_id}",
                index=int(event["seq"]),
                details={"incumbent": incumbent, "conflicting_event": event},
            )

        record = transition_record(
            lease_id=lease_id,
            before_state=None,
            after_state="granted",
            reason="lease_granted",
            generation_before=None,
            generation_after=generation,
            event_seq=int(event["seq"]),
            request_id=event.get("request_id"),
        )
        lease = {
            "lease_id": lease_id,
            "node_id": node_id,
            "slot_id": slot_id,
            "state": "granted",
            "generation": generation,
            "quota_version": event.get("quota_version", self.spec.get("quota_version")),
            "last_event_seq": int(event["seq"]),
            "history": [record],
        }
        self.leases[lease_id] = lease
        owner = {
            "slot_id": slot_id,
            "lease_id": lease_id,
            "node_id": node_id,
            "generation": generation,
            "event_seq": int(event["seq"]),
            "basis_generation": generation,
        }
        self.owners[slot_id] = owner
        self._new_change(
            event,
            "ownership_assigned",
            slot_id=slot_id,
            to_owner=dict(owner),
            from_owner=None,
            basis_generation=generation,
        )

    def _active_lease(self, event: dict[str, Any]) -> dict[str, Any]:
        lease_id = event["lease_id"]
        lease = self.leases.get(lease_id)
        if lease is None:
            raise VerifyError(f"unknown lease {lease_id}", index=int(event["seq"]), field="lease_id")
        if lease["state"] not in ACTIVE_STATES:
            raise VerifyError(
                f"lease {lease_id} is not active: {lease['state']}",
                index=int(event["seq"]),
                field="state",
            )
        return lease

    def _renew(self, event: dict[str, Any]) -> None:
        lease = self._active_lease(event)
        generation_after = int(event["generation"])
        record = transition_record(
            lease_id=lease["lease_id"],
            before_state=lease["state"],
            after_state="renewed",
            reason="lease_renewed",
            generation_before=int(lease["generation"]),
            generation_after=generation_after,
            event_seq=int(event["seq"]),
            request_id=event.get("request_id"),
        )
        lease["history"].append(record)
        lease["state"] = "renewed"
        lease["generation"] = generation_after
        lease["last_event_seq"] = int(event["seq"])
        owner = self.owners.get(lease["slot_id"])
        if owner and owner["lease_id"] == lease["lease_id"]:
            owner["generation"] = generation_after
            owner["event_seq"] = int(event["seq"])
            owner["basis_generation"] = event.get("basis_generation", generation_after - 1)
        self._new_change(
            event,
            "generation_advanced",
            slot_id=lease["slot_id"],
            lease_id=lease["lease_id"],
            generation=generation_after,
            basis_generation=record["generation_before"],
        )

    def _release(self, event: dict[str, Any]) -> None:
        lease = self._active_lease(event)
        generation_after = int(event.get("generation", int(lease["generation"]) + 1))
        record = transition_record(
            lease_id=lease["lease_id"],
            before_state=lease["state"],
            after_state="released",
            reason="lease_released",
            generation_before=int(lease["generation"]),
            generation_after=generation_after,
            event_seq=int(event["seq"]),
            request_id=event.get("request_id"),
        )
        lease["history"].append(record)
        lease["state"] = "released"
        lease["generation"] = generation_after
        lease["last_event_seq"] = int(event["seq"])
        previous_owner = self.owners.pop(lease["slot_id"], None)
        self._new_change(
            event,
            "ownership_released",
            slot_id=lease["slot_id"],
            lease_id=lease["lease_id"],
            from_owner=previous_owner,
            to_owner=None,
            basis_generation=record["generation_before"],
        )

    def _expire(self, event: dict[str, Any]) -> None:
        lease_id = event["lease_id"]
        slot_id = event["slot_id"]
        generation = int(event.get("generation", 0))
        key = (slot_id, lease_id, generation or int(self.leases.get(lease_id, {}).get("generation", -1)))
        if key in self.decisions_by_key:
            return
        lease = self._active_lease(event)
        basis_generation = int(lease["generation"])
        reason = event.get("reason", "ttl_expiry")
        permission_reason = "ttl_expiry" if reason in {"ttl_expiry", "expired", "expiration_reclaim"} else reason
        record = transition_record(
            lease_id=lease_id,
            before_state=lease["state"],
            after_state="expired",
            reason=permission_reason,
            generation_before=basis_generation,
            generation_after=max(basis_generation, generation),
            event_seq=int(event["seq"]),
            request_id=event.get("request_id"),
        )
        lease["history"].append(record)
        lease["state"] = "expired"
        lease["generation"] = max(basis_generation, generation)
        lease["fence_reason"] = "expired"
        lease["last_event_seq"] = int(event["seq"])
        previous_owner = self.owners.pop(slot_id, None)
        decision_id = f"D{int(event['seq']):04d}"
        decision = {
            "decision_id": decision_id,
            "key": {"slot_id": slot_id, "lease_id": lease_id, "generation": basis_generation},
            "slot_id": slot_id,
            "lease_id": lease_id,
            "generation": basis_generation,
            "event_seq": int(event["seq"]),
            "winning_claim_id": event.get("claim_id", f"event-{event['seq']}"),
            "reason_category": "expired",
            "trigger_reason": "expiration_reclaim",
            "suppressed_claim_ids": [],
        }
        self.decisions.append(decision)
        self.decisions_by_key[(slot_id, lease_id, basis_generation)] = decision_id
        self._new_change(
            event,
            "ownership_reclaimed",
            decision_id=decision_id,
            slot_id=slot_id,
            lease_id=lease_id,
            from_owner=previous_owner,
            to_owner=None,
            basis_generation=basis_generation,
            reason_category="expired",
        )

    def _propose_reclaim(self, event: dict[str, Any]) -> None:
        seq = int(event["seq"])
        slot_id = event["slot_id"]
        lease_id = event["lease_id"]
        generation = int(event["generation"])
        claim_id = event["claim_id"]
        category = event["reason"]
        key: ReclaimKey = (slot_id, lease_id, generation)
        existing_decision_id = self.decisions_by_key.get(key)
        if existing_decision_id is not None:
            decision = next(item for item in self.decisions if item["decision_id"] == existing_decision_id)
            if claim_id not in decision["suppressed_claim_ids"]:
                decision["suppressed_claim_ids"].append(claim_id)
            self.stale_claims.append(
                {
                    "claim_id": claim_id,
                    "event_seq": seq,
                    "status": "suppressed_duplicate_epoch",
                    "decision_id": existing_decision_id,
                    "slot_id": slot_id,
                    "lease_id": lease_id,
                    "generation": generation,
                    "reason_category": category,
                }
            )
            return

        lease = self.leases.get(lease_id)
        mismatch: Optional[str] = None
        if lease is None:
            mismatch = "unknown_lease"
        elif lease["state"] not in ACTIVE_STATES:
            mismatch = f"lease_{lease['state']}"
        elif int(lease["generation"]) != generation:
            mismatch = "generation_mismatch"
        elif lease["slot_id"] != slot_id or lease["node_id"] != event["node_id"]:
            mismatch = "ownership_reference_mismatch"
        if mismatch is not None:
            self.stale_claims.append(
                {
                    "claim_id": claim_id,
                    "event_seq": seq,
                    "status": "rejected",
                    "reason": mismatch,
                    "slot_id": slot_id,
                    "lease_id": lease_id,
                    "generation": generation,
                    "reason_category": category,
                }
            )
            return

        assert lease is not None
        trigger_reason = "expiration_reclaim" if category == "expired" else "quota_fenced"
        record = transition_record(
            lease_id=lease_id,
            before_state=lease["state"],
            after_state="expired",
            reason=trigger_reason,
            generation_before=generation,
            generation_after=generation,
            event_seq=seq,
            request_id=event.get("request_id"),
        )
        lease["history"].append(record)
        lease["state"] = "expired"
        lease["fence_reason"] = category
        lease["last_event_seq"] = seq
        previous_owner = self.owners.pop(slot_id, None)
        decision_id = f"D{seq:04d}"
        decision = {
            "decision_id": decision_id,
            "key": {"slot_id": slot_id, "lease_id": lease_id, "generation": generation},
            "slot_id": slot_id,
            "lease_id": lease_id,
            "generation": generation,
            "event_seq": seq,
            "winning_claim_id": claim_id,
            "reason_category": category,
            "trigger_reason": trigger_reason,
            "suppressed_claim_ids": [],
        }
        self.decisions.append(decision)
        self.decisions_by_key[key] = decision_id
        self._new_change(
            event,
            "ownership_reclaimed",
            decision_id=decision_id,
            slot_id=slot_id,
            lease_id=lease_id,
            from_owner=previous_owner,
            to_owner=None,
            basis_generation=generation,
            reason_category=category,
        )

    def assert_invariants(self) -> None:
        slot_counts = Counter(owner["slot_id"] for owner in self.owners.values())
        double_held = [slot for slot, count in slot_counts.items() if count > 1]
        if double_held:
            raise VerifyError(f"double-held slots: {double_held}", details={"slots": double_held})
        seen_keys: set[tuple[str, str, int]] = set()
        for decision in self.decisions:
            key = (decision["slot_id"], decision["lease_id"], int(decision["generation"]))
            if key in seen_keys:
                raise VerifyError(f"duplicate reclamation accounting for {key}")
            seen_keys.add(key)

    def result(self) -> dict[str, Any]:
        all_slots = sorted(set(self.spec.get("slots", [])) | set(self.owners) | {lease["slot_id"] for lease in self.leases.values()})
        node_limits = {node["node_id"]: int(node.get("lease_limit", self.spec.get("per_node_lease_limit", 0))) for node in self.spec.get("nodes", [])}
        active_by_node = Counter(
            lease["node_id"] for lease in self.leases.values() if lease["state"] in ACTIVE_STATES
        )
        node_capacity = {
            node_id: {
                "active_leases": int(active_by_node.get(node_id, 0)),
                "lease_limit": int(node_limits.get(node_id, self.spec.get("per_node_lease_limit", 0))),
                "margin": int(node_limits.get(node_id, self.spec.get("per_node_lease_limit", 0)))
                - int(active_by_node.get(node_id, 0)),
            }
            for node_id in sorted(set(node_limits) | set(active_by_node))
        }
        decisions = sorted(self.decisions, key=lambda item: item["event_seq"])
        for decision in decisions:
            decision["suppressed_claim_ids"] = sorted(decision["suppressed_claim_ids"])
        expired = sum(1 for item in decisions if item["reason_category"] == "expired")
        fenced = sum(1 for item in decisions if item["reason_category"] == "quota_fenced")
        return {
            "owners": sorted(self.owners.values(), key=lambda item: item["slot_id"]),
            "leases": sorted(self.leases.values(), key=lambda item: item["lease_id"]),
            "ownership_changes": sorted(self.changes, key=lambda item: item["event_seq"]),
            "decisions": decisions,
            "stale_claims": sorted(self.stale_claims, key=lambda item: item["event_seq"]),
            "capacity": {
                "total_slots": len(all_slots),
                "owned_slots": len(self.owners),
                "free_slots": len(all_slots) - len(self.owners),
                "active_leases": sum(active_by_node.values()),
                "by_node": dict(sorted(node_capacity.items())),
            },
            "reclamation_totals": {
                "expired": expired,
                "quota_fenced": fenced,
                "unique_epochs": len(decisions),
                "overlap": [],
            },
            "invariants": {
                "no_double_hold": not any(
                    count > 1 for count in Counter(owner["slot_id"] for owner in self.owners.values()).values()
                ),
                "no_reclaim_double_count": len(self.decisions) == len(self.decisions_by_key),
            },
        }

    def canonical(self) -> bytes:
        return canonical_text(self.result()).encode("utf-8")


def adjudicate_events(events: list[dict[str, Any]], spec: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return OwnershipAdjudicator(spec).apply_all(events).result()
