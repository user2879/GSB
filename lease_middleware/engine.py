"""Deterministic lease generation, ownership and replay engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import canonical_text, content_digest
from .errors import LeaseError, SpecError
from .propagation import build_propagation_report, changed_node_ids
from .spec import SpecBundle
from .statemachine import (
    ACTIVE_STATES,
    EXPIRED,
    match_version_tag,
    require_transition,
    transition_record,
)


class ReplayEngine:
    def __init__(self, bundle: SpecBundle) -> None:
        self.bundle = bundle
        self.nodes_doc = bundle.nodes_doc
        self.requests = {item["lease_id"]: item for item in bundle.requests}
        self.node_by_id = bundle.node_by_id
        self.leases: dict[str, dict[str, Any]] = {}
        self.owners: dict[str, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.propagation_reports: list[dict[str, Any]] = []
        self.topology_version = int(self.nodes_doc.get("topology_version", 1))
        self.quota_version = int(self.nodes_doc.get("quota_version", 1))
        self.node_generations = {
            node["node_id"]: int(node["node_generation"]) for node in bundle.nodes
        }
        self._effective_reclamations: set[tuple[str, int]] = set()
        self._suppressed_alternatives: set[tuple[str, int, str]] = set()

    def request_for(self, lease_id: str) -> dict[str, Any]:
        if lease_id not in self.requests:
            raise SpecError(
                "event lease is absent from request manifest",
                field="lease_id",
                details={"lease_id": lease_id},
            )
        return self.requests[lease_id]

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        return record

    def _ownership_record(
        self,
        *,
        event_seq: int,
        lease: dict[str, Any],
        generation: int | None,
        node_generation: int,
        change: str,
        reason: str,
        at_ms: int,
        effective: bool = True,
        counted_reclamation: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner = self.owners.get(lease["equity_id"])
        record: dict[str, Any] = {
            "entry_kind": "ownership",
            "event_seq": event_seq,
            "equity_id": lease["equity_id"],
            "lease_id": lease["lease_id"],
            "node_id": lease["node_id"],
            "generation": generation,
            "node_generation": node_generation,
            "change": change,
            "reason": reason,
            "effective": effective,
            "counted_reclamation": counted_reclamation,
            "owner_after_lease": owner["lease_id"] if owner else None,
            "owner_after_node": owner["node_id"] if owner else None,
            "at_ms": at_ms,
        }
        if extra:
            record.update(extra)
        return self._append(record)

    def _assert_single_owner(self, equity_id: str) -> None:
        owner = self.owners.get(equity_id)
        if owner is None:
            return
        holders = [
            lease_id
            for lease_id, lease in self.leases.items()
            if lease["equity_id"] == equity_id
            and lease["state"] in ACTIVE_STATES
            and self.owners.get(equity_id, {}).get("lease_id") == lease_id
        ]
        if len(holders) > 1:
            raise LeaseError(
                "capacity equity is held by more than one lease",
                field="equity_id",
                details={"equity_id": equity_id, "holders": sorted(holders)},
            )

    def _grant(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        lease_id = event["lease_id"]
        request = self.request_for(lease_id)
        if lease_id in self.leases:
            raise LeaseError(
                "lease generation chain has already been granted",
                field="lease_id",
                details={"lease_id": lease_id},
            )
        node = self.node_by_id(request["node_id"])
        active_on_node = sum(
            1
            for lease in self.leases.values()
            if lease["node_id"] == node["node_id"] and lease["state"] in ACTIVE_STATES
        )
        if active_on_node >= node["lease_limit"]:
            raise LeaseError(
                "node lease_limit would be exceeded",
                field="nodes.lease_limit",
                details={"node_id": node["node_id"], "lease_limit": node["lease_limit"]},
            )

        at_ms = int(event.get("at_ms", request["at_ms"]))
        generation = 1
        expires_at_ms = at_ms + int(request["ttl_ms"])
        node_generation = self.node_generations[node["node_id"]]
        lease = {
            "lease_id": lease_id,
            "request_id": request["request_id"],
            "equity_id": request["equity_id"],
            "node_id": node["node_id"],
            "node_generation": node_generation,
            "node_generation_at_grant": node_generation,
            "state": "granted",
            "generation": generation,
            "ttl_ms": int(request["ttl_ms"]),
            "granted_at_ms": at_ms,
            "expires_at_ms": expires_at_ms,
            "history": [],
        }
        require_transition(None, "grant")
        transition = transition_record(
            event_seq=event["seq"],
            lease=lease,
            trigger="grant",
            reason="initial_grant",
            at_ms=at_ms,
            matched_tag=None,
            from_state=None,
            to_state="granted",
            generation_after=generation,
            expires_at_ms=expires_at_ms,
            generation_before=None,
        )
        lease["history"].append(transition)
        self.leases[lease_id] = lease
        self.records.append(transition)

        existing_owner = self.owners.get(request["equity_id"])
        if existing_owner is not None:
            raise LeaseError(
                "capacity equity is already held by another node",
                field="equity_id",
                details={
                    "equity_id": request["equity_id"],
                    "current_owner": existing_owner,
                    "requested_node": request["node_id"],
                },
            )
        self.owners[request["equity_id"]] = {
            "equity_id": request["equity_id"],
            "lease_id": lease_id,
            "node_id": request["node_id"],
            "generation": generation,
            "node_generation": node_generation,
            "event_seq": event["seq"],
        }
        self._ownership_record(
            event_seq=event["seq"],
            lease=lease,
            generation=generation,
            node_generation=node_generation,
            change="acquired",
            reason="initial_grant",
            at_ms=at_ms,
        )
        self._assert_single_owner(request["equity_id"])
        return [transition]

    def _tagged_transition(self, event: dict[str, Any], trigger: str) -> list[dict[str, Any]]:
        lease_id = event["lease_id"]
        lease = self.leases.get(lease_id)
        if lease is None:
            raise LeaseError(
                "unknown lease",
                field="lease_id",
                details={"lease_id": lease_id, "current_tag": 0},
            )
        before_state = lease["state"]
        current_tag = int(lease["generation"])
        requested_tag = event.get("version_tag")
        requested_tag = None if requested_tag is None else int(requested_tag)
        matched_tag = match_version_tag(
            lease_id=lease_id,
            current_tag=current_tag,
            requested_tag=requested_tag,
        )
        to_state = require_transition(before_state, trigger)
        at_ms = int(event.get("at_ms", lease["expires_at_ms"] - lease["ttl_ms"]))
        generation_after = current_tag + 1
        expires_at_ms = lease["expires_at_ms"]
        reason = "renew_accepted" if trigger == "renew" else "explicit_release"
        if trigger == "renew":
            expires_at_ms = at_ms + int(lease["ttl_ms"])

        transition = transition_record(
            event_seq=event["seq"],
            lease=lease,
            trigger=trigger,
            reason=reason,
            at_ms=at_ms,
            matched_tag=matched_tag,
            from_state=before_state,
            to_state=to_state,
            generation_after=generation_after,
            expires_at_ms=expires_at_ms,
        )
        lease["state"] = to_state
        lease["generation"] = generation_after
        lease["expires_at_ms"] = expires_at_ms
        lease["history"].append(transition)
        self.records.append(transition)

        if trigger == "release":
            self.owners.pop(lease["equity_id"], None)
            self._ownership_record(
                event_seq=event["seq"],
                lease=lease,
                generation=generation_after,
                node_generation=lease["node_generation_at_grant"],
                change="released",
                reason="explicit_release",
                at_ms=at_ms,
            )
        else:
            owner = self.owners.get(lease["equity_id"])
            if owner and owner["lease_id"] == lease_id:
                owner["generation"] = generation_after
                owner["event_seq"] = event["seq"]
            self._ownership_record(
                event_seq=event["seq"],
                lease=lease,
                generation=generation_after,
                node_generation=lease["node_generation_at_grant"],
                change="renewed",
                reason="renew_accepted",
                at_ms=at_ms,
            )
        return [transition]

    def _effective_reclaim(
        self,
        *,
        event_seq: int,
        lease: dict[str, Any],
        at_ms: int,
        reason: str,
    ) -> dict[str, Any]:
        old_generation = int(lease["generation"])
        uniqueness_key = (lease["equity_id"], old_generation)
        if uniqueness_key in self._effective_reclamations:
            raise LeaseError(
                "same equity generation was reclaimed twice",
                field="equity_id",
                details={"equity_id": lease["equity_id"], "generation": old_generation},
            )
        require_transition(lease["state"], "expire")
        new_generation = old_generation + 1
        transition = transition_record(
            event_seq=event_seq,
            lease=lease,
            trigger="expire",
            reason=reason,
            at_ms=at_ms,
            matched_tag=old_generation,
            from_state=lease["state"],
            to_state=EXPIRED,
            generation_after=new_generation,
            expires_at_ms=lease["expires_at_ms"],
        )
        lease["state"] = EXPIRED
        lease["generation"] = new_generation
        lease["history"].append(transition)
        self.records.append(transition)
        self._effective_reclamations.add(uniqueness_key)
        self.owners.pop(lease["equity_id"], None)
        return self._ownership_record(
            event_seq=event_seq,
            lease=lease,
            generation=old_generation,
            node_generation=self.node_generations[lease["node_id"]],
            change="reclaimed",
            reason=reason,
            at_ms=at_ms,
            counted_reclamation=True,
        )

    def _suppress_alternative(
        self,
        *,
        event_seq: int,
        lease: dict[str, Any],
        at_ms: int,
        candidate_reason: str,
        prior_event_seq: int,
        prior_reason: str,
    ) -> dict[str, Any] | None:
        old_generation = int(lease["history"][-1]["generation_before"])
        key = (lease["equity_id"], old_generation, candidate_reason)
        if key in self._suppressed_alternatives:
            return None
        self._suppressed_alternatives.add(key)
        return self._ownership_record(
            event_seq=event_seq,
            lease=lease,
            generation=old_generation,
            node_generation=self.node_generations[lease["node_id"]],
            change="suppressed",
            reason=candidate_reason,
            at_ms=at_ms,
            effective=False,
            counted_reclamation=False,
            extra={
                "decision": "unique_ownership_award",
                "decided_by_event_seq": prior_event_seq,
                "decided_reason": prior_reason,
            },
        )

    def _expire_scan(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        at_ms = int(event["at_ms"])
        output: list[dict[str, Any]] = []
        candidates = sorted(
            (
                lease
                for lease in self.leases.values()
                if lease["expires_at_ms"] is not None
                and int(lease["expires_at_ms"]) <= at_ms
            ),
            key=lambda item: item["lease_id"],
        )
        for lease in candidates:
            if lease["state"] in ACTIVE_STATES:
                output.append(
                    self._effective_reclaim(
                        event_seq=event["seq"],
                        lease=lease,
                        at_ms=at_ms,
                        reason="ttl_expired",
                    )
                )
            elif lease["state"] == EXPIRED and lease["history"][-1]["reason"] == (
                "generation_superseded"
            ):
                suppressed = self._suppress_alternative(
                    event_seq=event["seq"],
                    lease=lease,
                    at_ms=at_ms,
                    candidate_reason="ttl_expired",
                    prior_event_seq=int(lease["history"][-1]["event_seq"]),
                    prior_reason="generation_superseded",
                )
                if suppressed is not None:
                    output.append(suppressed)
        return output

    def _version_evolved(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        requested_topology = event.get("topology_version")
        requested_quota = event.get("quota_version")
        if requested_topology is not None:
            requested_topology = int(requested_topology)
            if requested_topology <= self.topology_version:
                raise SpecError(
                    "topology_version must increase monotonically",
                    field="topology_version",
                    details={"current": self.topology_version, "requested": requested_topology},
                )
            self.topology_version = requested_topology
        if requested_quota is not None:
            requested_quota = int(requested_quota)
            if requested_quota <= self.quota_version:
                raise SpecError(
                    "quota_version must increase monotonically",
                    field="quota_version",
                    details={"current": self.quota_version, "requested": requested_quota},
                )
            self.quota_version = requested_quota

        node_ids = [node["node_id"] for node in self.bundle.nodes]
        changed_nodes = changed_node_ids(event, node_ids)
        for node_id in changed_nodes:
            self.node_generations[node_id] += 1

        report = build_propagation_report(
            event=event,
            leases=self.leases,
            known_node_ids=node_ids,
            topology_version=self.topology_version,
            quota_version=self.quota_version,
        )
        report["changed_node_generations"] = {
            node_id: self.node_generations[node_id] for node_id in changed_nodes
        }
        self.propagation_reports.append(report)
        self.records.append(report)

        at_ms = int(event.get("at_ms", 0))
        output: list[dict[str, Any]] = [report]
        for lease_id in report["affected_leases"]:
            lease = self.leases[lease_id]
            output.append(
                self._effective_reclaim(
                    event_seq=event["seq"],
                    lease=lease,
                    at_ms=at_ms,
                    reason="generation_superseded",
                )
            )

        for lease_id in sorted(report["unchanged_leases"]):
            lease = self.leases[lease_id]
            if (
                lease["state"] == EXPIRED
                and lease["history"][-1]["reason"] == "ttl_expired"
                and lease["node_id"] in changed_nodes
                and int(lease["expires_at_ms"]) <= at_ms
            ):
                suppressed = self._suppress_alternative(
                    event_seq=event["seq"],
                    lease=lease,
                    at_ms=at_ms,
                    candidate_reason="generation_superseded",
                    prior_event_seq=int(lease["history"][-1]["event_seq"]),
                    prior_reason="ttl_expired",
                )
                if suppressed is not None:
                    output.append(suppressed)
        return output

    def _node_generation_bump(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        node_id = event["node_id"]
        adapted = dict(event)
        adapted["type"] = "version_evolved"
        adapted["changed_node_ids"] = [node_id]
        if "node_generation" in event:
            requested = int(event["node_generation"])
            current = self.node_generations[node_id]
            if requested <= current:
                raise SpecError(
                    "node_generation must increase monotonically",
                    field="node_generation",
                    details={"current": current, "requested": requested},
                )
            self.node_generations[node_id] = requested - 1
        adapted.setdefault("topology_version", self.topology_version + 1)
        return self._version_evolved(adapted)

    def apply_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = event["type"]
        if event_type == "grant":
            return self._grant(event)
        if event_type in {"renew", "release"}:
            return self._tagged_transition(event, event_type)
        if event_type == "expire_scan":
            return self._expire_scan(event)
        if event_type == "version_evolved":
            return self._version_evolved(event)
        if event_type == "node_generation_bump":
            return self._node_generation_bump(event)
        raise SpecError(f"unsupported event type {event_type!r}", field="type")

    def process_all(self) -> dict[str, Any]:
        for event in self.bundle.events:
            self.apply_event(event)
        return self.result()

    def capacity(self) -> list[dict[str, Any]]:
        report = []
        for node in self.bundle.nodes:
            active = sum(
                1
                for lease in self.leases.values()
                if lease["node_id"] == node["node_id"] and lease["state"] in ACTIVE_STATES
            )
            report.append(
                {
                    "node_id": node["node_id"],
                    "lease_limit": node["lease_limit"],
                    "active_leases": active,
                    "available_lease_slots": node["lease_limit"] - active,
                    "in_flight_limit": node["in_flight_limit"],
                }
            )
        return report

    def result(self) -> dict[str, Any]:
        ownership = [
            {
                "equity_id": equity_id,
                "lease_id": owner["lease_id"],
                "node_id": owner["node_id"],
                "generation": owner["generation"],
                "node_generation": owner["node_generation"],
                "event_seq": owner["event_seq"],
            }
            for equity_id, owner in sorted(self.owners.items())
        ]
        effective_reclamations = [
            record
            for record in self.records
            if record.get("entry_kind") == "ownership"
            and record.get("counted_reclamation") is True
        ]
        suppressed = [
            record
            for record in self.records
            if record.get("entry_kind") == "ownership"
            and record.get("effective") is False
        ]
        return {
            "format_version": 1,
            "versions": {
                "topology_version": self.topology_version,
                "quota_version": self.quota_version,
                "node_generations": dict(sorted(self.node_generations.items())),
            },
            "leases": [deepcopy(self.leases[key]) for key in sorted(self.leases)],
            "ownership": ownership,
            "capacity": self.capacity(),
            "propagation": deepcopy(self.propagation_reports),
            "records": deepcopy(self.records),
            "summary": {
                "lease_count": len(self.leases),
                "active_lease_count": sum(
                    lease["state"] in ACTIVE_STATES for lease in self.leases.values()
                ),
                "effective_reclamation_count": len(effective_reclamations),
                "suppressed_alternative_count": len(suppressed),
                "record_count": len(self.records),
            },
        }


def replay_bundle(bundle: SpecBundle) -> dict[str, Any]:
    engine = ReplayEngine(bundle)
    return engine.process_all()


def result_digest(result: dict[str, Any]) -> str:
    conclusion = {
        "versions": result["versions"],
        "leases": result["leases"],
        "ownership": result["ownership"],
        "capacity": result["capacity"],
    }
    return content_digest(conclusion)


def assert_byte_identical(left: dict[str, Any], right: dict[str, Any]) -> None:
    if canonical_text(left) != canonical_text(right):
        raise AssertionError("replay conclusions differ")
