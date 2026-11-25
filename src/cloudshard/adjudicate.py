from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes
from .errors import LeaseError
from .spec import Spec


@dataclass
class CapacityLimits:
    global_budget_units: int
    node_lease_limits: dict[str, int]
    node_concurrency_quotas: dict[str, int]
    node_renewal_limits: dict[str, int]


def limits_from_spec(spec: Spec) -> CapacityLimits:
    return CapacityLimits(
        global_budget_units=spec.global_budget_units,
        node_lease_limits={node.node_id: node.lease_limit for node in spec.nodes},
        node_concurrency_quotas={node.node_id: node.concurrency_quota for node in spec.nodes},
        node_renewal_limits={node.node_id: node.renewal_limit for node in spec.nodes},
    )


class Adjudicator:
    """Deterministic lease/capacity ownership state machine.

    The first valid terminal reclamation wins. Competing expiry and quota events
    are retained as decisions but only one record enters the ownership sequence.
    """

    def __init__(self, limits: CapacityLimits, *, default_ttl_ms: int = 1_000) -> None:
        self.limits = limits
        self.default_ttl_ms = default_ttl_ms
        self.quota_versions: dict[str, int] = {"global": 0}
        for node_id in limits.node_lease_limits:
            self.quota_versions[node_id] = 0
        self.claims: dict[str, dict[str, Any]] = {}
        self.entitlement_holders: dict[str, str] = {}
        self.ownership_sequence: list[dict[str, Any]] = []
        self.timeline: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []

    @classmethod
    def from_spec(cls, spec: Spec) -> "Adjudicator":
        adjudic = cls(limits_from_spec(spec), default_ttl_ms=spec.lease_ttl_ms)
        for node in spec.nodes:
            adjudic.quota_versions[node.node_id] = node.capacity_version
            adjudic.quota_versions["global"] = max(adjudic.quota_versions["global"], 1)
        return adjudic

    def replay(self, events: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> "Adjudicator":
        for event in events:
            self.handle_event(event)
        return self

    def ownership_bytes(self) -> bytes:
        return canonical_bytes({"format_version": 1, "ownership_sequence": self.ownership_sequence})

    def result_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "ownership_sequence": self.ownership_sequence,
            "timeline": self.timeline,
            "decisions": self.decisions,
            "claims": sorted(self.claims.values(), key=lambda c: c["lease_id"]),
            "entitlement_holders": dict(sorted(self.entitlement_holders.items())),
            "quota_versions": dict(sorted(self.quota_versions.items())),
        }

    def handle_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event = dict(event)
        seq = int(event["seq"])
        event_type = str(event["type"])
        at_ms = int(event.get("at_ms", seq))
        handler = {
            "lease_apply": self._apply,
            "lease_renew": self._renew,
            "lease_release": self._release,
            "lease_expire": self._expire,
            "expire_due": self._expire_due,
            "quota_change": self._quota_change,
            "cancel_shard": self._cancel_shard,
        }[event_type]
        return handler(event, seq, at_ms)

    def _qv(self, event: dict[str, Any]) -> int:
        return int(event.get("quota_version", self.quota_versions.get("global", 1)))

    def _decision(self, seq: int, outcome: str, basis: dict[str, Any]) -> dict[str, Any]:
        record = {"event_seq": seq, "outcome": outcome, "basis": basis}
        self.decisions.append(record)
        return record

    def _allocate_entitlements(
        self, lease_id: str, node_id: str, units: int
    ) -> dict[str, list[str]]:
        allocated = {"global_units": [], "lease_slots": [], "concurrency_slots": []}
        budget = self.limits.global_budget_units
        for unit_index in range(budget):
            ent_id = f"global:unit:{unit_index:04d}"
            if ent_id not in self.entitlement_holders:
                self.entitlement_holders[ent_id] = lease_id
                allocated["global_units"].append(ent_id)
                if len(allocated["global_units"]) == units:
                    break
        if len(allocated["global_units"]) != units:
            self._rollback_allocation(allocated)
            raise LeaseError("global resource budget has insufficient free units", field="resource_units")

        lease_limit = self.limits.node_lease_limits[node_id]
        for slot in range(lease_limit):
            ent_id = f"node:{node_id}:lease-slot:{slot:03d}"
            if ent_id not in self.entitlement_holders:
                self.entitlement_holders[ent_id] = lease_id
                allocated["lease_slots"].append(ent_id)
                break
        if not allocated["lease_slots"]:
            self._rollback_allocation(allocated)
            raise LeaseError("node lease_limit exhausted", field="lease_limit")

        concurrency = self.limits.node_concurrency_quotas[node_id]
        for slot in range(concurrency):
            ent_id = f"node:{node_id}:concurrency-slot:{slot:03d}"
            if ent_id not in self.entitlement_holders:
                self.entitlement_holders[ent_id] = lease_id
                allocated["concurrency_slots"].append(ent_id)
                break
        if not allocated["concurrency_slots"]:
            self._rollback_allocation(allocated)
            raise LeaseError("node concurrency_quota exhausted", field="concurrency_quota")
        return allocated

    def _rollback_allocation(self, allocated: dict[str, list[str]]) -> None:
        for ent_id in [*allocated.get("global_units", []), *allocated.get("lease_slots", []), *allocated.get("concurrency_slots", [])]:
            self.entitlement_holders.pop(ent_id, None)

    def _active_claim_for_shard(self, shard_id: str) -> dict[str, Any] | None:
        for claim in sorted(self.claims.values(), key=lambda c: c["lease_id"]):
            if claim["status"] == "active" and claim["shard_id"] == shard_id:
                return claim
        return None

    def _apply(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        lease_id = event["lease_id"]
        shard_id = event["shard_id"]
        node_id = event["node_id"]
        if lease_id in self.claims:
            raise LeaseError(f"lease_id {lease_id!r} already exists", shard_id=shard_id, field="lease_id")
        existing = self._active_claim_for_shard(shard_id)
        if existing is not None:
            raise LeaseError(
                f"shard {shard_id!r} already has active lease {existing['lease_id']}",
                shard_id=shard_id,
                field="shard_id",
                basis={"existing_lease_id": existing["lease_id"]},
            )
        active_on_node = [c for c in self.claims.values() if c["status"] == "active" and c["node_id"] == node_id]
        if len(active_on_node) >= self.limits.node_lease_limits[node_id]:
            raise LeaseError("node lease_limit exhausted", shard_id=shard_id, field="lease_limit")
        if len(active_on_node) >= self.limits.node_concurrency_quotas[node_id]:
            raise LeaseError("node concurrency_quota exhausted", shard_id=shard_id, field="concurrency_quota")

        units = int(event.get("resource_units", 1))
        used_units = sum(int(c["resource_units"]) for c in self.claims.values() if c["status"] == "active")
        if used_units + units > self.limits.global_budget_units:
            raise LeaseError("global resource budget exhausted", shard_id=shard_id, field="global_budget_units")

        allocated = self._allocate_entitlements(lease_id, node_id, units)
        ttl_ms = int(event.get("ttl_ms", self.default_ttl_ms))
        qv = self._qv(event)
        claim = {
            "lease_id": lease_id,
            "shard_id": shard_id,
            "node_id": node_id,
            "resource_units": units,
            "priority": int(event.get("priority", 0)),
            "granted_at_ms": at_ms,
            "expires_at_ms": at_ms + ttl_ms,
            "ttl_ms": ttl_ms,
            "renewal_count": 0,
            "lease_version": 1,
            "quota_version": qv,
            "status": "active",
            "allocated_entitlements": allocated,
            "last_event_seq": seq,
        }
        self.claims[lease_id] = claim
        record = {
            "sequence": len(self.ownership_sequence) + 1,
            "event_seq": seq,
            "quota_version": qv,
            "at_ms": at_ms,
            "entitlement_id": f"lease:{lease_id}",
            "lease_id": lease_id,
            "shard_id": shard_id,
            "node_id": node_id,
            "previous_holder_key": None,
            "holder_key": f"{node_id}::{shard_id}",
            "change": "granted",
            "reason": "lease_apply_accepted",
            "basis": {
                "ttl_ms": ttl_ms,
                "expires_at_ms": claim["expires_at_ms"],
                "allocated_entitlements": allocated,
            },
        }
        self.ownership_sequence.append(record)
        self.timeline.append({**record, "timeline": "lease"})
        return [record]

    def _renew(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        lease_id = event["lease_id"]
        claim = self.claims.get(lease_id)
        if claim is None or claim["status"] != "active":
            raise LeaseError(f"lease {lease_id!r} is not active", field="lease_id")
        expected_version = int(event["expected_version"])
        if expected_version != claim["lease_version"]:
            raise LeaseError(
                "optimistic lease version mismatch; expiry was not extended",
                field="lease_version",
                shard_id=claim["shard_id"],
                basis={
                    "lease_id": lease_id,
                    "expected_version": expected_version,
                    "actual_version": claim["lease_version"],
                    "expires_at_ms": claim["expires_at_ms"],
                },
            )
        if at_ms >= claim["expires_at_ms"]:
            raise LeaseError("cannot renew an expired lease", field="expires_at_ms", shard_id=claim["shard_id"])
        limit = self.limits.node_renewal_limits[claim["node_id"]]
        if claim["renewal_count"] >= limit:
            raise LeaseError(
                f"renewal count {claim['renewal_count']} exceeds declared limit {limit}",
                field="renewal_count",
                shard_id=claim["shard_id"],
            )
        old_expires = claim["expires_at_ms"]
        claim["lease_version"] += 1
        claim["renewal_count"] += 1
        claim["expires_at_ms"] = at_ms + int(claim["ttl_ms"])
        claim["last_event_seq"] = seq
        record = {
            "event_seq": seq,
            "quota_version": self._qv(event),
            "at_ms": at_ms,
            "lease_id": lease_id,
            "shard_id": claim["shard_id"],
            "node_id": claim["node_id"],
            "change": "renewed",
            "old_version": expected_version,
            "new_version": claim["lease_version"],
            "old_expires_at_ms": old_expires,
            "new_expires_at_ms": claim["expires_at_ms"],
            "basis": {"optimistic_check": "lease_version matched node and ledger"},
        }
        self.timeline.append(record)
        return [record]

    def _free_claim(self, claim: dict[str, Any]) -> None:
        allocated = claim.get("allocated_entitlements", {})
        for ent_id in [
            *allocated.get("global_units", []),
            *allocated.get("lease_slots", []),
            *allocated.get("concurrency_slots", []),
        ]:
            if self.entitlement_holders.get(ent_id) == claim["lease_id"]:
                del self.entitlement_holders[ent_id]

    def _reclaim(
        self,
        lease_id: str,
        *,
        seq: int,
        at_ms: int,
        quota_version: int,
        reason: str,
        basis: dict[str, Any],
    ) -> dict[str, Any] | None:
        claim = self.claims.get(lease_id)
        if claim is None:
            self._decision(seq, "reclaim_unknown_lease", {"lease_id": lease_id, "reason": reason})
            return None
        if claim["status"] != "active":
            self._decision(
                seq,
                "duplicate_reclamation_skipped",
                {
                    "lease_id": lease_id,
                    "requested_reason": reason,
                    "terminal_status": claim["status"],
                    "terminal_event_seq": claim["last_event_seq"],
                },
            )
            return None

        previous_holder = f"{claim['node_id']}::{claim['shard_id']}"
        self._free_claim(claim)
        claim["status"] = "reclaimed"
        claim["reclaim_reason"] = reason
        claim["last_event_seq"] = seq
        claim["quota_version_at_reclamation"] = quota_version
        record = {
            "sequence": len(self.ownership_sequence) + 1,
            "event_seq": seq,
            "quota_version": quota_version,
            "at_ms": at_ms,
            "entitlement_id": f"lease:{lease_id}",
            "lease_id": lease_id,
            "shard_id": claim["shard_id"],
            "node_id": claim["node_id"],
            "previous_holder_key": previous_holder,
            "holder_key": None,
            "change": "reclaimed",
            "reason": reason,
            "basis": basis,
        }
        self.ownership_sequence.append(record)
        self.timeline.append({**record, "timeline": "ownership"})
        return record

    def _release(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        qv = self._qv(event)
        reason = str(event.get("reason", "released"))
        basis: dict[str, Any] = {"event_type": "lease_release"}
        if event.get("recovery"):
            basis["recovery"] = True
            basis["rule"] = str(event.get("rule", "committed dispatch without outcome is unknown and reclaimed before rerun"))
        record = self._reclaim(
            event["lease_id"],
            seq=seq,
            at_ms=at_ms,
            quota_version=qv,
            reason=reason,
            basis=basis,
        )
        return [record] if record else []

    def _expire(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        lease_id = event["lease_id"]
        claim = self.claims.get(lease_id)
        if claim is None:
            raise LeaseError(f"unknown lease {lease_id!r}", field="lease_id")
        if claim["status"] != "active":
            self._decision(seq, "expire_after_terminal_skipped", {"lease_id": lease_id, "status": claim["status"]})
            return []
        if at_ms < int(claim["expires_at_ms"]):
            raise LeaseError(
                "lease is not due according to monotonic milliseconds",
                field="at_ms",
                shard_id=claim["shard_id"],
                basis={"at_ms": at_ms, "expires_at_ms": claim["expires_at_ms"]},
            )
        record = self._reclaim(
            lease_id,
            seq=seq,
            at_ms=at_ms,
            quota_version=self._qv(event),
            reason="lease_expired",
            basis={
                "observed_monotonic_ms": at_ms,
                "expires_at_ms": claim["expires_at_ms"],
                "rule": "observed >= expires",
            },
        )
        return [record] if record else []

    def _expire_due(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        records = []
        due = [
            claim
            for claim in self.claims.values()
            if claim["status"] == "active" and at_ms >= int(claim["expires_at_ms"])
        ]
        for claim in sorted(due, key=lambda c: (c["expires_at_ms"], c["lease_id"])):
            record = self._reclaim(
                claim["lease_id"],
                seq=seq,
                at_ms=at_ms,
                quota_version=self._qv(event),
                reason="lease_expired",
                basis={"observed_monotonic_ms": at_ms, "expires_at_ms": claim["expires_at_ms"]},
            )
            if record:
                records.append(record)
        return records

    def _cancel_shard(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        claim = self._active_claim_for_shard(event["shard_id"])
        if claim is None:
            self._decision(seq, "cancel_queued_without_active_lease", {"shard_id": event["shard_id"]})
            return []
        record = self._reclaim(
            claim["lease_id"],
            seq=seq,
            at_ms=at_ms,
            quota_version=self._qv(event),
            reason="canceled",
            basis={"shard_id": event["shard_id"], "rule": "cancel_releases_capacity_and_forbids_retry"},
        )
        return [record] if record else []

    def _used_units(self) -> int:
        return sum(int(c["resource_units"]) for c in self.claims.values() if c["status"] == "active")

    def _active_node_claims(self, node_id: str) -> list[dict[str, Any]]:
        return [c for c in self.claims.values() if c["status"] == "active" and c["node_id"] == node_id]

    def _choose_revocation(self, node_id: str | None) -> dict[str, Any] | None:
        candidates = [
            c
            for c in self.claims.values()
            if c["status"] == "active" and (node_id is None or c["node_id"] == node_id)
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda c: (int(c.get("priority", 0)), -int(c["resource_units"]), int(c["expires_at_ms"]), c["lease_id"]),
        )[0]

    def _quota_change(self, event: dict[str, Any], seq: int, at_ms: int) -> list[dict[str, Any]]:
        qv = int(event["quota_version"])
        node_id = event.get("node_id")
        scope = node_id or "global"
        previous_qv = self.quota_versions.get(scope, 0)
        if qv < previous_qv:
            raise LeaseError(
                f"quota_version for {scope} cannot decrease: {qv} < {previous_qv}",
                field="quota_version",
                basis={"scope": scope, "previous": previous_qv},
            )
        self.quota_versions[scope] = qv
        self.quota_versions["global"] = max(self.quota_versions["global"], qv)

        old_limits = {
            "global_budget_units": self.limits.global_budget_units,
            "node": {
                nid: {
                    "lease_limit": self.limits.node_lease_limits[nid],
                    "concurrency_quota": self.limits.node_concurrency_quotas[nid],
                }
                for nid in sorted(self.limits.node_lease_limits)
            },
        }
        if "global_budget_units" in event and event["global_budget_units"] is not None:
            self.limits.global_budget_units = int(event["global_budget_units"])
        if node_id is not None:
            if "lease_limit" in event and event["lease_limit"] is not None:
                self.limits.node_lease_limits[node_id] = int(event["lease_limit"])
            if "concurrency_quota" in event and event["concurrency_quota"] is not None:
                self.limits.node_concurrency_quotas[node_id] = int(event["concurrency_quota"])

        records: list[dict[str, Any]] = []
        explicit = event.get("revoke_lease_ids", []) or []
        for lease_id in explicit:
            record = self._reclaim(
                lease_id,
                seq=seq,
                at_ms=at_ms,
                quota_version=qv,
                reason="quota_revocation",
                basis={
                    "quota_version": qv,
                    "scope": scope,
                    "explicit_revocation": True,
                    "new_limits": {
                        "global_budget_units": self.limits.global_budget_units,
                        "lease_limit": self.limits.node_lease_limits.get(node_id) if node_id else None,
                        "concurrency_quota": self.limits.node_concurrency_quotas.get(node_id) if node_id else None,
                    },
                },
            )
            if record:
                records.append(record)

        while True:
            violation: dict[str, Any] | None = None
            target_node: str | None = None
            if self._used_units() > self.limits.global_budget_units:
                violation = {
                    "kind": "global_budget",
                    "used": self._used_units(),
                    "limit": self.limits.global_budget_units,
                }
            else:
                for nid in sorted(self.limits.node_lease_limits):
                    active = self._active_node_claims(nid)
                    if len(active) > self.limits.node_lease_limits[nid]:
                        target_node = nid
                        violation = {
                            "kind": "node_lease_limit",
                            "node_id": nid,
                            "used": len(active),
                            "limit": self.limits.node_lease_limits[nid],
                        }
                        break
                    if len(active) > self.limits.node_concurrency_quotas[nid]:
                        target_node = nid
                        violation = {
                            "kind": "node_concurrency_quota",
                            "node_id": nid,
                            "used": len(active),
                            "limit": self.limits.node_concurrency_quotas[nid],
                        }
                        break
            if violation is None:
                break
            claim = self._choose_revocation(target_node)
            if claim is None:
                break
            record = self._reclaim(
                claim["lease_id"],
                seq=seq,
                at_ms=at_ms,
                quota_version=qv,
                reason="quota_revocation",
                basis={
                    "quota_version": qv,
                    "scope": scope,
                    "violation": violation,
                    "selection_policy": "priority_asc,then_resource_units_desc,then_expiry_asc,then_lease_id",
                    "old_limits": old_limits,
                },
            )
            if record:
                records.append(record)
        self._decision(
            seq,
            "quota_version_committed",
            {
                "quota_version": qv,
                "scope": scope,
                "new_global_budget_units": self.limits.global_budget_units,
                "new_node_limits": {
                    nid: {
                        "lease_limit": self.limits.node_lease_limits[nid],
                        "concurrency_quota": self.limits.node_concurrency_quotas[nid],
                    }
                    for nid in sorted(self.limits.node_lease_limits)
                },
                "ownership_records_at_event": len(records),
            },
        )
        return records


def assert_unique_ownership(adj: Adjudicator) -> None:
    active_pairs: set[tuple[str, str]] = set()
    for claim in adj.claims.values():
        if claim["status"] != "active":
            continue
        pair = (claim["node_id"], claim["shard_id"])
        if pair in active_pairs:
            raise AssertionError(f"duplicate active node/shard ownership: {pair}")
        active_pairs.add(pair)
    reverse: dict[str, str] = {}
    for ent_id, lease_id in adj.entitlement_holders.items():
        if ent_id in reverse and reverse[ent_id] != lease_id:
            raise AssertionError(f"entitlement {ent_id} held twice")
        reverse[ent_id] = lease_id
