from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import DomainState
from .errors import GenerationChainError, VerificationError
from .hashing import sha256_digest
from .ledger import Ledger
from .node_server import NodeCore


LIFECYCLE_KINDS = {
    "grant",
    "renew",
    "release",
    "expire",
    "revoke",
    "grant_failed",
    "renew_rejected",
    "release_rejected",
}

COMPARE_FIELDS = (
    "kind",
    "ok",
    "error_code",
    "state_before",
    "state_after",
    "generation_used",
    "generation_before",
    "generation_after",
    "global_used",
    "global_remaining",
    "node_used",
    "node_remaining",
)


class DeterministicReplay:
    def __init__(self, spec: dict[str, Any], events: list[dict[str, Any]]):
        self.spec = spec
        self.events = events
        self.domain = DomainState(spec)
        self.node_cores = {
            node["node_id"]: NodeCore(node["node_id"], int(node["seed"]))
            for node in spec["nodes"]
        }
        self.apply_seq = {
            event["lease_id"]: event["seq"] for event in events if event["type"] == "apply"
        }
        self.facts_by_lease: dict[str, list[dict[str, Any]]] = {}
        self.upgrade_facts: list[dict[str, Any]] = []

    def _fact(self, lease_id: str, fact: dict[str, Any]) -> None:
        self.facts_by_lease.setdefault(lease_id, []).append(fact)

    def _transition_fact(
        self,
        *,
        lease_id: str,
        kind: str,
        transition: Any,
        quota: dict[str, Any],
        event_seq: int | None,
    ) -> None:
        self._fact(
            lease_id,
            {
                "kind": kind,
                "ok": True,
                "error_code": None,
                "event_seq": event_seq,
                "state_before": transition.from_state,
                "state_after": transition.to_state,
                "generation_used": transition.generation_used,
                "generation_before": transition.generation_before,
                "generation_after": transition.generation_after,
                "global_used": quota.get("global_used"),
                "global_remaining": quota.get("global_remaining"),
                "node_used": quota.get("node_used"),
                "node_remaining": quota.get("node_remaining"),
            },
        )

    def _rejected_fact(
        self,
        *,
        lease_id: str,
        kind: str,
        event_seq: int | None,
        error_code: str,
        request_generation: int | None,
        quota: dict[str, Any] | None = None,
    ) -> None:
        lease = self.domain.leases[lease_id]
        quota = quota or self.domain.quota_snapshot(lease)
        self._fact(
            lease_id,
            {
                "kind": kind,
                "ok": False,
                "error_code": error_code,
                "event_seq": event_seq,
                "state_before": lease.state,
                "state_after": lease.state,
                "generation_used": request_generation,
                "generation_before": lease.generation_current,
                "generation_after": lease.generation_current,
                "global_used": quota.get("global_used"),
                "global_remaining": quota.get("global_remaining"),
                "node_used": quota.get("node_used"),
                "node_remaining": quota.get("node_remaining"),
            },
        )

    def admit_pending(self, now_ms: int) -> None:
        while True:
            winner, decision = self.domain.choose_winner()
            if winner is None:
                return
            event_seq = self.apply_seq.get(winner)
            reservation = self.domain.reserve(winner, now_ms, event_seq)
            if reservation is None:
                return
            lease = self.domain.leases[winner]
            response = self.node_cores[lease.node_id].handle(
                {
                    "type": "apply",
                    "lease_id": winner,
                    "units": lease.units,
                    "generation": 1,
                }
            )
            if not response.get("ok"):
                self.domain.rollback_reservation(winner, now_ms, str(response.get("error")))
                self._rejected_fact(
                    lease_id=winner,
                    kind="grant_failed",
                    event_seq=event_seq,
                    error_code=str(response.get("error")),
                    request_generation=1,
                    quota=reservation["quota"],
                )
                continue
            transition = self.domain.confirm_grant(winner, response, now_ms, event_seq or 0)
            self._transition_fact(
                lease_id=winner,
                kind="grant",
                transition=transition,
                quota=reservation["quota"],
                event_seq=event_seq,
            )

    def expire_due(self, now_ms: int) -> None:
        for lease in self.domain.due_leases(now_ms):
            current_generation = lease.generation_current
            response = self.node_cores[lease.node_id].handle(
                {
                    "type": "release",
                    "lease_id": lease.lease_id,
                    "generation": current_generation,
                }
            )
            transition, quota = self.domain.apply_cas_response(
                lease.lease_id,
                "release",
                "hold_timeout_monotonic_deadline",
                current_generation,
                response,
                now_ms,
                None,
            )
            if transition is not None:
                self._transition_fact(
                    lease_id=lease.lease_id,
                    kind="expire",
                    transition=transition,
                    quota=quota,
                    event_seq=None,
                )

    def process_cas_event(self, event: dict[str, Any]) -> None:
        lease_id = event["lease_id"]
        action = event["type"]
        kind = action
        lease = self.domain.leases[lease_id]
        requested = event.get("generation")
        try:
            request_generation, _classification = self.domain.classify_generation(lease, requested)
        except GenerationChainError as exc:
            self._rejected_fact(
                lease_id=lease_id,
                kind=f"{kind}_rejected",
                event_seq=event["seq"],
                error_code="E_GEN",
                request_generation=requested,
            )
            return

        response = self.node_cores[lease.node_id].handle(
            {
                "type": action,
                "lease_id": lease_id,
                "generation": request_generation,
            }
        )
        transition, quota = self.domain.apply_cas_response(
            lease_id,
            action,
            f"user_{action}",
            request_generation,
            response,
            event["arrival_ms"],
            event["seq"],
        )
        if transition is None:
            self._rejected_fact(
                lease_id=lease_id,
                kind=f"{kind}_rejected",
                event_seq=event["seq"],
                error_code=str(response.get("error", "E_NODE")),
                request_generation=request_generation,
                quota=quota,
            )
        else:
            self._transition_fact(
                lease_id=lease_id,
                kind=kind,
                transition=transition,
                quota=quota,
                event_seq=event["seq"],
            )

    def process_upgrade(self, event: dict[str, Any]) -> None:
        now = event["arrival_ms"]
        version = int(event["quota_version"])
        plan = self.domain.upgrade_policy(version, now)
        for lease_id in plan["active_to_revoke"]:
            lease = self.domain.leases[lease_id]
            generation = lease.generation_current
            response = self.node_cores[lease.node_id].handle(
                {
                    "type": "revoke",
                    "lease_id": lease_id,
                    "generation": generation,
                }
            )
            transition, quota = self.domain.apply_cas_response(
                lease_id,
                "revoke",
                f"quota_upgrade_whole_replacement:{plan['old_version']}->{version}",
                generation,
                response,
                now,
                event["seq"],
            )
            if transition is not None:
                self._transition_fact(
                    lease_id=lease_id,
                    kind="revoke",
                    transition=transition,
                    quota=quota,
                    event_seq=event["seq"],
                )
        for lease_id in plan["pending_to_cancel"]:
            lease = self.domain.leases[lease_id]
            if lease_id in self.domain.pending:
                self.domain.pending.remove(lease_id)
            if lease.state == "queued":
                transition = self.domain.transition(
                    lease,
                    "revoked",
                    f"quota_upgrade_cancels_pending:{plan['old_version']}->{version}",
                    now,
                    event_seq=event["seq"],
                    generation_before=0,
                    generation_after=0,
                )
                self._transition_fact(
                    lease_id=lease_id,
                    kind="revoke",
                    transition=transition,
                    quota=self.domain.quota_snapshot(lease),
                    event_seq=event["seq"],
                )
        self.domain.commit_policy_replacement(plan["policy"])
        self.upgrade_facts.append(
            {
                "event_seq": event["seq"],
                "state_before": f"version:{plan['old_version']}",
                "state_after": f"version:{version}",
            }
        )

    def run(self) -> dict[str, list[dict[str, Any]]]:
        index = 0
        while index < len(self.events):
            arrival = self.events[index]["arrival_ms"]
            group = []
            while index < len(self.events) and self.events[index]["arrival_ms"] == arrival:
                group.append(self.events[index])
                index += 1
            self.expire_due(arrival)
            self.admit_pending(arrival)
            applies = [event for event in group if event["type"] == "apply"]
            for event in applies:
                self.domain.register_apply(event["lease_id"], arrival, event["seq"])
            self.admit_pending(arrival)
            for event in group:
                if event["type"] in {"renew", "release"}:
                    self.process_cas_event(event)
                    if event["type"] == "release":
                        self.admit_pending(arrival)
                elif event["type"] == "upgrade_quota":
                    self.process_upgrade(event)
                    self.admit_pending(arrival)
        horizon = int(self.spec.get("replay_horizon_ms", 0))
        self.expire_due(horizon)
        self.admit_pending(horizon)
        self.expire_due(horizon)
        return self.facts_by_lease


def _actual_fact(row: Any) -> dict[str, Any]:
    return {field: row[field] for field in COMPARE_FIELDS}


def verify_ledger(
    spec: dict[str, Any],
    events: list[dict[str, Any]],
    ledger_path: str | Path,
    *,
    check_digests: bool = True,
) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise VerificationError(f"ledger not found: {ledger_path}", path=str(ledger_path))
    ledger = Ledger(ledger_path)
    try:
        try:
            ledger.verify_hash_chain()
        except ValueError as exc:
            raise VerificationError(
                str(exc),
                path="$.ledger.hash_chain",
                details={"field": "entry_digest", "lease_id": None, "generation": None},
            ) from exc
        if check_digests:
            meta = {
                row["key"]: row["value"]
                for row in ledger.conn.execute("SELECT key,value FROM meta").fetchall()
            }
            if meta.get("spec_digest") != sha256_digest(spec):
                raise VerificationError(
                    "original spec digest does not match ledger metadata",
                    path="$.spec",
                    details={"field": "spec_digest"},
                )
            if meta.get("events_digest") != sha256_digest(events):
                raise VerificationError(
                    "original event stream digest does not match ledger metadata",
                    path="$.events",
                    details={"field": "events_digest"},
                )

        replay = DeterministicReplay(spec, events)
        expected = replay.run()
        actual: dict[str, list[Any]] = {}
        upgrade_rows = []
        for row in ledger.entries():
            if row["kind"] in LIFECYCLE_KINDS and row["lease_id"]:
                actual.setdefault(row["lease_id"], []).append(row)
            if row["kind"] == "quota_upgrade":
                upgrade_rows.append(row)

        lease_ids = sorted(set(expected) | set(actual))
        for lease_id in lease_ids:
            expected_rows = expected.get(lease_id, [])
            actual_rows = actual.get(lease_id, [])
            if len(expected_rows) != len(actual_rows):
                generation = (
                    actual_rows[-1]["generation_after"]
                    if actual_rows
                    else expected_rows[-1].get("generation_after")
                    if expected_rows
                    else None
                )
                raise VerificationError(
                    f"entry count mismatch for lease {lease_id}: "
                    f"expected {len(expected_rows)}, ledger has {len(actual_rows)}",
                    path=f"$.leases[?lease_id={lease_id}].entries",
                    details={
                        "lease_id": lease_id,
                        "field": "entry_count",
                        "generation": generation,
                        "expected": len(expected_rows),
                        "actual": len(actual_rows),
                    },
                )
            for index, (expected_fact, row) in enumerate(zip(expected_rows, actual_rows, strict=True)):
                actual_fact = _actual_fact(row)
                for field in COMPARE_FIELDS:
                    expected_value = expected_fact.get(field)
                    actual_value = actual_fact.get(field)
                    if field == "ok":
                        actual_value = bool(actual_value)
                    if expected_value != actual_value:
                        raise VerificationError(
                            f"ledger mismatch at lease {lease_id}, entry {index}, field {field}",
                            path=f"$.leases[?lease_id={lease_id}].entry[{index}].{field}",
                            details={
                                "lease_id": lease_id,
                                "field": field,
                                "generation": row["generation_after"],
                                "entry_index": index,
                                "expected": expected_value,
                                "actual": actual_value,
                            },
                        )

        expected_upgrades = replay.upgrade_facts
        if len(expected_upgrades) != len(upgrade_rows):
            raise VerificationError(
                "quota upgrade entry count mismatch",
                path="$.quota_versions",
                details={
                    "lease_id": None,
                    "field": "quota_upgrade_count",
                    "generation": None,
                    "expected": len(expected_upgrades),
                    "actual": len(upgrade_rows),
                },
            )
        for expected_upgrade, row in zip(expected_upgrades, upgrade_rows, strict=True):
            if row["state_after"] != expected_upgrade["state_after"]:
                raise VerificationError(
                    "quota version mismatch in ledger",
                    path="$.quota_versions",
                    details={
                        "lease_id": None,
                        "field": "quota_version",
                        "generation": None,
                        "expected": expected_upgrade["state_after"],
                        "actual": row["state_after"],
                    },
                )

        return {
            "ok": True,
            "verified_leases": lease_ids,
            "entry_count": len(ledger.entries()),
            "cursor": ledger.cursor(),
            "final_view": replay.domain.current_view(),
        }
    finally:
        ledger.close()
