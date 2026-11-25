from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import digest
from .errors import LeaseError
from .spec import Event, Spec
from .state import ACTIVE_STATES, LeaseState, SlotPool, transition


BATCH_RULE = {
    "event_count_min": 10,
    "node_count_min": 2,
    "remaining_capacity_min": 2,
    "operator": "AND",
    "default_when_not_triggered": "sequential",
}


@dataclass
class MutationBatch:
    transitions: list[dict[str, Any]] = field(default_factory=list)
    ownership_changes: list[dict[str, Any]] = field(default_factory=list)
    reclaims: list[dict[str, Any]] = field(default_factory=list)
    outcome: dict[str, Any] | None = None


def choose_path(
    *,
    requested_path: str,
    node_count: int,
    event_count: int,
    remaining_capacity: int,
) -> dict[str, Any]:
    if requested_path not in {"auto", "sequential", "batch"}:
        raise ValueError("path must be auto, sequential, or batch")
    thresholds = {
        "event_count": BATCH_RULE["event_count_min"],
        "node_count": BATCH_RULE["node_count_min"],
        "remaining_capacity": BATCH_RULE["remaining_capacity_min"],
    }
    actual = {
        "event_count": event_count,
        "node_count": node_count,
        "remaining_capacity": remaining_capacity,
    }
    conditions = {
        name: {
            "actual": actual[name],
            "threshold": thresholds[name],
            "met": actual[name] >= thresholds[name],
            "margin": actual[name] - thresholds[name],
            "boundary_value": thresholds[name],
        }
        for name in ("event_count", "node_count", "remaining_capacity")
    }
    auto_triggered = all(item["met"] for item in conditions.values())
    if requested_path == "auto":
        selected = "batch" if auto_triggered else "sequential"
        if auto_triggered:
            reason = "all explicit batch scale thresholds are met"
        else:
            first_unmet = next(
                name for name in ("event_count", "node_count", "remaining_capacity") if not conditions[name]["met"]
            )
            reason = f"{first_unmet} is below its batch threshold"
    else:
        selected = requested_path
        reason = f"path explicitly forced to {requested_path}"
    return {
        "format_version": 1,
        "selected": selected,
        "requested": requested_path,
        "auto_triggered": auto_triggered,
        "reason": reason,
        "rule": dict(BATCH_RULE),
        "params": actual,
        "conditions": conditions,
        "boundary": thresholds,
        "not_triggered_boundary": {
            name: {
                "threshold": thresholds[name],
                "needed_delta": max(0, thresholds[name] - actual[name]),
            }
            for name in ("event_count", "node_count", "remaining_capacity")
        },
    }


class LeaseArbiterEngine:
    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self.leases: dict[str, LeaseState] = {
            lease.lease_id: LeaseState(
                lease_id=lease.lease_id,
                node_id=lease.node_id,
                lease_type=lease.lease_type,
                ttl_ms=lease.ttl_ms,
                renew_interval_ms=lease.renew_interval_ms,
                quota_version=lease.quota_version,
            )
            for lease in spec.leases
        }
        limits = {node.node_id: node.max_leases for node in spec.nodes}
        self.slots = SlotPool(spec.global_concurrency, limits)
        self.quota_epochs: dict[tuple[str, str], int] = {}
        for lease in sorted(
            spec.leases,
            key=lambda item: (item.node_id, item.lease_type, item.quota_version, item.lease_id),
        ):
            key = (lease.node_id, lease.lease_type)
            self.quota_epochs[key] = max(self.quota_epochs.get(key, 0), lease.quota_version)
        self.last_event_seq = 0
        self.transitions: list[dict[str, Any]] = []
        self.ownership_changes: list[dict[str, Any]] = []
        self.reclaims: list[dict[str, Any]] = []
        self.event_outcomes: list[dict[str, Any]] = []

    @property
    def lease_by_id(self) -> dict[str, Any]:
        return self.spec.lease_by_id

    def expire_due(self, now_ms: int, trigger_event_seq: int) -> MutationBatch:
        batch = MutationBatch()
        due_ids = sorted(
            lease_id
            for lease_id, record in self.leases.items()
            if record.active and record.expires_at_ms is not None and record.expires_at_ms <= now_ms
        )
        for lease_id in due_ids:
            record = self.leases[lease_id]
            if not record.active:
                continue
            owner_before = record.owner
            expires_at_ms = record.expires_at_ms
            global_slot = record.global_slot
            node_slot = record.node_slot
            item = transition(
                record,
                "expired",
                "expired",
                record.label,
                trigger_event_seq,
                now_ms,
            )
            record.expires_at_ms = None
            record.global_slot = None
            record.node_slot = None
            record.owner = None
            self.slots.release(lease_id, record.node_id, global_slot, node_slot)
            reclaim = {
                "event_seq": trigger_event_seq,
                "lease_id": lease_id,
                "node_id": record.node_id,
                "lease_type": record.lease_type,
                "reason": "expired",
                "label_before": item["label_before"],
                "label_after": item["label_after"],
                "owner_before": owner_before,
                "global_slot": global_slot,
                "node_slot": node_slot,
                "expires_at_ms": expires_at_ms,
            }
            ownership = self._ownership_change(
                event_seq=trigger_event_seq,
                lease_id=lease_id,
                node_id=record.node_id,
                owner_before=owner_before,
                owner_after=None,
                label=record.label,
                reason="expired",
                global_slot=global_slot,
                node_slot=node_slot,
            )
            batch.transitions.append(item)
            batch.reclaims.append(reclaim)
            batch.ownership_changes.append(ownership)
        self.transitions.extend(batch.transitions)
        self.reclaims.extend(batch.reclaims)
        self.ownership_changes.extend(batch.ownership_changes)
        self.slots.assert_consistent(self.leases)
        return batch

    def apply_event(self, event: Event) -> MutationBatch:
        if event.seq <= self.last_event_seq:
            raise LeaseError(
                "event sequence would move backwards",
                event_seq=event.seq,
                last_event_seq=self.last_event_seq,
            )
        if event.op == "quota_invalidate":
            result = self._apply_quota_invalidate(event)
            self.last_event_seq = event.seq
            self.event_outcomes.append(result.outcome)  # type: ignore[arg-type]
            return result

        record = self.leases.get(event.lease_id or "")
        if record is None:
            raise LeaseError("event references an unknown lease", event_seq=event.seq, lease_id=event.lease_id)
        batch = MutationBatch()
        expected_label = int(event.expected_label)
        if event.op == "grant":
            if record.active:
                batch.outcome = self._outcome(event, record, "E_LEASE", "lease is already active")
            elif expected_label != record.label:
                batch.outcome = self._outcome(event, record, "E_STALE", "request label is stale")
            elif record.quota_version < self.quota_epochs.get((record.node_id, record.lease_type), 0):
                batch.outcome = self._outcome(event, record, "E_LEASE", "quota generation is obsolete")
            else:
                slots = self.slots.acquire(record.lease_id, record.node_id)
                if slots is None:
                    batch.outcome = self._outcome(event, record, "E_QUOTA", "lease quota exhausted")
                else:
                    global_slot, node_slot = slots
                    item = transition(
                        record,
                        "granted",
                        "granted",
                        expected_label,
                        event.seq,
                        event.at_ms,
                    )
                    record.expires_at_ms = event.at_ms + record.ttl_ms
                    record.global_slot = global_slot
                    record.node_slot = node_slot
                    record.owner = record.node_id
                    ownership = self._ownership_change(
                        event_seq=event.seq,
                        lease_id=record.lease_id,
                        node_id=record.node_id,
                        owner_before=None,
                        owner_after=record.node_id,
                        label=record.label,
                        reason="granted",
                        global_slot=global_slot,
                        node_slot=node_slot,
                    )
                    batch.transitions.append(item)
                    batch.ownership_changes.append(ownership)
                    batch.outcome = self._outcome(event, record, "granted", event.reason or "granted")
        elif event.op == "renew":
            if expected_label != record.label:
                batch.outcome = self._outcome(event, record, "E_STALE", "request label is stale")
            elif record.state not in ACTIVE_STATES:
                batch.outcome = self._outcome(event, record, "E_LEASE", "lease is not active")
            elif record.quota_version < self.quota_epochs.get((record.node_id, record.lease_type), 0):
                batch.outcome = self._outcome(event, record, "E_LEASE", "quota generation is obsolete")
            else:
                item = transition(
                    record,
                    "renewed",
                    "renewed",
                    expected_label,
                    event.seq,
                    event.at_ms,
                )
                record.expires_at_ms = event.at_ms + record.ttl_ms
                batch.transitions.append(item)
                batch.outcome = self._outcome(event, record, "renewed", event.reason or "renewed")
        elif event.op == "release":
            if expected_label != record.label:
                batch.outcome = self._outcome(event, record, "E_STALE", "request label is stale")
            elif record.state not in ACTIVE_STATES:
                batch.outcome = self._outcome(event, record, "E_LEASE", "lease is not active")
            else:
                owner_before = record.owner
                global_slot = record.global_slot
                node_slot = record.node_slot
                expires_at_ms = record.expires_at_ms
                item = transition(
                    record,
                    "released",
                    "released",
                    expected_label,
                    event.seq,
                    event.at_ms,
                )
                record.expires_at_ms = None
                record.global_slot = None
                record.node_slot = None
                record.owner = None
                self.slots.release(record.lease_id, record.node_id, global_slot, node_slot)
                ownership = self._ownership_change(
                    event_seq=event.seq,
                    lease_id=record.lease_id,
                    node_id=record.node_id,
                    owner_before=owner_before,
                    owner_after=None,
                    label=record.label,
                    reason="released",
                    global_slot=global_slot,
                    node_slot=node_slot,
                )
                batch.transitions.append(item)
                batch.ownership_changes.append(ownership)
                batch.outcome = self._outcome(event, record, "released", event.reason or "released")
                batch.outcome["released_global_slot"] = global_slot
                batch.outcome["released_node_slot"] = node_slot
                batch.outcome["released_expires_at_ms"] = expires_at_ms
        else:
            raise LeaseError("unsupported event op", event_seq=event.seq, op=event.op)

        self.transitions.extend(batch.transitions)
        self.ownership_changes.extend(batch.ownership_changes)
        self.event_outcomes.append(batch.outcome)  # type: ignore[arg-type]
        self.last_event_seq = event.seq
        self.slots.assert_consistent(self.leases)
        return batch

    def _apply_quota_invalidate(self, event: Event) -> MutationBatch:
        batch = MutationBatch()
        key = (event.node_id or "", event.lease_type or "")
        target_version = int(event.quota_version)
        previous_epoch = self.quota_epochs.get(key, 0)
        self.quota_epochs[key] = max(previous_epoch, target_version)
        target_ids = sorted(
            (
                record.lease_id,
                record.quota_version,
            )
            for record in self.leases.values()
            if record.node_id == key[0]
            and record.lease_type == key[1]
            and record.active
            and record.quota_version < target_version
        )
        target_ids = [lease_id for lease_id, _ in sorted(target_ids, key=lambda item: (item[1], item[0]))]
        for lease_id in target_ids:
            record = self.leases[lease_id]
            if not record.active:
                continue
            owner_before = record.owner
            expires_at_ms = record.expires_at_ms
            global_slot = record.global_slot
            node_slot = record.node_slot
            item = transition(
                record,
                "expired",
                "quota_invalidated",
                record.label,
                event.seq,
                event.at_ms,
            )
            record.expires_at_ms = None
            record.global_slot = None
            record.node_slot = None
            record.owner = None
            self.slots.release(record.lease_id, record.node_id, global_slot, node_slot)
            reclaim = {
                "event_seq": event.seq,
                "lease_id": record.lease_id,
                "node_id": record.node_id,
                "lease_type": record.lease_type,
                "reason": "quota_invalidated",
                "label_before": item["label_before"],
                "label_after": item["label_after"],
                "owner_before": owner_before,
                "global_slot": global_slot,
                "node_slot": node_slot,
                "expires_at_ms": expires_at_ms,
            }
            ownership = self._ownership_change(
                event_seq=event.seq,
                lease_id=record.lease_id,
                node_id=record.node_id,
                owner_before=owner_before,
                owner_after=None,
                label=record.label,
                reason="quota_invalidated",
                global_slot=global_slot,
                node_slot=node_slot,
            )
            batch.transitions.append(item)
            batch.reclaims.append(reclaim)
            batch.ownership_changes.append(ownership)
        batch.outcome = {
            "seq": event.seq,
            "op": event.op,
            "lease_id": None,
            "node_id": key[0],
            "lease_type": key[1],
            "quota_version": target_version,
            "status": "invalidated",
            "label": None,
            "state": None,
            "global_slot": None,
            "node_slot": None,
            "expires_at_ms": None,
            "reclaimed_count": len(batch.reclaims),
            "reason": event.reason or "quota_invalidated",
        }
        self.transitions.extend(batch.transitions)
        self.reclaims.extend(batch.reclaims)
        self.ownership_changes.extend(batch.ownership_changes)
        self.slots.assert_consistent(self.leases)
        return batch

    def _outcome(self, event: Event, record: LeaseState, status: str, reason: str) -> dict[str, Any]:
        return {
            "seq": event.seq,
            "op": event.op,
            "lease_id": record.lease_id,
            "node_id": record.node_id,
            "lease_type": record.lease_type,
            "status": status,
            "label": record.label,
            "state": record.state,
            "global_slot": record.global_slot,
            "node_slot": record.node_slot,
            "expires_at_ms": record.expires_at_ms,
            "reason": reason,
        }

    @staticmethod
    def _ownership_change(
        *,
        event_seq: int,
        lease_id: str,
        node_id: str,
        owner_before: str | None,
        owner_after: str | None,
        label: int,
        reason: str,
        global_slot: int | None,
        node_slot: int | None,
    ) -> dict[str, Any]:
        return {
            "event_seq": event_seq,
            "lease_id": lease_id,
            "node_id": node_id,
            "owner_before": owner_before,
            "owner_after": owner_after,
            "label": label,
            "reason": reason,
            "global_slot": global_slot,
            "node_slot": node_slot,
        }

    def snapshot(self) -> dict[str, Any]:
        self.slots.assert_consistent(self.leases)
        return {
            "format_version": 1,
            "last_event_seq": self.last_event_seq,
            "leases": [
                self.leases[lease_id].to_snapshot() for lease_id in sorted(self.leases)
            ],
            "quota_epochs": [
                {
                    "node_id": key[0],
                    "lease_type": key[1],
                    "quota_version": self.quota_epochs[key],
                }
                for key in sorted(self.quota_epochs)
            ],
            "slots": self.slots.snapshot(),
        }

    def restore_snapshot(self, payload: dict[str, Any]) -> None:
        if payload.get("format_version") != 1:
            raise LeaseError("unsupported snapshot format_version", format_version=payload.get("format_version"))
        restored: dict[str, LeaseState] = {}
        for item in payload.get("leases", []):
            record = LeaseState.from_snapshot(item)
            if record.lease_id not in self.leases:
                raise LeaseError("snapshot contains an undeclared lease", lease_id=record.lease_id)
            restored[record.lease_id] = record
        if set(restored) != set(self.leases):
            raise LeaseError("snapshot does not contain exactly the declared leases")
        pool = SlotPool.from_snapshot(payload["slots"])
        pool.assert_consistent(restored)
        quota_epochs: dict[tuple[str, str], int] = {}
        for item in payload.get("quota_epochs", []):
            quota_epochs[(item["node_id"], item["lease_type"])] = int(item["quota_version"])
        self.leases = restored
        self.slots = pool
        self.quota_epochs = quota_epochs
        self.last_event_seq = int(payload.get("last_event_seq", 0))

    def final_leases(self) -> list[dict[str, Any]]:
        return [
            {
                "lease_id": record.lease_id,
                "node_id": record.node_id,
                "lease_type": record.lease_type,
                "state": record.state,
                "label": record.label,
                "owner": record.owner,
                "global_slot": record.global_slot,
                "node_slot": record.node_slot,
                "expires_at_ms": record.expires_at_ms,
            }
            for record in (self.leases[lease_id] for lease_id in sorted(self.leases))
        ]

    def comparison_projection(self) -> dict[str, Any]:
        fields = (
            "seq",
            "op",
            "lease_id",
            "node_id",
            "lease_type",
            "status",
            "label",
            "state",
            "global_slot",
            "node_slot",
            "expires_at_ms",
        )
        events = [
            {field: outcome.get(field) for field in fields}
            for outcome in sorted(self.event_outcomes, key=lambda item: item["seq"])
        ]
        reclaim_fields = (
            "event_seq",
            "lease_id",
            "node_id",
            "lease_type",
            "reason",
            "label_after",
            "owner_before",
            "global_slot",
            "node_slot",
            "expires_at_ms",
        )
        reclaims = [
            {field: reclaim.get(field) for field in reclaim_fields}
            for reclaim in self.reclaims
        ]
        return {
            "format_version": 1,
            "events": events,
            "reclaims": reclaims,
            "quota_epochs": [
                {
                    "node_id": key[0],
                    "lease_type": key[1],
                    "quota_version": self.quota_epochs[key],
                }
                for key in sorted(self.quota_epochs)
            ],
            "final_leases": self.final_leases(),
        }

    def result(self, path: str, selection: dict[str, Any], batch_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "path": path,
            "selection": selection,
            "selection_reason": selection["reason"],
            "event_outcomes": sorted(self.event_outcomes, key=lambda item: item["seq"]),
            "reclaims": list(self.reclaims),
            "transitions": list(self.transitions),
            "ownership_changes": list(self.ownership_changes),
            "final_state": self.snapshot(),
            "comparison_projection": self.comparison_projection(),
            "summary": {
                "event_count": len(self.event_outcomes),
                "granted": sum(1 for item in self.event_outcomes if item["status"] == "granted"),
                "renewed": sum(1 for item in self.event_outcomes if item["status"] == "renewed"),
                "released": sum(1 for item in self.event_outcomes if item["status"] == "released"),
                "rejected": sum(1 for item in self.event_outcomes if item["status"].startswith("E_")),
                "reclaim_count": len(self.reclaims),
                "ownership_change_count": len(self.ownership_changes),
                "transition_count": len(self.transitions),
            },
        }
        if batch_stats is not None:
            payload["batch_stats"] = batch_stats
        payload["comparison_sha256"] = digest(payload["comparison_projection"])
        return payload


def run_arbitration(spec: Spec, requested_path: str = "auto") -> dict[str, Any]:
    selection = choose_path(
        requested_path=requested_path,
        node_count=len(spec.nodes),
        event_count=len(spec.events),
        remaining_capacity=spec.global_concurrency,
    )
    path = selection["selected"]
    engine = LeaseArbiterEngine(spec)
    batch_stats: dict[str, Any] | None = None
    if path == "sequential":
        for event in spec.events:
            engine.expire_due(event.at_ms, event.seq)
            engine.apply_event(event)
    else:
        group_count = 0
        frame_count = 0
        current_at: int | None = None
        frame: list[Event] = []
        for event in spec.events:
            if current_at is None or event.at_ms == current_at:
                current_at = event.at_ms
                frame.append(event)
                continue
            frame_count += 1
            group_count += _apply_batch_frame(engine, frame)
            frame = [event]
            current_at = event.at_ms
        if frame:
            frame_count += 1
            group_count += _apply_batch_frame(engine, frame)
        batch_stats = {
            "frame_count": frame_count,
            "group_count": group_count,
            "aggregation_keys": "node_id+lease_type",
            "backfill_order": "event seq",
        }
    return engine.result(path, selection, batch_stats)


def _apply_batch_frame(engine: LeaseArbiterEngine, frame: list[Event]) -> int:
    if not frame:
        return 0
    # Batch path phase 1: aggregate the whole time frame by node + lease type.
    groups: dict[tuple[str, str], list[Event]] = {}
    for event in frame:
        if event.op == "quota_invalidate":
            key = (event.node_id or "", event.lease_type or "")
        else:
            lease = engine.lease_by_id[event.lease_id or ""]
            key = (lease.node_id, lease.lease_type)
        groups.setdefault(key, []).append(event)
    # Phase 2: expiry is adjudicated once at the frame boundary.
    engine.expire_due(frame[0].at_ms, frame[0].seq)
    # Phase 3: unified adjudication.  Sorting by seq after aggregation keeps
    # shared global quota decisions independent of hash/group iteration order.
    merged = sorted(
        (event for group_events in groups.values() for event in group_events),
        key=lambda item: item.seq,
    )
    for event in merged:
        engine.apply_event(event)
    return len(groups)


def replay_events(
    spec: Spec,
    path: str = "sequential",
    *,
    stop_after_seq: int | None = None,
) -> tuple[LeaseArbiterEngine, dict[str, Any]]:
    selection = choose_path(
        requested_path=path,
        node_count=len(spec.nodes),
        event_count=len(spec.events),
        remaining_capacity=spec.global_concurrency,
    )
    engine = LeaseArbiterEngine(spec)
    for event in spec.events:
        if stop_after_seq is not None and event.seq > stop_after_seq:
            break
        engine.expire_due(event.at_ms, event.seq)
        engine.apply_event(event)
    return engine, selection
