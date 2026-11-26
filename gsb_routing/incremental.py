"""Invalidation propagation and affected-only reconstruction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, first_field_difference
from .fixedpoint import build_waterline
from .routing import build_event_conclusion, build_route_report
from .spec import EventSpec, RoutingSpec


def batch_ranges(events: Iterable[EventSpec], batch_size: int) -> list[tuple[int, int]]:
    events_list = list(events)
    if not events_list:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ranges: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(events_list):
        end_index = min(len(events_list), start_index + batch_size) - 1
        ranges.append((events_list[start_index].seq, events_list[end_index].seq))
        start_index = end_index + 1
    return ranges


def batch_id_for(spec: RoutingSpec, start_seq: int, end_seq: int) -> str:
    return f"batch-{spec.digest[:16]}-{start_seq}-{end_seq}"


def lease_id_for(spec: RoutingSpec, event_seq: int, database_id: str) -> str:
    return f"lease-{spec.digest[:16]}-{event_seq}-{database_id}"


@dataclass
class _Node:
    kind: str
    identifier: str
    event_seq: int | None = None
    database_id: str | None = None
    batch_start: int | None = None
    batch_end: int | None = None


def _event_dependencies(events: Iterable[EventSpec]) -> dict[int, int]:
    """Return query seq -> nearest preceding write seq for the same key."""

    latest_write: dict[tuple[str, int], int] = {}
    dependency: dict[int, int] = {}
    for event in sorted(events, key=lambda item: item.seq):
        if event.op == "write":
            latest_write[(event.table, event.key)] = event.seq
        elif event.op == "query":
            previous = latest_write.get((event.table, event.key))
            if previous is not None:
                dependency[event.seq] = previous
    return dependency


def _database_changed_fields(before: RoutingSpec, after: RoutingSpec, database_id: str) -> list[str]:
    before_database = next((item for item in before.databases if item.id == database_id), None)
    after_database = next((item for item in after.databases if item.id == database_id), None)
    if before_database is None or after_database is None:
        return ["presence"]
    fields: list[str] = []
    for field_name in (
        "port",
        "seed",
        "lease_limit",
        "concurrency_quota",
        "route_version",
    ):
        if getattr(before_database, field_name) != getattr(after_database, field_name):
            fields.append(field_name)
    return fields


def _event_change_reason(before: RoutingSpec, after: RoutingSpec) -> str:
    before_tables = {table.id: table for table in before.tables}
    after_tables = {table.id: table for table in after.tables}
    for table_id in sorted(set(before_tables) | set(after_tables)):
        old = before_tables.get(table_id)
        new = after_tables.get(table_id)
        if old is None or new is None:
            return "table_presence_changed"
        if old.boundaries != new.boundaries:
            return "boundary_sequence_changed"
        if old.boundary_rule != new.boundary_rule:
            return "boundary_ownership_rule_changed"
        if old.database_assignment != new.database_assignment:
            return "database_assignment_changed"
        if old.key_column != new.key_column:
            return "shard_key_column_changed"
    if before.route_version != after.route_version:
        return "route_version_changed"
    return "route_conclusion_changed"


class InvalidationPlan:
    def __init__(self, before: RoutingSpec, after: RoutingSpec, previous: RoutingSpec | None) -> None:
        self.before = before
        self.after = after
        self.previous = previous
        self.old_report = build_route_report(before, previous)
        self.new_full_report = build_route_report(after, previous)
        self.old_events = {event.seq: event for event in before.events}
        self.new_events = {event.seq: event for event in after.events}
        self.old_conclusions = {event["seq"]: event for event in self.old_report["events"]}
        self.new_conclusions = {event["seq"]: event for event in self.new_full_report["events"]}
        self.direct_events: dict[int, str] = {}
        self.indirect_events: dict[int, str] = {}
        self.direct_leases: list[dict[str, Any]] = []
        self.indirect_leases: list[dict[str, Any]] = []
        self.affected_batches: list[dict[str, Any]] = []
        self.affected_waterline: list[str] = []
        self.steps: list[dict[str, Any]] = []
        self._invalid_nodes: set[tuple[str, str]] = set()
        self._lease_records: dict[str, dict[str, Any]] = {}
        self._batch_records: dict[str, dict[str, Any]] = {}
        self._step_number = 0
        self._build()

    def _next_step(self) -> int:
        self._step_number += 1
        return self._step_number

    def _mark_event(self, seq: int, reason: str, source: str, *, direct: bool) -> bool:
        node = ("event", str(seq))
        if node in self._invalid_nodes:
            return False
        self._invalid_nodes.add(node)
        if direct:
            self.direct_events[seq] = reason
        else:
            self.indirect_events[seq] = reason
        self.steps.append(
            {
                "step": self._next_step(),
                "kind": "event",
                "id": seq,
                "reason": reason,
                "source": source,
                "direct": direct,
            }
        )
        return True

    def _mark_lease(self, record: dict[str, Any], reason: str, source: str, *, direct: bool) -> bool:
        node = ("lease", record["lease_id"])
        if node in self._invalid_nodes:
            return False
        self._invalid_nodes.add(node)
        self._lease_records[record["lease_id"]] = record
        if direct:
            self.direct_leases.append(record)
        else:
            self.indirect_leases.append(record)
        self.steps.append(
            {
                "step": self._next_step(),
                "kind": "lease",
                "id": record["lease_id"],
                "reason": reason,
                "source": source,
                "direct": direct,
                "event_seq": record.get("event_seq"),
                "database_id": record.get("database_id"),
            }
        )
        return True

    def _mark_batch(self, record: dict[str, Any], reason: str, source: str) -> None:
        node = ("batch", record["batch_id"])
        if node in self._invalid_nodes:
            return
        self._invalid_nodes.add(node)
        self._batch_records[record["batch_id"]] = record
        self.affected_batches.append(record)
        self.steps.append(
            {
                "step": self._next_step(),
                "kind": "batch",
                "id": record["batch_id"],
                "reason": reason,
                "source": source,
            }
        )

    def _mark_waterline(self, database_id: str, reason: str, source: str) -> None:
        node = ("waterline", database_id)
        if node in self._invalid_nodes:
            return
        self._invalid_nodes.add(node)
        self.affected_waterline.append(database_id)
        self.steps.append(
            {
                "step": self._next_step(),
                "kind": "waterline",
                "id": database_id,
                "reason": reason,
                "source": source,
            }
        )

    def _lease_record_for(self, spec: RoutingSpec, seq: int, database_id: str, reason: str) -> dict[str, Any]:
        lease_id = lease_id_for(spec, seq, database_id)
        return {
            "lease_id": lease_id,
            "event_seq": seq,
            "database_id": database_id,
            "route_version": spec.route_version,
            "status": "version_invalidated" if reason == "route_version_changed" else "invalidated",
            "decision_reason": reason,
            "decision_event_seq": seq,
            "decision_route_version": spec.route_version,
        }

    def _build(self) -> None:
        old_batches = batch_ranges(self.before.events, 3)
        new_batches = batch_ranges(self.after.events, 3)

        for seq in sorted(set(self.old_events) | set(self.new_events)):
            if seq not in self.old_events:
                self._mark_event(seq, "event_added", "event_stream", direct=True)
            elif seq not in self.new_events:
                self._mark_event(seq, "event_removed", "event_stream", direct=True)
            elif self.old_events[seq] != self.new_events[seq]:
                self._mark_event(seq, "event_replaced", "event_stream", direct=True)
            else:
                old = self.old_conclusions.get(seq)
                new = self.new_conclusions.get(seq)
                if canonical_json(old) != canonical_json(new):
                    reason = _event_change_reason(self.before, self.after)
                    self._mark_event(seq, reason, "route_conclusion", direct=True)

        old_dependencies = _event_dependencies(self.before.events)
        new_dependencies = _event_dependencies(self.after.events)
        dependencies_changed = old_dependencies != new_dependencies

        database_ids = sorted({item.id for item in self.before.databases} | {item.id for item in self.after.databases})
        for database_id in database_ids:
            fields = _database_changed_fields(self.before, self.after, database_id)
            if not fields:
                continue
            for seq in sorted(self.old_events):
                old_conclusion = self.old_conclusions.get(seq)
                if not old_conclusion:
                    continue
                if old_conclusion["route"]["database_id"] != database_id:
                    continue
                reason = "database_lease_policy_changed:" + ",".join(fields)
                self._mark_lease(
                    self._lease_record_for(self.before, seq, database_id, reason),
                    reason,
                    "database_policy",
                    direct=True,
                )

        queue: deque[tuple[str, str]] = deque()
        for seq in sorted(self.direct_events):
            queue.append(("event", str(seq)))
        for record in self.direct_leases:
            queue.append(("lease", record["lease_id"]))

        event_to_batch: dict[int, list[dict[str, Any]]] = {}
        for start_seq, end_seq in old_batches:
            record = {
                "batch_id": batch_id_for(self.before, start_seq, end_seq),
                "start_seq": start_seq,
                "end_seq": end_seq,
                "spec_side": "before",
            }
            for seq in range(start_seq, end_seq + 1):
                event_to_batch.setdefault(seq, []).append(record)
        new_event_to_batch: dict[int, list[dict[str, Any]]] = {}
        for start_seq, end_seq in new_batches:
            record = {
                "batch_id": batch_id_for(self.after, start_seq, end_seq),
                "start_seq": start_seq,
                "end_seq": end_seq,
                "spec_side": "after",
            }
            for event in self.after.events:
                if start_seq <= event.seq <= end_seq:
                    new_event_to_batch.setdefault(event.seq, []).append(record)

        while queue:
            kind, identifier = queue.popleft()
            if kind == "event":
                seq = int(identifier)
                event = self.new_events.get(seq)
                if event is None:
                    event = self.old_events.get(seq)
                if event is not None:
                    if seq in self.old_conclusions:
                        old_database = self.old_conclusions[seq]["route"]["database_id"]
                        record = self._lease_record_for(
                            self.before, seq, old_database, self.direct_events.get(seq, "event_invalidated")
                        )
                        if self._mark_lease(
                            record,
                            "event_route_invalidated",
                            f"event:{seq}",
                            direct=False,
                        ):
                            queue.append(("lease", record["lease_id"]))
                    if seq in self.new_conclusions:
                        new_database = self.new_conclusions[seq]["route"]["database_id"]
                        record = self._lease_record_for(
                            self.after, seq, new_database, self.direct_events.get(seq, "event_invalidated")
                        )
                        # A new lease is flagged in the same step without adding a duplicate node id when possible.
                        if record["lease_id"] not in self._lease_records:
                            self._mark_lease(
                                record,
                                "event_route_invalidated",
                                f"event:{seq}",
                                direct=False,
                            )
                            queue.append(("lease", record["lease_id"]))
                    for batch_record in event_to_batch.get(seq, []):
                        self._mark_batch(batch_record, "batch_contains_invalid_event", f"event:{seq}")
                    for batch_record in new_event_to_batch.get(seq, []):
                        self._mark_batch(batch_record, "batch_contains_invalid_event", f"event:{seq}")
                    if seq in self.old_conclusions:
                        self._mark_waterline(
                            self.old_conclusions[seq]["route"]["database_id"],
                            "event_route_invalidated",
                            f"event:{seq}",
                        )
                if dependencies_changed and seq in old_dependencies:
                    dependency = old_dependencies[seq]
                    if ("event", str(dependency)) in self._invalid_nodes:
                        if self._mark_event(
                            seq,
                            "write_dependency_invalidated",
                            f"event:{dependency}",
                            direct=False,
                        ):
                            queue.append(("event", str(seq)))
            elif kind == "lease":
                record = self._lease_records.get(identifier)
                if record is not None and record.get("database_id") is not None:
                    self._mark_waterline(
                        record["database_id"],
                        "lease_invalidated",
                        f"lease:{identifier}",
                    )
        self.affected_waterline.sort()

    def result(self) -> dict[str, Any]:
        affected_events = sorted(set(self.direct_events) | set(self.indirect_events))
        return {
            "before_spec_digest": self.before.digest,
            "after_spec_digest": self.after.digest,
            "before_route_version": self.before.route_version,
            "after_route_version": self.after.route_version,
            "direct": {
                "events": [{"seq": seq, "reason": self.direct_events[seq]} for seq in sorted(self.direct_events)],
                "leases": sorted(self.direct_leases, key=lambda item: (item["event_seq"], item["database_id"], item["lease_id"])),
            },
            "indirect": {
                "events": [{"seq": seq, "reason": self.indirect_events[seq]} for seq in sorted(self.indirect_events)],
                "leases": sorted(self.indirect_leases, key=lambda item: (item["event_seq"], item["database_id"], item["lease_id"])),
            },
            "affected_events": affected_events,
            "affected_leases": sorted(self._lease_records.values(), key=lambda item: (item["event_seq"], item["database_id"], item["lease_id"])) if False else sorted(
                self._lease_records.values(), key=lambda item: (int(item["event_seq"]), str(item["database_id"]), str(item["lease_id"]))
            ),
            "affected_batches": sorted(
                self._batch_records.values(), key=lambda item: (int(item["start_seq"]), int(item["end_seq"]), item["batch_id"])
            ),
            "affected_waterline": self.affected_waterline,
            "steps": self.steps,
        }


def plan_invalidation(
    before: RoutingSpec,
    after: RoutingSpec,
    previous: RoutingSpec | None = None,
) -> dict[str, Any]:
    return InvalidationPlan(before, after, previous).result()


def apply_incremental(
    before: RoutingSpec,
    after: RoutingSpec,
    previous: RoutingSpec | None = None,
    *,
    raise_on_mismatch: bool = False,
) -> dict[str, Any]:
    plan = InvalidationPlan(before, after, previous)
    affected = set(plan.direct_events) | set(plan.indirect_events)
    old_by_seq = plan.old_conclusions
    rebuilt: list[dict[str, Any]] = []
    recalculated_seq: list[int] = []
    reused_seq: list[int] = []
    for event in sorted(after.events, key=lambda item: item.seq):
        if event.seq in affected or event.seq not in old_by_seq:
            rebuilt.append(build_event_conclusion(after, previous, event))
            recalculated_seq.append(event.seq)
        else:
            rebuilt.append(old_by_seq[event.seq])
            reused_seq.append(event.seq)
    assembled = {
        "format_version": after.format_version,
        "route_version": after.route_version,
        "previous_route_version": previous.route_version if previous is not None else None,
        "spec_digest": after.digest,
        "previous_spec_digest": previous.digest if previous is not None else None,
        "boundary_ownership": {
            table.id: table.boundary_rule for table in sorted(after.tables, key=lambda item: item.id)
        },
        "events": rebuilt,
        "waterline": build_waterline(after, rebuilt),
    }
    consistency = compare_reports(assembled, plan.new_full_report)
    result = {
        "invalidation": plan.result(),
        "report": assembled,
        "recalculated_events": recalculated_seq,
        "reused_events": reused_seq,
        "consistency": consistency,
    }
    if raise_on_mismatch and not consistency["consistent"]:
        from .errors import replay_error

        raise replay_error(
            consistency.get("event_seq"),
            consistency.get("field") or "$",
            "incremental reconstruction differs from blank full rebuild",
        )
    return result


def compare_reports(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_events = {int(event["seq"]): event for event in left.get("events", [])}
    right_events = {int(event["seq"]): event for event in right.get("events", [])}
    for seq in sorted(set(left_events) | set(right_events)):
        left_event = left_events.get(seq)
        right_event = right_events.get(seq)
        if left_event is None or right_event is None:
            return {
                "consistent": False,
                "event_seq": seq,
                "field": "presence",
                "left_present": left_event is not None,
                "right_present": right_event is not None,
            }
        difference = first_field_difference(left_event, right_event)
        if difference is not None:
            return {
                "consistent": False,
                "event_seq": seq,
                "field": difference,
            }
    report_fields = [key for key in set(left) | set(right) if key != "events"]
    for field_name in sorted(report_fields):
        difference = first_field_difference(left.get(field_name), right.get(field_name), field_name)
        if difference is not None:
            return {
                "consistent": False,
                "event_seq": None,
                "field": difference,
            }
    return {"consistent": True, "event_seq": None, "field": None}


__all__ = [
    "InvalidationPlan",
    "apply_incremental",
    "batch_id_for",
    "batch_ranges",
    "compare_reports",
    "lease_id_for",
    "plan_invalidation",
]
