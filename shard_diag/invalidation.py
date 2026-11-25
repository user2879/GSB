from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .canonical import canonical_json, sha256_digest
from .errors import DiagnosticError, ErrorCode
from .routing import first_boundary_diff
from .spec import TableRule


def _record_key(record: dict[str, Any]) -> tuple[int, str, str, int]:
    return (
        int(record.get("event_seq", -1)),
        str(record.get("db_id", "")),
        str(record.get("table", "")),
        int(record.get("route_key", 0)),
    )


def _lease_key(lease: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(lease.get("event_seq", -1)),
        str(lease.get("db_id", "")),
        str(lease.get("shard_id", "")),
        str(lease.get("lease_id", "")),
    )


@dataclass(frozen=True, slots=True)
class InvalidationReport:
    table: str
    old_version: int
    new_version: int
    boundary_index: int | None
    boundary_offset: int
    direct_record_keys: tuple[tuple[int, str, str, int], ...]
    invalid_record_keys: tuple[tuple[int, str, str, int], ...]
    invalid_event_seqs: tuple[int, ...]
    invalid_lease_ids: tuple[str, ...]
    invalid_response_keys: tuple[tuple[int, str, str], ...]
    steps: tuple[dict[str, Any], ...]
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "boundary_index": self.boundary_index,
            "boundary_offset": self.boundary_offset,
            "direct_record_keys": [list(key) for key in self.direct_record_keys],
            "invalid_record_keys": [list(key) for key in self.invalid_record_keys],
            "invalid_event_seqs": list(self.invalid_event_seqs),
            "invalid_lease_ids": list(self.invalid_lease_ids),
            "invalid_response_keys": [list(key) for key in self.invalid_response_keys],
            "steps": list(self.steps),
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class RecomputeResult:
    records: tuple[dict[str, Any], ...]
    recomputed_record_keys: tuple[tuple[int, str, str, int], ...]
    unaffected_before_digest: str
    unaffected_after_digest: str
    unaffected_byte_identical: bool
    canonical_bytes: bytes


class InvalidationEngine:
    """Deterministic propagation closure for route/lease invalidation."""

    def __init__(
        self,
        route_records: Iterable[dict[str, Any]],
        lease_records: Iterable[dict[str, Any]] = (),
        response_records: Iterable[dict[str, Any]] = (),
    ) -> None:
        self.route_records = sorted((copy.deepcopy(record) for record in route_records), key=_record_key)
        self.lease_records = sorted((copy.deepcopy(lease) for lease in lease_records), key=_lease_key)
        self.response_records = sorted(
            (copy.deepcopy(record) for record in response_records),
            key=lambda item: (int(item.get("event_seq", -1)), str(item.get("db_id", "")), str(item.get("table", ""))),
        )
        if not self.response_records:
            self.response_records = [
                {"event_seq": record["event_seq"], "db_id": record["db_id"], "table": record["table"]}
                for record in self.route_records
            ]

    def _direct_stale_records(self, table: str, old_version: int) -> list[dict[str, Any]]:
        return [
            record
            for record in self.route_records
            if record.get("table") == table
            and record.get("new_route_version") is not None
            and int(record["new_route_version"]) <= old_version
        ]

    def invalidate_table_version(
        self,
        previous_rule: TableRule,
        current_rule: TableRule,
    ) -> InvalidationReport:
        if previous_rule.name != current_rule.name:
            raise DiagnosticError(ErrorCode.STATE.value, "cannot evolve different tables", field="table")
        if current_rule.route_version <= previous_rule.route_version:
            raise DiagnosticError(
                ErrorCode.STATE.value,
                "route table version must increase",
                field="route_version",
                key_column=current_rule.key_column,
            )
        table = previous_rule.name
        old_version = previous_rule.route_version
        new_version = current_rule.route_version
        diff = first_boundary_diff(previous_rule, current_rule)
        invalid_keys: set[tuple[int, str, str, int]] = set()
        direct_keys: set[tuple[int, str, str, int]] = set()
        steps: list[dict[str, Any]] = []
        reason_by_key: dict[tuple[int, str, str, int], list[str]] = {}

        def add_step(
            node_type: str,
            node_id: str,
            basis: str,
            *,
            event_seq: int | None = None,
            db_id: str | None = None,
        ) -> None:
            steps.append(
                {
                    "order": len(steps) + 1,
                    "node_type": node_type,
                    "node_id": node_id,
                    "event_seq": event_seq,
                    "db_id": db_id,
                    "basis": basis,
                }
            )

        add_step(
            "table_rule",
            f"{table}@{old_version}->{new_version}",
            (
                f"route table version increased {old_version}->{new_version}; "
                f"first boundary difference index={diff.index}, offset={diff.offset:+d}"
            ),
        )

        def invalidate(record: dict[str, Any], causes: list[str], *, direct: bool) -> bool:
            key = _record_key(record)
            if key in invalid_keys:
                return False
            invalid_keys.add(key)
            if direct:
                direct_keys.add(key)
            reason_by_key[key] = sorted(set(causes))
            add_step(
                "route_result",
                f"event:{key[0]}/{key[1]}/{key[2]}",
                "; ".join(reason_by_key[key]),
                event_seq=key[0],
                db_id=key[1],
            )
            return True

        direct_records = self._direct_stale_records(table, old_version)
        for record in direct_records:
            invalidate(
                record,
                [
                    f"record used table {table!r} generation {record.get('new_route_version')} "
                    f"which is superseded by {new_version}"
                ],
                direct=True,
            )

        changed = True
        while changed:
            changed = False
            invalid_records = [record for record in self.route_records if _record_key(record) in invalid_keys]
            invalid_key_columns = sorted(
                {str(record["key_column"]) for record in invalid_records if record.get("key_column") is not None}
            )
            invalid_route_keys = sorted({int(record["route_key"]) for record in invalid_records})
            for record in self.route_records:
                key = _record_key(record)
                if key in invalid_keys:
                    continue
                key_column = str(record.get("key_column", ""))
                route_key = int(record.get("route_key", 0))
                causes: list[str] = []
                if key_column in invalid_key_columns:
                    causes.append(f"shard key column {key_column!r} is referenced by an invalid route result")
                if route_key in invalid_route_keys:
                    causes.append(f"route key {route_key} is referenced by an invalid route result")
                if causes:
                    changed = invalidate(record, causes, direct=False) or changed

        invalid_event_seqs = sorted({key[0] for key in invalid_keys})
        invalid_response_keys = tuple(
            sorted(
                (int(response["event_seq"]), str(response["db_id"]), str(response["table"]))
                for response in self.response_records
                if int(response.get("event_seq", -1)) in invalid_event_seqs
            )
        )
        invalid_leases = [
            lease
            for lease in self.lease_records
            if int(lease.get("event_seq", -1)) in invalid_event_seqs
        ]
        for event_seq in invalid_event_seqs:
            add_step(
                "response",
                f"response:event:{event_seq}",
                "event response is invalid because its route result is in the invalidation closure",
                event_seq=event_seq,
            )
        for lease in invalid_leases:
            add_step(
                "lease",
                str(lease["lease_id"]),
                "lease references an invalid event sequence through its route-lease ownership relation",
                event_seq=int(lease["event_seq"]),
                db_id=str(lease["db_id"]),
            )

        invalid_keys_sorted = tuple(sorted(invalid_keys))
        report_payload = {
            "table": table,
            "old_version": old_version,
            "new_version": new_version,
            "boundary_index": diff.index,
            "boundary_offset": diff.offset,
            "direct_record_keys": [list(key) for key in sorted(direct_keys)],
            "invalid_record_keys": [list(key) for key in invalid_keys_sorted],
            "invalid_event_seqs": invalid_event_seqs,
            "invalid_lease_ids": sorted(str(lease["lease_id"]) for lease in invalid_leases),
            "invalid_response_keys": [list(key) for key in invalid_response_keys],
            "steps": steps,
        }
        return InvalidationReport(
            table=table,
            old_version=old_version,
            new_version=new_version,
            boundary_index=diff.index,
            boundary_offset=diff.offset,
            direct_record_keys=tuple(sorted(direct_keys)),
            invalid_record_keys=invalid_keys_sorted,
            invalid_event_seqs=tuple(invalid_event_seqs),
            invalid_lease_ids=tuple(sorted(report_payload["invalid_lease_ids"])),
            invalid_response_keys=invalid_response_keys,
            steps=tuple(steps),
            digest=sha256_digest(report_payload),
        )

    def recompute_invalid_subset(
        self,
        report: InvalidationReport,
        recompute: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> RecomputeResult:
        invalid_keys = set(report.invalid_record_keys)
        unaffected_before = [record for record in self.route_records if _record_key(record) not in invalid_keys]
        unaffected_digest_before = sha256_digest(unaffected_before)
        replacements: dict[tuple[int, str, str, int], dict[str, Any]] = {}
        for record in self.route_records:
            key = _record_key(record)
            if key not in invalid_keys:
                continue
            replacement = recompute(copy.deepcopy(record))
            replacement_key = _record_key(replacement)
            if replacement_key != key:
                raise DiagnosticError(
                    ErrorCode.STATE.value,
                    "recomputed record changed its stable event/database/table/key identity",
                    db_id=key[1],
                    field="record_key",
                )
            replacements[key] = replacement
        final_records = sorted(
            [replacements.get(_record_key(record), copy.deepcopy(record)) for record in self.route_records],
            key=_record_key,
        )
        unaffected_after = [record for record in final_records if _record_key(record) not in invalid_keys]
        unaffected_digest_after = sha256_digest(unaffected_after)
        identical = canonical_json(unaffected_before) == canonical_json(unaffected_after)
        return RecomputeResult(
            records=tuple(final_records),
            recomputed_record_keys=tuple(sorted(replacements)),
            unaffected_before_digest=unaffected_digest_before,
            unaffected_after_digest=unaffected_digest_after,
            unaffected_byte_identical=identical,
            canonical_bytes=canonical_json(final_records).encode("utf-8"),
        )
