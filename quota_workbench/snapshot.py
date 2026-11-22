"""Versioned, digest-protected whole-session snapshots and restore semantics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import (
    FORMAT_VERSION,
    canonical_json,
    chain_digest,
    digest_value,
    make_readonly,
    sha256_bytes,
    write_canonical,
)
from .engine import RuntimeState, build_matrix
from .errors import WorkbenchError
from .ledger import EMPTY_LEDGER_DIGEST, Ledger
from .session import WorkbenchSession
from .spec import Declaration, load_declaration, load_events, parse_declaration


SNAPSHOT_KIND = "quota-workbench-snapshot"


@dataclass
class Snapshot:
    path: Path
    declaration: Declaration
    versions: dict[str, Any]
    records: list[dict[str, Any]]
    ledger_cursor: int
    ledger_head_digest: str
    runtime_state: dict[str, Any]
    final_matrix: dict[str, Any]
    content_digest: str
    payload: dict[str, Any]


@dataclass
class RestoreResult:
    session: WorkbenchSession
    snapshot: Snapshot
    change_report: dict[str, Any]


def build_snapshot_payload(session: WorkbenchSession) -> dict[str, Any]:
    recovery = session.recovery_report()
    if recovery["status"] != "CLEAN":
        raise WorkbenchError(
            "E_SNAPSHOT",
            "cannot snapshot a session with an uncommitted or unknown event attempt",
            object_type="snapshot",
            field="recovery",
            details=recovery,
        )
    session.ledger.verify_chain()
    records = session.ledger.export_snapshot_events()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": SNAPSHOT_KIND,
        "declaration": session.declaration.to_dict(),
        "versions": session.declaration.versions(),
        "ledger": {
            "format_version": FORMAT_VERSION,
            "cursor": session.cursor,
            "head_digest": session.ledger.head_digest,
            "event_count": len(records),
            "event_records": records,
        },
        "runtime_state": session.state.to_dict(session.declaration),
        "final_matrix": session.current_matrix(),
    }
    return payload


def save_snapshot(session: WorkbenchSession, path: Path | str) -> Snapshot:
    target = Path(path)
    if target.exists():
        raise WorkbenchError(
            "E_SNAPSHOT",
            "snapshot destination already exists; snapshots are write-once",
            object_type="snapshot",
            object_id=str(target),
            field="path",
        )
    payload = build_snapshot_payload(session)
    snapshot = dict(payload)
    snapshot["content_digest"] = digest_value(payload)
    temp = target.with_name(target.name + ".tmp")
    if temp.exists():
        temp.unlink()
    write_canonical(temp, snapshot)
    try:
        os.replace(temp, target)
        make_readonly(target)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    return load_snapshot(target)


def _snapshot_error(message: str, field: str, details: dict[str, Any] | None = None) -> WorkbenchError:
    return WorkbenchError(
        "E_SNAPSHOT",
        message,
        object_type="snapshot",
        field=field,
        details=details or {},
    )


def load_snapshot(path: Path | str, *, force_digest_mismatch: bool = False) -> Snapshot:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise WorkbenchError(
            "E_IO", f"cannot read snapshot {source}: {exc}", field="path"
        ) from exc
    except json.JSONDecodeError as exc:
        raise _snapshot_error(
            f"invalid snapshot JSON: {exc.msg}", "json", {"line": exc.lineno, "column": exc.colno}
        ) from exc
    if not isinstance(data, dict):
        raise _snapshot_error("snapshot root must be an object", "$")
    if data.get("kind") != SNAPSHOT_KIND:
        raise _snapshot_error("snapshot kind is not recognized", "kind")
    version = data.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != FORMAT_VERSION:
        raise _snapshot_error("snapshot format_version is unsupported", "format_version")
    content_digest = data.get("content_digest")
    if not isinstance(content_digest, str):
        raise _snapshot_error("snapshot content_digest is missing", "content_digest")
    payload = {key: value for key, value in data.items() if key != "content_digest"}
    computed = digest_value(payload)
    if force_digest_mismatch or computed != content_digest:
        raise _snapshot_error(
            "snapshot content digest does not match its payload",
            "content_digest",
            {"expected": content_digest, "computed": computed, "injected": force_digest_mismatch},
        )

    declaration_data = payload.get("declaration")
    try:
        declaration = parse_declaration(declaration_data)
    except WorkbenchError as exc:
        raise _snapshot_error(
            "snapshot declaration is invalid",
            "declaration",
            {"cause": exc.to_dict()},
        ) from exc
    versions = payload.get("versions")
    if not isinstance(versions, dict):
        raise _snapshot_error("snapshot versions object is missing", "versions")
    expected_versions = declaration.versions()
    for field, expected in expected_versions.items():
        if versions.get(field) != expected:
            raise _snapshot_error(
                f"snapshot version field {field} does not match declaration",
                f"versions.{field}",
                {"expected": expected, "actual": versions.get(field)},
            )

    ledger_data = payload.get("ledger")
    if not isinstance(ledger_data, dict):
        raise _snapshot_error("snapshot ledger object is missing", "ledger")
    if ledger_data.get("format_version") != FORMAT_VERSION:
        raise _snapshot_error("snapshot ledger format_version is unsupported", "ledger.format_version")
    cursor = ledger_data.get("cursor")
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        raise _snapshot_error("snapshot ledger cursor is invalid", "ledger.cursor")
    head = ledger_data.get("head_digest")
    if not isinstance(head, str):
        raise _snapshot_error("snapshot ledger head_digest is missing", "ledger.head_digest")
    records_raw = ledger_data.get("event_records")
    if not isinstance(records_raw, list):
        raise _snapshot_error("snapshot event_records must be an array", "ledger.event_records")
    if len(records_raw) != cursor:
        raise _snapshot_error(
            "snapshot event_records length does not match cursor",
            "ledger.cursor",
            {"cursor": cursor, "records": len(records_raw)},
        )
    previous = EMPTY_LEDGER_DIGEST
    records: list[dict[str, Any]] = []
    for index, item in enumerate(records_raw, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("record"), dict):
            raise _snapshot_error("snapshot event record is malformed", f"ledger.event_records[{index}]")
        event_id = item.get("event_id")
        if event_id != index:
            raise _snapshot_error(
                "snapshot event ids are not continuous",
                f"ledger.event_records[{index}].event_id",
                {"expected": index, "actual": event_id},
            )
        raw = canonical_json(item["record"])
        record_digest = sha256_bytes(raw.encode("utf-8"))
        if item.get("record_digest") != record_digest:
            raise _snapshot_error(
                "snapshot event record digest is invalid",
                f"ledger.event_records[{index}].record_digest",
            )
        expected_chain = chain_digest(previous, record_digest)
        if item.get("chain_digest") != expected_chain:
            raise _snapshot_error(
                "snapshot event chain digest is invalid",
                f"ledger.event_records[{index}].chain_digest",
            )
        records.append(
            {
                "event_id": event_id,
                "record": item["record"],
                "record_digest": record_digest,
                "chain_digest": expected_chain,
            }
        )
        previous = expected_chain
    if previous != head:
        raise _snapshot_error(
            "snapshot ledger head digest does not match event records",
            "ledger.head_digest",
            {"expected": previous, "actual": head},
        )

    runtime_state_raw = payload.get("runtime_state")
    if not isinstance(runtime_state_raw, dict):
        raise _snapshot_error("snapshot runtime_state is missing", "runtime_state")
    try:
        runtime_state = RuntimeState.from_dict(runtime_state_raw, declaration)
    except ValueError as exc:
        raise _snapshot_error("snapshot runtime_state is invalid", "runtime_state") from exc
    expected_runtime = (
        records[-1]["record"].get("workload_state")
        if records
        else runtime_state.to_dict(declaration)
    )
    actual_runtime = runtime_state.to_dict(declaration)
    if expected_runtime != actual_runtime:
        raise _snapshot_error(
            "snapshot runtime_state differs from the last committed event",
            "runtime_state",
        )
    computed_matrix = build_matrix(declaration, runtime_state)
    final_matrix = payload.get("final_matrix")
    if not isinstance(final_matrix, dict) or canonical_json(final_matrix) != canonical_json(computed_matrix):
        raise _snapshot_error(
            "snapshot final_matrix differs from its runtime state",
            "final_matrix",
        )
    return Snapshot(
        path=source,
        declaration=declaration,
        versions=versions,
        records=records,
        ledger_cursor=cursor,
        ledger_head_digest=head,
        runtime_state=actual_runtime,
        final_matrix=computed_matrix,
        content_digest=content_digest,
        payload=payload,
    )


def compare_snapshot_to_declaration(
    snapshot: Snapshot, current: Declaration
) -> dict[str, Any]:
    current_versions = current.versions()
    version_fields = (
        "tenant_set_version",
        "workload_declaration_version",
        "quota_context_version",
        "cluster_total_units",
        "spec_version",
    )
    changed_fields = [
        field
        for field in version_fields
        if snapshot.versions.get(field) != current_versions.get(field)
    ]
    snapshot_tenants = {item.tenant_id: item.to_dict() for item in snapshot.declaration.tenants}
    current_tenants = {item.tenant_id: item.to_dict() for item in current.tenants}
    affected = sorted(
        tenant_id
        for tenant_id in set(snapshot_tenants) | set(current_tenants)
        if snapshot_tenants.get(tenant_id) != current_tenants.get(tenant_id)
    )
    return {
        "mode": "WHOLE_REPLACE",
        "changed_fields": changed_fields,
        "affected_tenant_ids": affected,
        "snapshot_versions": snapshot.versions,
        "current_versions": current_versions,
        "declaration_changed": bool(changed_fields or affected),
        "old_snapshot_policy": "IMMUTABLE_READ_ONLY",
        "snapshot_path": str(snapshot.path),
    }


def restore_snapshot(
    snapshot_path: Path | str,
    events_path: Path | str,
    db_path: Path | str,
    *,
    current_declaration_path: Path | str | None = None,
) -> RestoreResult:
    snapshot = load_snapshot(snapshot_path)
    current = (
        load_declaration(current_declaration_path)
        if current_declaration_path is not None
        else snapshot.declaration
    )
    change_report = compare_snapshot_to_declaration(snapshot, current)
    events = load_events(events_path, snapshot.declaration)
    if len(events) < snapshot.ledger_cursor:
        raise WorkbenchError(
            "E_CONTEXT",
            "event stream is shorter than the snapshot ledger cursor",
            object_type="event_stream",
            field="event_count",
            details={
                "unreproducible": True,
                "snapshot_cursor": snapshot.ledger_cursor,
                "event_count": len(events),
            },
        )
    for item in snapshot.records:
        event_id = int(item["event_id"])
        if item["record"].get("event") != events[event_id - 1].to_dict():
            raise WorkbenchError(
                "E_CONTEXT",
                "event stream cannot reproduce the snapshot event context",
                object_type="event",
                object_id=str(event_id),
                field="event",
                details={
                    "unreproducible": True,
                    "reason": "EVENT_STREAM_CHANGED",
                    "snapshot_event": item["record"].get("event"),
                    "supplied_event": events[event_id - 1].to_dict(),
                },
            )
    destination = Path(db_path)
    if destination.exists():
        raise _snapshot_error(
            "restore destination already exists; whole replacement needs a fresh database",
            "destination",
            {"path": str(destination)},
        )
    ledger = Ledger(destination)
    try:
        ledger.import_snapshot_events(
            snapshot.records,
            expected_cursor=snapshot.ledger_cursor,
            expected_head=snapshot.ledger_head_digest,
            versions=snapshot.versions,
        )
        state = RuntimeState.from_dict(snapshot.runtime_state, snapshot.declaration)
        session = WorkbenchSession(
            snapshot.declaration,
            events,
            ledger,
            state=state,
        )
        session.spec_path = None
        session.events_path = str(Path(events_path))
    except Exception:
        ledger.close()
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    return RestoreResult(session=session, snapshot=snapshot, change_report=change_report)


def snapshot_diff_against_current(
    snapshot_path: Path | str,
    current_declaration_path: Path | str,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path)
    current = load_declaration(current_declaration_path)
    return compare_snapshot_to_declaration(snapshot, current)
