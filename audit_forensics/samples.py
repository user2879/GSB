"""Deterministic local sample generation and sync-marker truth verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import make_sync_marker, normalize_timestamp
from .errors import AuditConfigError, business_error
from .storage import atomic_write_json, atomic_write_text


SAMPLE_SEED = 20261119
BASE_TS_MS = 1_795_046_400_000  # 2026-11-19T00:00:00Z


def _iso(milliseconds: int) -> str:
    from datetime import datetime, timezone

    value = datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _v1_event(
    event_id: str,
    trace_id: str,
    actor: str,
    object_name: str,
    action: str,
    ts: int,
    seq: int,
    parent_event_id: str | None = None,
    severity: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": event_id,
        "trace_id": trace_id,
        "actor": actor,
        "object": object_name,
        "action": action,
        "ts": ts,
        "seq": seq,
    }
    if parent_event_id is not None:
        event["parent_event_id"] = parent_event_id
    if severity is not None:
        event["severity"] = severity
    if source is not None:
        event["source"] = source
    return event


def _v2_event(
    event_id: str,
    trace_id: str,
    actor: str,
    object_name: str,
    action: str,
    ts: int | str,
    seq: int,
    parent_event_id: str | None = None,
    severity: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    return _v1_event(
        event_id,
        trace_id,
        actor,
        object_name,
        action,
        ts,  # type: ignore[arg-type]
        seq,
        parent_event_id,
        severity,
        source,
    )


def _rules() -> dict[str, Any]:
    return {
        "rules_version": "2026.11.1",
        "rules": [
            {
                "rule_id": "R-VAULT-ACCESS",
                "name": "Vault access correlation",
                "conditions": {
                    "actors": ["alice"],
                    "objects": ["vault"],
                    "time_window": {
                        "start": "2026-11-19T00:00:00Z",
                        "end": "2026-11-19T23:59:59Z",
                        "window_ms": 600_000,
                    },
                },
                "group_by": ["object"],
            },
            {
                "rule_id": "R-HIGH-SEVERITY",
                "name": "High severity by actor",
                "conditions": {
                    "severities": ["high", "critical"],
                    "time_window": {
                        "start_ms": BASE_TS_MS,
                        "end_ms": BASE_TS_MS + 86_400_000,
                        "window_ms": 300_000,
                    },
                },
                "group_by": ["actor"],
            },
            {
                "rule_id": "R-DB-OBJECT",
                "name": "Database object activity",
                "conditions": {"objects": ["db"]},
                "group_by": ["object"],
            },
        ],
    }


def _occurrence_record(
    event: dict[str, Any],
    *,
    source_format: str,
    source_file: str,
    position: str,
) -> dict[str, Any]:
    marker = make_sync_marker(
        event_id=event["event_id"],
        source_format=source_format,
        position=position,
        original_ts=event["ts"],
        source=Path(source_file).name,
    )
    return {
        "sync_marker": marker,
        "source_format": source_format,
        "source_file": source_file,
        "source_name": Path(source_file).name,
        "position": position,
        "event_id": event["event_id"],
        "original_ts": event["ts"],
    }


def generate_sample_dataset(
    output: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write a deterministic mixed-version sample and sync manifest."""

    root = Path(output)
    marker_file = root / "sync_manifest.json"
    if marker_file.exists() and not force:
        raise AuditConfigError(
            f"sample dataset already exists at {root}; pass force=True to replace"
        )
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    t100_003 = _v1_event(
        "evt-100-003",
        "T-100",
        "alice",
        "vault",
        "export",
        BASE_TS_MS + 3_000,
        3,
        "evt-100-002",
        "high",
        "endpoint-1",
    )
    t100_001 = _v1_event(
        "evt-100-001",
        "T-100",
        "alice",
        "vault",
        "login",
        BASE_TS_MS + 1_000,
        1,
        None,
        "low",
        "endpoint-1",
    )
    t100_002 = _v1_event(
        "evt-100-002",
        "T-100",
        "alice",
        "vault",
        "read",
        BASE_TS_MS + 2_000,
        2,
        "evt-100-001",
        None,
        "endpoint-1",
    )
    v1_primary = [t100_003, t100_001, t100_002]

    t600_001 = _v1_event(
        "evt-600-001",
        "T-600",
        "alice",
        "vault",
        "read",
        BASE_TS_MS + 9_000,
        1,
        None,
        "high",
        "endpoint-2",
    )
    conflict_v1 = _v1_event(
        "evt-700-001",
        "T-700",
        "frank",
        "host",
        "login",
        BASE_TS_MS + 11_000,
        1,
        None,
        "medium",
        "endpoint-1",
    )
    v1_conflict_events = [t600_001, conflict_v1]
    malformed_valid = _v1_event(
        "evt-800-001",
        "T-800",
        "grace",
        "archive",
        "read",
        BASE_TS_MS + 12_000,
        1,
        None,
        "info",
        "endpoint-3",
    )

    t100_002_duplicate = _v2_event(
        "evt-100-002",
        "T-100",
        "alice",
        "vault",
        "read",
        _iso(BASE_TS_MS + 2_000),
        2,
        "evt-100-001",
        "high",
        "collector-2",
    )
    t200_001 = _v2_event(
        "evt-200-001",
        "T-200",
        "bob",
        "db",
        "query",
        BASE_TS_MS + 4_000,
        1,
        None,
        "high",
        "collector-2",
    )
    t200_003 = _v2_event(
        "evt-200-003",
        "T-200",
        "bob",
        "db",
        "export",
        _iso(BASE_TS_MS + 6_000),
        3,
        "evt-200-001",
        "critical",
        "collector-2",
    )
    t300_001 = _v2_event(
        "evt-300-001",
        "T-300",
        "carol",
        "file",
        "read",
        BASE_TS_MS + 5_000,
        1,
        None,
        "low",
        "collector-3",
    )
    t300_002 = _v2_event(
        "evt-300-002",
        "T-300",
        "carol",
        "file",
        "copy",
        _iso(BASE_TS_MS + 5_500),
        2,
        "evt-200-001",
        "medium",
        "collector-3",
    )
    t400_001 = _v2_event(
        "evt-400-001",
        "T-400",
        "dave",
        "host",
        "scan",
        BASE_TS_MS + 7_000,
        1,
        "evt-missing",
        "medium",
        "collector-3",
    )
    t500_001 = _v2_event(
        "evt-500-001",
        "T-500",
        "erin",
        "mailbox",
        "delete",
        BASE_TS_MS + 8_000,
        1,
        "evt-500-001",
        "medium",
        "collector-3",
    )
    t600_002 = _v2_event(
        "evt-600-002",
        "T-600",
        "alice",
        "vault",
        "export",
        _iso(BASE_TS_MS + 10_000),
        2,
        "evt-600-001",
        "critical",
        "collector-2",
    )
    conflict_v2 = _v2_event(
        "evt-700-001",
        "T-700",
        "mallory",
        "host",
        "login",
        _iso(BASE_TS_MS + 11_000),
        1,
        None,
        "high",
        "collector-2",
    )
    bad_timestamp = _v2_event(
        "evt-bad-ts",
        "T-900",
        "heidi",
        "db",
        "query",
        "not-an-iso-time",
        1,
        None,
        "low",
        "collector-4",
    )
    v2_events = [
        t100_002_duplicate,
        t200_001,
        t200_003,
        t300_001,
        t300_002,
        t400_001,
        t500_001,
        t600_002,
        conflict_v2,
        bad_timestamp,
        {"event_id": "evt-missing-fields", "trace_id": "T-900"},
    ]

    v1_primary_path = input_dir / "v1_primary.jsonl"
    v1_conflict_path = input_dir / "v1_conflict.jsonl"
    v1_malformed_path = input_dir / "v1_malformed.jsonl"
    v2_path = input_dir / "v2_batch.json"
    rules_path = root / "rules.json"
    manifest_path = root / "sync_manifest.json"

    atomic_write_text(
        v1_primary_path,
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in v1_primary),
    )
    atomic_write_text(
        v1_conflict_path,
        "".join(
            json.dumps(event, separators=(",", ":")) + "\n"
            for event in v1_conflict_events
        ),
    )
    atomic_write_text(
        v1_malformed_path,
        '{"event_id":"broken-line"\n'
        + json.dumps(malformed_valid, separators=(",", ":"))
        + "\n",
    )
    atomic_write_json(v2_path, v2_events)
    atomic_write_json(rules_path, _rules())

    occurrences: list[dict[str, Any]] = []
    for index, event in enumerate(v1_primary, start=1):
        occurrences.append(
            _occurrence_record(
                event,
                source_format="v1",
                source_file=str(v1_primary_path),
                position=f"line:{index}",
            )
        )
    for index, event in enumerate(v1_conflict_events, start=1):
        occurrences.append(
            _occurrence_record(
                event,
                source_format="v1",
                source_file=str(v1_conflict_path),
                position=f"line:{index}",
            )
        )
    occurrences.append(
        _occurrence_record(
            malformed_valid,
            source_format="v1",
            source_file=str(v1_malformed_path),
            position="line:2",
        )
    )
    for index, event in enumerate(v2_events):
        if not isinstance(event, dict) or "ts" not in event:
            continue
        try:
            normalize_timestamp(event["ts"], allow_iso=True)
        except ValueError:
            continue
        occurrences.append(
            _occurrence_record(
                event,
                source_format="v2",
                source_file=str(v2_path),
                position=f"element:{index}",
            )
        )

    manifest = {
        "sync_protocol": "audit-sync.v1",
        "seed": SAMPLE_SEED,
        "base_ts_ms": BASE_TS_MS,
        "generated_by": "audit_forensics.samples.generate_sample_dataset",
        "occurrences": occurrences,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def verify_sync_manifest(
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Compare sample sync markers with normalized state.

    The verifier consumes the manifest's sender-side representation and the
    report's normalized event/provenance records. It never asks the generator
    for an expected normalized event.
    """

    errors: list[dict[str, Any]] = []
    events = {
        item.get("event", {}).get("event_id"): item
        for item in report.get("events", [])
        if isinstance(item, dict)
    }
    markers: dict[str, dict[str, Any]] = {}
    for item in report.get("events", []):
        for provenance in item.get("provenance", []):
            markers[provenance.get("sync_marker")] = provenance
    checked = 0
    for occurrence in manifest.get("occurrences", []):
        source_format = occurrence.get("source_format")
        source_name = occurrence.get("source_name", Path(occurrence.get("source_file", "")).name)
        expected_marker = make_sync_marker(
            event_id=occurrence["event_id"],
            source_format=source_format,
            position=occurrence["position"],
            original_ts=occurrence["original_ts"],
            source=source_name,
        )
        if expected_marker != occurrence.get("sync_marker"):
            errors.append(
                business_error(
                    "E_SYNC_MARKER",
                    "manifest sync marker does not match its sender-side fields",
                    source=source_name,
                    position=occurrence.get("position"),
                    event_id=occurrence.get("event_id"),
                )
            )
            continue
        provenance = markers.get(expected_marker)
        if provenance is None:
            errors.append(
                business_error(
                    "E_SYNC_MISSING",
                    "normalized state does not contain the synchronized occurrence",
                    source=source_name,
                    position=occurrence.get("position"),
                    event_id=occurrence.get("event_id"),
                )
            )
            continue
        try:
            expected_ts = normalize_timestamp(
                occurrence["original_ts"], allow_iso=(source_format == "v2")
            )
        except ValueError as exc:
            errors.append(
                business_error(
                    "E_SYNC_TIMESTAMP",
                    str(exc),
                    source=source_name,
                    position=occurrence.get("position"),
                    event_id=occurrence.get("event_id"),
                )
            )
            continue
        normalized_core = provenance.get("event_core") or {}
        if normalized_core.get("ts") != expected_ts:
            errors.append(
                business_error(
                    "E_SYNC_TIMESTAMP",
                    "normalized timestamp differs from sender-side time representation",
                    source=source_name,
                    position=occurrence.get("position"),
                    event_id=occurrence.get("event_id"),
                    details={
                        "expected_ts": expected_ts,
                        "actual_ts": normalized_core.get("ts"),
                    },
                )
            )
            continue
        event_state = events.get(occurrence["event_id"])
        if event_state is None:
            errors.append(
                business_error(
                    "E_SYNC_MISSING_EVENT",
                    "normalized event is absent",
                    event_id=occurrence.get("event_id"),
                )
            )
            continue
        if event_state.get("status") != "conflict":
            if normalized_core != event_state.get("event"):
                errors.append(
                    business_error(
                        "E_SYNC_CORE_MISMATCH",
                        "non-conflicting occurrence core differs from normalized event",
                        event_id=occurrence.get("event_id"),
                    )
                )
                continue
        checked += 1

    return {
        "ok": not errors,
        "checked": checked,
        "errors": errors,
    }
