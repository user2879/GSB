"""Snapshot persistence, content verification, and whole-set restoration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import validate_peer_script, validate_session_config
from .errors import E_REPLAY, E_VERSION, BusinessError, version_error
from .models import (
    Event,
    FieldChange,
    ItemResult,
    ReplacementReport,
    canonical_json,
    flatten_values,
)
from .projection import ITEM_ORDER, compare_items, replay, state_snapshot

ITEM_COMPARE_FIELDS = (
    "value",
    "revision",
    "basis_event_seq",
    "depends_on_items",
    "depends_on_events",
    "config_version",
    "script_version",
    "stale",
    "last_affected_event_seq",
)


def _item_from_dict(value: dict[str, Any]) -> ItemResult:
    return ItemResult(
        key=str(value["key"]),
        value=value["value"],
        revision=int(value["revision"]),
        basis_event_seq=int(value["basis_event_seq"]),
        depends_on_items=tuple(value.get("depends_on_items", [])),
        depends_on_events=tuple(
            int(item) for item in value.get("depends_on_events", [])
        ),
        config_version=int(value["config_version"]),
        script_version=int(value["script_version"]),
        stale=bool(value.get("stale", False)),
        last_affected_event_seq=int(value.get("last_affected_event_seq", 0)),
    )


def _replay_error(
    event_seq: int, field: str, expected: Any, actual: Any, reason: str
) -> BusinessError:
    return BusinessError(
        E_REPLAY,
        f"snapshot replay mismatch at event {event_seq}, field {field}: {reason}",
        [f"event:{event_seq}", field],
        {
            "first_event_seq": event_seq,
            "field": field,
            "expected": expected,
            "actual": actual,
            "basis": "snapshot replay_points versus deterministic replay",
        },
    )


def load_and_verify_snapshot(row: Any, ledger: Any) -> tuple[dict[str, Any], Any]:
    if row["config_version"] is None or row["script_version"] is None:
        raise version_error(
            ["snapshot.config_version", "snapshot.script_version"],
            "snapshot version is missing",
            snapshot_id=int(row["snapshot_id"]),
        )
    if row["config_digest"] is None or row["script_digest"] is None:
        raise version_error(
            ["snapshot.config_digest", "snapshot.script_digest"],
            "snapshot content digest is missing",
            snapshot_id=int(row["snapshot_id"]),
        )
    payload = ledger.snapshot_payload(row)

    config, config_errors = validate_session_config(
        payload["config"], source_path="<snapshot>"
    )
    if config_errors:
        raise BusinessError(E_VERSION, "snapshot config is invalid", config_errors)
    script, script_errors = validate_peer_script(
        payload["script"], config, source_path="<snapshot>"
    )
    if script_errors:
        raise BusinessError(E_VERSION, "snapshot peer script is invalid", script_errors)
    if (
        config.digest != payload["config_digest"]
        or config.config_version != payload["config_version"]
    ):
        raise version_error(
            ["snapshot.config_version", "snapshot.config_digest"],
            "snapshot config version/content digest mismatch",
            snapshot_id=int(row["snapshot_id"]),
        )
    if (
        script.digest != payload["script_digest"]
        or script.script_version != payload["script_version"]
    ):
        raise version_error(
            ["snapshot.script_version", "snapshot.script_digest"],
            "snapshot script version/content digest mismatch",
            snapshot_id=int(row["snapshot_id"]),
        )

    events = tuple(Event.from_dict(item) for item in payload["events"])
    if len(events) != int(payload["event_count"]):
        raise BusinessError(
            E_REPLAY,
            "snapshot event_count does not match events",
            ["snapshot.event_count"],
        )
    recomputed = replay(events, config, script, capture_checkpoints=True)

    stored_points = payload["replay_points"]
    actual_points = recomputed.checkpoints
    for index in range(max(len(stored_points), len(actual_points))):
        if index >= len(stored_points):
            point = actual_points[index]
            raise _replay_error(
                int(point["event_seq"]),
                "replay_point.missing",
                None,
                point["event_seq"],
                "missing replay point",
            )
        if index >= len(actual_points):
            point = stored_points[index]
            raise _replay_error(
                int(point["event_seq"]),
                "replay_point.extra",
                point["event_seq"],
                None,
                "extra replay point",
            )
        stored = stored_points[index]
        actual = actual_points[index]
        event_seq = int(stored.get("event_seq", 0))
        if int(stored.get("event_seq", -1)) != int(actual.get("event_seq", -2)):
            raise _replay_error(
                event_seq,
                "event_seq",
                stored.get("event_seq"),
                actual.get("event_seq"),
                "event sequence shifted",
            )
        stored_items = stored.get("items", {})
        actual_items = actual.get("items", {})
        for key in sorted(set(stored_items) | set(actual_items)):
            left = stored_items.get(key)
            right = actual_items.get(key)
            if left is None or right is None:
                raise _replay_error(
                    event_seq, f"{key}.presence", left, right, "item presence differs"
                )
            for field in ITEM_COMPARE_FIELDS:
                if canonical_json(left.get(field)) != canonical_json(right.get(field)):
                    raise _replay_error(
                        event_seq,
                        f"{key}.{field}",
                        left.get(field),
                        right.get(field),
                        "conclusion differs",
                    )

    expected_items = {
        key: _item_from_dict(value) for key, value in payload["conclusions"].items()
    }
    differences = compare_items(expected_items, recomputed.items)
    if differences:
        first = differences[0]
        item_key = first["field"].split(".", 1)[0]
        expected = expected_items.get(item_key)
        event_seq = 0 if expected is None else expected.basis_event_seq
        raise _replay_error(
            event_seq,
            first["field"],
            first["expected"],
            first["actual"],
            "final conclusion differs",
        )

    if canonical_json(payload["state"]) != canonical_json(
        state_snapshot(recomputed.state)
    ):
        raise _replay_error(
            recomputed.events[-1].event_seq if recomputed.events else 0,
            "state",
            payload["state"],
            state_snapshot(recomputed.state),
            "final state differs",
        )
    return payload, recomputed


def verify_snapshot_row(row: Any, ledger: Any) -> dict[str, Any]:
    payload, recomputed = load_and_verify_snapshot(row, ledger)
    return {
        "snapshot_id": int(row["snapshot_id"]),
        "name": payload["name"],
        "event_count": len(recomputed.events),
        "config_version": payload["config_version"],
        "script_version": payload["script_version"],
        "content_digest": row["content_digest"],
        "verified": True,
    }


def restore_snapshot_row(service: Any, row: Any) -> ReplacementReport:
    payload, recomputed = load_and_verify_snapshot(row, service.ledger)
    config = recomputed.state.config
    script = recomputed.state.script
    snapshot_items = {
        key: _item_from_dict(value) for key, value in payload["conclusions"].items()
    }
    old_items = dict(service.items)
    old_config_version = service.config.config_version
    old_script_version = service.script.script_version

    current_matches = (
        service.config.config_version == config.config_version
        and service.script.script_version == script.script_version
        and service.config.digest == config.digest
        and service.script.digest == script.digest
    )
    reason = (
        "snapshot_restore_same_version"
        if current_matches
        else "snapshot_version_content_replacement"
    )
    last_event_seq = recomputed.events[-1].event_seq if recomputed.events else 0

    # Register a historical snapshot's version if this database has not seen it.
    for kind, digest, version in (
        ("config", config.digest, config.config_version),
        ("script", script.digest, script.script_version),
    ):
        record = service.ledger.version_record(kind, digest)
        if record is not None:
            if int(record["version"]) != version:
                raise version_error(
                    [f"snapshot.{kind}_version"],
                    "snapshot version conflicts with version history",
                )
        else:
            by_number = service.ledger.version_by_number(kind, version)
            if by_number is not None and by_number["digest"] != digest:
                raise version_error(
                    [f"snapshot.{kind}_version"],
                    "snapshot version number is already used by other content",
                )
            service.ledger.register_version(kind, digest, version, service.clock_fn())

    service.ledger.replace_session(
        config,
        script,
        service.clock_fn(),
        current_event_seq=last_event_seq,
    )
    service.ledger.replace_events(
        recomputed.events,
        config_version=config.config_version,
        script_version=script.script_version,
    )
    archived = service.ledger.replace_conclusions(
        snapshot_items,
        reason=reason,
        mono_ms=service.clock_fn(),
        archive=True,
    )

    changes = _value_changes(old_items, snapshot_items)
    report = ReplacementReport(
        reason=reason,
        affected_fields=tuple(changes),
        old_config_version=old_config_version,
        old_script_version=old_script_version,
        new_config_version=config.config_version,
        new_script_version=script.script_version,
        archived_fields=archived,
    )
    service.config = config
    service.script = script
    service.state = recomputed.state
    service.events = recomputed.events
    service.items = snapshot_items
    service.needs_rerun = False
    service.recovery_report = None
    service.logger.info(
        "snapshot_restored",
        snapshot_id=int(row["snapshot_id"]),
        reason=reason,
        old_config_version=old_config_version,
        old_script_version=old_script_version,
        new_config_version=config.config_version,
        new_script_version=script.script_version,
        affected_fields=[change.field for change in changes],
    )
    return report


def seed_snapshot(
    ledger: Any, seed: dict[str, Any], *, mono_ms: int, base_dir: str | Path = "."
) -> int:
    base = Path(base_dir)
    config_raw = seed.get("config")
    if config_raw is None:
        config_raw = json.loads(
            (base / seed["config_file"]).read_text(encoding="utf-8")
        )
    config, config_errors = validate_session_config(
        config_raw, source_path="<snapshot-seed>"
    )
    if config_errors:
        raise BusinessError(E_VERSION, "snapshot seed config is invalid", config_errors)
    script_raw = seed.get("script")
    if script_raw is None:
        script_raw = json.loads(
            (base / seed["script_file"]).read_text(encoding="utf-8")
        )
    script, script_errors = validate_peer_script(
        script_raw, config, source_path="<snapshot-seed>"
    )
    if script_errors:
        raise BusinessError(
            E_VERSION, "snapshot seed peer script is invalid", script_errors
        )
    events_raw = seed.get("events")
    if events_raw is None:
        events_raw = json.loads(
            (base / seed["events_file"]).read_text(encoding="utf-8")
        )["events"]
    events = tuple(Event.from_dict(item) for item in events_raw)
    recomputed = replay(events, config, script, capture_checkpoints=True)
    for kind, digest, version in (
        ("config", config.digest, config.config_version),
        ("script", script.digest, script.script_version),
    ):
        existing = ledger.version_record(kind, digest)
        if existing is None:
            conflict = ledger.version_by_number(kind, version)
            if conflict is not None:
                raise version_error(
                    ["seed.version"],
                    "snapshot seed version conflicts",
                    kind=kind,
                    version=version,
                )
            ledger.register_version(kind, digest, version, mono_ms)
        elif int(existing["version"]) != version:
            raise version_error(
                ["seed.version"],
                "snapshot seed version disagrees with history",
                kind=kind,
            )
    payload = {
        "name": seed.get(
            "name", f"snapshot-v{config.config_version}-{script.script_version}"
        ),
        "created_mono_ms": int(seed.get("created_mono_ms", mono_ms)),
        "config_version": config.config_version,
        "script_version": script.script_version,
        "config_digest": config.digest,
        "script_digest": script.digest,
        "event_count": len(recomputed.events),
        "state": state_snapshot(recomputed.state),
        "conclusions": {key: recomputed.items[key].as_dict() for key in ITEM_ORDER},
        "events": [event.as_dict() for event in recomputed.events],
        "replay_points": list(recomputed.checkpoints),
        "config": config.raw,
        "script": script.raw,
    }
    return ledger.save_snapshot(payload)


def _value_changes(
    old_items: dict[str, ItemResult], new_items: dict[str, ItemResult]
) -> list[FieldChange]:
    old_flat: dict[str, Any] = {}
    new_flat: dict[str, Any] = {}
    for key, item in old_items.items():
        for path, value in flatten_values(f"items.{key}", item.value):
            old_flat[path] = value
        old_flat[f"items.{key}.revision"] = item.revision
    for key, item in new_items.items():
        for path, value in flatten_values(f"items.{key}", item.value):
            new_flat[path] = value
        new_flat[f"items.{key}.revision"] = item.revision
    changes: list[FieldChange] = []
    for field in sorted(set(old_flat) | set(new_flat)):
        old_value = old_flat.get(field)
        new_value = new_flat.get(field)
        if canonical_json(old_value) != canonical_json(new_value):
            changes.append(FieldChange(field, old_value, new_value))
    return changes
