"""Post-recovery deterministic replay and first-divergence localization."""

from __future__ import annotations

import json
from typing import Any

from .canonical import canonical_json
from .engine import run_engine
from .errors import ReplayError
from .ledger import LedgerReader
from .spec import LEDGER_SPEC_VERSION, SessionBundle


def _mismatch(
    *,
    field: str,
    expected: Any,
    actual: Any,
    evidence: str,
    partition_label: str | None = None,
    window_id: str | None = None,
    window_boundary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "partition_label": partition_label,
        "window_id": window_id,
        "window_boundary": window_boundary,
        "field": field,
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
    }


def _raise_first(first: dict[str, Any], *, checked: int) -> None:
    raise ReplayError(
        "replay diverged from the committed ledger; rerun the located window",
        details={"first_mismatch": first, "checked_items": checked},
    )


def _window_field_value(window: dict[str, Any], field: str) -> Any:
    if field in window:
        return window[field]
    content = window.get("content") or {}
    if field in content:
        return content[field]
    return None


def replay_and_compare(bundle: SessionBundle, ledger_path: str) -> dict[str, Any]:
    reader = LedgerReader(ledger_path)
    meta = reader.meta()
    if not meta:
        raise ReplayError(
            "ledger has no session metadata",
            details={"ledger": ledger_path, "first_mismatch": {"field": "meta"}},
        )
    status = meta.get("run_status")
    if status != "complete":
        raise ReplayError(
            "ledger does not contain one complete committed run; rerun is required",
            details={
                "ledger": ledger_path,
                "run_status": status,
                "required_action": "rerun",
            },
        )
    stored_windows = reader.windows(include_unknown=False)
    unknown = [item for item in reader.windows() if item["status"] == "unknown"]
    if unknown:
        raise ReplayError(
            "ledger contains unknown windows; replay cannot claim a complete exit report",
            details={
                "ledger": ledger_path,
                "unknown_windows": sorted(item["window_id"] for item in unknown),
                "required_action": "rerun",
            },
        )
    checked = 0
    if int(meta.get("format_version", -1)) != LEDGER_SPEC_VERSION:
        _raise_first(
            _mismatch(
                field="meta.format_version",
                expected=LEDGER_SPEC_VERSION,
                actual=meta.get("format_version"),
                evidence="current decoder requires integer ledger format version 1",
            ),
            checked=checked,
        )
    checked += 1
    if str(meta.get("rules_version")) != bundle.rules_version:
        _raise_first(
            _mismatch(
                field="meta.rules_version",
                expected=bundle.rules_version,
                actual=meta.get("rules_version"),
                evidence="ledger replay uses its recorded rule version",
            ),
            checked=checked,
        )
    checked += 1
    if int(meta.get("scale")) != bundle.scale:
        _raise_first(
            _mismatch(
                field="meta.scale",
                expected=bundle.scale,
                actual=meta.get("scale"),
                evidence="ledger replay uses its recorded fixed-point scale",
            ),
            checked=checked,
        )
    checked += 1
    if canonical_json(meta.get("strategy")) != canonical_json(bundle.strategy.to_dict()):
        _raise_first(
            _mismatch(
                field="meta.strategy",
                expected=bundle.strategy.to_dict(),
                actual=meta.get("strategy"),
                evidence="ledger replay uses its recorded watermark strategy",
            ),
            checked=checked,
        )
    checked += 1
    if str(meta.get("spec_digest")) != bundle.spec_digest:
        _raise_first(
            _mismatch(
                field="meta.spec_digest",
                expected=bundle.spec_digest,
                actual=meta.get("spec_digest"),
                evidence="loaded source/rules/budget digest differs from the ledger header",
            ),
            checked=checked,
        )
    checked += 1
    cursor = int(meta.get("event_cursor", -1))
    events_generated = int(meta.get("events_generated", -1))
    if cursor != events_generated - 1:
        _raise_first(
            _mismatch(
                field="meta.event_cursor",
                expected=events_generated - 1,
                actual=cursor,
                evidence="a complete run must have consumed every generated event",
            ),
            checked=checked,
        )
    checked += 1

    replay = run_engine(bundle)
    checked += len(replay.watermarks)
    expected_points = reader.watermark_points()
    actual_points = [
        {
            "event_seq": point.event_seq,
            "event_id": point.event_id,
            "phase": point.phase,
            "partition_label": point.partition_label,
            "partition": json.loads(point.partition_json),
            "partition_watermark": point.partition_watermark,
            "global_watermark": point.global_watermark,
            "strategy": point.strategy,
        }
        for point in replay.watermarks
    ]
    if len(expected_points) != len(actual_points):
        _raise_first(
            _mismatch(
                field="watermark_trace.length",
                expected=len(expected_points),
                actual=len(actual_points),
                evidence="watermark trace cardinality differs after replay",
            ),
            checked=checked,
        )
    for index, (expected, actual) in enumerate(zip(expected_points, actual_points)):
        for field in (
            "event_seq",
            "event_id",
            "phase",
            "partition_label",
            "partition_watermark",
            "global_watermark",
            "strategy",
        ):
            checked += 1
            if expected.get(field) != actual.get(field):
                _raise_first(
                    _mismatch(
                        field=f"watermark.{field}",
                        expected=expected.get(field),
                        actual=actual.get(field),
                        evidence=f"first divergent watermark trace row at index {index}",
                        partition_label=expected.get("partition_label"),
                    ),
                    checked=checked,
                )
        if expected.get("partition") != actual.get("partition"):
            _raise_first(
                _mismatch(
                    field="watermark.partition",
                    expected=expected.get("partition"),
                    actual=actual.get("partition"),
                    evidence=f"partition identity differs at watermark trace index {index}",
                    partition_label=expected.get("partition_label"),
                ),
                checked=checked,
            )

    expected_by_id = {item["window_id"]: item for item in stored_windows}
    actual_by_id = {item["window_id"]: item for item in replay.windows}
    checked += len(expected_by_id)
    for window_id in sorted(set(expected_by_id) | set(actual_by_id)):
        expected = expected_by_id.get(window_id)
        actual = actual_by_id.get(window_id)
        if expected is None or actual is None:
            anchor = expected or actual or {}
            _raise_first(
                _mismatch(
                    field="window.presence",
                    expected=expected is not None,
                    actual=actual is not None,
                    evidence="finalized window set differs after replay",
                    partition_label=anchor.get("partition_label"),
                    window_id=window_id,
                    window_boundary={
                        "start_ms": anchor.get("start_ms"),
                        "end_ms": anchor.get("end_ms"),
                    },
                ),
                checked=checked,
            )
        fields = (
            "format_version",
            "rules_version",
            "scale",
            "rule_id",
            "partition",
            "start_ms",
            "nominal_end_ms",
            "end_ms",
            "parent_window_id",
            "finalization_order",
            "trigger_seq",
            "trigger_event_id",
            "strategy",
            "version",
            "status",
            "digest",
            "progress_cursor",
            "stats",
            "late_decisions",
            "errors",
            "displacement_trajectory",
        )
        for field in fields:
            checked += 1
            expected_value = _window_field_value(expected, field)
            actual_value = _window_field_value(actual, field)
            if expected_value != actual_value:
                _raise_first(
                    _mismatch(
                        field=f"window.{field}",
                        expected=expected_value,
                        actual=actual_value,
                        evidence="first divergent field in deterministic window replay",
                        partition_label=expected.get("partition_label"),
                        window_id=window_id,
                        window_boundary={
                           "start_ms": expected.get("start_ms"),
                            "end_ms": expected.get("end_ms"),
                        },
                    ),
                    checked=checked,
                )
    expected_order = [
        item["window_id"]
        for item in sorted(
            stored_windows,
            key=lambda item: (int(item["finalization_order"]), item["window_id"]),
        )
    ]
    actual_order = [
        item["window_id"]
        for item in sorted(
            replay.windows,
            key=lambda item: (int(item["finalization_order"]), item["window_id"]),
        )
    ]
    if expected_order != actual_order:
        for index, (expected_id, actual_id) in enumerate(
            zip(expected_order, actual_order)
        ):
            if expected_id != actual_id:
                anchor = expected_by_id[expected_id]
                _raise_first(
                    _mismatch(
                        field="window.finalization_order",
                        expected=expected_order[index : index + 2],
                        actual=actual_order[index : index + 2],
                        evidence=f"stable finalization order first diverges at index {index}",
                        partition_label=anchor.get("partition_label"),
                        window_id=expected_id,
                        window_boundary={
                            "start_ms": anchor.get("start_ms"),
                            "end_ms": anchor.get("end_ms"),
                        },
                    ),
                    checked=checked + index,
                )
        _raise_first(
            _mismatch(
                field="window.finalization_order.length",
                expected=len(expected_order),
                actual=len(actual_order),
                evidence="finalization order length differs",
            ),
            checked=checked,
        )
    return {
        "ledger": ledger_path,
        "session_id": bundle.session_id,
        "strategy": bundle.strategy.kind,
        "scale": bundle.scale,
        "rules_version": bundle.rules_version,
        "event_cursor": replay.event_cursor,
        "windows_checked": len(expected_by_id),
        "watermark_rows_checked": len(expected_points),
        "checks_performed": checked,
        "result": "MATCH",
    }
