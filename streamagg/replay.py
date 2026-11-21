from __future__ import annotations

import json
from typing import Any

from .engine import Engine
from .errors import BusinessError
from .generator import generate_session
from .ledger import Ledger


WINDOW_FIELDS = (
    ("status", "status"),
    ("count", "count"),
    ("sum_fixed", "sum_fixed"),
    ("mean_fixed", "mean_fixed"),
    ("late_accepted_count", "late_accepted_count"),
    ("late_abs_amount", "late_accepted_abs_amount"),
    ("boundary_abs_amount", "boundary_abs_amount"),
)

ATTRIBUTION_FIELDS = (
    "quantization_truncation",
    "late_admission",
    "window_boundary",
    "sum_bound",
    "mean_bound",
    "single_value_quantization",
    "dominant_segment",
    "dominant_share_bp",
    "threshold_bp",
    "quantization_share_bp",
    "late_share_bp",
    "boundary_share_bp",
    "formula_version",
)

DECISION_FIELDS = (
    "decision",
    "reason",
    "fixed_value",
    "budget_count_before",
    "budget_count_after",
    "budget_amount_before",
    "budget_amount_after",
    "cumulative_late_abs_amount",
    "was_late",
)


def _window_id_from_key(key: Any) -> tuple[str, int, int, str, int]:
    if hasattr(key, "rule_id"):
        return (
            key.partition_key,
            key.window_start_ms,
            key.window_end_ms,
            key.rule_id,
            key.rule_version,
        )
    return (
        key["partition_key"],
        key["window_start_ms"],
        key["window_end_ms"],
        key["rule_id"],
        key["rule_version"],
    )


def _decision_id(decision: Any) -> tuple[str, int, int, str, int, str]:
    if hasattr(decision, "rule_id"):
        return (
            decision.partition_key,
            decision.window_start_ms,
            decision.window_end_ms,
            decision.rule_id,
            decision.rule_version,
            decision.event_id,
        )
    return (
        decision["partition_key"],
        decision["window_start_ms"],
        decision["window_end_ms"],
        decision["rule_id"],
        decision["rule_version"],
        decision["event_id"],
    )


def _value(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    try:
        return obj[field]
    except (KeyError, IndexError, TypeError):
        return None


def validate_replay(
    engine: Engine,
    ledger: Ledger,
    *,
    session_id: str | None = None,
    view_id: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    session = session_id or engine.session_id
    runtime = ledger.get_runtime(session)
    view = view_id or runtime["current_view_id"]
    ledger.verify_content_digest(session, view)
    manifest = ledger.get_view_manifest(session, view)
    manifest_checks = (
        ("rules_version", engine.spec.rules_version, manifest["rules_version"]),
        ("fixed_scale", engine.spec.fixed_scale, manifest["fixed_scale"]),
        ("spec_digest", engine.spec.spec_digest, manifest["spec_digest"]),
    )
    for field, actual_value, expected_value in manifest_checks:
        if actual_value == expected_value:
            continue
        mismatch = {
            "kind": "manifest",
            "partition_key": None,
            "window_start_ms": None,
            "window_end_ms": None,
            "field": field,
            "expected": expected_value,
            "actual": actual_value,
        }
        if strict:
            raise BusinessError(
                code="E_REPLAY",
                message="replay fixed-point or rule manifest does not match the ledger",
                field=field,
                details={"first_mismatch": mismatch},
            )
        return {"consistent": False, "first_mismatch": mismatch}

    events, _ = generate_session(engine.spec)
    events.sort(key=lambda event: (event.arrival_no, event.event_id))
    cursor = int(runtime["event_cursor"])
    if cursor < 0 or cursor > len(events):
        raise BusinessError(
            code="E_REPLAY",
            message="ledger event cursor is outside the generated event stream",
            field="event_cursor",
            details={"event_cursor": cursor, "event_count": len(events)},
        )
    replay_engine = Engine(
        engine.spec,
        session_id=f"replay-{session}",
        ledger=None,
        view_id=f"replay-{view}",
    )
    replay_engine.run_events(events[:cursor])

    mismatches: list[dict[str, Any]] = []

    ledger_watermarks = ledger.final_watermarks(session, view)
    for partition_key in sorted(set(ledger_watermarks) | set(replay_engine.watermarks)):
        expected = ledger_watermarks.get(partition_key)
        actual = replay_engine.watermarks.get(partition_key)
        if expected != actual:
            mismatches.append(
                {
                    "kind": "watermark",
                    "partition_key": partition_key,
                    "window_start_ms": None,
                    "window_end_ms": None,
                    "field": "watermark",
                    "expected": expected,
                    "actual": actual,
                }
            )

    ledger_windows = {
        _window_id_from_key(row): row for row in ledger.current_windows(session, view)
    }
    replay_windows = {
        _window_id_from_key(result.key): result.to_dict()
        for result in replay_engine.finalized_windows.values()
    }
    ledger_attributions = {
        _window_id_from_key(row): row for row in ledger.current_attributions(session, view)
    }
    for window_id in sorted(set(ledger_windows) | set(replay_windows)):
        expected_row = ledger_windows.get(window_id)
        actual_row = replay_windows.get(window_id)
        if expected_row is None or actual_row is None:
            mismatches.append(
                {
                    "kind": "window",
                    "partition_key": window_id[0],
                    "window_start_ms": window_id[1],
                    "window_end_ms": window_id[2],
                    "field": "window_presence",
                    "expected": expected_row is not None,
                    "actual": actual_row is not None,
                }
            )
            continue
        for expected_field, actual_field in WINDOW_FIELDS:
            expected = expected_row[expected_field]
            actual = actual_row.get(actual_field)
            if expected != actual:
                mismatches.append(
                    {
                        "kind": "window",
                        "partition_key": window_id[0],
                        "window_start_ms": window_id[1],
                        "window_end_ms": window_id[2],
                        "field": f"window.{expected_field}",
                        "expected": expected,
                        "actual": actual,
                    }
                )
        expected_attr = ledger_attributions.get(window_id)
        actual_attr = actual_row.get("attribution", {})
        if expected_attr is None:
            mismatches.append(
                {
                    "kind": "attribution",
                    "partition_key": window_id[0],
                    "window_start_ms": window_id[1],
                    "window_end_ms": window_id[2],
                    "field": "attribution_presence",
                    "expected": False,
                    "actual": True,
                }
            )
        else:
            for field in ATTRIBUTION_FIELDS:
                expected = expected_attr[field]
                actual = actual_attr.get(field)
                if expected != actual:
                    mismatches.append(
                        {
                            "kind": "attribution",
                            "partition_key": window_id[0],
                            "window_start_ms": window_id[1],
                            "window_end_ms": window_id[2],
                            "field": f"attribution.{field}",
                            "expected": expected,
                            "actual": actual,
                        }
                    )

    ledger_decisions = {
        _decision_id(row): row for row in ledger.current_decisions(session, view)
    }
    replay_decisions = {
        _decision_id(decision): decision.to_dict() for decision in replay_engine.decisions
    }
    for decision_key in sorted(set(ledger_decisions) | set(replay_decisions)):
        expected_row = ledger_decisions.get(decision_key)
        actual_row = replay_decisions.get(decision_key)
        if expected_row is None or actual_row is None:
            mismatches.append(
                {
                    "kind": "decision",
                    "partition_key": decision_key[0],
                    "window_start_ms": decision_key[1],
                    "window_end_ms": decision_key[2],
                    "field": "decision_presence",
                    "expected": expected_row is not None,
                    "actual": actual_row is not None,
                }
            )
            continue
        for field in DECISION_FIELDS:
            expected = expected_row[field]
            actual = actual_row.get(field)
            if field == "was_late":
                expected = bool(expected)
                actual = bool(actual)
            if expected != actual:
                mismatches.append(
                    {
                        "kind": "decision",
                        "partition_key": decision_key[0],
                        "window_start_ms": decision_key[1],
                        "window_end_ms": decision_key[2],
                        "field": f"decision.{field}",
                        "expected": expected,
                        "actual": actual,
                    }
                )

    ledger_pairs = {
        json.dumps((row["key_a_json"], row["key_b_json"]), sort_keys=True)
        for row in ledger.current_ambiguities(session, view)
    }
    replay_pairs = {
        json.dumps(
            (
                json.dumps(pair["key_a"], sort_keys=True, separators=(",", ":")),
                json.dumps(pair["key_b"], sort_keys=True, separators=(",", ":")),
            ),
            sort_keys=True,
        )
        for pair in replay_engine.ambiguity_report()
    }
    if ledger_pairs != replay_pairs:
        mismatches.append(
            {
                "kind": "ambiguity",
                "partition_key": None,
                "window_start_ms": None,
                "window_end_ms": None,
                "field": "ambiguity_pairs",
                "expected": sorted(ledger_pairs),
                "actual": sorted(replay_pairs),
            }
        )

    mismatches.sort(
        key=lambda item: (
            item["partition_key"] or "",
            item["window_start_ms"] if item["window_start_ms"] is not None else -(1 << 63),
            item["window_end_ms"] if item["window_end_ms"] is not None else -(1 << 63),
            item["field"],
            item["kind"],
        )
    )
    report = {
        "consistent": not mismatches,
        "session_id": session,
        "view_id": view,
        "event_cursor": cursor,
        "rules_version": manifest["rules_version"],
        "fixed_scale": manifest["fixed_scale"],
        "compared_watermarks": len(set(ledger_watermarks) | set(replay_engine.watermarks)),
        "compared_windows": len(set(ledger_windows) | set(replay_windows)),
        "compared_decisions": len(set(ledger_decisions) | set(replay_decisions)),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }
    if mismatches and strict:
        first = mismatches[0]
        raise BusinessError(
            code="E_REPLAY",
            message="replay diverged from the ledger",
            field=first["field"],
            details={
                "partition_key": first["partition_key"],
                "window_start_ms": first["window_start_ms"],
                "window_end_ms": first["window_end_ms"],
                "field": first["field"],
                "expected": first["expected"],
                "actual": first["actual"],
                "mismatch_count": len(mismatches),
                "first_mismatch": first,
            },
        )
    return report
