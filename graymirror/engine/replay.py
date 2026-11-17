"""Deterministic replay verification after snapshot recovery."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from .errors import E_REPLAY, GrayError
from .session import RuleView

# Fields that must match for every individual request.
DECISION_FIELDS = (
    "rule_id", "bucket", "gray_fixed_point", "scale_factor", "target",
    "channel", "basis", "attempts", "status",
)
MIRROR_FIELDS = (
    "rule_id", "bucket", "mirror_fixed_point", "scale_factor", "target",
    "timeout_ms", "status", "basis",
)


def _normalized(value: Any) -> Any:
    """Collapse in memory Fraction values to their exact serialized string form."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def verify_replay(view: RuleView, snapshot: dict[str, Any]) -> dict[str, Any]:
    _compare_sequences(
        snapshot.get("decisions", []), view.decisions,
        key_field="request_id", fields=DECISION_FIELDS, label="route_decision")
    _compare_sequences(
        snapshot.get("mirrors", []), view.mirrors,
        key_field="request_id", fields=MIRROR_FIELDS, label="mirror_record")
    _compare_aggregates(snapshot, view)
    return {
        "status": "MATCH",
        "checked_decisions": len(view.decisions),
        "checked_mirrors": len(view.mirrors),
        "rule_version": view.version,
        "scale_exponent": view.scale_exponent,
    }


def _compare_sequences(snapshot_rows: list[dict[str, Any]],
                       replay_rows: list[dict[str, Any]],
                       key_field: str, fields: tuple[str, ...],
                       label: str) -> None:
    snap_by_key = {row[key_field]: row for row in snapshot_rows}
    replay_by_key = {row[key_field]: row for row in replay_rows}
    if set(snap_by_key) != set(replay_by_key):
        only_snapshot = sorted(set(snap_by_key) - set(replay_by_key))
        only_replay = sorted(set(replay_by_key) - set(snap_by_key))
        first = (only_snapshot or only_replay)[0]
        raise GrayError(
            E_REPLAY, f"{label} membership mismatch", field_path=key_field,
            details={"first_request_id": first,
                     "only_in_snapshot": only_snapshot[:5],
                     "only_in_replay": only_replay[:5]})
    for row in snapshot_rows:  # snapshot order defines the first mismatch
        key = row[key_field]
        replay_row = replay_by_key[key]
        for field_name in fields:
            expected = _normalized(row.get(field_name))
            actual = _normalized(replay_row.get(field_name))
            if expected != actual:
                raise GrayError(
                    E_REPLAY,
                    f"{label} field diverged at first inconsistent request",
                    field_path=f"{key}.{field_name}",
                    details={"first_request_id": key, "field": field_name,
                             "snapshot": expected, "replay": actual})


def _compare_aggregates(snapshot: dict[str, Any], view: RuleView) -> None:
    snapshot_stats = _normalized(snapshot.get("statistics"))
    replay_stats = _normalized(view.statistics)
    if snapshot_stats != replay_stats:
        raise GrayError(E_REPLAY, "aggregate statistics diverged on replay",
                        field_path="statistics",
                        details={"snapshot": snapshot_stats,
                                 "replay": replay_stats})
    snapshot_rules = _normalized(snapshot.get("quantized_rules"))
    replay_rules = _normalized(view.quantized_report)
    if snapshot_rules != replay_rules:
        raise GrayError(E_REPLAY, "quantized rules diverged on replay",
                        field_path="quantized_rules")
