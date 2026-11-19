"""Versioned rule configuration and deterministic cross-trace case grouping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, parse_rules_timestamp, sha256_text
from .errors import AuditConfigError


GROUP_FIELDS = ("actor", "object", "action", "severity", "trace_id")
CONDITION_LIST_FIELDS = ("actors", "objects", "actions", "severities")


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AuditConfigError(f"{field} must be an array of non-empty strings")
    return list(value)


def _optional_timestamp(value: Any, field: str) -> int:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AuditConfigError(f"{field} must be milliseconds or ISO-8601")
    return parse_rules_timestamp(value)


def _normalize_conditions(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AuditConfigError("rule conditions must be an object")

    conditions: dict[str, Any] = {}
    for field in CONDITION_LIST_FIELDS:
        values = _string_list(raw.get(field), f"conditions.{field}")
        if values:
            conditions[field] = values

    window = raw.get("time_window")
    if window is None and any(
        key in raw for key in ("start_ms", "end_ms", "window_ms", "start", "end")
    ):
        window = {
            key: raw[key]
            for key in ("start_ms", "end_ms", "window_ms", "start", "end")
            if key in raw
        }
    if window is not None:
        if not isinstance(window, dict):
            raise AuditConfigError("conditions.time_window must be an object")
        normalized_window: dict[str, int] = {}
        starts = window.get("start_ms", window.get("start"))
        ends = window.get("end_ms", window.get("end"))
        if starts is not None:
            normalized_window["start_ms"] = _optional_timestamp(
                starts, "conditions.time_window.start"
            )
        if ends is not None:
            normalized_window["end_ms"] = _optional_timestamp(
                ends, "conditions.time_window.end"
            )
        if "window_ms" in window:
            gap = window["window_ms"]
            if not isinstance(gap, int) or isinstance(gap, bool) or gap < 0:
                raise AuditConfigError(
                    "conditions.time_window.window_ms must be a non-negative integer"
                )
            normalized_window["window_ms"] = gap
        if (
            "start_ms" in normalized_window
            and "end_ms" in normalized_window
            and normalized_window["start_ms"] > normalized_window["end_ms"]
        ):
            raise AuditConfigError("rule time window start must not exceed end")
        conditions["time_window"] = normalized_window

    return conditions


@dataclass(frozen=True)
class RuleConfig:
    version: str
    rules: tuple[dict[str, Any], ...]
    raw: dict[str, Any]

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(self.raw))

    def as_state(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "digest": self.digest,
            "config": self.raw,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RuleConfig":
        if not isinstance(raw, dict):
            raise AuditConfigError("rules configuration root must be an object")
        version_value = raw.get("rules_version", raw.get("rule_version"))
        if version_value is None:
            raise AuditConfigError("rules_version is required")
        if isinstance(version_value, bool) or not isinstance(
            version_value, (str, int)
        ):
            raise AuditConfigError("rules_version must be a string or integer")
        version = str(version_value)
        if not version:
            raise AuditConfigError("rules_version must not be empty")

        raw_rules = raw.get("rules")
        if not isinstance(raw_rules, list):
            raise AuditConfigError("rules must be an array")
        normalized_rules: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, candidate in enumerate(raw_rules):
            if not isinstance(candidate, dict):
                raise AuditConfigError(f"rules[{index}] must be an object")
            rule_id = candidate.get("rule_id")
            if not isinstance(rule_id, str) or not rule_id:
                raise AuditConfigError(f"rules[{index}].rule_id is required")
            if rule_id in seen_ids:
                raise AuditConfigError(f"duplicate rule_id: {rule_id}")
            seen_ids.add(rule_id)

            conditions = _normalize_conditions(candidate.get("conditions"))
            group_by = candidate.get("group_by", [])
            if not isinstance(group_by, list) or any(
                field not in GROUP_FIELDS for field in group_by
            ):
                raise AuditConfigError(
                    f"rule {rule_id} group_by must use {', '.join(GROUP_FIELDS)}"
                )
            if len(group_by) != len(set(group_by)):
                raise AuditConfigError(f"rule {rule_id} group_by has duplicates")

            normalized = {
                "rule_id": rule_id,
                "name": str(candidate.get("name", rule_id)),
                "conditions": conditions,
                "group_by": list(group_by),
            }
            if "description" in candidate:
                normalized["description"] = str(candidate["description"])
            normalized_rules.append(normalized)

        return cls(version=version, rules=tuple(normalized_rules), raw=raw)


def load_rules(path_like: str | Path) -> RuleConfig:
    path = Path(path_like)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise AuditConfigError(f"cannot read rules file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AuditConfigError(f"invalid rules JSON in {path}: {exc}") from exc
    return RuleConfig.from_dict(raw)


def _rule_matches(event: dict[str, Any], conditions: dict[str, Any]) -> bool:
    list_to_event_field = {
        "actors": "actor",
        "objects": "object",
        "actions": "action",
        "severities": "severity",
    }
    for condition_field, event_field in list_to_event_field.items():
        allowed = conditions.get(condition_field)
        if allowed and event.get(event_field) not in allowed:
            return False
    window = conditions.get("time_window")
    if window:
        if "start_ms" in window and event["ts"] < window["start_ms"]:
            return False
        if "end_ms" in window and event["ts"] > window["end_ms"]:
            return False
    return True


def _group_values(event: dict[str, Any], group_by: Iterable[str]) -> dict[str, Any]:
    return {field: event.get(field) for field in group_by}


def evaluate_rules(
    event_states: Iterable[dict[str, Any]],
    rule_config: RuleConfig,
    *,
    excluded_event_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Recompute every case from the complete event set.

    This function intentionally has no incremental mode. It is deterministic
    over ``event_id`` ordering and timestamp ordering for any arrival order.
    """

    excluded = excluded_event_ids or set()
    enriched: list[dict[str, Any]] = []
    for state in event_states:
        event = state.get("event", {})
        event_id = event.get("event_id")
        if not event_id or event_id in excluded:
            continue
        item = dict(event)
        attributes = state.get("attributes") or {}
        item["severity"] = attributes.get("severity")
        item["source"] = attributes.get("source")
        item["_state_status"] = state.get("status", "accepted")
        enriched.append(item)
    enriched.sort(key=lambda item: (item["ts"], item["event_id"]))

    cases: list[dict[str, Any]] = []
    for rule in rule_config.rules:
        conditions = rule["conditions"]
        selected = [
            event for event in enriched if _rule_matches(event, conditions)
        ]
        if not selected:
            continue

        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in selected:
            group = _group_values(event, rule["group_by"])
            key = canonical_json(group)
            grouped.setdefault(key, []).append(event)

        window_ms = (conditions.get("time_window") or {}).get("window_ms")
        for group_key in sorted(grouped):
            group_events = grouped[group_key]
            segments: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            previous_ts: int | None = None
            for event in group_events:
                if (
                    window_ms is not None
                    and previous_ts is not None
                    and event["ts"] - previous_ts > window_ms
                ):
                    if current:
                        segments.append(current)
                    current = []
                current.append(event)
                previous_ts = event["ts"]
            if current:
                segments.append(current)

            for segment in segments:
                member_ids = sorted(event["event_id"] for event in segment)
                group = _group_values(segment[0], rule["group_by"])
                identity = {
                    "rule_id": rule["rule_id"],
                    "rules_version": rule_config.version,
                    "group": group,
                    "member_event_ids": member_ids,
                }
                case_id = "CASE-" + sha256_text(canonical_json(identity))[:24].upper()
                members = []
                for event in sorted(
                    segment, key=lambda item: (item["ts"], item["event_id"])
                ):
                    members.append(
                        {
                            "event_id": event["event_id"],
                            "trace_id": event["trace_id"],
                            "ts": event["ts"],
                            "seq": event["seq"],
                            "actor": event["actor"],
                            "object": event["object"],
                            "action": event["action"],
                            "severity": event.get("severity"),
                        }
                    )
                traces = sorted({event["trace_id"] for event in segment})
                cases.append(
                    {
                        "case_id": case_id,
                        "rule_id": rule["rule_id"],
                        "rule_name": rule["name"],
                        "rules_version": rule_config.version,
                        "group": group,
                        "member_event_ids": member_ids,
                        "members": members,
                        "trace_ids": traces,
                        "start_ts": min(event["ts"] for event in segment),
                        "end_ts": max(event["ts"] for event in segment),
                        "association_basis": {
                            "rule_id": rule["rule_id"],
                            "rules_version": rule_config.version,
                            "conditions": conditions,
                            "group_by": rule["group_by"],
                        },
                    }
                )

    cases.sort(key=lambda case: case["case_id"])
    return cases
