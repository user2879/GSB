"""Loading and strict validation of event-source and aggregation-rule declarations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .errors import SUPPORTED_AGGREGATIONS, BusinessError
from .util import canonical_json, sha256_hex


class _UniqueKeyChecker:
    """Reject JSON objects with duplicate keys (specification ambiguity)."""

    def __init__(self) -> None:
        self.duplicate_path: str | None = None

    def __call__(self, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                self.duplicate_path = key
            result[key] = value
        return result


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    seed: int
    event_count: int


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    window_seconds: int
    group_key: str
    aggregations: tuple[str, ...]
    concurrency: int
    memory_budget: int
    version: int = 1
    source_ids: tuple[str, ...] = field(default_factory=tuple)

    def definition_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "window_seconds": self.window_seconds,
            "group_key": self.group_key,
            "aggregations": list(self.aggregations),
            "concurrency": self.concurrency,
            "memory_budget": self.memory_budget,
            "version": self.version,
            "source_ids": list(self.source_ids),
        }

    def signature(self) -> str:
        return sha256_hex(canonical_json(self.definition_dict()))


@dataclass(frozen=True)
class Spec:
    sources: tuple[SourceSpec, ...]
    rules: tuple[RuleSpec, ...]
    value_min: int
    value_max: int
    group_keys: tuple[str, ...]
    raw: dict[str, Any]

    def fingerprint(self) -> str:
        material = {
            "sources": [
                {"source_id": s.source_id, "seed": s.seed, "event_count": s.event_count}
                for s in self.sources
            ],
            "rules": [r.definition_dict() for r in self.rules],
            "value_min": self.value_min,
            "value_max": self.value_max,
            "group_keys": list(self.group_keys),
        }
        return sha256_hex(canonical_json(material))


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fail(message: str, path: str, details: dict[str, Any] | None = None) -> BusinessError:
    return BusinessError("E_SPEC", message, path, details)


def load_json_file(path: str, *, injection: str | None = None) -> Any:
    """Read a JSON file, surfacing parse failures as E_SPEC."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise _fail(f"cannot read declaration file: {exc}", "file") from exc
    checker = _UniqueKeyChecker()
    try:
        data = json.loads(text, object_pairs_hook=checker)
    except json.JSONDecodeError as exc:
        raise _fail(
            f"malformed JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            "file",
            {"injection": injection} if injection else None,
        ) from exc
    if checker.duplicate_path is not None:
        raise _fail(
            f"duplicate JSON key '{checker.duplicate_path}' makes the declaration ambiguous",
            checker.duplicate_path,
            {"injection": injection} if injection else None,
        )
    return data


def load_sample_overrides(path: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Load optional deterministic per-event overrides keyed by (source_id, source_seq)."""
    data = load_json_file(path)
    if not isinstance(data, dict) or "events" not in data or not isinstance(data["events"], list):
        raise _fail("sample override file must contain an 'events' array", "events")
    overrides: dict[tuple[str, int], dict[str, Any]] = {}
    for index, item in enumerate(data["events"]):
        item_path = f"events[{index}]"
        if not isinstance(item, dict):
            raise _fail("sample override entry must be an object", item_path)
        source_id = item.get("source_id")
        source_seq = item.get("source_seq")
        if not _is_identifier(source_id):
            raise _fail("sample override requires a non-empty source_id", f"{item_path}.source_id")
        if not _is_positive_int(source_seq):
            raise _fail("sample override requires a positive source_seq", f"{item_path}.source_seq")
        key = (source_id, source_seq)
        if key in overrides:
            raise _fail(
                "duplicate sample override for the same source event",
                f"{item_path}",
                {"source_id": source_id, "source_seq": source_seq},
            )
        allowed = {"source_id", "source_seq", "timestamp_ms", "group_key", "value"}
        unknown = set(item) - allowed
        if unknown:
            raise _fail(f"unknown sample override fields: {sorted(unknown)}", item_path)
        overrides[key] = dict(item)
    return overrides


def parse_spec(data: Any, *, injection: str | None = None) -> Spec:
    """Validate the raw declaration structure and return an immutable Spec."""
    if not isinstance(data, dict):
        raise _fail("declaration root must be a JSON object", "root")

    errors: list[dict[str, str]] = []

    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise _fail("'sources' must be a non-empty array", "sources")
    sources: list[SourceSpec] = []
    seen_source_ids: set[str] = set()
    for index, raw in enumerate(raw_sources):
        path = f"sources[{index}]"
        if not isinstance(raw, dict):
            errors.append({"path": path, "reason": "source entry must be an object"})
            continue
        if "source_id" in raw and "id" in raw and raw.get("source_id") != raw.get("id"):
            errors.append({"path": f"{path}.source_id", "reason": "both source_id and alias id provided with different values"})
        source_id = raw.get("source_id", raw.get("id"))
        seed = raw.get("seed")
        event_count = raw.get("event_count", raw.get("count"))
        if not _is_identifier(source_id):
            errors.append({"path": f"{path}.source_id", "reason": "source_id must be a non-empty string"})
        elif source_id in seen_source_ids:
            errors.append({"path": f"{path}.source_id", "reason": f"duplicate source_id '{source_id}'"})
        else:
            seen_source_ids.add(source_id)
        if not _is_positive_int(seed):
            errors.append({"path": f"{path}.seed", "reason": "seed must be a positive integer"})
        if not _is_positive_int(event_count):
            errors.append({"path": f"{path}.event_count", "reason": "event_count must be a positive integer"})
        if not errors or all(e["path"].startswith(path) for e in errors):
            if _is_identifier(source_id) and _is_positive_int(seed) and _is_positive_int(event_count):
                sources.append(SourceSpec(source_id=source_id, seed=seed, event_count=event_count))

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise _fail("'rules' must be a non-empty array", "rules")
    rules: list[RuleSpec] = []
    seen_rule_ids: set[str] = set()
    for index, raw in enumerate(raw_rules):
        path = f"rules[{index}]"
        local_errors: list[dict[str, str]] = []
        if not isinstance(raw, dict):
            errors.append({"path": path, "reason": "rule entry must be an object"})
            continue
        if "rule_id" in raw and "id" in raw and raw.get("rule_id") != raw.get("id"):
            local_errors.append({"path": f"{path}.rule_id", "reason": "both rule_id and alias id provided with different values"})
        rule_id = raw.get("rule_id", raw.get("id"))
        if not _is_identifier(rule_id):
            local_errors.append({"path": f"{path}.rule_id", "reason": "rule_id must be a non-empty string"})
        elif rule_id in seen_rule_ids:
            local_errors.append({"path": f"{path}.rule_id", "reason": f"duplicate rule_id '{rule_id}'"})
        else:
            seen_rule_ids.add(rule_id)

        window_seconds = raw.get("window_seconds")
        if not _is_positive_int(window_seconds):
            local_errors.append({"path": f"{path}.window_seconds", "reason": "window_seconds must be a positive integer"})

        group_key = raw.get("group_key")
        if not _is_identifier(group_key):
            local_errors.append({"path": f"{path}.group_key", "reason": "group_key must be a non-empty string"})

        aggregations = raw.get("aggregations", raw.get("aggregation_functions"))
        if not isinstance(aggregations, list) or not aggregations:
            local_errors.append({"path": f"{path}.aggregations", "reason": "aggregations must be a non-empty array"})
            aggregations = ()
        else:
            bad = [item for item in aggregations if item not in SUPPORTED_AGGREGATIONS]
            if bad:
                local_errors.append(
                    {
                        "path": f"{path}.aggregations",
                        "reason": f"unsupported aggregations {bad}; supported={list(SUPPORTED_AGGREGATIONS)}",
                    }
                )
            aggregations = tuple(dict.fromkeys(aggregations)) if not bad else tuple(aggregations)

        concurrency = raw.get("concurrency", raw.get("per_rule_concurrency"))
        if not _is_positive_int(concurrency):
            local_errors.append({"path": f"{path}.concurrency", "reason": "per-rule concurrency quota must be a positive integer"})

        memory_budget = raw.get("memory_budget", raw.get("per_window_memory_items"))
        if not _is_positive_int(memory_budget):
            local_errors.append({"path": f"{path}.memory_budget", "reason": "per-window memory item budget must be a positive integer"})

        version = raw.get("version", 1)
        if not _is_positive_int(version):
            local_errors.append({"path": f"{path}.version", "reason": "rule version must be a positive integer"})

        source_ids = raw.get("source_ids", [])
        if not isinstance(source_ids, list) or any(not _is_identifier(item) for item in source_ids):
            local_errors.append({"path": f"{path}.source_ids", "reason": "source_ids must be an array of non-empty strings"})
            source_ids = []

        errors.extend(local_errors)
        if not local_errors:
            rules.append(
                RuleSpec(
                    rule_id=rule_id,
                    window_seconds=window_seconds,
                    group_key=group_key,
                    aggregations=aggregations,
                    concurrency=concurrency,
                    memory_budget=memory_budget,
                    version=version,
                    source_ids=tuple(source_ids),
                )
            )

    value_min = data.get("value_min", 0)
    value_max = data.get("value_max", 100)
    if not isinstance(value_min, int) or isinstance(value_min, bool) or not isinstance(value_max, int) or isinstance(value_max, bool) or value_min > value_max:
        errors.append({"path": "value_range", "reason": "value_min/value_max must be integers with value_min <= value_max"})

    group_keys = data.get("group_keys", ["a", "b", "c", "d"])
    if not isinstance(group_keys, list) or not group_keys or any(not _is_identifier(item) for item in group_keys):
        errors.append({"path": "group_keys", "reason": "group_keys must be a non-empty array of non-empty strings"})
        group_keys = ["a", "b", "c", "d"]

    if errors:
        raise _fail(
            "declaration rejected with one or more invalid fields",
            errors[0]["path"],
            {"field_errors": errors, "injection": injection} if injection else {"field_errors": errors},
        )

    rule_sources = {item for rule in rules for item in rule.source_ids}
    unknown_sources = rule_sources - seen_source_ids
    if unknown_sources:
        raise _fail(
            "rule references unknown event source(s)",
            "rules.source_ids",
            {"unknown_source_ids": sorted(unknown_sources)},
        )

    return Spec(
        sources=tuple(sources),
        rules=tuple(rules),
        value_min=value_min,
        value_max=value_max,
        group_keys=tuple(group_keys),
        raw=data,
    )


def load_spec(path: str, *, injection: str | None = None) -> Spec:
    return parse_spec(load_json_file(path, injection=injection), injection=injection)
