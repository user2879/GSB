"""Strict local JSON loading and validation for the diagnostic session."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import digest_object
from .errors import SpecError
from .fixedpoint import SUPPORTED_AGGREGATES

SUPPORTED_STRATEGIES = ("source_max", "partition_min", "fixed_step")
SUPPORTED_BOUNDARY_MODES = ("end_from_parent_end",)
LEDGER_SPEC_VERSION = 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    value: Any,
    *,
    field_name: str,
    location: dict[str, Any],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not _is_int(value):
        raise SpecError(
            f"{field_name} must be an integer",
            details={**location, "field": field_name, "value": value},
        )
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise SpecError(
            f"{field_name} must be >= {minimum}",
            details={**location, "field": field_name, "value": integer},
        )
    if maximum is not None and integer > maximum:
        raise SpecError(
            f"{field_name} must be <= {maximum}",
            details={**location, "field": field_name, "value": integer},
        )
    return integer


def _require_string(
    value: Any, *, field_name: str, location: dict[str, Any]
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(
            f"{field_name} must be a non-empty string",
            details={**location, "field": field_name, "value": value},
        )
    return value.strip()


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_utf8(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise SpecError(
            f"cannot read JSON file: {file_path}",
            details={"file": str(file_path), "cause": str(exc)},
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpecError(
            f"JSON file is not valid UTF-8: {file_path}",
            details={"file": str(file_path), "cause": str(exc)},
        ) from exc
    try:
        return json.loads(text, object_pairs_hook=_duplicate_rejecting_pairs)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {file_path}",
            details={"file": str(file_path), "cause": str(exc)},
        ) from exc
    except ValueError as exc:
        raise SpecError(
            f"invalid JSON in {file_path}: {exc}",
            details={"file": str(file_path), "cause": str(exc)},
        ) from exc


@dataclass(frozen=True)
class StrategySpec:
    kind: str
    step_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.kind == "fixed_step":
            payload["step_ms"] = self.step_ms
        return payload


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    seed: int
    event_count: int
    out_of_order_ppm: int
    start_ms: int
    pace_ms: int
    partition_values: dict[str, tuple[Any, ...]]
    measure_range: tuple[int, int]
    measure_decimals: int
    weight_range: tuple[int, int]
    weight_decimals: int
    declaration_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "seed": self.seed,
            "event_count": self.event_count,
            "out_of_order_ppm": self.out_of_order_ppm,
            "start_ms": self.start_ms,
            "pace_ms": self.pace_ms,
            "partition_values": {
                key: list(values) for key, values in self.partition_values.items()
            },
            "measure_range": list(self.measure_range),
            "measure_decimals": self.measure_decimals,
            "weight_range": list(self.weight_range),
            "weight_decimals": self.weight_decimals,
        }


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    width_ms: int
    allowed_lateness_ms: int
    partition_keys: tuple[str, ...]
    aggregate_functions: tuple[str, ...]
    measure_column: str
    weight_column: str
    boundary_origin_ms: int
    declaration_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "width_ms": self.width_ms,
            "allowed_lateness_ms": self.allowed_lateness_ms,
            "partition_keys": list(self.partition_keys),
            "aggregate_functions": list(self.aggregate_functions),
            "measure_column": self.measure_column,
            "weight_column": self.weight_column,
            "boundary_origin_ms": self.boundary_origin_ms,
        }


@dataclass(frozen=True)
class DerivationSpec:
    derivation_id: str
    parent_rule_id: str
    child_rule_id: str
    boundary_mode: str
    offset_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            "parent_rule_id": self.parent_rule_id,
            "child_rule_id": self.child_rule_id,
            "boundary_mode": self.boundary_mode,
            "offset_ms": self.offset_ms,
        }


@dataclass(frozen=True)
class BudgetSpec:
    event_count_limit: int
    absolute_measure_limit_fixed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "event_count_limit": self.event_count_limit,
            "absolute_measure_limit_fixed": self.absolute_measure_limit_fixed,
        }


@dataclass(frozen=True)
class LateBudgetPolicy:
    default: BudgetSpec
    rule_overrides: dict[str, BudgetSpec] = field(default_factory=dict)
    window_overrides: dict[str, BudgetSpec] = field(default_factory=dict)

    def resolve(self, rule_id: str, window_id: str) -> BudgetSpec:
        return (
            self.window_overrides.get(window_id)
            or self.rule_overrides.get(rule_id)
            or self.default
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "default": self.default.to_dict(),
            "rule_overrides": {
                key: value.to_dict()
                for key, value in sorted(self.rule_overrides.items())
            },
            "window_overrides": {
                key: value.to_dict()
                for key, value in sorted(self.window_overrides.items())
            },
        }


@dataclass(frozen=True)
class SchedulerSpec:
    global_limit: int = 4
    per_partition_limit: int = 1
    task_timeout_ms: int = 1000
    retries: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "global_limit": self.global_limit,
            "per_partition_limit": self.per_partition_limit,
            "task_timeout_ms": self.task_timeout_ms,
            "retries": self.retries,
        }


@dataclass(frozen=True)
class ProbeSpec:
    enabled: bool = False
    timeout_ms: int = 150
    attempts: int = 2
    backoff_ms: int = 5
    server_delay_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timeout_ms": self.timeout_ms,
            "attempts": self.attempts,
            "backoff_ms": self.backoff_ms,
            "server_delay_ms": self.server_delay_ms,
        }


@dataclass
class SessionBundle:
    session_id: str
    scale: int
    strategy: StrategySpec
    rules_version: str
    sources: tuple[SourceSpec, ...]
    rules: tuple[RuleSpec, ...]
    derivations: tuple[DerivationSpec, ...]
    budgets: LateBudgetPolicy
    scheduler: SchedulerSpec
    probe: ProbeSpec
    source_files: tuple[str, ...]
    rules_file: str
    budget_file: str
    spec_digest: str = ""
    topological_rules: tuple[RuleSpec, ...] = ()
    derivation_by_child: dict[str, DerivationSpec] = field(default_factory=dict)

    def rule_by_id(self, rule_id: str) -> RuleSpec:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise KeyError(rule_id)

    def public_spec(self) -> dict[str, Any]:
        return {
            "format_version": LEDGER_SPEC_VERSION,
            "session_id": self.session_id,
            "scale": self.scale,
            "strategy": self.strategy.to_dict(),
            "rules_version": self.rules_version,
            "sources": [
                source.to_dict() for source in sorted(self.sources, key=lambda s: s.source_id)
            ],
            "rules": [
                rule.to_dict() for rule in sorted(self.rules, key=lambda r: r.rule_id)
            ],
            "derivations": [
                derivation.to_dict()
                for derivation in sorted(
                    self.derivations, key=lambda d: d.derivation_id
                )
            ],
            "late_budgets": self.budgets.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "probe": self.probe.to_dict(),
        }


def _parse_partition_values(
    raw: Any, *, source_id: str, index: int
) -> dict[str, tuple[Any, ...]]:
    location = {"kind": "source", "index": index, "source_id": source_id}
    if not isinstance(raw, dict) or not raw:
        raise SpecError(
            "partition_values must be a non-empty object",
            details={**location, "field": "partition_values"},
        )
    parsed: dict[str, tuple[Any, ...]] = {}
    for key, values in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise SpecError(
                "partition column name must be a non-empty string",
                details={**location, "field": "partition_values.key", "value": key},
            )
        if not isinstance(values, list) or not values:
            raise SpecError(
                "partition value list must be non-empty",
                details={
                    **location,
                    "field": f"partition_values.{key}",
                    "value": values,
                },
            )
        for value in values:
            if isinstance(value, (dict, list)) or value is None or isinstance(value, float):
                raise SpecError(
                    "partition values must be stable string or integer scalars",
                    details={
                        **location,
                        "field": f"partition_values.{key}",
                        "value": value,
                    },
                )
            if isinstance(value, bool):
                raise SpecError(
                    "partition values must not be booleans",
                    details={
                        **location,
                        "field": f"partition_values.{key}",
                        "value": value,
                    },
                )
        parsed[key] = tuple(values)
    return parsed


def _parse_range(
    raw: Any,
    *,
    field_name: str,
    default: tuple[int, int],
    location: dict[str, Any],
) -> tuple[int, int]:
    if raw is None:
        return default
    if not isinstance(raw, list) or len(raw) != 2:
        raise SpecError(
            f"{field_name} must be a two-integer array",
            details={**location, "field": field_name, "value": raw},
        )
    minimum = _require_int(raw[0], field_name=f"{field_name}[0]", location=location, minimum=0)
    maximum = _require_int(raw[1], field_name=f"{field_name}[1]", location=location, minimum=0)
    if minimum > maximum:
        raise SpecError(
            f"{field_name}[0] must be <= {field_name}[1]",
            details={**location, "field": field_name, "value": raw},
        )
    return minimum, maximum


def _parse_source(
    raw: Any, *, index: int, source_id_seen: set[str]
) -> SourceSpec:
    location = {"kind": "source", "index": index}
    if not isinstance(raw, dict):
        raise SpecError("source declaration must be an object", details=location)
    source_id = _require_string(raw.get("source_id"), field_name="source_id", location=location)
    location["source_id"] = source_id
    if source_id in source_id_seen:
        raise SpecError(
            "duplicate source_id",
            details={**location, "field": "source_id", "value": source_id},
        )
    source_id_seen.add(source_id)
    seed = _require_int(raw.get("seed"), field_name="seed", location=location, minimum=0)
    event_count = _require_int(
        raw.get("event_count"), field_name="event_count", location=location, minimum=0
    )
    ppm = _require_int(
        raw.get("out_of_order_ppm"),
        field_name="out_of_order_ppm",
        location=location,
        minimum=0,
        maximum=10000,
    )
    start_ms = _require_int(
        raw.get("start_ms", 0), field_name="start_ms", location=location, minimum=0
    )
    pace_ms = _require_int(
        raw.get("pace_ms", 100), field_name="pace_ms", location=location, minimum=1
    )
    partition_values = _parse_partition_values(
        raw.get("partition_values"), source_id=source_id, index=index
    )
    measure_range = _parse_range(
        raw.get("measure_range"),
        field_name="measure_range",
        default=(1, 9999),
        location=location,
    )
    measure_decimals = _require_int(
        raw.get("measure_decimals", 4),
        field_name="measure_decimals",
        location=location,
        minimum=0,
        maximum=9,
    )
    weight_range = _parse_range(
        raw.get("weight_range"),
        field_name="weight_range",
        default=(1, 9999),
        location=location,
    )
    weight_decimals = _require_int(
        raw.get("weight_decimals", 4),
        field_name="weight_decimals",
        location=location,
        minimum=0,
        maximum=9,
    )
    return SourceSpec(
        source_id=source_id,
        seed=seed,
        event_count=event_count,
        out_of_order_ppm=ppm,
        start_ms=start_ms,
        pace_ms=pace_ms,
        partition_values=partition_values,
        measure_range=measure_range,
        measure_decimals=measure_decimals,
        weight_range=weight_range,
        weight_decimals=weight_decimals,
        declaration_index=index,
    )


def _parse_strategy(raw: Any, *, file_path: str) -> StrategySpec:
    location = {"kind": "strategy", "file": file_path}
    if isinstance(raw, str):
        kind = raw
        step_ms = None
    elif isinstance(raw, dict):
        kind = raw.get("kind")
        step_ms = raw.get("step_ms")
    else:
        raise SpecError("strategy must be a string or object", details=location)
    if kind not in SUPPORTED_STRATEGIES:
        raise SpecError(
            "unsupported watermark strategy",
            details={**location, "field": "kind", "value": kind},
        )
    if kind == "fixed_step":
        step_ms = _require_int(
            step_ms,
            field_name="step_ms",
            location=location,
            minimum=1,
        )
    elif step_ms is not None:
        raise SpecError(
            "step_ms is only valid for fixed_step",
            details={**location, "field": "step_ms", "value": step_ms},
        )
    return StrategySpec(kind=kind, step_ms=step_ms if kind == "fixed_step" else None)


def _parse_rule(raw: Any, *, index: int, rule_id_seen: set[str]) -> RuleSpec:
    location = {"kind": "window", "index": index}
    if not isinstance(raw, dict):
        raise SpecError("window rule must be an object", details=location)
    rule_id = _require_string(raw.get("rule_id"), field_name="rule_id", location=location)
    location["rule_id"] = rule_id
    if rule_id in rule_id_seen:
        raise SpecError(
            "duplicate rule_id",
            details={**location, "field": "rule_id", "value": rule_id},
        )
    rule_id_seen.add(rule_id)
    width_ms = _require_int(
        raw.get("width_ms"), field_name="width_ms", location=location, minimum=1
    )
    allowed = _require_int(
        raw.get("allowed_lateness_ms"),
        field_name="allowed_lateness_ms",
        location=location,
        minimum=1,
    )
    keys_raw = raw.get("partition_keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise SpecError(
            "partition_keys must be a non-empty array",
            details={**location, "field": "partition_keys", "value": keys_raw},
        )
    keys: list[str] = []
    for key in keys_raw:
        if not isinstance(key, str) or not key.strip():
            raise SpecError(
                "partition key must be a non-empty string",
                details={**location, "field": "partition_keys", "value": key},
            )
        if key in keys:
            raise SpecError(
                "duplicate partition key",
                details={**location, "field": "partition_keys", "value": key},
            )
        keys.append(key)
    aggregates_raw = raw.get("aggregate_functions")
    if not isinstance(aggregates_raw, list) or not aggregates_raw:
        raise SpecError(
            "aggregate_functions must be a non-empty array",
            details={**location, "field": "aggregate_functions", "value": aggregates_raw},
        )
    aggregates: list[str] = []
    for aggregate in aggregates_raw:
        if aggregate not in SUPPORTED_AGGREGATES:
            raise SpecError(
                "unsupported aggregate function",
                details={
                    **location,
                    "field": "aggregate_functions",
                    "value": aggregate,
                },
            )
        if aggregate not in aggregates:
            aggregates.append(aggregate)
    measure_column = _require_string(
        raw.get("measure_column", "measure"),
        field_name="measure_column",
        location=location,
    )
    weight_column = _require_string(
        raw.get("weight_column", "weight"),
        field_name="weight_column",
        location=location,
    )
    if measure_column == weight_column:
        raise SpecError(
            "measure_column and weight_column must differ",
            details={**location, "field": "weight_column", "value": weight_column},
        )
    origin_ms = _require_int(
        raw.get("boundary_origin_ms", 0),
        field_name="boundary_origin_ms",
        location=location,
        minimum=0,
    )
    return RuleSpec(
        rule_id=rule_id,
        width_ms=width_ms,
        allowed_lateness_ms=allowed,
        partition_keys=tuple(keys),
        aggregate_functions=tuple(aggregates),
        measure_column=measure_column,
        weight_column=weight_column,
        boundary_origin_ms=origin_ms,
        declaration_index=index,
    )


def _parse_budget(raw: Any, *, location: dict[str, Any]) -> BudgetSpec:
    if not isinstance(raw, dict):
        raise SpecError("late budget must be an object", details=location)
    count = _require_int(
        raw.get("event_count_limit"),
        field_name="event_count_limit",
        location=location,
        minimum=0,
    )
    absolute = _require_int(
        raw.get("absolute_measure_limit_fixed"),
        field_name="absolute_measure_limit_fixed",
        location=location,
        minimum=0,
    )
    return BudgetSpec(event_count_limit=count, absolute_measure_limit_fixed=absolute)


def _parse_budget_policy(raw: Any) -> LateBudgetPolicy:
    location = {"kind": "late_budgets"}
    if not isinstance(raw, dict):
        raise SpecError("late_budgets must be an object", details=location)
    default = _parse_budget(raw.get("default"), location={**location, "scope": "default"})
    rule_overrides: dict[str, BudgetSpec] = {}
    raw_rule_overrides = raw.get("rule_overrides", {})
    if not isinstance(raw_rule_overrides, dict):
        raise SpecError(
            "rule_overrides must be an object",
            details={**location, "field": "rule_overrides"},
        )
    for key, value in raw_rule_overrides.items():
        rule_overrides[key] = _parse_budget(
            value, location={**location, "scope": "rule", "rule_id": key}
        )
    window_overrides: dict[str, BudgetSpec] = {}
    raw_window_overrides = raw.get("window_overrides", {})
    if not isinstance(raw_window_overrides, dict):
        raise SpecError(
            "window_overrides must be an object",
            details={**location, "field": "window_overrides"},
        )
    for key, value in raw_window_overrides.items():
        window_overrides[key] = _parse_budget(
            value, location={**location, "scope": "window", "window_id": key}
        )
    return LateBudgetPolicy(
        default=default,
        rule_overrides=rule_overrides,
        window_overrides=window_overrides,
    )


def _find_rule_cycle(
    rules: Iterable[RuleSpec], derivation_by_child: Mapping[str, DerivationSpec]
) -> list[str]:
    rule_order = [rule.rule_id for rule in rules]
    graph: dict[str, list[str]] = {rule_id: [] for rule_id in rule_order}
    for child, derivation in sorted(derivation_by_child.items()):
        graph.setdefault(derivation.parent_rule_id, []).append(child)
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        visiting.add(node)
        stack.append(node)
        for neighbor in sorted(graph.get(node, [])):
            if neighbor in visiting:
                start = stack.index(neighbor)
                return stack[start:]
            if neighbor not in visited:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for rule_id in rule_order:
        if rule_id not in visited:
            cycle = dfs(rule_id)
            if cycle:
                return cycle
    return []


def _topological_rules(
    rules: tuple[RuleSpec, ...], derivation_by_child: Mapping[str, DerivationSpec]
) -> tuple[RuleSpec, ...]:
    rule_map = {rule.rule_id: rule for rule in rules}
    ready = [
        rule
        for rule in sorted(rules, key=lambda item: item.rule_id)
        if rule.rule_id not in derivation_by_child
    ]
    children_by_parent: dict[str, list[str]] = {}
    for child, derivation in derivation_by_child.items():
        children_by_parent.setdefault(derivation.parent_rule_id, []).append(child)
    ordered: list[RuleSpec] = []
    emitted: set[str] = set()
    while ready:
        rule = ready.pop(0)
        if rule.rule_id in emitted:
            continue
        emitted.add(rule.rule_id)
        ordered.append(rule)
        for child in sorted(children_by_parent.get(rule.rule_id, [])):
            if child not in emitted and child in rule_map:
                ready.append(rule_map[child])
        ready.sort(key=lambda item: item.rule_id)
    if len(ordered) != len(rules):
        raise SpecError(
            "window derivation graph is cyclic",
            details={"kind": "derivation", "field": "cycle"},
        )
    return tuple(ordered)


def load_bundle(
    source_paths: Iterable[str | Path],
    rules_path: str | Path,
    budget_path: str | Path,
) -> SessionBundle:
    source_docs: list[tuple[str, Any]] = []
    for source_path in source_paths:
        path = Path(source_path)
        source_docs.append((str(path), load_json_utf8(path)))
    if not source_docs:
        raise SpecError("at least one source declaration file is required")
    rules_doc_path = Path(rules_path)
    budget_doc_path = Path(budget_path)
    rules_doc = load_json_utf8(rules_doc_path)
    budget_doc = load_json_utf8(budget_doc_path)

    session_ids: list[str] = []
    scales: list[int] = []
    strategies: list[StrategySpec] = []
    sources_raw: list[tuple[str, Any, int]] = []
    for file_path, doc in source_docs:
        if not isinstance(doc, dict):
            raise SpecError("source document must be an object", details={"file": file_path})
        _require_int(
            doc.get("format_version"),
            field_name="format_version",
            location={"file": file_path},
            minimum=1,
            maximum=1,
        )
        session_ids.append(
            _require_string(
                doc.get("session_id"), field_name="session_id", location={"file": file_path}
            )
        )
        scales.append(
            _require_int(
                doc.get("scale"),
                field_name="scale",
                location={"file": file_path},
                minimum=0,
                maximum=6,
            )
        )
        strategies.append(_parse_strategy(doc.get("strategy"), file_path=file_path))
        raw_sources = doc.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise SpecError(
                "sources must be a non-empty array",
                details={"file": file_path, "field": "sources"},
            )
        for offset, raw_source in enumerate(raw_sources):
            sources_raw.append((file_path, raw_source, offset))

    if len(set(session_ids)) != 1:
        raise SpecError(
            "all source files must declare the same session_id",
            details={"kind": "source", "field": "session_id", "values": session_ids},
        )
    if len(set(scales)) != 1:
        raise SpecError(
            "all source files must declare the same scale",
            details={"kind": "source", "field": "scale", "values": scales},
        )
    strategy_payloads = [strategy.to_dict() for strategy in strategies]
    if any(payload != strategy_payloads[0] for payload in strategy_payloads):
        raise SpecError(
            "all source files must declare the same watermark strategy",
            details={"kind": "source", "field": "strategy", "values": strategy_payloads},
        )
    session_id = session_ids[0]
    scale = scales[0]
    strategy = strategies[0]

    source_seen: set[str] = set()
    sources: list[SourceSpec] = []
    for index, (_file_path, raw_source, _offset) in enumerate(sources_raw):
        sources.append(_parse_source(raw_source, index=index, source_id_seen=source_seen))

    if not isinstance(rules_doc, dict):
        raise SpecError(
            "window rules document must be an object", details={"file": str(rules_doc_path)}
        )
    _require_int(
        rules_doc.get("format_version"),
        field_name="format_version",
        location={"file": str(rules_doc_path)},
        minimum=1,
        maximum=1,
    )
    rules_version = _require_string(
        rules_doc.get("rules_version"),
        field_name="rules_version",
        location={"file": str(rules_doc_path)},
    )
    raw_rules = rules_doc.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise SpecError(
            "rules must be a non-empty array",
            details={"file": str(rules_doc_path), "field": "rules"},
        )
    rule_seen: set[str] = set()
    rules: list[RuleSpec] = []
    for index, raw_rule in enumerate(raw_rules):
        rules.append(_parse_rule(raw_rule, index=index, rule_id_seen=rule_seen))
    rules_tuple = tuple(rules)

    for source in sources:
        for rule in rules_tuple:
            for key in rule.partition_keys:
                if key not in source.partition_values:
                    raise SpecError(
                        "partition key column is absent from source declaration",
                        details={
                            "kind": "source",
                            "index": source.declaration_index,
                            "source_id": source.source_id,
                            "field": f"partition_values.{key}",
                        },
                    )

    if not isinstance(budget_doc, dict):
        raise SpecError(
            "derivation and late-budget document must be an object",
            details={"file": str(budget_doc_path)},
        )
    _require_int(
        budget_doc.get("format_version"),
        field_name="format_version",
        location={"file": str(budget_doc_path)},
        minimum=1,
        maximum=1,
    )
    raw_derivations = budget_doc.get("derivations", [])
    if not isinstance(raw_derivations, list):
        raise SpecError(
            "derivations must be an array",
            details={"file": str(budget_doc_path), "field": "derivations"},
        )
    rule_map = {rule.rule_id: rule for rule in rules_tuple}
    derivations: list[DerivationSpec] = []
    derivation_ids: set[str] = set()
    derivation_by_child: dict[str, DerivationSpec] = {}
    for index, raw_derivation in enumerate(raw_derivations):
        location = {"kind": "window", "index": index}
        if not isinstance(raw_derivation, dict):
            raise SpecError("derivation must be an object", details=location)
        derivation_id = _require_string(
            raw_derivation.get("derivation_id"), field_name="derivation_id", location=location
        )
        location["derivation_id"] = derivation_id
        if derivation_id in derivation_ids:
            raise SpecError(
                "duplicate derivation_id",
                details={**location, "field": "derivation_id", "value": derivation_id},
            )
        derivation_ids.add(derivation_id)
        parent = _require_string(
            raw_derivation.get("parent_rule_id"),
            field_name="parent_rule_id",
            location=location,
        )
        child = _require_string(
            raw_derivation.get("child_rule_id"),
            field_name="child_rule_id",
            location=location,
        )
        if parent not in rule_map:
            raise SpecError(
                "derivation parent window rule does not exist",
                details={**location, "field": "parent_rule_id", "value": parent},
            )
        if child not in rule_map:
            raise SpecError(
                "derivation target window rule does not exist",
                details={**location, "field": "child_rule_id", "value": child},
            )
        if parent == child:
            raise SpecError(
                "derivation cannot target itself",
                details={**location, "field": "child_rule_id", "value": child},
            )
        mode = _require_string(
            raw_derivation.get("boundary_mode", "end_from_parent_end"),
            field_name="boundary_mode",
            location=location,
        )
        if mode not in SUPPORTED_BOUNDARY_MODES:
            raise SpecError(
                "unsupported boundary mode",
                details={**location, "field": "boundary_mode", "value": mode},
            )
        offset_ms = _require_int(
            raw_derivation.get("offset_ms", 0),
            field_name="offset_ms",
            location=location,
            minimum=0,
        )
        if child in derivation_by_child:
            raise SpecError(
                "window rule has more than one incoming derivation",
                details={**location, "field": "child_rule_id", "value": child},
            )
        parent_rule = rule_map[parent]
        child_rule = rule_map[child]
        if parent_rule.partition_keys != child_rule.partition_keys:
            raise SpecError(
                "derived windows must share partition keys with their parent",
                details={
                    **location,
                    "field": "partition_keys",
                    "parent": list(parent_rule.partition_keys),
                    "child": list(child_rule.partition_keys),
                },
            )
        derivation = DerivationSpec(
            derivation_id=derivation_id,
            parent_rule_id=parent,
            child_rule_id=child,
            boundary_mode=mode,
            offset_ms=offset_ms,
        )
        derivations.append(derivation)
        derivation_by_child[child] = derivation

    cycle = _find_rule_cycle(rules_tuple, derivation_by_child)
    if cycle:
        first_cycle_rule = next(
            rule for rule in rules_tuple if rule.rule_id in set(cycle)
        )
        raise SpecError(
            "window derivation graph is cyclic",
            details={
                "kind": "window",
                "index": first_cycle_rule.declaration_index,
                "rule_id": first_cycle_rule.rule_id,
                "field": "derivations",
                "cycle": cycle,
            },
        )
    topo_rules = _topological_rules(rules_tuple, derivation_by_child)

    budgets = _parse_budget_policy(budget_doc.get("late_budgets"))
    for rule_id in budgets.rule_overrides:
        if rule_id not in rule_map:
            raise SpecError(
                "late-budget rule override references an unknown rule",
                details={"kind": "window", "field": "late_budgets.rule_overrides", "rule_id": rule_id},
            )

    raw_scheduler = budget_doc.get("finalization_scheduler", {})
    if not isinstance(raw_scheduler, dict):
        raise SpecError(
            "finalization_scheduler must be an object",
            details={"kind": "scheduler", "field": "finalization_scheduler"},
        )
    global_limit = _require_int(
        raw_scheduler.get("global_limit", 4),
        field_name="global_limit",
        location={"kind": "scheduler"},
        minimum=1,
    )
    per_partition_limit = _require_int(
        raw_scheduler.get("per_partition_limit", 1),
        field_name="per_partition_limit",
        location={"kind": "scheduler"},
        minimum=1,
    )
    if per_partition_limit > global_limit:
        raise SpecError(
            "per_partition_limit must not exceed global_limit",
            details={
                "kind": "scheduler",
                "field": "per_partition_limit",
                "value": per_partition_limit,
                "global_limit": global_limit,
            },
        )
    scheduler = SchedulerSpec(
        global_limit=global_limit,
        per_partition_limit=per_partition_limit,
        task_timeout_ms=_require_int(
            raw_scheduler.get("task_timeout_ms", 1000),
            field_name="task_timeout_ms",
            location={"kind": "scheduler"},
            minimum=0,
        ),
        retries=_require_int(
            raw_scheduler.get("retries", 0),
            field_name="retries",
            location={"kind": "scheduler"},
            minimum=0,
        ),
    )

    raw_probe = budget_doc.get("source_probe", {})
    if not isinstance(raw_probe, dict):
        raise SpecError(
            "source_probe must be an object", details={"kind": "probe"}
        )
    enabled = raw_probe.get("enabled", False)
    if not isinstance(enabled, bool):
        raise SpecError(
            "source_probe.enabled must be a boolean",
            details={"kind": "probe", "field": "enabled", "value": enabled},
        )
    probe = ProbeSpec(
        enabled=enabled,
        timeout_ms=_require_int(
            raw_probe.get("timeout_ms", 150),
            field_name="timeout_ms",
            location={"kind": "probe"},
            minimum=1,
        ),
        attempts=_require_int(
            raw_probe.get("attempts", 2),
            field_name="attempts",
            location={"kind": "probe"},
            minimum=1,
        ),
        backoff_ms=_require_int(
            raw_probe.get("backoff_ms", 5),
            field_name="backoff_ms",
            location={"kind": "probe"},
            minimum=0,
        ),
        server_delay_ms=_require_int(
            raw_probe.get("server_delay_ms", 0),
            field_name="server_delay_ms",
            location={"kind": "probe"},
            minimum=0,
        ),
    )

    bundle = SessionBundle(
        session_id=session_id,
        scale=scale,
        strategy=strategy,
        rules_version=rules_version,
        sources=tuple(sources),
        rules=rules_tuple,
        derivations=tuple(derivations),
        budgets=budgets,
        scheduler=scheduler,
        probe=probe,
        source_files=tuple(str(path) for path, _ in source_docs),
        rules_file=str(rules_doc_path),
        budget_file=str(budget_doc_path),
        topological_rules=topo_rules,
        derivation_by_child=derivation_by_child,
    )
    bundle.spec_digest = digest_object(bundle.public_spec())
    return bundle
