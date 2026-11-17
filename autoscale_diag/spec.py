"""Load and strongly validate a local workload declaration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_SPEC, spec_error


METRICS = frozenset({"requests", "latency", "error_rate"})


@dataclass(frozen=True)
class Observation:
    minute: int
    requests: int
    avg_latency_ms: int
    errors: int

    def to_dict(self) -> dict[str, int]:
        return {
            "minute": self.minute,
            "requests": self.requests,
            "avg_latency_ms": self.avg_latency_ms,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class Rule:
    id: str
    metric: str
    target_value: int
    per_replica_capacity: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "metric": self.metric,
            "target_value": self.target_value,
            "per_replica_capacity": self.per_replica_capacity,
        }


@dataclass(frozen=True)
class Workload:
    id: str
    observations: tuple[Observation, ...]
    rules: tuple[Rule, ...]
    concurrency_per_replica: int
    min_replicas: int
    max_replicas: int
    cooldown_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "observations": [item.to_dict() for item in self.observations],
            "rules": [rule.to_dict() for rule in self.rules],
            "concurrency_per_replica": self.concurrency_per_replica,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "cooldown_ms": self.cooldown_ms,
        }


@dataclass(frozen=True)
class ClusterSpec:
    declaration_version: str
    global_replica_budget: int
    aggregation_threshold: int
    window_size: int
    workloads: tuple[Workload, ...]

    @property
    def timeline(self) -> tuple[int, ...]:
        return tuple(item.minute for item in self.workloads[0].observations)

    @property
    def minute_count(self) -> int:
        return len(self.workloads[0].observations)

    @property
    def workload_by_id(self) -> dict[str, Workload]:
        return {workload.id: workload for workload in self.workloads}

    def observation(self, workload_id: str, minute_index: int) -> Observation:
        return self.workload_by_id[workload_id].observations[minute_index]

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "declaration_version": self.declaration_version,
            "global_replica_budget": self.global_replica_budget,
            "aggregation_threshold": self.aggregation_threshold,
            "window_size": self.window_size,
            "workloads": [item.to_dict() for item in self.workloads],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def policy_payload(self) -> dict[str, Any]:
        return {
            "declaration_version": self.declaration_version,
            "global_replica_budget": self.global_replica_budget,
            "aggregation_threshold": self.aggregation_threshold,
            "window_size": self.window_size,
        }


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise spec_error(path, "must be an object")
    return value


def _require_list(value: Any, path: str, *, non_empty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise spec_error(path, "must be an array")
    if non_empty and not value:
        raise spec_error(path, "must not be empty")
    return value


def _string(value: Any, path: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise spec_error(path, "must be a string")
    if non_empty and not value.strip():
        raise spec_error(path, "must not be empty")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise spec_error(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise spec_error(path, f"must be >= {minimum}")
    return value


def _required(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise spec_error(f"{path}.{key}", "is required")
    return obj[key]


def _parse_rule(raw: Any, path: str) -> Rule:
    obj = _require_object(raw, path)
    rule_id = _string(_required(obj, "id", path), f"{path}.id")
    metric = _string(_required(obj, "metric", path), f"{path}.metric")
    if metric not in METRICS:
        raise spec_error(
            f"{path}.metric",
            "must be one of requests, latency, error_rate",
            allowed=sorted(METRICS),
        )
    target = _integer(
        _required(obj, "target_value", path), f"{path}.target_value", minimum=1
    )
    if target > 100:
        raise spec_error(f"{path}.target_value", "must be in the inclusive range 1..100")
    capacity = _integer(
        _required(obj, "per_replica_capacity", path),
        f"{path}.per_replica_capacity",
        minimum=1,
    )
    return Rule(rule_id, metric, target, capacity)


def _parse_observation(raw: Any, path: str) -> Observation:
    obj = _require_object(raw, path)
    minute = _integer(_required(obj, "minute", path), f"{path}.minute")
    requests = _integer(
        _required(obj, "requests", path), f"{path}.requests", minimum=0
    )
    latency = _integer(
        _required(obj, "avg_latency_ms", path),
        f"{path}.avg_latency_ms",
        minimum=0,
    )
    errors = _integer(_required(obj, "errors", path), f"{path}.errors", minimum=0)
    if errors > requests:
        raise spec_error(
            f"{path}.errors",
            "errors must not exceed requests",
            errors=errors,
            requests=requests,
        )
    return Observation(minute, requests, latency, errors)


def _parse_workload(raw: Any, path: str) -> Workload:
    obj = _require_object(raw, path)
    workload_id = _string(_required(obj, "id", path), f"{path}.id")

    raw_observations = _require_list(
        _required(obj, "observations", path), f"{path}.observations", non_empty=True
    )
    observations: list[Observation] = []
    previous_minute: int | None = None
    for index, item in enumerate(raw_observations):
        observation_path = f"{path}.observations[{index}]"
        observation = _parse_observation(item, observation_path)
        if previous_minute is not None and observation.minute != previous_minute + 1:
            raise spec_error(
                f"{observation_path}.minute",
                "must be exactly one greater than the previous minute",
                previous=previous_minute,
                actual=observation.minute,
            )
        previous_minute = observation.minute
        observations.append(observation)

    raw_rules = _require_list(
        _required(obj, "rules", path), f"{path}.rules", non_empty=True
    )
    rules: list[Rule] = []
    rule_ids: set[str] = set()
    for index, item in enumerate(raw_rules):
        rule_path = f"{path}.rules[{index}]"
        rule = _parse_rule(item, rule_path)
        if rule.id in rule_ids:
            raise spec_error(f"{rule_path}.id", "rule id must be unique within workload")
        rule_ids.add(rule.id)
        rules.append(rule)

    concurrency = _integer(
        _required(obj, "concurrency_per_replica", path),
        f"{path}.concurrency_per_replica",
        minimum=1,
    )
    min_replicas = _integer(
        _required(obj, "min_replicas", path), f"{path}.min_replicas", minimum=1
    )
    max_replicas = _integer(
        _required(obj, "max_replicas", path), f"{path}.max_replicas", minimum=1
    )
    if min_replicas > max_replicas:
        raise spec_error(
            f"{path}.min_replicas",
            "min_replicas must be <= max_replicas",
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
    cooldown_ms = _integer(
        _required(obj, "cooldown_ms", path), f"{path}.cooldown_ms", minimum=1
    )
    return Workload(
        id=workload_id,
        observations=tuple(observations),
        rules=tuple(rules),
        concurrency_per_replica=concurrency,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        cooldown_ms=cooldown_ms,
    )


def parse_spec(payload: Any) -> ClusterSpec:
    """Validate a decoded JSON value as a complete cluster declaration."""

    root = _require_object(payload, "$")
    version = _string(
        _required(root, "declaration_version", "$"), "declaration_version"
    )
    budget = _integer(
        _required(root, "global_replica_budget", "$"),
        "global_replica_budget",
        minimum=1,
    )
    threshold = _integer(
        _required(root, "aggregation_threshold", "$"),
        "aggregation_threshold",
        minimum=1,
    )
    window_size = _integer(
        _required(root, "window_size", "$"), "window_size", minimum=1
    )
    raw_workloads = _require_list(
        _required(root, "workloads", "$"), "workloads", non_empty=True
    )

    workloads: list[Workload] = []
    workload_ids: set[str] = set()
    for index, item in enumerate(raw_workloads):
        path = f"workloads[{index}]"
        workload = _parse_workload(item, path)
        if workload.id in workload_ids:
            raise spec_error(f"{path}.id", "workload id must be unique")
        workload_ids.add(workload.id)
        workloads.append(workload)

    baseline = workloads[0]
    for index, workload in enumerate(workloads[1:], start=1):
        if len(workload.observations) != len(baseline.observations):
            raise spec_error(
                f"workloads[{index}].observations",
                "all workloads must share the same minute grid",
                expected_minutes=len(baseline.observations),
                actual_minutes=len(workload.observations),
            )
        for obs_index, (left, right) in enumerate(
            zip(baseline.observations, workload.observations)
        ):
            if left.minute != right.minute:
                raise spec_error(
                    f"workloads[{index}].observations[{obs_index}].minute",
                    "all workloads must share the same minute grid",
                    expected=left.minute,
                    actual=right.minute,
                )

    minimum_total = sum(workload.min_replicas for workload in workloads)
    if budget < minimum_total:
        raise spec_error(
            "global_replica_budget",
            "global budget cannot be smaller than the sum of workload minimum replicas",
            budget=budget,
            minimum_total=minimum_total,
        )

    return ClusterSpec(
        declaration_version=version,
        global_replica_budget=budget,
        aggregation_threshold=threshold,
        window_size=window_size,
        workloads=tuple(workloads),
    )


def load_spec(path: str | Path) -> ClusterSpec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise spec_error("$", f"cannot read declaration file: {exc}", file=str(source))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise spec_error(
            "$",
            f"invalid JSON: {exc.msg}",
            line=exc.lineno,
            column=exc.colno,
            file=str(source),
        )
    return parse_spec(payload)

