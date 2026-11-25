from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import canonical_bytes
from .errors import SpecError


SPEC_FORMAT = "scale-review-spec"
SUPPORTED_SPEC_VERSION = 1
ESTIMATION_RULE_VERSION = "demand-ceil-v1"


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fail(field: str, message: str) -> None:
    raise SpecError(message, field=field)


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(field, f"{field} must be a JSON object")
    return value


def _require_int(value: Any, field: str) -> int:
    if not is_int(value):
        _fail(field, f"{field} must be an integer")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    number = _require_int(value, field)
    if number <= 0:
        _fail(field, f"{field} must be a positive integer")
    return number


def _require_non_negative_int(value: Any, field: str) -> int:
    number = _require_int(value, field)
    if number < 0:
        _fail(field, f"{field} must be a non-negative integer")
    return number


def _reject_unknown(obj: Mapping[str, Any], allowed: set[str], field: str) -> None:
    for key in obj:
        if key not in allowed:
            _fail(f"{field}.{key}", f"unknown field {field}.{key}")


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    requests_per_minute: tuple[int, ...]
    concurrency_per_replica: int
    request_duration_ms: int
    min_replicas: int
    max_replicas: int


@dataclass(frozen=True)
class DecisionRules:
    estimation_rule_version: str
    scale_up_threshold: int
    scale_down_threshold: int


@dataclass(frozen=True)
class AlgorithmConfig:
    window_length: int
    path_switch_length: int
    effort_tie_boundary: int
    tie_break_path: str


@dataclass(frozen=True)
class BudgetConfig:
    global_replica_budget: int
    service_concurrency_quotas: Mapping[str, int]


@dataclass(frozen=True)
class LeasePolicy:
    lease_ttl_ms: int
    renew_interval_ms: int
    max_inflight_tasks: int
    max_task_replicas: int
    simulated_task_ms: int = 0


@dataclass(frozen=True)
class Spec:
    format: str
    schema_version: int
    scaling_factor_permille: int
    decision_rules: DecisionRules
    algorithm: AlgorithmConfig
    budget: BudgetConfig
    lease_policy: LeasePolicy
    services: tuple[ServiceSpec, ...]

    @property
    def service_ids(self) -> list[str]:
        return [service.service_id for service in self.services]

    @property
    def minute_count(self) -> int:
        return len(self.services[0].requests_per_minute) if self.services else 0

    def service(self, service_id: str) -> ServiceSpec:
        for service in self.services:
            if service.service_id == service_id:
                return service
        raise KeyError(service_id)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "scaling_factor_permille": self.scaling_factor_permille,
            "decision_rules": {
                "estimation_rule_version": self.decision_rules.estimation_rule_version,
                "scale_down_threshold": self.decision_rules.scale_down_threshold,
                "scale_up_threshold": self.decision_rules.scale_up_threshold,
            },
            "algorithm": {
                "effort_tie_boundary": self.algorithm.effort_tie_boundary,
                "path_switch_length": self.algorithm.path_switch_length,
                "tie_break_path": self.algorithm.tie_break_path,
                "window_length": self.algorithm.window_length,
            },
            "budget": {
                "global_replica_budget": self.budget.global_replica_budget,
                "service_concurrency_quotas": dict(
                    sorted(self.budget.service_concurrency_quotas.items())
                ),
            },
            "lease_policy": {
                "lease_ttl_ms": self.lease_policy.lease_ttl_ms,
                "max_inflight_tasks": self.lease_policy.max_inflight_tasks,
                "max_task_replicas": self.lease_policy.max_task_replicas,
                "renew_interval_ms": self.lease_policy.renew_interval_ms,
                "simulated_task_ms": self.lease_policy.simulated_task_ms,
            },
            "services": [
                {
                    "concurrency_per_replica": service.concurrency_per_replica,
                    "max_replicas": service.max_replicas,
                    "min_replicas": service.min_replicas,
                    "request_duration_ms": service.request_duration_ms,
                    "requests_per_minute": list(service.requests_per_minute),
                    "service_id": service.service_id,
                }
                for service in self.services
            ],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.canonical_dict())


def validate_spec_dict(data: Any) -> Spec:
    root = _require_object(data, "$")
    allowed_root = {
        "format",
        "schema_version",
        "scaling_factor_permille",
        "decision_rules",
        "algorithm",
        "budget",
        "lease_policy",
        "services",
    }
    _reject_unknown(root, allowed_root, "$")

    if root.get("format") != SPEC_FORMAT:
        _fail("format", f"format must be {SPEC_FORMAT!r}")
    schema_version = _require_int(root.get("schema_version"), "schema_version")
    if schema_version != SUPPORTED_SPEC_VERSION:
        _fail("schema_version", "schema_version must be 1")

    scaling_factor = _require_positive_int(
        root.get("scaling_factor_permille"), "scaling_factor_permille"
    )

    rules_obj = _require_object(root.get("decision_rules"), "decision_rules")
    _reject_unknown(
        rules_obj,
        {
            "estimation_rule_version",
            "scale_up_threshold",
            "scale_down_threshold",
        },
        "decision_rules",
    )
    rule_version = rules_obj.get("estimation_rule_version")
    if rule_version != ESTIMATION_RULE_VERSION:
        _fail(
            "decision_rules.estimation_rule_version",
            f"estimation_rule_version must be {ESTIMATION_RULE_VERSION!r}",
        )
    up_threshold = _require_int(
        rules_obj.get("scale_up_threshold"), "decision_rules.scale_up_threshold"
    )
    down_threshold = _require_int(
        rules_obj.get("scale_down_threshold"), "decision_rules.scale_down_threshold"
    )

    algorithm_obj = _require_object(root.get("algorithm"), "algorithm")
    _reject_unknown(
        algorithm_obj,
        {
            "window_length",
            "path_switch_length",
            "effort_tie_boundary",
            "tie_break_path",
        },
        "algorithm",
    )
    window_length = _require_positive_int(
        algorithm_obj.get("window_length"), "algorithm.window_length"
    )
    path_switch_length = _require_positive_int(
        algorithm_obj.get("path_switch_length"), "algorithm.path_switch_length"
    )
    effort_tie_boundary = _require_non_negative_int(
        algorithm_obj.get("effort_tie_boundary"),
        "algorithm.effort_tie_boundary",
    )
    tie_break_path = algorithm_obj.get("tie_break_path")
    if tie_break_path not in {"full", "incremental"}:
        _fail(
            "algorithm.tie_break_path",
            "algorithm.tie_break_path must be 'full' or 'incremental'",
        )

    budget_obj = _require_object(root.get("budget"), "budget")
    _reject_unknown(
        budget_obj,
        {"global_replica_budget", "service_concurrency_quotas"},
        "budget",
    )
    global_budget = _require_non_negative_int(
        budget_obj.get("global_replica_budget"),
        "budget.global_replica_budget",
    )
    quotas_obj = _require_object(
        budget_obj.get("service_concurrency_quotas"),
        "budget.service_concurrency_quotas",
    )

    lease_obj = _require_object(root.get("lease_policy"), "lease_policy")
    _reject_unknown(
        lease_obj,
        {
            "lease_ttl_ms",
            "renew_interval_ms",
            "max_inflight_tasks",
            "max_task_replicas",
            "simulated_task_ms",
        },
        "lease_policy",
    )
    lease_ttl_ms = _require_positive_int(
        lease_obj.get("lease_ttl_ms"), "lease_policy.lease_ttl_ms"
    )
    renew_interval_ms = _require_positive_int(
        lease_obj.get("renew_interval_ms"), "lease_policy.renew_interval_ms"
    )
    if renew_interval_ms >= lease_ttl_ms:
        _fail(
            "lease_policy.renew_interval_ms",
            "lease_policy.renew_interval_ms must be strictly less than lease_ttl_ms",
        )
    max_inflight = _require_positive_int(
        lease_obj.get("max_inflight_tasks"), "lease_policy.max_inflight_tasks"
    )
    max_task_replicas = _require_positive_int(
        lease_obj.get("max_task_replicas"), "lease_policy.max_task_replicas"
    )
    simulated_task_ms = _require_non_negative_int(
        lease_obj.get("simulated_task_ms", 0), "lease_policy.simulated_task_ms"
    )

    services_value = root.get("services")
    if not isinstance(services_value, list) or not services_value:
        _fail("services", "services must be a non-empty JSON array")

    seen_ids: set[str] = set()
    services: list[ServiceSpec] = []
    expected_length: int | None = None
    for index, item in enumerate(services_value):
        field = f"services[{index}]"
        service_obj = _require_object(item, field)
        allowed = {
            "service_id",
            "requests_per_minute",
            "concurrency_per_replica",
            "request_duration_ms",
            "min_replicas",
            "max_replicas",
        }
        _reject_unknown(service_obj, allowed, field)

        service_id = service_obj.get("service_id")
        if not isinstance(service_id, str) or not service_id:
            _fail(f"{field}.service_id", f"{field}.service_id must be a non-empty string")
        id_field = f"services[{index}].service_id"
        if service_id in seen_ids:
            _fail(id_field, f"duplicate service_id {service_id!r}")
        seen_ids.add(service_id)

        requests_value = service_obj.get("requests_per_minute")
        requests_field = f"services[{index}].requests_per_minute"
        if not isinstance(requests_value, list):
            _fail(requests_field, f"{requests_field} must be an array")
        requests: list[int] = []
        for minute, request_count in enumerate(requests_value):
            minute_field = f"{requests_field}[{minute}]"
            value = _require_non_negative_int(request_count, minute_field)
            requests.append(value)
        if expected_length is None:
            expected_length = len(requests)
        elif len(requests) != expected_length:
            _fail(
                requests_field,
                "all services must have equally indexed minute sequences",
            )

        capacity = _require_positive_int(
            service_obj.get("concurrency_per_replica"),
            f"{field}.concurrency_per_replica",
        )
        duration_ms = _require_positive_int(
            service_obj.get("request_duration_ms"),
            f"{field}.request_duration_ms",
        )
        min_replicas = _require_non_negative_int(
            service_obj.get("min_replicas"), f"{field}.min_replicas"
        )
        max_replicas = _require_non_negative_int(
            service_obj.get("max_replicas"), f"{field}.max_replicas"
        )
        if min_replicas > max_replicas:
            _fail(
                f"{field}.min_replicas",
                f"{field}.min_replicas must be <= {field}.max_replicas",
            )

        quota_field = f"budget.service_concurrency_quotas.{service_id}"
        if service_id not in quotas_obj:
            _fail(quota_field, f"missing concurrency quota for service {service_id!r}")
        _require_non_negative_int(quotas_obj[service_id], quota_field)

        services.append(
            ServiceSpec(
                service_id=service_id,
                requests_per_minute=tuple(requests),
                concurrency_per_replica=capacity,
                request_duration_ms=duration_ms,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            )
        )

    for quota_service_id in quotas_obj:
        if quota_service_id not in seen_ids:
            _fail(
                f"budget.service_concurrency_quotas.{quota_service_id}",
                "quota key does not match a declared service_id",
            )
        _require_non_negative_int(
            quotas_obj[quota_service_id],
            f"budget.service_concurrency_quotas.{quota_service_id}",
        )

    ordered_services = tuple(sorted(services, key=lambda service: service.service_id))
    return Spec(
        format=SPEC_FORMAT,
        schema_version=schema_version,
        scaling_factor_permille=scaling_factor,
        decision_rules=DecisionRules(
            estimation_rule_version=rule_version,
            scale_up_threshold=up_threshold,
            scale_down_threshold=down_threshold,
        ),
        algorithm=AlgorithmConfig(
            window_length=window_length,
            path_switch_length=path_switch_length,
            effort_tie_boundary=effort_tie_boundary,
            tie_break_path=tie_break_path,
        ),
        budget=BudgetConfig(
            global_replica_budget=global_budget,
            service_concurrency_quotas=MappingProxyType(dict(quotas_obj)),
        ),
        lease_policy=LeasePolicy(
            lease_ttl_ms=lease_ttl_ms,
            renew_interval_ms=renew_interval_ms,
            max_inflight_tasks=max_inflight,
            max_task_replicas=max_task_replicas,
            simulated_task_ms=simulated_task_ms,
        ),
        services=ordered_services,
    )


def load_spec_dict(data: Any) -> Spec:
    return validate_spec_dict(data)


def load_spec_file(path: str | Path) -> Spec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read spec file: {exc}", field="$file") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            field="$json",
        ) from exc
    return validate_spec_dict(data)
