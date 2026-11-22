"""Strong validation loader for cluster declaration JSON files."""

from __future__ import annotations

import json
from typing import Any

from .errors import LengthError, SchemaError
from .models import ClusterSpec, ServiceSpec, _is_int

DEFAULT_LATENCY_BUDGET_MS = 50
DEFAULT_INCREMENTAL_THRESHOLD = 12
DEFAULT_WINDOW_SIZE = 8

_SERVICE_FIELDS = (
    "service_id",
    "requests",
    "concurrency_per_replica",
    "latency_ms",
    "scale_up_threshold",
    "scale_down_threshold",
    "min_replicas",
    "max_replicas",
    "concurrency_quota",
)


def _fail(field: str, reason: str, *, service: str | None = None, index: int | None = None) -> None:
    details: dict[str, Any] = {"field": field, "reason": reason}
    if service is not None:
        details["service_id"] = service
    if index is not None:
        details["service_index"] = index
    raise SchemaError(f"invalid declaration field '{field}': {reason}", details=details)


def _require_int(value: Any, field: str, *, service: str | None, index: int | None) -> int:
    if not _is_int(value):
        _fail(field, "expected integer", service=service, index=index)
    return value


def _validate_service(raw: Any, index: int, seen_ids: set[str]) -> ServiceSpec:
    if not isinstance(raw, dict):
        _fail("services", "each service must be an object", index=index)

    for field_name in _SERVICE_FIELDS:
        if field_name not in raw:
            _fail(field_name, "missing required field",
                  service=raw.get("service_id") if isinstance(raw, dict) else None,
                  index=index)

    service_id = raw["service_id"]
    if not isinstance(service_id, str) or not service_id.strip():
        _fail("service_id", "expected non-empty string", index=index)
    if service_id in seen_ids:
        _fail("service_id", "duplicate service identifier", service=service_id, index=index)
    seen_ids.add(service_id)

    requests = raw["requests"]
    if not isinstance(requests, list) or not requests:
        _fail("requests", "expected non-empty list of integers", service=service_id, index=index)
    checked_requests: list[int] = []
    for minute, value in enumerate(requests):
        if not _is_int(value) or value < 0:
            _fail("requests", f"element at minute {minute} must be a non-negative integer",
                  service=service_id, index=index)
        checked_requests.append(value)

    concurrency = _require_int(raw["concurrency_per_replica"], "concurrency_per_replica",
                               service=service_id, index=index)
    if concurrency < 0:
        _fail("concurrency_per_replica", "must be non-negative integer",
              service=service_id, index=index)

    latency = _require_int(raw["latency_ms"], "latency_ms",
                           service=service_id, index=index)
    if latency < 0:
        _fail("latency_ms", "must be non-negative integer", service=service_id, index=index)

    scale_up = _require_int(raw["scale_up_threshold"], "scale_up_threshold",
                            service=service_id, index=index)
    if not 0 <= scale_up <= 10000:
        _fail("scale_up_threshold", "must be an integer in [0, 10000] basis points",
              service=service_id, index=index)

    scale_down = _require_int(raw["scale_down_threshold"], "scale_down_threshold",
                              service=service_id, index=index)
    if not 0 <= scale_down <= 10000:
        _fail("scale_down_threshold", "must be an integer in [0, 10000] basis points",
              service=service_id, index=index)
    if scale_down >= scale_up:
        _fail("scale_down_threshold",
              "must be strictly smaller than scale_up_threshold",
              service=service_id, index=index)

    min_replicas = _require_int(raw["min_replicas"], "min_replicas",
                                service=service_id, index=index)
    if min_replicas < 0:
        _fail("min_replicas", "must be non-negative integer", service=service_id, index=index)

    max_replicas = _require_int(raw["max_replicas"], "max_replicas",
                                service=service_id, index=index)
    if max_replicas < 0:
        _fail("max_replicas", "must be non-negative integer", service=service_id, index=index)
    if min_replicas > max_replicas:
        _fail("min_replicas", "min_replicas must not exceed max_replicas",
              service=service_id, index=index)

    quota = _require_int(raw["concurrency_quota"], "concurrency_quota",
                         service=service_id, index=index)
    if quota <= 0:
        _fail("concurrency_quota", "must be a positive integer", service=service_id, index=index)

    return ServiceSpec(
        service_id=service_id,
        requests=tuple(checked_requests),
        concurrency_per_replica=concurrency,
        latency_ms=latency,
        scale_up_threshold=scale_up,
        scale_down_threshold=scale_down,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        concurrency_quota=quota,
    )


def check_series_lengths(spec: ClusterSpec) -> None:
    """Raise E_LENGTH when service load series disagree on length.

    Kept separate from :func:`load_declaration` so tampered/replay inputs can
    surface the dedicated E_LENGTH diagnostic even though initial loading
    rejects malformed files with E_SCHEMA.
    """

    if not spec.services:
        raise LengthError("declaration contains no services", details={"field": "services"})
    expected = len(spec.services[0].requests)
    for service in spec.services[1:]:
        if len(service.requests) != expected:
            raise LengthError(
                "load series length mismatch",
                details={
                    "service_id": service.service_id,
                    "field": "requests",
                    "expected_length": expected,
                    "actual_length": len(service.requests),
                },
            )


def load_declaration(path: str) -> ClusterSpec:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise SchemaError(f"declaration file not found: {path}",
                          details={"field": "file", "path": path})
    except json.JSONDecodeError as exc:
        raise SchemaError(f"declaration is not valid JSON: {exc.msg}",
                          details={"field": "file", "line": exc.lineno, "column": exc.colno})

    if not isinstance(raw, dict):
        _fail("root", "declaration root must be an object")

    services_raw = raw.get("services")
    if not isinstance(services_raw, list) or not services_raw:
        _fail("services", "expected a non-empty list of service objects")

    budget = _require_int(raw.get("global_replica_budget"), "global_replica_budget",
                          service=None, index=None)
    if budget <= 0:
        _fail("global_replica_budget", "must be a positive integer")

    scale = raw.get("fixed_point_scale", 1000)
    if not _is_int(scale) or scale <= 0:
        _fail("fixed_point_scale", "must be a positive integer")

    latency_budget = raw.get("latency_budget_ms", DEFAULT_LATENCY_BUDGET_MS)
    if not _is_int(latency_budget) or latency_budget <= 0:
        _fail("latency_budget_ms", "must be a positive integer")

    threshold = raw.get("incremental_threshold_minutes", DEFAULT_INCREMENTAL_THRESHOLD)
    if not _is_int(threshold) or threshold < 0:
        _fail("incremental_threshold_minutes", "must be a non-negative integer")

    window = raw.get("window_size", DEFAULT_WINDOW_SIZE)
    if not _is_int(window) or window <= 0:
        _fail("window_size", "must be a positive integer")

    seen_ids: set[str] = set()
    services = tuple(
        _validate_service(raw_service, position, seen_ids)
        for position, raw_service in enumerate(services_raw)
    )

    spec = ClusterSpec(
        services=services,
        global_replica_budget=budget,
        fixed_point_scale=scale,
        latency_budget_ms=latency_budget,
        incremental_threshold_minutes=threshold,
        window_size=window,
    )
    # Initial load enforces equal lengths inside E_SCHEMA's strong contract.
    try:
        check_series_lengths(spec)
    except LengthError as exc:
        raise SchemaError(exc.message, details=exc.details)
    return spec
