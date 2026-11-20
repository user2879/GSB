"""Cluster declaration loading with all-or-nothing strong validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    canonical_bytes,
    is_nonneg_int,
    is_pos_int,
)
from .errors import BudgetError, SpecError

_MS_PER_MINUTE = 60_000
_SERVICE_FIELDS = (
    "name",
    "requests",
    "concurrency_limit",
    "request_ms",
    "scale_threshold_pct",
    "min_replicas",
    "max_replicas",
    "concurrency_quota",
)


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    requests: tuple[int, ...]
    concurrency_limit: int
    request_ms: int
    scale_threshold_pct: int
    min_replicas: int
    max_replicas: int
    concurrency_quota: int

    @property
    def quota_replica_cap(self) -> int:
        """Largest replica count that still fits the per-service concurrency quota."""
        return self.concurrency_quota // self.concurrency_limit

    @property
    def per_replica_minute_capacity(self) -> int:
        """Whole requests one replica can finish per minute (floor division)."""
        return (self.concurrency_limit * _MS_PER_MINUTE) // self.request_ms


@dataclass(frozen=True)
class ClusterSpec:
    global_replica_budget: int
    services: tuple[ServiceSpec, ...]
    label: str = ""
    digest: str = ""
    horizon: int = field(default=0)

    def service(self, name: str) -> ServiceSpec:
        for svc in self.services:
            if svc.name == name:
                return svc
        raise KeyError(name)

    def requests_at(self, minute: int) -> dict[str, int]:
        return {svc.name: svc.requests[minute] for svc in self.services}


def _validate_service(raw: Any, index: int, errors: list[tuple[str, str]]) -> None:
    base = f"/services/{index}"
    if not isinstance(raw, dict):
        errors.append((base, "service definition must be an object"))
        return

    for key in _SERVICE_FIELDS:
        if key not in raw:
            errors.append((f"{base}/{key}", "missing required field"))

    name = raw.get("name")
    if "name" in raw and (not isinstance(name, str) or not name.strip()):
        errors.append((f"{base}/name", "service name must be a non-empty string"))

    requests = raw.get("requests")
    if "requests" in raw:
        path = f"{base}/requests"
        if not isinstance(requests, list) or not requests:
            errors.append((path, "requests must be a non-empty array indexed by minute"))
        else:
            for minute, value in enumerate(requests):
                if value is None:
                    errors.append((f"{path}[{minute}]", "load gap: missing request count for minute"))
                elif not is_nonneg_int(value):
                    errors.append((f"{path}[{minute}]", "request count must be a non-negative integer"))

    if "concurrency_limit" in raw and not is_pos_int(raw["concurrency_limit"]):
        errors.append((f"{base}/concurrency_limit", "must be a positive integer"))
    if "request_ms" in raw and not is_pos_int(raw["request_ms"]):
        errors.append((f"{base}/request_ms", "must be a positive integer (milliseconds)"))
    if "scale_threshold_pct" in raw:
        threshold = raw["scale_threshold_pct"]
        if not (is_pos_int(threshold) and 1 <= threshold <= 100):
            errors.append((f"{base}/scale_threshold_pct", "must be an integer in [1, 100]"))
    if "min_replicas" in raw and not is_nonneg_int(raw["min_replicas"]):
        errors.append((f"{base}/min_replicas", "must be a non-negative integer"))
    if "max_replicas" in raw and not is_pos_int(raw["max_replicas"]):
        errors.append((f"{base}/max_replicas", "must be a positive integer"))
    if "concurrency_quota" in raw and not is_pos_int(raw["concurrency_quota"]):
        errors.append((f"{base}/concurrency_quota", "must be a positive integer"))

    if (
        is_nonneg_int(raw.get("min_replicas"))
        and is_pos_int(raw.get("max_replicas"))
        and raw["min_replicas"] > raw["max_replicas"]
    ):
        errors.append((f"{base}/min_replicas", "min_replicas must be <= max_replicas"))


def _build_spec(raw: dict[str, Any]) -> ClusterSpec:
    errors: list[tuple[str, str]] = []
    budget_errors: list[tuple[str, str]] = []

    if not isinstance(raw, dict):
        raise SpecError("cluster declaration must be a JSON object", fields=["/"])
    if "global_replica_budget" not in raw:
        errors.append(("/global_replica_budget", "missing required field"))
    elif not is_pos_int(raw["global_replica_budget"]):
        budget_errors.append(("/global_replica_budget", "must be a positive integer"))

    services_raw = raw.get("services")
    if not isinstance(services_raw, list) or not services_raw:
        errors.append(("/services", "must be a non-empty array of service definitions"))
        services_raw = []

    for index, svc_raw in enumerate(services_raw):
        _validate_service(svc_raw, index, errors)

    # Name uniqueness (stable indices are reported for both offenders).
    seen: dict[str, int] = {}
    for index, svc_raw in enumerate(services_raw):
        name = svc_raw.get("name") if isinstance(svc_raw, dict) else None
        if isinstance(name, str) and name.strip():
            if name in seen:
                errors.append((f"/services/{index}/name", f"duplicate service name {name!r}"))
                errors.append((f"/services/{seen[name]}/name", f"duplicate service name {name!r}"))
            seen[name] = index

    if errors:
        raise SpecError(
            "cluster declaration rejected as a whole",
            fields=[path for path, _ in errors],
            context={"reasons": [{"field": path, "reason": reason} for path, reason in errors]},
        )

    if budget_errors:
        raise BudgetError(
            "budget or quota parameters are invalid",
            fields=[path for path, _ in budget_errors],
            context={"reasons": [{"field": path, "reason": reason}
                                  for path, reason in budget_errors]},
        )

    horizon = len(services_raw[0]["requests"])
    for index, svc_raw in enumerate(services_raw):
        if len(svc_raw["requests"]) != horizon:
            raise SpecError(
                "load sequences must cover the same contiguous minute horizon",
                fields=[f"/services/{index}/requests"],
            )

    services = tuple(
        ServiceSpec(
            name=svc_raw["name"],
            requests=tuple(svc_raw["requests"]),
            concurrency_limit=svc_raw["concurrency_limit"],
            request_ms=svc_raw["request_ms"],
            scale_threshold_pct=svc_raw["scale_threshold_pct"],
            min_replicas=svc_raw["min_replicas"],
            max_replicas=svc_raw["max_replicas"],
            concurrency_quota=svc_raw["concurrency_quota"],
        )
        for svc_raw in services_raw
    )
    services = tuple(sorted(services, key=lambda svc: svc.name))

    for index, svc in enumerate(services_raw):
        if "concurrency_quota" in svc and not is_pos_int(svc["concurrency_quota"]):
            budget_errors.append((
                f"/services/{index}/concurrency_quota",
                "must be a positive integer",
            ))
    for svc in services:
        if svc.min_replicas * svc.concurrency_limit > svc.concurrency_quota:
            budget_errors.append((
                f"/services/{svc.name}/concurrency_quota",
                "quota cannot hold the service's minimum replicas",
            ))
    reserved = sum(svc.min_replicas for svc in services)
    if reserved > raw["global_replica_budget"]:
        budget_errors.append((
            "/global_replica_budget",
            f"sum of minimum replicas ({reserved}) exceeds global budget",
        ))
    if budget_errors:
        raise BudgetError(
            "budget or quota parameters are infeasible",
            fields=[path for path, _ in budget_errors],
            context={"reasons": [{"field": path, "reason": reason} for path, reason in budget_errors]},
        )

    normalized = {
        "global_replica_budget": raw["global_replica_budget"],
        "services": [
            {
                "name": svc.name,
                "requests": list(svc.requests),
                "concurrency_limit": svc.concurrency_limit,
                "request_ms": svc.request_ms,
                "scale_threshold_pct": svc.scale_threshold_pct,
                "min_replicas": svc.min_replicas,
                "max_replicas": svc.max_replicas,
                "concurrency_quota": svc.concurrency_quota,
            }
            for svc in services
        ],
    }
    digest = hashlib.sha256(canonical_bytes(normalized)).hexdigest()
    return ClusterSpec(
        global_replica_budget=raw["global_replica_budget"],
        services=services,
        label=str(raw.get("version_label", "")),
        digest=digest,
        horizon=horizon,
    )


def parse_spec(raw: dict[str, Any]) -> ClusterSpec:
    return _build_spec(raw)


def load_spec(path: str | Path) -> ClusterSpec:
    """Load and strongly validate a cluster declaration from a local JSON file."""
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"spec file not found: {file_path}", fields=[str(file_path)]) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            fields=[str(file_path)],
        ) from exc
    return _build_spec(raw)
