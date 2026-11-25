from __future__ import annotations

from pathlib import Path
from typing import Any

from .algorithms import PathResult
from .allocation import MinuteAllocation, allocate_minute
from .canonical import sha256_hex, write_json_atomic
from .errors import DivergenceError
from .spec import Spec


DEMAND_COMPARE_FIELDS = (
    "request_count",
    "per_replica_capacity",
    "unclamped_replicas",
    "clamped_replicas",
    "baseline_replicas",
    "replica_delta",
    "scaling_action",
    "window_length",
    "window_size",
    "window_request_sum",
    "window_demand_sum",
    "ring_index",
    "estimation_rule_version",
)

ALLOCATION_COMPARE_FIELDS = (
    "clamped_replicas",
    "quota_capped_replicas",
    "global_remaining_before",
    "global_remaining_after",
    "service_quota_remaining_after",
    "concurrency_quota_remaining_after",
    "service_replica_quota",
    "allocated_replicas",
    "shortfall_replicas",
    "lease_count",
    "lease_ids",
)


def path_input_view(spec: Spec, path_result: PathResult) -> dict[str, Any]:
    return {
        "path": path_result.path,
        "spec_digest": sha256_hex(spec.canonical_bytes()),
        "scaling_factor_permille": spec.scaling_factor_permille,
        "algorithm": spec.canonical_dict()["algorithm"],
        "budget": spec.canonical_dict()["budget"],
        "services": [
            {
                "service_id": service.service_id,
                "requests_per_minute": list(service.requests_per_minute),
                "concurrency_per_replica": service.concurrency_per_replica,
                "min_replicas": service.min_replicas,
                "max_replicas": service.max_replicas,
            }
            for service in spec.services
        ],
    }


def _raise_divergence(
    spec: Spec,
    full: PathResult,
    incremental: PathResult,
    service_id: str,
    minute: int,
    field: str,
    full_value: Any,
    incremental_value: Any,
    *,
    artifact_dir: str | Path | None,
) -> None:
    views = {
        "full": path_input_view(spec, full),
        "incremental": path_input_view(spec, incremental),
    }
    artifact_path: str | None = None
    if artifact_dir is not None:
        path = Path(artifact_dir) / "e_diverge_input_views.json"
        write_json_atomic(
            path,
            {
                "code": "E_DIVERGE",
                "service_id": service_id,
                "minute": minute,
                "field": field,
                "full_value": full_value,
                "incremental_value": incremental_value,
                "input_views": views,
            },
        )
        artifact_path = str(path)
    raise DivergenceError(
        "algorithm paths disagree at "
        f"service={service_id!r}, minute={minute}, field={field!r}",
        field=f"services.{service_id}.minutes[{minute}].{field}",
        details={
            "service_id": service_id,
            "minute": minute,
            "field": field,
            "full_value": full_value,
            "incremental_value": incremental_value,
            "input_views": views,
            "input_view_artifact": artifact_path,
        },
    )


def verify_minute(
    spec: Spec,
    full: PathResult,
    incremental: PathResult,
    minute: int,
    *,
    artifact_dir: str | Path | None = None,
) -> tuple[list[MinuteAllocation], list[MinuteAllocation]]:
    for service_id in spec.service_ids:
        full_minutes = full.minutes[service_id]
        incremental_minutes = incremental.minutes[service_id]
        if minute >= len(full_minutes) or minute >= len(incremental_minutes):
            raise DivergenceError(
                "path minute is missing",
                field=f"services.{service_id}.minutes[{minute}]",
                details={"service_id": service_id, "minute": minute},
            )

        full_item = full_minutes[minute]
        incremental_item = incremental_minutes[minute]
        full_dict = full_item.comparable_dict()
        incremental_dict = incremental_item.comparable_dict()
        for field in DEMAND_COMPARE_FIELDS:
            if full_dict[field] != incremental_dict[field]:
                _raise_divergence(
                    spec,
                    full,
                    incremental,
                    service_id,
                    minute,
                    field,
                    full_dict[field],
                    incremental_dict[field],
                    artifact_dir=artifact_dir,
                )

    full_estimates = {
        service_id: full.minutes[service_id][minute]
        for service_id in spec.service_ids
    }
    incremental_estimates = {
        service_id: incremental.minutes[service_id][minute]
        for service_id in spec.service_ids
    }
    full_allocations = allocate_minute(spec, minute, full_estimates)
    incremental_allocations = allocate_minute(
        spec, minute, incremental_estimates
    )
    for full_allocation, incremental_allocation in zip(
        full_allocations, incremental_allocations
    ):
        service_id = full_allocation.service_id
        full_dict = full_allocation.comparable_dict()
        incremental_dict = incremental_allocation.comparable_dict()
        for field in ALLOCATION_COMPARE_FIELDS:
            if full_dict[field] != incremental_dict[field]:
                _raise_divergence(
                    spec,
                    full,
                    incremental,
                    service_id,
                    minute,
                    f"allocation.{field}",
                    full_dict[field],
                    incremental_dict[field],
                    artifact_dir=artifact_dir,
                )
    return full_allocations, incremental_allocations


def verify_minute_range(
    spec: Spec,
    full: PathResult,
    incremental: PathResult,
    start_minute: int,
    end_minute: int,
    *,
    artifact_dir: str | Path | None = None,
) -> dict[int, list[MinuteAllocation]]:
    allocations: dict[int, list[MinuteAllocation]] = {}
    for minute in range(start_minute, end_minute):
        full_allocations, _ = verify_minute(
            spec,
            full,
            incremental,
            minute,
            artifact_dir=artifact_dir,
        )
        allocations[minute] = full_allocations
    return allocations


def verify_paths(
    spec: Spec,
    full: PathResult,
    incremental: PathResult,
    *,
    artifact_dir: str | Path | None = None,
) -> tuple[dict[int, list[MinuteAllocation]], dict[int, list[MinuteAllocation]]]:
    full_allocations = verify_minute_range(
        spec,
        full,
        incremental,
        0,
        spec.minute_count,
        artifact_dir=artifact_dir,
    )
    incremental_allocations = full_allocations
    return full_allocations, incremental_allocations
