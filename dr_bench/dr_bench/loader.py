"""Strong, all-or-nothing validation and loading of JSON scene files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import SceneError
from .models import (
    OBSERVATIONS,
    ROLES,
    ROLE_PRIMARY,
    STRATEGIES,
    InstanceConfig,
    Observation,
    Scene,
    ServiceConfig,
)


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_scene(path: str | Path) -> Scene:
    scene_path = Path(path)
    try:
        raw = scene_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneError(f"cannot read scene file: {exc}", str(scene_path)) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SceneError(
            f"invalid JSON: {exc.msg}",
            str(scene_path),
            {"line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_scene(payload, source=str(scene_path))


def parse_scene(payload: Any, source: str = "<memory>") -> Scene:
    errors: list[dict[str, str]] = []

    def fail(path: str, message: str) -> None:
        errors.append({"path": path, "message": message})

    if not isinstance(payload, dict):
        raise SceneError("scene root must be a JSON object", source)

    scene_id = payload.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        fail("scene_id", "must be a non-empty string")
    scene_version = payload.get("scene_version")
    if not _is_nonneg_int(scene_version):
        fail("scene_version", "must be a non-negative integer")

    services_raw = payload.get("services")
    instances_raw = payload.get("instances")
    observations_raw = payload.get("observations")
    if not isinstance(services_raw, list):
        fail("services", "must be a list")
        services_raw = []
    if not isinstance(instances_raw, list):
        fail("instances", "must be a list")
        instances_raw = []
    if not isinstance(observations_raw, list):
        fail("observations", "must be a list")
        observations_raw = []

    services: dict[str, ServiceConfig] = {}
    seen_services: set[str] = set()
    for index, item in enumerate(services_raw):
        base = f"services[{index}]"
        if not isinstance(item, dict):
            fail(base, "service entry must be an object")
            continue
        service_id = item.get("service_id")
        if not isinstance(service_id, str) or not service_id:
            fail(f"{base}.service_id", "must be a non-empty string")
            continue
        if service_id in seen_services:
            fail(f"{base}.service_id", f"duplicate service id {service_id!r}")
            continue
        seen_services.add(service_id)
        strategy = item.get("strategy")
        if strategy not in STRATEGIES:
            fail(f"{base}.strategy", f"must be one of {list(STRATEGIES)}")
        for field_name in (
            "cooldown_ms",
            "min_healthy_replicas",
            "success_threshold",
            "failure_threshold",
        ):
            if not _is_nonneg_int(item.get(field_name)):
                fail(f"{base}.{field_name}", "must be a non-negative integer")
        if errors and any(error["path"].startswith(base) for error in errors):
            continue
        services[service_id] = ServiceConfig(
            service_id=service_id,
            strategy=strategy,
            cooldown_ms=item["cooldown_ms"],
            min_healthy_replicas=item["min_healthy_replicas"],
            success_threshold=item["success_threshold"],
            failure_threshold=item["failure_threshold"],
        )

    instances: dict[str, InstanceConfig] = {}
    seen_instances: set[str] = set()
    service_primary: dict[str, list[str]] = {}
    service_members: dict[str, list[str]] = {}
    for index, item in enumerate(instances_raw):
        base = f"instances[{index}]"
        if not isinstance(item, dict):
            fail(base, "instance entry must be an object")
            continue
        service_id = item.get("service_id")
        instance_id = item.get("instance_id")
        role = item.get("role")
        version = item.get("version")
        if not isinstance(service_id, str) or not service_id:
            fail(f"{base}.service_id", "must be a non-empty string")
        elif service_id not in services:
            fail(f"{base}.service_id", f"unknown service id {service_id!r}")
        if not isinstance(instance_id, str) or not instance_id:
            fail(f"{base}.instance_id", "must be a non-empty string")
        elif instance_id in seen_instances:
            fail(f"{base}.instance_id", f"duplicate instance id {instance_id!r}")
        else:
            seen_instances.add(instance_id)
        if role not in ROLES:
            fail(f"{base}.role", f"must be one of {list(ROLES)}")
        if not _is_nonneg_int(version):
            fail(f"{base}.version", "must be a non-negative integer")
        if service_id in services and isinstance(instance_id, str) and instance_id:
            service_members.setdefault(service_id, []).append(instance_id)
            if role == ROLE_PRIMARY:
                service_primary.setdefault(service_id, []).append(instance_id)

    for service_id in sorted(services):
        members = sorted(service_members.get(service_id, []))
        if not members:
            fail(f"services[{service_id}]", "service must contain at least one instance")
            continue
        primaries = sorted(service_primary.get(service_id, []))
        if len(primaries) == 0:
            fail(f"services[{service_id}]", "exactly one primary instance is required")
        elif len(primaries) > 1:
            fail(
                f"services[{service_id}]",
                f"at most one primary instance is allowed, got {primaries}",
            )
        valid_entries = [
            entry
            for entry in instances_raw
            if isinstance(entry, dict) and entry.get("service_id") == service_id
        ]
        if len(valid_entries) != len(members):
            continue
        for entry in valid_entries:
            instance_id = entry["instance_id"]
            role = entry["role"]
            version = entry["version"]
            if isinstance(instance_id, str) and role in ROLES and _is_nonneg_int(version):
                instances[instance_id] = InstanceConfig(
                    service_id=service_id,
                    instance_id=instance_id,
                    role=role,
                    version=version,
                )

    observations: list[Observation] = []
    per_instance_rounds: dict[str, set[int]] = {}
    per_instance_times: dict[str, list[tuple[int, int]]] = {}
    for index, item in enumerate(observations_raw):
        base = f"observations[{index}]"
        if not isinstance(item, dict):
            fail(base, "observation entry must be an object")
            continue
        service_id = item.get("service_id")
        instance_id = item.get("instance_id")
        round_index = item.get("round_index")
        observed_at_ms = item.get("observed_at_ms")
        result = item.get("result")
        if not isinstance(service_id, str) or service_id not in services:
            fail(f"{base}.service_id", "must reference a declared service id")
        if not isinstance(instance_id, str) or instance_id not in instances:
            fail(f"{base}.instance_id", "must reference a declared instance id")
        elif (
            isinstance(service_id, str)
            and service_id in services
            and instances[instance_id].service_id != service_id
        ):
            fail(
                f"{base}.service_id",
                f"instance {instance_id!r} does not belong to service {service_id!r}",
            )
        if not _is_nonneg_int(round_index):
            fail(f"{base}.round_index", "must be a non-negative integer")
        if not _is_int(observed_at_ms):
            fail(f"{base}.observed_at_ms", "must be an integer millisecond timestamp")
        if result not in OBSERVATIONS:
            fail(f"{base}.result", f"must be one of {list(OBSERVATIONS)}")
        if any(error["path"].startswith(base) for error in errors):
            continue
        if round_index in per_instance_rounds.setdefault(instance_id, set()):
            fail(f"{base}.round_index", f"duplicate round for instance {instance_id!r}")
        else:
            per_instance_rounds[instance_id].add(round_index)
        times = per_instance_times.setdefault(instance_id, [])
        times.append((round_index, observed_at_ms))
        ordered = sorted(times)
        for earlier, later in zip(ordered, ordered[1:]):
            if later[1] < earlier[1]:
                fail(
                    f"{base}.observed_at_ms",
                    "timestamps must be non-decreasing per instance",
                )
                break
        observations.append(
            Observation(
                service_id=service_id,
                instance_id=instance_id,
                round_index=round_index,
                observed_at_ms=observed_at_ms,
                result=result,
            )
        )

    if errors:
        raise SceneError(
            f"scene validation failed with {len(errors)} error(s)",
            source,
            {"errors": errors},
        )

    by_instance_round: dict[str, dict[int, Observation]] = {}
    all_rounds: set[int] = set()
    for observation in observations:
        by_instance_round.setdefault(observation.instance_id, {})[
            observation.round_index
        ] = observation
        all_rounds.add(observation.round_index)
    rounds = tuple(sorted(all_rounds))
    ordered_members = {
        service_id: sorted(member_ids) for service_id, member_ids in service_members.items()
    }
    return Scene(
        scene_id=scene_id,
        scene_version=scene_version,
        services=services,
        instances=instances,
        observations=observations,
        by_instance_round=by_instance_round,
        rounds=rounds,
        service_members=ordered_members,
    )
