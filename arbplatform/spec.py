"""Environment / instance / view declaration loading and strong validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SpecError

POLICIES = ("latest_revision", "weighted_majority", "interval_midpoint")


@dataclass(frozen=True)
class EnvironmentSpec:
    env_id: str
    host: str
    port: int
    lease_capacity: int
    seed: int
    weight: int = 1


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    env_ref: str
    seq_low: int
    seq_high: int
    time_base_ms: int


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    env_refs: tuple[str, ...]
    policies: dict[str, str]  # config key -> policy
    tolerance: int


@dataclass(frozen=True)
class Manifest:
    environments: tuple[EnvironmentSpec, ...]
    instances: tuple[InstanceSpec, ...]
    views: tuple[ViewSpec, ...]
    scale_power: int
    global_concurrency: int
    rejected_ratio_threshold: float = 0.5

    def env(self, env_id: str) -> EnvironmentSpec:
        for env in self.environments:
            if env.env_id == env_id:
                return env
        raise SpecError(f"unknown environment {env_id!r}", field=f"envs.{env_id}")

    def view(self, view_id: str) -> ViewSpec:
        for view in self.views:
            if view.view_id == view_id:
                return view
        raise SpecError(f"unknown view {view_id!r}", field=f"views.{view_id}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(obj: Any, key: str, path: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise SpecError(f"missing required field {key!r}", field=f"{path}.{key}")
    return obj[key]


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"manifest not found: {path}", field="<file>") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc.msg}", field=f"<line {exc.lineno}>") from exc
    return parse_manifest(data)


def parse_manifest(data: Any) -> Manifest:
    if not isinstance(data, dict):
        raise SpecError("manifest root must be an object", field="<root>")

    env_raw = _require(data, "environments", "$")
    inst_raw = _require(data, "instances", "$")
    view_raw = _require(data, "views", "$")
    scale_power = _require(data, "scale_power", "$")
    global_concurrency = _require(data, "global_concurrency", "$")

    if not _is_int(scale_power) or scale_power not in (0, 1, 2, 3):
        raise SpecError(
            "scale_power must be an integer in 0..3", field="$.scale_power"
        )
    if not _is_int(global_concurrency) or global_concurrency <= 0:
        raise SpecError(
            "global_concurrency must be a positive integer",
            field="$.global_concurrency",
        )
    threshold = data.get("rejected_ratio_threshold", 0.5)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
            or not 0.0 <= float(threshold) <= 1.0:
        raise SpecError(
            "rejected_ratio_threshold must be a number in [0,1]",
            field="$.rejected_ratio_threshold",
        )

    environments = _parse_environments(env_raw)
    instances = _parse_instances(inst_raw, {e.env_id for e in environments})
    views = _parse_views(view_raw, {e.env_id for e in environments})
    return Manifest(
        environments=environments,
        instances=instances,
        views=views,
        scale_power=scale_power,
        global_concurrency=global_concurrency,
        rejected_ratio_threshold=float(threshold),
    )


def _parse_environments(raw: Any) -> tuple[EnvironmentSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SpecError("environments must be a non-empty list", field="$.environments")
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    out: list[EnvironmentSpec] = []
    for idx, item in enumerate(raw):
        path = f"$.environments[{idx}]"
        if not isinstance(item, dict):
            raise SpecError("environment must be an object", field=path)
        env_id = _require(item, "env_id", path)
        if not isinstance(env_id, str) or not env_id:
            raise SpecError("env_id must be a non-empty string", field=f"{path}.env_id")
        if env_id in seen_ids:
            raise SpecError(f"duplicate environment id {env_id!r}", field=f"{path}.env_id")
        seen_ids.add(env_id)

        host = _require(item, "host", path)
        if host != "127.0.0.1":
            raise SpecError("host must be the loopback 127.0.0.1", field=f"{path}.host")
        port = _require(item, "port", path)
        if not _is_int(port) or not (1 <= port <= 65535):
            raise SpecError("port must be an integer in 1..65535", field=f"{path}.port")
        if port in seen_ports:
            raise SpecError(f"duplicate loopback port {port}", field=f"{path}.port")
        seen_ports.add(port)

        lease = _require(item, "lease_capacity", path)
        if not _is_int(lease) or lease <= 0:
            raise SpecError(
                "lease_capacity must be a positive integer",
                field=f"{path}.lease_capacity",
            )
        seed = _require(item, "seed", path)
        if not _is_int(seed):
            raise SpecError("seed must be an integer", field=f"{path}.seed")
        weight = item.get("weight", 1)
        if not _is_int(weight) or weight <= 0:
            raise SpecError("weight must be a positive integer", field=f"{path}.weight")
        out.append(EnvironmentSpec(env_id, host, port, lease, seed, weight))
    return tuple(out)


def _parse_instances(raw: Any, env_ids: set[str]) -> tuple[InstanceSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SpecError("instances must be a non-empty list", field="$.instances")
    seen: set[str] = set()
    out: list[InstanceSpec] = []
    for idx, item in enumerate(raw):
        path = f"$.instances[{idx}]"
        if not isinstance(item, dict):
            raise SpecError("instance must be an object", field=path)
        instance_id = _require(item, "instance_id", path)
        if not isinstance(instance_id, str) or not instance_id:
            raise SpecError(
                "instance_id must be a non-empty string", field=f"{path}.instance_id"
            )
        if instance_id in seen:
            raise SpecError(
                f"duplicate instance id {instance_id!r}", field=f"{path}.instance_id"
            )
        seen.add(instance_id)

        env_ref = _require(item, "env_ref", path)
        if not isinstance(env_ref, str) or env_ref not in env_ids:
            raise SpecError(
                f"env_ref {env_ref!r} does not reference an existing environment",
                field=f"{path}.env_ref",
            )
        seq_low = _require(item, "seq_low", path)
        seq_high = _require(item, "seq_high", path)
        if not _is_int(seq_low):
            raise SpecError("seq_low must be an integer", field=f"{path}.seq_low")
        if not _is_int(seq_high):
            raise SpecError("seq_high must be an integer", field=f"{path}.seq_high")
        if seq_low > seq_high:
            raise SpecError(
                "seq_low must be <= seq_high", field=f"{path}.seq_low"
            )
        time_base = _require(item, "time_base_ms", path)
        if not _is_int(time_base):
            raise SpecError(
                "time_base_ms must be an integer", field=f"{path}.time_base_ms"
            )
        out.append(InstanceSpec(instance_id, env_ref, seq_low, seq_high, time_base))
    return tuple(out)


def _parse_views(raw: Any, env_ids: set[str]) -> tuple[ViewSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SpecError("views must be a non-empty list", field="$.views")
    seen: set[str] = set()
    out: list[ViewSpec] = []
    for idx, item in enumerate(raw):
        path = f"$.views[{idx}]"
        if not isinstance(item, dict):
            raise SpecError("view must be an object", field=path)
        view_id = _require(item, "view_id", path)
        if not isinstance(view_id, str) or not view_id:
            raise SpecError(
                "view_id must be a non-empty string", field=f"{path}.view_id"
            )
        if view_id in seen:
            raise SpecError(
                f"duplicate view id {view_id!r}", field=f"{path}.view_id"
            )
        seen.add(view_id)

        refs = _require(item, "env_refs", path)
        if not isinstance(refs, list) or not refs:
            raise SpecError(
                "env_refs must be a non-empty list", field=f"{path}.env_refs"
            )
        ref_tuple: list[str] = []
        for ref in refs:
            if not isinstance(ref, str) or ref not in env_ids:
                raise SpecError(
                    f"view env_ref {ref!r} does not exist", field=f"{path}.env_refs"
                )
            if ref not in ref_tuple:
                ref_tuple.append(ref)

        policies = _require(item, "policies", path)
        if not isinstance(policies, dict) or not policies:
            raise SpecError(
                "policies must be a non-empty key->policy object",
                field=f"{path}.policies",
            )
        policy_map: dict[str, str] = {}
        for key in sorted(policies):
            policy = policies[key]
            if not isinstance(key, str) or not key:
                raise SpecError(
                    "policy config key must be a non-empty string",
                    field=f"{path}.policies",
                )
            if policy not in POLICIES:
                raise SpecError(
                    f"policy {policy!r} not in {POLICIES}",
                    field=f"{path}.policies.{key}",
                )
            policy_map[key] = policy

        tolerance = _require(item, "tolerance", path)
        if not _is_int(tolerance) or tolerance < 0:
            raise SpecError(
                "tolerance must be a non-negative integer", field=f"{path}.tolerance"
            )
        out.append(
            ViewSpec(view_id, tuple(sorted(ref_tuple)), policy_map, tolerance)
        )
    return tuple(out)
