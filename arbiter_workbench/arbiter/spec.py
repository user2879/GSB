"""Local JSON declaration loading with strong validation.

Declaration files (all integer timestamps/revisions/leases/tolerances):

environments.json -> {"environments": [
    {"env_id", "host" (optional, must equal 127.0.0.1), "port",
     "lease_capacity", "seed", "weight"}]}
instances.json    -> {"instances": [
    {"instance_id", "env_ref", "obs_count", "base_ts_ms"}]}
views.json        -> {"views": [
    {"view_id", "envs": [...env ids...],
     "policies": {key: "latest" | "weighted_majority" | "interval_midpoint"},
     "tolerance"}]}

The first invalid field/object is reported as E_SPEC.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ErrorContext, SpecError

POLICIES = frozenset({"latest", "weighted_majority", "interval_midpoint"})
MIN_PORT, MAX_PORT = 1024, 65535


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    host: str
    port: int
    lease_capacity: int
    seed: int
    weight: int


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    env_ref: str
    obs_count: int
    base_ts_ms: int


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    envs: tuple[str, ...]
    policies: dict[str, str]
    tolerance: int


@dataclass
class Declarations:
    environments: dict[str, EnvSpec]
    instances: list[InstanceSpec]
    views: dict[str, ViewSpec]
    keys: tuple[str, ...] = field(default_factory=tuple)

    def policy_for(self, view_id: str, key: str) -> str:
        view = self.views[view_id]
        if key in view.policies:
            return view.policies[key]
        if "*" in view.policies:
            return view.policies["*"]
        raise SpecError(
            "view declares no policy for key",
            ErrorContext(view_id, f"policies.{key}"),
        )


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _fail(obj_id: str, field_name: str, detail: str) -> None:
    raise SpecError(
        "illegal declaration field",
        ErrorContext(obj_id, field_name, detail),
    )


def _require(obj: dict, field_name: str, obj_id: str, predicate, detail: str):
    if field_name not in obj:
        _fail(obj_id, field_name, "missing required field")
    if not predicate(obj[field_name]):
        _fail(obj_id, field_name, detail)
    return obj[field_name]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SpecError(
            "declaration file not found", ErrorContext(str(path), "", "")
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            "declaration file is not valid JSON",
            ErrorContext(str(path), "", str(exc)),
        ) from exc
    if not isinstance(data, dict):
        raise SpecError(
            "declaration root must be an object", ErrorContext(str(path))
        )
    return data


def _validate_env(raw: dict) -> EnvSpec:
    if not isinstance(raw, dict):
        _fail("<environment>", "<element>", "environment must be an object")
    env_id = _require(raw, "env_id", "<environment>", _is_str, "non-empty string required")
    host = raw.get("host", "127.0.0.1")
    if not _is_str(host) or host != "127.0.0.1":
        _fail(env_id, "host", "only loopback host 127.0.0.1 is allowed")
    port = _require(raw, "port", env_id, _is_int, "integer port required")
    if not (MIN_PORT <= port <= MAX_PORT):
        _fail(env_id, "port", f"port must be within [{MIN_PORT}, {MAX_PORT}]")
    lease = _require(
        raw, "lease_capacity", env_id, _is_int, "integer lease capacity required"
    )
    if lease <= 0:
        _fail(env_id, "lease_capacity", "must be a positive integer")
    seed = _require(raw, "seed", env_id, _is_int, "integer seed required")
    if seed < 0:
        _fail(env_id, "seed", "seed must be a non-negative integer")
    weight = _require(raw, "weight", env_id, _is_int, "integer weight required")
    if weight <= 0:
        _fail(env_id, "weight", "weight must be a positive integer")
    return EnvSpec(env_id, host, port, lease, seed, weight)


def _validate_instance(raw: dict, env_ids: set[str]) -> InstanceSpec:
    if not isinstance(raw, dict):
        _fail("<instance>", "<element>", "instance must be an object")
    instance_id = _require(
        raw, "instance_id", "<instance>", _is_str, "non-empty string required"
    )
    env_ref = _require(
        raw, "env_ref", instance_id, _is_str, "non-empty string required"
    )
    if env_ref not in env_ids:
        _fail(instance_id, "env_ref", f"unknown environment reference {env_ref!r}")
    obs_count = _require(
        raw, "obs_count", instance_id, _is_int, "integer obs_count required"
    )
    if not (1 <= obs_count <= 10000):
        _fail(instance_id, "obs_count", "must be within [1, 10000]")
    base_ts = _require(
        raw, "base_ts_ms", instance_id, _is_int, "integer base_ts_ms required"
    )
    if base_ts < 0:
        _fail(instance_id, "base_ts_ms", "must be a non-negative integer")
    return InstanceSpec(instance_id, env_ref, obs_count, base_ts)


def _validate_view(raw: dict, env_ids: set[str], keys: set[str]) -> ViewSpec:
    if not isinstance(raw, dict):
        _fail("<view>", "<element>", "view must be an object")
    view_id = _require(raw, "view_id", "<view>", _is_str, "non-empty string required")
    envs_raw = _require(
        raw, "envs", view_id, lambda v: isinstance(v, list) and len(v) > 0,
        "non-empty list of env ids required",
    )
    envs: list[str] = []
    for env_id in envs_raw:
        if not _is_str(env_id) or env_id not in env_ids:
            _fail(view_id, "envs", f"unknown environment reference {env_id!r}")
        if env_id in envs:
            _fail(view_id, "envs", f"duplicate environment {env_id!r}")
        envs.append(env_id)
    policies = _require(
        raw, "policies", view_id, lambda v: isinstance(v, dict) and len(v) > 0,
        "non-empty policy map required",
    )
    for key, policy in policies.items():
        if not _is_str(key):
            _fail(view_id, "policies", "policy keys must be non-empty strings")
        if key != "*" and key not in keys:
            _fail(view_id, f"policies.{key}", "unknown configuration key")
        if not _is_str(policy) or policy not in POLICIES:
            _fail(
                view_id, f"policies.{key}",
                f"policy must be one of {sorted(POLICIES)}",
            )
    tolerance = _require(
        raw, "tolerance", view_id, _is_int, "integer tolerance required"
    )
    if tolerance < 0:
        _fail(view_id, "tolerance", "tolerance must be a non-negative integer")
    return ViewSpec(view_id, tuple(envs), dict(policies), tolerance)


def load_declarations(decl_dir: str | Path, keys: tuple[str, ...]) -> Declarations:
    """Load and cross-validate the three declaration files."""
    decl_dir = Path(decl_dir)
    env_data = _load_json(decl_dir / "environments.json")
    inst_data = _load_json(decl_dir / "instances.json")
    view_data = _load_json(decl_dir / "views.json")

    env_list_raw = env_data.get("environments")
    if not isinstance(env_list_raw, list) or not env_list_raw:
        raise SpecError(
            "environments must be a non-empty list",
            ErrorContext("environments.json", "environments"),
        )
    environments: dict[str, EnvSpec] = {}
    seen_ports: set[int] = set()
    for raw in env_list_raw:
        env = _validate_env(raw)
        if env.env_id in environments:
            _fail(env.env_id, "env_id", "duplicate environment id")
        if env.port in seen_ports:
            _fail(env.env_id, "port", f"duplicate port {env.port}")
        seen_ports.add(env.port)
        environments[env.env_id] = env

    inst_list_raw = inst_data.get("instances")
    if not isinstance(inst_list_raw, list) or not inst_list_raw:
        raise SpecError(
            "instances must be a non-empty list",
            ErrorContext("instances.json", "instances"),
        )
    instances: list[InstanceSpec] = []
    seen_instances: set[str] = set()
    for raw in inst_list_raw:
        inst = _validate_instance(raw, set(environments))
        if inst.instance_id in seen_instances:
            _fail(inst.instance_id, "instance_id", "duplicate instance id")
        seen_instances.add(inst.instance_id)
        instances.append(inst)

    view_list_raw = view_data.get("views")
    if not isinstance(view_list_raw, list) or not view_list_raw:
        raise SpecError(
            "views must be a non-empty list",
            ErrorContext("views.json", "views"),
        )
    views: dict[str, ViewSpec] = {}
    key_set = set(keys)
    for raw in view_list_raw:
        view = _validate_view(raw, set(environments), key_set)
        if view.view_id in views:
            _fail(view.view_id, "view_id", "duplicate view id")
        views[view.view_id] = view

    return Declarations(environments, instances, views, tuple(keys))
