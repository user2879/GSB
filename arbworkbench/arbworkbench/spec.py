"""Domain models and strict specification loading / validation.

The loader performs *whole-input rejection*: the first structural violation
raises :class:`SpecError` (code ``E_SPEC``) with a JSON-pointer-like location
identifying the offending element, and nothing is partially accepted.

Validation split
----------------
* E_SPEC : shape, types, uniqueness, references, supported policy names.
* E_GRID : structurally valid but numerically out-of-grid values that can only
  be detected while arbitration runs (scale range, reversed intervals).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SpecError

POLICY_LATEST = "latest_wins"
POLICY_MAJORITY = "weighted_majority"
POLICY_INTERVAL = "interval_intersection"
SUPPORTED_POLICIES = (POLICY_LATEST, POLICY_MAJORITY, POLICY_INTERVAL)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


@dataclass(frozen=True)
class Observation:
    env_id: str
    key: str
    seq: int
    timestamp_ms: int
    value: object          # numeric scalar or [lo, hi] interval
    obs_id: str            # stable identity: env/key/seq

    @property
    def is_interval(self) -> bool:
        return isinstance(self.value, list)


@dataclass
class Environment:
    env_id: str
    weight: int
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True)
class ViewPolicy:
    key: str
    policy: str


@dataclass
class View:
    view_id: str
    env_ids: tuple[str, ...]
    policies: dict[str, str]            # key -> policy name


@dataclass
class Specification:
    scale_exponent: int
    environments: dict[str, Environment]
    views: dict[str, View]

    @property
    def env_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.environments))

    def all_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for env in self.environments.values():
            for obs in env.observations:
                keys.add(obs.key)
        for view in self.views.values():
            keys.update(view.policies)
        return tuple(sorted(keys))


def load_json(path: str | Path) -> dict:
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise SpecError(f"input file not found: {p}", location=str(p)) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "JSON document is malformed",
            location=f"{p}:{exc.lineno}:{exc.colno}",
            detail={"reason": exc.msg},
        ) from exc
    if not isinstance(data, dict):
        raise SpecError("top-level JSON node must be an object", location="$")
    return data


def _validate_observation(raw: dict, env_id: str, key: str, seen_seq: set[int]) -> Observation:
    base = f"environments[{env_id}].observations[{key}]"
    if not isinstance(raw, dict):
        raise SpecError("observation must be an object", location=base)
    for field_name in ("seq", "timestamp_ms", "value"):
        if field_name not in raw:
            raise SpecError(f"observation missing field '{field_name}'", location=f"{base}.<{field_name}>")
    seq = raw["seq"]
    if not _is_int(seq):
        raise SpecError("observation seq must be an integer", location=f"{base}.seq")
    if seq in seen_seq:
        raise SpecError(
            f"duplicate observation seq {seq} for env '{env_id}' key '{key}'",
            location=f"{base}.seq",
        )
    seen_seq.add(seq)
    ts = raw["timestamp_ms"]
    if not _is_int(ts) or ts < 0:
        raise SpecError(
            "timestamp_ms must be a non-negative integer", location=f"{base}.timestamp_ms"
        )
    value = raw["value"]
    if isinstance(value, list):
        if len(value) != 2 or not all(_is_number(v) for v in value):
            raise SpecError(
                "interval value must be a two-element numeric array [lo, hi]",
                location=f"{base}.value",
            )
    elif not _is_number(value):
        raise SpecError("observation value must be numeric or [lo, hi]", location=f"{base}.value")
    return Observation(env_id=env_id, key=key, seq=seq, timestamp_ms=ts, value=value,
                       obs_id=f"{env_id}/{key}#{seq}")


def parse_specification(env_doc: dict, view_doc: dict) -> Specification:
    # ---- scale ----------------------------------------------------------
    scale_exponent = env_doc.get("scale_exponent", 0)
    if not _is_int(scale_exponent):
        raise SpecError("scale_exponent must be an integer", location="scale_exponent")
    # Range itself is an E_GRID concern (valid type, out of grid).

    # ---- environments ---------------------------------------------------
    raw_envs = env_doc.get("environments")
    if not isinstance(raw_envs, list):
        raise SpecError("environments must be an array", location="environments")
    environments: dict[str, Environment] = {}
    for idx, raw_env in enumerate(raw_envs):
        loc = f"environments[{idx}]"
        if not isinstance(raw_env, dict):
            raise SpecError("environment entry must be an object", location=loc)
        env_id = raw_env.get("env_id")
        if not isinstance(env_id, str) or not env_id:
            raise SpecError("environment env_id must be a non-empty string", location=f"{loc}.env_id")
        if env_id in environments:
            raise SpecError(f"duplicate environment id '{env_id}'", location=f"{loc}.env_id")
        weight = raw_env.get("weight", 1)
        if not _is_int(weight) or weight < 0:
            raise SpecError(
                "environment weight must be a non-negative integer", location=f"{loc}.weight"
            )
        raw_obs = raw_env.get("observations")
        if not isinstance(raw_obs, dict):
            raise SpecError("observations must be an object keyed by config key", location=f"{loc}.observations")
        env = Environment(env_id=env_id, weight=weight)
        for key, obs_list in raw_obs.items():
            if not isinstance(obs_list, list) or not obs_list:
                raise SpecError(
                    f"observations for key '{key}' must be a non-empty array",
                    location=f"{loc}.observations[{key}]",
                )
            seen_seq: set[int] = set()
            for raw_obs_entry in obs_list:
                env.observations.append(
                    _validate_observation(raw_obs_entry, env_id, key, seen_seq)
                )
        environments[env_id] = env

    # ---- views ----------------------------------------------------------
    raw_views = view_doc.get("views")
    if not isinstance(raw_views, list):
        raise SpecError("views must be an array", location="views")
    views: dict[str, View] = {}
    for idx, raw_view in enumerate(raw_views):
        loc = f"views[{idx}]"
        if not isinstance(raw_view, dict):
            raise SpecError("view entry must be an object", location=loc)
        view_id = raw_view.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise SpecError("view_id must be a non-empty string", location=f"{loc}.view_id")
        if view_id in views:
            raise SpecError(f"duplicate view id '{view_id}'", location=f"{loc}.view_id")
        env_ids = raw_view.get("env_ids")
        if not isinstance(env_ids, list) or not env_ids:
            raise SpecError("view env_ids must be a non-empty array", location=f"{loc}.env_ids")
        resolved: list[str] = []
        for ref in env_ids:
            if not isinstance(ref, str):
                raise SpecError("env_ids entries must be strings", location=f"{loc}.env_ids")
            if ref not in environments:
                raise SpecError(
                    f"view '{view_id}' references unknown environment '{ref}'",
                    location=f"{loc}.env_ids[{ref}]",
                )
            resolved.append(ref)
        if len(set(resolved)) != len(resolved):
            raise SpecError(f"view '{view_id}' lists a duplicated environment", location=f"{loc}.env_ids")
        raw_policies = raw_view.get("policies")
        if not isinstance(raw_policies, dict) or not raw_policies:
            raise SpecError("view policies must be a non-empty object", location=f"{loc}.policies")
        policies: dict[str, str] = {}
        for key, policy in raw_policies.items():
            if policy not in SUPPORTED_POLICIES:
                raise SpecError(
                    f"unsupported policy '{policy}' for key '{key}'",
                    location=f"{loc}.policies[{key}]",
                    detail={"supported": list(SUPPORTED_POLICIES)},
                )
            policies[key] = policy
        views[view_id] = View(view_id=view_id, env_ids=tuple(sorted(resolved)), policies=policies)

    return Specification(
        scale_exponent=scale_exponent,
        environments=environments,
        views=views,
    )


def load_specification(env_path: str | Path, view_path: str | Path) -> Specification:
    env_doc = load_json(env_path)
    view_doc = load_json(view_path)
    return parse_specification(env_doc, view_doc)
