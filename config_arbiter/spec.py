from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import RefError, SpecError
from .util import decimal_text, parse_decimal

SUPPORTED_SOURCE_TYPES = ("baseline", "override", "runtime")
SUPPORTED_STRATEGIES = ("latest", "weighted_majority", "interval_midpoint")
ENV_SCHEMA = "config-arbiter.environments/v1"
VIEW_SCHEMA = "config-arbiter.views/v1"


@dataclass(frozen=True)
class Observation:
    key: str
    value: Decimal
    value_text: str
    observed_seq: int
    captured_at_ms: int


@dataclass(frozen=True)
class Source:
    source_id: str
    source_type: str
    priority_bias: int
    observations: dict[str, Observation]


@dataclass(frozen=True)
class Binding:
    required_source_types: tuple[str, ...]
    min_observed_seq: int | None


@dataclass(frozen=True)
class Environment:
    environment_id: str
    depth: int
    available_sources: tuple[str, ...]
    sources: dict[str, Source]
    bindings: dict[str, Binding]


@dataclass(frozen=True)
class KeyPolicy:
    strategy: str
    tolerance: int


@dataclass(frozen=True)
class View:
    view_id: str
    environments: tuple[str, ...]
    environment_priority: tuple[str, ...]
    environment_weights: dict[str, int]
    key_policies: dict[str, KeyPolicy]


@dataclass(frozen=True)
class Session:
    scale_power: int
    scale_policy_version: str
    arbitration_policy_version: str
    truncation: str
    overflow: str


@dataclass(frozen=True)
class Specs:
    environment_document: dict[str, Any]
    view_document: dict[str, Any]
    environments: dict[str, Environment]
    views: dict[str, View]
    source_type_order: tuple[str, ...]
    environment_set_version: int
    source_priority_version: int
    session: Session


def _spec(path: str, message: str, *, entry: str | None = None, **extra: Any) -> SpecError:
    details: dict[str, Any] = {"path": path}
    if entry is not None:
        details["entry"] = entry
    details.update(extra)
    return SpecError(message, details=details)


def _object(value: Any, path: str, entry: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _spec(path, "expected a JSON object", entry=entry, actual_type=type(value).__name__)
    return value


def _list(value: Any, path: str, entry: str) -> list[Any]:
    if not isinstance(value, list):
        raise _spec(path, "expected a JSON array", entry=entry, actual_type=type(value).__name__)
    return value


def _string(value: Any, path: str, entry: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise _spec(path, "expected a non-empty string" if nonempty else "expected a string", entry=entry)
    return value


def _integer(value: Any, path: str, entry: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _spec(path, "expected an integer", entry=entry, actual_type=type(value).__name__)
    if minimum is not None and value < minimum:
        raise _spec(path, f"integer must be >= {minimum}", entry=entry, actual=value)
    return value


def _read_document(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _spec("$", f"cannot load JSON document: {exc}", entry=str(path)) from exc
    if not isinstance(value, dict):
        raise _spec("$", "document root must be a JSON object", entry=str(path))
    return value


def _apply_injections(
    environment_document: dict[str, Any],
    view_document: dict[str, Any],
    *,
    inject_spec_corrupt: bool,
    inject_priority_ref_missing_env: bool,
) -> None:
    if inject_spec_corrupt:
        environments = environment_document.get("environments")
        if isinstance(environments, list) and environments and isinstance(environments[0], dict):
            environments[0]["depth"] = -1
        else:
            environment_document["environments"] = [{"environment_id": "injected", "depth": -1}]
    if inject_priority_ref_missing_env:
        views = view_document.get("views")
        if isinstance(views, list) and views and isinstance(views[0], dict):
            priority = views[0].get("environment_priority")
            if not isinstance(priority, list):
                priority = []
                views[0]["environment_priority"] = priority
            priority.append("__injected_missing_environment__")
        else:
            view_document["views"] = [
                {
                    "view_id": "injected",
                    "environments": [],
                    "environment_priority": ["__injected_missing_environment__"],
                    "key_policies": {},
                }
            ]


def _parse_session(view_document: dict[str, Any]) -> Session:
    entry = "$.session"
    raw = _object(view_document.get("session"), entry, entry)
    power = _integer(raw.get("scale_power"), entry + ".scale_power", entry, minimum=0)
    if power > 9:
        raise _spec(entry + ".scale_power", "scale_power must be between 0 and 9", entry=entry, actual=power)
    scale_version = _string(raw.get("scale_policy_version"), entry + ".scale_policy_version", entry)
    arbitration_version = _string(
        view_document.get("arbitration_policy_version"),
        "$.arbitration_policy_version",
        "$",
    )
    truncation = _string(raw.get("truncation"), entry + ".truncation", entry)
    if truncation != "toward_zero":
        raise _spec(entry + ".truncation", "only toward_zero truncation is supported", entry=entry)
    overflow = _string(raw.get("overflow"), entry + ".overflow", entry)
    if overflow != "saturate_int64":
        raise _spec(entry + ".overflow", "only saturate_int64 overflow is supported", entry=entry)
    return Session(
        scale_power=power,
        scale_policy_version=scale_version,
        arbitration_policy_version=arbitration_version,
        truncation=truncation,
        overflow=overflow,
    )


def _parse_environments(document: dict[str, Any]) -> tuple[dict[str, Environment], tuple[str, ...], int, int]:
    if document.get("schema") != ENV_SCHEMA:
        raise _spec("$.schema", f"schema must be {ENV_SCHEMA!r}", entry="$", actual=document.get("schema"))
    env_version = _integer(document.get("environment_set_version"), "$.environment_set_version", "$", minimum=0)
    priority_version = _integer(document.get("source_priority_version"), "$.source_priority_version", "$", minimum=0)

    raw_order = _list(document.get("source_type_order"), "$.source_type_order", "$")
    source_type_order: list[str] = []
    for index, raw_type in enumerate(raw_order):
        source_type = _string(raw_type, f"$.source_type_order[{index}]", f"source_type_order[{index}]")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise _spec(
                f"$.source_type_order[{index}]",
                "unsupported source type",
                entry=f"source_type_order[{index}]",
                actual=source_type,
            )
        if source_type in source_type_order:
            raise _spec(
                f"$.source_type_order[{index}]",
                "duplicate source type in source_type_order",
                entry=f"source_type_order[{index}]",
            )
        source_type_order.append(source_type)
    if set(source_type_order) != set(SUPPORTED_SOURCE_TYPES):
        raise _spec(
            "$.source_type_order",
            "source_type_order must contain baseline, override and runtime exactly once",
            entry="source_type_order",
            actual=source_type_order,
        )

    raw_environments = _list(document.get("environments"), "$.environments", "$")
    if not raw_environments:
        raise _spec("$.environments", "at least one environment is required", entry="environments")

    environments: dict[str, Environment] = {}
    for env_index, raw_env in enumerate(raw_environments):
        env_path = f"$.environments[{env_index}]"
        env_entry = f"environments[{env_index}]"
        env_obj = _object(raw_env, env_path, env_entry)
        env_id = _string(env_obj.get("environment_id"), env_path + ".environment_id", env_entry)
        if env_id in environments:
            raise _spec(env_path + ".environment_id", "duplicate environment_id", entry=env_entry, actual=env_id)
        depth = _integer(env_obj.get("depth"), env_path + ".depth", env_entry, minimum=0)

        raw_available = _list(env_obj.get("available_sources"), env_path + ".available_sources", env_entry)
        available: list[str] = []
        for source_index, raw_source_type in enumerate(raw_available):
            source_type = _string(
                raw_source_type,
                f"{env_path}.available_sources[{source_index}]",
                env_entry,
            )
            if source_type not in source_type_order:
                raise _spec(
                    f"{env_path}.available_sources[{source_index}]",
                    "source type is not supported by source_type_order",
                    entry=env_entry,
                    actual=source_type,
                )
            if source_type in available:
                raise _spec(
                    f"{env_path}.available_sources[{source_index}]",
                    "duplicate available source type",
                    entry=env_entry,
                    actual=source_type,
                )
            available.append(source_type)

        raw_sources = _list(env_obj.get("sources"), env_path + ".sources", env_entry)
        sources: dict[str, Source] = {}
        all_keys: set[str] = set()
        for source_index, raw_source in enumerate(raw_sources):
            source_path = f"{env_path}.sources[{source_index}]"
            source_entry = f"{env_entry}.sources[{source_index}]"
            source_obj = _object(raw_source, source_path, source_entry)
            source_id = _string(source_obj.get("source_id"), source_path + ".source_id", source_entry)
            if source_id in sources:
                raise _spec(source_path + ".source_id", "duplicate source_id in environment", entry=source_entry)
            source_type = _string(source_obj.get("source_type"), source_path + ".source_type", source_entry)
            if source_type not in source_type_order:
                raise _spec(source_path + ".source_type", "unsupported source_type", entry=source_entry, actual=source_type)
            if source_type not in available:
                raise _spec(
                    source_path + ".source_type",
                    "source_type is not in the environment's available_sources",
                    entry=source_entry,
                    actual=source_type,
                )
            bias = _integer(source_obj.get("priority_bias"), source_path + ".priority_bias", source_entry)
            raw_observations = _list(source_obj.get("observations"), source_path + ".observations", source_entry)
            observations: dict[str, Observation] = {}
            for observation_index, raw_observation in enumerate(raw_observations):
                observation_path = f"{source_path}.observations[{observation_index}]"
                observation_entry = f"{source_entry}.observations[{observation_index}]"
                observation_obj = _object(raw_observation, observation_path, observation_entry)
                key = _string(observation_obj.get("key"), observation_path + ".key", observation_entry)
                if key in observations:
                    raise _spec(
                        observation_path + ".key",
                        "duplicate key in the same environment/source",
                        entry=observation_entry,
                        key=key,
                    )
                parsed = parse_decimal(observation_obj.get("value"))
                if parsed is None:
                    raise _spec(
                        observation_path + ".value",
                        "value must be a finite JSON integer or float (booleans are rejected)",
                        entry=observation_entry,
                        actual_type=type(observation_obj.get("value")).__name__,
                    )
                observed_seq = _integer(
                    observation_obj.get("observed_seq"),
                    observation_path + ".observed_seq",
                    observation_entry,
                    minimum=0,
                )
                captured_at_ms = _integer(
                    observation_obj.get("captured_at_ms"),
                    observation_path + ".captured_at_ms",
                    observation_entry,
                    minimum=0,
                )
                observations[key] = Observation(
                    key=key,
                    value=parsed,
                    value_text=decimal_text(parsed),
                    observed_seq=observed_seq,
                    captured_at_ms=captured_at_ms,
                )
                all_keys.add(key)
            sources[source_id] = Source(
                source_id=source_id,
                source_type=source_type,
                priority_bias=bias,
                observations=observations,
            )

        raw_bindings = env_obj.get("bindings", {})
        bindings_obj = _object(raw_bindings, env_path + ".bindings", env_entry)
        bindings: dict[str, Binding] = {}
        for key in sorted(bindings_obj):
            binding_path = f"{env_path}.bindings.{key}"
            binding_entry = f"{env_entry}.bindings[{key}]"
            if key not in all_keys:
                raise _spec(binding_path, "binding key has no observation in this environment", entry=binding_entry)
            binding_obj = _object(bindings_obj[key], binding_path, binding_entry)
            raw_required = binding_obj.get("required_source_types", list(source_type_order))
            required_values = _list(raw_required, binding_path + ".required_source_types", binding_entry)
            required: list[str] = []
            for required_index, raw_required_type in enumerate(required_values):
                required_type = _string(
                    raw_required_type,
                    f"{binding_path}.required_source_types[{required_index}]",
                    binding_entry,
                )
                if required_type not in source_type_order:
                    raise _spec(
                        f"{binding_path}.required_source_types[{required_index}]",
                        "unsupported source type in binding",
                        entry=binding_entry,
                        actual=required_type,
                    )
                if required_type in required:
                    raise _spec(
                        f"{binding_path}.required_source_types[{required_index}]",
                        "duplicate source type in binding",
                        entry=binding_entry,
                    )
                required.append(required_type)
            raw_min_seq = binding_obj.get("min_observed_seq")
            min_seq = None if raw_min_seq is None else _integer(
                raw_min_seq,
                binding_path + ".min_observed_seq",
                binding_entry,
                minimum=0,
            )
            bindings[key] = Binding(tuple(required), min_seq)

        environments[env_id] = Environment(
            environment_id=env_id,
            depth=depth,
            available_sources=tuple(available),
            sources=sources,
            bindings=bindings,
        )

    return environments, tuple(source_type_order), env_version, priority_version


def _parse_views(
    document: dict[str, Any],
    environments: dict[str, Environment],
) -> tuple[dict[str, View], Session]:
    if document.get("schema") != VIEW_SCHEMA:
        raise _spec("$.schema", f"schema must be {VIEW_SCHEMA!r}", entry="$", actual=document.get("schema"))
    session = _parse_session(document)
    raw_views = _list(document.get("views"), "$.views", "$")
    if not raw_views:
        raise _spec("$.views", "at least one view is required", entry="views")
    views: dict[str, View] = {}
    for view_index, raw_view in enumerate(raw_views):
        view_path = f"$.views[{view_index}]"
        view_entry = f"views[{view_index}]"
        view_obj = _object(raw_view, view_path, view_entry)
        view_id = _string(view_obj.get("view_id"), view_path + ".view_id", view_entry)
        if view_id in views:
            raise _spec(view_path + ".view_id", "duplicate view_id", entry=view_entry, actual=view_id)

        raw_environments = _list(view_obj.get("environments"), view_path + ".environments", view_entry)
        participant_ids: list[str] = []
        for participant_index, raw_participant in enumerate(raw_environments):
            participant = _string(
                raw_participant,
                f"{view_path}.environments[{participant_index}]",
                view_entry,
            )
            if participant not in environments:
                raise _spec(
                    f"{view_path}.environments[{participant_index}]",
                    "view references a missing environment",
                    entry=view_entry,
                    actual=participant,
                )
            if participant in participant_ids:
                raise _spec(
                    f"{view_path}.environments[{participant_index}]",
                    "duplicate environment in view",
                    entry=view_entry,
                    actual=participant,
                )
            participant_ids.append(participant)
        if not participant_ids:
            raise _spec(view_path + ".environments", "view must contain at least one environment", entry=view_entry)

        raw_priority = _list(view_obj.get("environment_priority"), view_path + ".environment_priority", view_entry)
        priority: list[str] = []
        for priority_index, raw_priority_env in enumerate(raw_priority):
            priority_env = _string(
                raw_priority_env,
                f"{view_path}.environment_priority[{priority_index}]",
                view_entry,
            )
            if priority_env in priority:
                raise _spec(
                    f"{view_path}.environment_priority[{priority_index}]",
                    "duplicate environment in environment_priority",
                    entry=view_entry,
                    actual=priority_env,
                )
            priority.append(priority_env)
        unknown_priority = sorted(set(priority) - set(environments))
        if unknown_priority:
            raise RefError(
                "environment priority references a missing environment",
                details={
                    "path": f"{view_path}.environment_priority",
                    "entry": view_entry,
                    "missing_environment_id": unknown_priority[0],
                    "missing_environment_ids": unknown_priority,
                },
            )
        if set(priority) != set(participant_ids):
            raise RefError(
                "environment_priority must reference each participating environment exactly once",
                details={
                    "path": f"{view_path}.environment_priority",
                    "entry": view_entry,
                    "participants": sorted(participant_ids),
                    "priority": priority,
                },
            )

        raw_weights = view_obj.get("environment_weights", {})
        weights_obj = _object(raw_weights, view_path + ".environment_weights", view_entry)
        weights: dict[str, int] = {env_id: 1 for env_id in participant_ids}
        unknown_weights = sorted(set(weights_obj) - set(environments))
        if unknown_weights:
            raise RefError(
                "environment weight references a missing environment",
                details={
                    "path": f"{view_path}.environment_weights",
                    "entry": view_entry,
                    "missing_environment_id": unknown_weights[0],
                },
            )
        for weight_env in sorted(weights_obj):
            weight_path = f"{view_path}.environment_weights.{weight_env}"
            weight = _integer(weights_obj[weight_env], weight_path, view_entry, minimum=1)
            if weight_env not in participant_ids:
                raise RefError(
                    "environment weight references an environment outside the view",
                    details={"path": weight_path, "entry": view_entry, "environment_id": weight_env},
                )
            weights[weight_env] = weight

        raw_policies = _object(view_obj.get("key_policies"), view_path + ".key_policies", view_entry)
        if not raw_policies:
            raise _spec(view_path + ".key_policies", "at least one key policy is required", entry=view_entry)
        policies: dict[str, KeyPolicy] = {}
        observed_keys = {
            observation.key
            for env_id in participant_ids
            for source in environments[env_id].sources.values()
            for observation in source.observations.values()
        }
        for key in sorted(raw_policies):
            policy_path = f"{view_path}.key_policies.{key}"
            policy_entry = f"{view_entry}.key_policies[{key}]"
            if not key:
                raise _spec(policy_path, "key must be a non-empty string", entry=policy_entry)
            if key not in observed_keys:
                raise _spec(policy_path, "key has no observations in participating environments", entry=policy_entry, key=key)
            policy_obj = _object(raw_policies[key], policy_path, policy_entry)
            strategy = _string(policy_obj.get("strategy"), policy_path + ".strategy", policy_entry)
            if strategy not in SUPPORTED_STRATEGIES:
                raise _spec(
                    policy_path + ".strategy",
                    "unsupported arbitration strategy",
                    entry=policy_entry,
                    actual=strategy,
                    supported=list(SUPPORTED_STRATEGIES),
                )
            tolerance = _integer(policy_obj.get("tolerance"), policy_path + ".tolerance", policy_entry, minimum=0)
            policies[key] = KeyPolicy(strategy=strategy, tolerance=tolerance)

        views[view_id] = View(
            view_id=view_id,
            environments=tuple(participant_ids),
            environment_priority=tuple(priority),
            environment_weights=weights,
            key_policies=policies,
        )
    return views, session


def load_specs(
    environment_path: str | Path,
    views_path: str | Path,
    *,
    inject_spec_corrupt: bool = False,
    inject_priority_ref_missing_env: bool = False,
) -> Specs:
    environment_document = _read_document(environment_path)
    view_document = _read_document(views_path)
    _apply_injections(
        environment_document,
        view_document,
        inject_spec_corrupt=inject_spec_corrupt,
        inject_priority_ref_missing_env=inject_priority_ref_missing_env,
    )
    environments, source_type_order, env_version, priority_version = _parse_environments(environment_document)
    views, session = _parse_views(view_document, environments)
    return Specs(
        environment_document=environment_document,
        view_document=view_document,
        environments=environments,
        views=views,
        source_type_order=source_type_order,
        environment_set_version=env_version,
        source_priority_version=priority_version,
        session=session,
    )
