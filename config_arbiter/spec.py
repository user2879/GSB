from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import canonical_json, normalize_decimal, read_json_strict, sha256_hex
from .errors import SpecError
from .fixedpoint import quantize_decimal

ARBITRATION_MODES = ("latest", "weighted_majority", "interval_midpoint")
LOOPBACK_HOST = "127.0.0.1"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SessionSpec:
    scale_exponent: int
    seed: int
    sample_count: int
    group_size: int
    noise_bound: int
    global_concurrency: int
    per_environment_concurrency: int
    handshake_timeout_ms: int


@dataclass(frozen=True)
class SliceSpec:
    version: int
    values: Mapping[str, str]

    def digest(self) -> str:
        return sha256_hex({"version": self.version, "values": dict(self.values)})


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    host: str
    port: int
    weight_text: str
    weight: Decimal
    weight_units: int
    weight_error_units: Decimal
    slices: tuple[SliceSpec, ...]

    @property
    def current_slice(self) -> SliceSpec:
        return self.slices[-1]


@dataclass(frozen=True)
class KeyRule:
    mode: str
    tolerance: int


@dataclass(frozen=True)
class ViewSpec:
    id: str
    environments: tuple[str, ...]
    keys: Mapping[str, KeyRule]

    def config_payload(self) -> dict[str, Any]:
        return {
            "environments": sorted(self.environments),
            "keys": {
                key: {"mode": rule.mode, "tolerance": rule.tolerance}
                for key, rule in sorted(self.keys.items())
            },
        }

    def config_digest(self) -> str:
        return sha256_hex(
            {
                "id": self.id,
                **self.config_payload(),
            }
        )

    @classmethod
    def from_config_payload(
        cls, view_id: str, payload: Any
    ) -> "ViewSpec | None":
        if not isinstance(payload, dict):
            return None
        environments = payload.get("environments")
        keys_payload = payload.get("keys")
        if not isinstance(environments, list) or not isinstance(keys_payload, dict):
            return None
        if not all(isinstance(item, str) and item for item in environments):
            return None
        keys: dict[str, KeyRule] = {}
        for key, rule_payload in keys_payload.items():
            if not isinstance(key, str) or not isinstance(rule_payload, dict):
                return None
            mode = rule_payload.get("mode")
            tolerance = rule_payload.get("tolerance")
            if mode not in ARBITRATION_MODES:
                return None
            if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
                return None
            keys[key] = KeyRule(mode=mode, tolerance=tolerance)
        return cls(
            id=view_id,
            environments=tuple(environments),
            keys=MappingProxyType(keys),
        )


@dataclass(frozen=True)
class DeploymentSpec:
    source_path: Path
    format_version: int
    session: SessionSpec
    environments: Mapping[str, EnvironmentSpec]
    views: Mapping[str, ViewSpec]

    def current_slices(self) -> dict[str, SliceSpec]:
        return {env_id: env.current_slice for env_id, env in self.environments.items()}


def _expect_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(field, "must be a JSON object")
    return value


def _expect_list(value: Any, field: str, *, non_empty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(field, "must be a JSON array")
    if non_empty and not value:
        raise SpecError(field, "must not be empty")
    return value


def _expect_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(field, "must be an integer")
    if minimum is not None and value < minimum:
        raise SpecError(field, f"must be >= {minimum}")
    return value


def _expect_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(field, "must be a non-empty string")
    return value


def _parse_session(raw: Any) -> SessionSpec:
    session = _expect_object(raw, "$.session")
    scale_exponent = _expect_int(session.get("scale_exponent"), "$.session.scale_exponent", minimum=0)
    if scale_exponent > 12:
        raise SpecError("$.session.scale_exponent", "must be <= 12")
    seed = _expect_int(session.get("seed"), "$.session.seed", minimum=0)
    sample_count = _expect_int(session.get("sample_count"), "$.session.sample_count", minimum=2)
    group_size = _expect_int(session.get("group_size"), "$.session.group_size", minimum=2)
    if sample_count % group_size != 0:
        raise SpecError(
            "$.session.sample_count",
            "must be divisible by $.session.group_size",
        )
    noise_bound = _expect_int(session.get("noise_bound"), "$.session.noise_bound", minimum=0)
    global_concurrency = _expect_int(
        session.get("global_concurrency"),
        "$.session.global_concurrency",
        minimum=1,
    )
    per_environment = _expect_int(
        session.get("per_environment_concurrency"),
        "$.session.per_environment_concurrency",
        minimum=1,
    )
    timeout_ms = _expect_int(
        session.get("handshake_timeout_ms"),
        "$.session.handshake_timeout_ms",
        minimum=1,
    )
    return SessionSpec(
        scale_exponent=scale_exponent,
        seed=seed,
        sample_count=sample_count,
        group_size=group_size,
        noise_bound=noise_bound,
        global_concurrency=global_concurrency,
        per_environment_concurrency=per_environment,
        handshake_timeout_ms=timeout_ms,
    )


def parse_slice(version: Any, values: Any, field: str) -> SliceSpec:
    parsed_version = _expect_int(version, f"{field}.version", minimum=1)
    value_object = _expect_object(values, f"{field}.values")
    if not value_object:
        raise SpecError(f"{field}.values", "must not be empty")
    normalized: dict[str, str] = {}
    for key in sorted(value_object):
        if not isinstance(key, str) or not key.strip():
            raise SpecError(f"{field}.values", "key names must be non-empty strings")
        text, _ = normalize_decimal(value_object[key], f"{field}.values.{key}")
        normalized[key] = text
    return SliceSpec(version=parsed_version, values=MappingProxyType(normalized))


def _parse_environments(
    raw: Any, session: SessionSpec
) -> tuple[dict[str, EnvironmentSpec], set[int]]:
    items = _expect_list(raw, "$.environments", non_empty=True)
    result: dict[str, EnvironmentSpec] = {}
    used_ports: set[int] = set()
    for index, item in enumerate(items):
        base = f"$.environments[{index}]"
        env = _expect_object(item, base)
        env_id = _expect_id(env.get("id"), f"{base}.id")
        if env_id in result:
            raise SpecError(f"{base}.id", f"duplicate environment id {env_id!r}")
        host = env.get("host")
        if host != LOOPBACK_HOST:
            raise SpecError(f"{base}.host", "must be the literal string '127.0.0.1'")
        port = _expect_int(env.get("port"), f"{base}.port", minimum=1024)
        if port > 65535:
            raise SpecError(f"{base}.port", "must be <= 65535")
        if port in used_ports:
            raise SpecError(f"{base}.port", f"duplicate loopback port {port}")
        used_ports.add(port)

        weight_text, weight = normalize_decimal(env.get("weight"), f"{base}.weight")
        if weight <= 0:
            raise SpecError(f"{base}.weight", "must be > 0")
        quantized_weight = quantize_decimal(weight_text, session.scale_exponent, f"{base}.weight")
        if quantized_weight.units <= 0:
            raise SpecError(
                f"{base}.weight",
                "quantizes to zero at the session scale; use a larger weight",
            )

        slice_items = _expect_list(env.get("slices"), f"{base}.slices", non_empty=True)
        slices: list[SliceSpec] = []
        previous_version = 0
        for slice_index, slice_item in enumerate(slice_items):
            slice_base = f"{base}.slices[{slice_index}]"
            slice_obj = _expect_object(slice_item, slice_base)
            parsed = parse_slice(
                slice_obj.get("version"),
                slice_obj.get("values"),
                slice_base,
            )
            if parsed.version <= previous_version:
                raise SpecError(
                    f"{slice_base}.version",
                    f"must be strictly increasing; previous version is {previous_version}",
                )
            previous_version = parsed.version
            slices.append(parsed)

        result[env_id] = EnvironmentSpec(
            id=env_id,
            host=host,
            port=port,
            weight_text=weight_text,
            weight=weight,
            weight_units=quantized_weight.units,
            weight_error_units=quantized_weight.error_units,
            slices=tuple(slices),
        )
    return result, used_ports


def _parse_views(
    raw: Any, environments: Mapping[str, EnvironmentSpec]
) -> dict[str, ViewSpec]:
    items = _expect_list(raw, "$.views", non_empty=True)
    result: dict[str, ViewSpec] = {}
    for index, item in enumerate(items):
        base = f"$.views[{index}]"
        view = _expect_object(item, base)
        view_id = _expect_id(view.get("id"), f"{base}.id")
        if view_id in result:
            raise SpecError(f"{base}.id", f"duplicate view id {view_id!r}")

        participants_raw = _expect_list(
            view.get("environments"), f"{base}.environments", non_empty=True
        )
        participants: list[str] = []
        seen_participants: set[str] = set()
        for participant_index, participant in enumerate(participants_raw):
            participant_field = f"{base}.environments[{participant_index}]"
            if not isinstance(participant, str) or not participant:
                raise SpecError(participant_field, "must be a non-empty environment id")
            if participant in seen_participants:
                raise SpecError(participant_field, f"duplicate participant {participant!r}")
            if participant not in environments:
                raise SpecError(participant_field, f"unknown environment {participant!r}")
            seen_participants.add(participant)
            participants.append(participant)

        keys_raw = _expect_object(view.get("keys"), f"{base}.keys")
        if not keys_raw:
            raise SpecError(f"{base}.keys", "must not be empty")
        keys: dict[str, KeyRule] = {}
        for key in sorted(keys_raw):
            key_field = f"{base}.keys.{key}"
            if not isinstance(key, str) or not key.strip():
                raise SpecError(f"{base}.keys", "key names must be non-empty strings")
            rule_raw = _expect_object(keys_raw[key], key_field)
            mode = rule_raw.get("mode")
            if mode not in ARBITRATION_MODES:
                raise SpecError(
                    f"{key_field}.mode",
                    f"must be one of {', '.join(ARBITRATION_MODES)}",
                )
            tolerance = _expect_int(
                rule_raw.get("tolerance"), f"{key_field}.tolerance", minimum=0
            )
            keys[key] = KeyRule(mode=mode, tolerance=tolerance)

        result[view_id] = ViewSpec(
            id=view_id,
            environments=tuple(participants),
            keys=MappingProxyType(keys),
        )
    return result


def load_spec(path: str | Path) -> DeploymentSpec:
    source = Path(path).resolve()
    raw = read_json_strict(source)
    root = _expect_object(raw, "$")
    format_version = _expect_int(root.get("format_version"), "$.format_version", minimum=1)
    if format_version != FORMAT_VERSION:
        raise SpecError(
            "$.format_version",
            f"unsupported format version {format_version}; expected {FORMAT_VERSION}",
        )
    session = _parse_session(root.get("session"))
    environments, _ = _parse_environments(root.get("environments"), session)
    views = _parse_views(root.get("views"), environments)
    return DeploymentSpec(
        source_path=source,
        format_version=format_version,
        session=session,
        environments=MappingProxyType(environments),
        views=MappingProxyType(views),
    )


def spec_fingerprint(spec: DeploymentSpec) -> str:
    return sha256_hex(
        {
            "format_version": spec.format_version,
            "session": {
                "scale_exponent": spec.session.scale_exponent,
                "seed": spec.session.seed,
                "sample_count": spec.session.sample_count,
                "group_size": spec.session.group_size,
                "noise_bound": spec.session.noise_bound,
            },
            "environments": {
                env_id: {
                    "port": env.port,
                    "weight": env.weight_text,
                    "slices": [
                        {"version": item.version, "values": dict(item.values)}
                        for item in env.slices
                    ],
                }
                for env_id, env in sorted(spec.environments.items())
            },
            "views": {
                view_id: json.loads(canonical_json({
                    "environments": sorted(view.environments),
                    "keys": {
                        key: {"mode": rule.mode, "tolerance": rule.tolerance}
                        for key, rule in sorted(view.keys.items())
                    },
                }))
                for view_id, view in sorted(spec.views.items())
            },
        }
    )
