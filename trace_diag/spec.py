from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .canonical import sha256_json
from .errors import BucketError, SpecError

REQUIRED_FIELDS = (
    "timestamp_ms",
    "magnitude",
    "client_key",
    "duration_us",
    "trace_id",
    "sampled",
)

FIELD_DEFAULTS: dict[str, dict[str, Any]] = {
    "timestamp_ms": {"type": "int", "start": 1_700_000_000_000, "max_step_ms": 37},
    "magnitude": {"type": "int", "min": 0, "max": 9},
    "client_key": {"type": "string", "key_space": 257},
    "duration_us": {"type": "int", "min_us": 500, "max_us": 500_000},
    "trace_id": {"type": "string"},
    "sampled": {"type": "bool", "rate_bp": 2_500},
}

FIELD_PARAM_KEYS: dict[str, set[str]] = {
    "timestamp_ms": {"type", "start", "max_step_ms"},
    "magnitude": {"type", "min", "max"},
    "client_key": {"type", "key_space"},
    "duration_us": {"type", "min_us", "max_us"},
    "trace_id": {"type"},
    "sampled": {"type", "rate_bp"},
}

SERVICES_ROOT_KEYS = {"version", "services", "description"}
SERVICE_KEYS = {"id", "seed", "record_count", "fields", "description"}
POLICIES_ROOT_KEYS = {"version", "policies", "description"}
POLICY_KEYS = {
    "id",
    "version",
    "head_ratio_bp",
    "tail_duration_threshold_us",
    "tail_ratio_bp",
    "scale_power",
    "bucket_count",
    "description",
}

SUPPORTED_BUCKET_MIN = 2
SUPPORTED_BUCKET_MAX = 256
SUPPORTED_RECORD_MAX = 1_000_000


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def read_json_document(path: str | Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError("$", f"cannot read JSON file: {source}") from exc
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_object)
    except _DuplicateKey as exc:
        raise SpecError(f"$.{exc.key}", "duplicate JSON key") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "$",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(path, "expected a JSON object")
    return value


def _require_array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(path, "expected a JSON array")
    return value


def _reject_unknown(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise SpecError(f"{path}.{unknown[0]}", "unknown field")


def _required(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise SpecError(f"{path}.{key}", "required field is missing")
    return obj[key]


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SpecError(path, "expected a string")
    if not allow_empty and not value.strip():
        raise SpecError(path, "must not be empty")
    if any(ord(char) < 32 for char in value):
        raise SpecError(path, "must not contain control characters")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    error: Callable[[str, str], Exception] | None = None,
) -> int:
    if not _is_int(value):
        exc = error or SpecError
        raise exc(path, "expected an integer")
    if minimum is not None and value < minimum:
        exc = error or SpecError
        raise exc(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        exc = error or SpecError
        raise exc(path, f"must be <= {maximum}")
    return value


def _description(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise SpecError(path, "expected a string")
    return value


@dataclass(frozen=True)
class ServiceSpec:
    id: str
    seed: int
    record_count: int
    fields: dict[str, dict[str, Any]]
    description: str = ""

    def field(self, name: str) -> dict[str, Any]:
        return self.fields[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed": self.seed,
            "record_count": self.record_count,
            "fields": {name: dict(self.fields[name]) for name in REQUIRED_FIELDS},
            "description": self.description,
        }

    def content_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class ServicesDoc:
    version: int
    services: tuple[ServiceSpec, ...]
    description: str = ""

    def service_map(self) -> dict[str, ServiceSpec]:
        return {service.id: service for service in self.services}

    def content_dict(self) -> dict[str, Any]:
        return {
            "services": [service.content_dict() for service in self.services],
            "description": self.description,
        }

    @property
    def content_digest(self) -> str:
        return sha256_json(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "services": [service.to_dict() for service in self.services],
            "description": self.description,
        }

    def with_version(self, version: int) -> "ServicesDoc":
        return replace(self, version=version)


@dataclass(frozen=True)
class SamplingPolicy:
    id: str
    version: int
    head_ratio_bp: int
    tail_duration_threshold_us: int
    tail_ratio_bp: int
    scale_power: int
    bucket_count: int
    description: str = ""

    @property
    def quantum_us(self) -> int:
        return 10**self.scale_power

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "head_ratio_bp": self.head_ratio_bp,
            "tail_duration_threshold_us": self.tail_duration_threshold_us,
            "tail_ratio_bp": self.tail_ratio_bp,
            "scale_power": self.scale_power,
            "bucket_count": self.bucket_count,
            "description": self.description,
        }

    def content_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("version")
        return result

    def with_version(self, version: int) -> "SamplingPolicy":
        return replace(self, version=version)

    def variant(self, **changes: Any) -> "SamplingPolicy":
        return replace(self, **changes)


@dataclass(frozen=True)
class PoliciesDoc:
    version: int
    policies: tuple[SamplingPolicy, ...]
    description: str = ""

    def policy_map(self) -> dict[str, SamplingPolicy]:
        return {policy.id: policy for policy in self.policies}

    def content_dict(self) -> dict[str, Any]:
        return {
            "policies": [policy.content_dict() for policy in self.policies],
            "description": self.description,
        }

    @property
    def content_digest(self) -> str:
        return sha256_json(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policies": [policy.to_dict() for policy in self.policies],
            "description": self.description,
        }

    def with_version(self, version: int) -> "PoliciesDoc":
        return replace(
            self,
            version=version,
            policies=tuple(policy.with_version(max(policy.version, version)) for policy in self.policies),
        )


def _parse_field(name: str, value: Any, path: str) -> dict[str, Any]:
    obj = _require_object(value, path)
    _reject_unknown(obj, FIELD_PARAM_KEYS[name], path)
    field_type = _required(obj, "type", path)
    expected = FIELD_DEFAULTS[name]["type"]
    if field_type != expected:
        raise SpecError(f"{path}.type", f"expected {expected!r}")
    result = dict(FIELD_DEFAULTS[name])
    for key, item in obj.items():
        if key != "type":
            result[key] = item

    if name == "timestamp_ms":
        result["start"] = _integer(result["start"], f"{path}.start", minimum=0)
        result["max_step_ms"] = _integer(
            result["max_step_ms"], f"{path}.max_step_ms", minimum=0
        )
    elif name == "magnitude":
        result["min"] = _integer(result["min"], f"{path}.min", minimum=0)
        result["max"] = _integer(result["max"], f"{path}.max", minimum=0)
        if result["max"] < result["min"]:
            raise SpecError(f"{path}.max", "must be >= min")
    elif name == "client_key":
        result["key_space"] = _integer(
            result["key_space"], f"{path}.key_space", minimum=1
        )
    elif name == "duration_us":
        result["min_us"] = _integer(
            result["min_us"], f"{path}.min_us", minimum=1
        )
        result["max_us"] = _integer(
            result["max_us"], f"{path}.max_us", minimum=1
        )
        if result["max_us"] < result["min_us"]:
            raise SpecError(f"{path}.max_us", "must be >= min_us")
    elif name == "sampled":
        result["rate_bp"] = _integer(
            result["rate_bp"], f"{path}.rate_bp", minimum=0, maximum=10_000
        )
    return result


def parse_services_document(raw: Any, *, source: str = "$") -> ServicesDoc:
    root = _require_object(raw, source)
    _reject_unknown(root, SERVICES_ROOT_KEYS, source)
    version = _integer(
        _required(root, "version", source),
        f"{source}.version",
        minimum=1,
    )
    description = _description(root.get("description", ""), f"{source}.description")
    services_raw = _require_array(_required(root, "services", source), f"{source}.services")
    if not services_raw:
        raise SpecError(f"{source}.services", "at least one service is required")

    seen: set[str] = set()
    services: list[ServiceSpec] = []
    for index, raw_service in enumerate(services_raw):
        path = f"{source}.services[{index}]"
        obj = _require_object(raw_service, path)
        _reject_unknown(obj, SERVICE_KEYS, path)
        service_id = _string(_required(obj, "id", path), f"{path}.id")
        if service_id in seen:
            raise SpecError(f"{path}.id", f"duplicate service id {service_id!r}")
        seen.add(service_id)
        seed = _integer(_required(obj, "seed", path), f"{path}.seed", minimum=1)
        record_count = _integer(
            _required(obj, "record_count", path),
            f"{path}.record_count",
            minimum=1,
            maximum=SUPPORTED_RECORD_MAX,
        )
        fields_raw = _require_object(_required(obj, "fields", path), f"{path}.fields")
        missing = [name for name in REQUIRED_FIELDS if name not in fields_raw]
        if missing:
            raise SpecError(f"{path}.fields.{missing[0]}", "required field declaration is missing")
        extra = sorted(set(fields_raw) - set(REQUIRED_FIELDS))
        if extra:
            raise SpecError(f"{path}.fields.{extra[0]}", "unknown record field")
        fields = {
            name: _parse_field(name, fields_raw[name], f"{path}.fields.{name}")
            for name in REQUIRED_FIELDS
        }
        services.append(
            ServiceSpec(
                id=service_id,
                seed=seed,
                record_count=record_count,
                fields=fields,
                description=_description(obj.get("description", ""), f"{path}.description"),
            )
        )
    return ServicesDoc(version=version, services=tuple(services), description=description)


def parse_policies_document(
    raw: Any,
    *,
    source: str = "$",
    ratio_exception: type[BucketError] | None = None,
) -> PoliciesDoc:
    root = _require_object(raw, source)
    _reject_unknown(root, POLICIES_ROOT_KEYS, source)
    version = _integer(
        _required(root, "version", source),
        f"{source}.version",
        minimum=1,
    )
    description = _description(root.get("description", ""), f"{source}.description")
    policies_raw = _require_array(
        _required(root, "policies", source), f"{source}.policies"
    )
    if not policies_raw:
        raise SpecError(f"{source}.policies", "at least one policy is required")

    ratio_error: Callable[[str, str], Exception] | None = None
    if ratio_exception is not None:
        ratio_error = lambda path, message: ratio_exception(path, message)

    seen: set[str] = set()
    policies: list[SamplingPolicy] = []
    for index, raw_policy in enumerate(policies_raw):
        path = f"{source}.policies[{index}]"
        obj = _require_object(raw_policy, path)
        _reject_unknown(obj, POLICY_KEYS, path)
        policy_id = _string(_required(obj, "id", path), f"{path}.id")
        if policy_id in seen:
            raise SpecError(f"{path}.id", f"duplicate policy id {policy_id!r}")
        seen.add(policy_id)
        policy_version = _integer(
            obj.get("version", version), f"{path}.version", minimum=1
        )
        head_ratio_bp = _integer(
            _required(obj, "head_ratio_bp", path),
            f"{path}.head_ratio_bp",
            minimum=0,
            maximum=10_000,
            error=ratio_error,
        )
        tail_ratio_bp = _integer(
            _required(obj, "tail_ratio_bp", path),
            f"{path}.tail_ratio_bp",
            minimum=0,
            maximum=10_000,
            error=ratio_error,
        )
        threshold = _integer(
            _required(obj, "tail_duration_threshold_us", path),
            f"{path}.tail_duration_threshold_us",
            minimum=1,
        )
        scale_power = _integer(
            _required(obj, "scale_power", path),
            f"{path}.scale_power",
            minimum=0,
            maximum=6,
        )
        bucket_count = _integer(
            _required(obj, "bucket_count", path),
            f"{path}.bucket_count",
            minimum=SUPPORTED_BUCKET_MIN,
            maximum=SUPPORTED_BUCKET_MAX,
        )
        policies.append(
            SamplingPolicy(
                id=policy_id,
                version=policy_version,
                head_ratio_bp=head_ratio_bp,
                tail_duration_threshold_us=threshold,
                tail_ratio_bp=tail_ratio_bp,
                scale_power=scale_power,
                bucket_count=bucket_count,
                description=_description(obj.get("description", ""), f"{path}.description"),
            )
        )
    return PoliciesDoc(version=version, policies=tuple(policies), description=description)


def load_services_file(
    path: str | Path, *, inject_missing_id: bool = False
) -> ServicesDoc:
    raw = read_json_document(path)
    if inject_missing_id:
        services = raw.get("services") if isinstance(raw, dict) else None
        if isinstance(services, list) and services and isinstance(services[0], dict):
            services[0].pop("id", None)
    return parse_services_document(raw)


def load_policies_file(
    path: str | Path, *, inject_ratio_bucket_error: bool = False
) -> PoliciesDoc:
    raw = read_json_document(path)
    if inject_ratio_bucket_error:
        policies = raw.get("policies") if isinstance(raw, dict) else None
        if isinstance(policies, list) and policies and isinstance(policies[0], dict):
            policies[0]["head_ratio_bp"] = 10_001
    return parse_policies_document(
        raw,
        ratio_exception=BucketError if inject_ratio_bucket_error else None,
    )
