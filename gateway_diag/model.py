"""Domain models, canonical serialization, and strict local file validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import SpecError

SUPPORTED_QUANTIZATIONS = ("fp32", "int8", "int4")
LOOPBACK_HOST = "127.0.0.1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fail(entry: str, field: str, message: str) -> None:
    raise SpecError(message, entry=entry, field=field)


def _require_object(value: Any, entry: str, field: str = "") -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(entry, field, "must be a JSON object")
    return value


def _get_alias(
    obj: Mapping[str, Any],
    names: Sequence[str],
    entry: str,
    field: str,
) -> Any:
    for name in names:
        if name in obj:
            return obj[name]
    _fail(entry, field, "field is required")


def _integer(
    value: Any,
    entry: str,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(entry, field, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(entry, field, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(entry, field, f"must be <= {maximum}")
    return value


def _string(value: Any, entry: str, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(entry, field, "must be a string")
    if nonempty and not value:
        _fail(entry, field, "must not be empty")
    return value


@dataclass(frozen=True)
class Limits:
    global_concurrency: int
    backend_concurrency: Mapping[str, int]

    def quota_for(self, backend_id: str) -> int:
        return self.backend_concurrency[backend_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_concurrency": self.global_concurrency,
            "backend_concurrency": {
                key: self.backend_concurrency[key]
                for key in sorted(self.backend_concurrency)
            },
        }


@dataclass(frozen=True)
class Backend:
    id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    quantization: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "quantization": self.quantization,
        }

    @property
    def content_digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


@dataclass(frozen=True)
class RouteRule:
    id: str
    prefix: str
    stable_backend: str
    canary_backend: str
    canary_weight_bp: int
    batch_size: int
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prefix": self.prefix,
            "stable_backend": self.stable_backend,
            "canary_backend": self.canary_backend,
            "canary_weight_bp": self.canary_weight_bp,
            "batch_size": self.batch_size,
            "version": self.version,
        }

    @property
    def content_digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


@dataclass(frozen=True)
class RequestSample:
    request_id: int
    client_key: str
    input_vector: tuple[float, ...]
    arrival_ms: int
    selector: str | None = None
    source_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "client_key": self.client_key,
            "input": list(self.input_vector),
            "arrival_ms": self.arrival_ms,
        }
        if self.selector is not None:
            result["prefix"] = self.selector
        return result


@dataclass(frozen=True)
class Spec:
    revision: int
    limits: Limits
    backends: tuple[Backend, ...]
    rules: tuple[RouteRule, ...]

    def backend(self, backend_id: str) -> Backend:
        for backend in self.backends:
            if backend.id == backend_id:
                return backend
        raise KeyError(backend_id)

    def rule(self, rule_id: str) -> RouteRule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    @property
    def backends_by_id(self) -> dict[str, Backend]:
        return {backend.id: backend for backend in self.backends}

    @property
    def rules_by_id(self) -> dict[str, RouteRule]:
        return {rule.id: rule for rule in self.rules}

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "limits": self.limits.to_dict(),
            "backends": [backend.to_dict() for backend in self.backends],
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @property
    def fingerprint(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def inventory(self) -> dict[str, dict[str, str]]:
        return {
            "backends": {
                backend.id: backend.content_digest
                for backend in sorted(self.backends, key=lambda item: item.id)
            },
            "rules": {
                rule.id: str(rule.version)
                for rule in sorted(self.rules, key=lambda item: item.id)
            },
        }


def _parse_limits(
    root: Mapping[str, Any], backend_ids: Sequence[str]
) -> Limits:
    raw_limits: Any = root.get("limits", {})
    if raw_limits is None:
        raw_limits = {}
    limits_obj = _require_object(raw_limits, "limits")

    global_value = _get_alias(
        limits_obj,
        ("global_concurrency", "global_concurrency_limit", "max_global_concurrency"),
        "limits",
        "global_concurrency",
    ) if any(
        key in limits_obj
        for key in (
            "global_concurrency",
            "global_concurrency_limit",
            "max_global_concurrency",
        )
    ) else root.get("global_concurrency", 1)
    global_limit = _integer(
        global_value, "limits", "global_concurrency", minimum=1
    )

    quotas_raw = limits_obj.get("backend_concurrency", {})
    quotas_obj = _require_object(quotas_raw, "limits", "backend_concurrency")
    default_quota = limits_obj.get("default_backend_concurrency", 1)
    default_quota = _integer(
        default_quota, "limits", "default_backend_concurrency", minimum=1
    )
    quotas: dict[str, int] = {}
    known = set(backend_ids)
    for backend_id in sorted(quotas_obj):
        if backend_id not in known:
            _fail(
                "limits",
                f"backend_concurrency.{backend_id}",
                "references an unknown backend id",
            )
        quotas[backend_id] = _integer(
            quotas_obj[backend_id],
            "limits",
            f"backend_concurrency.{backend_id}",
            minimum=1,
        )
    for backend_id in backend_ids:
        quotas.setdefault(backend_id, default_quota)
    return Limits(global_concurrency=global_limit, backend_concurrency=quotas)


def parse_spec_obj(value: Any, *, source: str = "<memory>") -> Spec:
    root = _require_object(value, "spec")
    revision_value = 1
    for name in ("revision", "spec_revision", "version"):
        if name in root:
            revision_value = root[name]
            break
    revision = _integer(revision_value, "spec", "revision", minimum=1)

    raw_backends = root.get("backends")
    if not isinstance(raw_backends, list) or not raw_backends:
        _fail("backends", "", "must be a non-empty array")
    raw_rules = root.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        _fail("rules", "", "must be a non-empty array")

    backends: list[Backend] = []
    backend_ids: set[str] = set()
    ports: dict[int, str] = {}
    for index, raw_backend in enumerate(raw_backends):
        entry = f"backends[{index}]"
        obj = _require_object(raw_backend, entry)
        backend_id = _string(
            _get_alias(obj, ("id", "backend_id"), entry, "id"), entry, "id"
        )
        if backend_id in backend_ids:
            _fail(entry, "id", f"duplicate backend id {backend_id!r}")
        backend_ids.add(backend_id)
        host = obj.get("host", LOOPBACK_HOST)
        host = _string(host, entry, "host")
        if host != LOOPBACK_HOST:
            _fail(entry, "host", "must be the literal IPv4 loopback address 127.0.0.1")
        port = _integer(_get_alias(obj, ("port",), entry, "port"), entry, "port", minimum=1, maximum=65535)
        if port in ports:
            _fail(
                entry,
                "port",
                f"duplicate port {port}; already used by {ports[port]!r}",
            )
        ports[port] = backend_id
        model_dim = _integer(
            _get_alias(obj, ("model_dim", "model_dimension", "input_dim"), entry, "model_dim"),
            entry,
            "model_dim",
            minimum=1,
        )
        hidden_dim = _integer(
            _get_alias(obj, ("hidden_dim", "hidden_dimension"), entry, "hidden_dim"),
            entry,
            "hidden_dim",
            minimum=1,
        )
        seed = _integer(_get_alias(obj, ("seed", "random_seed"), entry, "seed"), entry, "seed")
        quantization = _string(
            _get_alias(
                obj,
                ("quantization", "quantization_profile", "quant"),
                entry,
                "quantization",
            ),
            entry,
            "quantization",
        )
        if quantization not in SUPPORTED_QUANTIZATIONS:
            _fail(
                entry,
                "quantization",
                f"unsupported value; expected one of {', '.join(SUPPORTED_QUANTIZATIONS)}",
            )
        backends.append(
            Backend(
                id=backend_id,
                host=host,
                port=port,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                quantization=quantization,
            )
        )

    backend_id_set = {backend.id for backend in backends}
    rules: list[RouteRule] = []
    rule_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        entry = f"rules[{index}]"
        obj = _require_object(raw_rule, entry)
        rule_id = _string(
            _get_alias(obj, ("id", "rule_id"), entry, "id"), entry, "id"
        )
        if rule_id in rule_ids:
            _fail(entry, "id", f"duplicate rule id {rule_id!r}")
        rule_ids.add(rule_id)
        prefix = _string(
            _get_alias(obj, ("prefix", "request_prefix"), entry, "prefix"),
            entry,
            "prefix",
        )
        stable_backend = _string(
            _get_alias(
                obj,
                ("stable_backend", "stable_backend_id", "stable_target"),
                entry,
                "stable_backend",
            ),
            entry,
            "stable_backend",
        )
        if stable_backend not in backend_id_set:
            _fail(entry, "stable_backend", f"unknown backend {stable_backend!r}")
        canary_backend = _string(
            _get_alias(
                obj,
                ("canary_backend", "canary_backend_id", "gray_backend", "canary_target"),
                entry,
                "canary_backend",
            ),
            entry,
            "canary_backend",
        )
        if canary_backend not in backend_id_set:
            _fail(entry, "canary_backend", f"unknown backend {canary_backend!r}")
        weight = _integer(
            _get_alias(
                obj,
                ("canary_weight_bp", "gray_weight_bp", "canary_weight"),
                entry,
                "canary_weight_bp",
            ),
            entry,
            "canary_weight_bp",
            minimum=0,
            maximum=10000,
        )
        batch_size = _integer(
            _get_alias(obj, ("batch_size",), entry, "batch_size"),
            entry,
            "batch_size",
            minimum=1,
        )
        version = _integer(
            _get_alias(obj, ("version", "rule_version"), entry, "version"),
            entry,
            "version",
            minimum=1,
        )
        rules.append(
            RouteRule(
                id=rule_id,
                prefix=prefix,
                stable_backend=stable_backend,
                canary_backend=canary_backend,
                canary_weight_bp=weight,
                batch_size=batch_size,
                version=version,
            )
        )

    limits = _parse_limits(root, [backend.id for backend in backends])
    return Spec(
        revision=revision,
        limits=limits,
        backends=tuple(backends),
        rules=tuple(rules),
    )


def _read_json_file(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(
            f"cannot read {file_path}: {exc}", entry=str(file_path), field="file"
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            entry=str(file_path),
            field=f"line {exc.lineno} column {exc.colno}",
        ) from exc


def load_spec(path: str | Path) -> Spec:
    return parse_spec_obj(_read_json_file(path), source=str(path))


def _parse_input_vector(value: Any, entry: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        _fail(entry, "input", "must be a non-empty numeric array")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _fail(entry, f"input[{index}]", "must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            _fail(entry, f"input[{index}]", "must be a finite number")
        result.append(number)
    return tuple(result)


def parse_request_objs(lines: Iterable[tuple[int, str]], spec: Spec) -> tuple[RequestSample, ...]:
    from .routing import decide_route

    requests: list[RequestSample] = []
    seen_ids: set[int] = set()
    for source_index, line in lines:
        entry = f"requests[{source_index}]"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(entry, "json", f"invalid JSON: {exc.msg}")
        if not isinstance(obj, dict):
            _fail(entry, "", "must be a JSON object")
        request_id = _integer(
            _get_alias(obj, ("request_id", "request_seq", "sequence", "seq"), entry, "request_id"),
            entry,
            "request_id",
            minimum=1,
        )
        if request_id in seen_ids:
            _fail(entry, "request_id", f"duplicate request id {request_id}")
        seen_ids.add(request_id)
        client_key = _string(
            _get_alias(obj, ("client_key", "client"), entry, "client_key"),
            entry,
            "client_key",
        )
        vector = _parse_input_vector(
            _get_alias(obj, ("input", "input_vector", "vector"), entry, "input"), entry
        )
        arrival_ms = _integer(
            _get_alias(
                obj,
                ("arrival_ms", "arrived_at_ms", "arrival_time_ms"),
                entry,
                "arrival_ms",
            ),
            entry,
            "arrival_ms",
            minimum=0,
        )
        selector_value = None
        for name in ("prefix", "request_prefix", "path", "rule_id"):
            if name in obj:
                selector_value = _string(obj[name], entry, name)
                break

        sample = RequestSample(
            request_id=request_id,
            client_key=client_key,
            input_vector=vector,
            arrival_ms=arrival_ms,
            selector=selector_value,
            source_index=source_index,
        )
        decision = decide_route(spec, sample)
        backend = spec.backend(decision.backend_id)
        if len(vector) != backend.model_dim:
            _fail(
                entry,
                "input",
                f"vector length {len(vector)} does not match routed backend {backend.id!r} model_dim {backend.model_dim}",
            )
        requests.append(sample)
    if not requests:
        _fail("requests", "", "request stream must contain at least one request")
    return tuple(sorted(requests, key=lambda item: (item.arrival_ms, item.request_id)))


def load_requests(path: str | Path, spec: Spec) -> tuple[RequestSample, ...]:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(
            f"cannot read {file_path}: {exc}", entry=str(file_path), field="file"
        ) from exc
    lines: list[tuple[int, str]] = []
    for source_index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        lines.append((source_index, line))
    return parse_request_objs(lines, spec)


def request_stream_digest(requests: Sequence[RequestSample]) -> str:
    ordered = sorted(requests, key=lambda item: (item.arrival_ms, item.request_id))
    return sha256_text(
        canonical_json([request.to_dict() for request in ordered])
    )
