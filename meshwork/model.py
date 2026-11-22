from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import sha256_hex
from .errors import MeshError, spec_error

LOOPBACK_HOST = "127.0.0.1"
SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BehaviorRule:
    request_prefix: str
    outcome: str
    latency_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_prefix": self.request_prefix,
            "outcome": self.outcome,
            "latency_class": self.latency_class,
        }


@dataclass(frozen=True)
class BehaviorSpec:
    fast_ms: int
    slow_ms: int
    success_percent: int
    slow_percent: int
    error_percent: int
    rules: tuple[BehaviorRule, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_ms": self.fast_ms,
            "slow_ms": self.slow_ms,
            "success_percent": self.success_percent,
            "slow_percent": self.slow_percent,
            "error_percent": self.error_percent,
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass(frozen=True)
class InstanceSpec:
    id: str
    host: str
    port: int
    weight: int
    seed: int
    behavior: BehaviorSpec
    order: int
    service_id: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "seed": self.seed,
            "behavior": self.behavior.to_dict(),
            "order": self.order,
            "service": self.service_id,
        }


@dataclass(frozen=True)
class ServiceSpec:
    id: str
    quota: int
    instances: tuple[InstanceSpec, ...]
    order: int

    def instance_by_id(self) -> dict[str, InstanceSpec]:
        return {instance.id: instance for instance in self.instances}


@dataclass(frozen=True)
class RouteRule:
    service: str
    weights: dict[str, int] | None


@dataclass(frozen=True)
class RouteRevision:
    revision: int
    rules: dict[str, RouteRule]
    description: str | None = None


@dataclass(frozen=True)
class BreakerConfig:
    failure_threshold: int
    cooldown_ms: int
    half_open_probes: int


@dataclass(frozen=True)
class SessionConfig:
    scale: int
    global_concurrency: int
    attempt_timeout_ms: int
    retry_backoff_ms: tuple[int, ...]
    breaker: BreakerConfig

    @property
    def max_attempts(self) -> int:
        return 1 + len(self.retry_backoff_ms)


@dataclass(frozen=True)
class MeshSpec:
    schema_version: int
    session: SessionConfig
    services: dict[str, ServiceSpec]
    service_order: tuple[str, ...]
    route_revisions: tuple[RouteRevision, ...]
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def ordered_services(self) -> tuple[ServiceSpec, ...]:
        return tuple(self.services[service_id] for service_id in self.service_order)

    def revision_map(self) -> dict[int, RouteRevision]:
        return {revision.revision: revision for revision in self.route_revisions}

    def active_revision(self, requested: int | None = None) -> RouteRevision:
        if requested is None:
            return self.route_revisions[-1]
        by_revision = self.revision_map()
        if requested not in by_revision:
            field = "revision"
            location = f"$.route_revisions[revision={requested}]"
            raise spec_error(
                f"route revision {requested} does not exist",
                field,
                location,
            )
        return by_revision[requested]

    def effective_weights(self, service_id: str, revision: RouteRevision) -> dict[str, int]:
        service = self.services[service_id]
        rule = revision.rules.get(service_id)
        if rule is None or rule.weights is None:
            return {instance.id: instance.weight for instance in service.instances}
        return {instance.id: int(rule.weights.get(instance.id, 0)) for instance in service.instances}

    def instance_version(self) -> str:
        payload = [
            {
                "service": service.id,
                "quota": service.quota,
                "order": service.order,
                "instances": [instance.identity_dict() for instance in service.instances],
            }
            for service in self.ordered_services()
        ]
        return sha256_hex({"kind": "instance-set-v1", "services": payload})

    def config_digest(self) -> str:
        return sha256_hex(self.raw)


@dataclass(frozen=True)
class RequestSpec:
    id: str
    service: str
    payload: dict[str, Any]
    order: int
    source_path: str | None = None


def _location(path: str, key: str | None = None) -> str:
    if key is None:
        return path
    if path.endswith("]"):
        return f"{path}.{key}"
    return f"{path}.{key}"


def _field_from_location(location: str) -> str:
    field = location[2:] if location.startswith("$.") else location
    return field.replace("[", ".").replace("]", "").replace("..", ".")


def _raise(message: str, location: str) -> None:
    raise spec_error(message, _field_from_location(location), location)


def _as_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _raise("must be a JSON object", location)
    return value


def _as_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _raise("must be a JSON array", location)
    return value


def _as_str(value: Any, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _raise("must be a string", location)
    if nonempty and not value.strip():
        _raise("must not be empty", location)
    return value


def _as_int(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise("must be an integer", location)
    if minimum is not None and value < minimum:
        _raise(f"must be >= {minimum}", location)
    return value


def _optional_int(obj: dict[str, Any], key: str, location: str, default: int) -> int:
    if key not in obj:
        return default
    return _as_int(obj[key], _location(location, key))


def load_json_file(path: str | Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise MeshError(
            "E_SPEC",
            f"cannot read JSON file: {exc}",
            field="file",
            location=str(source),
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MeshError(
            "E_SPEC",
            f"invalid JSON: {exc.msg}",
            field="json",
            location=f"{source}:{exc.lineno}:{exc.colno}",
        ) from exc


def _parse_behavior(value: Any, location: str) -> BehaviorSpec:
    obj = _as_object(value, location)
    fast_ms = _as_int(obj.get("fast_ms"), _location(location, "fast_ms"), minimum=0)
    slow_ms = _as_int(obj.get("slow_ms"), _location(location, "slow_ms"), minimum=0)
    if slow_ms < fast_ms:
        _raise("must be >= fast_ms", _location(location, "slow_ms"))
    success = _as_int(
        obj.get("success_percent", 0),
        _location(location, "success_percent"),
        minimum=0,
    )
    slow = _as_int(
        obj.get("slow_percent", 0),
        _location(location, "slow_percent"),
        minimum=0,
    )
    error = _as_int(
        obj.get("error_percent", 0),
        _location(location, "error_percent"),
        minimum=0,
    )
    if success + slow + error != 100:
        _raise(
            "success_percent + slow_percent + error_percent must equal 100",
            location,
        )
    raw_rules = obj.get("rules", [])
    rules: list[BehaviorRule] = []
    for rule_index, raw_rule in enumerate(_as_list(raw_rules, _location(location, "rules"))):
        rule_location = _location(location, f"rules[{rule_index}]")
        rule_obj = _as_object(raw_rule, rule_location)
        prefix = _as_str(
            rule_obj.get("request_prefix"),
            _location(rule_location, "request_prefix"),
        )
        outcome = _as_str(rule_obj.get("outcome"), _location(rule_location, "outcome"))
        if outcome not in {"success", "error"}:
            _raise("must be 'success' or 'error'", _location(rule_location, "outcome"))
        latency_class = _as_str(
            rule_obj.get("latency_class", "fast"),
            _location(rule_location, "latency_class"),
        )
        if latency_class not in {"fast", "slow"}:
            _raise(
                "must be 'fast' or 'slow'",
                _location(rule_location, "latency_class"),
            )
        rules.append(BehaviorRule(prefix, outcome, latency_class))
    return BehaviorSpec(fast_ms, slow_ms, success, slow, error, tuple(rules))


def _parse_service(
    value: Any,
    location: str,
    order: int,
    service_ids: set[str],
    instance_ids: set[str],
    ports: dict[int, str],
) -> ServiceSpec:
    obj = _as_object(value, location)
    service_id = _as_str(obj.get("id"), _location(location, "id"))
    if service_id in service_ids:
        _raise(f"duplicate service id {service_id!r}", _location(location, "id"))
    service_ids.add(service_id)
    quota = _as_int(obj.get("quota"), _location(location, "quota"), minimum=1)
    raw_instances = _as_list(obj.get("instances"), _location(location, "instances"))
    if not raw_instances:
        _raise("must contain at least one instance", _location(location, "instances"))
    instances: list[InstanceSpec] = []
    for instance_index, raw_instance in enumerate(raw_instances):
        instance_location = _location(location, f"instances[{instance_index}]")
        instance_obj = _as_object(raw_instance, instance_location)
        instance_id = _as_str(
            instance_obj.get("id"), _location(instance_location, "id")
        )
        if instance_id in instance_ids:
            _raise(
                f"duplicate instance id {instance_id!r}",
                _location(instance_location, "id"),
            )
        instance_ids.add(instance_id)
        host = _as_str(instance_obj.get("host"), _location(instance_location, "host"))
        if host != LOOPBACK_HOST:
            _raise(
                "host must be the literal loopback address 127.0.0.1",
                _location(instance_location, "host"),
            )
        port = _as_int(
            instance_obj.get("port"),
            _location(instance_location, "port"),
            minimum=1,
        )
        if port > 65535:
            _raise("must be <= 65535", _location(instance_location, "port"))
        if port in ports:
            _raise(
                f"port {port} is already used by instance {ports[port]!r}",
                _location(instance_location, "port"),
            )
        ports[port] = instance_id
        weight = _as_int(
            instance_obj.get("weight"),
            _location(instance_location, "weight"),
            minimum=0,
        )
        seed = _as_int(instance_obj.get("seed"), _location(instance_location, "seed"))
        behavior = _parse_behavior(
            instance_obj.get("behavior"), _location(instance_location, "behavior")
        )
        instances.append(
            InstanceSpec(
                id=instance_id,
                host=host,
                port=port,
                weight=weight,
                seed=seed,
                behavior=behavior,
                order=instance_index,
                service_id=service_id,
            )
        )
    if not any(instance.weight > 0 for instance in instances):
        _raise(
            "at least one instance weight must be positive",
            _location(location, "instances"),
        )
    return ServiceSpec(service_id, quota, tuple(instances), order)


def _parse_route_revision(
    value: Any,
    location: str,
    services: dict[str, ServiceSpec],
    previous_revision: int | None,
) -> RouteRevision:
    obj = _as_object(value, location)
    revision = _as_int(obj.get("revision"), _location(location, "revision"), minimum=1)
    if previous_revision is not None and revision <= previous_revision:
        _raise(
            f"revision must be strictly increasing (previous {previous_revision}, got {revision})",
            _location(location, "revision"),
        )
    raw_rules = _as_list(obj.get("rules"), _location(location, "rules"))
    if not raw_rules:
        _raise("must contain at least one route rule", _location(location, "rules"))
    rules: dict[str, RouteRule] = {}
    for rule_index, raw_rule in enumerate(raw_rules):
        rule_location = _location(location, f"rules[{rule_index}]")
        rule_obj = _as_object(raw_rule, rule_location)
        service_id = _as_str(
            rule_obj.get("service"), _location(rule_location, "service")
        )
        if service_id not in services:
            _raise(
                f"unknown service {service_id!r}",
                _location(rule_location, "service"),
            )
        if service_id in rules:
            _raise(
                f"duplicate route rule for service {service_id!r}",
                _location(rule_location, "service"),
            )
        weights: dict[str, int] | None = None
        if "weights" in rule_obj:
            weights_location = _location(rule_location, "weights")
            weights_obj = _as_object(rule_obj["weights"], weights_location)
            instance_map = services[service_id].instance_by_id()
            weights = {}
            for instance_id, raw_weight in weights_obj.items():
                if instance_id not in instance_map:
                    _raise(
                        f"unknown instance {instance_id!r} for service {service_id!r}",
                        _location(weights_location, instance_id),
                    )
                weights[instance_id] = _as_int(
                    raw_weight,
                    _location(weights_location, instance_id),
                    minimum=0,
                )
            if not any(weight > 0 for weight in weights.values()):
                _raise(
                    "at least one route weight must be positive",
                    weights_location,
                )
        rules[service_id] = RouteRule(service_id, weights)
    description_value = obj.get("description")
    description = None
    if description_value is not None:
        description = _as_str(description_value, _location(location, "description"), nonempty=False)
    return RouteRevision(revision, rules, description)


def parse_spec(raw: Any, source_path: str | None = None) -> MeshSpec:
    root = _as_object(raw, "$")
    schema_version = _as_int(
        root.get("schema_version"), "$.schema_version", minimum=1
    )
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        _raise(
            f"unsupported schema_version {schema_version}; expected {SUPPORTED_SCHEMA_VERSION}",
            "$.schema_version",
        )
    session_obj = _as_object(root.get("session"), "$.session")
    scale = _as_int(session_obj.get("scale"), "$.session.scale", minimum=1)
    global_concurrency = _as_int(
        session_obj.get("global_concurrency"),
        "$.session.global_concurrency",
        minimum=1,
    )
    attempt_timeout_ms = _as_int(
        session_obj.get("attempt_timeout_ms"),
        "$.session.attempt_timeout_ms",
        minimum=1,
    )
    backoff_values = _as_list(
        session_obj.get("retry_backoff_ms"), "$.session.retry_backoff_ms"
    )
    backoffs: list[int] = []
    for index, value in enumerate(backoff_values):
        backoffs.append(
            _as_int(
                value,
                f"$.session.retry_backoff_ms[{index}]",
                minimum=0,
            )
        )
    breaker_obj = _as_object(session_obj.get("breaker"), "$.session.breaker")
    breaker = BreakerConfig(
        failure_threshold=_as_int(
            breaker_obj.get("failure_threshold"),
            "$.session.breaker.failure_threshold",
            minimum=1,
        ),
        cooldown_ms=_as_int(
            breaker_obj.get("cooldown_ms"),
            "$.session.breaker.cooldown_ms",
            minimum=0,
        ),
        half_open_probes=_as_int(
            breaker_obj.get("half_open_probes"),
            "$.session.breaker.half_open_probes",
            minimum=1,
        ),
    )
    raw_services = _as_list(root.get("services"), "$.services")
    if not raw_services:
        _raise("must contain at least one service", "$.services")
    service_ids: set[str] = set()
    instance_ids: set[str] = set()
    ports: dict[int, str] = {}
    services: dict[str, ServiceSpec] = {}
    order: list[str] = []
    for service_index, raw_service in enumerate(raw_services):
        service = _parse_service(
            raw_service,
            f"$.services[{service_index}]",
            service_index,
            service_ids,
            instance_ids,
            ports,
        )
        services[service.id] = service
        order.append(service.id)
    raw_revisions = _as_list(root.get("route_revisions"), "$.route_revisions")
    if not raw_revisions:
        _raise("must contain at least one revision", "$.route_revisions")
    revisions: list[RouteRevision] = []
    previous: int | None = None
    for revision_index, raw_revision in enumerate(raw_revisions):
        revision = _parse_route_revision(
            raw_revision,
            f"$.route_revisions[{revision_index}]",
            services,
            previous,
        )
        revisions.append(revision)
        previous = revision.revision
    session = SessionConfig(
        scale,
        global_concurrency,
        attempt_timeout_ms,
        tuple(backoffs),
        breaker,
    )
    return MeshSpec(
        schema_version,
        session,
        services,
        tuple(order),
        tuple(revisions),
        source_path,
        raw,
    )


def load_spec(path: str | Path) -> MeshSpec:
    raw = load_json_file(path)
    return parse_spec(raw, source_path=str(Path(path)))


def parse_requests(raw: Any, known_services: Iterable[str], source_path: str | None = None) -> list[RequestSpec]:
    if isinstance(raw, dict):
        raw_requests = _as_list(raw.get("requests"), "$.requests")
        root_location = "$.requests"
    else:
        raw_requests = _as_list(raw, "$")
        root_location = "$"
    known = set(known_services)
    ids: set[str] = set()
    requests: list[RequestSpec] = []
    for index, raw_request in enumerate(raw_requests):
        location = f"{root_location}[{index}]"
        obj = _as_object(raw_request, location)
        request_id = _as_str(obj.get("id"), _location(location, "id"))
        if request_id in ids:
            _raise(f"duplicate request id {request_id!r}", _location(location, "id"))
        ids.add(request_id)
        service_id = _as_str(obj.get("service"), _location(location, "service"))
        if service_id not in known:
            _raise(f"unknown service {service_id!r}", _location(location, "service"))
        payload_value = obj.get("payload", {})
        payload = _as_object(payload_value, _location(location, "payload"))
        requests.append(RequestSpec(request_id, service_id, payload, index, source_path))
    return requests


def load_requests(path: str | Path, known_services: Iterable[str]) -> list[RequestSpec]:
    raw = load_json_file(path)
    return parse_requests(raw, known_services, source_path=str(Path(path)))


def requests_digest(requests: Iterable[RequestSpec]) -> str:
    payload = [
        {"id": request.id, "service": request.service, "payload": request.payload}
        for request in sorted(requests, key=lambda item: (item.order, item.id))
    ]
    return sha256_hex({"kind": "request-sequence-v1", "requests": payload})
