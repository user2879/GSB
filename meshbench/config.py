"""Service/route/simulated-upstream configuration with strong validation.

The JSON document is parsed and *fully* validated before any dispatch record is
created.  Any malformed declaration raises :class:`RuleError` (``E_RULE``) and
the caller must reject the whole document, leaving the ledger untouched.
"""

import json
from dataclasses import dataclass, field

from .errors import RuleError

LOOPBACK = "127.0.0.1"
MAX_BUDGET_PERMYRIAD = 10000
BUCKET_SPAN = 10000


def _is_int(value):
    # bool is a subclass of int; explicitly reject booleans as numbers.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class UpstreamSpec:
    """Deterministic simulated-upstream behaviour for one instance."""

    process_ms: int
    error_permyriad: int
    slow_permyriad: int
    slow_extra_ms: int


@dataclass(frozen=True)
class InstanceConfig:
    instance_id: str
    host: str
    port: int
    weight: int
    failure_threshold: int
    cooldown_ms: int
    half_open_probes: int
    upstream: UpstreamSpec


@dataclass(frozen=True)
class ServiceConfig:
    service_id: str
    concurrency_quota: int
    instances: tuple

    def instance(self, instance_id):
        for inst in self.instances:
            if inst.instance_id == instance_id:
                return inst
        raise KeyError(instance_id)


@dataclass(frozen=True)
class RouteConfig:
    route_id: str
    prefix: str
    target_service: str
    attempt_timeout_ms: int
    retries: int
    window_minutes: int
    budget_permyriad: int
    min_samples: int
    slow_threshold_ms: int


@dataclass(frozen=True)
class MeshConfig:
    version: int
    global_concurrency: int
    services: dict = field(default_factory=dict)
    routes: dict = field(default_factory=dict)

    def match_route(self, path):
        """Longest-prefix match; ties broken by the smallest route_id."""
        candidates = [r for r in self.routes.values() if path.startswith(r.prefix)]
        if not candidates:
            raise RuleError("no route matches path", path=path)
        candidates.sort(key=lambda r: (-len(r.prefix), r.route_id))
        return candidates[0]


def _require(condition, message, **context):
    if not condition:
        raise RuleError(message, **context)


def _parse_upstream(raw, where):
    _require(isinstance(raw, dict), "upstream must be an object", where=where)
    process_ms = raw.get("process_ms")
    error_pm = raw.get("error_permyriad", 0)
    slow_pm = raw.get("slow_permyriad", 0)
    slow_extra = raw.get("slow_extra_ms", 0)
    _require(_is_int(process_ms) and process_ms >= 0, "process_ms must be >=0 int", where=where)
    _require(_is_int(error_pm) and 0 <= error_pm <= MAX_BUDGET_PERMYRIAD,
             "error_permyriad must be 0..10000", where=where)
    _require(_is_int(slow_pm) and 0 <= slow_pm <= MAX_BUDGET_PERMYRIAD,
             "slow_permyriad must be 0..10000", where=where)
    _require(_is_int(slow_extra) and slow_extra >= 0, "slow_extra_ms must be >=0 int", where=where)
    _require(error_pm + slow_pm <= MAX_BUDGET_PERMYRIAD,
             "error_permyriad + slow_permyriad must not exceed 10000", where=where)
    return UpstreamSpec(process_ms, error_pm, slow_pm, slow_extra)


def _parse_instance(raw, service_id, seen_ports):
    where = f"service={service_id} instance={raw.get('instance_id') if isinstance(raw, dict) else '?'}"
    _require(isinstance(raw, dict), "instance must be an object", where=where)
    instance_id = raw.get("instance_id")
    host = raw.get("host")
    port = raw.get("port")
    weight = raw.get("weight")
    threshold = raw.get("failure_threshold")
    cooldown = raw.get("cooldown_ms")
    probes = raw.get("half_open_probes", 1)
    _require(isinstance(instance_id, str) and instance_id.strip(), "bad instance_id", where=where)
    _require(host == LOOPBACK, "host must be the loopback literal 127.0.0.1", where=where)
    _require(_is_int(port) and 1 <= port <= 65535, "port must be 1..65535", where=where)
    _require(port not in seen_ports, "duplicate instance port across services", where=where, port=port)
    _require(_is_int(weight) and weight >= 0, "weight must be a non-negative int", where=where)
    _require(_is_int(threshold) and threshold >= 1, "failure_threshold must be >=1", where=where)
    _require(_is_int(cooldown) and cooldown >= 0, "cooldown_ms must be >=0 int", where=where)
    _require(_is_int(probes) and probes >= 1, "half_open_probes must be >=1", where=where)
    upstream = _parse_upstream(raw.get("upstream"), where)
    seen_ports.add(port)
    return InstanceConfig(instance_id, host, port, weight, threshold, cooldown, probes, upstream)


def _parse_service(raw, seen_ports):
    _require(isinstance(raw, dict), "service must be an object")
    service_id = raw.get("service_id")
    _require(isinstance(service_id, str) and service_id.strip(), "bad service_id")
    raw_instances = raw.get("instances")
    _require(isinstance(raw_instances, list) and raw_instances, "service needs >=1 instance",
             service=service_id)
    instances = tuple(_parse_instance(item, service_id, seen_ports) for item in raw_instances)
    local_ids = [inst.instance_id for inst in instances]
    _require(len(set(local_ids)) == len(local_ids), "duplicate instance_id in service",
             service=service_id)
    total_weight = sum(inst.weight for inst in instances)
    _require(total_weight > 0, "sum of instance weights must be non-zero", service=service_id)
    quota = raw.get("concurrency_quota")
    _require(_is_int(quota) and quota >= 1, "concurrency_quota must be >=1", service=service_id)
    return ServiceConfig(service_id, quota, instances)


def _parse_route(raw, service_ids):
    _require(isinstance(raw, dict), "route must be an object")
    route_id = raw.get("route_id")
    prefix = raw.get("prefix")
    target = raw.get("target_service")
    timeout = raw.get("attempt_timeout_ms")
    retries = raw.get("retries")
    window = raw.get("window_minutes")
    budget = raw.get("budget_permyriad")
    min_samples = raw.get("min_samples")
    slow = raw.get("slow_threshold_ms")
    _require(isinstance(route_id, str) and route_id.strip(), "bad route_id")
    _require(isinstance(prefix, str) and prefix.strip(), "bad prefix", route=route_id)
    _require(target in service_ids, "route target_service must reference a declared service",
             route=route_id)
    _require(_is_int(timeout) and timeout >= 1, "attempt_timeout_ms must be positive int", route=route_id)
    _require(_is_int(retries) and retries >= 0, "retries must be >=0 int", route=route_id)
    _require(_is_int(window) and window >= 1, "window_minutes must be positive int", route=route_id)
    _require(_is_int(budget) and 0 <= budget <= MAX_BUDGET_PERMYRIAD,
             "budget_permyriad must be 0..10000", route=route_id)
    _require(_is_int(min_samples) and min_samples >= 1, "min_samples must be >=1 int", route=route_id)
    _require(_is_int(slow) and slow >= 1, "slow_threshold_ms must be positive int", route=route_id)
    return RouteConfig(route_id, prefix, target, timeout, retries, window, budget, min_samples, slow)


def parse_config(document) -> MeshConfig:
    """Validate a decoded JSON document and return an immutable MeshConfig."""
    _require(isinstance(document, dict), "top-level config must be an object")
    version = document.get("version", 1)
    _require(_is_int(version) and version >= 1, "version must be a positive int")
    global_concurrency = document.get("global_concurrency")
    _require(_is_int(global_concurrency) and global_concurrency >= 1,
             "global_concurrency must be >=1")
    raw_services = document.get("services")
    _require(isinstance(raw_services, list) and raw_services, "services must be a non-empty list")
    seen_ports = set()
    services = {}
    for raw in raw_services:
        service = _parse_service(raw, seen_ports)
        _require(service.service_id not in services, "duplicate service_id",
                 service=service.service_id)
        services[service.service_id] = service
    raw_routes = document.get("routes")
    _require(isinstance(raw_routes, list) and raw_routes, "routes must be a non-empty list")
    routes = {}
    for raw in raw_routes:
        route = _parse_route(raw, set(services))
        _require(route.route_id not in routes, "duplicate route_id", route=route.route_id)
        _require(not any(existing.prefix == route.prefix for existing in routes.values()),
                 "duplicate route prefix", route=route.route_id)
        routes[route.route_id] = route
    return MeshConfig(version, global_concurrency, services, routes)


def load_config(path) -> MeshConfig:
    """Read and validate a local JSON rule file; raise RuleError on any defect."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError as exc:
        raise RuleError(f"rule file not found: {path}", path=path) from exc
    except json.JSONDecodeError as exc:
        raise RuleError("rule file is not valid JSON", path=path, line=exc.lineno,
                        column=exc.colno) from exc
    return parse_config(document)
