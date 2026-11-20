from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import RuleError
from .util import (
    load_json_strict,
    require_int,
    require_list,
    require_mapping,
    require_string,
    sha256_bytes,
)


LOOPBACK_HOST = "127.0.0.1"
SERVICE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class InstanceRule:
    service: str
    name: str
    host: str
    port: int
    weight: int
    failure_threshold: int
    cooldown_ms: int

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class ServiceRule:
    name: str
    concurrency_quota: int
    instances: tuple[InstanceRule, ...]

    @property
    def total_weight(self) -> int:
        return sum(item.weight for item in self.instances)


@dataclass(frozen=True)
class RouteRule:
    prefix: str
    service: str
    timeout_ms: int
    max_retries: int


@dataclass(frozen=True)
class RuleSet:
    schema_version: int
    rules_version: str
    source_sha256: str
    global_concurrency: int
    default_service_quota: int
    services: tuple[ServiceRule, ...]
    routes: tuple[RouteRule, ...]
    raw_document: dict[str, Any]

    def service(self, name: str) -> ServiceRule:
        for item in self.services:
            if item.name == name:
                return item
        raise KeyError(name)

    def match_route(self, path: str) -> RouteRule:
        candidates = [
            route for route in self.routes if path.startswith(route.prefix)
        ]
        if not candidates:
            raise RuleError(
                "request path does not match any route prefix",
                "$.path",
                {"path": path},
            )
        return max(candidates, key=lambda route: (len(route.prefix), route.prefix))

    def semantic_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "limits": {
                "global_concurrency": self.global_concurrency,
                "default_service_quota": self.default_service_quota,
            },
            "services": [
                {
                    "name": service.name,
                    "concurrency_quota": service.concurrency_quota,
                    "instances": [
                        {
                            "name": instance.name,
                            "host": instance.host,
                            "port": instance.port,
                            "weight": instance.weight,
                            "failure_threshold": instance.failure_threshold,
                            "cooldown_ms": instance.cooldown_ms,
                        }
                        for instance in service.instances
                    ],
                }
                for service in self.services
            ],
            "routes": [
                {
                    "prefix": route.prefix,
                    "service": route.service,
                    "timeout_ms": route.timeout_ms,
                    "max_retries": route.max_retries,
                }
                for route in self.routes
            ],
        }


def _instance_name(
    item: dict[str, Any],
    path: str,
    service: str,
    port: int,
) -> str:
    if "name" not in item:
        return f"{service}-{port}"
    return require_string(item["name"], f"{path}.name")


def parse_rules(raw: bytes, source: str) -> RuleSet:
    document = load_json_strict(raw, source)
    root = require_mapping(document, "$")
    schema_version = require_int(
        root.get("schema_version"), "$.schema_version", minimum=1, maximum=1
    )
    limits = require_mapping(root.get("limits"), "$.limits")
    global_concurrency = require_int(
        limits.get("global_concurrency"),
        "$.limits.global_concurrency",
        minimum=1,
        maximum=256,
    )
    default_service_quota = require_int(
        limits.get("default_service_quota", global_concurrency),
        "$.limits.default_service_quota",
        minimum=1,
        maximum=global_concurrency,
    )

    services_raw = require_list(root.get("services"), "$.services")
    if not services_raw:
        raise RuleError("at least one service is required", "$.services")

    seen_services: set[str] = set()
    seen_ports: dict[int, str] = {}
    services: list[ServiceRule] = []
    for service_index, service_raw in enumerate(services_raw):
        service_path = f"$.services[{service_index}]"
        service_obj = require_mapping(service_raw, service_path)
        name = require_string(service_obj.get("name"), f"{service_path}.name")
        if not SERVICE_NAME_RE.fullmatch(name):
            raise RuleError(
                "service name must match ^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
                f"{service_path}.name",
            )
        if name in seen_services:
            raise RuleError(
                "service name must be unique", f"{service_path}.name", {"name": name}
            )
        seen_services.add(name)
        quota = require_int(
            service_obj.get("concurrency_quota", default_service_quota),
            f"{service_path}.concurrency_quota",
            minimum=1,
            maximum=global_concurrency,
        )
        instances_raw = require_list(
            service_obj.get("instances"), f"{service_path}.instances"
        )
        if not instances_raw:
            raise RuleError(
                "at least one instance is required", f"{service_path}.instances"
            )
        instances: list[InstanceRule] = []
        seen_instance_names: set[str] = set()
        for instance_index, instance_raw in enumerate(instances_raw):
            instance_path = f"{service_path}.instances[{instance_index}]"
            instance_obj = require_mapping(instance_raw, instance_path)
            host = require_string(
                instance_obj.get("host", LOOPBACK_HOST), f"{instance_path}.host"
            )
            if host != LOOPBACK_HOST:
                raise RuleError(
                    "instance host must be the loopback literal 127.0.0.1",
                    f"{instance_path}.host",
                    {"host": host},
                )
            port = require_int(
                instance_obj.get("port"),
                f"{instance_path}.port",
                minimum=1,
                maximum=65535,
            )
            if port in seen_ports:
                raise RuleError(
                    "instance port must be globally unique",
                    f"{instance_path}.port",
                    {"port": port, "already_used_by": seen_ports[port]},
                )
            seen_ports[port] = f"{service_path}.instances[{instance_index}]"
            weight = require_int(
                instance_obj.get("weight"),
                f"{instance_path}.weight",
                minimum=0,
                maximum=1_000_000,
            )
            failure_threshold = require_int(
                instance_obj.get("failure_threshold"),
                f"{instance_path}.failure_threshold",
                minimum=1,
                maximum=1000,
            )
            cooldown_ms = require_int(
                instance_obj.get("cooldown_ms"),
                f"{instance_path}.cooldown_ms",
                minimum=1,
                maximum=3_600_000,
            )
            instance_name = _instance_name(instance_obj, instance_path, name, port)
            if instance_name in seen_instance_names:
                raise RuleError(
                    "instance name must be unique within a service",
                    f"{instance_path}.name",
                )
            seen_instance_names.add(instance_name)
            instances.append(
                InstanceRule(
                    service=name,
                    name=instance_name,
                    host=host,
                    port=port,
                    weight=weight,
                    failure_threshold=failure_threshold,
                    cooldown_ms=cooldown_ms,
                )
            )
        if sum(instance.weight for instance in instances) <= 0:
            raise RuleError(
                "sum of instance weights must be greater than zero",
                f"{service_path}.instances",
            )
        services.append(
            ServiceRule(
                name=name,
                concurrency_quota=quota,
                instances=tuple(sorted(instances, key=lambda item: item.port)),
            )
        )

    routes_raw = require_list(root.get("routes"), "$.routes")
    if not routes_raw:
        raise RuleError("at least one route is required", "$.routes")
    routes: list[RouteRule] = []
    seen_prefixes: set[str] = set()
    for route_index, route_raw in enumerate(routes_raw):
        route_path = f"$.routes[{route_index}]"
        route_obj = require_mapping(route_raw, route_path)
        prefix = require_string(route_obj.get("prefix"), f"{route_path}.prefix")
        if not prefix.startswith("/"):
            raise RuleError(
                "route prefix must start with /", f"{route_path}.prefix"
            )
        if "?" in prefix or "#" in prefix:
            raise RuleError(
                "route prefix must not contain query or fragment",
                f"{route_path}.prefix",
            )
        if prefix in seen_prefixes:
            raise RuleError(
                "route prefix must be unique", f"{route_path}.prefix", {"prefix": prefix}
            )
        seen_prefixes.add(prefix)
        service_name = require_string(
            route_obj.get("service"), f"{route_path}.service"
        )
        if service_name not in seen_services:
            raise RuleError(
                "route target service does not exist",
                f"{route_path}.service",
                {"service": service_name},
            )
        timeout_ms = require_int(
            route_obj.get("timeout_ms"),
            f"{route_path}.timeout_ms",
            minimum=1,
            maximum=600_000,
        )
        max_retries = require_int(
            route_obj.get("max_retries"),
            f"{route_path}.max_retries",
            minimum=0,
            maximum=10,
        )
        routes.append(
            RouteRule(
                prefix=prefix,
                service=service_name,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
            )
        )

    rules_version = sha256_bytes(raw)
    return RuleSet(
        schema_version=schema_version,
        rules_version=f"sha256:{rules_version}",
        source_sha256=rules_version,
        global_concurrency=global_concurrency,
        default_service_quota=default_service_quota,
        services=tuple(services),
        routes=tuple(sorted(routes, key=lambda item: (-len(item.prefix), item.prefix))),
        raw_document=root,
    )


def load_rules(path: str | Path, raw_override: bytes | None = None) -> RuleSet:
    rule_path = Path(path)
    try:
        raw = raw_override if raw_override is not None else rule_path.read_bytes()
    except OSError as exc:
        raise RuleError(
            "unable to read rules file",
            "$",
            {"source": str(rule_path), "reason": str(exc)},
        ) from exc
    return parse_rules(raw, str(rule_path))


@dataclass(frozen=True)
class RequestSpec:
    index: int
    request_id: str
    method: str
    path: str
    route: RouteRule


def parse_requests(raw: bytes, source: str, rules: RuleSet) -> tuple[RequestSpec, ...]:
    document = load_json_strict(raw, source)
    root = require_mapping(document, "$")
    requests_raw = root.get("requests")
    if not isinstance(requests_raw, list) or not requests_raw:
        raise RuleError("requests must be a non-empty JSON array", "$.requests")
    seen_ids: set[str] = set()
    result: list[RequestSpec] = []
    for index, request_raw in enumerate(requests_raw):
        path = f"$.requests[{index}]"
        obj = require_mapping(request_raw, path)
        request_id = require_string(obj.get("id"), f"{path}.id")
        if request_id in seen_ids:
            raise RuleError(
                "request id must be unique", f"{path}.id", {"id": request_id}
            )
        seen_ids.add(request_id)
        method = require_string(obj.get("method", "GET"), f"{path}.method").upper()
        if method != "GET":
            raise RuleError(
                "the deterministic sample supports GET only", f"{path}.method"
            )
        request_path = require_string(obj.get("path"), f"{path}.path")
        if not request_path.startswith("/"):
            raise RuleError("request path must start with /", f"{path}.path")
        try:
            route = rules.match_route(request_path)
        except RuleError as exc:
            raise RuleError(
                "request path does not match any route prefix",
                f"{path}.path",
                {"path": request_path},
            ) from exc
        result.append(
            RequestSpec(
                index=index,
                request_id=request_id,
                method=method,
                path=request_path,
                route=route,
            )
        )
    return tuple(result)


def load_requests(path: str | Path, rules: RuleSet) -> tuple[RequestSpec, ...]:
    request_path = Path(path)
    try:
        raw = request_path.read_bytes()
    except OSError as exc:
        raise RuleError(
            "unable to read requests file",
            "$",
            {"source": str(request_path), "reason": str(exc)},
        ) from exc
    return parse_requests(raw, str(request_path), rules)


def route_map(rules: RuleSet | dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(rules, RuleSet):
        document: dict[str, Any] = rules.raw_document
    else:
        document = rules
    output: dict[str, dict[str, Any]] = {}
    for route in document.get("routes", []):
        if isinstance(route, dict) and isinstance(route.get("prefix"), str):
            output[route["prefix"]] = dict(route)
    return output


def diff_rules(
    old: RuleSet | dict[str, Any], new: RuleSet | dict[str, Any]
) -> dict[str, Any]:
    old_map = route_map(old)
    new_map = route_map(new)
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    changed = sorted(
        prefix
        for prefix in set(old_map) & set(new_map)
        if old_map[prefix] != new_map[prefix]
    )
    return {
        "added_prefixes": added,
        "removed_prefixes": removed,
        "changed_prefixes": changed,
        "routes": [
            {
                "prefix": prefix,
                "before": old_map.get(prefix),
                "after": new_map.get(prefix),
            }
            for prefix in added + removed + changed
        ],
    }


def all_instances(rules: RuleSet) -> Iterable[InstanceRule]:
    for service in rules.services:
        yield from service.instances
