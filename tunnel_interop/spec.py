"""Local JSON declaration loading and fail-fast validation."""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import E_SPEC, fail
from .util import sha256_json


_MISSING = object()


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    host: str
    port: int
    mtu: int
    encapsulation_overhead: int
    replay_window: int
    seed: int

    @property
    def effective_payload_limit(self) -> int:
        return self.mtu - self.encapsulation_overhead

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.endpoint_id,
            "host": self.host,
            "port": self.port,
            "mtu": self.mtu,
            "encapsulation_overhead": self.encapsulation_overhead,
            "replay_window": self.replay_window,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Route:
    prefix: str
    network: ipaddress.IPv4Network | ipaddress.IPv6Network
    next_hop: str
    priority: int
    tunnel_id: str

    def canonical(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "next_hop": self.next_hop,
            "priority": self.priority,
            "tunnel_id": self.tunnel_id,
        }


@dataclass(frozen=True)
class FragmentPolicy:
    fragment_size: int
    max_retries: int
    reassembly_timeout_ms: int

    def canonical(self) -> dict[str, Any]:
        return {
            "fragment_size": self.fragment_size,
            "max_retries": self.max_retries,
            "reassembly_timeout_ms": self.reassembly_timeout_ms,
        }


@dataclass(frozen=True)
class RuntimePolicy:
    global_inflight_limit: int = 4
    per_tunnel_inflight_quota: int = 2
    ack_timeout_ms: int = 500
    fragment_format: str = "f3"
    max_message_bytes: int = 8 * 1024 * 1024
    receiver_idle_timeout_ms: int = 10_000

    def canonical(self) -> dict[str, Any]:
        return {
            "global_inflight_limit": self.global_inflight_limit,
            "per_tunnel_inflight_quota": self.per_tunnel_inflight_quota,
            "ack_timeout_ms": self.ack_timeout_ms,
            "fragment_format": self.fragment_format,
            "max_message_bytes": self.max_message_bytes,
            "receiver_idle_timeout_ms": self.receiver_idle_timeout_ms,
        }


@dataclass(frozen=True)
class SamplePolicy:
    bytes_per_message: int | None = None

    def message_size(self, fragment_size: int) -> int:
        return self.bytes_per_message if self.bytes_per_message is not None else 3 * fragment_size + 29

    def canonical(self) -> dict[str, Any]:
        return {"bytes_per_message": self.bytes_per_message}


@dataclass(frozen=True)
class Versions:
    declaration: int = 1
    endpoint_set: int = 1
    route_table: int = 1
    fragment_policy: int = 1

    def canonical(self) -> dict[str, Any]:
        return {
            "declaration": self.declaration,
            "endpoint_set": self.endpoint_set,
            "route_table": self.route_table,
            "fragment_policy": self.fragment_policy,
        }


@dataclass(frozen=True)
class TunnelSpec:
    source_path: str
    endpoints: tuple[Endpoint, ...]
    routes: tuple[Route, ...]
    fragment_policy: FragmentPolicy
    runtime: RuntimePolicy
    samples: SamplePolicy
    versions: Versions

    @property
    def endpoint_map(self) -> dict[str, Endpoint]:
        return {item.endpoint_id: item for item in self.endpoints}

    @property
    def route_map(self) -> dict[str, Route]:
        return {item.tunnel_id: item for item in self.routes}

    @property
    def effective_payload_limit(self) -> int:
        return min(item.effective_payload_limit for item in self.endpoints)

    def endpoint_fingerprint(self) -> str:
        return sha256_json([item.canonical() for item in sorted(self.endpoints, key=lambda ep: ep.endpoint_id)])

    def route_fingerprint(self) -> str:
        return sha256_json([item.canonical() for item in sorted(self.routes, key=lambda route: route.tunnel_id)])

    def fragment_fingerprint(self) -> str:
        return sha256_json(self.fragment_policy.canonical())

    def declaration_fingerprint(self) -> str:
        return sha256_json(
            {
                "endpoints": [item.canonical() for item in sorted(self.endpoints, key=lambda ep: ep.endpoint_id)],
                "routes": [item.canonical() for item in sorted(self.routes, key=lambda route: route.tunnel_id)],
                "fragment_policy": self.fragment_policy.canonical(),
                "runtime": self.runtime.canonical(),
                "samples": self.samples.canonical(),
            }
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if not _is_int(value):
        fail(E_SPEC, path, "value must be an integer")
    if positive and value <= 0:
        fail(E_SPEC, path, "value must be a positive integer", actual=value)
    if nonnegative and value < 0:
        fail(E_SPEC, path, "value must be a non-negative integer", actual=value)
    return value


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        fail(E_SPEC, path, "value must be a non-empty string")
    return value


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(E_SPEC, path, "value must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        fail(E_SPEC, path, "value must be an array")
    return value


def _field(obj: Mapping[str, Any], name: str, path: str, default: Any = _MISSING) -> Any:
    if name in obj:
        return obj[name]
    if default is not _MISSING:
        return default
    fail(E_SPEC, f"{path}.{name}", "required field is missing")


def load_spec(path: str | Path, *, reference_error_code: str = E_SPEC) -> TunnelSpec:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
    except OSError as exc:
        fail(E_SPEC, "$", "cannot read declaration file", error=str(exc))
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        fail(E_SPEC, "$", "declaration is not valid UTF-8", error=str(exc))
    except json.JSONDecodeError as exc:
        fail(E_SPEC, "$", "declaration is not valid JSON", error=str(exc), line=exc.lineno, column=exc.colno)
    return load_spec_mapping(raw, source_path=str(source), reference_error_code=reference_error_code)


def load_spec_mapping(
    raw: Any,
    *,
    source_path: str = "<memory>",
    reference_error_code: str = E_SPEC,
) -> TunnelSpec:
    root = _require_object(raw, "$")

    endpoint_values = _require_list(_field(root, "endpoints", "$"), "$.endpoints")
    if not endpoint_values:
        fail(E_SPEC, "$.endpoints", "at least one endpoint is required")

    endpoints: list[Endpoint] = []
    seen_endpoint_ids: dict[str, int] = {}
    seen_ports: dict[int, int] = {}
    for index, item in enumerate(endpoint_values):
        path = f"$.endpoints[{index}]"
        obj = _require_object(item, path)
        endpoint_id = _require_text(_field(obj, "id", path), f"{path}.id")
        if endpoint_id in seen_endpoint_ids:
            fail(
                E_SPEC,
                f"{path}.id",
                "endpoint id must be unique",
                value=endpoint_id,
                first_index=seen_endpoint_ids[endpoint_id],
            )
        seen_endpoint_ids[endpoint_id] = index

        host = _require_text(_field(obj, "host", path, "127.0.0.1"), f"{path}.host")
        if host != "127.0.0.1":
            fail(E_SPEC, f"{path}.host", "host must be the loopback address 127.0.0.1", actual=host)

        port = _require_int(_field(obj, "port", path), f"{path}.port")
        if port < 1 or port > 65535:
            fail(E_SPEC, f"{path}.port", "port must be in [1, 65535]", actual=port)
        if port in seen_ports:
            fail(
                E_SPEC,
                f"{path}.port",
                "loopback UDP port must be globally unique",
                actual=port,
                first_index=seen_ports[port],
            )
        seen_ports[port] = index

        mtu = _require_int(_field(obj, "mtu", path), f"{path}.mtu", positive=True)
        overhead = _require_int(
            _field(obj, "encapsulation_overhead", path),
            f"{path}.encapsulation_overhead",
            positive=True,
        )
        if overhead >= mtu:
            fail(
                E_SPEC,
                f"{path}.encapsulation_overhead",
                "encapsulation overhead must be strictly smaller than MTU",
                overhead=overhead,
                mtu=mtu,
            )
        replay_window = _require_int(
            _field(obj, "replay_window", path),
            f"{path}.replay_window",
            positive=True,
        )
        seed = _require_int(_field(obj, "seed", path), f"{path}.seed")
        endpoints.append(
            Endpoint(
                endpoint_id=endpoint_id,
                host=host,
                port=port,
                mtu=mtu,
                encapsulation_overhead=overhead,
                replay_window=replay_window,
                seed=seed,
            )
        )

    endpoint_ids = {item.endpoint_id for item in endpoints}
    route_values = _require_list(_field(root, "routes", "$"), "$.routes")
    if not route_values:
        fail(E_SPEC, "$.routes", "at least one route is required")

    routes: list[Route] = []
    seen_tunnel_ids: dict[str, int] = {}
    for index, item in enumerate(route_values):
        path = f"$.routes[{index}]"
        obj = _require_object(item, path)
        prefix = _require_text(_field(obj, "prefix", path), f"{path}.prefix")
        try:
            network = ipaddress.ip_network(prefix, strict=False)
        except ValueError as exc:
            fail(E_SPEC, f"{path}.prefix", "route prefix is invalid", error=str(exc), actual=prefix)

        next_hop = _require_text(_field(obj, "next_hop", path), f"{path}.next_hop")
        if next_hop not in endpoint_ids:
            fail(
                reference_error_code,
                f"{path}.next_hop",
                "route next hop does not reference a declared endpoint",
                next_hop=next_hop,
            )

        priority = _require_int(_field(obj, "priority", path), f"{path}.priority")
        tunnel_id = _require_text(_field(obj, "tunnel_id", path), f"{path}.tunnel_id")
        if tunnel_id in seen_tunnel_ids:
            fail(
                E_SPEC,
                f"{path}.tunnel_id",
                "tunnel id must be unique",
                value=tunnel_id,
                first_index=seen_tunnel_ids[tunnel_id],
            )
        seen_tunnel_ids[tunnel_id] = index
        routes.append(
            Route(
                prefix=prefix,
                network=network,
                next_hop=next_hop,
                priority=priority,
                tunnel_id=tunnel_id,
            )
        )

    policy_obj = _require_object(_field(root, "fragment_policy", "$"), "$.fragment_policy")
    fragment_size = _require_int(
        _field(policy_obj, "fragment_size", "$.fragment_policy"),
        "$.fragment_policy.fragment_size",
        positive=True,
    )
    effective_limit = min(item.effective_payload_limit for item in endpoints)
    if fragment_size > effective_limit:
        fail(
            E_SPEC,
            "$.fragment_policy.fragment_size",
            "fragment size must not exceed the smallest endpoint effective payload limit",
            fragment_size=fragment_size,
            effective_payload_limit=effective_limit,
        )
    max_retries = _require_int(
        _field(policy_obj, "max_retries", "$.fragment_policy"),
        "$.fragment_policy.max_retries",
        nonnegative=True,
    )
    reassembly_timeout_ms = _require_int(
        _field(policy_obj, "reassembly_timeout_ms", "$.fragment_policy"),
        "$.fragment_policy.reassembly_timeout_ms",
        nonnegative=True,
    )
    fragment_policy = FragmentPolicy(
        fragment_size=fragment_size,
        max_retries=max_retries,
        reassembly_timeout_ms=reassembly_timeout_ms,
    )

    runtime_obj = _require_object(_field(root, "runtime", "$", {}), "$.runtime")
    global_limit = _require_int(
        _field(runtime_obj, "global_inflight_limit", "$.runtime", 4),
        "$.runtime.global_inflight_limit",
        positive=True,
    )
    per_tunnel_quota = _require_int(
        _field(runtime_obj, "per_tunnel_inflight_quota", "$.runtime", 2),
        "$.runtime.per_tunnel_inflight_quota",
        positive=True,
    )
    if per_tunnel_quota > global_limit:
        fail(
            E_SPEC,
            "$.runtime.per_tunnel_inflight_quota",
            "per-tunnel quota must not exceed the global inflight limit",
            per_tunnel_quota=per_tunnel_quota,
            global_inflight_limit=global_limit,
        )
    ack_timeout_ms = _require_int(
        _field(runtime_obj, "ack_timeout_ms", "$.runtime", 500),
        "$.runtime.ack_timeout_ms",
        positive=True,
    )
    fragment_format = _require_text(
        _field(runtime_obj, "fragment_format", "$.runtime", "f3"),
        "$.runtime.fragment_format",
    )
    if fragment_format not in {"f1", "f2", "f3"}:
        fail(
            E_SPEC,
            "$.runtime.fragment_format",
            "fragment format must be one of f1, f2, f3",
            actual=fragment_format,
        )
    max_message_bytes = _require_int(
        _field(runtime_obj, "max_message_bytes", "$.runtime", 8 * 1024 * 1024),
        "$.runtime.max_message_bytes",
        positive=True,
    )
    receiver_idle_timeout_ms = _require_int(
        _field(runtime_obj, "receiver_idle_timeout_ms", "$.runtime", 10_000),
        "$.runtime.receiver_idle_timeout_ms",
        positive=True,
    )
    runtime = RuntimePolicy(
        global_inflight_limit=global_limit,
        per_tunnel_inflight_quota=per_tunnel_quota,
        ack_timeout_ms=ack_timeout_ms,
        fragment_format=fragment_format,
        max_message_bytes=max_message_bytes,
        receiver_idle_timeout_ms=receiver_idle_timeout_ms,
    )

    sample_obj = _require_object(_field(root, "sample", "$", {}), "$.sample")
    bytes_value = _field(sample_obj, "bytes_per_message", "$.sample", None)
    if bytes_value is not None:
        bytes_value = _require_int(bytes_value, "$.sample.bytes_per_message", positive=True)
        if bytes_value > max_message_bytes:
            fail(
                E_SPEC,
                "$.sample.bytes_per_message",
                "sample size must not exceed runtime.max_message_bytes",
                actual=bytes_value,
                max_message_bytes=max_message_bytes,
            )
    samples = SamplePolicy(bytes_per_message=bytes_value)

    versions_obj = _require_object(_field(root, "versions", "$", {}), "$.versions")
    versions = Versions(
        declaration=_require_int(
            _field(versions_obj, "declaration", "$.versions", 1),
            "$.versions.declaration",
            positive=True,
        ),
        endpoint_set=_require_int(
            _field(versions_obj, "endpoint_set", "$.versions", 1),
            "$.versions.endpoint_set",
            positive=True,
        ),
        route_table=_require_int(
            _field(versions_obj, "route_table", "$.versions", 1),
            "$.versions.route_table",
            positive=True,
        ),
        fragment_policy=_require_int(
            _field(versions_obj, "fragment_policy", "$.versions", 1),
            "$.versions.fragment_policy",
            positive=True,
        ),
    )

    return TunnelSpec(
        source_path=source_path,
        endpoints=tuple(endpoints),
        routes=tuple(routes),
        fragment_policy=fragment_policy,
        runtime=runtime,
        samples=samples,
        versions=versions,
    )


def validate_versions_monotonic(old: TunnelSpec, new: TunnelSpec) -> None:
    """Reject a changed declaration whose relevant version did not advance."""

    if new.versions.declaration < old.versions.declaration:
        fail(
            E_SPEC,
            "$.versions.declaration",
            "declaration version must be monotonic",
            old=old.versions.declaration,
            new=new.versions.declaration,
        )
    if new.versions.declaration == old.versions.declaration and new.declaration_fingerprint() != old.declaration_fingerprint():
        fail(
            E_SPEC,
            "$.versions.declaration",
            "declaration content changed without a declaration version increment",
            old=old.versions.declaration,
            new=new.versions.declaration,
        )

    checks = (
        (old.endpoint_fingerprint(), new.endpoint_fingerprint(), "endpoint_set"),
        (old.route_fingerprint(), new.route_fingerprint(), "route_table"),
        (old.fragment_fingerprint(), new.fragment_fingerprint(), "fragment_policy"),
    )
    for old_hash, new_hash, version_name in checks:
        old_version = getattr(old.versions, version_name)
        new_version = getattr(new.versions, version_name)
        if new_version < old_version:
            fail(
                E_SPEC,
                f"$.versions.{version_name}",
                f"{version_name} version must be monotonic",
                old=old_version,
                new=new_version,
            )
        if old_hash != new_hash and new_version == old_version:
            fail(
                E_SPEC,
                f"$.versions.{version_name}",
                f"{version_name} changed without a version increment",
                old=old_version,
                new=new_version,
            )
