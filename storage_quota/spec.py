from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import AppError, SpecError


FORMAT_VERSION = 1
LOOPBACK_HOST = "127.0.0.1"
EVENT_TYPES = {
    "dispatch",
    "renew",
    "expire",
    "release",
    "capacity_change",
    "quota_change",
    "barrier",
    "crash",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_json(path: str | Path, role: str) -> Any:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise AppError("E_IO", f"cannot read {role}: {exc}", path=str(p)) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(f"$ (line {exc.lineno}, column {exc.colno})", "document",
                        f"invalid JSON in {role}: {exc.msg}") from exc


def _object(value: Any, path: str, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(path, field_name, "expected object")
    return value


def _keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    for key in obj:
        if key not in allowed:
            raise SpecError(f"{path}.{key}", key, "unknown field")


def _required(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise SpecError(f"{path}.{key}", key, "missing required field")
    return obj[key]


def _string(value: Any, path: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(path, field_name, "expected non-empty string")
    return value


def _bool(value: Any, path: str, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SpecError(path, field_name, "expected boolean")
    return value


def _int(value: Any, path: str, field_name: str, *, minimum: int | None = None,
         strict_minimum: int | None = None) -> int:
    if not _is_int(value):
        raise SpecError(path, field_name, "expected integer")
    if minimum is not None and value < minimum:
        raise SpecError(path, field_name, f"expected integer >= {minimum}")
    if strict_minimum is not None and value <= strict_minimum:
        raise SpecError(path, field_name, f"expected integer > {strict_minimum}")
    return value


def _list(value: Any, path: str, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(path, field_name, "expected array")
    return value


@dataclass(frozen=True)
class NodeSpec:
    id: str
    host: str
    port: int
    fallback_port: int | None
    concurrency: int
    lease_limit: int
    seed: int
    capacity_version: int

    def effective_port(self, use_fallback: bool) -> int:
        if use_fallback:
            if self.fallback_port is None:
                raise SpecError(f"nodes[{self.id}].fallback_port", "fallback_port",
                                "fallback port was requested but is not declared")
            return self.fallback_port
        return self.port


@dataclass(frozen=True)
class ShardSpec:
    id: str
    node_id: str
    rows: int
    resource_units: int
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class RestartSpec:
    node_id: str
    at_seq: int


@dataclass(frozen=True)
class EventSpec:
    seq: int
    type: str
    payload: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


@dataclass(frozen=True)
class Spec:
    format_version: int
    global_resource_budget: int
    lease_duration_ms: int
    socket_timeout_ms: int
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    events: tuple[EventSpec, ...]
    restarts: tuple[RestartSpec, ...] = field(default_factory=tuple)

    def node(self, node_id: str) -> NodeSpec:
        for item in self.nodes:
            if item.id == node_id:
                return item
        raise KeyError(node_id)

    def shard(self, shard_id: str) -> ShardSpec:
        for item in self.shards:
            if item.id == shard_id:
                return item
        raise KeyError(shard_id)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(shard.id for shard in self.shards)

    def event_json(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "events": [{"seq": event.seq, "type": event.type, **event.payload}
                       for event in self.events],
        }

    def declaration_json(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "global_resource_budget": self.global_resource_budget,
            "lease_duration_ms": self.lease_duration_ms,
            "socket_timeout_ms": self.socket_timeout_ms,
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "port": node.port,
                    **({"fallback_port": node.fallback_port}
                       if node.fallback_port is not None else {}),
                    "concurrency": node.concurrency,
                    "lease_limit": node.lease_limit,
                    "seed": node.seed,
                    "capacity_version": node.capacity_version,
                }
                for node in self.nodes
            ],
            "shards": [
                {
                    "id": shard.id,
                    "node_id": shard.node_id,
                    "rows": shard.rows,
                    "resource_units": shard.resource_units,
                    **({"depends_on": list(shard.depends_on)} if shard.depends_on else {}),
                }
                for shard in self.shards
            ],
            "restarts": [
                {"node_id": restart.node_id, "at_seq": restart.at_seq}
                for restart in self.restarts
            ],
        }


def _parse_node(raw: Any, index: int) -> NodeSpec:
    path = f"nodes[{index}]"
    obj = _object(raw, path, "node")
    _keys(obj, {"id", "host", "port", "fallback_port", "concurrency", "lease_limit",
                "seed", "capacity_version"}, path)
    node_id = _string(_required(obj, "id", path), f"{path}.id", "id")
    host = _string(_required(obj, "host", path), f"{path}.host", "host")
    if host != LOOPBACK_HOST:
        raise SpecError(f"{path}.host", "host", "host must be the literal 127.0.0.1")
    port = _int(_required(obj, "port", path), f"{path}.port", "port", minimum=1)
    if port > 65535:
        raise SpecError(f"{path}.port", "port", "port must be <= 65535")
    fallback_port: int | None = None
    if "fallback_port" in obj:
        fallback_port = _int(obj["fallback_port"], f"{path}.fallback_port",
                             "fallback_port", minimum=1)
        if fallback_port > 65535:
            raise SpecError(f"{path}.fallback_port", "fallback_port", "port must be <= 65535")
        if fallback_port == port:
            raise SpecError(f"{path}.fallback_port", "fallback_port",
                            "fallback_port must differ from port")
    concurrency = _int(_required(obj, "concurrency", path), f"{path}.concurrency",
                       "concurrency", minimum=0)
    lease_limit = _int(_required(obj, "lease_limit", path), f"{path}.lease_limit",
                       "lease_limit", minimum=0)
    seed = _int(_required(obj, "seed", path), f"{path}.seed", "seed")
    capacity_version = _int(_required(obj, "capacity_version", path),
                            f"{path}.capacity_version", "capacity_version", minimum=0)
    return NodeSpec(node_id, host, port, fallback_port, concurrency, lease_limit, seed,
                    capacity_version)


def _parse_shard(raw: Any, index: int) -> ShardSpec:
    path = f"shards[{index}]"
    obj = _object(raw, path, "shard")
    _keys(obj, {"id", "node_id", "rows", "resource_units", "depends_on"}, path)
    shard_id = _string(_required(obj, "id", path), f"{path}.id", "id")
    node_id = _string(_required(obj, "node_id", path), f"{path}.node_id", "node_id")
    rows = _int(_required(obj, "rows", path), f"{path}.rows", "rows", strict_minimum=0)
    units = _int(_required(obj, "resource_units", path), f"{path}.resource_units",
                 "resource_units", strict_minimum=0)
    depends_on: list[str] = []
    if "depends_on" in obj:
        deps = _list(obj["depends_on"], f"{path}.depends_on", "depends_on")
        for dep_index, dep in enumerate(deps):
            dep_path = f"{path}.depends_on[{dep_index}]"
            dep_id = _string(dep, dep_path, "depends_on")
            if dep_id in depends_on:
                raise SpecError(dep_path, "depends_on", f"duplicate shard id {dep_id}")
            depends_on.append(dep_id)
    return ShardSpec(shard_id, node_id, rows, units, tuple(depends_on))


def _parse_event(raw: Any, index: int, *, previous_seq: int,
                 capacity_versions: dict[str, int], node_ids: set[str],
                 shard_ids: set[str]) -> EventSpec:
    path = f"events[{index}]"
    obj = _object(raw, path, "event")
    seq = _int(_required(obj, "seq", path), f"{path}.seq", "seq", strict_minimum=0)
    if index == 0 and seq != 1:
        raise SpecError(f"{path}.seq", "seq", "first event seq must be 1")
    if seq <= previous_seq:
        raise SpecError(f"{path}.seq", "seq", "event seq must be strictly increasing")
    event_type = _string(_required(obj, "type", path), f"{path}.type", "type")
    if event_type not in EVENT_TYPES:
        raise SpecError(f"{path}.type", "type", f"unknown event type {event_type}")
    allowed = {"seq", "type"}
    payload: dict[str, Any] = {}
    if event_type == "dispatch":
        allowed |= {"shard_id"}
        shard_id = _string(_required(obj, "shard_id", path), f"{path}.shard_id", "shard_id")
        if shard_id not in shard_ids:
            raise SpecError(f"{path}.shard_id", "shard_id", f"unknown shard {shard_id}")
        payload = {"shard_id": shard_id}
    elif event_type in {"renew", "expire", "release"}:
        allowed |= {"shard_id"}
        if event_type == "renew":
            allowed.add("expected_lease_version")
        if event_type == "expire":
            allowed.add("wait_until_expired")
        if event_type == "release":
            allowed.add("reason")
        shard_id = _string(_required(obj, "shard_id", path), f"{path}.shard_id", "shard_id")
        if shard_id not in shard_ids:
            raise SpecError(f"{path}.shard_id", "shard_id", f"unknown shard {shard_id}")
        payload = {"shard_id": shard_id}
        if event_type == "renew":
            version = _int(_required(obj, "expected_lease_version", path),
                           f"{path}.expected_lease_version", "expected_lease_version",
                           strict_minimum=0)
            payload["expected_lease_version"] = version
        elif event_type == "expire":
            wait = False
            if "wait_until_expired" in obj:
                wait = _bool(obj["wait_until_expired"], f"{path}.wait_until_expired",
                             "wait_until_expired")
            payload["wait_until_expired"] = wait
        elif event_type == "release":
            reason = "released"
            if "reason" in obj:
                reason = _string(obj["reason"], f"{path}.reason", "reason")
            payload["reason"] = reason
    elif event_type == "capacity_change":
        allowed |= {"node_id", "new_capacity_version"}
        node_id = _string(_required(obj, "node_id", path), f"{path}.node_id", "node_id")
        if node_id not in node_ids:
            raise SpecError(f"{path}.node_id", "node_id", f"unknown node {node_id}")
        version = _int(_required(obj, "new_capacity_version", path),
                       f"{path}.new_capacity_version", "new_capacity_version", minimum=0)
        if version <= capacity_versions[node_id]:
            raise SpecError(f"{path}.new_capacity_version", "new_capacity_version",
                            "capacity version must increase monotonically")
        capacity_versions[node_id] = version
        payload = {"node_id": node_id, "new_capacity_version": version}
    elif event_type == "quota_change":
        allowed |= {"node_id", "concurrency", "lease_limit"}
        node_id = _string(_required(obj, "node_id", path), f"{path}.node_id", "node_id")
        if node_id not in node_ids:
            raise SpecError(f"{path}.node_id", "node_id", f"unknown node {node_id}")
        payload = {"node_id": node_id}
        if "concurrency" not in obj and "lease_limit" not in obj:
            raise SpecError(f"{path}.concurrency", "concurrency",
                            "at least one quota field is required")
        if "concurrency" in obj:
            payload["concurrency"] = _int(obj["concurrency"], f"{path}.concurrency",
                                           "concurrency", minimum=0)
        if "lease_limit" in obj:
            payload["lease_limit"] = _int(obj["lease_limit"], f"{path}.lease_limit",
                                           "lease_limit", minimum=0)
    elif event_type == "crash":
        allowed |= {"node_id"}
        node_id = _string(_required(obj, "node_id", path), f"{path}.node_id", "node_id")
        if node_id not in node_ids:
            raise SpecError(f"{path}.node_id", "node_id", f"unknown node {node_id}")
        payload = {"node_id": node_id}
    elif event_type == "barrier":
        payload = {}
    _keys(obj, allowed, path)
    return EventSpec(seq, event_type, payload)


def _parse_restart(raw: Any, index: int, *, node_ids: set[str],
                   event_seqs: set[int]) -> RestartSpec:
    path = f"restarts[{index}]"
    obj = _object(raw, path, "restart")
    _keys(obj, {"node_id", "at_seq"}, path)
    node_id = _string(_required(obj, "node_id", path), f"{path}.node_id", "node_id")
    if node_id not in node_ids:
        raise SpecError(f"{path}.node_id", "node_id", f"unknown node {node_id}")
    at_seq = _int(_required(obj, "at_seq", path), f"{path}.at_seq", "at_seq", strict_minimum=0)
    if at_seq not in event_seqs:
        raise SpecError(f"{path}.at_seq", "at_seq", f"no event with seq {at_seq}")
    return RestartSpec(node_id, at_seq)


def _detect_dependency_cycle(shards: tuple[ShardSpec, ...]) -> None:
    by_id = {shard.id: shard for shard in shards}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(shard_id: str, trail: list[str]) -> None:
        if shard_id in visiting:
            cycle = trail[trail.index(shard_id):] + [shard_id]
            first = cycle[0]
            index = next(i for i, shard in enumerate(shards) if shard.id == first)
            raise SpecError(f"shards[{index}].depends_on", "depends_on",
                            "dependency cycle: " + " -> ".join(cycle))
        if shard_id in visited:
            return
        visiting.add(shard_id)
        for dep in by_id[shard_id].depends_on:
            if dep not in by_id:
                index = next(i for i, shard in enumerate(shards) if shard.id == shard_id)
                raise SpecError(f"shards[{index}].depends_on", "depends_on",
                                f"unknown shard dependency {dep}")
            if dep == shard_id:
                index = next(i for i, shard in enumerate(shards) if shard.id == shard_id)
                raise SpecError(f"shards[{index}].depends_on", "depends_on",
                                "shard cannot depend on itself")
            visit(dep, trail + [shard_id])
        visiting.remove(shard_id)
        visited.add(shard_id)

    for shard in shards:
        visit(shard.id, [])


def validate_documents(spec_document: Any, events_document: Any) -> Spec:
    path = "$"
    obj = _object(spec_document, path, "document")
    _keys(obj, {"format_version", "global_resource_budget", "lease_duration_ms",
                "socket_timeout_ms", "nodes", "shards", "restarts"}, path)
    format_version = _int(_required(obj, "format_version", path), "$.format_version",
                          "format_version", minimum=1)
    if format_version != FORMAT_VERSION:
        raise SpecError("$.format_version", "format_version",
                        f"unsupported format_version {format_version}")
    global_budget = _int(_required(obj, "global_resource_budget", path),
                         "$.global_resource_budget", "global_resource_budget", minimum=0)
    lease_duration = _int(_required(obj, "lease_duration_ms", path),
                          "$.lease_duration_ms", "lease_duration_ms", strict_minimum=0)
    socket_timeout = _int(_required(obj, "socket_timeout_ms", path),
                          "$.socket_timeout_ms", "socket_timeout_ms", strict_minimum=0)

    raw_nodes = _list(_required(obj, "nodes", path), "$.nodes", "nodes")
    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    used_ports: dict[int, str] = {}
    for index, raw in enumerate(raw_nodes):
        node = _parse_node(raw, index)
        if node.id in node_ids:
            raise SpecError(f"nodes[{index}].id", "id", f"duplicate node id {node.id}")
        node_ids.add(node.id)
        for field_name, port in (("port", node.port), ("fallback_port", node.fallback_port)):
            if port is None:
                continue
            if port in used_ports:
                raise SpecError(f"nodes[{index}].{field_name}", field_name,
                                f"port {port} already used by {used_ports[port]}")
            used_ports[port] = node.id
        nodes.append(node)

    raw_shards = _list(_required(obj, "shards", path), "$.shards", "shards")
    shards: list[ShardSpec] = []
    shard_ids: set[str] = set()
    for index, raw in enumerate(raw_shards):
        shard = _parse_shard(raw, index)
        if shard.id in shard_ids:
            raise SpecError(f"shards[{index}].id", "id", f"duplicate shard id {shard.id}")
        shard_ids.add(shard.id)
        if shard.node_id not in node_ids:
            raise SpecError(f"shards[{index}].node_id", "node_id",
                            f"unknown node {shard.node_id}")
        shards.append(shard)
    shards_tuple = tuple(shards)
    _detect_dependency_cycle(shards_tuple)

    restarts_raw = obj.get("restarts", [])
    restarts_list = _list(restarts_raw, "$.restarts", "restarts")

    events_obj = _object(events_document, "$events", "document")
    _keys(events_obj, {"format_version", "events"}, "$events")
    events_format = _int(_required(events_obj, "format_version", "$events"),
                         "$events.format_version", "format_version", minimum=1)
    if events_format != FORMAT_VERSION:
        raise SpecError("$events.format_version", "format_version",
                        f"unsupported format_version {events_format}")
    raw_events = _list(_required(events_obj, "events", "$events"), "$events.events", "events")
    events: list[EventSpec] = []
    previous_seq = 0
    capacity_versions = {node.id: node.capacity_version for node in nodes}
    for index, raw in enumerate(raw_events):
        event = _parse_event(raw, index, previous_seq=previous_seq,
                             capacity_versions=capacity_versions,
                             node_ids=node_ids, shard_ids=shard_ids)
        previous_seq = event.seq
        events.append(event)
    event_seqs = {event.seq for event in events}
    restarts = tuple(_parse_restart(raw, index, node_ids=node_ids, event_seqs=event_seqs)
                     for index, raw in enumerate(restarts_list))
    return Spec(format_version, global_budget, lease_duration, socket_timeout,
                tuple(nodes), shards_tuple, tuple(events), restarts)


def load_spec(spec_path: str | Path, events_path: str | Path) -> Spec:
    return validate_documents(_load_json(spec_path, "spec"),
                              _load_json(events_path, "event stream"))
