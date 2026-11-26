"""Local JSON declaration loading and all-or-nothing strong validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import Any

from .errors import ShardMeshError, spec_error
from .fixedpoint import validate_scale
from .graph import topological_order

SUPPORTED_EVENT_TYPES = {
    "arrive",
    "set_budget",
    "set_node_quota",
    "set_node_lease_limit",
    "set_scale",
    "set_max_renewals",
    "set_lease_ttl",
    "bump_capacity_version",
}


@dataclass(frozen=True)
class NodeSpec:
    id: str
    host: str
    port: int
    concurrency: int
    lease_limit: int
    seed: int
    capacity_version: int


@dataclass(frozen=True)
class ShardSpec:
    id: str
    node_id: str
    deps: tuple[str, ...]
    rows: int
    units: int
    priority: int


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass(frozen=True)
class Spec:
    global_budget: int
    scale: int
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    events: tuple[Event, ...]
    seed: int = 365
    host: str = "127.0.0.1"
    max_renewals: int = 1
    lease_ttl_ms: int = 2_000
    request_timeout_ms: int = 3_000
    format_version: int = 1
    source_path: str | None = None

    @property
    def node_map(self) -> dict[str, NodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def shard_map(self) -> dict[str, ShardSpec]:
        return {shard.id: shard for shard in self.shards}

    @property
    def deps_map(self) -> dict[str, tuple[str, ...]]:
        return {shard.id: shard.deps for shard in self.shards}

    @property
    def topology(self) -> list[str]:
        return topological_order(self.deps_map)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        data["nodes"] = [asdict(node) for node in self.nodes]
        data["shards"] = [
            {**asdict(shard), "deps": list(shard.deps)} for shard in self.shards
        ]
        data["events"] = [
            {"seq": event.seq, "type": event.type, **event.params} for event in self.events
        ]
        return data


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, index: int | None, field_name: str, message: str, category: str) -> None:
    if not condition:
        raise spec_error(index, field_name, message, category=category)


def _required_key(raw: dict[str, Any], key: str, index: int | None, category: str) -> Any:
    if key not in raw:
        raise spec_error(index, key, f"missing required field {key!r}", category=category)
    return raw[key]


def _parse_node(raw: Any, index: int, default_host: str = "127.0.0.1") -> NodeSpec:
    category = "node"
    _require(isinstance(raw, dict), index, "node", "node entry must be an object", category)
    node_id = _required_key(raw, "id", index, category)
    _require(isinstance(node_id, str) and node_id, index, "id", "node id must be a non-empty string", category)
    host = raw.get("host", default_host)
    _require(host == "127.0.0.1", index, "host", "host must be the literal 127.0.0.1", category)
    port = _required_key(raw, "port", index, category)
    _require(_is_int(port), index, "port", "port must be an integer", category)
    _require(1024 <= port <= 65535, index, "port", "port must be an unprivileged high port in [1024,65535]", category)
    concurrency = _required_key(raw, "concurrency", index, category)
    _require(_is_int(concurrency) and concurrency >= 0, index, "concurrency", "concurrency must be a non-negative integer", category)
    lease_limit = _required_key(raw, "lease_limit", index, category)
    _require(_is_int(lease_limit) and lease_limit >= 0, index, "lease_limit", "lease_limit must be a non-negative integer", category)
    seed = _required_key(raw, "seed", index, category)
    _require(_is_int(seed) and seed >= 0, index, "seed", "seed must be a non-negative integer", category)
    capacity_version = _required_key(raw, "capacity_version", index, category)
    _require(_is_int(capacity_version) and capacity_version >= 1, index, "capacity_version", "capacity_version must be a positive integer", category)
    return NodeSpec(node_id, host, port, concurrency, lease_limit, seed, capacity_version)


def _parse_shard(raw: Any, index: int, node_ids: set[str]) -> ShardSpec:
    category = "shard"
    _require(isinstance(raw, dict), index, "shard", "shard entry must be an object", category)
    shard_id = _required_key(raw, "id", index, category)
    _require(isinstance(shard_id, str) and shard_id, index, "id", "shard id must be a non-empty string", category)
    if "node_id" in raw:
        node_id = raw["node_id"]
        _require(isinstance(node_id, str) and node_id in node_ids, index, "node_id", "node_id must reference a declared node", category)
    else:
        node_id = ""
    deps = _required_key(raw, "deps", index, category)
    _require(isinstance(deps, list), index, "deps", "deps must be a list of shard ids", category)
    _require(all(isinstance(dep, str) for dep in deps), index, "deps", "every dependency must be a string", category)
    _require(len(deps) == len(set(deps)), index, "deps", "dependencies must not be duplicated", category)
    _require(shard_id not in deps, index, "deps", "a shard cannot depend on itself", category)
    rows = _required_key(raw, "rows", index, category)
    _require(_is_int(rows) and rows >= 1, index, "rows", "rows must be a positive integer", category)
    units = _required_key(raw, "units", index, category)
    _require(_is_int(units) and units >= 0, index, "units", "units must be a non-negative integer", category)
    priority = _required_key(raw, "priority", index, category)
    _require(_is_int(priority), index, "priority", "priority must be an integer", category)
    return ShardSpec(shard_id, node_id, tuple(deps), rows, units, priority)


def _parse_event(raw: Any, index: int, known_shards: set[str], known_nodes: set[str]) -> Event:
    category = "event"
    _require(isinstance(raw, dict), index, "event", "event entry must be an object", category)
    seq = _required_key(raw, "seq", index, category)
    _require(_is_int(seq) and seq >= 1, index, "seq", "event seq must be a positive integer", category)
    event_type = _required_key(raw, "type", index, category)
    _require(isinstance(event_type, str) and event_type in SUPPORTED_EVENT_TYPES, index, "type", f"unsupported event type {event_type!r}", category)
    params = {key: value for key, value in raw.items() if key not in {"seq", "type"}}

    def field_int(name: str, minimum: int | None = 0) -> int:
        value = _required_key(raw, name, index, category)
        _require(_is_int(value), index, name, f"{name} must be an integer", category)
        if minimum is not None:
            _require(value >= minimum, index, name, f"{name} must be >= {minimum}", category)
        return value

    def field_node() -> str:
        node_id = _required_key(raw, "node_id", index, category)
        _require(isinstance(node_id, str) and node_id in known_nodes, index, "node_id", "event node_id must reference a declared node", category)
        return node_id

    if event_type == "arrive":
        shard_id = _required_key(raw, "shard_id", seq, category)
        _require(isinstance(shard_id, str) and shard_id in known_shards, seq, "shard_id", "arrive event must reference a declared shard", category)
    elif event_type == "set_budget":
        field_int("budget", 0)
    elif event_type == "set_node_quota":
        field_node()
        field_int("concurrency", 0)
    elif event_type == "set_node_lease_limit":
        field_node()
        field_int("lease_limit", 0)
    elif event_type == "set_scale":
        validate_scale(_required_key(raw, "scale", seq, category), seq, "scale")
    elif event_type == "set_max_renewals":
        field_int("max_renewals", 0)
    elif event_type == "set_lease_ttl":
        field_int("lease_ttl_ms", 1)
    elif event_type == "bump_capacity_version":
        field_node()
        field_int("version", 1)
    return Event(seq=seq, type=event_type, params=params)


def parse_spec(raw: dict[str, Any], source_path: str | Path | None = None) -> Spec:
    """Validate a raw spec and return an immutable Spec.

    No runtime state is touched before the complete document validates.
    """
    category = "global"
    _require(isinstance(raw, dict), None, "root", "spec root must be a JSON object", category)
    budget = _required_key(raw, "global_budget", None, category)
    _require(_is_int(budget) and budget >= 0, None, "global_budget", "global_budget must be a non-negative integer", category)
    scale = validate_scale(_required_key(raw, "scale", None, category), None, "scale")

    seed = raw.get("seed", 365)
    _require(_is_int(seed) and seed >= 0, None, "seed", "seed must be a non-negative integer", category)
    host = raw.get("host", "127.0.0.1")
    _require(host == "127.0.0.1", None, "host", "host must be the literal 127.0.0.1", category)
    max_renewals = raw.get("max_renewals", 1)
    _require(_is_int(max_renewals) and max_renewals >= 0, None, "max_renewals", "max_renewals must be a non-negative integer", category)
    lease_ttl_ms = raw.get("lease_ttl_ms", 2000)
    _require(_is_int(lease_ttl_ms) and lease_ttl_ms >= 1, None, "lease_ttl_ms", "lease_ttl_ms must be a positive integer", category)
    request_timeout_ms = raw.get("request_timeout_ms", 3000)
    _require(_is_int(request_timeout_ms) and request_timeout_ms >= 50, None, "request_timeout_ms", "request_timeout_ms must be an integer >= 50", category)

    raw_nodes = _required_key(raw, "nodes", None, category)
    _require(isinstance(raw_nodes, list) and raw_nodes, None, "nodes", "nodes must be a non-empty list", category)
    nodes: list[NodeSpec] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    default_host = raw.get("host", "127.0.0.1")
    _require(default_host == "127.0.0.1", None, "host", "host must be the literal 127.0.0.1", "global")
    for index, raw_node in enumerate(raw_nodes):
        node = _parse_node(raw_node, index, default_host)
        _require(node.id not in seen_node_ids, index, "id", f"duplicate node id {node.id!r}", "node")
        _require(node.port not in seen_ports, index, "port", f"duplicate loopback port {node.port}", "node")
        seen_node_ids.add(node.id)
        seen_ports.add(node.port)
        nodes.append(node)

    raw_shards = _required_key(raw, "shards", None, category)
    _require(isinstance(raw_shards, list) and raw_shards, None, "shards", "shards must be a non-empty list", category)
    shards: list[ShardSpec] = []
    seen_shard_ids: set[str] = set()
    for index, raw_shard in enumerate(raw_shards):
        shard = _parse_shard(raw_shard, index, seen_node_ids)
        _require(shard.id not in seen_shard_ids, index, "id", f"duplicate shard id {shard.id!r}", "shard")
        for dep in shard.deps:
            _require(dep in seen_shard_ids or any(item.get("id") == dep for item in raw_shards if isinstance(item, dict)), index, "deps", f"dependency {dep!r} does not exist", "shard")
        seen_shard_ids.add(shard.id)
        shards.append(shard)

    deps_map = {shard.id: shard.deps for shard in shards}
    try:
        topological_order(deps_map)
    except Exception as exc:
        if isinstance(exc, Exception) and getattr(exc, "code", None) == "E_SPEC":
            cyclic = exc.details.get("cycle_candidates", [])
            raise spec_error(
                next((i for i, s in enumerate(shards) if s.id in cyclic), None),
                "deps",
                str(exc.message),
                category="dependency_graph",
                details=exc.details,
            ) from exc
        raise

    assigned_shards = []
    for index, shard in enumerate(shards):
        if shard.node_id:
            assigned_shards.append(shard)
        else:
            assigned_shards.append(replace(shard, node_id=nodes[index % len(nodes)].id))
    shards = tuple(assigned_shards)
    deps_map = {shard.id: shard.deps for shard in shards}

    raw_events = _required_key(raw, "events", None, category)
    _require(isinstance(raw_events, list) and raw_events, None, "events", "events must be a non-empty list", category)
    events: list[Event] = []
    known_shards = set(seen_shard_ids)
    known_nodes = set(seen_node_ids)
    last_seq = 0
    arrived: set[str] = set()
    capacity_versions = {node.id: node.capacity_version for node in nodes}
    for index, raw_event in enumerate(raw_events):
        try:
            event = _parse_event(raw_event, index, known_shards, known_nodes)
        except ShardMeshError as exc:
            if exc.category == "event" and exc.seq is None and isinstance(raw_event, dict):
                raw_seq = raw_event.get("seq")
                if _is_int(raw_seq):
                    exc.seq = raw_seq
            raise
        if event.seq <= last_seq:
            raise spec_error(
                index,
                "seq",
                f"event seq {event.seq} must be strictly increasing after {last_seq}",
                category="event",
                seq=event.seq,
            )
        last_seq = event.seq
        if event.type == "arrive":
            shard_id = event.params["shard_id"]
            _require(shard_id not in arrived, index, "shard_id", f"duplicate arrival for shard {shard_id!r}", "event")
            arrived.add(shard_id)
        elif event.type == "bump_capacity_version":
            node_id = event.params["node_id"]
            version = event.params["version"]
            expected = capacity_versions[node_id] + 1
            _require(version == expected, index, "version", f"capacity version must increment by one; expected {expected}", "event")
            capacity_versions[node_id] = version
        events.append(event)

    missing_arrivals = sorted(known_shards - arrived)
    _require(not missing_arrivals, None, "events", f"missing arrive events for shards {missing_arrivals}", "event")

    return Spec(
        global_budget=budget,
        scale=scale,
        nodes=tuple(nodes),
        shards=tuple(shards),
        events=tuple(events),
        seed=seed,
        host=host,
        max_renewals=max_renewals,
        lease_ttl_ms=lease_ttl_ms,
        request_timeout_ms=request_timeout_ms,
        source_path=str(source_path) if source_path else None,
    )


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise spec_error(None, "json", f"invalid JSON: {exc.msg} at line {exc.lineno}", category="file") from exc
    except OSError as exc:
        raise spec_error(None, "path", f"cannot read spec {path}: {exc}", category="file") from exc
    return parse_spec(raw, path)
