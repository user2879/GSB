from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest_json
from .errors import SpecError

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
MAX_INT64 = 2**63 - 1


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    concurrency_quota: int
    max_leases_per_node: int
    seed: int
    quota_version: int


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    dependencies: tuple[str, ...]
    sample_rows: int
    resource_units: int
    priority: int


@dataclass(frozen=True)
class RuntimeConfig:
    lease_ttl_ms: int = 5_000
    io_timeout_ms: int = 2_000
    ready_timeout_ms: int = 5_000
    batch_size: int = 32
    fixed_point_scale: int = 1_000_000
    wait_poll_ms: int = 5


@dataclass(frozen=True)
class Spec:
    format_version: int
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    global_budget: int
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def node_map(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    @property
    def shard_map(self) -> dict[str, ShardSpec]:
        return {shard.shard_id: shard for shard in self.shards}

    def digest(self) -> str:
        return digest_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "nodes": [asdict(node) for node in self.nodes],
            "shards": [
                {**asdict(shard), "dependencies": list(shard.dependencies)}
                for shard in self.shards
            ],
            "global_budget": self.global_budget,
            "runtime": asdict(self.runtime),
        }


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    data: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass(frozen=True)
class EventStream:
    events: tuple[Event, ...]

    def digest(self) -> str:
        return digest_json([event_to_dict(event) for event in self.events])


def event_to_dict(event: Event) -> dict[str, Any]:
    return {"seq": event.seq, "type": event.type, **event.data}


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SpecError(f"declaration file not found: {path}", field="path") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {path}: {exc.msg}",
            field=f"json:{exc.lineno}:{exc.colno}",
        ) from exc


def load_spec(path: str | Path) -> Spec:
    raw = load_json_file(path)
    return parse_spec(raw)


def load_events(path: str | Path, spec: Spec | None = None) -> EventStream:
    raw = load_json_file(path)
    events = parse_events(raw)
    if spec is not None:
        validate_event_references(events, spec)
    return events


def _err(message: str, *, field: str, object_id: str | None = None) -> SpecError:
    return SpecError(message, field=field, object_id=object_id)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field: str, object_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _err(f"{field} must be a JSON object", field=field, object_id=object_id)
    return value


def _require_id(value: Any, field: str, object_id: str | None = None) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise _err(
            f"{field} must match {ID_RE.pattern}",
            field=field,
            object_id=object_id,
        )
    return value


def _require_int(
    obj: dict[str, Any],
    field: str,
    object_id: str,
    *,
    minimum: int,
    maximum: int = MAX_INT64,
) -> int:
    if field not in obj:
        raise _err(f"missing integer field {field}", field=field, object_id=object_id)
    value = obj[field]
    if not _is_int(value) or value < minimum or value > maximum:
        raise _err(
            f"{field} must be an integer in [{minimum}, {maximum}]",
            field=field,
            object_id=object_id,
        )
    return value


def _reject_unknown(obj: dict[str, Any], allowed: set[str], field_prefix: str, object_id: str) -> None:
    for key in obj:
        if key not in allowed:
            raise _err(
                f"unknown field {key!r}; allowed fields are {sorted(allowed)}",
                field=f"{field_prefix}.{key}",
                object_id=object_id,
            )


def parse_spec(raw: Any) -> Spec:
    root = _require_object(raw, "$")
    for field_name in ("format_version", "nodes", "shards", "global_budget"):
        if field_name not in root:
            raise _err(f"missing required field {field_name}", field=field_name)
    format_version = _require_int(root, "format_version", "$", minimum=1, maximum=1)
    if not isinstance(root["nodes"], list) or not root["nodes"]:
        raise _err("nodes must be a non-empty array", field="nodes")
    if not isinstance(root["shards"], list) or not root["shards"]:
        raise _err("shards must be a non-empty array", field="shards")

    runtime = _parse_runtime(root.get("runtime", {}))
    nodes: list[NodeSpec] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw_node in enumerate(root["nodes"]):
        field_prefix = f"nodes[{index}]"
        node_hint = f"<node[{index}]>"
        node_obj = _require_object(raw_node, field_prefix, node_hint)
        if "node_id" not in node_obj:
            raise _err("missing node_id", field=f"{field_prefix}.node_id", object_id=node_hint)
        node_id = _require_id(node_obj["node_id"], f"{field_prefix}.node_id", node_hint)
        object_id = node_id
        if node_id in seen_node_ids:
            raise _err("duplicate node_id", field=f"{field_prefix}.node_id", object_id=object_id)
        _reject_unknown(
            node_obj,
            {
                "node_id", "host", "port", "concurrency_quota",
                "max_leases_per_node", "seed", "quota_version",
            },
            field_prefix,
            object_id,
        )
        if node_obj.get("host") != "127.0.0.1":
            raise _err(
                "host must be the literal 127.0.0.1",
                field=f"{field_prefix}.host",
                object_id=object_id,
            )
        port = _require_int(node_obj, "port", object_id, minimum=1, maximum=65535)
        if port in seen_ports:
            raise _err("loopback port is duplicated", field=f"{field_prefix}.port", object_id=object_id)
        concurrency = _require_int(node_obj, "concurrency_quota", object_id, minimum=1, maximum=4096)
        max_leases = _require_int(node_obj, "max_leases_per_node", object_id, minimum=1, maximum=8192)
        seed = _require_int(node_obj, "seed", object_id, minimum=0, maximum=2**32 - 1)
        quota_version = _require_int(node_obj, "quota_version", object_id, minimum=1, maximum=MAX_INT64)
        seen_node_ids.add(node_id)
        seen_ports.add(port)
        nodes.append(NodeSpec(node_id, "127.0.0.1", port, concurrency, max_leases, seed, quota_version))

    shards: list[ShardSpec] = []
    seen_shard_ids: set[str] = set()
    for index, raw_shard in enumerate(root["shards"]):
        field_prefix = f"shards[{index}]"
        shard_hint = f"<shard[{index}]>"
        shard_obj = _require_object(raw_shard, field_prefix, shard_hint)
        if "shard_id" not in shard_obj:
            raise _err("missing shard_id", field=f"{field_prefix}.shard_id", object_id=shard_hint)
        shard_id = _require_id(shard_obj["shard_id"], f"{field_prefix}.shard_id", shard_hint)
        object_id = shard_id
        if shard_id in seen_shard_ids:
            raise _err("duplicate shard_id", field=f"{field_prefix}.shard_id", object_id=object_id)
        _reject_unknown(
            shard_obj,
            {"shard_id", "dependencies", "sample_rows", "resource_units", "priority"},
            field_prefix,
            object_id,
        )
        dependencies_raw = shard_obj.get("dependencies")
        if not isinstance(dependencies_raw, list):
            raise _err("dependencies must be an array", field=f"{field_prefix}.dependencies", object_id=object_id)
        dependencies: list[str] = []
        seen_deps: set[str] = set()
        for dep_index, dep in enumerate(dependencies_raw):
            dep_field = f"{field_prefix}.dependencies[{dep_index}]"
            dep_id = _require_id(dep, dep_field, object_id)
            if dep_id in seen_deps:
                raise _err("dependency is duplicated", field=dep_field, object_id=object_id)
            if dep_id == shard_id:
                raise _err("shard cannot depend on itself", field=dep_field, object_id=object_id)
            seen_deps.add(dep_id)
            dependencies.append(dep_id)
        sample_rows = _require_int(shard_obj, "sample_rows", object_id, minimum=1, maximum=1_000_000)
        resource_units = _require_int(shard_obj, "resource_units", object_id, minimum=1, maximum=1_000_000_000)
        priority = _require_int(shard_obj, "priority", object_id, minimum=0, maximum=10_000)
        seen_shard_ids.add(shard_id)
        shards.append(ShardSpec(shard_id, tuple(dependencies), sample_rows, resource_units, priority))

    shard_ids = {shard.shard_id for shard in shards}
    for shard in shards:
        for dep in shard.dependencies:
            if dep not in shard_ids:
                raise _err(
                    f"dependency {dep!r} does not exist",
                    field="shards.dependencies",
                    object_id=shard.shard_id,
                )
    ensure_dag(shards)
    global_budget = _require_int(root, "global_budget", "$", minimum=1, maximum=MAX_INT64)
    _reject_unknown(root, {"format_version", "nodes", "shards", "global_budget", "runtime"}, "$", "$")
    return Spec(format_version, tuple(nodes), tuple(shards), global_budget, runtime)


def _parse_runtime(raw: Any) -> RuntimeConfig:
    if raw in (None, {}):
        return RuntimeConfig()
    obj = _require_object(raw, "runtime", "$")
    _reject_unknown(
        obj,
        {
            "lease_ttl_ms", "io_timeout_ms", "ready_timeout_ms",
            "batch_size", "fixed_point_scale", "wait_poll_ms",
        },
        "runtime",
        "$",
    )
    values = {}
    bounds = {
        "lease_ttl_ms": (20, 3_600_000),
        "io_timeout_ms": (20, 60_000),
        "ready_timeout_ms": (100, 60_000),
        "batch_size": (1, 100_000),
        "fixed_point_scale": (1, 1_000_000_000),
        "wait_poll_ms": (1, 10_000),
    }
    for key, (low, high) in bounds.items():
        if key in obj:
            values[key] = _plain_int(obj[key], f"runtime.{key}", "$", low, high)
    return RuntimeConfig(**values)


def ensure_dag(shards: Iterable[ShardSpec]) -> list[str]:
    shard_list = list(shards)
    shard_map = {shard.shard_id: shard for shard in shard_list}
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(shard_id: str) -> None:
        status = state.get(shard_id, 0)
        if status == 1:
            start = stack.index(shard_id)
            cycle = stack[start:] + [shard_id]
            raise _err(
                "dependency graph contains a cycle: " + " -> ".join(cycle),
                field="shards.dependencies",
                object_id=shard_id,
                details={"cycle": cycle},
            ) if False else SpecError(
                "dependency graph contains a cycle: " + " -> ".join(cycle),
                field="shards.dependencies",
                object_id=shard_id,
                details={"cycle": cycle},
            )
        if status == 2:
            return
        state[shard_id] = 1
        stack.append(shard_id)
        for dep in shard_map[shard_id].dependencies:
            visit(dep)
        stack.pop()
        state[shard_id] = 2

    for shard in sorted(shard_list, key=lambda item: item.shard_id):
        visit(shard.shard_id)
    return stable_topological_order(shard_list)


def stable_topological_order(shards: Iterable[ShardSpec]) -> list[str]:
    shard_list = list(shards)
    shard_map = {shard.shard_id: shard for shard in shard_list}
    indegree = {shard.shard_id: len(shard.dependencies) for shard in shard_list}
    dependents: dict[str, list[str]] = {shard.shard_id: [] for shard in shard_list}
    for shard in shard_list:
        for dep in shard.dependencies:
            dependents[dep].append(shard.shard_id)
    ready = sorted(shard_id for shard_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        newly_ready: list[str] = []
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        ready.extend(newly_ready)
        ready.sort()
    if len(order) != len(shard_map):
        remaining = sorted(shard_id for shard_id, degree in indegree.items() if degree > 0)
        raise SpecError(
            "dependency graph contains an unresolved cycle",
            field="shards.dependencies",
            object_id=remaining[0] if remaining else None,
            details={"remaining": remaining},
        )
    return order


def topo_rank_map(spec: Spec) -> dict[str, int]:
    order = stable_topological_order(spec.shards)
    return {shard_id: index for index, shard_id in enumerate(order)}


def parse_events(raw: Any) -> EventStream:
    root = _require_object(raw, "$")
    _reject_unknown(root, {"events"}, "$", "$")
    if "events" not in root or not isinstance(root["events"], list):
        raise _err("events must be an array", field="events")
    events: list[Event] = []
    for index, raw_event in enumerate(root["events"]):
        prefix = f"events[{index}]"
        hint = f"<event[{index}]>"
        event_obj = _require_object(raw_event, prefix, hint)
        seq = _require_int(event_obj, "seq", hint, minimum=1, maximum=10**9)
        object_id = f"event:{seq}"
        if seq != index + 1:
            raise _err("event seq must be contiguous and match arrival position", field=f"{prefix}.seq", object_id=object_id)
        event_type = event_obj.get("type")
        if not isinstance(event_type, str):
            raise _err("event type must be a string", field=f"{prefix}.type", object_id=object_id)
        allowed = {
            "dispatch", "cancel", "priority",
            "quota_upgrade", "node_offline", "rollback",
        }
        if event_type not in allowed:
            raise _err(f"unknown event type {event_type!r}", field=f"{prefix}.type", object_id=object_id)
        allowed_fields = {
            "seq", "type", "shard_id", "node_id", "priority",
            "quota_version", "target_version", "force_kill",
            "batch_size", "processing_delay_ms", "reason",
        }
        _reject_unknown(event_obj, allowed_fields, prefix, object_id)
        data = {key: value for key, value in event_obj.items() if key not in {"seq", "type"}}
        _validate_event_shape(event_type, data, prefix, object_id)
        events.append(Event(seq, event_type, data))
    return EventStream(tuple(events))


def _validate_event_shape(event_type: str, data: dict[str, Any], prefix: str, object_id: str) -> None:
    def need(field_name: str) -> Any:
        if field_name not in data:
            raise _err(f"missing {field_name}", field=f"{prefix}.{field_name}", object_id=object_id)
        return data[field_name]

    if event_type in {"dispatch", "cancel", "priority"}:
        _require_id(need("shard_id"), f"{prefix}.shard_id", object_id)
    if event_type == "dispatch":
        _require_id(need("node_id"), f"{prefix}.node_id", object_id)
        if "batch_size" in data:
            _require_int(data, "batch_size", object_id, minimum=1, maximum=100_000) if False else _plain_int(
                data["batch_size"], f"{prefix}.batch_size", object_id, 1, 100_000
            )
        if "processing_delay_ms" in data:
            _plain_int(data["processing_delay_ms"], f"{prefix}.processing_delay_ms", object_id, 0, 30_000)
    elif event_type == "cancel":
        if "reason" in data and not isinstance(data["reason"], str):
            raise _err("reason must be a string", field=f"{prefix}.reason", object_id=object_id)
    elif event_type == "priority":
        _plain_int(need("priority"), f"{prefix}.priority", object_id, 0, 10_000)
    elif event_type == "quota_upgrade":
        _plain_int(need("quota_version"), f"{prefix}.quota_version", object_id, 1, MAX_INT64)
        if "node_id" in data:
            _require_id(data["node_id"], f"{prefix}.node_id", object_id)
    elif event_type == "node_offline":
        _require_id(need("node_id"), f"{prefix}.node_id", object_id)
        if "force_kill" in data and not isinstance(data["force_kill"], bool):
            raise _err("force_kill must be a boolean", field=f"{prefix}.force_kill", object_id=object_id)
    elif event_type == "rollback":
        _plain_int(need("target_version"), f"{prefix}.target_version", object_id, 1, MAX_INT64)
        if "node_id" in data:
            _require_id(data["node_id"], f"{prefix}.node_id", object_id)


def _plain_int(value: Any, field_name: str, object_id: str, low: int, high: int) -> int:
    if not _is_int(value) or value < low or value > high:
        raise _err(f"{field_name} must be an integer in [{low}, {high}]", field=field_name, object_id=object_id)
    return value


def validate_event_references(events: EventStream, spec: Spec) -> None:
    nodes = spec.node_map
    shards = spec.shard_map
    dispatched: set[str] = set()
    offline: set[str] = set()
    known_versions: dict[str, set[int]] = {
        node_id: {node.quota_version} for node_id, node in nodes.items()
    }
    global_versions: set[int] = {node.quota_version for node in nodes.values()}
    for event in events.events:
        oid = f"event:{event.seq}"
        data = event.data
        if event.type in {"dispatch", "cancel", "priority"}:
            shard_id = data["shard_id"]
            if shard_id not in shards:
                raise _err("event references unknown shard", field="events.shard_id", object_id=oid)
        if event.type == "dispatch":
            node_id = data["node_id"]
            if node_id not in nodes:
                raise _err("dispatch references unknown node", field="events.node_id", object_id=oid)
            if shard_id in dispatched:
                raise _err("shard is dispatched more than once", field="events.shard_id", object_id=oid)
            if node_id in offline:
                raise _err("cannot dispatch to an offline node", field="events.node_id", object_id=oid)
            dispatched.add(shard_id)
        elif event.type == "cancel":
            if data["shard_id"] not in dispatched:
                raise _err("cannot cancel a shard before its dispatch event", field="events.shard_id", object_id=oid)
        elif event.type == "priority":
            if data["shard_id"] not in dispatched:
                raise _err("cannot adjust priority before dispatch", field="events.shard_id", object_id=oid)
        elif event.type == "quota_upgrade":
            version = data["quota_version"]
            node_id = data.get("node_id")
            if node_id is not None:
                if node_id not in nodes:
                    raise _err("quota_upgrade references unknown node", field="events.node_id", object_id=oid)
                known_versions[node_id].add(version)
            else:
                global_versions.add(version)
                for versions in known_versions.values():
                    versions.add(version)
        elif event.type == "node_offline":
            node_id = data["node_id"]
            if node_id not in nodes:
                raise _err("node_offline references unknown node", field="events.node_id", object_id=oid)
            if node_id in offline:
                raise _err("node is already offline", field="events.node_id", object_id=oid)
            offline.add(node_id)
        elif event.type == "rollback":
            target = data["target_version"]
            node_id = data.get("node_id")
            if node_id is not None:
                if node_id not in nodes:
                    raise _err("rollback references unknown node", field="events.node_id", object_id=oid)
                if target not in known_versions[node_id]:
                    raise _err("rollback target is not a historical quota version for node", field="events.target_version", object_id=oid)
            else:
                if target not in global_versions or any(target not in versions for versions in known_versions.values()):
                    raise _err("global rollback target is not a historical quota version on every node", field="events.target_version", object_id=oid)


def validate_spec_file(path: str | Path) -> tuple[Spec | None, SpecError | None]:
    try:
        return load_spec(path), None
    except SpecError as exc:
        return None, exc
