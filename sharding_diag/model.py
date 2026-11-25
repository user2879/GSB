"""Typed declarations plus strict JSON loading and validation.

A declaration document has three sections::

    {
      "nodes": [ {"node_id", "host", "port", "concurrency",
                  "lease_limit", "seed", "capacity_version",
                  "fixed_point_scale"?} ... ],
      "shards": [ {"shard_id", "depends_on"?, "rows", "demand",
                   "priority"} ... ],
      "global": {"budget", "freeze_threshold_ms", "revoke_tolerance",
                 "lease_ttl_ms"?, "fixed_point_scale"?}
    }

Validation walks sections in a fixed sorted order and rejects the *first*
illegal object with ``E_SPEC`` carrying its identifier and field name.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Dict, List, Optional, Tuple

from .errors import SpecError

_SCHEMA_VERSION = 1


def _is_int(value) -> bool:
    # bool is a subclass of int but is never an acceptable integer here.
    return isinstance(value, int) and not isinstance(value, bool)


def _reject(object_id: str, fld: str, reason: str):
    raise SpecError(
        f"illegal {fld!r} on {object_id}: {reason}",
        object_id=object_id,
        field=fld,
        detail=reason,
    )


def _require_int(object_id, fld, value, minimum):
    if not _is_int(value):
        _reject(object_id, fld, f"expected integer, got {type(value).__name__}")
    if value < minimum:
        _reject(object_id, fld, f"expected integer >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    concurrency: int
    lease_limit: int
    seed: int
    capacity_version: int
    fixed_point_scale: int


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    depends_on: Tuple[str, ...]
    rows: int
    demand: int
    priority: int


@dataclass(frozen=True)
class GlobalSpec:
    budget: int
    freeze_threshold_ms: int
    revoke_tolerance: int
    lease_ttl_ms: int
    fixed_point_scale: int


@dataclass(frozen=True)
class Declaration:
    nodes: Tuple[NodeSpec, ...]
    shards: Tuple[ShardSpec, ...]
    global_spec: GlobalSpec
    schema_version: int = _SCHEMA_VERSION

    def node_map(self) -> Dict[str, NodeSpec]:
        return {n.node_id: n for n in self.nodes}

    def shard_map(self) -> Dict[str, ShardSpec]:
        return {s.shard_id: s for s in self.shards}

    def node_set_version(self) -> str:
        """Stable fingerprint of the node set (id/host/port only)."""
        parts = [f"{n.node_id}@{n.host}:{n.port}" for n in sorted(self.nodes, key=lambda x: x.node_id)]
        return "ns:" + "|".join(parts)

    def capacity_version_map(self) -> Dict[str, int]:
        return {n.node_id: n.capacity_version for n in self.nodes}


def load_declaration(path: str) -> Declaration:
    if not os.path.exists(path):
        raise SpecError(f"declaration file not found: {path}", object_id=path, field="file")
    with open(path, "r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid JSON: {exc}", object_id=path, field="file")
    if not isinstance(raw, dict):
        raise SpecError("top-level declaration must be an object", object_id="<root>", field="root")
    return parse_declaration(raw)


def parse_declaration(raw: dict) -> Declaration:
    global_default_scale = 1000
    gsection = raw.get("global")
    if not isinstance(gsection, dict):
        _reject("<global>", "global", "section missing or not an object")
    global_spec = _parse_global(gsection)
    global_default_scale = global_spec.fixed_point_scale

    nsection = raw.get("nodes")
    if not isinstance(nsection, list) or not nsection:
        _reject("<nodes>", "nodes", "section must be a non-empty list")
    nodes: List[NodeSpec] = []
    seen_node_ids = set()
    seen_endpoints = set()
    for item in nsection:
        nodes.append(_parse_node(item, seen_node_ids, seen_endpoints, global_default_scale))

    ssection = raw.get("shards")
    if not isinstance(ssection, list) or not ssection:
        _reject("<shards>", "shards", "section must be a non-empty list")
    shards: List[ShardSpec] = []
    seen_shard_ids = set()
    for item in ssection:
        shards.append(_parse_shard(item, seen_shard_ids))

    # Cross-references and DAG check use the parsed identifier sets.
    shard_ids = {s.shard_id for s in shards}
    _validate_dependencies(shards, shard_ids)
    _validate_dag(shards, shard_ids)
    return Declaration(tuple(nodes), tuple(shards), global_spec)


def _parse_global(section: dict) -> GlobalSpec:
    oid = "<global>"
    budget = _require_int(oid, "budget", section.get("budget"), 1)
    freeze = _require_int(oid, "freeze_threshold_ms", section.get("freeze_threshold_ms"), 0)
    tolerance = _require_int(oid, "revoke_tolerance", section.get("revoke_tolerance"), 0)
    ttl = _require_int(oid, "lease_ttl_ms", section.get("lease_ttl_ms", 30_000), 1)
    scale = _require_int(oid, "fixed_point_scale", section.get("fixed_point_scale", 1000), 1)
    return GlobalSpec(budget, freeze, tolerance, ttl, scale)


def _parse_node(item, seen_ids, seen_endpoints, default_scale) -> NodeSpec:
    if not isinstance(item, dict):
        _reject("<node>", "node", "each node must be an object")
    node_id = item.get("node_id")
    if not isinstance(node_id, str) or not node_id:
        _reject("<node>", "node_id", "must be a non-empty string")
    if node_id in seen_ids:
        _reject(node_id, "node_id", "duplicate node identifier")
    seen_ids.add(node_id)

    host = item.get("host")
    if host != "127.0.0.1":
        _reject(node_id, "host", f"only the literal 127.0.0.1 loopback is allowed, got {host!r}")
    port = _require_int(node_id, "port", item.get("port"), 1)
    if not 1024 <= port <= 65535:
        _reject(node_id, "port", f"must be a high port in [1024, 65535], got {port}")
    endpoint = (host, port)
    if endpoint in seen_endpoints:
        _reject(node_id, "port", f"duplicate loopback endpoint {host}:{port}")
    seen_endpoints.add(endpoint)

    concurrency = _require_int(node_id, "concurrency", item.get("concurrency"), 1)
    lease_limit = _require_int(node_id, "lease_limit", item.get("lease_limit"), 1)
    seed = _require_int(node_id, "seed", item.get("seed"), 0)
    capacity_version = _require_int(node_id, "capacity_version", item.get("capacity_version"), 1)
    scale = _require_int(
        node_id,
        "fixed_point_scale",
        item.get("fixed_point_scale", default_scale),
        1,
    )
    return NodeSpec(node_id, host, port, concurrency, lease_limit, seed, capacity_version, scale)


def _parse_shard(item, seen_ids) -> ShardSpec:
    if not isinstance(item, dict):
        _reject("<shard>", "shard", "each shard must be an object")
    shard_id = item.get("shard_id")
    if not isinstance(shard_id, str) or not shard_id:
        _reject("<shard>", "shard_id", "must be a non-empty string")
    if shard_id in seen_ids:
        _reject(shard_id, "shard_id", "duplicate shard identifier")
    seen_ids.add(shard_id)

    deps_raw = item.get("depends_on", [])
    if not isinstance(deps_raw, list) or any(not isinstance(d, str) or not d for d in deps_raw):
        _reject(shard_id, "depends_on", "must be a list of non-empty strings")
    rows = _require_int(shard_id, "rows", item.get("rows"), 1)
    demand = _require_int(shard_id, "demand", item.get("demand"), 1)
    priority = _require_int(shard_id, "priority", item.get("priority"), 0)
    return ShardSpec(shard_id, tuple(deps_raw), rows, demand, priority)


def _validate_dependencies(shards: List[ShardSpec], shard_ids):
    for shard in shards:
        for dep in shard.depends_on:
            if dep == shard.shard_id:
                _reject(shard.shard_id, "depends_on", f"self dependency {dep!r}")
            if dep not in shard_ids:
                _reject(shard.shard_id, "depends_on", f"unknown upstream reference {dep!r}")
        if len(set(shard.depends_on)) != len(shard.depends_on):
            _reject(shard.shard_id, "depends_on", "duplicate upstream reference")


def _validate_dag(shards: List[ShardSpec], shard_ids):
    # Iterate deterministically so the first cyclic node reported is stable.
    graph = {s.shard_id: list(s.depends_on) for s in shards}
    state: Dict[str, int] = {}  # 0=unvisited, 1=on stack, 2=done

    def visit(node_id, stack):
        state[node_id] = 1
        stack.append(node_id)
        for dep in graph[node_id]:
            mark = state.get(dep, 0)
            if mark == 0:
                visit(dep, stack)
            elif mark == 1:
                cycle = stack[stack.index(dep):] + [dep]
                _reject(node_id, "depends_on", "dependency cycle: " + " -> ".join(cycle))
        stack.pop()
        state[node_id] = 2

    for node_id in sorted(graph):
        if state.get(node_id, 0) == 0:
            visit(node_id, [])


def topological_order(shards) -> List[str]:
    """Kahn sort with identifier tie-breaking; independent of dict order."""
    specs = sorted(shards, key=lambda s: s.shard_id)
    deps = {s.shard_id: set(s.depends_on) for s in specs}
    dependents: Dict[str, List[str]] = {s.shard_id: [] for s in specs}
    for s in specs:
        for dep in s.depends_on:
            dependents[dep].append(s.shard_id)
    ready = sorted(sid for sid, d in deps.items() if not d)
    order: List[str] = []
    while ready:
        sid = ready.pop(0)
        order.append(sid)
        for child in sorted(dependents[sid]):
            deps[child].discard(sid)
            if not deps[child]:
                ready.append(child)
        ready.sort()
    if len(order) != len(deps):
        raise SpecError("cannot topologically sort a cyclic graph", field="depends_on")
    return order
