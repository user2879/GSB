"""Local JSON specification loading and strict validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SpecError

SPEC_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CapacityLevel:
    version: int
    concurrency: int
    lease_limit: int


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    seed: int
    initial_capacity_version: int
    capacities: tuple[CapacityLevel, ...]

    def level(self, version: int) -> CapacityLevel:
        for level in self.capacities:
            if level.version == version:
                return level
        raise SpecError(
            f"node {self.node_id} has no capacity level {version}",
            path=f"nodes.{self.node_id}.capacities",
        )


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    depends_on: tuple[str, ...]
    sample_rows: int
    resource_units: int
    priority: int


@dataclass(frozen=True)
class EventSpec:
    seq: int
    kind: str
    shard_id: str | None
    capacity_version: int | None
    lease_id: str | None
    ms: int | None = None


@dataclass(frozen=True)
class Spec:
    format_version: int
    global_budget: dict[int, int]
    lease_ttl_ms: int
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    events: tuple[EventSpec, ...]
    raw: dict
    node_ids: tuple[str, ...] = field(default_factory=tuple)
    shard_ids: tuple[str, ...] = field(default_factory=tuple)

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise SpecError(f"unknown node {node_id}", path="nodes")

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise SpecError(f"unknown shard {shard_id}", path="shards")

    def node_set_version(self) -> str:
        parts = [
            f"{n.node_id}|{n.host}|{n.port}|{n.seed}|{n.initial_capacity_version}"
            for n in self.nodes
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]

    def content_digest(self) -> str:
        canonical = json.dumps(
            self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def structural_digest(self) -> str:
        """Digest of durable topology/quota inputs, excluding transient events."""
        structural = {
            "format_version": self.format_version,
            "lease_ttl_ms": self.lease_ttl_ms,
            "global_budget": self.global_budget,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "host": node.host,
                    "port": node.port,
                    "seed": node.seed,
                    "initial_capacity_version": node.initial_capacity_version,
                    "capacities": [level.__dict__ for level in node.capacities],
                }
                for node in self.nodes
            ],
            "shards": [shard.__dict__ for shard in self.shards],
        }
        canonical = json.dumps(
            structural, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def capacity_versions(self) -> tuple[int, ...]:
        return tuple(sorted(self.global_budget.keys()))


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"spec file not found: {path}", path="<file>") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}", path=f"<file>:{exc.line}:{exc.col}"
        ) from exc
    return parse_spec(raw)


def _is_nonneg_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_pos_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_ident(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def parse_spec(raw: dict) -> Spec:
    if not isinstance(raw, dict):
        raise SpecError("top-level declaration must be a JSON object", path="$")

    format_version = raw.get("format_version", 1)
    if not _is_nonneg_int(format_version):
        raise SpecError("format_version must be a non-negative integer", path="format_version")

    ttl = raw.get("lease_ttl_ms", 300)
    if not _is_pos_int(ttl):
        raise SpecError("lease_ttl_ms must be a positive integer", path="lease_ttl_ms")

    global_budget = _parse_budget(raw)
    nodes = _parse_nodes(raw, global_budget)
    shards = _parse_shards(raw)
    _validate_dependencies(shards)
    events = _parse_events(raw, nodes, shards, global_budget)

    return Spec(
        format_version=int(format_version),
        global_budget=global_budget,
        lease_ttl_ms=int(ttl),
        nodes=nodes,
        shards=shards,
        events=events,
        raw=raw,
        node_ids=tuple(n.node_id for n in nodes),
        shard_ids=tuple(s.shard_id for s in shards),
    )


def _parse_budget(raw: dict) -> dict[int, int]:
    if "global_budget" not in raw or not isinstance(raw["global_budget"], dict):
        raise SpecError("global_budget must be an object", path="global_budget")
    budget: dict[int, int] = {}
    for key, value in raw["global_budget"].items():
        if not str(key).lstrip("-").isdigit():
            raise SpecError("capacity version key must be an integer", path=f"global_budget.{key}")
        version = int(key)
        if version < 0:
            raise SpecError("capacity version must be non-negative", path=f"global_budget.{key}")
        if not _is_nonneg_int(value):
            raise SpecError(
                "budget must be a non-negative integer", path=f"global_budget.{key}"
            )
        budget[version] = int(value)
    if not budget:
        raise SpecError("global_budget must declare at least one version", path="global_budget")
    return dict(sorted(budget.items()))


def _parse_nodes(raw: dict, global_budget: dict[int, int]) -> tuple[NodeSpec, ...]:
    if "nodes" not in raw or not isinstance(raw["nodes"], list):
        raise SpecError("nodes must be a list", path="nodes")
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    nodes: list[NodeSpec] = []
    for index, item in enumerate(raw["nodes"]):
        base = f"nodes[{index}]"
        if not isinstance(item, dict):
            raise SpecError("node must be an object", path=base)
        node_id = item.get("node_id")
        if not _is_ident(node_id):
            raise SpecError("node_id must be a non-empty string", path=f"{base}.node_id")
        if node_id in seen_ids:
            raise SpecError(f"duplicate node_id {node_id}", path=f"{base}.node_id")
        seen_ids.add(node_id)

        host = item.get("host")
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", path=f"{base}.host")
        port = item.get("port")
        if not _is_nonneg_int(port) or not (1 <= int(port) <= 65535):
            raise SpecError("port must be an integer in 1..65535", path=f"{base}.port")
        if int(port) in seen_ports:
            raise SpecError(f"duplicate port {port}", path=f"{base}.port")
        seen_ports.add(int(port))

        seed = item.get("seed")
        if not _is_nonneg_int(seed):
            raise SpecError("seed must be a non-negative integer", path=f"{base}.seed")

        initial = item.get("initial_capacity_version")
        if not _is_nonneg_int(initial):
            raise SpecError(
                "initial_capacity_version must be a non-negative integer",
                path=f"{base}.initial_capacity_version",
            )
        if int(initial) not in global_budget:
            raise SpecError(
                "initial capacity version missing from global_budget",
                path=f"{base}.initial_capacity_version",
            )

        capacities = _parse_capacities(item, int(initial), global_budget, base)
        nodes.append(
            NodeSpec(
                node_id=str(node_id),
                host="127.0.0.1",
                port=int(port),
                seed=int(seed),
                initial_capacity_version=int(initial),
                capacities=capacities,
            )
        )
    if not nodes:
        raise SpecError("at least one node is required", path="nodes")
    return tuple(nodes)


def _parse_capacities(
    item: dict, initial: int, global_budget: dict[int, int], base: str
) -> tuple[CapacityLevel, ...]:
    raw_levels = item.get("capacities")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise SpecError("capacities must be a non-empty list", path=f"{base}.capacities")
    levels: list[CapacityLevel] = []
    seen_versions: set[int] = set()
    for idx, level_raw in enumerate(raw_levels):
        path = f"{base}.capacities[{idx}]"
        if not isinstance(level_raw, dict):
            raise SpecError("capacity level must be an object", path=path)
        version = level_raw.get("version")
        if not _is_nonneg_int(version):
            raise SpecError("capacity version must be a non-negative integer", path=f"{path}.version")
        if int(version) in seen_versions:
            raise SpecError("duplicate capacity version", path=f"{path}.version")
        if int(version) not in global_budget:
            raise SpecError(
                "capacity version missing from global_budget", path=f"{path}.version"
            )
        seen_versions.add(int(version))
        concurrency = level_raw.get("concurrency")
        if not _is_nonneg_int(concurrency):
            raise SpecError(
                "concurrency must be a non-negative integer", path=f"{path}.concurrency"
            )
        lease_limit = level_raw.get("lease_limit")
        if not _is_nonneg_int(lease_limit):
            raise SpecError(
                "lease_limit must be a non-negative integer", path=f"{path}.lease_limit"
            )
        levels.append(
            CapacityLevel(
                version=int(version),
                concurrency=int(concurrency),
                lease_limit=int(lease_limit),
            )
        )
    levels.sort(key=lambda level: level.version)
    if initial not in seen_versions:
        raise SpecError(
            "initial_capacity_version has no matching capacity level",
            path=f"{base}.initial_capacity_version",
        )
    return tuple(levels)


def _parse_shards(raw: dict) -> tuple[ShardSpec, ...]:
    if "shards" not in raw or not isinstance(raw["shards"], list):
        raise SpecError("shards must be a list", path="shards")
    seen: set[str] = set()
    shards: list[ShardSpec] = []
    for index, item in enumerate(raw["shards"]):
        base = f"shards[{index}]"
        if not isinstance(item, dict):
            raise SpecError("shard must be an object", path=base)
        shard_id = item.get("shard_id")
        if not _is_ident(shard_id):
            raise SpecError("shard_id must be a non-empty string", path=f"{base}.shard_id")
        if shard_id in seen:
            raise SpecError(f"duplicate shard_id {shard_id}", path=f"{base}.shard_id")
        seen.add(shard_id)

        depends = item.get("depends_on", [])
        if not isinstance(depends, list) or not all(_is_ident(d) for d in depends):
            raise SpecError(
                "depends_on must be a list of shard id strings", path=f"{base}.depends_on"
            )
        rows = item.get("sample_rows")
        if not _is_pos_int(rows):
            raise SpecError("sample_rows must be a positive integer", path=f"{base}.sample_rows")
        units = item.get("resource_units")
        if not _is_pos_int(units):
            raise SpecError(
                "resource_units must be a positive integer", path=f"{base}.resource_units"
            )
        priority = item.get("priority", 0)
        if not _is_nonneg_int(priority):
            raise SpecError("priority must be a non-negative integer", path=f"{base}.priority")
        shards.append(
            ShardSpec(
                shard_id=str(shard_id),
                depends_on=tuple(str(d) for d in depends),
                sample_rows=int(rows),
                resource_units=int(units),
                priority=int(priority),
            )
        )
    if not shards:
        raise SpecError("at least one shard is required", path="shards")
    return tuple(shards)


def _validate_dependencies(shards: tuple[ShardSpec, ...]) -> None:
    ids = {shard.shard_id for shard in shards}
    by_id = {shard.shard_id: shard for shard in shards}
    for index, shard in enumerate(shards):
        base = f"shards[{index}].depends_on"
        for dep in shard.depends_on:
            if dep not in ids:
                raise SpecError(f"dependency {dep} does not exist", path=base)
            if dep == shard.shard_id:
                raise SpecError("a shard cannot depend on itself", path=base)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(shard_id: str, stack: tuple[str, ...]) -> None:
        if shard_id in visited:
            return
        if shard_id in visiting:
            cycle = " -> ".join(stack + (shard_id,))
            raise SpecError(f"dependency cycle detected: {cycle}", path="shards.depends_on")
        visiting.add(shard_id)
        for dep in by_id[shard_id].depends_on:
            visit(dep, stack + (shard_id,))
        visiting.discard(shard_id)
        visited.add(shard_id)

    for shard in shards:
        visit(shard.shard_id, ())


_EVENT_KINDS = {
    "arrive",
    "cancel",
    "capacity",
    "renew",
    "freeze",
    "unfreeze",
    "_sleep",
}


def _parse_events(
    raw: dict,
    nodes: tuple[NodeSpec, ...],
    shards: tuple[ShardSpec, ...],
    global_budget: dict[int, int],
) -> tuple[EventSpec, ...]:
    if "events" not in raw or not isinstance(raw["events"], list):
        raise SpecError("events must be a list", path="events")
    shard_ids = {shard.shard_id for shard in shards}
    events: list[EventSpec] = []
    for index, item in enumerate(raw["events"]):
        base = f"events[{index}]"
        if not isinstance(item, dict):
            raise SpecError("event must be an object", path=base)
        kind = item.get("kind")
        if kind not in _EVENT_KINDS:
            raise SpecError(
                f"event kind must be one of {sorted(_EVENT_KINDS)}", path=f"{base}.kind"
            )
        shard_id = item.get("shard_id")
        if shard_id is not None:
            if not _is_ident(shard_id) or shard_id not in shard_ids:
                raise SpecError("event shard_id must reference a declared shard", path=f"{base}.shard_id")
            shard_id = str(shard_id)
        if kind in {"arrive", "cancel", "freeze", "unfreeze"} and shard_id is None:
            raise SpecError(f"{kind} event requires shard_id", path=f"{base}.shard_id")
        if kind == "_sleep":
            ms = item.get("ms", 20)
            if not _is_pos_int(ms):
                raise SpecError("_sleep event requires positive integer ms", path=f"{base}.ms")
        capacity_version = item.get("capacity_version")
        if capacity_version is not None:
            if not _is_nonneg_int(capacity_version):
                raise SpecError(
                    "capacity_version must be a non-negative integer",
                    path=f"{base}.capacity_version",
                )
            if int(capacity_version) not in global_budget:
                raise SpecError(
                    "event capacity version missing from global_budget",
                    path=f"{base}.capacity_version",
                )
            capacity_version = int(capacity_version)
        if kind == "capacity" and capacity_version is None:
            raise SpecError("capacity event requires capacity_version", path=f"{base}.capacity_version")
        lease_id = item.get("lease_id")
        if lease_id is not None and not _is_ident(lease_id):
            raise SpecError("lease_id must be a non-empty string", path=f"{base}.lease_id")
        if kind == "renew" and (lease_id is None or shard_id is None):
            raise SpecError("renew event requires lease_id and shard_id", path=base)
        events.append(
            EventSpec(
                seq=index,
                kind=str(kind),
                shard_id=shard_id,
                capacity_version=capacity_version,
                lease_id=str(lease_id) if lease_id is not None else None,
                ms=int(item["ms"]) if item.get("ms") is not None else None,
            )
        )
    return tuple(events)


def topological_order(spec: Spec) -> tuple[str, ...]:
    """Stable dependency order: dependencies first, ties broken by shard id."""
    by_id = {shard.shard_id: shard for shard in spec.shards}
    visited: set[str] = set()
    order: list[str] = []

    def visit(shard_id: str) -> None:
        if shard_id in visited:
            return
        for dep in sorted(by_id[shard_id].depends_on):
            visit(dep)
        visited.add(shard_id)
        order.append(shard_id)

    for shard_id in sorted(by_id):
        visit(shard_id)
    return tuple(order)
