"""Local JSON / NDJSON loading with deterministic strong validation."""

import json
from dataclasses import dataclass, field

from .errors import CycleError, DependError, SpecError


def _is_pos_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    concurrency: int
    lease_limit: int
    seed: int
    capacity_version: int
    work_ms: int = 0
    hold_ms: int = 0


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    deps: tuple
    units: int
    targets: tuple


@dataclass(frozen=True)
class Revision:
    version: int
    parent: int
    affects: tuple
    params: dict


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    shard_id: str | None = None
    revision: int | None = None
    node_id: str | None = None
    concurrency: int | None = None
    lease_limit: int | None = None


@dataclass
class Manifest:
    global_budget: int
    lease_ttl_ms: int
    nodes: dict
    shards: dict
    revisions: dict
    order: list = field(default_factory=list)

    def topo_order(self):
        return topo_sort(self.shards)

    def dependents(self):
        graph = {sid: [] for sid in self.shards}
        for sid, spec in self.shards.items():
            for dep in spec.deps:
                graph[dep].append(sid)
        return {k: tuple(sorted(v)) for k, v in graph.items()}


def topo_sort(shards: dict) -> list:
    indegree = {sid: len(spec.deps) for sid, spec in shards.items()}
    dependents = {sid: [] for sid in shards}
    for sid, spec in shards.items():
        for dep in spec.deps:
            dependents[dep].append(sid)
    ready = sorted(sid for sid, deg in indegree.items() if deg == 0)
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in sorted(dependents[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    if len(order) != len(shards):
        remaining = sorted(sid for sid, deg in indegree.items() if deg > 0)
        raise CycleError(
            "shard dependency graph contains a cycle",
            entity=remaining[0],
            field="deps",
            cycle=remaining,
        )
    return order


def load_manifest(path: str) -> Manifest:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise SpecError(f"manifest file not found: {path}", entity=path, field="file") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "manifest JSON is corrupted and cannot be parsed",
            entity=path,
            field="json",
            line=exc.lineno,
            column=exc.colno,
        ) from exc

    if not isinstance(raw, dict):
        raise SpecError("manifest root must be an object", entity=path, field="root")

    if not _is_pos_int(raw.get("global_budget")):
        raise SpecError("global_budget must be a positive integer", entity="global", field="global_budget")
    if not _is_pos_int(raw.get("lease_ttl_ms")):
        raise SpecError("lease_ttl_ms must be a positive integer", entity="global", field="lease_ttl_ms")

    nodes = _parse_nodes(raw.get("nodes"))
    shards = _parse_shards(raw.get("shards"), nodes)
    revisions = _parse_revisions(raw.get("revisions"))
    return Manifest(
        global_budget=raw["global_budget"],
        lease_ttl_ms=raw["lease_ttl_ms"],
        nodes=nodes,
        shards=shards,
        revisions=revisions,
        order=topo_sort(shards),
    )


def _parse_nodes(raw_nodes) -> dict:
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SpecError("nodes must be a non-empty array", entity="nodes", field="nodes")
    nodes = {}
    seen_ports = set()
    required = ("node_id", "host", "port", "concurrency", "lease_limit", "seed", "capacity_version")
    for entry in raw_nodes:
        if not isinstance(entry, dict):
            raise SpecError("node entry must be an object", entity="nodes", field="node")
        for key in required:
            if key not in entry:
                raise SpecError(f"node missing field {key}", entity=entry.get("node_id", "?"), field=key)
        node_id = entry["node_id"]
        if not isinstance(node_id, str) or not node_id:
            raise SpecError("node_id must be a non-empty string", entity=node_id, field="node_id")
        if node_id in nodes:
            raise SpecError("duplicate node identifier", entity=node_id, field="node_id")
        if entry["host"] != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", entity=node_id, field="host")
        if not _is_pos_int(entry["port"]) or entry["port"] > 65535:
            raise SpecError("port must be an integer in 1..65535", entity=node_id, field="port")
        if entry["port"] in seen_ports:
            raise SpecError("port must be globally unique", entity=node_id, field="port")
        seen_ports.add(entry["port"])
        if not _is_pos_int(entry["concurrency"]):
            raise SpecError("concurrency must be a positive integer", entity=node_id, field="concurrency")
        if not _is_pos_int(entry["lease_limit"]):
            raise SpecError("lease_limit must be a positive integer", entity=node_id, field="lease_limit")
        if not _is_int(entry["seed"]):
            raise SpecError("seed must be an integer", entity=node_id, field="seed")
        if not _is_pos_int(entry["capacity_version"]):
            raise SpecError("capacity_version must be a positive integer", entity=node_id, field="capacity_version")
        work_ms = entry.get("work_ms", 0)
        hold_ms = entry.get("hold_ms", 0)
        if not _is_int(work_ms) or work_ms < 0 or not _is_int(hold_ms) or hold_ms < 0:
            raise SpecError("work_ms/hold_ms must be non-negative integers", entity=node_id, field="work_ms")
        nodes[node_id] = NodeSpec(
            node_id=node_id,
            host=entry["host"],
            port=entry["port"],
            concurrency=entry["concurrency"],
            lease_limit=entry["lease_limit"],
            seed=entry["seed"],
            capacity_version=entry["capacity_version"],
            work_ms=work_ms,
            hold_ms=hold_ms,
        )
    return nodes


def _parse_shards(raw_shards, nodes: dict) -> dict:
    if not isinstance(raw_shards, list) or not raw_shards:
        raise SpecError("shards must be a non-empty array", entity="shards", field="shards")
    shards = {}
    for entry in raw_shards:
        if not isinstance(entry, dict):
            raise SpecError("shard entry must be an object", entity="shards", field="shard")
        sid = entry.get("shard_id")
        if not isinstance(sid, str) or not sid:
            raise SpecError("shard_id must be a non-empty string", entity=sid, field="shard_id")
        if sid in shards:
            raise SpecError("duplicate shard identifier", entity=sid, field="shard_id")
        deps = entry.get("deps", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SpecError("deps must be an array of strings", entity=sid, field="deps")
        if not _is_pos_int(entry.get("units")):
            raise SpecError("units must be a positive integer", entity=sid, field="units")
        targets = entry.get("targets")
        if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
            raise SpecError("targets must be a non-empty string array", entity=sid, field="targets")
        shards[sid] = ShardSpec(
            shard_id=sid,
            deps=tuple(deps),
            units=entry["units"],
            targets=tuple(targets),
        )
    for sid, spec in shards.items():
        for dep in spec.deps:
            if dep not in shards:
                raise DependError("dependency references an unknown shard", entity=sid, field="deps", ref=dep)
            if dep == sid:
                raise CycleError("shard depends on itself", entity=sid, field="deps")
        for target in spec.targets:
            if target not in nodes:
                raise DependError("target references an unknown node", entity=sid, field="targets", ref=target)
    topo_sort(shards)
    return shards


def _parse_revisions(raw_revisions) -> dict:
    if not isinstance(raw_revisions, list) or not raw_revisions:
        raise SpecError("revisions must be a non-empty array", entity="revisions", field="revisions")
    revisions = {}
    last_version = 0
    for entry in raw_revisions:
        if not isinstance(entry, dict):
            raise SpecError("revision entry must be an object", entity="revisions", field="revision")
        version = entry.get("version")
        if not _is_pos_int(version):
            raise SpecError("revision version must be a positive integer", entity=version, field="version")
        if version in revisions:
            raise SpecError("duplicate revision version", entity=version, field="version")
        if version <= last_version:
            raise SpecError("revision versions must be strictly increasing", entity=version, field="version")
        parent = entry.get("parent")
        if last_version == 0:
            if parent not in (None, 0):
                raise SpecError("first revision parent must be null or 0", entity=version, field="parent")
        elif parent != last_version:
            raise SpecError("revision parent must reference the previous revision", entity=version, field="parent")
        affects = entry.get("affects", [])
        if not isinstance(affects, list) or not all(isinstance(a, str) for a in affects):
            raise SpecError("affects must be an array of node ids", entity=version, field="affects")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise SpecError("params must be an object", entity=version, field="params")
        for key, value in params.items():
            if not _is_pos_int(value):
                raise SpecError("revision params must be positive integers", entity=version, field=f"params.{key}")
        revisions[version] = Revision(
            version=version,
            parent=parent or 0,
            affects=tuple(sorted(affects)),
            params=params,
        )
        last_version = version
    return revisions


def load_events(path: str, manifest: Manifest) -> list:
    events = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise SpecError(f"event file not found: {path}", entity=path, field="file") from exc
    with handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpecError("event NDJSON line is corrupted", entity=path, field=f"line:{line_no}") from exc
            events.append(_validate_event(raw, line_no, manifest, len(events)))
    return events


def _validate_event(raw, line_no, manifest: Manifest, expected_seq) -> Event:
    if not isinstance(raw, dict):
        raise SpecError("event must be an object", entity=f"line:{line_no}", field="event")
    seq = raw.get("seq")
    if not _is_int(seq) or seq != expected_seq:
        raise SpecError("event seq must be consecutive starting at 0", entity=f"line:{line_no}", field="seq")
    kind = raw.get("type")
    if kind not in ("submit", "cancel", "revise", "quota"):
        raise SpecError("unknown event type", entity=seq, field="type")
    if kind in ("submit", "cancel"):
        sid = raw.get("shard_id")
        if sid not in manifest.shards:
            raise DependError("event references an unknown shard", entity=seq, field="shard_id", ref=sid)
        return Event(seq=seq, kind=kind, shard_id=sid)
    if kind == "revise":
        version = raw.get("revision")
        if version not in manifest.revisions:
            raise DependError("event references an unknown revision", entity=seq, field="revision", ref=version)
        return Event(seq=seq, kind=kind, revision=version)
    node_id = raw.get("node_id")
    if node_id not in manifest.nodes:
        raise DependError("quota event references an unknown node", entity=seq, field="node_id", ref=node_id)
    if not _is_pos_int(raw.get("concurrency")) or not _is_pos_int(raw.get("lease_limit")):
        raise SpecError("quota values must be positive integers", entity=seq, field="concurrency")
    return Event(
        seq=seq,
        kind=kind,
        node_id=node_id,
        concurrency=raw["concurrency"],
        lease_limit=raw["lease_limit"],
    )
