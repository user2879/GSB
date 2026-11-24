"""Loading and strong validation of the four deterministic input documents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import BusinessError
from .util import (
    canonical_json,
    deep_merge,
    digest_text,
    find_cycle,
    is_int,
    is_nonneg_int,
    is_pos_int,
    stable_topological_order,
)

INPUT_FILES = {
    "nodes": "nodes.json",
    "shards": "shards.json",
    "revisions": "revisions.json",
    "events": "events.json",
}


def _err(
    code: str,
    message: str,
    *,
    position: str,
    field: str | None,
    shard_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> BusinessError:
    return BusinessError(code, message, position=position, field=field, shard_id=shard_id, detail=detail)


def _load_json(path: Path, *, component: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _err("E_SPEC", f"cannot read {component} file: {exc}", position=str(path), field="$") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _err(
            "E_SPEC",
            f"invalid JSON: {exc.msg}",
            position=f"{path.name}:{exc.lineno}:{exc.colno}",
            field="$",
        ) from exc


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
    dependencies: tuple[str, ...]
    units: int
    sample_count: int
    seed_offset: int


@dataclass(frozen=True)
class RevisionSpec:
    version: int
    parent: int | None
    node_changes: Mapping[str, Mapping[str, int]]
    global_budget: int
    aggregation: Mapping[str, int]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class EventSpec:
    seq: int
    type: str
    raw: Mapping[str, Any]

    @property
    def shard_id(self) -> str | None:
        value = self.raw.get("shard_id")
        return value if isinstance(value, str) else None

    @property
    def node_id(self) -> str | None:
        value = self.raw.get("node_id")
        return value if isinstance(value, str) else None


@dataclass
class Spec:
    nodes: dict[str, NodeSpec]
    shards: dict[str, ShardSpec]
    revisions: dict[int, RevisionSpec]
    events: list[EventSpec]
    global_budget: int
    default_lease_ttl_ms: int
    topo_order: list[str]
    source_digest: str
    source_label: str
    raw_documents: dict[str, Any] = field(repr=False)

    def node_config(self, version: int) -> dict[str, Any]:
        revision = self.revisions[version]
        return {
            "global_budget": revision.global_budget,
            "aggregation": dict(revision.aggregation),
            "nodes": {
                node_id: dict(values)
                for node_id, values in sorted(revision.node_changes.items())
            },
        }

    def public_summary(self) -> dict[str, Any]:
        return {
            "source": self.source_label,
            "source_digest": self.source_digest,
            "nodes": [self.nodes[node_id].id for node_id in sorted(self.nodes)],
            "shards": [self.shards[shard_id].id for shard_id in self.topo_order],
            "topological_order": self.topo_order,
            "revisions": sorted(self.revisions),
            "event_count": len(self.events),
            "global_budget": self.global_budget,
        }


def _read_source(sample_path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(sample_path)
    if source.is_dir():
        docs: dict[str, Any] = {}
        for component, filename in INPUT_FILES.items():
            docs[component] = _load_json(source / filename, component=component)
        return docs, source
    if not source.is_file():
        raise _err("E_SPEC", "sample path does not exist", position=str(source), field="$")
    combined = _load_json(source, component="sample")
    if not isinstance(combined, dict):
        raise _err("E_SPEC", "combined sample must be an object", position=source.name, field="$")
    missing = [name for name in INPUT_FILES if name not in combined]
    if missing:
        raise _err(
            "E_SPEC",
            "combined sample is missing component documents",
            position=source.name,
            field=missing[0],
        )
    return {name: combined[name] for name in INPUT_FILES}, source


def _as_list(value: Any, *, position: str, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise _err("E_SPEC", "expected an array", position=position, field=field_name)
    return value


def _need_obj(value: Any, *, position: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _err("E_SPEC", "expected an object", position=position, field="$")
    return value


def _check_format_version(root: dict[str, Any], *, position: str) -> None:
    if "format_version" in root and root.get("format_version") != 1:
        raise _err("E_SPEC", "unsupported input format_version (expected integer 1)", position=position, field="format_version")


def _need_str(value: Any, *, position: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err("E_SPEC", "expected a non-empty string", position=position, field=field_name)
    return value


def _need_int(value: Any, *, position: str, field_name: str, positive: bool = True) -> int:
    valid = is_pos_int(value) if positive else is_nonneg_int(value)
    if not valid:
        qualifier = "positive" if positive else "non-negative"
        raise _err("E_SPEC", f"expected a {qualifier} integer", position=position, field=field_name)
    return int(value)


def _validate_nodes(doc: Any) -> tuple[dict[str, NodeSpec], int, int]:
    root = _need_obj(doc, position="nodes.json")
    _check_format_version(root, position="nodes.json")
    position_root = "nodes.json"
    global_budget = _need_int(root.get("global_budget"), position=position_root, field_name="global_budget")
    ttl = _need_int(
        root.get("default_lease_ttl_ms", 5000),
        position=position_root,
        field_name="default_lease_ttl_ms",
    )
    rows = _as_list(root.get("nodes"), position=position_root, field_name="nodes")
    if not rows:
        raise _err("E_SPEC", "at least one node is required", position=position_root, field="nodes")
    nodes: dict[str, NodeSpec] = {}
    ports: set[int] = set()
    for index, value in enumerate(rows):
        pos = f"nodes.json:nodes[{index}]"
        row = _need_obj(value, position=pos)
        node_id = _need_str(row.get("id"), position=pos, field_name="id")
        if node_id in nodes:
            raise _err("E_SPEC", f"duplicate node id: {node_id}", position=pos, field="id")
        host = row.get("host")
        if host != "127.0.0.1":
            raise _err("E_SPEC", "host must be the literal 127.0.0.1", position=pos, field="host")
        port = _need_int(row.get("port"), position=pos, field_name="port")
        if port > 65535:
            raise _err("E_SPEC", "port must be <= 65535", position=pos, field="port")
        if port in ports:
            raise _err("E_SPEC", f"duplicate loopback port: {port}", position=pos, field="port")
        ports.add(port)
        concurrency = _need_int(row.get("concurrency"), position=pos, field_name="concurrency")
        lease_limit = _need_int(row.get("lease_limit"), position=pos, field_name="lease_limit")
        seed = _need_int(row.get("seed"), position=pos, field_name="seed", positive=False)
        capacity_version = _need_int(row.get("capacity_version"), position=pos, field_name="capacity_version")
        nodes[node_id] = NodeSpec(
            id=node_id,
            host=host,
            port=port,
            concurrency=concurrency,
            lease_limit=lease_limit,
            seed=seed,
            capacity_version=capacity_version,
        )
    return nodes, global_budget, ttl


def _validate_shards(doc: Any, nodes: Mapping[str, NodeSpec]) -> tuple[dict[str, ShardSpec], list[str]]:
    root = _need_obj(doc, position="shards.json")
    _check_format_version(root, position="shards.json")
    rows = _as_list(root.get("shards"), position="shards.json", field_name="shards")
    if not rows:
        raise _err("E_SPEC", "at least one shard is required", position="shards.json", field="shards")
    shards: dict[str, ShardSpec] = {}
    for index, value in enumerate(rows):
        pos = f"shards.json:shards[{index}]"
        row = _need_obj(value, position=pos)
        shard_id = _need_str(row.get("id"), position=pos, field_name="id")
        if shard_id in shards:
            raise _err("E_SPEC", f"duplicate shard id: {shard_id}", position=pos, field="id", shard_id=shard_id)
        node_id = _need_str(row.get("node_id"), position=pos, field_name="node_id")
        if node_id not in nodes:
            raise _err(
                "E_SPEC",
                f"unknown node reference: {node_id}",
                position=pos,
                field="node_id",
                shard_id=shard_id,
            )
        deps_raw = _as_list(row.get("dependencies"), position=pos, field_name="dependencies")
        deps: list[str] = []
        for dep_index, dep in enumerate(deps_raw):
            if not isinstance(dep, str) or not dep:
                raise _err(
                    "E_GRAPH",
                    "dependency must be a non-empty shard id",
                    position=f"{pos}.dependencies[{dep_index}]",
                    field="dependencies",
                    shard_id=shard_id,
                )
            if dep == shard_id:
                raise _err("E_GRAPH", "a shard cannot depend on itself", position=pos, field="dependencies", shard_id=shard_id)
            if dep in deps:
                raise _err("E_GRAPH", f"duplicate dependency: {dep}", position=pos, field="dependencies", shard_id=shard_id)
            deps.append(dep)
        units = _need_int(row.get("units"), position=pos, field_name="units")
        sample_count = _need_int(row.get("sample_count"), position=pos, field_name="sample_count")
        seed_offset = _need_int(row.get("seed_offset"), position=pos, field_name="seed_offset", positive=False)
        shards[shard_id] = ShardSpec(
            id=shard_id,
            node_id=node_id,
            dependencies=tuple(deps),
            units=units,
            sample_count=sample_count,
            seed_offset=seed_offset,
        )
    for shard in shards.values():
        for dep in shard.dependencies:
            if dep not in shards:
                raise _err(
                    "E_GRAPH",
                    f"dependency does not exist: {dep}",
                    position=f"shards.json:shards[{list(shards).index(shard.id)}].dependencies",
                    field="dependencies",
                    shard_id=shard.id,
                )
    dependency_map = {shard_id: shard.dependencies for shard_id, shard in shards.items()}
    cycle = find_cycle(shards, dependency_map)
    if cycle:
        cycle_text = " -> ".join(cycle)
        cycle_members = cycle[:-1] if cycle and cycle[0] == cycle[-1] else cycle
        first_shard = min(cycle_members, key=lambda shard_id: list(shards).index(shard_id))
        index = list(shards).index(first_shard)
        raise _err(
            "E_GRAPH",
            f"dependency cycle: {cycle_text}",
            position=f"shards.json:shards[{index}].dependencies",
            field="dependencies",
            shard_id=first_shard,
            detail={"cycle": cycle},
        )
    topo = stable_topological_order(shards, dependency_map)
    return shards, topo


def _validate_revisions(
    doc: Any,
    nodes: Mapping[str, NodeSpec],
    fallback_budget: int,
) -> dict[int, RevisionSpec]:
    root = _need_obj(doc, position="revisions.json")
    _check_format_version(root, position="revisions.json")
    rows = _as_list(root.get("revisions"), position="revisions.json", field_name="revisions")
    if not rows:
        raise _err("E_SPEC", "at least one revision is required", position="revisions.json", field="revisions")
    base_nodes = {
        node.id: {
            "concurrency": node.concurrency,
            "lease_limit": node.lease_limit,
            "capacity_version": node.capacity_version,
        }
        for node in nodes.values()
    }
    base = {
        "global_budget": fallback_budget,
        "aggregation": {"scale": 1000, "group_modulus": 7},
        "nodes": base_nodes,
    }
    revisions: dict[int, RevisionSpec] = {}
    effective: dict[str, Any] = base
    previous_version = 0
    for index, value in enumerate(rows):
        pos = f"revisions.json:revisions[{index}]"
        row = _need_obj(value, position=pos)
        version = _need_int(row.get("version"), position=pos, field_name="version")
        if version <= previous_version:
            raise _err("E_SPEC", "revision versions must strictly increase", position=pos, field="version")
        if index == 0 and version != 1:
            raise _err("E_SPEC", "the root revision version must be 1", position=pos, field="version")
        parent_raw = row.get("parent")
        parent: int | None
        if parent_raw is None:
            if index != 0:
                raise _err("E_SPEC", "only the root revision may have a null parent", position=pos, field="parent")
            parent = None
        else:
            if not is_pos_int(parent_raw):
                raise _err("E_SPEC", "parent must be a positive integer or null", position=pos, field="parent")
            parent = int(parent_raw)
            if parent not in revisions:
                raise _err("E_SPEC", f"parent revision does not exist: {parent}", position=pos, field="parent")
            if parent >= version:
                raise _err("E_SPEC", "parent revision must be lower than version", position=pos, field="parent")
        global_budget: int | None = None
        if "global_budget" in row:
            global_budget = _need_int(row.get("global_budget"), position=pos, field_name="global_budget")
        aggregation: dict[str, int] = {}
        if "aggregation" in row:
            agg_pos = f"{pos}.aggregation"
            agg = _need_obj(row.get("aggregation"), position=agg_pos)
            for key in ("scale", "group_modulus"):
                if key in agg:
                    aggregation[key] = _need_int(agg.get(key), position=agg_pos, field_name=key)
            allowed = {"scale", "group_modulus"}
            unknown = sorted(set(agg) - allowed)
            if unknown:
                raise _err("E_SPEC", f"unknown aggregation field: {unknown[0]}", position=agg_pos, field=unknown[0])
        node_changes_raw = row.get("nodes", {})
        node_changes_obj = _need_obj(node_changes_raw, position=f"{pos}.nodes")
        node_changes: dict[str, dict[str, int]] = {}
        for node_id, change_value in node_changes_obj.items():
            if node_id not in nodes:
                raise _err("E_SPEC", f"unknown node in revision: {node_id}", position=f"{pos}.nodes", field=node_id)
            change_pos = f"{pos}.nodes.{node_id}"
            change_obj = _need_obj(change_value, position=change_pos)
            allowed = {"concurrency", "lease_limit", "capacity_version"}
            unknown = sorted(set(change_obj) - allowed)
            if unknown:
                raise _err("E_SPEC", f"unknown node config field: {unknown[0]}", position=change_pos, field=unknown[0])
            normalized: dict[str, int] = {}
            for key in ("concurrency", "lease_limit", "capacity_version"):
                if key in change_obj:
                    normalized[key] = _need_int(change_obj.get(key), position=change_pos, field_name=key)
            node_changes[node_id] = normalized
        if index == 0:
            if global_budget is None:
                raise _err("E_SPEC", "root revision must declare global_budget", position=pos, field="global_budget")
            if "scale" not in aggregation or "group_modulus" not in aggregation:
                raise _err(
                    "E_SPEC",
                    "root revision must declare aggregation.scale and aggregation.group_modulus",
                    position=f"{pos}.aggregation",
                    field="aggregation",
                )
            missing_nodes = sorted(set(nodes) - set(node_changes))
            if missing_nodes:
                raise _err(
                    "E_SPEC",
                    f"root revision is missing node config: {missing_nodes[0]}",
                    position=f"{pos}.nodes",
                    field=missing_nodes[0],
                )
            for node_id in sorted(nodes):
                for key in ("concurrency", "lease_limit", "capacity_version"):
                    if key not in node_changes[node_id]:
                        raise _err(
                            "E_SPEC",
                            f"root revision is missing {key}",
                            position=f"{pos}.nodes.{node_id}",
                            field=key,
                        )
        effective = deep_merge(
            effective,
            {
                **({"global_budget": global_budget} if global_budget is not None else {}),
                **({"aggregation": aggregation} if aggregation else {}),
                **({"nodes": node_changes} if node_changes else {}),
            },
        )
        effective_budget = _need_int(effective.get("global_budget"), position=pos, field_name="global_budget")
        revisions[version] = RevisionSpec(
            version=version,
            parent=parent,
            node_changes={
                node_id: {
                    **dict(base_nodes[node_id]),
                    **dict(node_changes.get(node_id, {})),
                }
                for node_id in sorted(nodes)
            },
            global_budget=effective_budget,
            aggregation=dict(effective["aggregation"]),
            raw=row,
        )
        previous_version = version
    return revisions


def _validate_events(
    doc: Any,
    shards: Mapping[str, ShardSpec],
    nodes: Mapping[str, NodeSpec],
    revisions: Mapping[int, RevisionSpec],
) -> list[EventSpec]:
    root = _need_obj(doc, position="events.json")
    _check_format_version(root, position="events.json")
    rows = _as_list(root.get("events"), position="events.json", field_name="events")
    events: list[EventSpec] = []
    previous_seq = 0
    active_revision = 1
    seen_seq: set[int] = set()
    for index, value in enumerate(rows):
        pos = f"events.json:events[{index}]"
        row = _need_obj(value, position=pos)
        seq = _need_int(row.get("seq"), position=pos, field_name="seq")
        if seq in seen_seq:
            raise _err("E_SPEC", "duplicate event sequence number", position=pos, field="seq")
        if seq <= previous_seq:
            raise _err("E_SPEC", "event sequence numbers must increase in arrival order", position=pos, field="seq")
        seen_seq.add(seq)
        event_type = _need_str(row.get("type"), position=pos, field_name="type")
        if event_type == "dispatch":
            shard_id = _need_str(row.get("shard_id"), position=pos, field_name="shard_id")
            if shard_id not in shards:
                raise _err(
                    "E_SPEC",
                    f"dispatch references unknown shard: {shard_id}",
                    position=pos,
                    field="shard_id",
                    shard_id=shard_id,
                )
            if "ttl_ms" in row:
                _need_int(row.get("ttl_ms"), position=pos, field_name="ttl_ms")
            if "node_id" in row and row.get("node_id") != shards[shard_id].node_id:
                raise _err(
                    "E_SPEC",
                    "dispatch node_id must match the shard assignment",
                    position=pos,
                    field="node_id",
                    shard_id=shard_id,
                )
        elif event_type == "cancel":
            shard_id = _need_str(row.get("shard_id"), position=pos, field_name="shard_id")
            if shard_id not in shards:
                raise _err("E_SPEC", f"cancel references unknown shard: {shard_id}", position=pos, field="shard_id", shard_id=shard_id)
            if "reason" in row and not isinstance(row.get("reason"), str):
                raise _err("E_SPEC", "cancel reason must be a string", position=pos, field="reason", shard_id=shard_id)
        elif event_type == "quota":
            node_id = _need_str(row.get("node_id"), position=pos, field_name="node_id")
            if node_id not in nodes:
                raise _err("E_SPEC", f"quota references unknown node: {node_id}", position=pos, field="node_id")
            changed = False
            for field_name in ("concurrency", "lease_limit"):
                if field_name in row:
                    _need_int(row.get(field_name), position=pos, field_name=field_name)
                    changed = True
            if "global_budget" in row:
                _need_int(row.get("global_budget"), position=pos, field_name="global_budget")
                changed = True
            if not changed:
                raise _err("E_SPEC", "quota event must change at least one quota field", position=pos, field="node_id")
        elif event_type == "revision":
            version = _need_int(row.get("version"), position=pos, field_name="version")
            if version not in revisions:
                raise _err("E_SPEC", f"unknown revision version: {version}", position=pos, field="version")
            if version <= active_revision:
                raise _err("E_SPEC", "revision events must advance the active revision", position=pos, field="version")
            active_revision = version
        else:
            raise _err("E_SPEC", f"unknown event type: {event_type}", position=pos, field="type")
        events.append(EventSpec(seq=seq, type=event_type, raw=row))
        previous_seq = seq
    return events


def load_spec(sample_path: str | Path) -> Spec:
    """Load and validate every document before any child process can start."""
    docs, source = _read_source(sample_path)
    nodes, budget, ttl = _validate_nodes(docs["nodes"])
    shards, topo = _validate_shards(docs["shards"], nodes)
    revisions = _validate_revisions(docs["revisions"], nodes, budget)
    events = _validate_events(docs["events"], shards, nodes, revisions)
    source_digest = digest_text(canonical_json(docs))
    return Spec(
        nodes=nodes,
        shards=shards,
        revisions=revisions,
        events=events,
        global_budget=budget,
        default_lease_ttl_ms=ttl,
        topo_order=topo,
        source_digest=source_digest,
        source_label=str(source),
        raw_documents=docs,
    )


def revision_node_values(spec: Spec, version: int) -> dict[str, dict[str, int]]:
    revision = spec.revisions[version]
    return {node_id: dict(values) for node_id, values in revision.node_changes.items()}


def direct_invalidation_reasons(
    spec: Spec,
    old_version: int,
    new_version: int,
) -> dict[str, list[str]]:
    """Compute direct invalidation reasons from a revision transition."""
    old = spec.revisions[old_version]
    new = spec.revisions[new_version]
    reasons: dict[str, list[str]] = {}
    if old.global_budget != new.global_budget:
        for shard in spec.shards.values():
            reasons.setdefault(shard.id, []).append(f"revision:{new_version}:global_budget")
    if dict(old.aggregation) != dict(new.aggregation):
        for shard in spec.shards.values():
            reasons.setdefault(shard.id, []).append(f"revision:{new_version}:aggregation")
    for node_id in sorted(spec.nodes):
        if dict(old.node_changes[node_id]) != dict(new.node_changes[node_id]):
            for shard in spec.shards.values():
                if shard.node_id == node_id:
                    reasons.setdefault(shard.id, []).append(f"revision:{new_version}:node:{node_id}")
    return reasons
