from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError

MISSING = object()


def is_int(value: Any) -> bool:
    return type(value) is int


def _get(obj: dict[str, Any], canonical: str, aliases: Iterable[str] = (), default: Any = MISSING) -> Any:
    if canonical in obj:
        return obj[canonical]
    for alias in aliases:
        if alias in obj:
            return obj[alias]
    if default is MISSING:
        raise SpecError(f"missing required field {canonical!r}", field=canonical)
    return default


def _require_type(
    value: Any,
    typ: type | tuple[type, ...],
    *,
    path: str,
    field_name: str,
    type_label: str,
) -> Any:
    if isinstance(typ, tuple) and int in typ and bool not in typ:
        pass
    if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
        raise SpecError(f"field {field_name!r} must be {type_label}", path=path, field=field_name)
    return value


def _positive_int(value: Any, *, path: str, field_name: str) -> int:
    if not is_int(value) or value <= 0:
        raise SpecError(f"field {field_name!r} must be a positive integer", path=path, field=field_name)
    return value


def _nonnegative_int(value: Any, *, path: str, field_name: str) -> int:
    if not is_int(value) or value < 0:
        raise SpecError(f"field {field_name!r} must be a non-negative integer", path=path, field=field_name)
    return value


def _nonempty_str(value: Any, *, path: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"field {field_name!r} must be a non-empty string", path=path, field=field_name)
    return value


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    concurrency_quota: int
    lease_limit: int
    seed: int
    capacity_version: int
    renewal_limit: int = 3
    fixed_point_scale: int = 1_000_000
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    dependencies: tuple[str, ...]
    sample_rows: int
    resource_units: int
    priority: int
    order_index: int
    node_refs: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RevisionSpec:
    revision: int
    parent_revision: int | None
    config: dict[str, Any]
    changed_shard_ids: tuple[str, ...]
    changed_node_ids: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class Spec:
    path: Path | None
    raw: dict[str, Any]
    format_version: int
    global_budget_units: int
    lease_ttl_ms: int
    socket_timeout_ms: int
    sample_seed: int
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    revisions: tuple[RevisionSpec, ...]
    events: tuple[dict[str, Any], ...]
    node_by_id: dict[str, NodeSpec]
    shard_by_id: dict[str, ShardSpec]

    @property
    def initial_revision(self) -> int:
        return self.revisions[0].revision

    @property
    def latest_revision(self) -> int:
        return self.revisions[-1].revision


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read spec file {path}: {exc}", path=str(path), field="$") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            path=f"{path}:{exc.lineno}:{exc.colno}",
            field="$",
        ) from exc
    return parse_spec(raw, path=path)


def parse_spec(raw: Any, *, path: str | Path | None = None) -> Spec:
    source = str(path) if path is not None else "<dict>"
    if not isinstance(raw, dict):
        raise SpecError("top-level JSON value must be an object", path=source, field="$")

    format_version = _positive_int(
        _get(raw, "format_version", ("formatVersion",), 1), path=source, field_name="format_version"
    )
    if format_version != 1:
        raise SpecError("only format_version=1 is supported", path=source, field="format_version")

    budget = _positive_int(
        _get(raw, "global_budget_units", ("global_budget", "budget_units"), MISSING),
        path=f"{source}$.global_budget_units",
        field_name="global_budget_units",
    )
    ttl = _positive_int(_get(raw, "lease_ttl_ms", ("ttl_ms",), 1_000), path=source, field_name="lease_ttl_ms")
    timeout_ms = _positive_int(
        _get(raw, "socket_timeout_ms", ("timeout_ms",), 2_000), path=source, field_name="socket_timeout_ms"
    )
    sample_seed = _nonnegative_int(
        _get(raw, "sample_seed", ("fixed_seed",), 20261126), path=source, field_name="sample_seed"
    )

    raw_nodes = _get(raw, "nodes", (), MISSING)
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SpecError("field 'nodes' must be a non-empty list", path=f"{source}$.nodes", field="nodes")

    nodes: list[NodeSpec] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, item in enumerate(raw_nodes):
        base = f"{source}$.nodes[{index}]"
        if not isinstance(item, dict):
            raise SpecError("node declaration must be an object", path=base, field="nodes")
        node_id = _nonempty_str(
            _get(item, "node_id", ("id",), MISSING), path=base, field_name="node_id"
        )
        if node_id in seen_node_ids:
            raise SpecError(f"duplicate node_id {node_id!r}", path=f"{base}.node_id", field="node_id")
        seen_node_ids.add(node_id)

        host = _nonempty_str(_get(item, "host", ("address", "bind"), "127.0.0.1"), path=base, field_name="host")
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", path=f"{base}.host", field="host")

        port = _positive_int(_get(item, "port", (), MISSING), path=f"{base}.port", field_name="port")
        if not (1024 <= port <= 65535):
            raise SpecError("port must be an unprivileged high port in [1024, 65535]", path=f"{base}.port", field="port")
        if port in seen_ports:
            raise SpecError(f"duplicate loopback port {port}", path=f"{base}.port", field="port")
        seen_ports.add(port)

        concurrency = _positive_int(
            _get(item, "concurrency_quota", ("concurrency", "max_concurrency"), MISSING),
            path=f"{base}.concurrency_quota",
            field_name="concurrency_quota",
        )
        lease_limit = _positive_int(
            _get(item, "lease_limit", ("max_leases", "lease_quota"), MISSING),
            path=f"{base}.lease_limit",
            field_name="lease_limit",
        )
        seed = _nonnegative_int(
            _get(item, "seed", ("random_seed",), MISSING), path=f"{base}.seed", field_name="seed"
        )
        if seed > 2**32 - 1:
            raise SpecError("seed must fit uint32 for deterministic NumPy seeding", path=f"{base}.seed", field="seed")
        capacity_version = _positive_int(
            _get(item, "capacity_version", ("capacityVersion",), MISSING),
            path=f"{base}.capacity_version",
            field_name="capacity_version",
        )
        renewal_limit = _nonnegative_int(
            _get(item, "renewal_limit", ("max_renewals",), 3),
            path=f"{base}.renewal_limit",
            field_name="renewal_limit",
        )
        scale = _positive_int(
            _get(item, "fixed_point_scale", ("scale_q",), 1_000_000),
            path=f"{base}.fixed_point_scale",
            field_name="fixed_point_scale",
        )
        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=port,
                concurrency_quota=concurrency,
                lease_limit=lease_limit,
                seed=seed,
                capacity_version=capacity_version,
                renewal_limit=renewal_limit,
                fixed_point_scale=scale,
                raw=item,
            )
        )

    raw_shards = _get(raw, "shards", ("shard_list",), MISSING)
    if not isinstance(raw_shards, list) or not raw_shards:
        raise SpecError("field 'shards' must be a non-empty list", path=f"{source}$.shards", field="shards")

    shards: list[ShardSpec] = []
    seen_shard_ids: set[str] = set()
    node_ids = seen_node_ids
    for index, item in enumerate(raw_shards):
        base = f"{source}$.shards[{index}]"
        if not isinstance(item, dict):
            raise SpecError("shard declaration must be an object", path=base, field="shards")
        shard_id = _nonempty_str(_get(item, "shard_id", ("id",), MISSING), path=base, field_name="shard_id")
        if shard_id in seen_shard_ids:
            raise SpecError(f"duplicate shard_id {shard_id!r}", path=f"{base}.shard_id", field="shard_id")
        seen_shard_ids.add(shard_id)

        deps_value = _get(item, "dependencies", ("deps", "upstream"), [])
        if not isinstance(deps_value, list) or not all(isinstance(dep, str) and dep for dep in deps_value):
            raise SpecError("dependencies must be a list of non-empty shard ids", path=f"{base}.dependencies", field="dependencies")
        if len(set(deps_value)) != len(deps_value):
            raise SpecError("dependencies must not contain duplicates", path=f"{base}.dependencies", field="dependencies")

        rows = _positive_int(
            _get(item, "sample_rows", ("rows",), MISSING), path=f"{base}.sample_rows", field_name="sample_rows"
        )
        if rows > 1_000_000:
            raise SpecError("sample_rows is capped at 1,000,000 for local demo safety", path=f"{base}.sample_rows", field="sample_rows")
        units = _positive_int(
            _get(item, "resource_units", ("units",), MISSING), path=f"{base}.resource_units", field_name="resource_units"
        )
        if units > budget:
            raise SpecError(
                f"resource_units {units} exceeds global_budget_units {budget}; shard could never run",
                path=f"{base}.resource_units",
                field="resource_units",
            )
        priority = _get(item, "priority", (), MISSING)
        if not is_int(priority):
            raise SpecError("priority must be an integer", path=f"{base}.priority", field="priority")

        node_refs_value = _get(item, "node_refs", ("nodes", "references"), [])
        if isinstance(node_refs_value, str):
            node_refs_value = [node_refs_value]
        if not isinstance(node_refs_value, list) or not all(isinstance(n, str) and n for n in node_refs_value):
            raise SpecError("node_refs must be a list of node ids", path=f"{base}.node_refs", field="node_refs")
        node_refs = tuple(node_refs_value)
        for ref_index, node_ref in enumerate(node_refs):
            if node_ref not in node_ids:
                raise SpecError(
                    f"node_ref {node_ref!r} does not exist",
                    path=f"{base}.node_refs[{ref_index}]",
                    field="node_refs",
                )
        explicit_node = _get(item, "node_id", ("node",), None)
        if explicit_node is not None:
            _nonempty_str(explicit_node, path=base, field_name="node_id")
            if explicit_node not in node_ids:
                raise SpecError(f"node_id {explicit_node!r} does not exist", path=f"{base}.node_id", field="node_id")
            if explicit_node not in node_refs:
                node_refs = node_refs + (explicit_node,)

        shards.append(
            ShardSpec(
                shard_id=shard_id,
                dependencies=tuple(deps_value),
                sample_rows=rows,
                resource_units=units,
                priority=priority,
                order_index=index,
                node_refs=node_refs,
                raw=item,
            )
        )

    shard_ids = seen_shard_ids
    for shard in shards:
        for dep_index, dep in enumerate(shard.dependencies):
            if dep not in shard_ids:
                index = shard.order_index
                raise SpecError(
                    f"dependency {dep!r} does not exist",
                    path=f"{source}$.shards[{index}].dependencies[{dep_index}]",
                    field="dependencies",
                )

    _assert_dag(shards, source)

    raw_revisions = _get(raw, "revisions", ("revision_chain",), [])
    if not isinstance(raw_revisions, list) or not raw_revisions:
        raise SpecError("field 'revisions' must be a non-empty list", path=f"{source}$.revisions", field="revisions")
    revisions = _parse_revisions(raw_revisions, source, shard_ids, node_ids)

    raw_events = _get(raw, "events", ("event_stream",), [])
    if not isinstance(raw_events, list):
        raise SpecError("field 'events' must be a list", path=f"{source}$.events", field="events")
    events = _parse_events(raw_events, source, shard_ids, node_ids)

    return Spec(
        path=Path(source) if path is not None else None,
        raw=raw,
        format_version=format_version,
        global_budget_units=budget,
        lease_ttl_ms=ttl,
        socket_timeout_ms=timeout_ms,
        sample_seed=sample_seed,
        nodes=tuple(nodes),
        shards=tuple(shards),
        revisions=tuple(revisions),
        events=events,
        node_by_id={node.node_id: node for node in nodes},
        shard_by_id={shard.shard_id: shard for shard in shards},
    )


def _assert_dag(shards: list[ShardSpec], source: str) -> None:
    by_id = {shard.shard_id: shard for shard in shards}
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(shard_id: str) -> None:
        marker = state.get(shard_id, 0)
        if marker == 1:
            cycle_start = stack.index(shard_id)
            cycle = stack[cycle_start:] + [shard_id]
            shard = by_id[shard_id]
            raise SpecError(
                "dependency cycle detected: " + " -> ".join(cycle),
                path=f"{source}$.shards[{shard.order_index}].dependencies",
                field="dependencies",
                basis={"cycle": cycle},
            )
        if marker == 2:
            return
        state[shard_id] = 1
        stack.append(shard_id)
        for dep in by_id[shard_id].dependencies:
            visit(dep)
        stack.pop()
        state[shard_id] = 2

    for shard in shards:
        visit(shard.shard_id)


def _parse_revisions(
    raw_revisions: list[Any], source: str, shard_ids: set[str], node_ids: set[str]
) -> list[RevisionSpec]:
    revisions: list[RevisionSpec] = []
    seen: set[int] = set()
    for index, item in enumerate(raw_revisions):
        base = f"{source}$.revisions[{index}]"
        if not isinstance(item, dict):
            raise SpecError("revision must be an object", path=base, field="revisions")
        revision = _positive_int(_get(item, "revision", ("revision_id", "version"), MISSING), path=base, field_name="revision")
        if revision in seen:
            raise SpecError(f"duplicate revision {revision}", path=f"{base}.revision", field="revision")
        if index > 0 and revision <= revisions[-1].revision:
            raise SpecError(
                "revisions must be strictly increasing in declaration order",
                path=f"{base}.revision",
                field="revision",
            )
        seen.add(revision)
        parent = _get(item, "parent_revision", ("parent",), None)
        if parent is not None:
            parent = _positive_int(parent, path=f"{base}.parent_revision", field_name="parent_revision")
            if parent == revision:
                raise SpecError("revision cannot be its own parent", path=f"{base}.parent_revision", field="parent_revision")
            if parent not in seen:
                raise SpecError(
                    f"parent_revision {parent} does not reference an earlier revision",
                    path=f"{base}.parent_revision",
                    field="parent_revision",
                )
        config = _get(item, "config", ("values",), {})
        if not isinstance(config, dict):
            raise SpecError("config must be an object", path=f"{base}.config", field="config")

        changed_shards: list[str] = []
        changed_nodes: list[str] = []
        for key, target in (("changed_shard_ids", "shard"), ("changed_node_ids", "node")):
            values = _get(item, key, (), [])
            if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
                raise SpecError(f"{key} must be a list of id strings", path=f"{base}.{key}", field=key)
            known = shard_ids if target == "shard" else node_ids
            for pos, value in enumerate(values):
                if value not in known:
                    raise SpecError(f"{key}[{pos}] references unknown {target} {value!r}", path=f"{base}.{key}[{pos}]", field=key)
            if target == "shard":
                changed_shards.extend(values)
            else:
                changed_nodes.extend(values)

        changes = _get(item, "changes", (), [])
        if not isinstance(changes, list):
            raise SpecError("changes must be a list", path=f"{base}.changes", field="changes")
        for change_index, change in enumerate(changes):
            change_base = f"{base}.changes[{change_index}]"
            if not isinstance(change, dict):
                raise SpecError("change must be an object", path=change_base, field="changes")
            kind = _get(change, "kind", ("type",), MISSING)
            if kind not in {"shard", "node", "global"}:
                raise SpecError("change.kind must be shard, node, or global", path=f"{change_base}.kind", field="kind")
            shard_ref = _get(change, "shard_id", ("id",), None) if kind == "shard" else _get(change, "shard_id", (), None)
            node_ref = _get(change, "node_id", ("id",), None) if kind == "node" else _get(change, "node_id", (), None)
            if kind == "shard":
                if not isinstance(shard_ref, str) or shard_ref not in shard_ids:
                    raise SpecError("change must reference a known shard_id", path=f"{change_base}.shard_id", field="shard_id")
                changed_shards.append(shard_ref)
            elif kind == "node":
                if not isinstance(node_ref, str) or node_ref not in node_ids:
                    raise SpecError("change must reference a known node_id", path=f"{change_base}.node_id", field="node_id")
                changed_nodes.append(node_ref)

        revisions.append(
            RevisionSpec(
                revision=revision,
                parent_revision=parent,
                config=config,
                changed_shard_ids=tuple(dict.fromkeys(changed_shards)),
                changed_node_ids=tuple(dict.fromkeys(changed_nodes)),
                raw=item,
            )
        )

    by_revision = {rev.revision: rev for rev in revisions}
    for rev in revisions:
        seen_chain: set[int] = set()
        current = rev.revision
        while current is not None:
            if current in seen_chain:
                raise SpecError(
                    "parent revision graph contains a cycle",
                    path=f"{source}$.revisions",
                    field="parent_revision",
                    basis={"revision": rev.revision, "cycle": sorted(seen_chain)},
                )
            seen_chain.add(current)
            parent = by_revision[current].parent_revision
            current = parent
    return revisions


def _parse_events(
    raw_events: list[Any], source: str, shard_ids: set[str], node_ids: set[str]
) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    seen_sequences: set[int] = set()
    known_types = {
        "lease_apply",
        "lease_renew",
        "lease_release",
        "lease_expire",
        "expire_due",
        "quota_change",
        "cancel_shard",
    }
    for index, item in enumerate(raw_events):
        base = f"{source}$.events[{index}]"
        if not isinstance(item, dict):
            raise SpecError("event must be an object", path=base, field="events")
        event = dict(item)
        seq = _get(event, "seq", ("arrival_seq", "sequence"), index + 1)
        if not is_int(seq) or seq <= 0:
            raise SpecError("event seq must be a positive integer", path=f"{base}.seq", field="seq")
        if seq != index + 1:
            raise SpecError(
                f"events must be ordered by contiguous arrival sequence; expected {index + 1}, got {seq}",
                path=f"{base}.seq",
                field="seq",
            )
        if seq in seen_sequences:
            raise SpecError(f"duplicate event seq {seq}", path=f"{base}.seq", field="seq")
        seen_sequences.add(seq)
        event["seq"] = seq

        event_type = _nonempty_str(_get(event, "type", ("event_type",), MISSING), path=base, field_name="type")
        if event_type not in known_types:
            raise SpecError(f"unknown event type {event_type!r}", path=f"{base}.type", field="type")
        event["type"] = event_type

        at_ms = _get(event, "at_ms", ("monotonic_ms", "at_monotonic_ms"), None)
        if at_ms is not None:
            _nonnegative_int(at_ms, path=f"{base}.at_ms", field_name="at_ms")
            event["at_ms"] = at_ms

        if event_type in {"lease_apply", "lease_renew", "lease_release", "lease_expire"}:
            lease_id = _nonempty_str(_get(event, "lease_id", (), MISSING), path=base, field_name="lease_id")
            event["lease_id"] = lease_id
        if event_type == "lease_apply":
            shard_id = _nonempty_str(_get(event, "shard_id", (), MISSING), path=base, field_name="shard_id")
            node_id = _nonempty_str(_get(event, "node_id", (), MISSING), path=base, field_name="node_id")
            if shard_id not in shard_ids:
                raise SpecError(f"event references unknown shard {shard_id!r}", path=f"{base}.shard_id", field="shard_id")
            if node_id not in node_ids:
                raise SpecError(f"event references unknown node {node_id!r}", path=f"{base}.node_id", field="node_id")
            event["shard_id"] = shard_id
            event["node_id"] = node_id
            if "ttl_ms" in event:
                _positive_int(event["ttl_ms"], path=f"{base}.ttl_ms", field_name="ttl_ms")
            if "resource_units" in event:
                _positive_int(event["resource_units"], path=f"{base}.resource_units", field_name="resource_units")
        if event_type == "lease_renew":
            version = _get(event, "expected_version", ("lease_version", "version"), MISSING)
            if not is_int(version) or version <= 0:
                raise SpecError("expected_version must be a positive integer", path=f"{base}.expected_version", field="expected_version")
            event["expected_version"] = version
        if event_type == "quota_change":
            version = _positive_int(
                _get(event, "quota_version", ("capacity_version",), MISSING),
                path=f"{base}.quota_version",
                field_name="quota_version",
            )
            event["quota_version"] = version
            node_id = _get(event, "node_id", (), None)
            if node_id is not None:
                _nonempty_str(node_id, path=base, field_name="node_id")
                if node_id not in node_ids:
                    raise SpecError(f"quota_change references unknown node {node_id!r}", path=f"{base}.node_id", field="node_id")
            for field_name in ("concurrency_quota", "lease_limit", "global_budget_units"):
                if field_name in event and event[field_name] is not None:
                    _positive_int(event[field_name], path=f"{base}.{field_name}", field_name=field_name)
            revoke = event.get("revoke_lease_ids", [])
            if not isinstance(revoke, list) or not all(isinstance(x, str) and x for x in revoke):
                raise SpecError("revoke_lease_ids must be a list of strings", path=f"{base}.revoke_lease_ids", field="revoke_lease_ids")
        if event_type == "cancel_shard":
            shard_id = _nonempty_str(_get(event, "shard_id", (), MISSING), path=base, field_name="shard_id")
            if shard_id not in shard_ids:
                raise SpecError(f"cancel_shard references unknown shard {shard_id!r}", path=f"{base}.shard_id", field="shard_id")
        events.append(event)
    return tuple(events)


def topological_shards(spec: Spec | Iterable[ShardSpec]) -> list[ShardSpec]:
    shards = list(spec.shards if isinstance(spec, Spec) else spec)
    by_id = {shard.shard_id: shard for shard in shards}
    visited: set[str] = set()
    result: list[ShardSpec] = []

    def visit(shard: ShardSpec) -> None:
        if shard.shard_id in visited:
            return
        visited.add(shard.shard_id)
        for dep_id in sorted(shard.dependencies):
            visit(by_id[dep_id])
        result.append(shard)

    for shard in sorted(shards, key=lambda s: (s.priority, s.shard_id)):
        visit(shard)
    return result


def stable_dispatch_order(spec: Spec) -> list[ShardSpec]:
    """Topological order with deterministic priority choice among available nodes."""

    by_id = spec.shard_by_id
    remaining = set(by_id)
    completed: set[str] = set()
    result: list[ShardSpec] = []
    while remaining:
        available = [
            by_id[shard_id]
            for shard_id in remaining
            if all(dep in completed for dep in by_id[shard_id].dependencies)
        ]
        if not available:
            raise SpecError("dependency cycle detected while ordering dispatch", field="dependencies")
        chosen = min(available, key=lambda shard: (-shard.priority, shard.shard_id))
        result.append(chosen)
        completed.add(chosen.shard_id)
        remaining.remove(chosen.shard_id)
    return result
