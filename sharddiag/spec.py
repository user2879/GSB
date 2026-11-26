"""Loading and strict validation of local shard diagnostic specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_digest
from .errors import SpecError


EVENT_TYPES = {
    "shard_arrived",
    "cancel",
    "fail_shard",
    "renew_lease",
    "advance_capacity",
}


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fail(message: str, *, field_name: str, location: str, **details: object) -> None:
    raise SpecError(message, field=field_name, location=location, **details)


def _require_dict(value: Any, *, field_name: str, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("must be a JSON object", field_name=field_name, location=location)
    return value


def _require_list(value: Any, *, field_name: str, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("must be a JSON array", field_name=field_name, location=location)
    return value


def _require_int(
    value: Any,
    *,
    field_name: str,
    location: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not is_int(value):
        _fail("must be an integer", field_name=field_name, location=location)
    assert isinstance(value, int)
    if minimum is not None and value < minimum:
        _fail(f"must be >= {minimum}", field_name=field_name, location=location)
    if maximum is not None and value > maximum:
        _fail(f"must be <= {maximum}", field_name=field_name, location=location)
    return value


def _require_nonempty_str(value: Any, *, field_name: str, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("must be a non-empty string", field_name=field_name, location=location)
    return value


def _no_extra_keys(
    value: dict[str, Any], *, allowed: set[str], field_name: str, location: str
) -> None:
    extra = next((key for key in value if key not in allowed), None)
    if extra is not None:
        child_location = f"{location}.{extra}" if location != "$" else f"$.{extra}"
        _fail(
            f"unknown field {extra!r}",
            field_name=f"{field_name}.{extra}",
            location=child_location,
        )


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    source_path: str | None
    nodes: dict[str, dict[str, Any]]
    shards: dict[str, dict[str, Any]]
    capacity_versions: dict[int, dict[str, Any]]
    event_list: list[dict[str, Any]]
    topo_order: list[str]
    topo_rank: dict[str, int]
    spec_digest: str
    shard_index: dict[str, int] = field(default_factory=dict)
    node_index: dict[str, int] = field(default_factory=dict)

    @property
    def initial_capacity_version(self) -> int:
        return min(self.capacity_versions)

    @property
    def current_capacity_version(self) -> int:
        return max(self.capacity_versions)

    def version_chain(self) -> list[int]:
        return sorted(self.capacity_versions)

    def descendants(self, root_ids: Iterable[str]) -> set[str]:
        """Return transitive shard descendants, including each root."""

        reverse: dict[str, list[str]] = {shard_id: [] for shard_id in self.shards}
        for shard_id, shard in self.shards.items():
            for upstream in shard["depends_on"]:
                reverse[upstream].append(shard_id)

        pending = list(dict.fromkeys(root_ids))
        seen: set[str] = set()
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            pending.extend(reverse.get(current, []))
        return seen


def stable_topological_order(shards: dict[str, dict[str, Any]]) -> list[str]:
    """Kahn order with lexicographic choice among currently ready shards."""

    indegree = {shard_id: len(shard["depends_on"]) for shard_id, shard in shards.items()}
    children: dict[str, list[str]] = {shard_id: [] for shard_id in shards}
    for shard_id, shard in shards.items():
        for upstream in shard["depends_on"]:
            children[upstream].append(shard_id)

    ready = sorted(shard_id for shard_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        newly_ready: list[str] = []
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                newly_ready.append(child)
        ready.extend(newly_ready)
        ready.sort()

    if len(order) != len(shards):
        remaining = sorted(shard_id for shard_id, degree in indegree.items() if degree > 0)
        first = remaining[0]
        raise SpecError(
            "dependency graph contains a cycle",
            field="shards.depends_on",
            location=f"$.shards[{shards[first].get('_index', '?')}].depends_on",
            cycle_shards=remaining,
        )
    return order


def validate_spec(data: Any, *, source_path: str | None = None) -> Spec:
    root = _require_dict(data, field_name="$", location="$")
    _no_extra_keys(
        root,
        allowed={
            "format_version",
            "session_scale",
            "global_budget_units",
            "max_renewals",
            "lease_ttl_ms",
            "handshake_timeout_ms",
            "io_timeout_ms",
            "startup_timeout_ms",
            "batch_rows",
            "nodes",
            "shards",
            "capacity_versions",
            "events",
        },
        field_name="$",
        location="$",
    )

    format_version = _require_int(
        root.get("format_version"), field_name="format_version", location="$.format_version", minimum=1
    )
    if format_version != 1:
        _fail("integer format_version must be 1 in this build", field_name="format_version", location="$.format_version")

    session_scale = _require_int(
        root.get("session_scale"), field_name="session_scale", location="$.session_scale", minimum=0, maximum=3
    )
    global_budget = _require_int(
        root.get("global_budget_units"),
        field_name="global_budget_units",
        location="$.global_budget_units",
        minimum=1,
    )
    max_renewals = _require_int(
        root.get("max_renewals"), field_name="max_renewals", location="$.max_renewals", minimum=0
    )
    lease_ttl = _require_int(
        root.get("lease_ttl_ms"), field_name="lease_ttl_ms", location="$.lease_ttl_ms", minimum=1
    )
    handshake_timeout = _require_int(
        root.get("handshake_timeout_ms"),
        field_name="handshake_timeout_ms",
        location="$.handshake_timeout_ms",
        minimum=1,
    )
    io_timeout = _require_int(
        root.get("io_timeout_ms"), field_name="io_timeout_ms", location="$.io_timeout_ms", minimum=1
    )
    startup_timeout = root.get("startup_timeout_ms", 5000)
    _require_int(
        startup_timeout, field_name="startup_timeout_ms", location="$.startup_timeout_ms", minimum=1
    )
    batch_rows = root.get("batch_rows", 16)
    _require_int(batch_rows, field_name="batch_rows", location="$.batch_rows", minimum=1)

    raw_versions = _require_list(
        root.get("capacity_versions"), field_name="capacity_versions", location="$.capacity_versions"
    )
    if not raw_versions:
        _fail("at least one capacity version is required", field_name="capacity_versions", location="$.capacity_versions")

    raw_nodes = _require_list(root.get("nodes"), field_name="nodes", location="$.nodes")
    if not raw_nodes:
        _fail("at least one node is required", field_name="nodes", location="$.nodes")

    nodes: dict[str, dict[str, Any]] = {}
    node_index: dict[str, int] = {}
    seen_ports: dict[int, str] = {}
    for index, raw_node in enumerate(raw_nodes):
        location = f"$.nodes[{index}]"
        node = _require_dict(raw_node, field_name="nodes", location=location)
        _no_extra_keys(
            node,
            allowed={
                "node_id",
                "host",
                "port",
                "concurrency_quota",
                "lease_limit",
                "seed",
                "capacity_version",
                "fixed_point_decimals",
            },
            field_name="nodes",
            location=location,
        )
        node_id = _require_nonempty_str(
            node.get("node_id"), field_name="nodes.node_id", location=f"{location}.node_id"
        )
        if node_id in nodes:
            _fail(
                f"duplicate node_id {node_id!r}",
                field_name="nodes.node_id",
                location=f"{location}.node_id",
            )
        node_index[node_id] = index

        host = _require_nonempty_str(node.get("host"), field_name="nodes.host", location=f"{location}.host")
        if host != "127.0.0.1":
            _fail(
                "host must be the literal 127.0.0.1",
                field_name="nodes.host",
                location=f"{location}.host",
                actual=host,
            )

        port = _require_int(
            node.get("port"), field_name="nodes.port", location=f"{location}.port", minimum=1, maximum=65535
        )
        if port in seen_ports:
            _fail(
                f"port {port} is already declared by node {seen_ports[port]!r}",
                field_name="nodes.port",
                location=f"{location}.port",
            )
        seen_ports[port] = node_id

        concurrency = _require_int(
            node.get("concurrency_quota"),
            field_name="nodes.concurrency_quota",
            location=f"{location}.concurrency_quota",
            minimum=0,
        )
        lease_limit = _require_int(
            node.get("lease_limit"),
            field_name="nodes.lease_limit",
            location=f"{location}.lease_limit",
            minimum=0,
        )
        seed = _require_int(
            node.get("seed"), field_name="nodes.seed", location=f"{location}.seed", minimum=0, maximum=2**64 - 1
        )
        capacity_version = _require_int(
            node.get("capacity_version"),
            field_name="nodes.capacity_version",
            location=f"{location}.capacity_version",
            minimum=1,
        )
        fixed_decimals = _require_int(
            node.get("fixed_point_decimals"),
            field_name="nodes.fixed_point_decimals",
            location=f"{location}.fixed_point_decimals",
            minimum=0,
            maximum=9,
        )

        nodes[node_id] = {
            **node,
            "node_id": node_id,
            "host": host,
            "port": port,
            "concurrency_quota": concurrency,
            "lease_limit": lease_limit,
            "seed": seed,
            "capacity_version": capacity_version,
            "fixed_point_decimals": fixed_decimals,
            "_index": index,
        }

    versions: dict[int, dict[str, Any]] = {}
    previous_version: int | None = None
    for index, raw_version in enumerate(raw_versions):
        location = f"$.capacity_versions[{index}]"
        version_entry = _require_dict(raw_version, field_name="capacity_versions", location=location)
        _no_extra_keys(
            version_entry,
            allowed={
                "version",
                "parent",
                "session_scale",
                "global_budget_units",
                "note",
                "node_calibers",
            },
            field_name="capacity_versions",
            location=location,
        )
        version = _require_int(
            version_entry.get("version"),
            field_name="capacity_versions.version",
            location=f"{location}.version",
            minimum=1,
        )
        if version in versions:
            _fail(
                f"duplicate capacity version {version}",
                field_name="capacity_versions.version",
                location=f"{location}.version",
            )
        if previous_version is not None and version <= previous_version:
            _fail(
                "capacity versions must be strictly increasing",
                field_name="capacity_versions.version",
                location=f"{location}.version",
            )

        parent_location = f"{location}.parent"
        parent = version_entry.get("parent")
        if index == 0:
            if parent is not None:
                _fail(
                    "the first capacity version must explicitly set parent to null",
                    field_name="capacity_versions.parent",
                    location=parent_location,
                )
        else:
            if not is_int(parent):
                _fail(
                    "non-initial capacity version must explicitly reference an integer parent",
                    field_name="capacity_versions.parent",
                    location=parent_location,
                )
            assert is_int(parent)
            if parent != previous_version:
                _fail(
                    f"parent must be the immediately preceding version {previous_version}",
                    field_name="capacity_versions.parent",
                    location=parent_location,
                    actual=parent,
                )

        version_scale = _require_int(
            version_entry.get("session_scale"),
            field_name="capacity_versions.session_scale",
            location=f"{location}.session_scale",
            minimum=0,
            maximum=3,
        )
        version_budget = _require_int(
            version_entry.get("global_budget_units"),
            field_name="capacity_versions.global_budget_units",
            location=f"{location}.global_budget_units",
            minimum=1,
        )
        calibers_raw = _require_dict(
            version_entry.get("node_calibers"),
            field_name="capacity_versions.node_calibers",
            location=f"{location}.node_calibers",
        )
        if set(calibers_raw) != set(nodes):
            _fail(
                "node_calibers must reference exactly the declared node ids",
                field_name="capacity_versions.node_calibers",
                location=f"{location}.node_calibers",
                missing=sorted(set(nodes) - set(calibers_raw)),
                unknown=sorted(set(calibers_raw) - set(nodes)),
            )

        calibers: dict[str, dict[str, int]] = {}
        for node_id in sorted(nodes):
            caliber_location = f"{location}.node_calibers.{node_id}"
            caliber = _require_dict(
                calibers_raw.get(node_id),
                field_name="capacity_versions.node_calibers",
                location=caliber_location,
            )
            _no_extra_keys(
                caliber,
                allowed={"concurrency_quota", "lease_limit", "fixed_point_decimals"},
                field_name="capacity_versions.node_calibers",
                location=caliber_location,
            )
            concurrency = _require_int(
                caliber.get("concurrency_quota"),
                field_name="capacity_versions.node_calibers.concurrency_quota",
                location=f"{caliber_location}.concurrency_quota",
                minimum=0,
            )
            lease_limit = _require_int(
                caliber.get("lease_limit"),
                field_name="capacity_versions.node_calibers.lease_limit",
                location=f"{caliber_location}.lease_limit",
                minimum=0,
            )
            decimals = _require_int(
                caliber.get("fixed_point_decimals"),
                field_name="capacity_versions.node_calibers.fixed_point_decimals",
                location=f"{caliber_location}.fixed_point_decimals",
                minimum=0,
                maximum=9,
            )
            calibers[node_id] = {
                "concurrency_quota": concurrency,
                "lease_limit": lease_limit,
                "fixed_point_decimals": decimals,
            }

        versions[version] = {
            **version_entry,
            "version": version,
            "parent": parent,
            "session_scale": version_scale,
            "global_budget_units": version_budget,
            "node_calibers": calibers,
        }
        previous_version = version

    first_version = min(versions)
    first_entry = versions[first_version]
    if first_entry["session_scale"] != session_scale:
        _fail(
            "session_scale must equal capacity version 1",
            field_name="session_scale",
            location="$.session_scale",
        )
    if first_entry["global_budget_units"] != global_budget:
        _fail(
            "global_budget_units must equal capacity version 1",
            field_name="global_budget_units",
            location="$.global_budget_units",
        )
    for node_id, node in nodes.items():
        caliber = first_entry["node_calibers"][node_id]
        for key in ("concurrency_quota", "lease_limit", "fixed_point_decimals"):
            if node[key] != caliber[key]:
                _fail(
                    f"node {node_id!r} {key} must equal capacity version {first_version}",
                    field_name=f"nodes.{key}",
                    location=f"$.nodes[{node['_index']}].{key}",
                )
        if node["capacity_version"] not in versions:
            _fail(
                f"capacity_version {node['capacity_version']} is absent from capacity_versions",
                field_name="nodes.capacity_version",
                location=f"$.nodes[{node['_index']}].capacity_version",
            )

    raw_shards = _require_list(root.get("shards"), field_name="shards", location="$.shards")
    if not raw_shards:
        _fail("at least one shard is required", field_name="shards", location="$.shards")

    shards: dict[str, dict[str, Any]] = {}
    shard_index: dict[str, int] = {}
    for index, raw_shard in enumerate(raw_shards):
        location = f"$.shards[{index}]"
        shard = _require_dict(raw_shard, field_name="shards", location=location)
        _no_extra_keys(
            shard,
            allowed={
                "shard_id",
                "node_id",
                "depends_on",
                "sample_rows",
                "resource_units",
                "priority",
                "batch_rows",
            },
            field_name="shards",
            location=location,
        )
        shard_id = _require_nonempty_str(
            shard.get("shard_id"), field_name="shards.shard_id", location=f"{location}.shard_id"
        )
        if shard_id in shards:
            _fail(
                f"duplicate shard_id {shard_id!r}",
                field_name="shards.shard_id",
                location=f"{location}.shard_id",
            )
        shard_index[shard_id] = index

        node_id = _require_nonempty_str(
            shard.get("node_id"), field_name="shards.node_id", location=f"{location}.node_id"
        )
        if node_id not in nodes:
            _fail(
                f"node_id {node_id!r} does not exist",
                field_name="shards.node_id",
                location=f"{location}.node_id",
            )

        depends_raw = shard.get("depends_on")
        depends = _require_list(
            depends_raw, field_name="shards.depends_on", location=f"{location}.depends_on"
        )
        normalized_depends: list[str] = []
        for dep_index, dependency in enumerate(depends):
            dep_location = f"{location}.depends_on[{dep_index}]"
            dependency_id = _require_nonempty_str(
                dependency, field_name="shards.depends_on", location=dep_location
            )
            if dependency_id == shard_id:
                _fail(
                    "a shard must not depend on itself",
                    field_name="shards.depends_on",
                    location=dep_location,
                )
            normalized_depends.append(dependency_id)

        rows = _require_int(
            shard.get("sample_rows"),
            field_name="shards.sample_rows",
            location=f"{location}.sample_rows",
            minimum=1,
        )
        units = _require_int(
            shard.get("resource_units"),
            field_name="shards.resource_units",
            location=f"{location}.resource_units",
            minimum=1,
        )
        priority = _require_int(
            shard.get("priority"), field_name="shards.priority", location=f"{location}.priority"
        )
        shard_batch_rows = shard.get("batch_rows", batch_rows)
        _require_int(
            shard_batch_rows, field_name="shards.batch_rows", location=f"{location}.batch_rows", minimum=1
        )
        if shard_batch_rows is not None and is_int(shard_batch_rows) and shard_batch_rows > rows:
            _fail(
                "batch_rows must not exceed sample_rows",
                field_name="shards.batch_rows",
                location=f"{location}.batch_rows",
            )

        shards[shard_id] = {
            **shard,
            "shard_id": shard_id,
            "node_id": node_id,
            "depends_on": normalized_depends,
            "sample_rows": rows,
            "resource_units": units,
            "priority": priority,
            "batch_rows": shard_batch_rows,
            "_index": index,
        }

    for shard_id, shard in shards.items():
        for dep_index, dependency_id in enumerate(shard["depends_on"]):
            if dependency_id not in shards:
                index = shard_index[shard_id]
                _fail(
                    f"dependency {dependency_id!r} does not exist",
                    field_name="shards.depends_on",
                    location=f"$.shards[{index}].depends_on[{dep_index}]",
                )

    topo_order = stable_topological_order(shards)
    topo_rank = {shard_id: index for index, shard_id in enumerate(topo_order)}

    raw_events = _require_list(root.get("events"), field_name="events", location="$.events")
    events: list[dict[str, Any]] = []
    previous_seq: int | None = None
    seen_seq: set[int] = set()
    for index, raw_event in enumerate(raw_events):
        location = f"$.events[{index}]"
        event = _require_dict(raw_event, field_name="events", location=location)
        seq = _require_int(event.get("seq"), field_name="events.seq", location=f"{location}.seq", minimum=1)
        if seq in seen_seq:
            _fail(f"duplicate event seq {seq}", field_name="events.seq", location=f"{location}.seq")
        if previous_seq is not None and seq <= previous_seq:
            _fail(
                "events must be ordered by strictly increasing arrival seq",
                field_name="events.seq",
                location=f"{location}.seq",
            )
        seen_seq.add(seq)
        previous_seq = seq

        event_type = _require_nonempty_str(event.get("type"), field_name="events.type", location=f"{location}.type")
        if event_type not in EVENT_TYPES:
            _fail(
                f"unknown event type {event_type!r}",
                field_name="events.type",
                location=f"{location}.type",
                allowed=sorted(EVENT_TYPES),
            )
        event_allowed = {"seq", "type"}
        if event_type in {"shard_arrived", "cancel", "fail_shard"}:
            event_allowed.update({"shard_id", "reason"})
        elif event_type == "renew_lease":
            event_allowed.update({"shard_id", "expected_version"})
        elif event_type == "advance_capacity":
            event_allowed.update({"to_version", "node_id"})
        _no_extra_keys(event, allowed=event_allowed, field_name="events", location=location)

        normalized = {**event, "seq": seq, "type": event_type, "_index": index}
        if event_type in {"shard_arrived", "cancel", "fail_shard", "renew_lease"}:
            shard_id = _require_nonempty_str(
                event.get("shard_id"), field_name="events.shard_id", location=f"{location}.shard_id"
            )
            if shard_id not in shards:
                _fail(
                    f"event references unknown shard {shard_id!r}",
                    field_name="events.shard_id",
                    location=f"{location}.shard_id",
                )
            normalized["shard_id"] = shard_id
        if event_type == "renew_lease":
            expected_version = _require_int(
                event.get("expected_version"),
                field_name="events.expected_version",
                location=f"{location}.expected_version",
                minimum=1,
            )
            normalized["expected_version"] = expected_version
        if event_type == "advance_capacity":
            to_version = _require_int(
                event.get("to_version"),
                field_name="events.to_version",
                location=f"{location}.to_version",
                minimum=1,
            )
            if to_version not in versions:
                _fail(
                    f"event references unknown capacity version {to_version}",
                    field_name="events.to_version",
                    location=f"{location}.to_version",
                )
            normalized["to_version"] = to_version
            if "node_id" in event and event["node_id"] is not None:
                event_node = _require_nonempty_str(
                    event.get("node_id"), field_name="events.node_id", location=f"{location}.node_id"
                )
                if event_node not in nodes:
                    _fail(
                        f"event references unknown node {event_node!r}",
                        field_name="events.node_id",
                        location=f"{location}.node_id",
                    )
                normalized["node_id"] = event_node
        events.append(normalized)

    spec_digest = canonical_digest(
        {key: value for key, value in root.items() if key not in {"_runtime"}}
    )
    return Spec(
        raw=root,
        source_path=source_path,
        nodes=nodes,
        shards=shards,
        capacity_versions=versions,
        event_list=events,
        topo_order=topo_order,
        topo_rank=topo_rank,
        spec_digest=spec_digest,
        shard_index=shard_index,
        node_index=node_index,
    )


def load_spec(path: str | Path) -> Spec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(
            f"cannot read specification: {exc}",
            field="$",
            location=str(source),
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            field="$",
            location=f"{source}:{exc.lineno}:{exc.colno}",
        ) from exc
    return validate_spec(data, source_path=str(source))
