from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import sha256_hex
from .errors import SpecError


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(obj: dict[str, Any], key: str, *, field_path: str) -> int:
    if key not in obj:
        raise SpecError(f"missing integer field {key}", field=f"{field_path}.{key}")
    value = obj[key]
    if not is_int(value):
        raise SpecError(
            f"field {key} must be an integer",
            field=f"{field_path}.{key}",
            details={"value_type": type(value).__name__},
        )
    return value


def require_str(obj: dict[str, Any], key: str, *, field_path: str) -> str:
    if key not in obj:
        raise SpecError(f"missing string field {key}", field=f"{field_path}.{key}")
    value = obj[key]
    if not isinstance(value, str) or not value:
        raise SpecError(
            f"field {key} must be a non-empty string",
            field=f"{field_path}.{key}",
            details={"value_type": type(value).__name__},
        )
    return value


def require_list(obj: dict[str, Any], key: str, *, field_path: str) -> list[Any]:
    if key not in obj:
        raise SpecError(f"missing list field {key}", field=f"{field_path}.{key}")
    value = obj[key]
    if not isinstance(value, list):
        raise SpecError(
            f"field {key} must be a list",
            field=f"{field_path}.{key}",
            details={"value_type": type(value).__name__},
        )
    return value


def load_json_file(path: str | Path, *, label: str) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read {label}: {path}", field=label) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"{label} is not valid JSON: {exc.msg}",
            field=f"{label}_json",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    concurrency_quota: int
    lease_limit: int
    seed: int
    capacity_version: int


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    node_id: str
    depends_on: tuple[str, ...]
    sample_rows: int
    resource_units: int
    priority: int
    planned_renewals: int = 0


@dataclass(frozen=True)
class ToleranceSpec:
    shard_q: int = 1
    node_watermark_q: int = 1
    global_budget_q: int = 1
    dominance_threshold_bp: int = 5000


@dataclass(frozen=True)
class AccountingSpec:
    pressure_den: int = 8
    group_count: int = 8


@dataclass(frozen=True)
class FaultSpec:
    port_conflict: bool = False
    timeout_shard_id: str | None = None
    timeout_delay_ms: int = 400
    kill_after_commits: int | None = None
    delay_ms_by_shard: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionSpec:
    schema_version: int
    session_id: str
    scale_power: int
    global_budget_units: int
    lease_ttl_ms: int
    lease_max_renewals: int
    socket_timeout_ms: int
    ready_timeout_ms: int
    worker_count: int
    tolerances: ToleranceSpec
    accounting: AccountingSpec
    fault_injection: FaultSpec
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    topo_order: tuple[str, ...]

    @property
    def q(self) -> int:
        return 10 ** self.scale_power

    @property
    def budget_q(self) -> int:
        return self.global_budget_units * self.q

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)

    def shards_for_node(self, node_id: str) -> tuple[ShardSpec, ...]:
        return tuple(shard for shard in self.shards if shard.node_id == node_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("topo_order", None)
        data["topo_order"] = list(self.topo_order)
        return data

    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "scale_power": self.scale_power,
            "global_budget_units": self.global_budget_units,
            "lease_ttl_ms": self.lease_ttl_ms,
            "lease_max_renewals": self.lease_max_renewals,
            "socket_timeout_ms": self.socket_timeout_ms,
            "ready_timeout_ms": self.ready_timeout_ms,
            "worker_count": self.worker_count,
            "tolerances": asdict(self.tolerances),
            "accounting": asdict(self.accounting),
            "nodes": [asdict(node) for node in self.nodes],
            "shards": [asdict(shard) for shard in self.shards],
        }
        return sha256_hex(payload)


def stable_topological_order(shards: Iterable[ShardSpec]) -> list[str]:
    shard_list = list(shards)
    by_id = {shard.shard_id: shard for shard in shard_list}
    missing = [
        dep
        for shard in shard_list
        for dep in shard.depends_on
        if dep not in by_id
    ]
    if missing:
        raise SpecError(
            "dependency reference does not exist",
            field="shards.depends_on",
            details={"missing": missing[0]},
        )

    dependents: dict[str, set[str]] = {shard.shard_id: set() for shard in shard_list}
    indegree: dict[str, int] = {shard.shard_id: 0 for shard in shard_list}
    for shard in shard_list:
        for dep in shard.depends_on:
            if shard.shard_id not in dependents[dep]:
                dependents[dep].add(shard.shard_id)
                indegree[shard.shard_id] += 1

    ready = [sid for sid, degree in indegree.items() if degree == 0]
    order: list[str] = []
    while ready:
        ready.sort(key=lambda sid: (-by_id[sid].priority, sid))
        current = ready.pop(0)
        order.append(current)
        for dependent in sorted(dependents[current]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(order) != len(shard_list):
        cycle = sorted(sid for sid, degree in indegree.items() if degree > 0)
        raise SpecError(
            "dependency graph contains a cycle",
            field="shards.depends_on",
            details={"cycle_candidates": cycle},
        )
    return order


def _parse_tolerances(raw: Any, path: str) -> ToleranceSpec:
    if raw is None:
        return ToleranceSpec()
    if not isinstance(raw, dict):
        raise SpecError("tolerances must be an object", field=path)
    values: dict[str, int] = {}
    mapping = {
        "shard_q": "shard_q",
        "node_watermark_q": "node_watermark_q",
        "global_budget_q": "global_budget_q",
        "dominance_threshold_bp": "dominance_threshold_bp",
    }
    for json_key, attr in mapping.items():
        full = f"{path}.{json_key}"
        if json_key not in raw:
            values[attr] = getattr(ToleranceSpec(), attr)
            continue
        value = require_int(raw, json_key, field_path=path)
        if json_key == "dominance_threshold_bp":
            if not 1 <= value <= 10000:
                raise SpecError("dominance threshold must be in [1,10000]", field=full)
        elif value < 0:
            raise SpecError("tolerance must be a non-negative integer", field=full)
        values[attr] = value
    return ToleranceSpec(**values)


def _parse_accounting(raw: Any, path: str) -> AccountingSpec:
    if raw is None:
        return AccountingSpec()
    if not isinstance(raw, dict):
        raise SpecError("accounting must be an object", field=path)
    pressure_den = raw.get("pressure_den", 8)
    group_count = raw.get("group_count", 8)
    for key, value in (("pressure_den", pressure_den), ("group_count", group_count)):
        full = f"{path}.{key}"
        if not is_int(value) or value <= 0:
            raise SpecError(f"{key} must be a positive integer", field=full)
    return AccountingSpec(pressure_den=pressure_den, group_count=group_count)  # type: ignore[arg-type]


def _parse_fault(raw: Any, path: str) -> FaultSpec:
    if raw is None:
        return FaultSpec()
    if not isinstance(raw, dict):
        raise SpecError("fault_injection must be an object", field=path)
    result = FaultSpec(
        port_conflict=bool(raw.get("port_conflict", False)),
        timeout_shard_id=raw.get("timeout_shard_id"),
        timeout_delay_ms=raw.get("timeout_delay_ms", 400),
        kill_after_commits=raw.get("kill_after_commits"),
        delay_ms_by_shard=dict(raw.get("delay_ms_by_shard", {})),
    )
    if not isinstance(result.port_conflict, bool):
        raise SpecError("port_conflict must be boolean", field=f"{path}.port_conflict")
    if result.timeout_shard_id is not None and not isinstance(result.timeout_shard_id, str):
        raise SpecError("timeout_shard_id must be string or null", field=f"{path}.timeout_shard_id")
    if not is_int(result.timeout_delay_ms) or result.timeout_delay_ms <= 0:
        raise SpecError("timeout_delay_ms must be positive integer", field=f"{path}.timeout_delay_ms")
    if result.kill_after_commits is not None and (
        not is_int(result.kill_after_commits) or result.kill_after_commits <= 0
    ):
        raise SpecError("kill_after_commits must be positive integer or null", field=f"{path}.kill_after_commits")
    if not isinstance(raw.get("delay_ms_by_shard", {}), dict):
        raise SpecError("delay_ms_by_shard must be object", field=f"{path}.delay_ms_by_shard")
    for shard_id, delay in result.delay_ms_by_shard.items():
        if not isinstance(shard_id, str) or not is_int(delay) or delay < 0:
            raise SpecError(
                "delay_ms_by_shard must map shard ids to non-negative integers",
                field=f"{path}.delay_ms_by_shard.{shard_id}",
            )
    return result


def parse_session(
    declaration: dict[str, Any],
    shards_doc: dict[str, Any],
    *,
    capacity_doc: dict[str, Any] | None = None,
) -> SessionSpec:
    if not isinstance(declaration, dict):
        raise SpecError("declaration root must be an object", field="declaration")
    if not isinstance(shards_doc, dict):
        raise SpecError("shards document root must be an object", field="shards")

    schema_version = require_int(declaration, "schema_version", field_path="declaration")
    if schema_version != 1:
        raise SpecError("unsupported schema_version", field="declaration.schema_version")

    session_id = require_str(declaration, "session_id", field_path="declaration")
    scale_power = require_int(declaration, "scale_power", field_path="declaration")
    if not 0 <= scale_power <= 3:
        raise SpecError("scale_power must be in [0,3]", field="declaration.scale_power")

    global_budget_units = require_int(declaration, "global_budget_units", field_path="declaration")
    if global_budget_units <= 0:
        raise SpecError("global_budget_units must be a positive integer", field="declaration.global_budget_units")

    lease_ttl_ms = require_int(declaration, "lease_ttl_ms", field_path="declaration")
    if lease_ttl_ms <= 0:
        raise SpecError("lease_ttl_ms must be positive integer", field="declaration.lease_ttl_ms")

    lease_max_renewals = require_int(declaration, "lease_max_renewals", field_path="declaration")
    if lease_max_renewals < 0:
        raise SpecError("lease_max_renewals must be non-negative integer", field="declaration.lease_max_renewals")

    socket_timeout_ms = declaration.get("socket_timeout_ms", 1000)
    if not is_int(socket_timeout_ms) or socket_timeout_ms <= 0:
        raise SpecError("socket_timeout_ms must be positive integer", field="declaration.socket_timeout_ms")

    ready_timeout_ms = declaration.get("ready_timeout_ms", 5000)
    if not is_int(ready_timeout_ms) or ready_timeout_ms <= 0:
        raise SpecError("ready_timeout_ms must be positive integer", field="declaration.ready_timeout_ms")

    worker_count = declaration.get("worker_count", 8)
    if not is_int(worker_count) or worker_count <= 0:
        raise SpecError("worker_count must be positive integer", field="declaration.worker_count")

    tolerance_raw = declaration.get("tolerances_q", declaration.get("tolerances"))
    tolerance_path = "declaration.tolerances_q" if "tolerances_q" in declaration else "declaration.tolerances"
    tolerances = _parse_tolerances(tolerance_raw, tolerance_path)
    accounting = _parse_accounting(declaration.get("accounting"), "declaration.accounting")
    faults = _parse_fault(declaration.get("fault_injection"), "declaration.fault_injection")

    raw_nodes = require_list(declaration, "nodes", field_path="declaration")
    if not raw_nodes:
        raise SpecError("at least one node is required", field="declaration.nodes")

    nodes: list[NodeSpec] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw_node in enumerate(raw_nodes):
        base = f"declaration.nodes[{index}]"
        if not isinstance(raw_node, dict):
            raise SpecError("node must be an object", field=base)
        node_id = require_str(raw_node, "node_id", field_path=base)
        id_field = f"{base}.node_id"
        if node_id in seen_node_ids:
            raise SpecError("duplicate node_id", field=id_field, details={"value": node_id})
        seen_node_ids.add(node_id)

        host = require_str(raw_node, "host", field_path=base)
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", field=f"{base}.host", details={"value": host})

        port = require_int(raw_node, "port", field_path=base)
        if not 1 <= port <= 65535:
            raise SpecError("port must be in [1,65535]", field=f"{base}.port", details={"value": port})
        if port in seen_ports:
            raise SpecError("duplicate loopback port", field=f"{base}.port", details={"value": port})
        seen_ports.add(port)

        concurrency_quota = require_int(raw_node, "concurrency_quota", field_path=base)
        if concurrency_quota < 0:
            raise SpecError("concurrency_quota must be non-negative integer", field=f"{base}.concurrency_quota")

        lease_limit = require_int(raw_node, "lease_limit", field_path=base)
        if lease_limit < 0:
            raise SpecError("lease_limit must be non-negative integer", field=f"{base}.lease_limit")

        seed = require_int(raw_node, "seed", field_path=base)
        if not 0 <= seed <= 2**32 - 1:
            raise SpecError("seed must be an integer in [0,2^32-1]", field=f"{base}.seed")

        capacity_version = require_int(raw_node, "capacity_version", field_path=base)
        if capacity_version < 1:
            raise SpecError("capacity_version must be a positive integer", field=f"{base}.capacity_version")

        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=port,
                concurrency_quota=concurrency_quota,
                lease_limit=lease_limit,
                seed=seed,
                capacity_version=capacity_version,
            )
        )

    if capacity_doc is not None:
        _validate_capacity_doc(capacity_doc, nodes)

    raw_shards = require_list(shards_doc, "shards", field_path="shards")
    if not raw_shards:
        raise SpecError("at least one shard is required", field="shards.shards")

    shards: list[ShardSpec] = []
    seen_shard_ids: set[str] = set()
    node_ids = seen_node_ids
    for index, raw_shard in enumerate(raw_shards):
        base = f"shards.shards[{index}]"
        if not isinstance(raw_shard, dict):
            raise SpecError("shard must be an object", field=base)
        shard_id = require_str(raw_shard, "shard_id", field_path=base)
        if shard_id in seen_shard_ids:
            raise SpecError("duplicate shard_id", field=f"{base}.shard_id", details={"value": shard_id})
        seen_shard_ids.add(shard_id)

        node_id = require_str(raw_shard, "node_id", field_path=base)
        if node_id not in node_ids:
            raise SpecError("shard node_id does not reference a declared node", field=f"{base}.node_id")

        raw_deps = require_list(raw_shard, "depends_on", field_path=base)
        deps: list[str] = []
        for dep_index, dep in enumerate(raw_deps):
            dep_field = f"{base}.depends_on[{dep_index}]"
            if not isinstance(dep, str) or not dep:
                raise SpecError("dependency reference must be a non-empty string", field=dep_field)
            if dep == shard_id:
                raise SpecError("shard cannot depend on itself", field=dep_field)
            deps.append(dep)

        sample_rows = require_int(raw_shard, "sample_rows", field_path=base)
        if sample_rows <= 0 or sample_rows > 1_000_000:
            raise SpecError("sample_rows must be an integer in [1,1000000]", field=f"{base}.sample_rows")

        resource_units = require_int(raw_shard, "resource_units", field_path=base)
        if resource_units <= 0:
            raise SpecError("resource_units must be a positive integer", field=f"{base}.resource_units")

        priority = require_int(raw_shard, "priority", field_path=base)
        planned_renewals = raw_shard.get("planned_renewals", 0)
        if not is_int(planned_renewals) or planned_renewals < 0:
            raise SpecError("planned_renewals must be non-negative integer", field=f"{base}.planned_renewals")
        if planned_renewals > lease_max_renewals:
            raise SpecError(
                "planned_renewals exceeds global lease_max_renewals",
                field=f"{base}.planned_renewals",
            )

        shards.append(
            ShardSpec(
                shard_id=shard_id,
                node_id=node_id,
                depends_on=tuple(deps),
                sample_rows=sample_rows,
                resource_units=resource_units,
                priority=priority,
                planned_renewals=planned_renewals,
            )
        )

    topo_order = stable_topological_order(shards)
    known_shards = set(seen_shard_ids)
    for shard in shards:
        for dep in shard.depends_on:
            if dep not in known_shards:
                raise SpecError(
                    "dependency reference does not exist",
                    field=f"shards.shards[?shard_id={shard.shard_id}].depends_on",
                    details={"missing": dep},
                )

    timeout_shard = faults.timeout_shard_id
    if timeout_shard is not None and timeout_shard not in seen_shard_ids:
        raise SpecError(
            "fault_injection.timeout_shard_id does not reference a shard",
            field="declaration.fault_injection.timeout_shard_id",
        )
    for shard_id in faults.delay_ms_by_shard:
        if shard_id not in seen_shard_ids:
            raise SpecError(
                "fault_injection.delay_ms_by_shard references unknown shard",
                field=f"declaration.fault_injection.delay_ms_by_shard.{shard_id}",
            )

    return SessionSpec(
        schema_version=schema_version,
        session_id=session_id,
        scale_power=scale_power,
        global_budget_units=global_budget_units,
        lease_ttl_ms=lease_ttl_ms,
        lease_max_renewals=lease_max_renewals,
        socket_timeout_ms=socket_timeout_ms,
        ready_timeout_ms=ready_timeout_ms,
        worker_count=worker_count,
        tolerances=tolerances,
        accounting=accounting,
        fault_injection=faults,
        nodes=tuple(nodes),
        shards=tuple(shards),
        topo_order=tuple(topo_order),
    )


def _validate_capacity_doc(capacity_doc: dict[str, Any], nodes: list[NodeSpec]) -> None:
    if not isinstance(capacity_doc, dict):
        raise SpecError("capacity document root must be object", field="capacity")
    versions = require_list(capacity_doc, "capacity_versions", field_path="capacity")
    by_node = {item.node_id: item.capacity_version for item in nodes}
    seen: set[str] = set()
    for index, item in enumerate(versions):
        base = f"capacity.capacity_versions[{index}]"
        if not isinstance(item, dict):
            raise SpecError("capacity version entry must be object", field=base)
        node_id = require_str(item, "node_id", field_path=base)
        capacity_version = require_int(item, "capacity_version", field_path=base)
        if node_id not in by_node:
            raise SpecError("capacity entry references unknown node", field=f"{base}.node_id")
        if node_id in seen:
            raise SpecError("duplicate capacity entry", field=f"{base}.node_id")
        seen.add(node_id)
        if capacity_version != by_node[node_id]:
            raise SpecError(
                "capacity version does not match node declaration",
                field=f"{base}.capacity_version",
                details={"declared": by_node[node_id], "capacity_doc": capacity_version},
            )
    missing = set(by_node) - seen
    if missing:
        first = sorted(missing)[0]
        raise SpecError(
            "capacity document is missing a node version",
            field="capacity.capacity_versions",
            details={"missing_node_id": first},
        )


def load_session_files(
    declaration_path: str | Path,
    shards_path: str | Path,
    *,
    capacity_path: str | Path | None = None,
) -> SessionSpec:
    declaration = load_json_file(declaration_path, label="declaration")
    shards_doc = load_json_file(shards_path, label="shards")
    capacity_doc = load_json_file(capacity_path, label="capacity") if capacity_path else None
    return parse_session(declaration, shards_doc, capacity_doc=capacity_doc)
