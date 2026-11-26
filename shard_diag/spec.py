from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .errors import SpecError
from .models import DelayBatch, NodeSpec, ShardSpec, Spec
from .topology import stable_topological_order


TOP_LEVEL_FIELDS = {
    "format_version",
    "session_id",
    "scale_power",
    "global_resource_budget",
    "lease_ttl_ms",
    "max_revision_lag",
    "dispatch_delay_limit_ms",
    "revision_chain",
    "nodes",
    "shards",
    "delay_batches",
}
NODE_FIELDS = {
    "node_id",
    "host",
    "port",
    "concurrency_quota",
    "lease_limit",
    "seed",
    "capacity_version",
}
SHARD_FIELDS = {
    "shard_id",
    "node_id",
    "dependencies",
    "sample_rows",
    "resource_units",
    "priority",
}
BATCH_FIELDS = {
    "batch_id",
    "node_id",
    "primary_revision",
    "replayed_revision",
    "delay_samples_ms",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bad(pointer: str, field: str, message: str, **details: Any) -> None:
    raise SpecError(message, object=pointer, field=field, details=details or None)


def _require_dict(value: Any, pointer: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _bad(pointer, "$", "value must be a JSON object")
    return value


def _check_unknown(obj: dict[str, Any], allowed: set[str], pointer: str) -> None:
    for key in obj:
        if key not in allowed:
            _bad(pointer, key, f"unknown field {key!r}")


def _nonempty_str(obj: dict[str, Any], field: str, pointer: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value:
        _bad(pointer, field, f"{field} must be a non-empty string")
    return value


def _positive_int(obj: dict[str, Any], field: str, pointer: str) -> int:
    value = obj.get(field)
    if not _is_int(value) or value <= 0:
        _bad(pointer, field, f"{field} must be a positive integer")
    return value


def _nonnegative_int(obj: dict[str, Any], field: str, pointer: str) -> int:
    value = obj.get(field)
    if not _is_int(value) or value < 0:
        _bad(pointer, field, f"{field} must be a non-negative integer")
    return value


def parse_spec(raw: Any) -> Spec:
    root = _require_dict(raw, "$")
    _check_unknown(root, TOP_LEVEL_FIELDS, "$")

    format_version = _positive_int(root, "format_version", "$")
    if format_version != 1:
        _bad("$", "format_version", "unsupported format version", expected=1)

    session_id = _nonempty_str(root, "session_id", "$")

    scale_power = _nonnegative_int(root, "scale_power", "$")
    if scale_power > 3:
        _bad("$", "scale_power", "scale_power must be between 0 and 3")

    budget = _positive_int(root, "global_resource_budget", "$")
    lease_ttl_ms = _positive_int(root, "lease_ttl_ms", "$")
    max_revision_lag = _nonnegative_int(root, "max_revision_lag", "$")
    delay_limit = _nonnegative_int(root, "dispatch_delay_limit_ms", "$")

    chain_raw = root.get("revision_chain")
    if not isinstance(chain_raw, list) or not chain_raw:
        _bad("$.revision_chain", "revision_chain", "revision_chain must be a non-empty list")
    revision_chain: list[int] = []
    for index, revision in enumerate(chain_raw):
        pointer = f"$.revision_chain[{index}]"
        if not _is_int(revision) or revision < 0:
            _bad(pointer, str(index), "revision must be a non-negative integer")
        if revision_chain and revision <= revision_chain[-1]:
            _bad(
                pointer,
                "revision_chain",
                "revision numbers must be strictly increasing",
                previous=revision_chain[-1],
                current=revision,
            )
        revision_chain.append(revision)

    nodes_raw = root.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        _bad("$.nodes", "nodes", "nodes must be a non-empty list")

    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    ports: set[int] = set()
    for index, item in enumerate(nodes_raw):
        pointer = f"$.nodes[{index}]"
        obj = _require_dict(item, pointer)
        _check_unknown(obj, NODE_FIELDS, pointer)
        node_id = _nonempty_str(obj, "node_id", pointer)
        if node_id in node_ids:
            _bad(pointer, "node_id", f"duplicate node_id {node_id!r}")
        node_ids.add(node_id)

        host = _nonempty_str(obj, "host", pointer)
        if host != "127.0.0.1":
            _bad(pointer, "host", "host must be the literal 127.0.0.1")

        port = obj.get("port")
        if not _is_int(port) or not (1024 <= port <= 65535):
            _bad(pointer, "port", "port must be an integer in [1024, 65535]")
        if port in ports:
            _bad(pointer, "port", f"port {port} is declared more than once")
        ports.add(port)

        concurrency_quota = _positive_int(obj, "concurrency_quota", pointer)
        lease_limit = _positive_int(obj, "lease_limit", pointer)

        seed = _nonnegative_int(obj, "seed", pointer)
        if seed > 2**64 - 1:
            _bad(pointer, "seed", "seed must fit in unsigned 64 bits")

        capacity_version = _positive_int(obj, "capacity_version", pointer)
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

    shards_raw = root.get("shards")
    if not isinstance(shards_raw, list):
        _bad("$.shards", "shards", "shards must be a list")

    shards: list[ShardSpec] = []
    shard_ids: set[str] = set()
    for index, item in enumerate(shards_raw):
        pointer = f"$.shards[{index}]"
        obj = _require_dict(item, pointer)
        _check_unknown(obj, SHARD_FIELDS, pointer)
        shard_id = _nonempty_str(obj, "shard_id", pointer)
        if shard_id in shard_ids:
            _bad(pointer, "shard_id", f"duplicate shard_id {shard_id!r}")
        shard_ids.add(shard_id)

        node_id = _nonempty_str(obj, "node_id", pointer)
        if node_id not in node_ids:
            _bad(pointer, "node_id", f"node_id {node_id!r} does not exist")

        dependencies_raw = obj.get("dependencies")
        if not isinstance(dependencies_raw, list):
            _bad(pointer, "dependencies", "dependencies must be a list")
        dependencies: list[str] = []
        seen_dependencies: set[str] = set()
        for dep_index, dependency in enumerate(dependencies_raw):
            dep_pointer = f"{pointer}.dependencies[{dep_index}]"
            if not isinstance(dependency, str) or not dependency:
                _bad(dep_pointer, "dependencies", "dependency ids must be non-empty strings")
            if dependency == shard_id:
                _bad(dep_pointer, "dependencies", "a shard cannot depend on itself")
            if dependency in seen_dependencies:
                _bad(dep_pointer, "dependencies", f"duplicate dependency {dependency!r}")
            seen_dependencies.add(dependency)
            dependencies.append(dependency)

        sample_rows = _nonnegative_int(obj, "sample_rows", pointer)
        if sample_rows > 10_000_000:
            _bad(pointer, "sample_rows", "sample_rows exceeds supported limit 10000000")

        resource_units = _positive_int(obj, "resource_units", pointer)
        if resource_units > budget:
            _bad(
                pointer,
                "resource_units",
                "resource_units cannot exceed global_resource_budget",
                resource_units=resource_units,
                budget=budget,
            )

        priority = obj.get("priority")
        if not _is_int(priority) or not (-1_000_000 <= priority <= 1_000_000):
            _bad(pointer, "priority", "priority must be an integer in [-1000000, 1000000]")

        shards.append(
            ShardSpec(
                shard_id=shard_id,
                node_id=node_id,
                dependencies=tuple(dependencies),
                sample_rows=sample_rows,
                resource_units=resource_units,
                priority=priority,
            )
        )

    shard_map = {shard.shard_id: shard for shard in shards}
    for shard in shards:
        for dependency in shard.dependencies:
            if dependency not in shard_map:
                _bad(
                    f"$.shards[{shard.shard_id}]",
                    "dependencies",
                    f"dependency {dependency!r} does not exist",
                )
    stable_topological_order(shards)

    batches_raw = root.get("delay_batches")
    if not isinstance(batches_raw, list):
        _bad("$.delay_batches", "delay_batches", "delay_batches must be a list")

    batches: list[DelayBatch] = []
    batch_ids: set[str] = set()
    revision_set = set(revision_chain)
    for index, item in enumerate(batches_raw):
        pointer = f"$.delay_batches[{index}]"
        obj = _require_dict(item, pointer)
        _check_unknown(obj, BATCH_FIELDS, pointer)
        batch_id = _nonempty_str(obj, "batch_id", pointer)
        if batch_id in batch_ids:
            _bad(pointer, "batch_id", f"duplicate batch_id {batch_id!r}")
        batch_ids.add(batch_id)

        node_id = _nonempty_str(obj, "node_id", pointer)
        if node_id not in node_ids:
            _bad(pointer, "node_id", f"node_id {node_id!r} does not exist")

        primary_revision = _nonnegative_int(obj, "primary_revision", pointer)
        if primary_revision not in revision_set:
            _bad(pointer, "primary_revision", "primary_revision must reference revision_chain")

        replayed_revision = _nonnegative_int(obj, "replayed_revision", pointer)
        if replayed_revision > primary_revision:
            _bad(
                pointer,
                "replayed_revision",
                "replayed_revision cannot be ahead of primary_revision",
                primary_revision=primary_revision,
                replayed_revision=replayed_revision,
            )

        samples_raw = obj.get("delay_samples_ms")
        if not isinstance(samples_raw, list) or not samples_raw:
            _bad(pointer, "delay_samples_ms", "delay_samples_ms must be a non-empty list")
        samples: list[int] = []
        for sample_index, sample in enumerate(samples_raw):
            sample_pointer = f"{pointer}.delay_samples_ms[{sample_index}]"
            if not _is_int(sample) or sample < 0:
                _bad(sample_pointer, "delay_samples_ms", "delay sample must be a non-negative integer")
            samples.append(sample)

        batches.append(
            DelayBatch(
                batch_id=batch_id,
                node_id=node_id,
                primary_revision=primary_revision,
                replayed_revision=replayed_revision,
                delay_samples_ms=tuple(samples),
            )
        )

    return Spec(
        format_version=format_version,
        session_id=session_id,
        scale_power=scale_power,
        global_resource_budget=budget,
        lease_ttl_ms=lease_ttl_ms,
        max_revision_lag=max_revision_lag,
        dispatch_delay_limit_ms=delay_limit,
        revision_chain=tuple(revision_chain),
        nodes=tuple(nodes),
        shards=tuple(shards),
        delay_batches=tuple(batches),
    )


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {path}: {exc.msg}",
            object=str(path),
            field="$",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc
    except OSError as exc:
        raise SpecError(f"cannot read spec {path}: {exc}", object=str(path), field="$") from exc
    return parse_spec(raw)


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def check_declared_ports(spec: Spec) -> None:
    for node in spec.nodes:
        if not is_port_available(node.host, node.port):
            raise SpecError(
                f"port {node.port} is not available for binding",
                object=f"$.nodes[{node.node_id}]",
                field="port",
            )
