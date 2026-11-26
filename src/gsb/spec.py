"""Strict local JSON schema loading.

The loader rejects unknown keys as well as type/range mistakes.  It processes
entries in declaration order so the reported first illegal entry is stable.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .errors import DependencyError, SpecError
from .model import NodeSpec, ShardSpec, Spec

SUPPORTED_FIXED_SCALES = frozenset({100, 1_000, 10_000, 100_000})
FORMAT_VERSION = 1

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ROOT_KEYS = {
    "format_version",
    "capacity_version",
    "fixed_scale",
    "global_budget",
    "max_workers",
    "lease_ttl_ms",
    "handshake_timeout_ms",
    "io_timeout_ms",
    "poll_interval_ms",
    "nodes",
    "shards",
}
_NODE_KEYS = {
    "id",
    "host",
    "port",
    "seed",
    "concurrency",
    "lease_limit",
    "quota",
    "used",
    "fixed_scale",
}
_SHARD_KEYS = {
    "id",
    "node_id",
    "dependencies",
    "demand",
    "sample_count",
    "group_count",
    "batch_size",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    obj: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in obj:
        raise SpecError("required integer is missing", path=f"{path}.{key}")
    value = obj[key]
    if not _is_int(value):
        raise SpecError("value must be an integer", path=f"{path}.{key}", details={"value": value})
    if minimum is not None and value < minimum:
        raise SpecError(
            f"integer must be >= {minimum}",
            path=f"{path}.{key}",
            details={"value": value, "minimum": minimum},
        )
    if maximum is not None and value > maximum:
        raise SpecError(
            f"integer must be <= {maximum}",
            path=f"{path}.{key}",
            details={"value": value, "maximum": maximum},
        )
    return value


def _require_str(obj: dict[str, Any], key: str, path: str) -> str:
    if key not in obj:
        raise SpecError("required string is missing", path=f"{path}.{key}")
    value = obj[key]
    if not isinstance(value, str) or not value:
        raise SpecError("value must be a non-empty string", path=f"{path}.{key}")
    return value


def _require_id(obj: dict[str, Any], key: str, path: str) -> str:
    value = _require_str(obj, key, path)
    if not _ID_RE.fullmatch(value):
        raise SpecError(
            "identifier must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
            path=f"{path}.{key}",
            details={"value": value},
        )
    return value


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("value must be a JSON object", path=path)
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError("value must be a JSON array", path=path)
    return value


def _reject_unknown(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise SpecError(
            "unknown field",
            path=f"{path}.{unknown[0]}",
            details={"unknown_fields": unknown},
        )


def _check_cycles(shards: tuple[ShardSpec, ...]) -> None:
    deps = {shard.id: shard.dependencies for shard in shards}
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(shard_id: str) -> None:
        status = state.get(shard_id, 0)
        if status == 2:
            return
        if status == 1:
            index = stack.index(shard_id)
            cycle = stack[index:] + [shard_id]
            raise DependencyError(
                "dependency cycle detected",
                path=f"shards.{shard_id}.dependencies",
                details={"cycle": cycle},
            )
        state[shard_id] = 1
        stack.append(shard_id)
        for dependency in deps[shard_id]:
            visit(dependency)
        stack.pop()
        state[shard_id] = 2

    for shard in shards:
        visit(shard.id)


def parse_spec(raw: Any) -> Spec:
    root = _require_object(raw, "$")
    _reject_unknown(root, _ROOT_KEYS, "$")

    format_version = _require_int(root, "format_version", "$")
    if format_version != FORMAT_VERSION:
        raise SpecError(
            f"unsupported format_version; expected {FORMAT_VERSION}",
            path="$.format_version",
            details={"value": format_version},
        )
    capacity_version = _require_str(root, "capacity_version", "$")
    fixed_scale = _require_int(root, "fixed_scale", "$")
    if fixed_scale not in SUPPORTED_FIXED_SCALES:
        raise SpecError(
            "fixed_scale is not supported",
            path="$.fixed_scale",
            details={"value": fixed_scale, "supported": sorted(SUPPORTED_FIXED_SCALES)},
        )
    global_budget = _require_int(root, "global_budget", "$", minimum=1)
    max_workers = _require_int(root, "max_workers", "$", minimum=1, maximum=1024)
    lease_ttl_ms = _require_int(root, "lease_ttl_ms", "$", minimum=10, maximum=86_400_000)
    handshake_timeout_ms = _require_int(
        root, "handshake_timeout_ms", "$", minimum=10, maximum=3_600_000
    )
    io_timeout_ms = _require_int(root, "io_timeout_ms", "$", minimum=10, maximum=3_600_000)
    poll_interval_ms = _require_int(root, "poll_interval_ms", "$", minimum=1, maximum=10_000)

    raw_nodes = _require_list(root.get("nodes"), "$.nodes")
    raw_shards = _require_list(root.get("shards"), "$.shards")
    if not raw_nodes:
        raise SpecError("at least one node is required", path="$.nodes")
    if not raw_shards:
        raise SpecError("at least one shard is required", path="$.shards")

    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    ports: set[int] = set()
    for index, raw_node in enumerate(raw_nodes):
        path = f"$.nodes[{index}]"
        node_obj = _require_object(raw_node, path)
        _reject_unknown(node_obj, _NODE_KEYS, path)
        node_id = _require_id(node_obj, "id", path)
        if node_id in node_ids:
            raise SpecError("duplicate node id", path=f"{path}.id", details={"id": node_id})
        node_ids.add(node_id)
        host = _require_str(node_obj, "host", path)
        if host != "127.0.0.1":
            raise SpecError(
                "node host must be the literal 127.0.0.1",
                path=f"{path}.host",
                details={"value": host},
            )
        port = _require_int(node_obj, "port", path, minimum=1, maximum=65535)
        if port in ports:
            raise SpecError(
                "loopback port must be globally unique",
                path=f"{path}.port",
                details={"port": port},
            )
        ports.add(port)
        seed = _require_int(node_obj, "seed", path, minimum=0)
        concurrency = _require_int(node_obj, "concurrency", path, minimum=1, maximum=4096)
        lease_limit = _require_int(node_obj, "lease_limit", path, minimum=1, maximum=4096)
        quota = _require_int(node_obj, "quota", path, minimum=1)
        used = _require_int(node_obj, "used", path, minimum=0, maximum=quota)
        node_scale = _require_int(node_obj, "fixed_scale", path)
        if node_scale not in SUPPORTED_FIXED_SCALES:
            raise SpecError(
                "fixed_scale is not supported",
                path=f"{path}.fixed_scale",
                details={"value": node_scale, "supported": sorted(SUPPORTED_FIXED_SCALES)},
            )
        nodes.append(
            NodeSpec(
                id=node_id,
                host=host,
                port=port,
                seed=seed,
                concurrency=concurrency,
                lease_limit=lease_limit,
                quota=quota,
                used=used,
                fixed_scale=node_scale,
            )
        )

    shards: list[ShardSpec] = []
    shard_ids: set[str] = set()
    for index, raw_shard in enumerate(raw_shards):
        path = f"$.shards[{index}]"
        shard_obj = _require_object(raw_shard, path)
        _reject_unknown(shard_obj, _SHARD_KEYS, path)
        shard_id = _require_id(shard_obj, "id", path)
        if shard_id in shard_ids:
            raise SpecError("duplicate shard id", path=f"{path}.id", details={"id": shard_id})
        shard_ids.add(shard_id)
        node_id = _require_id(shard_obj, "node_id", path)
        if node_id not in node_ids:
            raise DependencyError(
                "shard references an unknown node",
                path=f"{path}.node_id",
                details={"node_id": node_id},
            )
        raw_dependencies = _require_list(shard_obj.get("dependencies"), f"{path}.dependencies")
        dependencies: list[str] = []
        for dep_index, dependency in enumerate(raw_dependencies):
            dep_path = f"{path}.dependencies[{dep_index}]"
            if not isinstance(dependency, str) or not _ID_RE.fullmatch(dependency):
                raise DependencyError("dependency must be a valid shard id", path=dep_path)
            if dependency == shard_id:
                raise DependencyError(
                    "shard cannot depend on itself",
                    path=dep_path,
                    details={"shard_id": shard_id},
                )
            if dependency in dependencies:
                raise DependencyError(
                    "duplicate dependency reference",
                    path=dep_path,
                    details={"dependency": dependency},
                )
            dependencies.append(dependency)
        demand = _require_int(shard_obj, "demand", path, minimum=1)
        sample_count = _require_int(shard_obj, "sample_count", path, minimum=1)
        group_count = _require_int(shard_obj, "group_count", path, minimum=1)
        if group_count > sample_count:
            raise SpecError(
                "group_count cannot exceed sample_count",
                path=f"{path}.group_count",
                details={"group_count": group_count, "sample_count": sample_count},
            )
        batch_size = _require_int(shard_obj, "batch_size", path, minimum=1)
        if batch_size > sample_count:
            raise SpecError(
                "batch_size cannot exceed sample_count",
                path=f"{path}.batch_size",
                details={"batch_size": batch_size, "sample_count": sample_count},
            )
        shards.append(
            ShardSpec(
                id=shard_id,
                node_id=node_id,
                dependencies=tuple(dependencies),
                demand=demand,
                sample_count=sample_count,
                group_count=group_count,
                batch_size=batch_size,
            )
        )

    by_id = {shard.id: shard for shard in shards}
    for shard in shards:
        for index, dependency in enumerate(shard.dependencies):
            if dependency not in by_id:
                raise DependencyError(
                    "dependency references an unknown shard",
                    path=f"shards.{shard.id}.dependencies[{index}]",
                    details={"dependency": dependency},
                )
    _check_cycles(tuple(shards))

    return Spec(
        format_version=format_version,
        capacity_version=capacity_version,
        fixed_scale=fixed_scale,
        global_budget=global_budget,
        max_workers=max_workers,
        lease_ttl_ms=lease_ttl_ms,
        handshake_timeout_ms=handshake_timeout_ms,
        io_timeout_ms=io_timeout_ms,
        poll_interval_ms=poll_interval_ms,
        nodes=tuple(nodes),
        shards=tuple(shards),
    )


def load_spec(path: str | Path) -> Spec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError("cannot read spec file", path=str(source), details={"error": str(exc)}) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            "invalid JSON",
            path=f"$.line{exc.lineno}.column{exc.colno}",
            details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
        ) from exc
    return parse_spec(raw)

