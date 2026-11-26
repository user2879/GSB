from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest
from .errors import SpecError
from .graph import stable_topological_order


SUPPORTED_SCALE_FACTORS = tuple(
    sorted({2**power for power in range(21)} | {10**power for power in range(10)})
)
MIN_PORT = 1
MAX_PORT = 65535


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fail(field: str, position: str | int, message: str, **details: object) -> None:
    raise SpecError(message, field=field, position=str(position), **details)


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(
            f"cannot read spec {path}: {exc}",
            field="$",
            position=str(path),
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            field="$",
            position=f"{path}:{exc.lineno}:{exc.colno}",
            json_line=exc.lineno,
            json_column=exc.colno,
        ) from exc


def is_supported_scale(value: Any) -> bool:
    return _is_int(value) and value in SUPPORTED_SCALE_FACTORS


def _require_int(
    item: dict[str, Any],
    key: str,
    location: str,
    *,
    minimum: int | None = None,
    nonnegative: bool = False,
    positive: bool = False,
) -> int:
    field = f"{location}.{key}"
    if key not in item:
        _fail(field, field, f"missing required integer field {key}")
    value = item[key]
    if not _is_int(value):
        _fail(field, field, f"{key} must be an integer")
    if nonnegative and value < 0:
        _fail(field, field, f"{key} must be non-negative")
    if positive and value <= 0:
        _fail(field, field, f"{key} must be a positive integer")
    if minimum is not None and value < minimum:
        _fail(field, field, f"{key} must be >= {minimum}")
    return int(value)


def _require_str(item: dict[str, Any], key: str, location: str) -> str:
    field = f"{location}.{key}"
    if key not in item or not isinstance(item[key], str) or not item[key]:
        _fail(field, field, f"{key} must be a non-empty string")
    return item[key]


def load_spec(path: str | Path) -> dict[str, Any]:
    raw = _read_json(path)
    return parse_spec(raw, source_path=str(path))


def parse_spec(raw: Any, *, source_path: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        _fail("$", "$", "spec root must be a JSON object")

    session_id = raw.get("session_id", "local-session")
    if not isinstance(session_id, str) or not session_id:
        _fail("session_id", "session_id", "session_id must be a non-empty string")

    scale_factor = raw.get("scale_factor", 1000)
    if not is_supported_scale(scale_factor):
        _fail(
            "scale_factor",
            "scale_factor",
            "scale_factor must be a supported binary/decimal integer power "
            f"{SUPPORTED_SCALE_FACTORS}",
            supported=list(SUPPORTED_SCALE_FACTORS),
        )

    budget_raw = raw.get("budget", raw.get("global_budget"))
    if budget_raw is None:
        _fail("budget", "budget", "missing budget declaration")
    if isinstance(budget_raw, int) and not isinstance(budget_raw, bool):
        budget = {"resource_units": budget_raw}
        budget_location = "budget"
    elif isinstance(budget_raw, dict):
        budget = budget_raw
        budget_location = "budget"
    else:
        _fail("budget", "budget", "budget must be an integer or object")
    budget_units = _require_int(
        budget,
        "resource_units",
        budget_location,
        positive=True,
    )

    scheduling_raw = raw.get("scheduling", {})
    if scheduling_raw is None:
        scheduling_raw = {}
    if not isinstance(scheduling_raw, dict):
        _fail("scheduling", "scheduling", "scheduling must be an object")
    root_round_boundary = raw.get("round_boundary", raw.get("scheduling_boundary"))
    scheduling_defaults = {
        "lease_ttl_ms": 5_000,
        "request_timeout_ms": 3_000,
        "handshake_timeout_ms": 5_000,
        "round_boundary": scheduling_raw.get(
            "round_boundary",
            scheduling_raw.get(
                "max_admissions_per_round",
                root_round_boundary if root_round_boundary is not None else 1_000_000,
            ),
        ),
        "round_boundary_ms": scheduling_raw.get("round_boundary_ms", 1_000),
        "poll_interval_ms": scheduling_raw.get("poll_interval_ms", 5),
    }
    scheduling: dict[str, Any] = {}
    for key, default_value in scheduling_defaults.items():
        value = scheduling_raw.get(key, default_value)
        if key == "round_boundary" and key not in scheduling_raw and root_round_boundary is not None:
            value = root_round_boundary
        if not _is_int(value) or value <= 0:
            _fail(
                f"scheduling.{key}",
                f"scheduling.{key}",
                f"scheduling.{key} must be a positive integer",
            )
        scheduling[key] = int(value)
    scheduling["worker_pool_size"] = scheduling_raw.get("worker_pool_size")
    if scheduling["worker_pool_size"] is not None and (
        not _is_int(scheduling["worker_pool_size"])
        or scheduling["worker_pool_size"] <= 0
    ):
        _fail(
            "scheduling.worker_pool_size",
            "scheduling.worker_pool_size",
            "worker_pool_size must be a positive integer",
        )

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        _fail("nodes", "nodes", "nodes must be a non-empty array")

    nodes: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, item in enumerate(nodes_raw):
        location = f"nodes[{index}]"
        if not isinstance(item, dict):
            _fail(location, index, "node declaration must be an object")
        node_id = _require_str(item, "node_id", location)
        if node_id in seen_node_ids:
            _fail(f"{location}.node_id", f"{location}.node_id", f"duplicate node_id {node_id!r}")
        seen_node_ids.add(node_id)

        host = item.get("host", "127.0.0.1")
        if host != "127.0.0.1":
            _fail(f"{location}.host", f"{location}.host", "host must be the literal 127.0.0.1")

        port = _require_int(item, "port", location, minimum=MIN_PORT)
        if port > MAX_PORT:
            _fail(f"{location}.port", f"{location}.port", "port must be <= 65535")
        if port in seen_ports:
            _fail(f"{location}.port", f"{location}.port", f"duplicate loopback port {port}")
        seen_ports.add(port)

        concurrency = _require_int(item, "concurrency", location, nonnegative=True)
        if "concurrency_quota" in item:
            concurrency = _require_int(item, "concurrency_quota", location, nonnegative=True)
        lease_limit = _require_int(
            item,
            "lease_limit",
            location,
            nonnegative=True,
        )
        if "max_leases" in item:
            lease_limit = _require_int(item, "max_leases", location, nonnegative=True)
        seed = _require_int(item, "seed", location, minimum=0)
        if seed > 2**32 - 1:
            _fail(f"{location}.seed", f"{location}.seed", "seed must fit uint32")
        capacity_version = _require_int(item, "capacity_version", location, positive=True)

        fixed_scale = item.get("fixed_scale", scale_factor)
        if not is_supported_scale(fixed_scale):
            _fail(
                f"{location}.fixed_scale",
                f"{location}.fixed_scale",
                "fixed_scale must be a supported decimal power",
            )

        group_count = item.get("group_count", raw.get("default_group_count", 4))
        if not _is_int(group_count) or group_count <= 0:
            _fail(f"{location}.group_count", f"{location}.group_count", "group_count must be positive")

        processing_delay_ms = item.get("processing_delay_ms", 0)
        if not _is_int(processing_delay_ms) or processing_delay_ms < 0:
            _fail(
                f"{location}.processing_delay_ms",
                f"{location}.processing_delay_ms",
                "processing_delay_ms must be a non-negative integer",
            )

        nodes.append(
            {
                "node_id": node_id,
                "host": "127.0.0.1",
                "port": port,
                "concurrency": concurrency,
                "lease_limit": lease_limit,
                "seed": seed,
                "capacity_version": capacity_version,
                "fixed_scale": fixed_scale,
                "group_count": group_count,
                "processing_delay_ms": processing_delay_ms,
            }
        )

    shards_raw = raw.get("shards")
    if not isinstance(shards_raw, list) or not shards_raw:
        _fail("shards", "shards", "shards must be a non-empty array")

    shards: list[dict[str, Any]] = []
    seen_shard_ids: set[str] = set()
    node_ids = seen_node_ids
    for index, item in enumerate(shards_raw):
        location = f"shards[{index}]"
        if not isinstance(item, dict):
            _fail(location, index, "shard declaration must be an object")
        shard_id = _require_str(item, "shard_id", location)
        if shard_id in seen_shard_ids:
            _fail(
                f"{location}.shard_id",
                f"{location}.shard_id",
                f"duplicate shard_id {shard_id!r}",
            )
        seen_shard_ids.add(shard_id)

        dependencies = item.get("dependencies", item.get("depends_on", []))
        if not isinstance(dependencies, list) or not all(
            isinstance(dep, str) and dep for dep in dependencies
        ):
            _fail(
                f"{location}.dependencies",
                f"{location}.dependencies",
                "dependencies must be an array of non-empty strings",
            )
        if shard_id in dependencies:
            _fail(
                f"{location}.dependencies",
                f"{location}.dependencies",
                f"shard {shard_id!r} cannot depend on itself",
            )
        for dep_index, dep in enumerate(dependencies):
            if dep not in {x["shard_id"] for x in shards_raw if isinstance(x, dict)} and dep not in seen_shard_ids:
                # Forward references are allowed; final existence is checked below.
                pass
            if dep == shard_id:
                _fail(f"{location}.dependencies[{dep_index}]", dep_index, "self dependency")

        rows = _require_int(item, "rows", location, positive=True)
        resource_units = item.get("resource_units", item.get("units"))
        if resource_units is None:
            _fail(f"{location}.resource_units", location, "missing resource_units")
        if not _is_int(resource_units) or resource_units <= 0:
            _fail(
                f"{location}.resource_units",
                f"{location}.resource_units",
                "resource_units must be a positive integer",
            )
        priority = _require_int(item, "priority", location)

        node_ref = item.get("node_id", item.get("node_ref"))
        if node_ref is not None:
            if not isinstance(node_ref, str) or node_ref not in node_ids:
                _fail(
                    f"{location}.node_id",
                    f"{location}.node_id",
                    f"node reference {node_ref!r} does not exist",
                )

        group_count = item.get("group_count")
        if group_count is not None and (not _is_int(group_count) or group_count <= 0):
            _fail(f"{location}.group_count", f"{location}.group_count", "group_count must be positive")

        shards.append(
            {
                "shard_id": shard_id,
                "dependencies": list(dependencies),
                "rows": rows,
                "resource_units": int(resource_units),
                "priority": priority,
                "node_id": node_ref,
                "group_count": group_count,
            }
        )

    shard_ids = {shard["shard_id"] for shard in shards}
    for shard in shards:
        for dep_index, dep in enumerate(shard["dependencies"]):
            if dep not in shard_ids:
                location = f"shards[{_shard_index(shards, shard['shard_id'])}].dependencies[{dep_index}]"
                _fail(location, dep_index, f"dependency reference {dep!r} does not exist")

    try:
        topo_order = stable_topological_order(shards)
    except ValueError as exc:
        cycle = getattr(exc, "cycle", None)
        raise SpecError(
            str(exc),
            field="shards.dependencies",
            position="shards",
            cycle=cycle,
        ) from exc

    normalized = {
        "schema_version": int(raw.get("schema_version", 1)),
        "session_id": session_id,
        "scale_factor": int(scale_factor),
        "budget_units": budget_units,
        "scheduling": scheduling,
        "nodes": nodes,
        "shards": shards,
        "topo_order": topo_order,
        "source_path": source_path,
    }
    normalized["spec_digest"] = digest(_digest_payload(normalized))
    normalized["node_by_id"] = {node["node_id"]: node for node in nodes}
    normalized["shard_by_id"] = {shard["shard_id"]: shard for shard in shards}
    normalized["topo_index"] = {shard_id: i for i, shard_id in enumerate(topo_order)}
    return normalized


def _shard_index(shards: Iterable[dict[str, Any]], shard_id: str) -> int:
    for index, shard in enumerate(shards):
        if shard["shard_id"] == shard_id:
            return index
    return -1


def _digest_payload(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": spec["schema_version"],
        "session_id": spec["session_id"],
        "scale_factor": spec["scale_factor"],
        "budget_units": spec["budget_units"],
        "scheduling": spec["scheduling"],
        "nodes": spec["nodes"],
        "shards": spec["shards"],
        "topo_order": spec["topo_order"],
    }


def load_specs(old_path: str | Path, new_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_spec(old_path), load_spec(new_path)
