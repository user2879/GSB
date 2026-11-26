"""Shared NumPy integer probe evaluation and receipt fingerprints."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import numpy as np

from .canonical import canonical_json
from .errors import E_REPLY, RouteError
from .fixedpoint import quantize_ratio
from .routing import shard_id
from .spec import INT64_MAX, INT64_MIN, LOWER_EXCLUSIVE, RoutingSpec


def build_route_plan(spec: RoutingSpec) -> dict[str, Any]:
    return {
        "route_version": spec.route_version,
        "scale_factor": spec.scale_factor,
        "databases": [
            {
                "id": database.id,
                "seed": database.seed,
                "lease_limit": database.lease_limit,
                "concurrency_quota": database.concurrency_quota,
            }
            for database in sorted(spec.databases, key=lambda item: item.id)
        ],
        "tables": [
            {
                "id": table.id,
                "key_column": table.key_column,
                "boundaries": list(table.boundaries),
                "route_version": table.route_version,
                "boundary_rule": table.boundary_rule,
                "interval_databases": list(table.database_assignment),
            }
            for table in sorted(spec.tables, key=lambda item: item.id)
        ],
    }


def _validate_probe_keys(keys: Iterable[Mapping[str, Any]], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    table_map = {table["id"]: table for table in plan["tables"]}
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(keys):
        if not isinstance(item, Mapping):
            raise RouteError(E_REPLY, f"request.keys[{index}]", "probe key must be an object")
        table = item.get("table")
        if table not in table_map:
            raise RouteError(E_REPLY, f"request.keys[{index}].table", "unknown table in probe", {"table": table})
        key = item.get("key")
        seq = item.get("seq")
        if not isinstance(key, int) or isinstance(key, bool) or key < INT64_MIN or key > INT64_MAX:
            raise RouteError(E_REPLY, f"request.keys[{index}].key", "probe key must be a signed 64-bit integer")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise RouteError(E_REPLY, f"request.keys[{index}].seq", "probe seq must be a non-negative integer")
        validated.append({"seq": seq, "table": table, "key": key})
    return validated


def numpy_route_results(
    plan: Mapping[str, Any],
    keys: Iterable[Mapping[str, Any]],
    *,
    database_id: str,
    batch_id: str,
) -> dict[str, Any]:
    """Route a probe batch with NumPy and produce deterministic integer results."""

    validated = _validate_probe_keys(keys, plan)
    if plan.get("route_version") != plan.get("route_version") or not isinstance(plan.get("route_version"), int):
        raise RouteError(E_REPLY, "request.route_version", "invalid route plan version")
    route_version = int(plan["route_version"])
    scale_factor = int(plan["scale_factor"])
    table_map = {table["id"]: table for table in plan["tables"]}
    results: list[dict[str, Any]] = []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in validated:
        grouped.setdefault(item["table"], []).append(item)

    for table_id in sorted(grouped):
        table = table_map[table_id]
        items = grouped[table_id]
        values = np.asarray([item["key"] for item in items], dtype=np.int64)
        bounds = np.asarray(table["boundaries"], dtype=np.int64)
        side = "right" if table["boundary_rule"] == "lower_inclusive_upper_exclusive" else "left"
        raw = np.searchsorted(bounds, values, side=side) - 1
        if side == "right":
            valid = (raw >= 0) & (raw < len(bounds) - 1) & (values >= bounds[0]) & (values < bounds[-1])
        else:
            valid = (raw >= 0) & (raw < len(bounds) - 1) & (values > bounds[0]) & (values <= bounds[-1])
        invalid_index = int(np.flatnonzero(~valid)[0]) if not bool(np.all(valid)) else None
        if invalid_index is not None:
            item = items[invalid_index]
            raise RouteError(
                E_REPLY,
                f"request.keys[seq={item['seq']}].key",
                "key is not routable under the configured boundary rule",
                {"key": item["key"], "table": table_id},
            )
        lower = bounds[raw]
        upper = bounds[raw + 1]
        assigned = np.asarray(table["interval_databases"], dtype=object)[raw]
        for position, item in enumerate(items):
            interval_index = int(raw[position])
            assigned_database = str(assigned[position])
            if assigned_database != database_id:
                raise RouteError(
                    E_REPLY,
                    f"request.keys[seq={item['seq']}].database_id",
                    "probe key is not owned by this worker",
                    {"expected": assigned_database, "actual": database_id},
                )
            key_value = int(values[position])
            lower_value = int(lower[position])
            upper_value = int(upper[position])
            width = upper_value - lower_value
            offset = key_value - lower_value
            fixedpoint = quantize_ratio(offset, width, scale_factor)
            boundary_hit = None
            if key_value in table["boundaries"]:
                boundary_hit = {
                    "boundary": key_value,
                    "side": "lower" if side == "right" else "upper",
                    "ownership": "right_interval" if side == "right" else "left_interval",
                    "rule": table["boundary_rule"],
                    "interval_index": interval_index,
                }
            results.append(
                {
                    "seq": item["seq"],
                    "table": table_id,
                    "key": key_value,
                    "database_id": database_id,
                    "shard": shard_id(table_id, interval_index),
                    "interval_index": interval_index,
                    "lower_bound": lower_value,
                    "upper_bound": upper_value,
                    "interval_width": width,
                    "route_version": route_version,
                    "boundary_rule": table["boundary_rule"],
                    "boundary_hit": boundary_hit,
                    "relative_numerator": offset,
                    "relative_denominator": width,
                    "fixedpoint": fixedpoint,
                }
            )

    results.sort(key=lambda result: (result["seq"], result["table"], result["key"]))

    checksum_values = np.asarray([result["key"] for result in results], dtype=np.int64)
    checksum_seq = np.asarray([result["seq"] for result in results], dtype=np.int64)
    checksum_bucket = np.asarray([result["interval_index"] for result in results], dtype=np.int64)
    checksum_fp = np.asarray([result["fixedpoint"] for result in results], dtype=np.int64)
    if len(results):
        mixed = (
            checksum_values.astype(np.uint64)
            ^ (checksum_seq.astype(np.uint64) << np.uint64(1))
            ^ (checksum_bucket.astype(np.uint64) << np.uint64(7))
            ^ checksum_fp.astype(np.uint64)
        )
        initial = np.uint64(plan["databases"][0]["seed"] % (1 << 64)) if plan.get("databases") else np.uint64(0)
        for database in plan.get("databases", []):
            if database.get("id") == database_id:
                initial = np.uint64(int(database["seed"]) % (1 << 64))
                break
        checksum = np.bitwise_xor.reduce(mixed, initial=initial, dtype=np.uint64)
    else:
        checksum = np.uint64(0)
    context = {
        "batch_id": batch_id,
        "database_id": database_id,
        "route_version": route_version,
        "scale_factor": scale_factor,
        "results": results,
    }
    digest = hashlib.sha256(canonical_json(context).encode("utf-8")).hexdigest()
    return {
        "results": results,
        "probe_digest": digest,
        "fixedpoint_checksum_u64": str(int(checksum)),
        "route_version": route_version,
        "database_id": database_id,
    }


def expected_probe_receipt(
    spec: RoutingSpec,
    *,
    batch_id: str,
    database_id: str,
    keys: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = build_route_plan(spec)
    return numpy_route_results(plan, keys, database_id=database_id, batch_id=batch_id)


__all__ = ["build_route_plan", "expected_probe_receipt", "numpy_route_results"]
