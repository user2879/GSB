from __future__ import annotations

import bisect
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from .schema import BlockGroupDecl, Workload

_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def stable_int_hash(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def column_seed(block_group: BlockGroupDecl, column_name: str) -> int:
    return block_group.seed ^ stable_int_hash(block_group.id, column_name)


def generate_column(
    block_group: BlockGroupDecl,
    column_name: str,
    *,
    dtype: np.dtype[Any] | type[np.int64] | type[np.int32] = np.int64,
) -> np.ndarray[Any, Any]:
    column = block_group.column(column_name)
    counts = column.histogram.counts
    bounds = column.histogram.bounds
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)

    result = np.empty(block_group.row_count, dtype=dtype)
    seed = column_seed(block_group, column_name)
    for row_index in range(block_group.row_count):
        bucket_index = bisect.bisect_right(cumulative, row_index, 1, len(cumulative)) - 1
        lo = bounds[bucket_index]
        hi = bounds[bucket_index + 1]
        width = hi - lo
        if width <= 0:
            raise ValueError(f"invalid histogram interval for {block_group.id}.{column_name}")
        random_bits = _splitmix64(seed + row_index * 0xD1342543DE82EF95)
        value = lo + (random_bits % width)
        result[row_index] = value
    return result


def generate_block_group(block_group: BlockGroupDecl) -> dict[str, np.ndarray[Any, Any]]:
    columns: dict[str, np.ndarray[Any, Any]] = {}
    for column in block_group.columns:
        dtype = np.int32 if column.encoding == "int32" else np.int64
        columns[column.name] = generate_column(block_group, column.name, dtype=dtype)
    return columns


def generate_database(workload: Workload) -> dict[str, dict[str, np.ndarray[Any, Any]]]:
    return {group.id: generate_block_group(group) for group in workload.block_groups}


def block_group_data_digest(
    group: BlockGroupDecl,
    columns: Mapping[str, np.ndarray[Any, Any]],
) -> str:
    payload: dict[str, Any] = {"id": group.id, "row_count": group.row_count, "columns": {}}
    for column in group.columns:
        values = columns[column.name]
        payload["columns"][column.name] = [int(value) for value in values.tolist()]
    from .canonical import digest_json

    return digest_json(payload)


def _json_column(values: np.ndarray[Any, Any]) -> list[int]:
    return [int(value) for value in values.tolist()]


def build_shard_batches(workload: Workload) -> list[dict[str, Any]]:
    """Build deterministic co-partitioned sample batches.

    Tables listed in ``execution.partition.co_partition`` are split by the
    stable hash of their declared key. Replicated tables are copied in full.
    The sample workload declares enough partitioning to preserve two-hop joins.
    """

    if workload.execution is None:
        raise ValueError("workload.execution is required to build shard batches")
    spec = workload.execution
    database = generate_database(workload)
    partition = spec.partition
    driver_column = dict(partition.co_partition)[partition.driver]
    co_partition_hash_key = driver_column.qualified
    driver_values = database[partition.driver][driver_column.column]
    driver_rows_by_shard = [
        np.nonzero(
            np.fromiter(
                (
                    stable_int_hash("co-partition", co_partition_hash_key, int(value)) % spec.shard_count
                    for value in driver_values.tolist()
                ),
                dtype=np.int64,
                count=len(driver_values),
            )
            == shard_index
        )[0]
        for shard_index in range(spec.shard_count)
    ]

    batches: list[dict[str, Any]] = []
    for shard_index in range(spec.shard_count):
        tables: dict[str, Any] = {}
        for table in workload.query.tables:
            columns = database[table]
            if table in partition.replicate:
                indices: np.ndarray[Any, Any] | None = None
            else:
                ref = dict(partition.co_partition)[table]
                table_column = columns[ref.column]
                shard_values = np.fromiter(
                    (
                        stable_int_hash("co-partition", co_partition_hash_key, int(value)) % spec.shard_count
                        for value in table_column.tolist()
                    ),
                    dtype=np.int64,
                    count=len(table_column),
                )
                indices = np.nonzero(shard_values == shard_index)[0]
            table_payload: dict[str, Any] = {
                "row_count": int(len(next(iter(columns.values()))) if indices is None else len(indices)),
                "columns": {},
            }
            for column_name, values in columns.items():
                selected = values if indices is None else values[indices]
                table_payload["columns"][column_name] = _json_column(selected)
            tables[table] = table_payload
        batches.append(
            {
                "shard_index": shard_index,
                "total_shards": spec.shard_count,
                "shard_id": f"shard-{shard_index:03d}",
                "tables": tables,
                "driver_rows": int(len(driver_rows_by_shard[shard_index])),
            }
        )
    return batches


def batch_to_numpy(batch: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray[Any, Any]]]:
    result: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for table, payload in batch["tables"].items():
        result[table] = {
            column: np.asarray(values, dtype=np.int64)
            for column, values in payload["columns"].items()
        }
    return result
