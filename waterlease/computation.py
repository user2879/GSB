from __future__ import annotations

from typing import Any

from .hashing import sha256_hex
from .process_manager import ProcessManager
from .sampling import (
    build_summary_core,
    combine_aggregates,
    generate_samples,
    split_batches,
)
from .spec import NodeDecl, ShardDecl


def compute_shard_remotely(
    processes: ProcessManager,
    node: NodeDecl,
    shard: ShardDecl,
    upstream_digests: dict[str, str],
) -> dict[str, Any]:
    samples = generate_samples(
        node.seed,
        shard.shard_id,
        shard.sample_rows,
        fixed_point_scale=node.fixed_point_scale,
    )
    sample_set_digest = sha256_hex(samples)
    batches = split_batches(samples, shard.batch_size)
    batch_receipts: list[dict[str, Any]] = []
    batch_aggregates: list[dict[str, Any]] = []
    node_elapsed_ms = 0
    for batch_index, batch in enumerate(batches):
        response = processes.handles[node.node_id].client_message(
            "aggregate",
            shard_id=shard.shard_id,
            capacity_version=shard.capacity_version,
            samples=batch,
            batch_index=batch_index,
            batch_count=len(batches),
        )
        aggregate = response["aggregate"]
        batch_aggregates.append(aggregate)
        node_elapsed_ms += int(response.get("elapsed_ms", 0))
        batch_receipts.append(
            {
                "batch_index": batch_index,
                "node_id": response.get("node_id"),
                "shard_id": response.get("shard_id"),
                "capacity_version": response.get("capacity_version"),
                "elapsed_ms": response.get("elapsed_ms"),
                "worker_pid": response.get("worker_pid"),
                "aggregate_digest": sha256_hex(aggregate),
            }
        )

    combined = combine_aggregates(batch_aggregates)
    summary = build_summary_core(
        node_id=node.node_id,
        shard_id=shard.shard_id,
        capacity_version=shard.capacity_version,
        aggregate=combined,
        upstream_digests=upstream_digests,
    )
    return {
        "summary": summary,
        "sample_set_digest": sample_set_digest,
        "batch_count": len(batches),
        "batch_receipts": batch_receipts,
        "node_elapsed_ms": node_elapsed_ms,
    }
