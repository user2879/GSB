"""Deterministic ledger replay and field-level verification."""

from __future__ import annotations

from typing import Any

from .errors import ShardMeshError
from .ledger import Ledger, record_digest
from .samples import aggregate_batch


def replay_ledger(ledger: Ledger, samples: dict[str, Any]) -> dict[str, Any]:
    batches = samples.get("batches", samples) if isinstance(samples, dict) else {}
    digest_mismatches = ledger.verify_row_digests()
    if digest_mismatches:
        first = digest_mismatches[0]
        raise ShardMeshError(
            "E_REPLAY",
            "ledger content digest mismatch during replay",
            category="ledger",
            field_name="content_digest",
            details=first,
        )

    checked: list[dict[str, str]] = []
    for row in ledger.list_rows():
        record = ledger.get_record(row["shard_id"])
        if record is None or not record.get("outcome"):
            continue
        shard_id = record["shard_id"]
        node_id = record.get("node_id")
        stored_summary = record["outcome"]["summary"]
        scale = int(stored_summary.get("scale", 1))
        if shard_id not in batches:
            raise ShardMeshError(
                "E_REPLAY",
                f"deterministic sample for {shard_id} is unavailable",
                category="sample",
                field_name="samples",
                details={"shard_id": shard_id, "node_id": node_id},
            )
        expected_summary = aggregate_batch(batches[shard_id], scale)
        for field_name in sorted(set(stored_summary) | set(expected_summary)):
            if stored_summary.get(field_name) != expected_summary.get(field_name):
                raise ShardMeshError(
                    "E_REPLAY",
                    f"replayed aggregate field mismatch for {shard_id}: {field_name}",
                    category="aggregate",
                    field_name=field_name,
                    details={
                        "shard_id": shard_id,
                        "node_id": node_id,
                        "stored": stored_summary.get(field_name),
                        "recomputed": expected_summary.get(field_name),
                    },
                )
        if record_digest(record) != record.get("content_digest"):
            raise ShardMeshError(
                "E_REPLAY",
                "record digest does not match recomputed canonical digest",
                category="ledger",
                field_name="content_digest",
                details={"shard_id": shard_id, "node_id": node_id},
            )
        checked.append({"shard_id": shard_id, "node_id": str(node_id), "summary_checksum": stored_summary["checksum"]})
    return {"replayed_shards": checked, "checked_count": len(checked), "consistent": True}
