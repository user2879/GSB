"""Compare incremental commit output against a full recompute."""

from .changeset import full_recompute
from .errors import IncrementalError
from .state_store import reconstruct_state
from .model import content_digest, optimizer_digest, partition_digest
from .hashing import canon_bytes


def verify_checkpoint(ledger, checkpoint_id, change, decl):
    """Raise E_INCR at the first checkpoint/shard/field mismatch, else True."""
    entry = ledger.get_checkpoint(checkpoint_id)
    if entry["parent_id"] is None:
        parent_state, parent_entry = None, None
    else:
        parent_state = reconstruct_state(ledger, entry["parent_id"])
        parent_entry = ledger.get_checkpoint(entry["parent_id"])

    state, full_rows, step, partition_mode = full_recompute(
        change, decl, parent_state, parent_entry
    )

    if step != entry["step"]:
        raise IncrementalError("step mismatch", [
            f"checkpoint={checkpoint_id} field=step ledger={entry['step']} full={step}"
        ])
    if partition_mode != entry["partition_mode"]:
        raise IncrementalError("partition_mode mismatch", [
            f"checkpoint={checkpoint_id} field=partition_mode "
            f"ledger={entry['partition_mode']} full={partition_mode}"
        ])

    ledger_shards = {row["shard_id"]: row for row in entry["shards"]}
    full_ids = [row["shard_id"] for row in full_rows]
    if full_ids != sorted(ledger_shards):
        raise IncrementalError("shard id set/order mismatch", [
            f"checkpoint={checkpoint_id} field=shard_ids "
            f"ledger={sorted(ledger_shards)} full={full_ids}"
        ])

    for row in full_rows:
        stored = ledger_shards[row["shard_id"]]
        if stored["shard_digest"] != row["shard_digest"]:
            raise IncrementalError("shard digest mismatch", [
                f"checkpoint={checkpoint_id} shard={row['shard_id']} "
                f"field=shard_digest ledger={stored['shard_digest'][:12]} "
                f"full={row['shard_digest'][:12]}"
            ])
        if stored["tensors"] != row["tensors"]:
            raise IncrementalError("shard tensor membership mismatch", [
                f"checkpoint={checkpoint_id} shard={row['shard_id']} field=tensors"
            ])

    full_partition = partition_digest([row["shard_digest"] for row in full_rows])
    stored_partition = partition_digest([row["shard_digest"] for row in entry["shards"]])
    if full_partition != stored_partition:
        raise IncrementalError("partition derived digest mismatch", [
            f"checkpoint={checkpoint_id} field=partition_digest"
        ])

    full_optimizer = optimizer_digest(state, decl["layer_dims"])
    if full_optimizer != entry["optimizer_digest"]:
        raise IncrementalError("optimizer digest mismatch", [
            f"checkpoint={checkpoint_id} field=optimizer_digest"
        ])
    if state["cursor"] != entry["cursor"]:
        raise IncrementalError("cursor mismatch", [
            f"checkpoint={checkpoint_id} field=cursor ledger={entry['cursor']} full={state['cursor']}"
        ])
    if canon_bytes(state["rng_state"]) != canon_bytes(entry["rng_state"]):
        raise IncrementalError("rng_state mismatch", [
            f"checkpoint={checkpoint_id} field=rng_state"
        ])

    scalar = {key: (canon_bytes(value) if key == "rng_state" else value)
              for key, value in entry.items()
              if key not in ("shards", "content_digest", "rng_state_canon")}
    if content_digest(scalar) != entry["content_digest"]:
        raise IncrementalError("content digest mismatch", [
            f"checkpoint={checkpoint_id} field=content_digest"
        ])
    return True
