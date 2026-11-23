"""Reconstruct full in-memory model state from a committed checkpoint."""

import os

from .model import parse_shard_blob


def reconstruct_state(ledger, checkpoint_id):
    """Rebuild tensors, momenta, rng_state and cursor from a checkpoint."""
    entry = ledger.get_checkpoint(checkpoint_id)
    if entry is None:
        raise KeyError(f"unknown checkpoint {checkpoint_id}")
    tensors, momenta = {}, {}
    for shard in entry["shards"]:
        with open(shard["blob_path"], "rb") as handle:
            blob = handle.read()
        shard_tensors, shard_momenta = parse_shard_blob(blob)
        tensors.update(shard_tensors)
        momenta.update(shard_momenta)
    return {
        "tensors": tensors,
        "momenta": momenta,
        "rng_state": entry["rng_state"],
        "cursor": entry["cursor"],
    }


def shard_blob_path(shard_dir, checkpoint_id, shard_id):
    safe_cid = checkpoint_id.replace(os.sep, "_")
    safe_sid = shard_id.replace(os.sep, "_")
    return os.path.join(os.path.abspath(shard_dir), f"{safe_cid}__{safe_sid}.bin")
