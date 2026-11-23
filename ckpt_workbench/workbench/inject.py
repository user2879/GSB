"""Command-line fault injection for the three required anomaly classes."""

from .errors import ChainError, IntegrityError, SnapshotError
from .ledger import Ledger
from .snapshot import load_snapshot


def inject_shard_corruption(db_path, shard_dir, checkpoint_id, shard_id):
    """Corrupt a shard digest -> validation must report E_INTEGRITY."""
    ledger = Ledger(db_path, shard_dir)
    if ledger.get_shard(checkpoint_id, shard_id) is None:
        raise IntegrityError("target shard not found", [f"{checkpoint_id}/{shard_id}"])
    ledger.corrupt_shard_digest(checkpoint_id, shard_id)
    ledger.close()
    return {"injected": "shard_digest", "target": f"{checkpoint_id}/{shard_id}"}


def inject_dangling_parent(db_path, shard_dir, checkpoint_id):
    """Break a parent link -> validation must report E_CHAIN."""
    ledger = Ledger(db_path, shard_dir)
    ledger.corrupt_parent(checkpoint_id)
    ledger.close()
    return {"injected": "dangling_parent", "target": checkpoint_id}


def inject_snapshot_mismatch(snapshot_dir):
    """Tamper a snapshot content digest -> load must report E_SNAPSHOT."""
    import json
    import os
    path = os.path.join(snapshot_dir, "snapshot.json")
    with open(path, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    body["content_digest"] = "deadbeef" * 8
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
    # verify the loader indeed rejects it
    try:
        load_snapshot(snapshot_dir)
    except SnapshotError:
        return {"injected": "snapshot_digest", "target": path}
    raise SnapshotError("snapshot tamper was not detected", [path])


def run_validation_expect_failure(db_path, shard_dir):
    ledger = Ledger(db_path, shard_dir)
    try:
        ledger.validate()
    except (IntegrityError, ChainError) as error:
        ledger.close()
        return {"reported_code": error.code, "details": error.details}
    finally:
        pass
    ledger.close()
    raise RuntimeError("injected fault not detected")
