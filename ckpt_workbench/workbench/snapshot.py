"""Session snapshots: versioned full-state capture and whole-set replacement."""

import json
import os
import shutil
import stat

from .errors import SnapshotError
from .hashing import digest
from .model import partition_digest

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_FILE = "snapshot.json"
BACKUP_PREFIX = "backup_"


def _readonly_backup(src_dir, backup_dir):
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(src_dir, backup_dir)
    for root, _dirs, files in os.walk(backup_dir):
        for name in files:
            path = os.path.join(root, name)
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def build_snapshot(ledger):
    checkpoints = []
    for cid in ledger.list_checkpoint_ids():
        entry = ledger.get_checkpoint(cid)
        checkpoints.append({
            "checkpoint_id": entry["checkpoint_id"],
            "parent_id": entry["parent_id"],
            "step": entry["step"],
            "ledger_version": entry["ledger_version"],
            "partition_mode": entry["partition_mode"],
            "shard_count": entry["shard_count"],
            "optimizer_digest": entry["optimizer_digest"],
            "rng_state": entry["rng_state"],
            "cursor": entry["cursor"],
            "content_digest": entry["content_digest"],
            "shards": [
                {
                    "shard_id": shard["shard_id"],
                    "shard_index": shard["shard_index"],
                    "tensors": shard["tensors"],
                    "shard_version": shard["shard_version"],
                    "shard_digest": shard["shard_digest"],
                }
                for shard in entry["shards"]
            ],
        })

    changes = [
        {
            "seq": row["seq"],
            "checkpoint_id": row["checkpoint_id"],
            "parent_id": row["parent_id"],
            "change_set": row["change_set"],
            "ledger_version": row["ledger_version"],
        }
        for row in ledger.list_changes()
    ]

    checkpoint_set_version = digest([c["checkpoint_id"] for c in checkpoints])
    partition_version = digest([
        partition_digest([s["shard_digest"] for s in c["shards"]]) for c in checkpoints
    ])
    body = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "ledger_version": ledger.latest_version(),
        "checkpoint_set_version": checkpoint_set_version,
        "partition_version": partition_version,
        "checkpoints": checkpoints,
        "change_trace": changes,
    }
    body["content_digest"] = digest({k: v for k, v in body.items()})
    return body


def save_snapshot(ledger, snapshot_dir):
    os.makedirs(snapshot_dir, exist_ok=True)
    body = build_snapshot(ledger)
    path = os.path.join(snapshot_dir, SNAPSHOT_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
    # Materialize one blob per (checkpoint, shard) so the snapshot is fully
    # self-contained, even when an unchanged shard reuses a parent blob path.
    blob_target = os.path.join(snapshot_dir, "shards")
    if os.path.exists(blob_target):
        shutil.rmtree(blob_target)
    os.makedirs(blob_target, exist_ok=True)
    for cp in body["checkpoints"]:
        for shard in cp["shards"]:
            source = ledger.get_shard(cp["checkpoint_id"], shard["shard_id"])["blob_path"]
            destination = os.path.join(
                blob_target, f"{cp['checkpoint_id']}__{shard['shard_id']}.bin"
            )
            shutil.copyfile(source, destination)
    return path


def load_snapshot(snapshot_dir):
    path = os.path.join(snapshot_dir, SNAPSHOT_FILE)
    if not os.path.exists(path):
        raise SnapshotError("snapshot file not found", [path])
    with open(path, "r", encoding="utf-8") as handle:
        body = json.load(handle)
    if body.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotError("unsupported snapshot format version", [
            f"expected={SNAPSHOT_FORMAT_VERSION} got={body.get('format_version')}"
        ])
    stored = body.get("content_digest")
    recomputed = digest({k: v for k, v in body.items() if k != "content_digest"})
    if stored != recomputed:
        raise SnapshotError("snapshot content digest mismatch", [
            f"stored={stored[:12]} recomputed={recomputed[:12]}"
        ])
    return body


def compare_with_ledger(snapshot, ledger):
    """Return list of version differences (empty == identical)."""
    diffs = []
    if snapshot["ledger_version"] != ledger.latest_version():
        diffs.append(
            f"ledger_version snapshot={snapshot['ledger_version']} "
            f"current={ledger.latest_version()}"
        )
    current_set = digest(ledger.list_checkpoint_ids())
    if snapshot["checkpoint_set_version"] != current_set:
        diffs.append("checkpoint_set_version differs")
    current_partition = digest([
        partition_digest([s["shard_digest"] for s in ledger.get_checkpoint(cid)["shards"]])
        for cid in ledger.list_checkpoint_ids()
    ])
    if snapshot["partition_version"] != current_partition:
        diffs.append("partition_version differs")
    return diffs


def restore_snapshot(snapshot_dir, target_db, target_shard_dir, replace=True):
    """Whole-set replacement restore; old state is kept as a read-only backup."""
    snapshot = load_snapshot(snapshot_dir)
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(target_db)),
                              BACKUP_PREFIX + os.path.basename(snapshot_dir))
    replaced = []
    if os.path.exists(target_db):
        _readonly_backup(os.path.dirname(os.path.abspath(target_db)), backup_dir)
        os.remove(target_db)
        for extra in (target_db + "-wal", target_db + "-shm"):
            if os.path.exists(extra):
                os.remove(extra)
    if os.path.exists(target_shard_dir):
        shutil.rmtree(target_shard_dir)

    # rebuild ledger from snapshot (import here to avoid circular import)
    from .ledger import Ledger
    ledger = Ledger(target_db, target_shard_dir, create=True)
    trace_by_cid = {row["checkpoint_id"]: row for row in snapshot["change_trace"]}
    ledger.begin()
    for cp in snapshot["checkpoints"]:
        replaced.append(cp["checkpoint_id"])
        shards = []
        for shard in cp["shards"]:
            source_blob = os.path.join(snapshot_dir, "shards",
                                       f"{cp['checkpoint_id']}__{shard['shard_id']}.bin")
            target_blob = os.path.join(target_shard_dir,
                                       f"{cp['checkpoint_id']}__{shard['shard_id']}.bin")
            if not os.path.exists(source_blob):
                raise SnapshotError("snapshot shard blob missing", [source_blob])
            shutil.copyfile(source_blob, target_blob)
            shards.append({**shard, "blob_path": os.path.abspath(target_blob)})
        scalar = {k: v for k, v in cp.items() if k != "shards"}
        change_row = trace_by_cid[cp["checkpoint_id"]]
        ledger.insert_restored(scalar, shards, change_row)
    ledger.set_meta("restored_from", snapshot_dir)
    ledger.commit()
    ledger.close()
    return {
        "replaced_checkpoints": sorted(replaced),
        "backup_dir": backup_dir if os.path.exists(backup_dir) else None,
        "snapshot": snapshot,
    }
