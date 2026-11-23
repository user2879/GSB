"""Forced-terminination recovery and post-recovery replay validation."""

import glob
import os

from .errors import ReplayError
from .hashing import canon_bytes
from .model import content_digest, optimizer_digest, partition_digest, shard_digest


def detect_inflight(ledger):
    """Identify an interrupted commit: in-flight marker without committed row."""
    marker = ledger.get_meta("in_flight_checkpoint") or ""
    if not marker:
        return None
    if ledger.get_checkpoint(marker) is not None:
        return None
    blob_hits = sorted(
        path for path in glob.glob(os.path.join(ledger.shard_dir, f"{marker}__*.bin"))
    )
    return {
        "checkpoint_id": marker,
        "unknown_shards": [
            {
                "shard_id": os.path.basename(path).split("__", 1)[1][:-4],
                "blob_path": path,
                "status": "UNKNOWN",
            }
            for path in blob_hits
        ],
    }


def list_unknown_shards(ledger):
    inflight = detect_inflight(ledger)
    return inflight["unknown_shards"] if inflight else []


def rollback_to_parent(ledger):
    """Disposition A: discard in-flight blobs, keep only committed parent state."""
    inflight = detect_inflight(ledger)
    removed = []
    if inflight:
        for item in inflight["unknown_shards"]:
            if os.path.exists(item["blob_path"]):
                os.remove(item["blob_path"])
                removed.append(item["blob_path"])
        ledger.set_meta("in_flight_checkpoint", "")
    return {
        "action": "rollback",
        "head": ledger.latest_checkpoint_id(),
        "removed_unknown_blobs": removed,
        "note": "uncommitted work discarded; resume from committed parent",
    }


def rerun_pending(ledger, change, decl, commit_func):
    """Disposition B: clear unknown blobs and re-run the change set cleanly."""
    inflight = detect_inflight(ledger)
    if inflight:
        for item in inflight["unknown_shards"]:
            if os.path.exists(item["blob_path"]):
                os.remove(item["blob_path"])
        ledger.set_meta("in_flight_checkpoint", "")
    parent_id = ledger.latest_checkpoint_id()
    result = commit_func(ledger, change, decl, parent_id)
    result["action"] = "rerun"
    return result


def committed_state_summary(ledger):
    head = ledger.latest_checkpoint_id()
    if head is None:
        return {"head": None, "cursor": int(ledger.get_meta("progress_cursor") or 0)}
    entry = ledger.get_checkpoint(head)
    return {
        "head": head,
        "cursor": entry["cursor"],
        "ledger_version": entry["ledger_version"],
        "step": entry["step"],
    }


def replay_ledger(ledger, decl, apply_change_func):
    """Re-derive every checkpoint from the recorded change trace and compare.

    apply_change_func(change, decl, parent_state, parent_entry) must return
    (state, full_rows, step, partition_mode), matching changeset.full_recompute.
    Raises E_REPLAY at the first inconsistent checkpoint/shard/field.
    """
    changes = ledger.list_changes()
    parent_state, parent_entry = None, None
    report = []

    for record in changes:
        change = record["change_set"]
        state, rows, step, partition_mode = apply_change_func(
            change, decl, parent_state, parent_entry
        )
        cid = record["checkpoint_id"]
        entry = ledger.get_checkpoint(cid)
        if entry is None:
            raise ReplayError("checkpoint missing during replay", [f"checkpoint={cid}"])

        if entry["step"] != step:
            raise ReplayError("replayed step differs", [
                f"checkpoint={cid} field=step ledger={entry['step']} replay={step}"
            ])
        if entry["partition_mode"] != partition_mode:
            raise ReplayError("replayed partition mode differs", [
                f"checkpoint={cid} field=partition_mode"
            ])

        stored = {shard["shard_id"]: shard for shard in entry["shards"]}
        replayed = {row["shard_id"]: row for row in rows}
        if sorted(stored) != sorted(replayed):
            raise ReplayError("replayed shard set differs", [
                f"checkpoint={cid} field=shard_ids"
            ])
        for shard_id in sorted(replayed):
            if stored[shard_id]["shard_digest"] != replayed[shard_id]["shard_digest"]:
                raise ReplayError("replayed shard digest differs", [
                    f"checkpoint={cid} shard={shard_id} field=shard_digest "
                    f"ledger={stored[shard_id]['shard_digest'][:12]} "
                    f"replay={replayed[shard_id]['shard_digest'][:12]}"
                ])

        replayed_partition = partition_digest([r["shard_digest"] for r in rows])
        stored_partition = partition_digest([s["shard_digest"] for s in entry["shards"]])
        if replayed_partition != stored_partition:
            raise ReplayError("replayed partition digest differs", [
                f"checkpoint={cid} field=partition_digest"
            ])
        if optimizer_digest(state, decl["layer_dims"]) != entry["optimizer_digest"]:
            raise ReplayError("replayed optimizer digest differs", [
                f"checkpoint={cid} field=optimizer_digest"
            ])
        if state["cursor"] != entry["cursor"]:
            raise ReplayError("replayed cursor differs", [
                f"checkpoint={cid} field=cursor"
            ])
        if canon_bytes(state["rng_state"]) != canon_bytes(entry["rng_state"]):
            raise ReplayError("replayed rng state differs", [
                f"checkpoint={cid} field=rng_state"
            ])
        scalar = {key: (canon_bytes(value) if key == "rng_state" else value)
                  for key, value in entry.items()
                  if key not in ("shards", "content_digest", "rng_state_canon")}
        if content_digest(scalar) != entry["content_digest"]:
            raise ReplayError("replayed content digest differs", [
                f"checkpoint={cid} field=content_digest"
            ])

        report.append({"checkpoint_id": cid, "status": "MATCH", "shards": len(rows)})
        parent_state = state
        parent_entry = entry

    return {"status": "OK", "checkpoints": report}
