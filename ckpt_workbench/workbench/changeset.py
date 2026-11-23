"""Change sets, dependency invalidation and incremental checkpoint commits."""

import copy
import json
import os

from . import dependencies as deps
from .errors import DependencyError
from .hashing import digest
from .model import (
    build_shard_blob,
    content_digest,
    initial_state,
    optimizer_digest,
    partition_digest,
    plan_shards,
    shard_digest,
    tensor_names,
    train_step,
)
from .state_store import reconstruct_state, shard_blob_path
from .hashing import canon_bytes


VALID_OPS = {"train_steps", "set_partition_mode", "set_cursor", "set_rng_state", "drop_tensor"}


def load_change_set(path):
    with open(path, "r", encoding="utf-8") as handle:
        change = json.load(handle)
    validate_change_set(change)
    return change


def validate_change_set(change):
    if "change_id" not in change or "ops" not in change:
        raise ValueError("change set requires change_id and ops")
    if not isinstance(change["ops"], list) or not change["ops"]:
        raise ValueError("change set ops must be a non-empty list")
    for op in change["ops"]:
        if op.get("op") not in VALID_OPS:
            raise ValueError(f"unknown op {op.get('op')!r}; valid={sorted(VALID_OPS)}")


def fingerprint(change):
    """Stable idempotency fingerprint of the logical change set."""
    return {"change_id": change["change_id"], "ops": change["ops"]}


def changed_sources(change):
    """Map change-set ops to dependency source nodes."""
    sources = []
    for op in change["ops"]:
        kind = op["op"]
        if kind == "train_steps":
            targets = op.get("tensors")
            if targets is None:
                targets = op["_all_tensors"]
            for name in targets:
                sources.append(f"param:{name}")
                sources.append(f"momentum:{name}")
            sources.append("rng")
            sources.append("cursor")
        elif kind == "set_partition_mode":
            for shard_id in op.get("shards", []):
                sources.append(f"part:{shard_id}")
        elif kind == "set_cursor":
            sources.append("cursor")
        elif kind == "set_rng_state":
            sources.append("rng")
        elif kind == "drop_tensor":
            sources.append(f"param:{op['tensor']}")
            sources.append(f"momentum:{op['tensor']}")
    return sorted(set(sources))


def _apply_ops(state, change, decl):
    """Pure state mutation for ops that change tensors/rng/cursor."""
    for op in change["ops"]:
        kind = op["op"]
        if kind == "train_steps":
            count = int(op["count"])
            for index in range(count):
                train_step(state, index, decl)
        elif kind == "set_cursor":
            state["cursor"] = int(op["value"])
        elif kind == "set_rng_state":
            state["rng_state"] = op["state"]
        # partition / drop do not mutate tensors here
    return state


def _check_drop_references(ledger, change):
    refs = []
    for op in change["ops"]:
        if op["op"] != "drop_tensor":
            continue
        name = op["tensor"]
        for cid in ledger.list_checkpoint_ids():
            for shard in ledger.get_shards(cid):
                if name in shard["tensors"]:
                    refs.append(f"{cid}/{shard['shard_id']} still references tensor {name}")
    if refs:
        raise DependencyError("cannot drop tensor referenced by committed shards", refs)


def _new_plan(partition_mode, decl, state):
    dims = decl["layer_dims"]
    plans = plan_shards(partition_mode, dims)
    all_names = set(tensor_names(dims))
    live = set(state["tensors"].keys())
    filtered = []
    for plan in plans:
        members = [name for name in plan["tensors"] if name in live]
        if members:
            filtered.append({"id": plan["id"], "tensors": members})
    missing = sorted(all_names - live)
    return filtered, missing


def plan_commit(ledger, change, decl, parent_id):
    """Compute everything needed for a commit without writing anything."""
    if parent_id is None:
        state = initial_state(decl)
        # baseline: advance declaration steps deterministically
        from .model import advance_steps
        advance_steps(state, decl["steps"], decl)
        partition_mode = decl["partition_mode"]
        step = decl["steps"]
    else:
        state = reconstruct_state(ledger, parent_id)
        parent_entry = ledger.get_checkpoint(parent_id)
        partition_mode = parent_entry["partition_mode"]
        step = parent_entry["step"]

    _check_drop_references(ledger, change)
    enriched = copy.deepcopy(change)
    all_names = tensor_names(decl["layer_dims"])
    for op in enriched["ops"]:
        if op["op"] == "train_steps" and op.get("tensors") is None:
            op["_all_tensors"] = all_names
    sources = changed_sources(enriched)
    graph = deps.build_graph(partition_mode, decl["layer_dims"])
    prop_before = deps.propagate(graph, sources)

    # determine post-change partition mode
    new_partition = partition_mode
    for op in change["ops"]:
        if op["op"] == "set_partition_mode":
            new_partition = op["mode"]
            op.setdefault("shards", [s["id"] for s in plan_shards(partition_mode, decl["layer_dims"])])

    # apply state changes
    _apply_ops(state, change, decl)
    for op in change["ops"]:
        if op["op"] == "train_steps":
            step += int(op["count"])
        if op["op"] == "drop_tensor":
            name = op["tensor"]
            state["tensors"].pop(name, None)
            state["momenta"].pop(name, None)

    new_plans, _missing = _new_plan(new_partition, decl, state)
    new_graph = deps.build_graph(new_partition, decl["layer_dims"])

    # shard version mapping from parent (or empty for baseline)
    parent_versions = {}
    parent_digests = {}
    parent_blobs = {}
    if parent_id is not None:
        for shard in ledger.get_shards(parent_id):
            parent_versions[shard["shard_id"]] = shard["shard_version"]
            parent_digests[shard["shard_id"]] = shard["shard_digest"]
            parent_blobs[shard["shard_id"]] = shard

    # Build shard rows; only recompute digests/blobs for affected shards.
    affected_shard_ids = set(prop_before["affected_shards"])
    if partition_mode != new_partition:
        affected_shard_ids = {plan["id"] for plan in new_plans}
    shard_rows = []
    rewritten = []
    unchanged = []
    for index, plan in enumerate(new_plans):
        shard_id = plan["id"]
        new_digest = shard_digest(plan["tensors"], state)
        old_digest = parent_digests.get(shard_id)
        is_affected = shard_id in affected_shard_ids or partition_mode != new_partition
        if is_affected or old_digest is None:
            blob = build_shard_blob(shard_id, plan["tensors"], state)
            version = parent_versions.get(shard_id, 0) + 1
            rewritten.append(shard_id)
        else:
            blob = None
            version = parent_versions.get(shard_id, 1)
            unchanged.append(shard_id)
        shard_rows.append({
            "shard_id": shard_id,
            "shard_index": index,
            "tensors": plan["tensors"],
            "shard_version": version,
            "shard_digest": new_digest,
            "blob": blob,
            "old_digest": old_digest,
        })

    shard_digests = [row["shard_digest"] for row in shard_rows]
    part_digest = partition_digest(shard_digests)
    opt_digest = optimizer_digest(state, decl["layer_dims"]) if state["momenta"] else digest("empty")
    ledger_version = ledger.latest_version() + 1
    checkpoint_id = change["change_id"]

    entry = {
        "checkpoint_id": checkpoint_id,
        "parent_id": parent_id,
        "step": step,
        "shard_count": len(shard_rows),
        "partition_mode": new_partition,
        "optimizer_digest": opt_digest,
        "rng_state": state["rng_state"],
        "cursor": state["cursor"],
        "ledger_version": ledger_version,
    }
    scalar = dict(entry)
    scalar["rng_state"] = canon_bytes(entry["rng_state"])
    entry["content_digest"] = content_digest(scalar)

    return {
        "entry": entry,
        "shard_rows": shard_rows,
        "state": state,
        "rewritten": sorted(set(rewritten)),
        "unchanged": sorted(unchanged),
        "propagation": prop_before,
        "partition_changed": partition_mode != new_partition,
        "partition_digest": part_digest,
        "graph": new_graph,
    }


def commit(ledger, change, decl, parent_id=None, crash_point=None):
    """Idempotent incremental commit inside one SQLite transaction.

    crash_point: None | 'before_commit' simulates a forced kill after blobs are
    written but before the transaction commits.
    """
    fp = fingerprint(change)
    existing = ledger.find_change_checkpoint(fp)
    if existing is not None:
        return {"checkpoint_id": existing, "idempotent": True, "rewritten": []}

    plan = plan_commit(ledger, change, decl, parent_id)
    entry = plan["entry"]
    ledger.begin()
    try:
        shard_rows_persist = []
        for row in plan["shard_rows"]:
            blob_path = shard_blob_path(ledger.shard_dir, entry["checkpoint_id"], row["shard_id"])
            if row["blob"] is not None:
                with open(blob_path, "wb") as handle:
                    handle.write(row["blob"])
            else:
                # unchanged shard: reuse parent blob byte-for-byte
                parent = ledger.get_shard(entry["parent_id"], row["shard_id"])
                blob_path = parent["blob_path"]
            shard_rows_persist.append({
                "shard_id": row["shard_id"],
                "shard_index": row["shard_index"],
                "tensors": row["tensors"],
                "shard_version": row["shard_version"],
                "shard_digest": row["shard_digest"],
                "blob_path": blob_path,
            })

        ledger.set_meta("progress_cursor", entry["cursor"])
        ledger.set_meta("in_flight_checkpoint", entry["checkpoint_id"])

        if crash_point == "before_commit":
            # Simulate SIGKILL: SQLite rolls the open transaction back, but the
            # in-flight marker and partially written shard blobs survive so the
            # next process can classify them as UNKNOWN. We keep the marker by
            # committing only the bookkeeping meta row in a separate transaction.
            unknown_blobs = []
            ledger.rollback()
            ledger.begin()
            ledger.set_meta("in_flight_checkpoint", entry["checkpoint_id"])
            ledger.commit()
            return {
                "checkpoint_id": entry["checkpoint_id"],
                "crash": True,
                "would_rewrite_shards": plan["rewritten"],
                "unknown_shard_blobs_on_disk": unknown_blobs,
                "note": "transaction rolled back; marker retained for recovery",
            }

        ledger.insert_checkpoint(entry, shard_rows_persist, fp)
        ledger.set_meta("in_flight_checkpoint", "")
        ledger.commit()
    except Exception:
        ledger.rollback()
        raise

    return {
        "checkpoint_id": entry["checkpoint_id"],
        "idempotent": False,
        "rewritten": plan["rewritten"],
        "unchanged": plan["unchanged"],
        "propagation": {
            "affected_shards": plan["propagation"]["affected_shards"],
            "order": plan["propagation"]["order"],
            "evidence": plan["propagation"]["evidence"],
        },
        "ledger_version": entry["ledger_version"],
        "partition_mode": entry["partition_mode"],
    }


def full_recompute(change, decl, parent_state=None, parent_entry=None):
    """Recompute every shard from scratch for cross-checking incremental output."""
    if parent_state is None:
        state = initial_state(decl)
        from .model import advance_steps
        advance_steps(state, decl["steps"], decl)
        partition_mode = decl["partition_mode"]
        step = decl["steps"]
        parent_id = None
    else:
        state = copy.deepcopy(parent_state)
        partition_mode = parent_entry["partition_mode"]
        step = parent_entry["step"]
        parent_id = parent_entry["checkpoint_id"]

    new_partition = partition_mode
    for op in change["ops"]:
        if op["op"] == "set_partition_mode":
            new_partition = op["mode"]
    _apply_ops(state, change, decl)
    for op in change["ops"]:
        if op["op"] == "train_steps":
            step += int(op["count"])
        if op["op"] == "drop_tensor":
            name = op["tensor"]
            state["tensors"].pop(name, None)
            state["momenta"].pop(name, None)

    plans, _ = _new_plan(new_partition, decl, state)
    rows = []
    for index, plan in enumerate(plans):
        rows.append({
            "shard_id": plan["id"],
            "shard_index": index,
            "tensors": plan["tensors"],
            "shard_digest": shard_digest(plan["tensors"], state),
        })
    return state, rows, step, new_partition
