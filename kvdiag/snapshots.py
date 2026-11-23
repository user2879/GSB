"""Shard snapshot chain: creation, integrity checks and replay."""

from .canon import content_digest
from .errors import ChainError


def make_snapshot_id(batch_id, shard_id):
    return "snap-%04d-%s" % (batch_id, shard_id)


def build_snapshot_rows(spec, batch_id, shard_states, parent_of):
    """Build snapshot rows for a batch, chained in sorted shard-id order.

    shard_states: {shard_id: {"index_pos_lo","index_pos_hi","lease_expiry_pos",
                              "content_digest"}}
    parent_of: {shard_id: parent_snapshot_id_or_None}; parents must already
    exist in the ledger or in this batch's earlier links.
    """
    rows = []
    for shard_id in sorted(shard_states):
        state = shard_states[shard_id]
        rows.append({
            "snapshot_id": make_snapshot_id(batch_id, shard_id),
            "parent_id": parent_of.get(shard_id),
            "shard_id": shard_id,
            "index_pos_lo": state["index_pos_lo"],
            "index_pos_hi": state["index_pos_hi"],
            "lease_expiry_pos": state["lease_expiry_pos"],
            "content_digest": state["content_digest"],
        })
    return rows


def _all_snapshots(ledger):
    return {r["snapshot_id"]: r for r in ledger.query("SELECT * FROM snapshots")}


def check_chain(ledger, head_id):
    """Validate parent existence and acyclicity from head to root."""
    snaps = _all_snapshots(ledger)
    if head_id not in snaps:
        raise ChainError("snapshot not found", details={"snapshot_id": head_id})
    ordered = []
    seen = set()
    cur = head_id
    while cur is not None:
        if cur in seen:
            raise ChainError("snapshot chain has a cycle",
                             details={"snapshot_id": cur})
        seen.add(cur)
        snap = snaps.get(cur)
        if snap is None:
            raise ChainError("snapshot parent missing",
                             details={"snapshot_id": cur})
        ordered.append(snap)
        parent = snap["parent_id"]
        if parent is not None and parent not in snaps:
            raise ChainError("broken snapshot chain: parent missing",
                             shard_id=snap["shard_id"],
                             details={"snapshot_id": cur, "parent_id": parent})
        cur = parent
    return ordered


def replay_from_snapshot(ledger, spec, head_id):
    """Replay shard states along the chain; every declared shard must appear.

    Returns {shard_id: state} sorted-independent (dict keyed by shard id).
    Raises E_CHAIN naming the broken snapshot and the missing shard.
    """
    ordered = check_chain(ledger, head_id)
    states = {}
    for snap in ordered:
        sid = snap["shard_id"]
        if sid not in states:
            states[sid] = {
                "shard_id": sid,
                "index_pos_lo": snap["index_pos_lo"],
                "index_pos_hi": snap["index_pos_hi"],
                "lease_expiry_pos": snap["lease_expiry_pos"],
                "content_digest": snap["content_digest"],
            }
    for sid in spec.shard_ids():
        if sid not in states:
            last = ordered[-1] if ordered else None
            raise ChainError(
                "snapshot chain misses declared shard",
                shard_id=sid,
                details={
                    "snapshot_id": last["snapshot_id"] if last else head_id,
                    "missing_shard_id": sid,
                })
    return states


def shard_state_digest(entries, leases):
    """Content digest of one shard's index entries plus lease states."""
    return content_digest({
        "entries": sorted(
            (e["rule"], e["key_digest"], e["position"], e["index_version"])
            for e in entries),
        "leases": sorted(
            (l["lease_id"], l["version"], l["state"]) for l in leases),
    })
