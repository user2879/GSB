"""Session-snapshot recovery with version-identified wholesale replacement.

Reopening compares the session snapshot's integer format version and
content digest against the on-disk ledger. Any mismatch is reported as
E_SNAPSHOT and the index entries, lease set and conclusions are replaced
wholesale (never merged incrementally); the differing items are listed in
a stable sorted order. After a forced termination only fully committed
batches are authoritative: declared shards without a committed progress
cursor are marked unknown and must be rerun.
"""

import json
from pathlib import Path

from .errors import SnapshotError
from .ledger import FORMAT_VERSION, Ledger


def write_session_snapshot(ledger, path):
    state = ledger.dump_state()
    doc = {
        "format_version": FORMAT_VERSION,
        "content_digest": ledger.compute_digest(),
        "state": state,
    }
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True),
                          encoding="utf-8")
    return doc


def load_session_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_key(table, row):
    if table == "index_entries":
        return "%s|%s|%s" % (row["shard_id"], row["rule"], row["key_digest"])
    if table == "leases":
        return row["lease_id"]
    if table == "snapshots":
        return row["snapshot_id"]
    if table in ("conclusions", "progress"):
        return row["shard_id"]
    return json.dumps(row, sort_keys=True)


def diff_states(disk_state, session_state):
    """Stable sorted list of differing items between two state dumps."""
    diffs = []
    for table in sorted(set(disk_state) | set(session_state)):
        disk = {_row_key(table, r): r for r in disk_state.get(table, [])}
        sess = {_row_key(table, r): r for r in session_state.get(table, [])}
        for key in sorted(set(disk) | set(sess)):
            if key not in disk:
                diffs.append("%s:%s:only-in-session" % (table, key))
            elif key not in sess:
                diffs.append("%s:%s:only-on-disk" % (table, key))
            elif disk[key] != sess[key]:
                diffs.append("%s:%s:content-differs" % (table, key))
    return diffs


def recover(spec, ledger, session_path, logger=None):
    """Compare session snapshot with disk; replace wholesale on mismatch.

    Returns a report dict: {replaced, diffs, unknown_shards, replay_ok}.
    """
    session = load_session_snapshot(session_path)
    disk_version = ledger.format_version()
    disk_digest = ledger.compute_digest()
    replaced = False
    diffs = []
    if (session.get("format_version") != disk_version
            or session.get("content_digest") != disk_digest):
        err = SnapshotError(
            "session snapshot version/digest mismatch with ledger",
            details={"session_version": session.get("format_version"),
                     "disk_version": disk_version,
                     "session_digest": session.get("content_digest"),
                     "disk_digest": disk_digest})
        if logger:
            logger.emit_error(err, event="snapshot_mismatch")
        diffs = diff_states(ledger.dump_state(), session.get("state", {}))
        ledger.replace_all(session.get("state", {}))
        replaced = True
        if logger:
            logger.emit("ledger_replaced", code="E_SNAPSHOT", diffs=diffs)

    committed = set(ledger.committed_shards())
    unknown = [sid for sid in spec.shard_ids() if sid not in committed]
    if unknown:
        ledger.commit_batch(
            _next_batch_id(ledger),
            progress=[{"shard_id": sid, "state": "unknown"} for sid in unknown])
        if logger:
            for sid in unknown:
                logger.emit("shard_unknown", shard_id=sid, code="E_SNAPSHOT",
                            reason="no committed batch; rerun required")
    return {"replaced": replaced, "diffs": diffs, "unknown_shards": unknown}


def _next_batch_id(ledger):
    rows = ledger.query("SELECT COALESCE(MAX(batch_id), 0) AS m FROM progress")
    return int(rows[0]["m"]) + 1
