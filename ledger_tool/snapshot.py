"""会话快照: 版本识别、恢复与重放校验."""

import glob
import json
import os

from .canon import canonical_json, digest_of
from .errors import E_REPLAY, E_SNAPSHOT, LedgerError
from .normalize import order_views, view_digest

SNAPSHOT_VERSION = 1


def _snapshot_paths(snap_dir):
    return sorted(glob.glob(os.path.join(str(snap_dir), "snapshot-*.json")))


def write_snapshot(snap_dir, ledger, views, trail, cursor):
    os.makedirs(snap_dir, exist_ok=True)
    idx = len(_snapshot_paths(snap_dir)) + 1
    ordered = order_views([dict(v) for v in views])
    snap = {
        "snapshot_version": SNAPSHOT_VERSION,
        "format_versions": ledger.formats(),
        "content_digest": ledger.content_digest(),
        "view_digest": view_digest(ordered),
        "record_digests": {v["request_id"]: digest_of(v) for v in ordered},
        "upgrade_trail": list(trail),
        "cursor": int(cursor),
    }
    path = os.path.join(str(snap_dir), f"snapshot-{idx:04d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(snap))
    return path, snap


def load_latest(snap_dir):
    paths = _snapshot_paths(snap_dir)
    if not paths:
        raise LedgerError(E_SNAPSHOT, "no snapshot found", stage="snapshot")
    with open(paths[-1], "r", encoding="utf-8") as fh:
        return paths[-1], json.load(fh)


def list_snapshots(snap_dir):
    return _snapshot_paths(snap_dir)


def verify_snapshot(snap, ledger):
    """比较快照的版本集合与内容摘要同当前账本是否一致."""
    diffs = []
    if snap.get("format_versions") != ledger.formats():
        diffs.append("format_versions")
    if snap.get("content_digest") != ledger.content_digest():
        diffs.append("content_digest")
    if diffs:
        raise LedgerError(E_SNAPSHOT,
                          f"snapshot stale, wholesale replace required: {diffs}",
                          stage="snapshot", extra={"diffs": diffs})
    return True


def replay_snapshot(snap, pipeline):
    """按轨迹重放归一与推理, 逐项比对记录摘要."""
    views = pipeline.ordered_views()
    record_digests = snap.get("record_digests", {})
    for view in views:
        rid = view["request_id"]
        if record_digests.get(rid) != digest_of(view):
            raise LedgerError(E_REPLAY,
                              f"replay mismatch at record {rid}",
                              record_id=rid, stage="replay")
    if len(record_digests) != len(views):
        missing = sorted(set(record_digests) - {v["request_id"] for v in views})
        raise LedgerError(E_REPLAY, "replay record count mismatch",
                          record_id=missing[0] if missing else None, stage="replay")
    return len(views)
