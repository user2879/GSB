"""Atomic persistence, stable-state recovery and orphan reclamation.

Commit protocol (write-unfinished == uncommitted):
  1. every file is written to a unique ``*.tmp.<pid>`` name in the same
     directory and flushed+fsynced;
  2. publication is a single ``os.replace`` (atomic rename on NTFS and POSIX);
  3. the ledger is always committed *after* the frame files it references, so
     a crash can never leave a committed ledger pointing at missing data.

Restart policy:
  * the ledger is the single source of truth of the last committed state;
  * files not referenced by it (including leftover ``*.tmp`` files) are
    orphans: moved to ``orphans/`` and recorded in the recovery log;
  * committed frame files whose hash disagrees with the ledger are also
    quarantined and marked for recomputation;
  * a snapshot older than the committed ledger raises E_SNAPSHOT_STALE.
"""

import json
import os
import shutil
import tempfile

from .config import LEDGER_SCHEMA_VERSION
from .errors import SnapshotStaleError
from .hashing import canonical_json, hash_bytes, hash_json

LEDGER_NAME = "ledger.json"
CLOSED_NAME = "CLOSED"
RECOVERY_NAME = "recovery_log.json"
TMP_SUFFIX = ".tmp"


def atomic_write(path, data, *, binary=False):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    mode = "wb" if binary else "w"
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".",
                               suffix=TMP_SUFFIX, dir=directory)
    try:
        with os.fdopen(fd, mode) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path, obj):
    atomic_write(path, canonical_json(obj), binary=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_ledger(work_dir, ledger):
    ledger["ledger_schema_version"] = LEDGER_SCHEMA_VERSION
    ledger["ledger_hash"] = hash_json(
        {k: v for k, v in ledger.items() if k != "ledger_hash"}
    )
    atomic_write_json(os.path.join(work_dir, LEDGER_NAME), ledger)


def read_ledger(work_dir):
    return read_json(os.path.join(work_dir, LEDGER_NAME))


def ledger_exists(work_dir):
    return os.path.exists(os.path.join(work_dir, LEDGER_NAME))


def write_closed_marker(work_dir, ledger):
    payload = {
        "committed_seq": ledger["committed_seq"],
        "ledger_hash": ledger["ledger_hash"],
    }
    atomic_write_json(os.path.join(work_dir, CLOSED_NAME), payload)


def is_closed(work_dir):
    return os.path.exists(os.path.join(work_dir, CLOSED_NAME))


def _safe_name(path):
    flat = path.replace(os.sep, "_").lstrip("_")
    return flat or "orphan"


def reclaim_orphans(work_dir, referenced_files):
    """Move unreferenced files and leftover temp files into orphans/."""
    orphan_dir = os.path.join(work_dir, "orphans")
    referenced = set(referenced_files)
    reclaimed = []
    for root, _dirs, files in os.walk(work_dir):
        if os.path.abspath(root).startswith(os.path.abspath(orphan_dir)):
            continue
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, work_dir)
            is_tmp = fname.endswith(TMP_SUFFIX) or TMP_SUFFIX + "." in fname
            if is_tmp or rel not in referenced:
                os.makedirs(orphan_dir, exist_ok=True)
                dest = os.path.join(orphan_dir, _safe_name(rel))
                base, n = dest, 1
                while os.path.exists(dest):
                    dest = f"{base}.{n}"
                    n += 1
                shutil.move(full, dest)
                reclaimed.append({"original": rel, "quarantined":
                                  os.path.relpath(dest, work_dir),
                                  "reason": "temp" if is_tmp else "unreferenced"})
    # prune empty directories left behind (except the well-known ones)
    keep = {"frames", "frames_out", "injections", "orphans"}
    for root, dirs, files in os.walk(work_dir, topdown=False):
        if root == work_dir:
            continue
        if not os.listdir(root) and os.path.basename(root) not in keep:
            os.rmdir(root)
    return reclaimed


def append_recovery_log(work_dir, entries):
    path = os.path.join(work_dir, RECOVERY_NAME)
    log = read_json(path) if os.path.exists(path) else {"events": []}
    log["events"].extend(entries)
    atomic_write_json(path, log)


def recover(work_dir):
    """Load the last committed ledger and reclaim everything uncommitted.

    Returns (ledger, recovery_info). Raises SnapshotStaleError only when a
    caller-supplied external snapshot is checked separately via
    ``assert_snapshot_fresh``.
    """
    ledger = read_ledger(work_dir)
    referenced = {
        LEDGER_NAME, RECOVERY_NAME, "truth.json", CLOSED_NAME,
    }
    referenced = {p.replace("/", os.sep) for p in referenced}
    for entry in ledger.get("outputs", {}).values():
        referenced.add(entry["file"].replace("/", os.sep))
    for rec in ledger.get("injection_files", []):
        referenced.add(rec.replace("/", os.sep))
    # immutable generated inputs are always protected from reclamation
    frame_count = ledger.get("frame_count", 0)
    for idx in range(frame_count):
        referenced.add(os.path.join("frames", "frame_%04d.fp16" % idx))
    reclaimed = reclaim_orphans(work_dir, referenced)
    info = {"reclaimed": reclaimed, "committed_seq": ledger["committed_seq"]}
    if reclaimed:
        append_recovery_log(
            work_dir,
            [{"event": "orphan_reclaim", "items": reclaimed}],
        )
    # A CLOSED marker from a previous run is stale once we resume work.
    marker = os.path.join(work_dir, CLOSED_NAME)
    if os.path.exists(marker):
        os.unlink(marker)
    return ledger, info


def assert_snapshot_fresh(snapshot_seq, snapshot_hash, ledger):
    """Reject external snapshots older than the committed ledger."""
    if int(snapshot_seq) < int(ledger["committed_seq"]):
        raise SnapshotStaleError(
            f"snapshot seq {snapshot_seq} is older than committed "
            f"seq {ledger['committed_seq']}",
            details={"snapshot_seq": snapshot_seq,
                     "committed_seq": ledger["committed_seq"]},
        )
    if snapshot_hash is not None and snapshot_hash != ledger["ledger_hash"]:
        raise SnapshotStaleError(
            "snapshot content hash disagrees with committed ledger",
            details={"snapshot_hash": snapshot_hash,
                     "ledger_hash": ledger["ledger_hash"]},
        )


def file_sha256(path):
    with open(path, "rb") as fh:
        return hash_bytes(fh.read())
