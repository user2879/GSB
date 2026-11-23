"""High-level application orchestration used by the CLI and the demo."""

import os
import shutil
import threading

from .changeset import commit, fingerprint, load_change_set
from .ledger import Ledger
from .model import load_declaration
from .snapshot import compare_with_ledger, load_snapshot, restore_snapshot, save_snapshot
from .verifier import verify_checkpoint

BASELINE_ID = "cp_baseline"


def open_ledger(workdir, create=False):
    db_path = os.path.join(workdir, "ledger.db")
    shard_dir = os.path.join(workdir, "shards")
    return Ledger(db_path, shard_dir, create=create)


def initialize(workdir, declaration_path):
    """Create a fresh ledger and the deterministic baseline checkpoint."""
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)
    decl = load_declaration(declaration_path)
    ledger = open_ledger(workdir, create=True)
    baseline_change = {
        "change_id": BASELINE_ID,
        "ops": [{"op": "train_steps", "count": decl["steps"]}],
    }
    result = commit(ledger, baseline_change, decl, parent_id=None)
    ledger.validate()
    ledger.close()
    return {"workdir": workdir, "baseline": result["checkpoint_id"]}


def submit_change(workdir, declaration_path, change_path, crash_point=None):
    decl = load_declaration(declaration_path)
    change = load_change_set(change_path)
    ledger = open_ledger(workdir)
    ledger.validate()
    parent_id = ledger.latest_checkpoint_id()
    result = commit(ledger, change, decl, parent_id, crash_point=crash_point)
    if not result.get("crash"):
        ledger.validate()
    ledger.close()
    return result


def verify_all(workdir, declaration_path, change_paths):
    decl = load_declaration(declaration_path)
    ledger = open_ledger(workdir)
    ledger.validate()
    changes = [load_change_set(path) for path in change_paths]
    by_id = {fingerprint(change)["change_id"]: change for change in changes}
    by_id[BASELINE_ID] = {
        "change_id": BASELINE_ID,
        "ops": [{"op": "train_steps", "count": decl["steps"]}],
    }
    results = []
    for record in ledger.list_changes():
        cid = record["checkpoint_id"]
        change = by_id.get(cid, record["change_set"])
        verify_checkpoint(ledger, cid, change, decl)
        results.append(cid)
    ledger.close()
    return {"verified": results}


def snapshot_save(workdir, snapshot_dir):
    ledger = open_ledger(workdir)
    path = save_snapshot(ledger, snapshot_dir)
    ledger.close()
    return {"snapshot": path}


def snapshot_status(workdir, snapshot_dir):
    ledger = open_ledger(workdir)
    snapshot = load_snapshot(snapshot_dir)
    diffs = compare_with_ledger(snapshot, ledger)
    ledger.close()
    return {"identical": not diffs, "diffs": diffs}


def snapshot_restore(workdir, snapshot_dir):
    db_path = os.path.join(workdir, "ledger.db")
    shard_dir = os.path.join(workdir, "shards")
    return restore_snapshot(snapshot_dir, db_path, shard_dir)


def concurrent_submit(workdir, declaration_path, change_paths):
    """Serialize distinct change sets by ledger_version using a write lock."""
    decl = load_declaration(declaration_path)
    changes = [load_change_set(path) for path in change_paths]
    outcomes = []
    lock = threading.Lock()

    def worker(change):
        with lock:
            ledger = open_ledger(workdir)
            parent_before = ledger.latest_checkpoint_id()
            result = commit(ledger, change, decl, parent_before)
            ledger.validate()
            ledger.close()
            outcomes.append({
                "change_id": change["change_id"],
                "parent": parent_before,
                "result": result.get("checkpoint_id"),
                "idempotent": result.get("idempotent", False),
            })

    threads = [threading.Thread(target=worker, args=(change,)) for change in changes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def idempotent_submit(workdir, declaration_path, change_path):
    first = submit_change(workdir, declaration_path, change_path)
    second = submit_change(workdir, declaration_path, change_path)
    ledger = open_ledger(workdir)
    count = len(ledger.list_checkpoint_ids())
    ledger.close()
    return {"first": first, "second": second, "checkpoint_count": count}
