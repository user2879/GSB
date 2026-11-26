from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .hashing import canonical_hash
from .store import SnapshotStore


def simulated_recovery(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(root)

    txn_v1 = store.new_transaction()
    for phase in SnapshotStore.PHASES:
        store.write_phase(txn_v1, phase, {"phase": phase, "version": 1})
    pointer_v1 = store.commit(
        txn_v1,
        global_version=1,
        config_hash="cfg_v1",
        state_hash=canonical_hash({"v": 1}),
        expected_global_version=0,
    )

    incomplete = store.new_transaction()
    store.write_phase(incomplete, "01_validation", {"partial": True})

    prepared = store.new_transaction()
    for phase in SnapshotStore.PHASES:
        store.write_phase(prepared, phase, {"phase": phase, "version": 2})
    manifest = {
        "status": "prepared",
        "global_version": 2,
        "config_hash": "cfg_v2",
        "state_hash": canonical_hash({"v": 2}),
        "phases": list(SnapshotStore.PHASES),
    }
    store.write_phase(prepared, "manifest", manifest)

    recovery = store.recover(promote_prepared=True, clean_incomplete=True)
    shutdown = store.shutdown_status()
    current = store.current_pointer()
    return {
        "committed_v1_pointer": pointer_v1,
        "recovery_result": recovery,
        "current_pointer_after_recovery": current,
        "shutdown": shutdown,
        "recovered_to_v2": bool(current and current.get("global_version") == 2),
        "incomplete_cleaned": recovery["staging_residue_count"] == 0,
    }
