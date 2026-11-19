"""Crash recovery: rebuild the latest consistent state from durable files.

The recovery path never consults an in-memory engine: it only reads the
persisted manifest, snapshots and op logs, exactly like a fresh process
started after ``SIGKILL`` / TerminateProcess. A missing shutdown marker is
treated as a forced termination and no exit report is fabricated.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

from .errors import E_CHAIN_BROKEN, E_LOG_TRUNCATED, E_SNAPSHOT_HASH, make_error
from .state import apply_event, build_snapshot, canonical_bytes, state_digest
from .storage import Storage


def recover(storage: Storage) -> Dict[str, Any]:
    storage.ensure_dirs()
    # Atomic writes land in *.tmp siblings and rename into place; a process
    # killed mid-write leaves an unreferenced .tmp that restart cleans up.
    for directory, _sub, files in os.walk(storage.root):
        for name in files:
            if name.endswith(".tmp"):
                os.remove(os.path.join(directory, name))
    report: Dict[str, Any] = {
        "clean_shutdown": storage.has_shutdown_marker(),
        "forced_termination": not storage.has_shutdown_marker(),
        "truncated": False,
        "truncation_errors": [],
        "replayed_versions": [],
        "anchor_version": None,
        "recovered_version": 0,
        "recovered_state_hash": None,
        "quarantined_files": [],
        "errors": [],
    }

    manifest = storage.load_manifest()
    storage.verify_parent_chain(manifest)
    committed_version = manifest["head_version"]

    entries, truncation_errors = storage.read_all_log_entries()
    report["truncation_errors"] = truncation_errors
    report["errors"].extend(truncation_errors)
    if truncation_errors:
        report["truncated"] = True
        quarantine_dir = os.path.join(storage.root, "quarantine")
        if os.path.isdir(quarantine_dir):
            report["quarantined_files"] = sorted(
                os.path.relpath(os.path.join(quarantine_dir, name), storage.root)
                for name in os.listdir(quarantine_dir)
            )
        else:
            report["quarantined_files"] = []

    durable_versions = {entry["version"] for entry in entries}
    anchors = [v for v in storage.available_snapshot_versions() if v <= committed_version]
    if not anchors:
        error = make_error(
            E_CHAIN_BROKEN,
            "no usable snapshot found during recovery",
            version=committed_version,
        )
        report["errors"].append(error)
        return report
    anchor_version = max(anchors)
    report["anchor_version"] = anchor_version
    snapshot = storage.read_snapshot_file(anchor_version)
    if snapshot["state_hash"] != state_digest(snapshot["state"]):
        raise RuntimeError("unreachable: snapshot digest already verified on read")
    state = copy.deepcopy(snapshot["state"])

    entries = [entry for entry in entries if anchor_version < entry["version"] <= committed_version]
    for entry in entries:
        if entry["version"] not in durable_versions:
            error = make_error(
                E_LOG_TRUNCATED,
                f"missing log entry for v{entry['version']}",
                version=entry["version"],
            )
            report["errors"].append(error)
            report["truncated"] = True
            break
        if entry["status"] == "applied" and entry.get("event") is not None:
            state = apply_event(state, entry["event"])
        actual_hash = state_digest(state)
        if actual_hash != entry["state_hash"]:
            error = make_error(
                E_SNAPSHOT_HASH,
                f"replay produced {actual_hash[:12]} but log records {entry['state_hash'][:12]}",
                version=entry["version"],
                tick=entry["tick"],
            )
            report["errors"].append(error)
            break
        report["replayed_versions"].append(entry["version"])

    recovered_version = anchor_version + len(report["replayed_versions"])
    report["recovered_version"] = recovered_version
    report["recovered_state_hash"] = state_digest(state)

    expected = next(item for item in manifest["versions"] if item["version"] == recovered_version)
    if expected["state_hash"] != report["recovered_state_hash"]:
        report["errors"].append(
            make_error(
                E_SNAPSHOT_HASH,
                "recovered state hash disagrees with manifest record",
                version=recovered_version,
            )
        )

    # A recovered snapshot of the latest consistent version is written for
    # subsequent runs, but the shutdown marker is never created here.
    recovered_snapshot = build_snapshot(
        version=recovered_version,
        tick=expected["tick"],
        parent_version=recovered_version - 1,
        state=state,
        source_event=entries[-1]["event"] if entries else snapshot.get("source_event"),
    )
    if recovered_snapshot["snapshot_hash"] != expected["snapshot_hash"]:
        report["errors"].append(
            make_error(
                E_SNAPSHOT_HASH,
                "rebuilt snapshot hash disagrees with manifest record",
                version=recovered_version,
            )
        )
    recovered_dir = os.path.join(storage.root, "recovered")
    os.makedirs(recovered_dir, exist_ok=True)
    recovered_path = os.path.join(
        recovered_dir, f"recovered_v{recovered_version:06d}.json"
    )
    with open(recovered_path, "wb") as handle:
        handle.write(canonical_bytes(recovered_snapshot) + b"\n")
    report["recovered_snapshot_file"] = os.path.relpath(recovered_path, storage.root)
    return report


def replay_tail_from_zero(storage: Storage) -> List[Dict[str, Any]]:
    """Convenience helper used by tests: replay the entire log from genesis."""
    snapshot = storage.read_snapshot_file(0)
    state = copy.deepcopy(snapshot["state"])
    entries, _ = storage.read_all_log_entries()
    out = []
    for entry in entries:
        if entry["status"] == "applied" and entry.get("event") is not None:
            state = apply_event(state, entry["event"])
        out.append({"version": entry["version"], "state_hash": state_digest(state)})
    return out
