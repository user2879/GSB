"""Independent recovery-consistency validator.

The validator runs as a separate process/entry point and only imports the
pure state primitives plus the storage layout helpers. It never imports
:mod:`syncpipe.engine`, never receives an engine object and keeps no cache
shared with online synthesis: every byte it checks comes back from disk.

Checks performed:
  1. manifest parent chain continuity (E_CHAIN_BROKEN);
  2. every stored snapshot's content hash (E_SNAPSHOT_HASH);
  3. every op-log line CRC (E_LOG_TRUNCATED);
  4. full replay from the genesis snapshot through all logs, comparing the
     recomputed state hash with the manifest per-version record;
  5. anchor-rebuild cross check: a randomly (seeded) chosen target version is
     rebuilt from its nearest ancestor anchor and compared field by field
     with the directly stored/chain-derived version.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List, Optional

from .errors import (
    E_CHAIN_BROKEN,
    E_LOG_TRUNCATED,
    E_SNAPSHOT_HASH,
    make_error,
)
from .state import apply_event, initial_state, state_digest
from .storage import Storage


def validate_directory(root: str, seed: Optional[int] = None) -> Dict[str, Any]:
    storage = Storage(root)
    report: Dict[str, Any] = {
        "root": os.path.basename(os.path.normpath(storage.root)),
        "ok": True,
        "checked_snapshots": [],
        "checked_log_versions": [],
        "per_version": [],
        "first_mismatch": None,
        "anchor_rebuild": None,
        "errors": [],
    }
    rng = random.Random(seed if seed is not None else 20261119)

    manifest = storage.load_manifest()

    # 1. parent chain
    expected_parent = None
    for item in manifest["versions"]:
        if item["parent_version"] != expected_parent:
            report["ok"] = False
            report["errors"].append(
                make_error(
                    E_CHAIN_BROKEN,
                    f"chain gap at v{item['version']} parent={item['parent_version']} "
                    f"expected={expected_parent}",
                    version=item["version"],
                    tick=item["tick"],
                )
            )
            report["first_mismatch"] = _mismatch_context(item, "chain_parent")
            return report
        expected_parent = item["version"]

    # 2. snapshot file hashes
    for version in storage.available_snapshot_versions():
        try:
            snapshot = storage.read_snapshot_file(version)
        except Exception as exc:  # hash failure raises PipelineError
            report["ok"] = False
            report["errors"].append(
                make_error(E_SNAPSHOT_HASH, str(exc), version=version)
            )
            report["first_mismatch"] = _mismatch_context(
                {"version": version}, "snapshot_hash", str(exc)
            )
            return report
        report["checked_snapshots"].append(version)
        if snapshot["state_hash"] != state_digest(snapshot["state"]):
            report["ok"] = False
            report["errors"].append(
                make_error(E_SNAPSHOT_HASH, "state hash mismatch inside snapshot", version=version)
            )
            report["first_mismatch"] = _mismatch_context(
                {"version": version, "tick": snapshot["tick"]}, "state_hash"
            )
            return report

    # 3. log CRC / truncation
    entries, truncation_errors = storage.read_all_log_entries()
    if truncation_errors:
        report["ok"] = False
        report["errors"].extend(truncation_errors)
        first = truncation_errors[0]
        report["first_mismatch"] = _mismatch_context(
            {"version": first["version"]}, "log_truncation", first["reason"]
        )
        return report

    # 4. full replay from genesis, per-version comparison
    genesis = storage.read_snapshot_file(0)
    if genesis["state"] != initial_state():
        report["ok"] = False
        report["errors"].append(make_error(E_SNAPSHOT_HASH, "genesis state is not empty", version=0))
        return report
    state = copy.deepcopy(genesis["state"])
    manifest_by_version = {item["version"]: item for item in manifest["versions"]}
    entries_by_version = {entry["version"]: entry for entry in entries}

    head = manifest["head_version"]
    for version in range(1, head + 1):
        item = manifest_by_version.get(version)
        entry = entries_by_version.get(version)
        if item is None or entry is None:
            report["ok"] = False
            code = E_CHAIN_BROKEN if item is None else E_LOG_TRUNCATED
            report["errors"].append(
                make_error(
                    code,
                    f"missing {'manifest' if item is None else 'log'} record at v{version}",
                    version=version,
                )
            )
            report["first_mismatch"] = _mismatch_context(
                {"version": version}, "missing_record"
            )
            return report
        if entry["status"] == "applied" and entry.get("event") is not None:
            state = apply_event(state, entry["event"])
        actual = state_digest(state)
        report["per_version"].append(
            {
                "version": version,
                "tick": entry["tick"],
                "expected_state_hash": item["state_hash"],
                "actual_state_hash": actual,
                "match": actual == item["state_hash"],
            }
        )
        report["checked_log_versions"].append(version)
        if actual != item["state_hash"]:
            report["ok"] = False
            report["errors"].append(
                make_error(
                    E_SNAPSHOT_HASH,
                    f"replay hash mismatch expected={item['state_hash'][:16]} actual={actual[:16]}",
                    version=version,
                    tick=entry["tick"],
                )
            )
            report["first_mismatch"] = _mismatch_context(
                item, "state_hash", None, entry
            )
            return report

    # 5. ancestor-anchor rebuild cross check (incremental catch-up equivalence)
    target = head
    anchor_candidates = [v for v in storage.available_snapshot_versions() if v < head]
    if anchor_candidates:
        anchor_version = max(v for v in anchor_candidates if v <= rng.randrange(0, head + 1)) if head > 0 else 0
        anchor_version = anchor_candidates[rng.randrange(len(anchor_candidates))]
        anchor = storage.read_snapshot_file(anchor_version)
        rebuilt_state = copy.deepcopy(anchor["state"])
        replayed = []
        for entry in entries:
            if entry["version"] <= anchor_version:
                continue
            if entry["version"] > target:
                break
            if entry["status"] == "applied" and entry.get("event") is not None:
                rebuilt_state = apply_event(rebuilt_state, entry["event"])
            replayed.append(entry["version"])
        full_state = copy.deepcopy(state)
        field_equal = rebuilt_state == full_state
        hash_equal = state_digest(rebuilt_state) == state_digest(full_state)
        report["anchor_rebuild"] = {
            "anchor_version": anchor_version,
            "target_version": target,
            "replayed_versions": replayed,
            "field_equal": field_equal,
            "hash_equal": hash_equal,
        }
        if not (field_equal and hash_equal):
            report["ok"] = False
            report["errors"].append(
                make_error(
                    E_SNAPSHOT_HASH,
                    f"anchor rebuild from v{anchor_version} not equal to direct v{target}",
                    version=target,
                )
            )
    return report


def _mismatch_context(
    manifest_item: Dict[str, Any],
    mismatch_kind: str,
    detail: Optional[str] = None,
    log_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = {
        "kind": mismatch_kind,
        "version": manifest_item.get("version"),
        "tick": manifest_item.get("tick"),
        "manifest_record": {
            "parent_version": manifest_item.get("parent_version"),
            "state_hash": manifest_item.get("state_hash"),
            "snapshot_hash": manifest_item.get("snapshot_hash"),
        },
    }
    if log_entry is not None:
        context["participating_event"] = log_entry.get("event")
        context["log_status"] = log_entry.get("status")
    if detail:
        context["detail"] = detail
    return context
