"""Snapshot recovery: resume an interrupted run from the last committed point.

The manifest records a global version and per-stage artifact versions.  On
resume we:

1. verify definition/config hashes against the current inputs (E_SNAPSHOT_STALE),
2. reload the persisted feature observations and budget ledger,
3. continue the point stream strictly after the recorded cursor.
"""
from __future__ import annotations

import os
from typing import Any

from .common import (
    CODE_SNAPSHOT_STALE,
    Failure,
    read_json,
    read_jsonl,
    sha256_text,
    canonical_dumps,
)
from .snapshot import SnapshotStore


class RecoveryError(Exception):
    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.detail)
        self.failure = failure


def verify_manifest(snapshot: SnapshotStore, expected_def_hash: str,
                    expected_cfg_hash: str) -> dict[str, Any]:
    manifest = snapshot.load_manifest()
    if manifest is None:
        raise RecoveryError(Failure("snapshot", "manifest", CODE_SNAPSHOT_STALE,
                                    "no snapshot manifest found"))
    if manifest["definition_hash"] != expected_def_hash:
        raise RecoveryError(Failure(
            "snapshot", "definition_hash", CODE_SNAPSHOT_STALE,
            "definition changed while pipeline was stopped"))
    if manifest["config_hash"] != expected_cfg_hash:
        raise RecoveryError(Failure(
            "snapshot", "config_hash", CODE_SNAPSHOT_STALE,
            "config changed while pipeline was stopped"))
    return manifest


def resume_state(snapshot: SnapshotStore) -> dict[str, Any]:
    """Reload persisted state needed to continue after the cursor."""
    feature_path = os.path.join(snapshot.stage_dir, "feature", "features.jsonl")
    ledger_path = os.path.join(snapshot.stage_dir, "feature", "budget_ledger.jsonl")
    state: dict[str, Any] = {"observations": {}, "ledger": [], "cursor_seq": -1}
    if os.path.exists(feature_path):
        for row in read_jsonl(feature_path):
            state["observations"].setdefault(
                (row["feature"], row["entity_id"]), []).append(
                (row["anchor_ts"], row["point_seq"], row["value"], row["empty"]))
            state["cursor_seq"] = max(state["cursor_seq"], row["point_seq"])
    if os.path.exists(ledger_path):
        state["ledger"] = read_jsonl(ledger_path)
    return state
