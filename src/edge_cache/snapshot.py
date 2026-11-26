"""Staged atomic snapshots, version identification and crash recovery.

Every stage (configuration, lease algebra, receipts, conclusions, ledger) is
first written to a unique temporary directory and then committed with a single
``os.replace`` of a versioned manifest.  Recovery therefore trusts only fully
committed snapshots: a torn temp directory left behind by ``SIGKILL`` is
ignored, and the diagnostic report names the missing stages and unfinished
shards.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

from .errors import E_SNAPSHOT_VERSION, EdgeCacheError, Failure
from .ledger import Ledger

STAGES = ("config", "leases", "receipts", "conclusions", "ledger")
MANIFEST_NAME = "manifest.json"


@dataclass
class SnapshotHandle:
    snapshot_id: str
    path: str
    stages: set[str] = field(default_factory=set)


class SnapshotStore:
    def __init__(self, root: str) -> None:
        self.root = root
        self.tmp_root = os.path.join(root, "tmp")
        self.committed_root = os.path.join(root, "committed")
        os.makedirs(self.tmp_root, exist_ok=True)
        os.makedirs(self.committed_root, exist_ok=True)

    def begin(self, snapshot_id: str) -> SnapshotHandle:
        path = os.path.join(self.tmp_root, snapshot_id)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return SnapshotHandle(snapshot_id=snapshot_id, path=path)

    def write_stage(self, handle: SnapshotHandle, stage: str, payload: dict[str, Any]) -> None:
        if stage not in STAGES:
            raise EdgeCacheError(Failure(E_SNAPSHOT_VERSION, "unknown snapshot stage",
                                         {"stage": stage}))
        path = os.path.join(handle.path, stage + ".json")
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, path)
        handle.stages.add(stage)

    def commit(self, handle: SnapshotHandle, global_version: int,
               config_hash: str, cursor: dict[str, Any]) -> str:
        if not handle.stages:
            raise EdgeCacheError(Failure(E_SNAPSHOT_VERSION, "refusing empty snapshot"))
        manifest = {
            "snapshot_id": handle.snapshot_id,
            "global_version": global_version,
            "config_hash": config_hash,
            "stages": sorted(handle.stages),
            "cursor": cursor,
        }
        manifest_path = os.path.join(handle.path, MANIFEST_NAME)
        tmp = manifest_path + ".part"
        with open(tmp, "w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, manifest_path)
        target = os.path.join(self.committed_root, handle.snapshot_id)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.replace(handle.path, target)
        self._set_current(handle.snapshot_id)
        return target

    def _set_current(self, snapshot_id: str) -> None:
        pointer = os.path.join(self.root, "CURRENT")
        tmp = pointer + ".part"
        with open(tmp, "w", encoding="utf-8") as file_obj:
            file_obj.write(snapshot_id)
        os.replace(tmp, pointer)

    def current_id(self) -> str | None:
        pointer = os.path.join(self.root, "CURRENT")
        if not os.path.exists(pointer):
            return None
        with open(pointer, "r", encoding="utf-8") as file_obj:
            snapshot_id = file_obj.read().strip()
        target = os.path.join(self.committed_root, snapshot_id)
        if not os.path.isfile(os.path.join(target, MANIFEST_NAME)):
            return None
        return snapshot_id

    def load_current(self, expected_config_hash: str | None = None) -> dict[str, Any]:
        snapshot_id = self.current_id()
        if snapshot_id is None:
            return {"status": "no-snapshot", "stages": [], "missing_stages": list(STAGES)}
        target = os.path.join(self.committed_root, snapshot_id)
        with open(os.path.join(target, MANIFEST_NAME), "r", encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        if expected_config_hash is not None and manifest["config_hash"] != expected_config_hash:
            raise EdgeCacheError(Failure(
                E_SNAPSHOT_VERSION,
                "snapshot config hash differs from current configuration",
                {"snapshot": manifest["config_hash"], "current": expected_config_hash},
            ))
        stages: dict[str, Any] = {}
        for stage in STAGES:
            path = os.path.join(target, stage + ".json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file_obj:
                    stages[stage] = json.load(file_obj)
        missing = [stage for stage in STAGES if stage not in stages]
        return {
            "status": "recovered" if not missing else "partial",
            "snapshot_id": snapshot_id,
            "manifest": manifest,
            "stages": stages,
            "missing_stages": missing,
        }

    def cleanup_temp(self) -> list[str]:
        removed: list[str] = []
        if not os.path.isdir(self.tmp_root):
            return removed
        for name in os.listdir(self.tmp_root):
            path = os.path.join(self.tmp_root, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
                removed.append(name)
        return removed


def build_diagnostic_report(
    recovery: dict[str, Any],
    ledger: Ledger,
    format_counts: dict[str, int],
    node_advances: dict[str, int],
    budget_consumed: int,
) -> dict[str, Any]:
    # Re-read flushed streams so the report is correct even when the in-memory
    # buffers were drained during batch persistence.
    import os

    failure_path = os.path.join(ledger.root, "failures.jsonl")
    failures = Ledger._read_jsonl(failure_path) if os.path.exists(failure_path) else []
    code_counts: dict[str, int] = {}
    for row in failures:
        code_counts[row["code"]] = code_counts.get(row["code"], 0) + 1
    expiry_count = sum(1 for row in failures if row["code"] == "E_LEASE_EXPIRED")
    manifest = recovery.get("manifest", {})
    cursor = manifest.get("cursor", {})
    return {
        "recovery_status": recovery.get("status", "no-snapshot"),
        "snapshot_id": recovery.get("snapshot_id"),
        "missing_stages": recovery.get("missing_stages", list(STAGES)),
        "unfinished_shards": cursor.get("unfinished_shards", []),
        "completed_shards": cursor.get("completed_shards", []),
        "budget_consumed": budget_consumed,
        "format_counts": format_counts,
        "node_generation_advances": node_advances,
        "lease_expiry_reclamations": expiry_count,
        "business_code_counts": code_counts,
        "business_entries": len(ledger.business_rows()),
        "algebra_events": len(Ledger._read_jsonl(os.path.join(ledger.root, "algebra.jsonl"))),
    }


def _pending_failures(ledger: Ledger) -> list[dict[str, Any]]:
    # Failures already flushed to disk are read back through failure_rows();
    # the helper exists so future in-memory callers stay symmetric.
    return []


def write_exit_record(ledger: Ledger, code: str, message: str,
                      detail: dict[str, Any]) -> dict[str, Any]:
    record = {"code": code, "message": message, "detail": detail}
    ledger.add_exit(record)
    ledger.flush_exits()
    return record
