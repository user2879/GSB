from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import SNAPSHOT_FORMAT_VERSION
from .utils import atomic_write_json, read_json


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.releases = self.root / "releases"
        self.current_pointer = self.root / "current.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.releases.mkdir(parents=True, exist_ok=True)

    def load_pointer(self) -> dict[str, Any] | None:
        if not self.current_pointer.exists():
            releases = sorted(self.releases.glob("snap-*"))
            if not releases:
                return None
            latest = releases[-1] / "manifest.json"
            if not latest.exists():
                return None
            manifest = read_json(latest)
            atomic_write_json(self.current_pointer, {"release": latest.parent.name, "manifest": manifest})
            return {"release": latest.parent.name, "manifest": manifest}
        return read_json(self.current_pointer)

    def load_compatible_state(self, config_hash: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        pointer = self.load_pointer()
        if not pointer:
            return None, None, "no_committed_snapshot"
        release_dir = self.releases / pointer["release"]
        manifest_path = release_dir / "manifest.json"
        state_path = release_dir / "state.json"
        if not manifest_path.exists() or not state_path.exists():
            return None, None, "last_snapshot_incomplete"
        manifest = read_json(manifest_path)
        if manifest.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
            return None, manifest, "snapshot_format_version_changed"
        if manifest.get("config_hash") != config_hash:
            return None, manifest, "config_hash_changed"
        return read_json(state_path), manifest, None

    def commit(
        self,
        *,
        state: dict[str, Any],
        ledger: dict[str, Any],
        config_hash: str,
        command_cursor: int,
        global_version: int,
    ) -> dict[str, Any]:
        lineage = config_hash[:12]
        snapshot_id = f"snap-{lineage}-{global_version:06d}-cursor-{command_cursor:04d}"
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            from .utils import write_json

            write_json(staging / "state.json", state)
            write_json(staging / "ledger.json", ledger)
            write_json(staging / "01_parse.json", {
                "receipt_counts": ledger["receipt_counts"],
                "receipts": ledger["receipts"],
                "sidecars": ledger["sidecars"],
                "truncated_fields": ledger["r3_to_r1_truncated_fields"],
            })
            write_json(staging / "sidecars.json", {
                "sidecar_version": "receipt-sidecar.v1",
                "sidecars": ledger["sidecars"],
                "truncated_fields": ledger["r3_to_r1_truncated_fields"],
            })
            write_json(staging / "02_attribution.json", {
                "attributions": ledger["attributions"],
                "decision_log": [
                    {
                        "request_id": item["request_id"],
                        "status": item["status"],
                        "decision_basis": item["decision_basis"],
                        "candidate_receipt_uids": item["candidate_receipt_uids"],
                    }
                    for item in ledger["attributions"]
                ],
            })
            write_json(staging / "03_resources.json", {
                "batch_metrics": ledger["batch_metrics"],
                "worker_version_metrics": ledger["worker_version_metrics"],
                "latency_statistics": ledger["latency_statistics"],
            })
            write_json(staging / "04_summary.json", {
                "summary": ledger["summary"],
                "business_code_counts": ledger["business_code_counts"],
                "pending_queue": ledger["pending_queue"],
                "cycles": ledger["cycles"],
            })
            artifact_versions = {
                "state": 1,
                "01_parse": 1,
                "02_attribution": 1,
                "03_resources": 1,
                "04_summary": 1,
                "ledger": 1,
            }
            manifest = {
                "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
                "global_version": global_version,
                "snapshot_id": snapshot_id,
                "config_hash": config_hash,
                "completed_command_cursor": command_cursor,
                "artifact_versions": artifact_versions,
                "files": sorted(name for name in os.listdir(staging) if name.endswith(".json")),
            }
            write_json(staging / "manifest.json", manifest)

            final_dir = self.releases / snapshot_id
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(staging, final_dir)
            pointer = {"release": snapshot_id, "manifest_path": "manifest.json", "global_version": global_version}
            atomic_write_json(self.current_pointer, pointer)
            return manifest
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def clean_staging_directories(self) -> list[str]:
        removed = []
        for path in self.root.glob(".staging-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
        return removed


def compare_ledger_files(left: str | Path, right: str | Path) -> list[str]:
    left_data = read_json(Path(left))
    right_data = read_json(Path(right))
    return diff_objects(left_data, right_data)


def diff_objects(left: Any, right: Any, path: str = "$") -> list[str]:
    differences: list[str] = []
    if type(left) is not type(right):
        differences.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
        return differences
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                differences.append(f"{child}: missing in left")
            elif key not in right:
                differences.append(f"{child}: missing in right")
            else:
                differences.extend(diff_objects(left[key], right[key], child))
    elif isinstance(left, list):
        if len(left) != len(right):
            differences.append(f"{path}: list length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(diff_objects(left_item, right_item, f"{path}[{index}]"))
    elif left != right:
        differences.append(f"{path}: {left!r} != {right!r}")
    return differences
