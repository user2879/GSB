"""Atomic, revisioned ledger snapshots with crash recovery."""

from __future__ import annotations

import copy
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from .constants import LEDGER_SCHEMA
from .errors import BusinessError
from .util import (
    atomic_write_bytes,
    atomic_write_canonical_json,
    atomic_write_json,
    hash_object,
    json_load,
    remove_tree,
    sha256_file,
)


class LedgerStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.root = self.workspace / "ledger"
        self.snapshots = self.root / "snapshots"
        self.current_path = self.root / "CURRENT.json"
        self.closed_path = self.root / "closed.json"
        self.recovery_path = self.root / "recovery.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.run_dir: Path | None = None
        self.run_id: str | None = None
        self.revision: int | None = None
        self.parent_revision: int | None = None
        self.parent_manifest_hash: str | None = None
        self.shard_hashes: dict[str, str] = {}

    @staticmethod
    def _manifest_hash(manifest: dict[str, Any]) -> str:
        core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        return hash_object(core)

    def begin(self, expected_revision: int | None = None) -> dict[str, Any]:
        latest = self.load_latest_committed()
        current_revision = int(latest["manifest"]["revision"]) if latest else 0
        if expected_revision is not None and int(expected_revision) != current_revision:
            raise BusinessError(
                "E_SNAPSHOT_STALE",
                "expected ledger revision is stale",
                {"expected": int(expected_revision), "current": current_revision},
            )
        revision = current_revision + 1
        run_id = f"rev-{revision:06d}-{uuid.uuid4().hex[:12]}"
        run_dir = self.snapshots / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        self.run_id = run_id
        self.revision = revision
        self.parent_revision = current_revision if latest else None
        self.parent_manifest_hash = latest["manifest_hash"] if latest else None
        self.shard_hashes = {}
        # A new run invalidates the prior normal-exit marker before any work starts.
        self.closed_path.unlink(missing_ok=True)
        atomic_write_json(
            run_dir / "in_progress.json",
            {
                "schema_version": "ledger-in-progress/1.0",
                "run_id": run_id,
                "revision": revision,
                "parent_revision": self.parent_revision,
                "parent_manifest_hash": self.parent_manifest_hash,
            },
        )
        return {"revision": revision, "run_id": run_id, "parent_revision": self.parent_revision}

    def write_shard(self, name: str, value: Any) -> dict[str, Any]:
        if self.run_dir is None:
            raise BusinessError("E_IO", "ledger run was not started")
        path = self.run_dir / name
        atomic_write_json(path, value)
        data = path.read_bytes()
        self.shard_hashes[name] = hashlib.sha256(data).hexdigest()
        return {"name": name, "bytes": len(data), "sha256": self.shard_hashes[name]}

    def commit(
        self,
        content_hash: str,
        metric_ledger_digest: str,
        expected_revision: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.run_dir is None or self.revision is None:
            raise BusinessError("E_IO", "ledger run was not started")
        latest = self.load_latest_committed()
        current_revision = int(latest["manifest"]["revision"]) if latest else 0
        if self.parent_revision is None:
            expected_current = 0
        else:
            expected_current = self.parent_revision
        if expected_revision is not None:
            expected_current = int(expected_revision)
        if current_revision != expected_current:
            raise BusinessError(
                "E_SNAPSHOT_STALE",
                "ledger changed before commit",
                {"expected": expected_current, "current": current_revision, "run_revision": self.revision},
            )
        manifest: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "revision": self.revision,
            "run_id": self.run_id,
            "parent_revision": self.parent_revision,
            "parent_manifest_hash": self.parent_manifest_hash,
            "content_hash": content_hash,
            "metric_ledger_digest": metric_ledger_digest,
            "shard_hashes": copy.deepcopy(self.shard_hashes),
            "status": "committed",
        }
        if extra:
            manifest.update(extra)
        manifest_hash = self._manifest_hash(manifest)
        manifest["manifest_hash"] = manifest_hash
        atomic_write_json(self.run_dir / "manifest.json", manifest)
        current = {
            "schema_version": "ledger-current/1.0",
            "revision": self.revision,
            "run_id": self.run_id,
            "manifest_hash": manifest_hash,
            "content_hash": content_hash,
        }
        atomic_write_canonical_json(self.current_path, current)
        return manifest

    def mark_closed(self, manifest: dict[str, Any]) -> dict[str, Any]:
        closed = {
            "schema_version": "ledger-closed/1.0",
            "run_id": manifest["run_id"],
            "revision": manifest["revision"],
            "manifest_hash": manifest["manifest_hash"],
            "content_hash": manifest["content_hash"],
            "complete": True,
        }
        atomic_write_json(self.closed_path, closed)
        return closed

    def _read_snapshot_dir(self, directory: Path) -> dict[str, Any] | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json_load(manifest_path)
        except BusinessError:
            return None
        if not isinstance(manifest, dict) or manifest.get("status") != "committed":
            return None
        calculated = self._manifest_hash(manifest)
        if calculated != manifest.get("manifest_hash"):
            return None
        shard_hashes = manifest.get("shard_hashes")
        if not isinstance(shard_hashes, dict):
            return None
        for name, declared_hash in shard_hashes.items():
            path = directory / str(name)
            if not path.exists() or sha256_file(path) != declared_hash:
                return None
        for name in ("meta.json", "actual_frames.json", "actual_segments.json"):
            if name in shard_hashes:
                try:
                    shard = json_load(directory / name)
                except BusinessError:
                    return None
                if shard.get("content_hash") not in (None, manifest.get("content_hash")):
                    return None
        return {
            "manifest": manifest,
            "manifest_hash": calculated,
            "directory": str(directory),
            "run_id": manifest.get("run_id"),
        }

    def load_latest_committed(self) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for directory in self.snapshots.iterdir() if self.snapshots.exists() else []:
            if not directory.is_dir():
                continue
            snapshot = self._read_snapshot_dir(directory)
            if snapshot:
                candidates.append(snapshot)
        if not candidates:
            return None
        candidates.sort(
            key=lambda snapshot: (
                int(snapshot["manifest"]["revision"]),
                str(snapshot["manifest"].get("content_hash")),
                str(snapshot["manifest_hash"]),
            )
        )
        latest = candidates[-1]
        latest_dir = Path(latest["directory"])
        shards: dict[str, Any] = {}
        for name in latest["manifest"].get("shard_hashes", {}):
            shards[name] = json_load(latest_dir / name)
        latest["shards"] = shards
        latest["closed"] = False
        if self.closed_path.exists():
            try:
                closed = json_load(self.closed_path)
                latest["closed"] = (
                    closed.get("manifest_hash") == latest["manifest_hash"]
                    and closed.get("run_id") == latest["run_id"]
                    and closed.get("complete") is True
                )
            except BusinessError:
                latest["closed"] = False
        return latest

    def recover(self) -> dict[str, Any]:
        latest = self.load_latest_committed()
        latest_dir = Path(latest["directory"]).resolve() if latest else None
        orphans: list[dict[str, Any]] = []
        snapshot_dirs = sorted(self.snapshots.iterdir()) if self.snapshots.exists() else []
        for directory in snapshot_dirs:
            if not directory.is_dir():
                continue
            if latest_dir is not None and directory.resolve() == latest_dir:
                continue
            manifest_path = directory / "manifest.json"
            if manifest_path.exists():
                if self._read_snapshot_dir(directory) is not None:
                    continue
                orphan_kind = "corrupt_committed_snapshot"
            else:
                orphan_kind = "uncommitted_snapshot"
            inventory = remove_tree(directory)
            orphans.append({"kind": orphan_kind, "run_id": directory.name, "files": inventory})
        for temp_path in sorted(self.root.rglob("*.tmp-*")):
            if temp_path.is_file():
                inventory = remove_tree(temp_path)
                orphans.append({"kind": "temporary_file", "run_id": None, "files": inventory})
        recovery = {
            "schema_version": "ledger-recovery/1.0",
            "recovered": latest is not None,
            "latest_revision": latest["manifest"]["revision"] if latest else None,
            "latest_run_id": latest["run_id"] if latest else None,
            "latest_manifest_hash": latest["manifest_hash"] if latest else None,
            "latest_content_hash": latest["manifest"].get("content_hash") if latest else None,
            "normal_exit_confirmed": bool(latest and latest.get("closed")),
            "closed_marker_reused": False,
            "orphans_reclaimed": orphans,
            "orphan_count": len(orphans),
            "recovery_rule": "newest valid committed manifest by revision and content-addressed shard hashes",
        }
        recovery["report_hash"] = hash_object(recovery)
        atomic_write_json(self.recovery_path, recovery)
        return recovery

    def expected_revision(self) -> int:
        latest = self.load_latest_committed()
        return int(latest["manifest"]["revision"]) if latest else 0

    def shard_sha256(self, snapshot: dict[str, Any], name: str) -> str:
        return str(snapshot["manifest"]["shard_hashes"][name])
