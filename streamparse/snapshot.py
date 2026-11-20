from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .errors import E_SNAPSHOT_STALE
from .fingerprint import canonical_json, content_hash


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(value) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SnapshotManager:
    """Atomic directory snapshot writer.

    Every commit is prepared in a sibling temporary directory, fsynced, and
    then swapped with ``current`` under a lock.  A version callback is checked
    both before writing and immediately before the swap.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        version_provider: Optional[Callable[[], int]] = None,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = self.root / "current"
        self.version_provider = version_provider
        self.enabled = enabled
        self.lock = threading.RLock()
        self.events: list[Dict[str, Any]] = []
        self.last_manifest: Optional[Dict[str, Any]] = None

    def _current_version(self) -> Optional[int]:
        if self.version_provider is None:
            return None
        return int(self.version_provider())

    def _stale(self, stage: str, expected_version: int, observed: Optional[int], reason: str) -> Dict[str, Any]:
        event = {
            "status": "stale",
            "business_code": E_SNAPSHOT_STALE,
            "stage": stage,
            "expected_version": expected_version,
            "observed_version": observed,
            "reason": reason,
        }
        with self.lock:
            self.events.append(event)
        return event

    def commit_stage(
        self,
        *,
        stage: str,
        expected_version: int,
        config_hash: str,
        completed_frame_cursor: Dict[str, Any],
        payload: Dict[str, Any],
        extra_files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "stage": stage, "expected_version": expected_version}

        observed = self._current_version()
        if observed is not None and observed != expected_version:
            return self._stale(stage, expected_version, observed, "version changed before preparation")

        with self.lock:
            temp_path = Path(
                tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=str(self.root))
            )
        try:
            _write_json(temp_path / "stage.json", payload)
            files: Dict[str, str] = {"stage.json": _hash_file(temp_path / "stage.json")}
            for name, value in sorted((extra_files or {}).items()):
                safe_name = str(name).replace("\\", "_").replace("/", "_")
                destination = temp_path / safe_name
                _write_json(destination, value)
                files[safe_name] = _hash_file(destination)

            manifest = {
                "schema_version": "1.0",
                "stage": stage,
                "global_version": expected_version,
                "config_hash": config_hash,
                "completed_frame_cursor": completed_frame_cursor,
                "files": files,
                "status": "prepared",
            }
            _write_json(temp_path / "manifest.json", manifest)

            observed = self._current_version()
            if observed is not None and observed != expected_version:
                shutil.rmtree(temp_path, ignore_errors=True)
                return self._stale(stage, expected_version, observed, "version changed before commit")

            with self.lock:
                backup: Optional[Path] = None
                if self.current.exists():
                    backup = self.root / f".old-snapshot-{uuid.uuid4().hex}"
                    os.replace(self.current, backup)
                try:
                    os.replace(temp_path, self.current)
                except Exception:
                    if backup is not None and backup.exists() and not self.current.exists():
                        os.replace(backup, self.current)
                    raise
                finally:
                    if backup is not None:
                        shutil.rmtree(backup, ignore_errors=True)

                manifest = dict(manifest)
                manifest["status"] = "committed"
                manifest["directory"] = "current"
                manifest["manifest_sha256"] = content_hash(manifest)
                _write_json(self.current / "manifest.json", manifest)
                self.last_manifest = manifest
                event = {
                    "status": "committed",
                    "stage": stage,
                    "global_version": expected_version,
                    "config_hash": config_hash,
                    "completed_frame_cursor": completed_frame_cursor,
                }
                self.events.append(event)
                return event
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

    def report(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "root": str(self.root),
                "current_directory": str(self.current),
                "events": list(self.events),
                "last_manifest": self.last_manifest,
            }
