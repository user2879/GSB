"""Atomic, content-addressed snapshots for stage artifacts and progress."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .fingerprint import canonical_json, fingerprint, sha256_bytes


MANIFEST_SCHEMA_VERSION = 1


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not available on every filesystem.  The file
            # replacement itself remains atomic.
            pass
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


class SnapshotStore:
    """Immutable object store plus atomically replaced snapshot manifests.

    Artifacts are content-addressed and therefore never overwritten with
    different content.  A run commits a complete manifest snapshot; the root
    ``manifest.json`` is only replaced after the immutable snapshot file has
    been durably written.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.snapshots_dir = self.root / "snapshots"
        self.manifest_path = self.root / "manifest.json"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("snapshot manifest root is not an object")
        return manifest

    def write_artifact(self, payload: dict[str, Any]) -> str:
        data = (canonical_json(payload) + "\n").encode("utf-8")
        object_id = sha256_bytes(data)
        path = self.objects_dir / f"{object_id}.json"
        with self._lock:
            if not path.exists():
                _atomic_write_bytes(path, data)
            elif path.read_bytes() != data:
                # Content addressing makes an existing object sufficient.
                # Read-back comparison repairs a truncated/corrupted object
                # without ever overwriting one object with different content.
                _atomic_write_bytes(path, data)
        return object_id

    def read_artifact(
        self, manifest: dict[str, Any] | None, page: str, stage: str, cache_key: str
    ) -> dict[str, Any] | None:
        if not manifest:
            return None
        page_entry = manifest.get("pages", {}).get(page)
        if not isinstance(page_entry, dict):
            return None
        stage_entry = page_entry.get("stages", {}).get(stage)
        if not isinstance(stage_entry, dict):
            return None
        if stage_entry.get("cache_key") != cache_key:
            return None
        object_id = stage_entry.get("object_id")
        if not isinstance(object_id, str):
            return None
        path = self.objects_dir / f"{object_id}.json"
        if not path.exists():
            return None
        data = path.read_bytes()
        if sha256_bytes(data) != object_id:
            return None
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if payload.get("cache_key") != cache_key:
            return None
        return payload

    def _manifest_footprint(self, manifest: dict[str, Any]) -> str:
        comparable = {
            "schema_version": manifest.get("schema_version"),
            "config_hash": manifest.get("config_hash"),
            "completed_pages": manifest.get("completed_pages", []),
            "failed_pages": manifest.get("failed_pages", []),
            "pages": manifest.get("pages", {}),
            "progress": manifest.get("progress", {}),
        }
        return fingerprint(comparable)

    def commit(self, manifest_without_version: dict[str, Any]) -> dict[str, Any]:
        """Commit a complete state snapshot.

        If the committed state is byte-for-byte equivalent after canonical
        projection, the existing global version is returned and no files are
        replaced.
        """

        with self._lock:
            previous = self.load_manifest()
            if previous is not None:
                if self._manifest_footprint(previous) == self._manifest_footprint(
                    manifest_without_version
                ):
                    return previous
                version = int(previous.get("global_version", 0)) + 1
            else:
                version = 1
            snapshot_file = f"snapshots/v{version:06d}.json"
            manifest = dict(manifest_without_version)
            manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
            manifest["global_version"] = version
            manifest["snapshot_file"] = snapshot_file
            manifest = _sort_manifest(manifest)
            data = (canonical_json(manifest) + "\n").encode("utf-8")
            snapshot_path = self.root / snapshot_file
            _atomic_write_bytes(snapshot_path, data)
            _atomic_write_bytes(self.manifest_path, data)
            return manifest

    def status(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest is None:
            return {
                "available": False,
                "snapshot_version": 0,
                "completed_pages": [],
                "failed_pages": [],
                "last_completed_page": None,
            }
        progress = manifest.get("progress", {})
        return {
            "available": True,
            "snapshot_version": manifest.get("global_version", 0),
            "snapshot_file": manifest.get("snapshot_file"),
            "config_hash": manifest.get("config_hash"),
            "completed_pages": list(manifest.get("completed_pages", [])),
            "failed_pages": [item.get("page") for item in manifest.get("failed_pages", [])],
            "last_completed_page": progress.get("last_completed_page"),
            "interrupted": bool(progress.get("interrupted", False)),
            "recovery_ready": True,
        }


def _sort_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["completed_pages"] = sorted(result.get("completed_pages", []))
    result["failed_pages"] = sorted(
        result.get("failed_pages", []),
        key=lambda item: (str(item.get("page", "")), str(item.get("stage", ""))),
    )
    sorted_pages: dict[str, Any] = {}
    for page in sorted(result.get("pages", {})):
        page_entry = dict(result["pages"][page])
        page_entry["stages"] = {
            stage: page_entry.get("stages", {})[stage]
            for stage in sorted(page_entry.get("stages", {}))
        }
        sorted_pages[page] = page_entry
    result["pages"] = sorted_pages
    return result
