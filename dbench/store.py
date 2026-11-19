"""Versioned, atomic snapshot store for stage artifacts.

Each stage owns a directory. Every successful computation is committed as a
new immutable version:

    <store>/<stage>/v<NNNN>.manifest.json
    <store>/<stage>/v<NNNN>.data.npy        # array-bearing stages only

Writes always go to a unique temporary file in the same directory followed by
``os.replace`` (atomic on POSIX and Windows NTFS), so a half-written file can
never be observed. ``progress.json`` is committed the same way.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np

from .errors import E_SNAPSHOT_VERSION, WorkbenchError
from .hashing import fingerprint_array, sha256_hex

_VERSION_RE = re.compile(r"^v(\d+)\.manifest\.json$")
PROGRESS_NAME = "progress.json"


class SnapshotStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # ---- paths ----------------------------------------------------------
    def stage_dir(self, stage: str) -> str:
        path = os.path.join(self.root, stage)
        os.makedirs(path, exist_ok=True)
        return path

    def _manifest_path(self, stage: str, version: int) -> str:
        return os.path.join(self.stage_dir(stage), f"v{version:04d}.manifest.json")

    def _data_path(self, stage: str, version: int) -> str:
        return os.path.join(self.stage_dir(stage), f"v{version:04d}.data.npy")

    # ---- listing / continuity ------------------------------------------
    def list_versions(self, stage: str) -> list[int]:
        if not os.path.isdir(os.path.join(self.root, stage)):
            return []
        versions = []
        for name in os.listdir(os.path.join(self.root, stage)):
            match = _VERSION_RE.match(name)
            if match:
                versions.append(int(match.group(1)))
        return sorted(versions)

    def check_continuity(self, stage: str) -> None:
        """Reject runs when a stage's versions are missing or non-contiguous.

        Both manifest and data files are scanned independently, so deleting a
        manifest while its data remains (or vice versa) is also detected.
        """
        directory = os.path.join(self.root, stage)
        if not os.path.isdir(directory):
            return
        manifest_versions: set[int] = set()
        data_versions: set[int] = set()
        for name in os.listdir(directory):
            m = re.match(r"^v(\d+)\.manifest\.json$", name)
            if m:
                manifest_versions.add(int(m.group(1)))
            m = re.match(r"^v(\d+)\.data\.npy$", name)
            if m:
                data_versions.add(int(m.group(1)))
        all_versions = manifest_versions | data_versions
        if not all_versions:
            return
        highest = max(all_versions)
        expected = set(range(1, highest + 1))
        if manifest_versions != expected:
            raise WorkbenchError(
                E_SNAPSHOT_VERSION,
                f"snapshot manifest versions for stage '{stage}' are not contiguous",
                {"stage": stage, "present": sorted(manifest_versions),
                 "missing": sorted(expected - manifest_versions)},
            )
        for version in sorted(expected):
            manifest = self.read_manifest(stage, version)
            if manifest.get("has_data") and version not in data_versions:
                raise WorkbenchError(
                    E_SNAPSHOT_VERSION,
                    f"missing data file for {stage} v{version}",
                    {"stage": stage, "version": version},
                )

    def recover_pending_writes(self, stages) -> dict:
        """Clean artifacts of writes interrupted before atomic replacement.

        After a forced kill the only trustworthy products are fully replaced
        manifests. Temporary files and orphan data (a ``.npy`` whose manifest
        never landed) cannot represent a committed version and are removed;
        ``progress.json`` is then reconciled to the surviving manifests.
        Deliberately corrupted/non-contiguous committed versions are still
        rejected by :meth:`check_continuity`.
        """
        removed: list[str] = []
        for stage in stages:
            directory = os.path.join(self.root, stage)
            if not os.path.isdir(directory):
                continue
            manifests = set(self.list_versions(stage))
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                if name.startswith(".tmp.") or name.endswith(".tmp") \
                        or name.endswith(".tmp.png"):
                    os.remove(path)
                    removed.append(path)
                    continue
                m = re.match(r"^v(\d+)\.data\.npy$", name)
                if m and int(m.group(1)) not in manifests:
                    os.remove(path)
                    removed.append(path)
        # Reconcile progress ledger with surviving committed manifests.
        progress = self.read_progress()
        committed = progress.get("committed", {})
        for stage in list(committed):
            info = committed[stage]
            version = info.get("version")
            if stage not in stages or version not in self.list_versions(stage):
                committed.pop(stage, None)
                removed.append(f"progress:{stage}")
        progress["committed"] = committed
        self.write_progress(progress)
        return {"removed": removed, "committed": committed}

    def next_version(self, stage: str) -> int:
        versions = self.list_versions(stage)
        return (versions[-1] + 1) if versions else 1

    # ---- atomic IO ------------------------------------------------------
    def _atomic_write_bytes(self, path: str, payload: bytes) -> None:
        directory = os.path.dirname(path)
        temp = os.path.join(directory, f".tmp.{os.getpid()}.{sha256_hex(payload)[:16]}")
        with open(temp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _atomic_write_json(self, path: str, value: Any) -> None:
        payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
        self._atomic_write_bytes(path, payload)

    def commit(self, stage: str, manifest: dict, array: np.ndarray | None) -> dict:
        self.check_continuity(stage)
        version = self.next_version(stage)
        has_data = array is not None
        if has_data:
            arr = np.ascontiguousarray(array)
            data_path = self._data_path(stage, version)
            # allow_pickle=False keeps every artifact pure numeric data.
            self._atomic_write_bytes(data_path, np.save_to_buffer(arr)
                                     if hasattr(np, "save_to_buffer")
                                     else _npy_bytes(arr))
            manifest = dict(manifest)
            manifest["data_file"] = os.path.basename(data_path)
            manifest["array_fingerprint"] = fingerprint_array(arr)
        manifest = dict(manifest)
        manifest["stage"] = stage
        manifest["version"] = version
        manifest["has_data"] = has_data
        manifest["schema"] = "dbench.artifact.v1"
        self._atomic_write_json(self._manifest_path(stage, version), manifest)
        return manifest

    def read_manifest(self, stage: str, version: int) -> dict:
        with open(self._manifest_path(stage, version), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def latest_manifest(self, stage: str) -> dict | None:
        versions = self.list_versions(stage)
        if not versions:
            return None
        return self.read_manifest(stage, versions[-1])

    def load_array(self, stage: str, version: int) -> np.ndarray:
        with open(self._data_path(stage, version), "rb") as handle:
            return np.load(handle, allow_pickle=False)

    def find_version_by_fingerprint(self, stage: str, content_fingerprint: str) -> int | None:
        """Return an existing version whose content fingerprint matches."""
        for version in self.list_versions(stage):
            manifest = self.read_manifest(stage, version)
            if manifest.get("content_fingerprint") == content_fingerprint:
                return version
        return None

    # ---- progress ledger ------------------------------------------------
    def progress_path(self) -> str:
        return os.path.join(self.root, PROGRESS_NAME)

    def read_progress(self) -> dict:
        path = self.progress_path()
        if not os.path.isfile(path):
            return {"schema": "dbench.progress.v1", "committed": {}, "last_recovery": None}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_progress(self, progress: dict) -> None:
        progress = dict(progress)
        progress["schema"] = "dbench.progress.v1"
        self._atomic_write_json(self.progress_path(), progress)

    def mark_committed(self, stage: str, version: int, fingerprint: str) -> dict:
        progress = self.read_progress()
        progress.setdefault("committed", {})[stage] = {
            "version": version,
            "content_fingerprint": fingerprint,
        }
        self.write_progress(progress)
        return progress


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()
