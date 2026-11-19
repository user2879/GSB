"""Atomic on-disk persistence for snapshots, op logs and chain manifests.

Directory layout::

    <root>/
      manifest.json
      shutdown.marker
      snapshots/snapshot_v000001.json
      logs/log_00000000_v000001..v000010.jsonl
      errors/errors.jsonl
      quarantine/
      progress/

All writes are atomic at file granularity: content lands in a deterministic
``*.tmp`` sibling, is fsynced, then renamed into place. A process killed at
any point can therefore never observe a half-written snapshot or manifest;
only the trailing op-log line may be partial and is detected by CRC.
"""

from __future__ import annotations

import json
import os
import zlib
from typing import Any, Dict, List, Optional, Tuple

from .errors import (
    E_CHAIN_BROKEN,
    E_LOG_TRUNCATED,
    E_SNAPSHOT_HASH,
    E_VERSION_DIVERGE,
    PipelineError,
    make_error,
)
from .state import canonical_bytes, snapshot_digest

SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"
ERROR_DIR = "errors"
QUARANTINE_DIR = "quarantine"
PROGRESS_DIR = "progress"
MANIFEST_NAME = "manifest.json"
SHUTDOWN_MARKER = "shutdown.marker"
ERROR_LOG_NAME = "errors.jsonl"
LOG_LINES_PER_SEGMENT = 8
KEEP_ANCHOR_EVERY = 5


class Storage:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.snapshot_dir = os.path.join(self.root, SNAPSHOT_DIR)
        self.log_dir = os.path.join(self.root, LOG_DIR)
        self.error_dir = os.path.join(self.root, ERROR_DIR)
        self.quarantine_dir = os.path.join(self.root, QUARANTINE_DIR)
        self.progress_dir = os.path.join(self.root, PROGRESS_DIR)
        self._log_handle = None
        self._log_path: Optional[str] = None
        self._log_lines = 0
        self._log_start_version: Optional[int] = None

    # ---------- generic helpers ----------

    def ensure_dirs(self) -> None:
        for directory in (
            self.root,
            self.snapshot_dir,
            self.log_dir,
            self.error_dir,
            self.quarantine_dir,
            self.progress_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    def atomic_write_bytes(self, relative_path: str, payload: bytes) -> str:
        self.ensure_dirs()
        final_path = os.path.join(self.root, relative_path)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
        self._fsync_dir(os.path.dirname(final_path))
        return final_path

    def atomic_write_json(self, relative_path: str, value: Any) -> str:
        return self.atomic_write_bytes(relative_path, canonical_bytes(value) + b"\n")

    @staticmethod
    def _fsync_dir(directory: str) -> None:
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    # ---------- manifest ----------

    def manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST_NAME)

    def has_manifest(self) -> bool:
        return os.path.exists(self.manifest_path())

    def init_manifest(self, seed: int) -> Dict[str, Any]:
        manifest = {
            "schema": "syncpipe.manifest/v1",
            "seed": seed,
            "head_version": 0,
            "versions": [],
            "snapshot_files": [],
            "compactions": [],
            "shutdown": False,
        }
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        self.atomic_write_json(MANIFEST_NAME, manifest)

    def load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path(), "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))

    def write_shutdown_marker(self, manifest: Dict[str, Any]) -> None:
        marker = {
            "schema": "syncpipe.shutdown/v1",
            "head_version": manifest["head_version"],
            "head_snapshot_hash": manifest["versions"][-1]["snapshot_hash"],
        }
        self.atomic_write_json(SHUTDOWN_MARKER, marker)

    def has_shutdown_marker(self) -> bool:
        return os.path.exists(os.path.join(self.root, SHUTDOWN_MARKER))

    def remove_shutdown_marker(self) -> None:
        path = os.path.join(self.root, SHUTDOWN_MARKER)
        if os.path.exists(path):
            os.remove(path)

    # ---------- snapshots ----------

    @staticmethod
    def snapshot_filename(version: int) -> str:
        return os.path.join(SNAPSHOT_DIR, f"snapshot_v{version:06d}.json")

    def write_snapshot(self, snapshot: Dict[str, Any]) -> str:
        stored_hash = snapshot["snapshot_hash"]
        actual_hash = snapshot_digest(snapshot)
        if stored_hash != actual_hash:
            raise PipelineError(
                E_SNAPSHOT_HASH,
                f"snapshot content hash mismatch before write v{snapshot['version']}",
                version=snapshot["version"],
                tick=snapshot["tick"],
            )
        return self.atomic_write_json(self.snapshot_filename(snapshot["version"]), snapshot)

    def read_snapshot_file(self, version: int) -> Dict[str, Any]:
        path = os.path.join(self.root, self.snapshot_filename(version))
        with open(path, "rb") as handle:
            snapshot = json.loads(handle.read().decode("utf-8"))
        if snapshot_digest(snapshot) != snapshot["snapshot_hash"]:
            raise PipelineError(
                E_SNAPSHOT_HASH,
                f"snapshot file tampered or half-written v{version}",
                version=version,
                tick=snapshot.get("tick"),
            )
        return snapshot

    def available_snapshot_versions(self) -> List[int]:
        if not os.path.isdir(self.snapshot_dir):
            return []
        versions = []
        for name in os.listdir(self.snapshot_dir):
            if name.startswith("snapshot_v") and name.endswith(".json"):
                versions.append(int(name[len("snapshot_v") : -len(".json")]))
        return sorted(versions)

    def remove_snapshot_file(self, version: int) -> None:
        path = os.path.join(self.root, self.snapshot_filename(version))
        if os.path.exists(path):
            os.remove(path)

    # ---------- op logs ----------

    @staticmethod
    def log_filename(start_version: int, end_version: int) -> str:
        return os.path.join(
            LOG_DIR, f"log_{start_version:08d}_v{start_version:06d}..v{end_version:06d}.jsonl"
        )

    @staticmethod
    def _entry_with_crc(entry: Dict[str, Any]) -> Dict[str, Any]:
        body = canonical_bytes(entry)
        tagged = dict(entry)
        tagged["crc32"] = f"{zlib.crc32(body) & 0xFFFFFFFF:08x}"
        return tagged

    def append_log_entry(self, entry: Dict[str, Any]) -> None:
        if self._log_handle is None or self._log_lines >= LOG_LINES_PER_SEGMENT:
            self.rotate_log(entry["version"])
        tagged = self._entry_with_crc(entry)
        line = canonical_bytes(tagged) + b"\n"
        assert self._log_handle is not None
        self._log_handle.write(line)
        self._log_handle.flush()
        os.fsync(self._log_handle.fileno())
        self._log_lines += 1

    def rotate_log(self, next_version: int) -> None:
        if self._log_handle is not None:
            self._log_handle.flush()
            os.fsync(self._log_handle.fileno())
            self._log_handle.close()
            old_path = self._log_path
            start = self._log_start_version
            end = next_version - 1
            final_name = self.log_filename(start, end)
            os.replace(old_path, os.path.join(self.root, final_name))
            self._fsync_dir(self.log_dir)
        tmp_name = os.path.join(LOG_DIR, f"log_pending_v{next_version:06d}.jsonl.tmp")
        self.ensure_dirs()
        self._log_path = os.path.join(self.root, tmp_name)
        self._log_handle = open(self._log_path, "ab")
        self._log_lines = 0
        self._log_start_version = next_version

    def close_log(self) -> None:
        if self._log_handle is None:
            return
        self._log_handle.flush()
        os.fsync(self._log_handle.fileno())
        self._log_handle.close()
        start = self._log_start_version
        end = start + self._log_lines - 1
        if self._log_lines > 0:
            final_name = self.log_filename(start, end)
            os.replace(self._log_path, os.path.join(self.root, final_name))
            self._fsync_dir(self.log_dir)
        else:
            os.remove(self._log_path)
        self._log_handle = None
        self._log_path = None
        self._log_lines = 0
        self._log_start_version = None

    def list_log_segments(self) -> List[str]:
        if not os.path.isdir(self.log_dir):
            return []
        return sorted(
            os.path.join(self.log_dir, name)
            for name in os.listdir(self.log_dir)
            if name.endswith(".jsonl")
        )

    def list_pending_logs(self) -> List[str]:
        if not os.path.isdir(self.log_dir):
            return []
        return sorted(
            os.path.join(self.log_dir, name)
            for name in os.listdir(self.log_dir)
            if name.endswith(".tmp")
        )

    def read_all_log_entries(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Read every durable log line.

        Returns ``(entries, truncation_errors)``. The first line that fails
        CRC/JSON validation terminates that segment; the remainder is
        reported as E_LOG_TRUNCATED and the offending bytes are quarantined.
        """
        entries: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for segment in self.list_log_segments() + self.list_pending_logs():
            segment_entries, segment_errors = self._read_segment(segment)
            entries.extend(segment_entries)
            errors.extend(segment_errors)
        entries.sort(key=lambda item: item["version"])
        return entries, errors

    def _read_segment(self, segment_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entries: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        with open(segment_path, "rb") as handle:
            raw = handle.read()
        if not raw:
            return entries, errors
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]
        truncated_at = None
        for index, line in enumerate(lines):
            try:
                tagged = json.loads(line.decode("utf-8"))
                expected = tagged.pop("crc32")
                actual = f"{zlib.crc32(canonical_bytes(tagged)) & 0xFFFFFFFF:08x}"
                if expected != actual:
                    raise ValueError(f"crc mismatch expected={expected} actual={actual}")
            except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError) as exc:
                truncated_at = index
                bad_bytes = b"\n".join(lines[index:])
                quarantine_name = os.path.join(
                    QUARANTINE_DIR,
                    f"truncated_segment_{len(entries) + 1:06d}.part",
                )
                self.atomic_write_bytes(quarantine_name, bad_bytes)
                version = entries[-1]["version"] if entries else 0
                errors.append(
                    make_error(
                        E_LOG_TRUNCATED,
                        f"op_log segment line={index + 1}: {exc}",
                        version=version,
                    )
                )
                break
            entries.append(tagged)
        return entries, errors

    # ---------- errors / quarantine / progress ----------

    def append_error(self, record: Dict[str, Any]) -> None:
        self.ensure_dirs()
        with open(os.path.join(self.error_dir, ERROR_LOG_NAME), "ab") as handle:
            handle.write(canonical_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_errors(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.error_dir, ERROR_LOG_NAME)
        if not os.path.exists(path):
            return []
        with open(path, "rb") as handle:
            return [json.loads(line) for line in handle.read().splitlines() if line.strip()]

    def write_progress(self, name: str, payload: Any) -> None:
        self.atomic_write_json(os.path.join(PROGRESS_DIR, name), payload)

    def read_progress(self, name: str) -> Optional[Any]:
        path = os.path.join(self.progress_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))

    # ---------- chain validation helpers ----------

    @staticmethod
    def verify_parent_chain(manifest: Dict[str, Any]) -> None:
        expected_parent = None
        for record in manifest["versions"]:
            if record["parent_version"] != expected_parent:
                raise PipelineError(
                    E_CHAIN_BROKEN,
                    f"parent mismatch at v{record['version']} expected={expected_parent} "
                    f"actual={record['parent_version']}",
                    version=record["version"],
                    tick=record["tick"],
                )
            expected_parent = record["version"]

    @staticmethod
    def detect_replacement(old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> None:
        """Refuse to blindly ingest a replaced snapshot directory.

        The first version whose stored hash disagrees with the previously
        recorded hash proves a version divergence rather than a simple
        extension.
        """
        old_by_version = {item["version"]: item for item in old_manifest["versions"]}
        for item in new_manifest["versions"]:
            previous = old_by_version.get(item["version"])
            if previous is not None and previous["snapshot_hash"] != item["snapshot_hash"]:
                raise PipelineError(
                    E_VERSION_DIVERGE,
                    f"snapshot directory replaced: v{item['version']} changed hash "
                    f"old={previous['snapshot_hash'][:12]} new={item['snapshot_hash'][:12]}",
                    version=item["version"],
                    tick=item["tick"],
                )
