"""Atomic sample snapshots and resumable run-manifest management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache import StageCache
from .util import PRODUCT_VERSIONS, atomic_write_json, read_json, sha256_json


SNAPSHOT_SCHEMA_VERSION = "1.0"


class SnapshotStore:
    def __init__(self, state_dir: str | Path):
        self.root = Path(state_dir)
        self.snapshot_dir = self.root / "snapshot"
        self.samples_dir = self.snapshot_dir / "samples"
        self.manifest_path = self.snapshot_dir / "manifest.json"
        self.prior_manifest_path = self.snapshot_dir / "prior_manifest.json"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.cache = StageCache(self.root / "cache")

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = read_json(self.manifest_path)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def archive_prior_manifest(self, existing: dict[str, Any]) -> None:
        if existing:
            atomic_write_json(self.prior_manifest_path, existing)

    def start_manifest(
        self,
        *,
        config_hash: str,
        manifest_hash: str,
        total_samples: int,
        global_version: int,
        run_id: str,
    ) -> dict[str, Any]:
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "product_versions": PRODUCT_VERSIONS,
            "global_version": int(global_version),
            "run_id": run_id,
            "status": "running",
            "config_hash": config_hash,
            "injection_manifest_hash": manifest_hash,
            "cursor": 0,
            "total_samples": int(total_samples),
            "sample_entries": [],
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def write_snapshot(
        self,
        *,
        index: int,
        file: str,
        stable_sort_key: str,
        kind: str,
        content: dict[str, Any],
        config_hash: str,
        manifest_hash: str,
        global_version: int,
        run_id: str,
    ) -> str:
        name_hash = sha256_json({"file": file, "index": index, "kind": kind})[:12]
        filename = f"{index:06d}-{name_hash}.json"
        record = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_version": PRODUCT_VERSIONS["snapshot"],
            "index": int(index),
            "file": file,
            "stable_sort_key": stable_sort_key,
            "kind": kind,
            "status": "complete",
            "global_version": int(global_version),
            "run_id": run_id,
            "config_hash": config_hash,
            "injection_manifest_hash": manifest_hash,
            "content": content,
        }
        atomic_write_json(self.samples_dir / filename, record)
        return filename

    def read_snapshot(self, filename: str) -> dict[str, Any] | None:
        path = self.samples_dir / filename
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def validate_entry(
        self,
        entry: dict[str, Any],
        *,
        index: int,
        config_hash: str,
        manifest_hash: str,
    ) -> dict[str, Any] | None:
        if not isinstance(entry, dict):
            return None
        if entry.get("index") != index:
            return None
        filename = entry.get("snapshot")
        if not isinstance(filename, str):
            return None
        record = self.read_snapshot(filename)
        if record is None:
            return None
        if record.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            return None
        if record.get("status") != "complete":
            return None
        if record.get("index") != index:
            return None
        if record.get("file") != entry.get("file"):
            return None
        if record.get("kind") != entry.get("kind"):
            return None
        if record.get("config_hash") != config_hash:
            return None
        if record.get("injection_manifest_hash") != manifest_hash:
            return None
        content = record.get("content")
        if not isinstance(content, dict):
            return None
        return record

    def recover_prefix(
        self,
        manifest: dict[str, Any],
        *,
        config_hash: str,
        manifest_hash: str,
        expected_files: list[str] | None = None,
        current_audio_hashes: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int, list[str]]:
        entries = manifest.get("sample_entries") or []
        cursor = manifest.get("cursor", 0)
        if not isinstance(entries, list) or not isinstance(cursor, int):
            return [], 0, ["manifest_entries_or_cursor_invalid"]
        cursor = max(0, min(cursor, len(entries)))
        validated_records: list[dict[str, Any]] = []
        problems: list[str] = []
        for index in range(cursor):
            entry = entries[index]
            if expected_files is not None:
                if index >= len(expected_files) or entry.get("file") != expected_files[index]:
                    problems.append(f"input_file_order_changed_at_index_{index}")
                    break
            if current_audio_hashes is not None:
                fingerprints = entry.get("fingerprints") or {}
                expected_hash = (
                    current_audio_hashes[index]
                    if index < len(current_audio_hashes)
                    else None
                )
                if (
                    not expected_hash
                    or fingerprints.get("audio_sha256") != expected_hash
                ):
                    problems.append(f"audio_content_changed_at_index_{index}")
                    break
            record = self.validate_entry(
                entry,
                index=index,
                config_hash=config_hash,
                manifest_hash=manifest_hash,
            )
            if record is None:
                problems.append(f"snapshot_prefix_stops_at_index_{index}")
                break
            validated_records.append(record)
        return validated_records, len(validated_records), problems

    def commit_manifest(
        self,
        manifest: dict[str, Any],
        *,
        entries: list[dict[str, Any]],
        cursor: int,
        status: str,
        recovery: dict[str, Any],
    ) -> dict[str, Any]:
        committed = dict(manifest)
        committed["sample_entries"] = list(entries)
        committed["cursor"] = int(cursor)
        committed["status"] = status
        committed["recovery"] = recovery
        atomic_write_json(self.manifest_path, committed)
        return committed
