"""Atomic stage snapshots and manifest-based recovery checks."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .stable import sha256_json
from .stages import STAGE_ORDER


SCHEMA_VERSION = 1


class SnapshotStore:
    """A content-keyed stage cache with whole-snapshot atomic replacement."""

    def __init__(
        self,
        root: str | Path,
        *,
        config_hash: str,
        scale_tier: int,
        expected_version: int | None = None,
        fresh: bool = False,
    ) -> None:
        self.root = Path(root)
        self.config_hash = config_hash
        self.scale_tier = scale_tier
        self.expected_version = expected_version
        self.fresh = fresh
        self.stage_entries: dict[str, dict[str, dict[str, Any]]] = {
            stage: {} for stage in STAGE_ORDER
        }
        self.stage_versions: dict[str, int] = {}
        self.stage_hashes: dict[str, str] = {}
        self.global_version = 0
        self.completed_sequences: list[str] = []
        self.stale_events: list[dict[str, Any]] = []
        self.dirty_stages: set[str] = set()
        self._commit_counter = 0
        self._manifest_config_hash: str | None = None
        self._manifest_scale_tier: int | None = None

    def open(self) -> None:
        if self.fresh and self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "stages").mkdir(parents=True, exist_ok=True)
        (self.root / "tmp").mkdir(parents=True, exist_ok=True)
        self._clean_temporary_directories()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            orphaned = any(
                path.is_dir() and any(path.iterdir())
                for path in (self.root / "stages").iterdir()
            )
            if orphaned:
                self._stale("MISSING_MANIFEST_WITH_STAGE_SNAPSHOTS")
            if self.expected_version is not None and not self.fresh:
                self._stale(
                    "MISSING_MANIFEST",
                    expected=self.expected_version,
                )
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._stale("UNREADABLE_MANIFEST", detail=str(exc))
            return
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
            self._stale("UNSUPPORTED_MANIFEST_SCHEMA")
            return
        self.global_version = int(manifest.get("global_version", 0))
        self.completed_sequences = sorted(manifest.get("completed_sequences", []))
        self._manifest_config_hash = manifest.get("config_hash")
        self._manifest_scale_tier = manifest.get("scale_tier")
        versions = manifest.get("stage_versions", {})
        hashes = manifest.get("stage_hashes", {})
        if not isinstance(versions, dict) or not isinstance(hashes, dict):
            self._stale("INVALID_MANIFEST_STAGE_MAP")
            versions = {}
            hashes = {}
        if manifest.get("config_hash") != self.config_hash:
            self._stale(
                "CONFIG_HASH_MISMATCH",
                previous=manifest.get("config_hash"),
                current=self.config_hash,
            )
        if self.expected_version is not None and self.expected_version != self.global_version:
            self._stale(
                "EXPECTED_VERSION_MISMATCH",
                expected=self.expected_version,
                actual=self.global_version,
            )
            for stage in STAGE_ORDER:
                self.stage_entries[stage] = {}
            self.stage_versions = {}
            self.stage_hashes = {}
            return
        for stage in STAGE_ORDER:
            version = versions.get(stage)
            if version is None:
                self._stale("STAGE_VERSION_MISSING", stage=stage)
                continue
            try:
                version_number = int(version)
            except (TypeError, ValueError):
                self._stale("INVALID_STAGE_VERSION", stage=stage, version=version)
                continue
            index_path = self._stage_index_path(stage, version_number)
            if not index_path.exists():
                self._stale(
                    "STAGE_SNAPSHOT_MISSING", stage=stage, version=version_number
                )
                continue
            try:
                snapshot = json.loads(index_path.read_text(encoding="utf-8"))
                if snapshot.get("stage") != stage or int(snapshot.get("version")) != version_number:
                    raise ValueError("snapshot stage/version does not match manifest")
                entries = snapshot["entries"]
                if not isinstance(entries, dict):
                    raise ValueError("entries must be a mapping")
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._stale(
                    "STAGE_SNAPSHOT_INVALID",
                    stage=stage,
                    version=version_number,
                    detail=str(exc),
                )
                continue
            actual_hash = self._entries_hash(entries)
            expected_hash = hashes.get(stage)
            if expected_hash is not None and expected_hash != actual_hash:
                self._stale(
                    "STAGE_SNAPSHOT_HASH_MISMATCH",
                    stage=stage,
                    version=version_number,
                )
                continue
            self.stage_entries[stage] = entries
            self.stage_versions[stage] = version_number
            self.stage_hashes[stage] = actual_hash

    def _clean_temporary_directories(self) -> None:
        temp_root = self.root / "tmp"
        for path in temp_root.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except OSError:
                    pass

    def _stale(self, reason: str, **details: Any) -> None:
        event = {
            "business_code": "E_SNAPSHOT_STALE",
            "reason": reason,
            **details,
        }
        self.stale_events.append(event)

    def _stage_index_path(self, stage: str, version: int) -> Path:
        return self.root / "stages" / stage / f"{version:08d}" / "index.json"

    @staticmethod
    def _entries_hash(entries: dict[str, Any]) -> str:
        return sha256_json(entries)

    def lookup(self, stage: str, cache_key: str) -> dict[str, Any] | None:
        return self.stage_entries[stage].get(cache_key)

    def put(self, stage: str, cache_key: str, entry: dict[str, Any]) -> None:
        existing = self.stage_entries[stage].get(cache_key)
        if existing is None:
            self.stage_entries[stage][cache_key] = entry
            self.dirty_stages.add(stage)

    def previous_keys_for_sequence(self, stage: str, sequence_id: str) -> list[str]:
        return sorted(
            key
            for key, entry in self.stage_entries[stage].items()
            if entry.get("sequence_id") == sequence_id
        )

    def _next_stage_version(self, stage: str) -> int:
        manifest_version = self.stage_versions.get(stage, 0)
        stage_root = self.root / "stages" / stage
        maximum = manifest_version
        if stage_root.exists():
            for path in stage_root.iterdir():
                if not path.is_dir():
                    continue
                try:
                    maximum = max(maximum, int(path.name))
                except ValueError:
                    continue
        return maximum + 1

    def commit(self, completed_sequences: list[str]) -> None:
        dirty = sorted(self.dirty_stages, key=STAGE_ORDER.index)
        if (
            not dirty
            and (self.root / "manifest.json").exists()
            and self._manifest_config_hash == self.config_hash
            and self._manifest_scale_tier == self.scale_tier
        ):
            # A run may only advance recovery progress without changing artifacts.
            if sorted(completed_sequences) == sorted(self.completed_sequences):
                return
        self._commit_counter += 1
        self.global_version += 1
        for stage in dirty:
            version = self._next_stage_version(stage)
            entries = self.stage_entries[stage]
            final_dir = self.root / "stages" / stage / f"{version:08d}"
            while final_dir.exists():
                version += 1
                final_dir = self.root / "stages" / stage / f"{version:08d}"
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "stage": stage,
                "version": version,
                "entries": entries,
            }
            tmp_dir = self.root / "tmp" / f"{stage}-{version:08d}-{os.getpid()}-{self._commit_counter}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            index_path = tmp_dir / "index.json"
            self._write_json(index_path, snapshot)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_dir, final_dir)
            self.stage_versions[stage] = version
            self.stage_hashes[stage] = self._entries_hash(entries)
        self.completed_sequences = sorted(set(completed_sequences))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "scale_tier": self.scale_tier,
            "completed_sequences": self.completed_sequences,
            "stage_versions": {
                stage: self.stage_versions[stage]
                for stage in STAGE_ORDER
                if stage in self.stage_versions
            },
            "stage_hashes": {
                stage: self.stage_hashes[stage]
                for stage in STAGE_ORDER
                if stage in self.stage_hashes
            },
            "committed_at_monotonic_ns": time.monotonic_ns(),
        }
        tmp_manifest = self.root / "tmp" / f"manifest-{os.getpid()}-{self._commit_counter}.json"
        self._write_json(tmp_manifest, manifest)
        os.replace(tmp_manifest, self.root / "manifest.json")
        self.dirty_stages.clear()
        self._manifest_config_hash = self.config_hash
        self._manifest_scale_tier = self.scale_tier

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        text = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
