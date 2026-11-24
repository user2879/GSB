"""Whole-run staged snapshots with an atomically replaced CURRENT pointer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_json, canonical_json, read_json, sha256_text


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CurrentSnapshot:
    pointer: dict[str, Any]
    manifest: dict[str, Any]
    run_dir: Path


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.staging_dir = self.root / "staging"
        self.cache_dir = self.root / "cache"
        self.current_path = self.root / "CURRENT.json"
        self.last_current_error = ""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> CurrentSnapshot | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = read_json(self.current_path)
            run_dir = (self.root / pointer["run_dir"]).resolve()
            manifest = read_json(run_dir / "manifest.json")
            for stage_name, expected_version in manifest.get("stage_versions", {}).items():
                stage_path = run_dir / f"{stage_name}.json"
                if not stage_path.exists():
                    self.last_current_error = f"snapshot stage {stage_name!r} is missing"
                    return None
                actual = "sha256:" + hashlib.sha256(stage_path.read_bytes()).hexdigest()
                if actual != expected_version:
                    self.last_current_error = f"snapshot stage {stage_name!r} hash mismatch"
                    return None
            ledger = read_json(run_dir / "ledger.json")
            ledger_version = ledger.get("ledger_version")
            if ledger_version != manifest.get("ledger_version"):
                self.last_current_error = "ledger and manifest versions differ"
                return None
            return CurrentSnapshot(pointer=pointer, manifest=manifest, run_dir=run_dir)
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            self.last_current_error = "CURRENT pointer or referenced snapshot is unreadable"
            return None

    def stage_versions(self, run_dir: Path) -> dict[str, str]:
        versions: dict[str, str] = {}
        for path in sorted(run_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            versions[path.stem] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return versions

    def commit(
        self,
        *,
        global_version: str,
        config_hash: str,
        requests_hash: str,
        receipt_fingerprints: dict[str, str],
        fault_config_hash: str,
        command_cursor: int,
        stages: dict[str, Any],
        extra_manifest: dict[str, Any] | None = None,
    ) -> CurrentSnapshot:
        run_token = uuid.uuid4().hex
        staging = self.staging_dir / run_token
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for name in sorted(stages):
                path = staging / f"{name}.json"
                path.write_text(canonical_json(stages[name]) + "\n", encoding="utf-8")
                with path.open("rb+") as handle:
                    os.fsync(handle.fileno())
            stage_versions = {
                path.stem: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(staging.glob("*.json"))
            }
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "global_version": global_version,
                "config_hash": config_hash,
                "requests_hash": requests_hash,
                "receipt_file_fingerprints": dict(sorted(receipt_fingerprints.items())),
                "fault_config_hash": fault_config_hash,
                "command_cursor": command_cursor,
                "stage_versions": stage_versions,
                "ledger_version": stages["ledger"].get("ledger_version", ""),
            }
            if extra_manifest:
                manifest.update(extra_manifest)
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
            with manifest_path.open("rb+") as handle:
                os.fsync(handle.fileno())
            manifest_hash = sha256_text(canonical_json(manifest))
            run_name = f"run-{global_version.removeprefix('sha256:')[:20]}-{manifest_hash[:12]}"
            final_dir = self.runs_dir / run_name
            if final_dir.exists():
                shutil.rmtree(staging)
            else:
                os.replace(staging, final_dir)
            pointer = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "global_version": global_version,
                "config_hash": config_hash,
                "run_dir": final_dir.relative_to(self.root).as_posix(),
                "manifest_hash": manifest_hash,
            }
            atomic_write_json(self.current_path, pointer)
            return CurrentSnapshot(pointer=pointer, manifest=manifest, run_dir=final_dir)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _cache_path(self, namespace: str, key: str) -> Path:
        safe_key = key.removeprefix("sha256:")
        path = self.cache_dir / namespace / f"{safe_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_cache(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self._cache_path(namespace, key)
        if not path.exists():
            return None
        try:
            value = read_json(path)
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def save_cache(self, namespace: str, key: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self._cache_path(namespace, key), payload)

    def resume_status(
        self,
        *,
        global_version: str,
        config_hash: str,
        command_cursor: int | None = None,
    ) -> dict[str, Any]:
        self.last_current_error = ""
        current = self.load_current()
        if current is None:
            corrupt = self.current_path.exists()
            return {
                "status": "corrupt" if corrupt else "cold",
                "resume_available": False,
                "config_changed": False,
                "previous_global_version": None,
                "error_code": "E_SNAPSHOT_CORRUPT" if corrupt else None,
                "error": self.last_current_error if corrupt else None,
            }
        current_global = current.manifest.get("global_version")
        current_config = current.manifest.get("config_hash")
        current_cursor = current.manifest.get("command_cursor")
        if (
            current_global == global_version
            and current_config == config_hash
            and (command_cursor is None or current_cursor == command_cursor)
        ):
            status = "resume_available"
        elif current_global == global_version and current_config == config_hash:
            status = "cursor_changed"
        elif current_config != config_hash:
            status = "config_changed"
        else:
            status = "input_changed"
        return {
            "status": status,
            "resume_available": status == "resume_available",
            "config_changed": current_config != config_hash,
            "previous_global_version": current_global,
            "previous_config_hash": current_config,
            "command_cursor": current_cursor,
        }

    def load_resumed_ledger(
        self,
        *,
        global_version: str,
        config_hash: str,
        command_cursor: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        current = self.load_current()
        if current is None:
            return None
        if current.manifest.get("global_version") != global_version:
            return None
        if current.manifest.get("config_hash") != config_hash:
            return None
        if (
            command_cursor is not None
            and current.manifest.get("command_cursor") != command_cursor
        ):
            return None
        try:
            ledger = read_json(current.run_dir / "ledger.json")
        except (OSError, json.JSONDecodeError):
            return None
        return ledger, dict(current.manifest)
