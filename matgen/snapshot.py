"""Whole-directory staged snapshots with versioned recovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cache import CacheEntry, CacheStore
from .errors import E_SNAPSHOT_IO, E_SNAPSHOT_VERSION, Issue, make_issue
from .hashing import fingerprint, sha256_bytes
from .io_utils import atomic_write_json, replace_directory
from .models import GenerationConfig


class SnapshotManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.current = self.root / "current"
        self.recovery_issues: list[Issue] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover()

    def _read_manifest_at(self, directory: Path) -> dict[str, Any] | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                return None
            return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def _backup_directories(self) -> list[Path]:
        return sorted(
            (path for path in self.root.glob(".previous-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _recover(self) -> None:
        backups = self._backup_directories()
        current_manifest = self._read_manifest_at(self.current)
        if current_manifest is None:
            if self.current.exists():
                shutil.rmtree(self.current, ignore_errors=True)
            for backup in backups:
                if self._read_manifest_at(backup) is not None:
                    os.replace(backup, self.current)
                    current_manifest = self._read_manifest_at(self.current)
                    break
        if current_manifest is not None:
            for backup in backups:
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
        for staging in self.root.glob(".staging-*"):
            shutil.rmtree(staging, ignore_errors=True)

    def load_manifest(self) -> dict[str, Any] | None:
        return self._read_manifest_at(self.current)

    def check_resume(
        self,
        *,
        input_config_hash: str,
        commands_hash: str,
        expected_version: int | None = None,
    ) -> tuple[bool, dict[str, Any] | None, list[Issue]]:
        manifest = self.load_manifest()
        if manifest is None:
            return False, None, []
        issues: list[Issue] = []
        reasons: list[str] = []
        if manifest.get("schema_version") != 1:
            reasons.append("snapshot schema_version mismatch")
        if manifest.get("input_config_hash") != input_config_hash:
            reasons.append("input config hash mismatch")
        if manifest.get("commands_hash") != commands_hash:
            reasons.append("command stream hash mismatch")
        if expected_version is not None and manifest.get("global_version") != expected_version:
            reasons.append(
                f"global version mismatch: expected {expected_version}, found {manifest.get('global_version')}"
            )
        if reasons:
            issues.append(
                make_issue(
                    E_SNAPSHOT_VERSION,
                    "; ".join(reasons),
                    source=str(self.current),
                    details={
                        "expected_input_config_hash": input_config_hash,
                        "snapshot_input_config_hash": manifest.get("input_config_hash"),
                        "expected_commands_hash": commands_hash,
                        "snapshot_commands_hash": manifest.get("commands_hash"),
                    },
                )
            )
            return False, manifest, issues
        return True, manifest, issues

    def restore_cache(self, cache: CacheStore) -> list[Issue]:
        entries_path = self.current / "cache_entries.json"
        arrays_path = self.current / "cache.npz"
        if not entries_path.exists() or not arrays_path.exists():
            return []
        issues: list[Issue] = []
        try:
            with entries_path.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError("cache_entries.json root must be a list")
            with np.load(arrays_path, allow_pickle=False) as archive:
                for row in rows:
                    key = str(row["key"])
                    array_names = list(row.get("array_names", []))
                    arrays = {
                        name.removeprefix(f"{row['index']}__"): np.asarray(archive[f"{row['index']}__{name}"]).copy()
                        for name in array_names
                    }
                    cache.entries[key] = CacheEntry(
                        kind=str(row["kind"]),
                        key=key,
                        meta=dict(row.get("meta", {})),
                        arrays=arrays,
                    )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            cache.entries.clear()
            issues.append(
                make_issue(
                    E_SNAPSHOT_IO,
                    f"cannot restore snapshot cache: {exc}",
                    source=str(self.current),
                )
            )
        return issues

    def read_ledger(self) -> list[dict[str, Any]]:
        path = self.current / "ledger.json"
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, list) else []
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []

    def commit(
        self,
        *,
        cache: CacheStore,
        config: GenerationConfig,
        input_config_hash: str,
        commands_hash: str,
        command_cursor: int,
        completed_batches_cursor: int,
        phase_hashes: dict[str, str],
        ledger: list[dict[str, Any]],
        force: bool = False,
    ) -> dict[str, Any]:
        entries = cache.all_entries()
        entry_index: list[dict[str, Any]] = []
        arrays_out: dict[str, np.ndarray] = {}
        artifact_versions: dict[str, list[dict[str, str]]] = {}
        for index, entry in enumerate(entries):
            array_names: list[str] = []
            for name, values in sorted(entry.arrays.items()):
                archive_name = f"{index}__{name}"
                arrays_out[archive_name] = np.asarray(values)
                array_names.append(name)
            entry_index.append(
                {
                    "index": index,
                    "key": entry.key,
                    "kind": entry.kind,
                    "meta": entry.meta,
                    "array_names": array_names,
                }
            )
            version = hashlib.sha256(entry.key.encode("ascii")).hexdigest()[:16]
            artifact_versions.setdefault(entry.kind, []).append({"key": entry.key, "version": version})

        state_hash = fingerprint(
            {
                "keys": [(entry.kind, entry.key) for entry in entries],
                "config_hash": input_config_hash,
                "commands_hash": commands_hash,
                "command_cursor": command_cursor,
                "completed_batches_cursor": completed_batches_cursor,
                "phase_hashes": phase_hashes,
            }
        )
        previous = self.load_manifest()
        if previous is not None and previous.get("state_hash") == state_hash and not force:
            return {
                "status": "skipped",
                "global_version": previous.get("global_version"),
                "state_hash": state_hash,
            }
        previous_version = int(previous.get("global_version", 0)) if previous else 0
        global_version = previous_version + 1
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        backup = self.root / f".previous-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            np.savez_compressed(staging / "cache.npz", **arrays_out)
            with (staging / "cache.npz").open("rb") as handle:
                os.fsync(handle.fileno())
            atomic_write_json(staging / "cache_entries.json", entry_index)
            atomic_write_json(staging / "ledger.json", ledger)
            manifest = {
                "schema_version": 1,
                "global_version": global_version,
                "input_config_hash": input_config_hash,
                "config_hash": fingerprint(config.as_dict()),
                "commands_hash": commands_hash,
                "quantization": {
                    "bit_width": config.fixed_point.bit_width,
                    "fraction_bits": config.fixed_point.fraction_bits,
                    "quantization_level": config.fixed_point.quantization_level,
                    "quantization_tiers": list(config.fixed_point.quantization_tiers),
                },
                "command_cursor": command_cursor,
                "completed_batches_cursor": completed_batches_cursor,
                "phases": {
                    "load_validation": {
                        "version": 1,
                        "hash": phase_hashes.get("load_validation", ""),
                        "status": "committed",
                    },
                    "layer_noise": {
                        "version": 1,
                        "hash": phase_hashes.get("layer_noise", ""),
                        "status": "committed",
                    },
                    "material_mix": {
                        "version": 1,
                        "hash": phase_hashes.get("material_mix", ""),
                        "status": "committed",
                    },
                    "channel_synthesis": {
                        "version": 1,
                        "hash": phase_hashes.get("channel_synthesis", ""),
                        "status": "committed",
                    },
                },
                "artifact_versions": artifact_versions,
                "state_hash": state_hash,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(staging / "manifest.json", manifest)
            for path in staging.iterdir():
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            try:
                directory_fd = os.open(staging, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            replace_directory(staging, self.current, backup)
            return {
                "status": "committed",
                "global_version": global_version,
                "state_hash": state_hash,
                "artifact_count": len(entries),
                "command_cursor": command_cursor,
                "completed_batches_cursor": completed_batches_cursor,
                "phase_hashes": phase_hashes,
            }
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not self.current.exists():
                os.replace(backup, self.current)
            raise RuntimeError(f"snapshot commit failed: {exc}") from exc


def phase_hashes_from_result(result: Any) -> dict[str, str]:
    texture_keys = sorted(
        entry["key"] for entry in result.ledger_entries if entry["kind"] == "texture"
    )
    layer_keys = sorted(
        entry["key"] for entry in result.ledger_entries if entry["kind"] == "noise_layer"
    )
    channel_keys = sorted(
        entry["key"] for entry in result.ledger_entries if entry["kind"] == "material_channel"
    )
    final_keys = sorted(
        entry["key"] for entry in result.ledger_entries if entry["kind"] == "final_synthesis"
    )
    return {
        "load_validation": sha256_bytes("\n".join(texture_keys).encode("ascii")),
        "layer_noise": sha256_bytes("\n".join(layer_keys).encode("ascii")),
        "material_mix": sha256_bytes("\n".join(channel_keys).encode("ascii")),
        "channel_synthesis": sha256_bytes("\n".join(final_keys).encode("ascii")),
    }
