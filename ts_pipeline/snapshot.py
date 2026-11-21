"""Crash-safe staged snapshots and stale-generation detection."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, fingerprint, read_json, to_builtin
from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_STALE


SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotManager:
    def __init__(
        self,
        state_dir: Path,
        *,
        snapshot_version: int,
        config_hash: str,
        injection_hash: str,
        samples_hash: str,
    ):
        self.state_dir = state_dir
        self.snapshot_root = state_dir / "snapshot"
        self.generations_root = self.snapshot_root / "generations"
        self.generations_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.snapshot_root / "manifest.json"
        self.snapshot_version = snapshot_version
        self.config_hash = config_hash
        self.injection_hash = injection_hash
        self.samples_hash = samples_hash
        self.previous_manifest: dict[str, Any] | None = None
        self.previous_generation_dir: Path | None = None
        self.current_generation_dir: Path | None = None
        self.inventory: dict[tuple[str, str], str] = {}
        self.status_code = "EMPTY"
        self.stale_reasons: list[str] = []
        self.revision = 1
        self._stage_baseline_initialized = False
        self._load_previous()

    def _load_previous(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            manifest = read_json(self.manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("manifest root is not an object")
        except Exception as exc:
            self.status_code = E_SNAPSHOT_CORRUPT
            self.stale_reasons.append(f"manifest cannot be read: {exc}")
            return
        self.previous_manifest = manifest
        generation_name = manifest.get("generation")
        if isinstance(generation_name, str):
            path = self.generations_root / generation_name
            if path.is_dir():
                self.previous_generation_dir = path
        self.revision = int(manifest.get("revision", 0)) + 1
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            self.status_code = E_SNAPSHOT_STALE
            self.stale_reasons.append("snapshot schema version mismatch")
        if manifest.get("version") != self.snapshot_version:
            self.status_code = E_SNAPSHOT_STALE
            self.stale_reasons.append("configured snapshot version mismatch")
        if manifest.get("config_hash") != self.config_hash:
            self.status_code = E_SNAPSHOT_STALE
            self.stale_reasons.append("config hash mismatch")
        if self.status_code == "EMPTY":
            self.status_code = "RESUMED"
        for entry in manifest.get("cache_inventory", []):
            if not isinstance(entry, dict):
                continue
            namespace = entry.get("namespace")
            key = entry.get("key")
            if isinstance(namespace, str) and isinstance(key, str):
                self.inventory[(namespace, key)] = str(entry.get("sha256", ""))

    @property
    def committed_inventory(self) -> set[tuple[str, str]]:
        return set(self.inventory)

    @property
    def committed_inventory_map(self) -> dict[tuple[str, str], str]:
        return dict(self.inventory)

    def register_inventory(self, entries: list[dict[str, str]]) -> None:
        for entry in entries:
            self.inventory[(entry["namespace"], entry["key"])] = entry.get("sha256", "")

    def _new_generation_name(self) -> str:
        payload = fingerprint(
            {
                "revision": self.revision,
                "config_hash": self.config_hash,
                "injection_hash": self.injection_hash,
                "samples_hash": self.samples_hash,
                "nonce": uuid.uuid4().hex,
            }
        )[:16]
        return f"gen-{self.revision:06d}-{payload}"

    def commit(
        self,
        *,
        stage: str,
        stage_payload: dict[str, Any],
        progress: dict[str, Any],
        status: str = "complete",
    ) -> dict[str, Any]:
        generation_name = self._new_generation_name()
        generation_dir = self.generations_root / generation_name
        temporary = self.generations_root / f".tmp-{generation_name}-{uuid.uuid4().hex}"
        if self.current_generation_dir is not None and self.current_generation_dir.is_dir():
            shutil.copytree(self.current_generation_dir, temporary)
        else:
            temporary.mkdir(parents=True)
        stage_path = temporary / "stages" / f"{stage}.json"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage_path, stage_payload)
        shutil.move(str(temporary), str(generation_dir))

        previous_stages: dict[str, Any] = {}
        if not self._stage_baseline_initialized:
            if self.status_code not in {E_SNAPSHOT_STALE, E_SNAPSHOT_CORRUPT}:
                if self.previous_manifest is not None:
                    previous_stages.update(self.previous_manifest.get("stages", {}))
            self._stage_baseline_initialized = True
        else:
            if self.current_generation_dir is not None and self.manifest_path.exists():
                try:
                    previous_stages.update(read_json(self.manifest_path).get("stages", {}))
                except Exception:
                    pass
        previous_stages[stage] = {
            "status": status,
            "product_fingerprint": fingerprint(stage_payload),
        }
        inventory_entries = [
            {"namespace": namespace, "key": key, "sha256": digest}
            for (namespace, key), digest in sorted(self.inventory.items())
        ]
        progress_value = to_builtin(progress)
        task_fingerprints = progress_value.pop("task_fingerprints", {})
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "version": self.snapshot_version,
            "revision": self.revision,
            "generation": generation_name,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "samples_hash": self.samples_hash,
            "stages": previous_stages,
            "progress": progress_value,
            "task_fingerprints": task_fingerprints,
            "cache_inventory": inventory_entries,
            "status_code": self.status_code,
            "stale_reasons": self.stale_reasons,
        }
        atomic_write_json(self.manifest_path, manifest)
        old_generation = self.current_generation_dir
        self.current_generation_dir = generation_dir
        self.previous_manifest = manifest
        self.revision += 1
        if old_generation is not None and old_generation != generation_dir and old_generation.is_dir():
            try:
                shutil.rmtree(old_generation)
            except OSError:
                pass
        return manifest

    def snapshot_summary(self) -> dict[str, Any]:
        manifest = self.previous_manifest or {}
        return {
            "status_code": self.status_code,
            "stale": self.status_code == E_SNAPSHOT_STALE,
            "stale_reasons": list(self.stale_reasons),
            "version": f"{self.snapshot_version}:{manifest.get('revision', self.revision)}",
            "revision": manifest.get("revision"),
            "generation": manifest.get("generation"),
            "config_hash": manifest.get("config_hash", self.config_hash),
            "current_config_hash": self.config_hash,
            "injection_hash": manifest.get("injection_hash", self.injection_hash),
            "samples_hash": manifest.get("samples_hash", self.samples_hash),
            "stages": manifest.get("stages", {}),
            "progress": manifest.get("progress", {}),
        }

    def previous_task_fingerprints(self) -> dict[str, Any]:
        if not self.previous_manifest:
            return {}
        value = self.previous_manifest.get("task_fingerprints", {})
        return value if isinstance(value, dict) else {}
