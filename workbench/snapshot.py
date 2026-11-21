"""Atomic whole-snapshot state and content-addressed stage artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import E_CACHE_CORRUPT, E_SNAPSHOT_STALE
from .hashing import hash_json, sha256_bytes, digest_short


ENGINE_VERSION = "workbench-snapshot-v1"
MANIFEST_SCHEMA_VERSION = 1
STAGES = ("visual", "text", "alignment", "retrieval")


def _manifest_checksum(manifest: dict[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_checksum"}
    return hash_json(body)


class SnapshotStore:
    def __init__(
        self,
        output_dir: Path | str,
        *,
        config_hash: str,
        full_rebuild: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.state_root = self.output_dir / "state"
        self.current_dir = self.state_root / "current"
        self.backup_dir = self.state_root / "current.backup"
        self.config_hash = config_hash
        self.full_rebuild = full_rebuild
        self.staging_dir: Path | None = None
        self.manifest: dict[str, Any] = {}
        self.previous_manifest: dict[str, Any] | None = None
        self.status = "cold"
        self.stale_reasons: list[str] = []
        self.reuse_enabled = not full_rebuild
        self._cache: dict[str, dict[str, Any]] = {}
        self._artifact_index: dict[str, dict[str, set[str]]] = {
            stage: {} for stage in STAGES
        }
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._recover_backup()
        self._load_previous()
        self._prepare_staging(copy_existing=not full_rebuild)
        self._load_artifact_index()

    def _recover_backup(self) -> None:
        if self.current_dir.exists() and self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)
        elif not self.current_dir.exists() and self.backup_dir.exists():
            os.replace(self.backup_dir, self.current_dir)

    def _load_previous(self) -> None:
        manifest_path = self.current_dir / "manifest.json"
        if not manifest_path.exists():
            self.status = "cold"
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.status = E_SNAPSHOT_STALE
            self.stale_reasons.append(f"manifest unreadable: {exc}")
            self.reuse_enabled = False
            return
        self.previous_manifest = manifest
        checksum = manifest.get("manifest_checksum")
        if checksum != _manifest_checksum(manifest):
            self.status = E_SNAPSHOT_STALE
            self.stale_reasons.append("manifest checksum mismatch")
            self.reuse_enabled = False
            return
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            self.status = E_SNAPSHOT_STALE
            self.stale_reasons.append("manifest schema version changed")
            self.reuse_enabled = False
            return
        if manifest.get("engine_version") != ENGINE_VERSION:
            self.status = E_SNAPSHOT_STALE
            self.stale_reasons.append("snapshot engine version changed")
            self.reuse_enabled = False
            return
        if manifest.get("config_hash") != self.config_hash:
            self.status = E_SNAPSHOT_STALE
            self.stale_reasons.append("configuration hash changed")
            # Content-addressed artifacts remain individually safe to reuse.
            self.reuse_enabled = not self.full_rebuild
        else:
            self.status = "resumed"

    def _prepare_staging(self, *, copy_existing: bool = True) -> None:
        self.staging_dir = Path(
            tempfile.mkdtemp(prefix=".snapshot-staging-", dir=str(self.state_root))
        )
        if copy_existing and self.current_dir.exists():
            shutil.copytree(self.current_dir, self.staging_dir, dirs_exist_ok=True)

    def _stage_dir(self, stage: str, root: Path | None = None) -> Path:
        return (root or self.staging_dir) / "stages" / stage  # type: ignore[operator]

    def _load_artifact_index(self) -> None:
        if not self.reuse_enabled or not self.current_dir.exists():
            return
        for stage in STAGES:
            directory = self.current_dir / "stages" / stage
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                self._artifact_index[stage].setdefault(path.stem, set()).add("current")

    def _artifact_path(self, stage: str, key: str) -> Path:
        return self._stage_dir(stage) / f"{key}.json"

    def has(self, stage: str, key: str) -> bool:
        if not self.reuse_enabled:
            return False
        path = self._artifact_path(stage, key)
        return path.is_file() and key in self._artifact_index.get(stage, {})

    def get(self, stage: str, key: str) -> dict[str, Any] | None:
        cache_key = f"{stage}:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.has(stage, key):
            return None
        path = self._artifact_path(stage, key)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{E_CACHE_CORRUPT}: cannot read {stage}:{key}: {exc}") from exc
        self._cache[cache_key] = artifact
        return artifact

    def put(self, stage: str, key: str, artifact: dict[str, Any]) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        directory = self._stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.json"
        payload = json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        self._cache[f"{stage}:{key}"] = artifact
        self._artifact_index.setdefault(stage, {}).setdefault(key, set()).add("staging")

    def artifact_keys(self, stage: str) -> list[str]:
        keys = set(self._artifact_index.get(stage, {}))
        directory = self._stage_dir(stage)
        if directory.exists():
            keys.update(path.stem for path in directory.glob("*.json"))
        return sorted(keys)

    def stage_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        previous_stages = (self.previous_manifest or {}).get("stages", {})
        for stage in STAGES:
            keys = self.artifact_keys(stage)
            digest = hash_json(keys)
            metadata[stage] = {
                "artifact_version": f"stage-v1:{digest_short(digest)}",
                "count": len(keys),
                "keys": keys,
                "previous": (
                    {
                        "artifact_version": previous_stages[stage].get("artifact_version"),
                        "count": previous_stages[stage].get("count"),
                    }
                    if stage in previous_stages
                    else None
                ),
            }
        return metadata

    def commit(
        self,
        *,
        completed_samples: list[str],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert self.staging_dir is not None
        previous_version = int((self.previous_manifest or {}).get("global_version", 0))
        global_version = previous_version + 1
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "global_version": global_version,
            "config_hash": self.config_hash,
            "completed_samples": sorted(set(completed_samples)),
            "stages": self.stage_metadata(),
        }
        if extra:
            manifest.update(extra)
        manifest["manifest_checksum"] = _manifest_checksum(manifest)
        manifest_path = self.staging_dir / "manifest.json"
        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(self.staging_dir))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, manifest_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        if self.current_dir.exists():
            os.replace(self.current_dir, self.backup_dir)
        os.replace(self.staging_dir, self.current_dir)
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.previous_manifest = manifest
        self.manifest = manifest
        self._cache.clear()
        self._artifact_index = {stage: {} for stage in STAGES}
        self._prepare_staging()
        self._load_artifact_index()
        return manifest

    def snapshot_report(self) -> dict[str, Any]:
        previous_version = int((self.previous_manifest or {}).get("global_version", 0))
        return {
            "status": self.status,
            "business_code": E_SNAPSHOT_STALE if self.status == E_SNAPSHOT_STALE else None,
            "resume_enabled": self.status == "resumed" and self.reuse_enabled,
            "reuse_enabled": self.reuse_enabled,
            "stale_reasons": list(self.stale_reasons),
            "previous_global_version": previous_version,
            "current_global_version": int((self.manifest or {}).get("global_version", 0)),
            "config_hash": self.config_hash,
            "previous_config_hash": (self.previous_manifest or {}).get("config_hash"),
            "completed_samples": list((self.manifest or {}).get("completed_samples", [])),
            "stages": (self.manifest or {}).get("stages", {}),
        }

    def cleanup_uncommitted(self) -> None:
        if self.staging_dir and self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())
