from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_STALE, WorkbenchError


ARTIFACT_NAMES = ("load", "emission", "decode", "merge", "runtime")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class SnapshotPaths:
    root: Path
    versions: Path
    current: Path

    @classmethod
    def under(cls, output_dir: Path) -> "SnapshotPaths":
        root = output_dir / "snapshots"
        return cls(root=root, versions=root / "versions", current=root / "CURRENT")


class SnapshotManager:
    """Atomic, same-volume staged snapshot store.

    New artifacts are written under ``snapshots/.staging-*`` and renamed as a
    complete version directory.  A relative ``CURRENT`` symlink is then swapped
    with ``os.replace`` so readers never observe a half-written snapshot.
    """

    def __init__(self, output_dir: Path):
        self.paths = SnapshotPaths.under(output_dir)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.versions.mkdir(parents=True, exist_ok=True)

        root_device = self.paths.root.stat().st_dev
        output_device = output_dir.stat().st_dev
        if root_device != output_device:
            raise WorkbenchError(
                E_SNAPSHOT_STALE,
                "snapshot",
                "snapshot staging and formal directories are not on the same filesystem",
            )

    def current_manifest(self) -> dict[str, Any] | None:
        current = self.paths.current
        if not current.exists():
            return None
        manifest_path = current / "manifest.json"
        if not manifest_path.exists():
            raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", "CURRENT exists but manifest.json is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", f"current snapshot manifest is unreadable: {exc}") from exc
        if not isinstance(manifest, dict):
            raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", "snapshot manifest root is not an object")
        return manifest

    def validate_resume(
        self,
        *,
        snapshot_format_version: int,
        config_hash: str,
        param_version: str,
        param_hash: str,
        corpus_hash: str,
        expected_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = self.current_manifest()
        if manifest is None:
            raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", "no CURRENT snapshot is available to resume")
        checks: list[str] = []
        if int(manifest.get("snapshot_format_version", -1)) != int(snapshot_format_version):
            checks.append(
                f"snapshot format {manifest.get('snapshot_format_version')} != {snapshot_format_version}"
            )
        if expected_version is not None and int(manifest.get("global_version", -1)) != int(expected_version):
            checks.append(f"global version {manifest.get('global_version')} != expected {expected_version}")
        if manifest.get("config_hash") != config_hash:
            checks.append("config hash mismatch")
        if manifest.get("param_version") != param_version:
            checks.append("model version mismatch")
        if manifest.get("param_hash") != param_hash:
            checks.append("parameter package hash mismatch")
        if manifest.get("corpus_hash") != corpus_hash:
            checks.append("corpus hash mismatch")
        if checks:
            raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", "; ".join(checks))
        artifacts = self.read_artifacts(manifest)
        return manifest, artifacts

    def read_artifacts(self, manifest: dict[str, Any]) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        current = self.paths.current
        expected_hashes = manifest.get("artifact_sha256", {})
        for name in ARTIFACT_NAMES:
            path = current / f"{name}.json"
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", f"snapshot artifact missing: {name}.json") from exc
            actual_hash = sha256_bytes(raw)
            if expected_hashes.get(name) != actual_hash:
                raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", f"snapshot artifact hash mismatch: {name}.json")
            try:
                artifacts[name] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", f"snapshot artifact invalid JSON: {name}.json") from exc
        return artifacts

    def commit(
        self,
        artifacts: dict[str, Any],
        *,
        snapshot_format_version: int,
        config_hash: str,
        param_version: str,
        param_hash: str,
        corpus_hash: str,
        completed_sample_ids: list[str],
        consumed_char_budget: int,
        recovery_point_sample_id: str | None,
        status: str,
    ) -> dict[str, Any]:
        missing = [name for name in ARTIFACT_NAMES if name not in artifacts]
        if missing:
            raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", "missing artifacts: " + ", ".join(missing))

        previous = self.current_manifest()
        previous_version = int(previous.get("global_version", 0)) if previous else 0
        new_version = previous_version + 1
        previous_versions = previous.get("stage_versions", {}) if previous else {}
        previous_hashes = previous.get("artifact_sha256", {}) if previous else {}

        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.paths.root))
        artifact_hashes: dict[str, str] = {}
        try:
            for name in ARTIFACT_NAMES:
                payload = _json_bytes(artifacts[name])
                target = staging / f"{name}.json"
                with target.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                artifact_hashes[name] = sha256_bytes(payload)

            stage_versions: dict[str, int] = {}
            for name in ARTIFACT_NAMES:
                prior_version = int(previous_versions.get(name, 0))
                if artifact_hashes[name] != previous_hashes.get(name):
                    stage_versions[name] = prior_version + 1
                else:
                    stage_versions[name] = prior_version

            manifest = {
                "global_version": new_version,
                "snapshot_format_version": int(snapshot_format_version),
                "config_hash": config_hash,
                "param_version": param_version,
                "param_hash": param_hash,
                "corpus_hash": corpus_hash,
                "completed_sample_ids": sorted(set(completed_sample_ids)),
                "consumed_char_budget": int(consumed_char_budget),
                "stage_versions": stage_versions,
                "artifact_sha256": artifact_hashes,
                "recovery_point_sample_id": recovery_point_sample_id,
                "status": status,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = staging / "manifest.json"
            with manifest_path.open("wb") as handle:
                handle.write(_json_bytes(manifest))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(staging)

            target_version_dir = self.paths.versions / f"v{new_version:08d}"
            if target_version_dir.exists():
                raise WorkbenchError(E_SNAPSHOT_STALE, "snapshot", f"snapshot version already exists: {new_version}")
            os.replace(staging, target_version_dir)
            _fsync_directory(self.paths.versions)

            symlink_tmp = self.paths.root / f".CURRENT.tmp-{os.getpid()}-{new_version}"
            if symlink_tmp.exists() or symlink_tmp.is_symlink():
                symlink_tmp.unlink()
            os.symlink(os.path.join("versions", f"v{new_version:08d}"), symlink_tmp)
            os.replace(symlink_tmp, self.paths.current)
            _fsync_directory(self.paths.root)
            return manifest
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()
