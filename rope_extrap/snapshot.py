from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_json
from .cache import ResultCache, build_stage_keys
from .engine import DiagnosticEngine
from .errors import E_SNAPSHOT_STALE, Failure
from .models import CacheEntry, STAGES
from .serialization import cache_entry_from_dict, cache_entry_to_dict


ARTIFACT_FILENAMES = {
    "position_encoding": "position_encoding.json",
    "attention_score": "attention_score.json",
    "error_trace": "error_trace.json",
}


@dataclass(frozen=True)
class SnapshotLoadResult:
    ok: bool
    manifest: dict[str, Any] | None
    entries: tuple[CacheEntry, ...]
    failures: tuple[Failure, ...]

    def apply_to(self, cache: ResultCache) -> None:
        for entry in self.entries:
            cache.put(entry)


class SnapshotStore:
    """Versioned snapshot writer with an atomic manifest pointer.

    All stage files are first written and fsynced under a staging directory.
    That directory is renamed into its immutable version directory, then the
    small manifest pointer is atomically replaced.  Readers therefore see
    either the previous complete snapshot or the next complete snapshot.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def commit(
        self,
        engine: DiagnosticEngine,
        *,
        global_version: int,
        completed_samples: list[str] | None = None,
        failed_samples: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = engine.config
        config_hash = config.config_fingerprint()
        serialized_by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
        current_complete: list[str] = []
        for sample_id in sorted(engine.samples):
            sample = engine.samples[sample_id]
            keys = build_stage_keys(config, sample)
            present = True
            for stage in STAGES:
                entry = engine.cache.get(stage, keys.as_dict()[stage])
                if entry is None:
                    present = False
                    continue
                serialized_by_stage[stage].append(cache_entry_to_dict(entry))
            if present:
                current_complete.append(sample_id)
        for stage in STAGES:
            serialized_by_stage[stage].sort(key=lambda item: (item["scheme"], item["sample_id"], item["stage"]))

        artifact_versions = {
            stage: sha256_json(serialized_by_stage[stage]) for stage in STAGES
        }
        stage_payloads = {
            stage: {
                "schema_version": "1.0.0",
                "stage": stage,
                "artifact_version": artifact_versions[stage],
                "entries": serialized_by_stage[stage],
            }
            for stage in STAGES
        }

        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            for stage in STAGES:
                _write_json_synced(staging / ARTIFACT_FILENAMES[stage], stage_payloads[stage])
            version_dir_name = f"v{global_version:08d}-{config_hash[:12]}"
            version_dir = self.root / version_dir_name
            if version_dir.exists():
                # Repeating a commit for the same version must be idempotent.
                existing = self._read_versions(version_dir)
                if existing != artifact_versions:
                    shutil.rmtree(staging)
                    raise ValueError(f"snapshot version directory already exists with different content: {version_dir}")
                shutil.rmtree(staging)
            else:
                os.replace(staging, version_dir)

            manifest = {
                "schema_version": "1.0.0",
                "global_version": int(global_version),
                "config_hash": config_hash,
                "config": config.to_dict(),
                "snapshot_dir": version_dir_name,
                "artifact_versions": artifact_versions,
                "artifact_files": dict(ARTIFACT_FILENAMES),
                "completed_samples": sorted(completed_samples if completed_samples is not None else current_complete),
                "failed_samples": sorted(failed_samples or []),
                "metadata": metadata or {},
            }
            _write_json_atomic(self.manifest_path, manifest)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def load(
        self,
        *,
        expected_version: int | None = None,
        expected_config_hash: str | None = None,
    ) -> SnapshotLoadResult:
        if not self.manifest_path.exists():
            failure = Failure(
                E_SNAPSHOT_STALE,
                "snapshot manifest does not exist",
                stage="snapshot",
                scope="snapshot",
            )
            return SnapshotLoadResult(False, None, (), (failure,))
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failure = Failure(
                E_SNAPSHOT_STALE,
                f"snapshot manifest is unreadable: {exc}",
                stage="snapshot",
                scope="snapshot",
            )
            return SnapshotLoadResult(False, None, (), (failure,))
        mismatches: list[str] = []
        if expected_version is not None and int(manifest.get("global_version", -1)) != int(expected_version):
            mismatches.append(
                f"version mismatch: manifest={manifest.get('global_version')} expected={expected_version}"
            )
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            mismatches.append("config hash mismatch")
        if mismatches:
            failure = Failure(
                E_SNAPSHOT_STALE,
                "; ".join(mismatches),
                stage="snapshot",
                scope="snapshot",
            )
            return SnapshotLoadResult(False, manifest, (), (failure,))

        version_dir = self.root / str(manifest.get("snapshot_dir", ""))
        entries: list[CacheEntry] = []
        try:
            for stage in STAGES:
                payload = json.loads(
                    (version_dir / ARTIFACT_FILENAMES[stage]).read_text(encoding="utf-8")
                )
                actual_version = sha256_json(payload.get("entries", []))
                expected_artifact_version = manifest.get("artifact_versions", {}).get(stage)
                if actual_version != expected_artifact_version:
                    raise ValueError(f"artifact hash mismatch for stage {stage}")
                entries.extend(cache_entry_from_dict(item) for item in payload.get("entries", []))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
            failure = Failure(
                E_SNAPSHOT_STALE,
                f"snapshot artifacts are stale or corrupt: {exc}",
                stage="snapshot",
                scope="snapshot",
            )
            return SnapshotLoadResult(False, manifest, (), (failure,))
        entries.sort(key=lambda entry: entry.stable_sort_key())
        return SnapshotLoadResult(True, manifest, tuple(entries), ())

    def _read_versions(self, version_dir: Path) -> dict[str, str]:
        versions: dict[str, str] = {}
        for stage in STAGES:
            payload = json.loads((version_dir / ARTIFACT_FILENAMES[stage]).read_text(encoding="utf-8"))
            versions[stage] = sha256_json(payload.get("entries", []))
        return versions


def _write_json_synced(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

