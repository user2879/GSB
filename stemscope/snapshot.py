from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .cache import CacheStore, entry_to_disk, load_stage_entries
from .errors import DiagnosticError
from .hashing import canonical_json_bytes, sha256_json


STAGES = ("parse", "convert", "gain", "mix")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


class SnapshotStore:
    def __init__(self, root: str | Path, *, schema: str = "stemscope-snapshot-v1") -> None:
        self.root = Path(root)
        self.schema = schema
        self.stages_root = self.root / "stages"
        self.manifest_path = self.root / "manifest.json"

    def compute_global_version(self, input_fingerprint: str, config_hash: str) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "input_fingerprint": input_fingerprint,
                "config_hash": config_hash,
            }
        )[:24]

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            import json

            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def audit(
        self,
        *,
        expected_global_version: str | None = None,
        expected_config_hash: str | None = None,
        expected_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        manifest = self.read_manifest()
        if manifest is None:
            return {
                "status": "empty",
                "reusable": False,
                "errors": [],
                "manifest": None,
                "temp_dirs_remaining": self._temp_dirs_remaining(),
            }
        errors: list[DiagnosticError] = []
        if expected_global_version is not None and manifest.get("global_version") != expected_global_version:
            errors.append(
                DiagnosticError(
                    "E_SNAPSHOT_STALE",
                    "snapshot global version does not match the requested version",
                    stage="snapshot",
                    details={
                        "expected": expected_global_version,
                        "actual": manifest.get("global_version"),
                    },
                )
            )
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            errors.append(
                DiagnosticError(
                    "E_SNAPSHOT_STALE",
                    "snapshot config hash does not match the current config",
                    stage="snapshot",
                    details={
                        "expected": expected_config_hash,
                        "actual": manifest.get("config_hash"),
                    },
                )
            )
        if expected_formats is not None:
            actual_formats = sorted(manifest.get("format_versions", []))
            if actual_formats != sorted(expected_formats):
                errors.append(
                    DiagnosticError(
                        "E_SNAPSHOT_STALE",
                        "snapshot format version set does not match the requested format set",
                        stage="snapshot",
                        details={
                            "expected": sorted(expected_formats),
                            "actual": actual_formats,
                        },
                    )
                )
        return {
            "status": "stale" if errors else "valid",
            "reusable": not errors,
            "errors": [error.to_dict() for error in errors],
            "manifest": manifest,
            "temp_dirs_remaining": self._temp_dirs_remaining(),
        }

    def load_cache(self, audit: dict[str, Any] | None = None) -> CacheStore:
        audit = audit or self.audit()
        cache = CacheStore()
        if not audit.get("reusable"):
            return cache
        for stage in STAGES:
            stage_dir = self._current_stage_dir(stage)
            if stage_dir is None:
                continue
            for entry in load_stage_entries(stage_dir):
                cache.put(entry)
        return cache

    def commit_stage(
        self,
        stage: str,
        cache: CacheStore,
        *,
        global_version: str,
        config_hash: str,
        format_versions: list[str],
        completed_segments: list[int] | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown snapshot stage {stage}")
        stage_root = self.stages_root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        entries = cache.entries_for_stage(stage)
        artifact_version = cache.stage_version(stage)
        current_manifest = self._current_stage_manifest(stage)
        if (
            current_manifest is not None
            and current_manifest.get("artifact_version") == artifact_version
            and current_manifest.get("global_version") == global_version
            and current_manifest.get("config_hash") == config_hash
        ):
            return current_manifest

        temporary = Path(tempfile.mkdtemp(prefix=".tmp-stage-", dir=stage_root))
        try:
            for entry in entries:
                entry_to_disk(entry, temporary)
            manifest = {
                "schema": self.schema,
                "stage": stage,
                "global_version": global_version,
                "config_hash": config_hash,
                "format_versions": sorted(set(format_versions)),
                "completed_segments": sorted(completed_segments or []),
                "entry_count": len(entries),
                "artifact_version": artifact_version,
            }
            (temporary / "stage_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
            generation = f"gen-{int(time.time_ns())}-{uuid.uuid4().hex[:8]}"
            generations = stage_root / "generations"
            generations.mkdir(parents=True, exist_ok=True)
            final_dir = generations / generation
            os.replace(temporary, final_dir)
            pointer_tmp = stage_root / f".CURRENT.tmp-{uuid.uuid4().hex}"
            pointer_tmp.write_text(generation, encoding="utf-8")
            os.replace(pointer_tmp, stage_root / "CURRENT")
            self._cleanup_generations(stage_root, keep=2)
            return manifest
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def commit_manifest(
        self,
        *,
        global_version: str,
        config_hash: str,
        input_fingerprint: str,
        format_versions: list[str],
        completed_segments: list[int],
        stage_versions: dict[str, str],
        stage_entry_counts: dict[str, int],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "global_version": global_version,
            "config_hash": config_hash,
            "input_fingerprint": input_fingerprint,
            "format_versions": sorted(set(format_versions)),
            "completed_segments": sorted(completed_segments),
            "stage_versions": dict(sorted(stage_versions.items())),
            "stage_entry_counts": dict(sorted(stage_entry_counts.items())),
        }
        if extra:
            payload["extra"] = extra
        atomic_write(self.manifest_path, canonical_json_bytes(payload) + b"\n")
        return payload

    def checkpoint(
        self,
        cache: CacheStore,
        *,
        global_version: str,
        config_hash: str,
        input_fingerprint: str,
        format_versions: list[str],
        completed_segments: list[int],
        stages: tuple[str, ...] = ("gain", "mix"),
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage_versions: dict[str, str] = {}
        stage_counts: dict[str, int] = {}
        for stage in STAGES:
            if stage in stages:
                manifest = self.commit_stage(
                    stage,
                    cache,
                    global_version=global_version,
                    config_hash=config_hash,
                    format_versions=format_versions,
                    completed_segments=completed_segments,
                )
                stage_versions[stage] = manifest["artifact_version"]
                stage_counts[stage] = int(manifest["entry_count"])
            else:
                current = self._current_stage_manifest(stage)
                if current is not None:
                    stage_versions[stage] = current["artifact_version"]
                    stage_counts[stage] = int(current["entry_count"])
        return self.commit_manifest(
            global_version=global_version,
            config_hash=config_hash,
            input_fingerprint=input_fingerprint,
            format_versions=format_versions,
            completed_segments=completed_segments,
            stage_versions=stage_versions,
            stage_entry_counts=stage_counts,
            extra=extra,
        )

    def _current_stage_dir(self, stage: str) -> Path | None:
        stage_root = self.stages_root / stage
        pointer = stage_root / "CURRENT"
        if not pointer.exists():
            return None
        generation = pointer.read_text(encoding="utf-8").strip()
        candidate = stage_root / "generations" / generation
        return candidate if candidate.is_dir() else None

    def _current_stage_manifest(self, stage: str) -> dict[str, Any] | None:
        stage_dir = self._current_stage_dir(stage)
        if stage_dir is None:
            return None
        import json

        try:
            return json.loads((stage_dir / "stage_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _temp_dirs_remaining(self) -> int:
        if not self.stages_root.exists():
            return 0
        return sum(1 for path in self.stages_root.rglob(".tmp-*") if path.is_dir())

    def _cleanup_generations(self, stage_root: Path, keep: int) -> None:
        generations = stage_root / "generations"
        if not generations.exists():
            return
        current = (stage_root / "CURRENT").read_text(encoding="utf-8").strip()
        candidates = sorted(
            (path for path in generations.iterdir() if path.is_dir() and path.name != current),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in candidates[keep - 1 :]:
            shutil.rmtree(path, ignore_errors=True)
