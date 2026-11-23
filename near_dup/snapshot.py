"""Whole-generation snapshot storage with atomic commit and recovery checks."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_json
from .errors import E_SNAPSHOT_VERSION_MISMATCH, BusinessError


PRIMARY_STAGES = ("normalization", "signatures", "buckets", "ledger")


class SnapshotStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.generations = self.root / "generations"
        self.latest_path = self.root / "LATEST"
        self.temp_root = self.root / "tmp"
        self.generations.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, payload: Any) -> None:
        encoded = canonical_json(payload) + "\n"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def load_latest(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        try:
            pointer = self._read_json(self.latest_path)
            generation = pointer["generation"]
            generation_dir = self.generations / generation
            manifest = self._read_json(generation_dir / "manifest.json")
            stages: dict[str, Any] = {}
            for stage in PRIMARY_STAGES:
                filename = manifest.get("stage_files", {}).get(stage, f"{stage}.json")
                stages[stage] = self._read_json(generation_dir / filename)
            return {
                "generation": generation,
                "manifest": manifest,
                "stages": stages,
            }
        except Exception:
            return None

    def commit(self, manifest: dict[str, Any], stages: dict[str, Any]) -> dict[str, Any]:
        version = int(manifest["global_version"])
        generation = f"g{version:08d}"
        while (self.generations / generation).exists():
            version += 1
            generation = f"g{version:08d}"
        staging = Path(tempfile.mkdtemp(prefix=f".{generation}-", dir=self.temp_root))
        try:
            stage_hashes: dict[str, str] = {}
            stage_files: dict[str, str] = {}
            for stage in PRIMARY_STAGES:
                payload = stages.get(stage, {})
                filename = f"{stage}.json"
                self._write_json(staging / filename, payload)
                stage_hashes[stage] = sha256_json(payload)
                stage_files[stage] = filename
            committed_manifest = dict(manifest)
            committed_manifest["global_version"] = version
            committed_manifest["stage_hashes"] = stage_hashes
            committed_manifest["stage_files"] = stage_files
            committed_manifest["generation"] = generation
            self._write_json(staging / "manifest.json", committed_manifest)
            destination = self.generations / generation
            if destination.exists():
                raise RuntimeError(f"snapshot generation already exists: {destination}")
            os.replace(staging, destination)
            pointer = {"generation": generation, "global_version": version}
            pointer_tmp = self.temp_root / f".LATEST-{version}-{os.getpid()}"
            self._write_json(pointer_tmp, pointer)
            os.replace(pointer_tmp, self.latest_path)
            return committed_manifest
        except Exception:
            if staging.exists():
                import shutil

                shutil.rmtree(staging, ignore_errors=True)
            raise

    def validate_recovery(
        self,
        snapshot: dict[str, Any] | None,
        *,
        expected_version: int | None = None,
        expected_config_hash: str | None = None,
    ) -> dict[str, Any]:
        if snapshot is None:
            raise BusinessError(
                E_SNAPSHOT_VERSION_MISMATCH,
                "no snapshot exists for recovery",
                detail={"expected_version": expected_version, "expected_config_hash": expected_config_hash},
            )
        manifest = snapshot["manifest"]
        mismatches: dict[str, Any] = {}
        if expected_version is not None and int(manifest.get("global_version", -1)) != expected_version:
            mismatches["version"] = {
                "expected": expected_version,
                "actual": manifest.get("global_version"),
            }
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            mismatches["config_hash"] = {
                "expected": expected_config_hash,
                "actual": manifest.get("config_hash"),
            }
        if mismatches:
            raise BusinessError(
                E_SNAPSHOT_VERSION_MISMATCH,
                "snapshot version or configuration hash does not match",
                detail=mismatches,
            )
        return manifest

    def list_generations(self) -> list[str]:
        return sorted(path.name for path in self.generations.iterdir() if path.is_dir())
