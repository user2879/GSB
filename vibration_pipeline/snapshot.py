from __future__ import annotations

import copy
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_STALE
from .util import atomic_write_json, read_json


CACHE_SCHEMA_VERSION = 1


@dataclass
class SnapshotState:
    root: Path
    manifest: dict[str, Any] | None
    cache: dict[str, Any]
    generation: int
    stale_reasons: list[str] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        return bool(self.stale_reasons)

    @property
    def stale_code(self) -> str | None:
        return E_SNAPSHOT_STALE if self.stale else None

    @property
    def previous_index(self) -> dict[str, Any]:
        value = self.cache.get("provenance_index", {})
        return value if isinstance(value, dict) else {}


def _empty_cache() -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "feature_subcache": {},
        "algorithm_raw_cache": {},
        "quantized_score_cache": {},
        "decision_cache": {},
        "provenance_index": {"segments": {}, "algorithm_params": {}, "global": {}},
    }


class SnapshotStore:
    def __init__(
        self,
        root: str | Path,
        *,
        schema_version: int,
        config_hash: str,
        waveform_fingerprint: str,
        segment_length: int,
        quantization_fingerprint: str,
    ) -> None:
        self.root = Path(root)
        self.schema_version = int(schema_version)
        self.config_hash = config_hash
        self.waveform_fingerprint = waveform_fingerprint
        self.segment_length = int(segment_length)
        self.quantization_fingerprint = quantization_fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self._verify_writable()
        self.manifest_path = self.root / "manifest.json"
        self.cache_path = self.root / "cache.json"
        self.stages_path = self.root / "stages"
        self.stages_path.mkdir(parents=True, exist_ok=True)

    def _verify_writable(self) -> None:
        probe = self.root / f".write-probe-{uuid.uuid4().hex}"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise OSError(f"snapshot directory is not writable: {self.root}") from exc

    @staticmethod
    def _load_json_if_present(path: Path) -> tuple[Any | None, str | None]:
        if not path.exists():
            return None, None
        try:
            return read_json(path), None
        except (OSError, ValueError) as exc:
            return None, f"unreadable_json:{path.name}:{type(exc).__name__}"

    def open(self) -> SnapshotState:
        manifest, manifest_error = self._load_json_if_present(self.manifest_path)
        cache, cache_error = self._load_json_if_present(self.cache_path)
        reasons: list[str] = []
        if manifest_error:
            reasons.append(manifest_error)
        if cache_error:
            reasons.append(cache_error)
        if manifest is not None:
            if manifest.get("snapshot_version") != self.schema_version:
                reasons.append(f"version_mismatch:{manifest.get('snapshot_version')}->{self.schema_version}")
            if manifest.get("config_hash") != self.config_hash:
                reasons.append("config_hash_mismatch")
            if manifest.get("waveform_fingerprint") != self.waveform_fingerprint:
                reasons.append("waveform_fingerprint_mismatch")
            if manifest.get("segment_length") != self.segment_length:
                reasons.append("segment_length_mismatch")
            if manifest.get("quantization_fingerprint") != self.quantization_fingerprint:
                reasons.append("quantization_fingerprint_mismatch")
        if cache is not None and cache.get("schema_version") != CACHE_SCHEMA_VERSION:
            reasons.append("cache_schema_mismatch")
            cache = None
        if cache is None:
            cache = _empty_cache()
        generation = int(manifest.get("generation", 0)) + 1 if isinstance(manifest, dict) else 1
        # Keep a stable de-duplicated order for diagnostics.
        reasons = list(dict.fromkeys(reasons))
        return SnapshotState(root=self.root, manifest=manifest, cache=cache, generation=generation, stale_reasons=reasons)

    def save_cache(self, cache: dict[str, Any]) -> None:
        payload = copy.deepcopy(cache)
        payload["schema_version"] = CACHE_SCHEMA_VERSION
        atomic_write_json(self.cache_path, payload)

    def _atomic_stage_replace(self, stage_name: str, temp_dir: Path) -> None:
        target = self.stages_path / stage_name
        backup = self.stages_path / f".backup-{stage_name}-{uuid.uuid4().hex}"
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temp_dir, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    def commit_stage(
        self,
        stage_name: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        temp_dir = self.stages_path / f".tmp-{stage_name}-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            data_path = temp_dir / "data.json"
            atomic_write_json(data_path, payload)
            from .util import hash_object

            artifact = {
                "stage": stage_name,
                "artifact_version": hash_object(payload, namespace=f"stage-{stage_name}-v1"),
                "metadata": metadata or {},
            }
            atomic_write_json(temp_dir / "artifact.json", artifact)
            self._atomic_stage_replace(stage_name, temp_dir)
            return artifact
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        payload = copy.deepcopy(manifest)
        payload["snapshot_version"] = self.schema_version
        payload["config_hash"] = self.config_hash
        payload["waveform_fingerprint"] = self.waveform_fingerprint
        payload["segment_length"] = self.segment_length
        payload["quantization_fingerprint"] = self.quantization_fingerprint
        atomic_write_json(self.manifest_path, payload)
