"""Versioned, atomically committed stage snapshots and cache restoration."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .util import atomic_write_text, pretty_json, read_json, sha256_obj, write_json


STAGES = (
    "load_validation",
    "version_conversion",
    "softmax",
    "layernorm",
    "summary",
)


class SnapshotStore:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir.resolve()
        self.snapshots_dir = self.runtime_dir / "snapshots"
        self.temp_dir = self.runtime_dir / "tmp"
        self.manifest_path = self.runtime_dir / "manifest.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        if not self.manifest_path.exists():
            return None
        try:
            manifest = read_json(self.manifest_path)
        except Exception:
            return None
        if not isinstance(manifest, dict) or manifest.get("schema") != "opexlab.manifest.v1":
            return None
        snapshot_path = manifest.get("snapshot_path")
        if not isinstance(snapshot_path, str):
            return None
        resolved = (self.runtime_dir / snapshot_path).resolve()
        try:
            resolved.relative_to(self.runtime_dir)
        except ValueError:
            return None
        if not resolved.is_dir():
            return None
        manifest["_resolved_snapshot_path"] = str(resolved)
        return manifest

    def resolve_snapshot_path(self, manifest: Dict[str, Any]) -> Path:
        value = manifest.get("_resolved_snapshot_path")
        if value:
            return Path(value)
        return (self.runtime_dir / str(manifest["snapshot_path"])).resolve()

    def load_stage(self, manifest: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
        path = self.resolve_snapshot_path(manifest) / f"{stage}.json"
        if not path.exists():
            return None
        try:
            value = read_json(path)
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def load_cache(self, manifest: Optional[Dict[str, Any]], config_hash: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        empty = {
            "version_conversion": {},
            "softmax": {},
            "layernorm_stats": {},
            "layernorm_output": {},
        }
        if manifest is None or manifest.get("config_hash") != config_hash:
            return empty
        stages = manifest.get("stage_versions")
        if not isinstance(stages, dict):
            return empty
        conversion = self.load_stage(manifest, "version_conversion") or {}
        softmax = self.load_stage(manifest, "softmax") or {}
        layernorm = self.load_stage(manifest, "layernorm") or {}
        result = {
            "version_conversion": {
                item.get("cache_key"): item
                for item in conversion.get("rows", [])
                if isinstance(item, dict) and item.get("cache_key")
            },
            "softmax": {
                item.get("cache_key"): item
                for item in softmax.get("rows", [])
                if isinstance(item, dict) and item.get("cache_key")
            },
            "layernorm_stats": {
                item.get("cache_key"): item
                for item in layernorm.get("stats_rows", [])
                if isinstance(item, dict) and item.get("cache_key")
            },
            "layernorm_output": {
                item.get("cache_key"): item
                for item in layernorm.get("output_rows", [])
                if isinstance(item, dict) and item.get("cache_key")
            },
        }
        return result

    def commit(
        self,
        *,
        config_hash: str,
        state_hash: str,
        state: Dict[str, Any],
        exchange_version: str,
        exchange_version_fingerprint: str,
        completed_layers: list[str],
        consumed_budget: int,
        stage_payloads: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        missing = [stage for stage in STAGES if stage not in stage_payloads]
        if missing:
            raise ValueError(f"missing stage payloads: {missing}")
        stage_versions = {
            stage: sha256_obj(stage_payloads[stage])
            for stage in STAGES
        }
        previous = self.load_manifest()
        if (
            previous is not None
            and previous.get("config_hash") == config_hash
            and previous.get("state_hash") == state_hash
            and previous.get("stage_versions") == stage_versions
        ):
            return dict(previous, commit_decision="unchanged_by_version_compare")
        sequence = 1
        if previous is not None:
            try:
                sequence = int(previous.get("snapshot_sequence", 0)) + 1
            except Exception:
                sequence = 1
        revision = int(state.get("revision", 0))
        snapshot_id = f"r{revision:04d}-s{sequence:04d}"
        temp_path = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=str(self.temp_dir)))
        try:
            for stage in STAGES:
                (temp_path / f"{stage}.json").write_text(
                    pretty_json(stage_payloads[stage]), encoding="utf-8", newline="\n"
                )
            final_path = self.snapshots_dir / snapshot_id
            if final_path.exists():
                suffix = 1
                while True:
                    candidate = self.snapshots_dir / f"{snapshot_id}-{suffix}"
                    if not candidate.exists():
                        final_path = candidate
                        break
                    suffix += 1
            os.replace(temp_path, final_path)
        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise
        manifest = {
            "schema": "opexlab.manifest.v1",
            "global_version": revision,
            "snapshot_sequence": sequence,
            "snapshot_id": final_path.name,
            "snapshot_path": str(final_path.relative_to(self.runtime_dir)).replace(os.sep, "/"),
            "config_hash": config_hash,
            "state_hash": state_hash,
            "exchange_version": exchange_version,
            "exchange_version_fingerprint": exchange_version_fingerprint,
            "completed_layers": list(completed_layers),
            "consumed_budget": int(consumed_budget),
            "stage_versions": stage_versions,
            "commit_decision": "version_changed",
        }
        atomic_write_text(self.manifest_path, pretty_json(manifest))
        return manifest


def save_state(path: Path, state: Dict[str, Any]) -> None:
    write_json(path, state)


def load_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None

