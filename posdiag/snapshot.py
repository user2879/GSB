from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import canonical_json, sha256_value


STAGE_ORDER = ("loading", "encoding", "attention", "error_trace")
CACHE_STAGE_ORDER = ("encoding", "attention", "error_trace")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.stages_root = self.root / "stages"
        self.temporary_root = self.root / "tmp"
        self.stages_root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def compatible(manifest: Mapping[str, Any] | None, *, global_version: int, config_hash: str) -> bool:
        if manifest is None:
            return False
        try:
            return (
                int(manifest.get("snapshot_schema_version", -1)) == 1
                and int(manifest.get("global_version", -1)) == int(global_version)
                and str(manifest.get("config_hash", "")) == config_hash
            )
        except (TypeError, ValueError):
            return False

    def commit_stage(
        self,
        stage: str,
        payload: Mapping[str, Any],
        *,
        global_version: int,
        config_hash: str,
        completed_samples: list[str],
        completed_schemes: list[str],
        consumed_budget: Mapping[str, Any],
        status: str,
        run_id: str,
        current_stage: str | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage: {stage}")

        version = sha256_value(payload)
        stage_parent = self.stages_root / stage
        stage_parent.mkdir(parents=True, exist_ok=True)
        final_directory = stage_parent / version
        if not final_directory.exists():
            temporary_directory = Path(
                tempfile.mkdtemp(prefix=f".{stage}-{version[:12]}-", dir=str(self.temporary_root))
            )
            try:
                _atomic_write_json(temporary_directory / "payload.json", payload)
                try:
                    os.replace(temporary_directory, final_directory)
                except OSError:
                    if not final_directory.exists():
                        raise
                    shutil.rmtree(temporary_directory, ignore_errors=True)
            except BaseException:
                shutil.rmtree(temporary_directory, ignore_errors=True)
                raise

        previous = self.load_manifest() or {}
        stages = dict(previous.get("stages", {})) if isinstance(previous.get("stages"), dict) else {}
        entries_count = len(payload.get("entries", {})) if isinstance(payload.get("entries"), dict) else 0
        stages[stage] = {
            "version": version,
            "path": f"stages/{stage}/{version}/payload.json",
            "entry_count": entries_count,
            "config_hash": config_hash,
        }
        manifest = {
            "snapshot_schema_version": 1,
            "global_version": int(global_version),
            "config_hash": config_hash,
            "run_id": run_id,
            "status": status,
            "current_stage": current_stage if current_stage is not None else stage,
            "completed_samples": sorted(set(completed_samples)),
            "completed_schemes": sorted(set(completed_schemes)),
            "consumed_budget": dict(consumed_budget),
            "stages": {name: stages[name] for name in STAGE_ORDER if name in stages},
        }
        _atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _payload_path(self, stage_record: Mapping[str, Any]) -> Path | None:
        relative = stage_record.get("path")
        if not isinstance(relative, str):
            return None
        candidate = (self.root / relative).resolve()
        root_resolved = self.root.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            return None
        return candidate

    def load_cached_entries(self) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
        cache: dict[str, dict[str, dict[str, Any]]] = {stage: {} for stage in CACHE_STAGE_ORDER}
        errors: list[str] = []
        manifest = self.load_manifest()
        if not manifest:
            return cache, errors
        stages = manifest.get("stages", {})
        if not isinstance(stages, dict):
            return cache, ["E_SNAPSHOT_READ"]
        for stage in CACHE_STAGE_ORDER:
            record = stages.get(stage)
            if not isinstance(record, dict):
                continue
            path = self._payload_path(record)
            if path is None:
                errors.append("E_SNAPSHOT_READ")
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                entries = payload.get("entries", {})
                if not isinstance(entries, dict):
                    raise ValueError("invalid entries")
                for key, entry in entries.items():
                    if isinstance(key, str) and isinstance(entry, dict):
                        cache[stage][key] = entry
            except (OSError, json.JSONDecodeError, ValueError):
                errors.append("E_SNAPSHOT_READ")
        return cache, errors

    def snapshot_view(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest is None:
            return {"exists": False}
        return {
            "exists": True,
            "path": str(self.manifest_path),
            "global_version": manifest.get("global_version"),
            "config_hash": manifest.get("config_hash"),
            "status": manifest.get("status"),
            "current_stage": manifest.get("current_stage"),
            "completed_samples": manifest.get("completed_samples", []),
            "completed_schemes": manifest.get("completed_schemes", []),
            "consumed_budget": manifest.get("consumed_budget", {}),
            "stages": manifest.get("stages", {}),
        }

    def cleanup_temporary_directories(self) -> None:
        if not self.temporary_root.exists():
            return
        for child in self.temporary_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
