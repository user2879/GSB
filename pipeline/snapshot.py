from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import SnapshotMismatchError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class SnapshotStore:
    def __init__(self, state_dir: str | Path, identity: dict) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.state_dir / "manifest.json"
        self.identity = identity
        if self.manifest_path.exists():
            self.manifest = self._read_manifest()
        else:
            self.manifest = {
                "schema_version": "1.0",
                "global_version": 0,
                "pipeline_version": identity["pipeline_version"],
                "config_hash": identity["config_hash"],
                "translation_hash": identity["translation_hash"],
                "injection_hash": identity["injection_hash"],
                "input_hashes": identity["input_hashes"],
                "completed_batches": {
                    "normalize": [],
                    "index_build": [],
                    "retrieval_scoring": [],
                    "evaluation": [],
                },
                "stage_artifacts": {},
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            self._persist_manifest()

    @classmethod
    def reset(cls, state_dir: str | Path, identity: dict) -> "SnapshotStore":
        state_dir = Path(state_dir)
        if state_dir.exists():
            shutil.rmtree(state_dir)
        return cls(state_dir, identity)

    def _read_manifest(self) -> dict:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotMismatchError(f"snapshot manifest is unreadable: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SnapshotMismatchError("snapshot manifest must be a JSON object")
        return manifest

    def _persist_manifest(self) -> None:
        self.manifest["updated_at"] = utc_now()
        atomic_write_json(self.manifest_path, self.manifest)

    def check_compatible(self) -> None:
        comparisons = {
            "pipeline_version": self.identity["pipeline_version"],
            "config_hash": self.identity["config_hash"],
            "translation_hash": self.identity["translation_hash"],
            "injection_hash": self.identity["injection_hash"],
            "input_hashes": self.identity["input_hashes"],
        }
        mismatches = []
        for key, expected in comparisons.items():
            actual = self.manifest.get(key)
            if actual != expected:
                mismatches.append({"field": key, "expected": expected, "actual": actual})
        if mismatches:
            error = SnapshotMismatchError(
                "snapshot identity does not match the current configuration/translation/input identity"
            )
            error.details = mismatches
            raise error

    def completed_batches(self, stage: str) -> set[int]:
        return set(int(item) for item in self.manifest.get("completed_batches", {}).get(stage, []))

    def stage_is_complete(self, stage: str, expected_batches: int) -> bool:
        return self.completed_batches(stage) >= set(range(1, expected_batches + 1))

    def load_stage_state(self, stage: str) -> Optional[dict]:
        artifact = self.manifest.get("stage_artifacts", {}).get(stage)
        if not artifact:
            return None
        relative = Path(artifact["path"]) / "artifact.json"
        path = self.state_dir / relative
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotMismatchError(f"stage artifact {path} is unreadable: {exc}") from exc
        return payload.get("data")

    def commit_batch(self, stage: str, batch_no: int, data: Any) -> dict:
        if batch_no in self.completed_batches(stage):
            return self.manifest["stage_artifacts"][stage]
        global_version = int(self.manifest.get("global_version", 0)) + 1
        stage_root = self.state_dir / "stages" / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        target_name = f"v{global_version:06d}"
        target = stage_root / target_name
        if target.exists():
            target = stage_root / f"v{global_version:06d}-{uuid.uuid4().hex[:8]}"
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-", dir=stage_root))
        try:
            with (temporary / "artifact.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": stage,
                        "batch_no": batch_no,
                        "global_version": global_version,
                        "committed_at": utc_now(),
                        "data": data,
                    },
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        relative_path = target.relative_to(self.state_dir).as_posix()
        completed = self.manifest.setdefault("completed_batches", {}).setdefault(stage, [])
        completed.append(batch_no)
        completed.sort()
        self.manifest["global_version"] = global_version
        self.manifest.setdefault("stage_artifacts", {})[stage] = {
            "version": target_name,
            "path": relative_path,
            "last_batch": batch_no,
            "global_version": global_version,
        }
        self._persist_manifest()
        return self.manifest["stage_artifacts"][stage]

    def mark_empty_stage(self, stage: str) -> None:
        if self.completed_batches(stage):
            return
        version = f"v{int(self.manifest.get('global_version', 0)):06d}"
        self.manifest.setdefault("completed_batches", {})[stage] = []
        self.manifest.setdefault("stage_artifacts", {})[stage] = {
            "version": version,
            "path": None,
            "last_batch": 0,
            "global_version": int(self.manifest.get("global_version", 0)),
        }
        self._persist_manifest()

    def snapshot_summary(self) -> dict:
        return {
            "version": int(self.manifest.get("global_version", 0)),
            "config_hash": self.manifest.get("config_hash"),
            "translation_hash": self.manifest.get("translation_hash"),
            "injection_hash": self.manifest.get("injection_hash"),
            "input_hashes": self.manifest.get("input_hashes"),
            "completed_batches": {
                stage: list(sorted(int(value) for value in batches))
                for stage, batches in sorted(self.manifest.get("completed_batches", {}).items())
            },
            "stage_artifacts": self.manifest.get("stage_artifacts", {}),
        }

