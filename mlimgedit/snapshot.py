"""Stage-based snapshot store with atomic replace and manifest recovery.

Layout::

    store/
      committed/
        artifacts/<sha256>.json     # immutable content-addressed artifacts
        manifest.json
      staging/
        artifacts/<sha256>.json
        manifest.json

A batch writes only into ``staging`` and the manifest is atomically renamed over
the committed manifest. Artifacts are content-addressed, so a commit compares
artifact-key sets/versions instead of blindly overwriting files.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hashing import fingerprint_json, fingerprint_strings

MANIFEST_NAME = "manifest.json"
ARTIFACT_DIR = "artifacts"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class StageState:
    version: str
    keys: dict[str, str] = field(default_factory=dict)  # logical key -> artifact id

    def to_dict(self) -> dict:
        return {"version": self.version, "keys": dict(sorted(self.keys.items()))}

    @classmethod
    def from_dict(cls, data: dict) -> "StageState":
        return cls(version=str(data.get("version", "")), keys=dict(data.get("keys", {})))


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.committed = self.root / "committed"
        self.staging = self.root / "staging"
        self.committed_artifacts = self.committed / ARTIFACT_DIR
        self.staging_artifacts = self.staging / ARTIFACT_DIR
        self.committed.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.committed_artifacts.mkdir(parents=True, exist_ok=True)
        self.staging_artifacts.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.committed / MANIFEST_NAME
        self.staging_manifest_path = self.staging / MANIFEST_NAME

    # ------------------------------------------------------------------ manifest
    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return read_json(self.manifest_path)
        return self._empty_manifest()

    @staticmethod
    def _empty_manifest() -> dict:
        return {
            "global_version": 0,
            "config_hash": "",
            "label_set_fingerprint": "",
            "batch_cursor": 0,
            "stage_versions": {},
            "stages": {},
            "pending": [],
            "deduction_cursor": 0,
        }

    # ----------------------------------------------------------------- artifacts
    def _artifact_path(self, artifact_id: str, staging: bool) -> Path:
        base = self.staging_artifacts if staging else self.committed_artifacts
        return base / f"{artifact_id}.json"

    def put_artifact(self, payload: Any) -> str:
        """Write a content-addressed artifact to staging and return its id."""
        artifact_id = fingerprint_json(payload)
        target = self._artifact_path(artifact_id, staging=True)
        if not target.exists():
            atomic_write_json(target, payload)
        # Mirror into committed store only at commit time via promote call.
        return artifact_id

    def get_artifact(self, artifact_id: str) -> Any:
        for base in (self.committed_artifacts, self.staging_artifacts):
            path = base / f"{artifact_id}.json"
            if path.exists():
                return read_json(path)
        raise KeyError(f"artifact {artifact_id} not found")

    def has_artifact(self, artifact_id: str) -> bool:
        return any(
            (base / f"{artifact_id}.json").exists()
            for base in (self.committed_artifacts, self.staging_artifacts)
        )

    # -------------------------------------------------------------------- commit
    def commit(
        self,
        previous: dict,
        *,
        config_hash: str,
        labels: tuple[str, ...],
        batch_cursor: int,
        stages: dict[str, StageState],
        pending: list[dict],
        deduction_cursor: int,
    ) -> tuple[dict, dict[str, bool]]:
        """Build a staging manifest and atomically promote it if versions changed."""
        stage_payload = {name: state.to_dict() for name, state in sorted(stages.items())}
        stage_versions = {name: state.version for name, state in stages.items()}
        manifest = {
            "global_version": int(previous.get("global_version", 0)) + 1,
            "config_hash": config_hash,
            "label_set_fingerprint": fingerprint_strings(labels),
            "batch_cursor": batch_cursor,
            "stage_versions": dict(sorted(stage_versions.items())),
            "stages": stage_payload,
            "pending": pending,
            "deduction_cursor": deduction_cursor,
        }
        changed = self._diff_manifest(previous, manifest)
        # Promote every staging artifact referenced by the new manifest.
        referenced = self._referenced_artifacts(stage_payload)
        for artifact_id in referenced:
            source = self._artifact_path(artifact_id, staging=True)
            target = self._artifact_path(artifact_id, staging=False)
            if not target.exists():
                if not source.exists():
                    raise KeyError(f"staging artifact {artifact_id} missing at commit")
                os.replace(source, target)
        atomic_write_json(self.staging_manifest_path, manifest)
        os.replace(self.staging_manifest_path, self.manifest_path)
        return manifest, changed

    @staticmethod
    def _diff_manifest(previous: dict, current: dict) -> dict[str, bool]:
        changed: dict[str, bool] = {}
        for name, state in current["stages"].items():
            old = previous.get("stages", {}).get(name)
            changed[name] = old != state
        changed["config"] = previous.get("config_hash") != current["config_hash"]
        changed["pending"] = previous.get("pending") != current["pending"]
        return changed

    @staticmethod
    def _referenced_artifacts(stage_payload: dict[str, dict]) -> set[str]:
        referenced: set[str] = set()
        for state in stage_payload.values():
            referenced.update(state.get("keys", {}).values())
        return referenced

    def reset(self) -> None:
        for base in (self.committed, self.staging):
            for path in base.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(base.rglob("*"), reverse=True):
                if path.is_dir() and path != base:
                    path.rmdir()
        self.committed_artifacts.mkdir(parents=True, exist_ok=True)
        self.staging_artifacts.mkdir(parents=True, exist_ok=True)
