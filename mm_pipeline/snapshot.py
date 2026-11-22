"""Atomic, content-addressed stage snapshots and recovery bookkeeping.

Each stage artifact is first written under a unique temporary directory and
then promoted with a single ``os.replace`` (atomic on POSIX within one
filesystem). A manifest only advances after a successful promotion, so a forced
termination can never expose a half-written stage: recovery resumes from the
last fully replaced snapshot version.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import STAGE_ORDER, SNAPSHOT_FORMAT_VERSION
from .errors import PipelineError, E_SNAPSHOT_STALE
from .hashing import hash_json

MANIFEST_NAME = "manifest.json"
STAGE_DIRS = {
    "load": "stage_load",
    "feature": "stage_feature",
    "fusion": "stage_fusion",
    "ledger": "stage_ledger",
}


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp-", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


@dataclass
class SnapshotHandle:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def version(self) -> int:
        return int(self.manifest["global_version"])

    def stage_dir(self, stage: str) -> Path:
        return self.root / STAGE_DIRS[stage]

    def artifact_path(self, stage: str, cache_key: str) -> Path:
        return self.stage_dir(stage) / f"{cache_key}.json"

    def has_artifact(self, stage: str, cache_key: str) -> bool:
        return self.artifact_path(stage, cache_key).exists()

    def load_artifact(self, stage: str, cache_key: str) -> Any:
        with self.artifact_path(stage, cache_key).open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_artifact_atomic(self, stage: str, cache_key: str, payload: Any) -> Path:
        target = self.artifact_path(stage, cache_key)
        _atomic_write_json(target, payload)
        return target

    def promote_stage(
        self,
        stage: str,
        config_hash: str,
        relation_hash: str,
        completed_samples: list[str],
        stage_versions: dict[str, str],
    ) -> int:
        """Advance the global manifest after a stage batch has been promoted."""
        next_version = self.version + 1
        manifest = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "global_version": next_version,
            "config_hash": config_hash,
            "relation_hash": relation_hash,
            "completed_samples": sorted(completed_samples),
            "stage_versions": dict(sorted(stage_versions.items())),
        }
        _atomic_write_json(self.manifest_path, manifest)
        self.manifest = manifest
        return next_version


def _empty_manifest() -> dict[str, Any]:
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "global_version": 0,
        "config_hash": None,
        "relation_hash": None,
        "completed_samples": [],
        "stage_versions": {stage: "" for stage in STAGE_ORDER},
    }


def open_snapshot(work_dir: str | Path) -> SnapshotHandle:
    """Open an existing snapshot, or initialize an empty one atomically."""
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    for stage in STAGE_ORDER:
        (root / STAGE_DIRS[stage]).mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        _atomic_write_json(manifest_path, _empty_manifest())
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return SnapshotHandle(root=root, manifest_path=manifest_path, manifest=manifest)


def resume_or_reset(
    work_dir: str | Path,
    config_hash: str,
    relation_hash: str,
    *,
    reset: bool = False,
) -> tuple[SnapshotHandle, Optional[PipelineError]]:
    """Decide whether an existing snapshot can be reused.

    Returns (handle, stale_error). A format/version mismatch yields
    E_SNAPSHOT_STALE and triggers a clean reset; a matching snapshot is resumed
    with its recovery version intact.
    """
    root = Path(work_dir)
    manifest_path = root / MANIFEST_NAME
    stale: Optional[PipelineError] = None

    if manifest_path.exists() and not reset:
        with manifest_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
        mismatch = (
            existing.get("format_version") != SNAPSHOT_FORMAT_VERSION
            or existing.get("config_hash") != config_hash
        )
        if mismatch:
            stale = PipelineError(
                E_SNAPSHOT_STALE,
                "snapshot format version or config hash differs; resetting store",
                stage="load",
            )
            reset = True

    if reset and root.exists():
        # Whole-directory replacement: remove then recreate via atomic rename of
        # a freshly built tree, so no partial stage directory ever remains.
        parent = root.parent
        replacement = Path(
            tempfile.mkdtemp(prefix=f".{root.name}-rebuild-", dir=str(parent))
        )
        shutil.rmtree(replacement)
        shutil.move(str(root), str(replacement))
        try:
            root.mkdir(parents=True, exist_ok=False)
            for stage in STAGE_ORDER:
                (root / STAGE_DIRS[stage]).mkdir(parents=True, exist_ok=True)
            _atomic_write_json(root / MANIFEST_NAME, _empty_manifest())
        finally:
            shutil.rmtree(replacement, ignore_errors=True)

    handle = open_snapshot(root)
    return handle, stale


def stage_version_hash(payloads: list[Any]) -> str:
    return hash_json(payloads, prefix="stage-version")
