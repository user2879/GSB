from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SnapshotState:
    global_version: int = 0
    config_hash: str = ""
    format_versions: list[str] = field(default_factory=list)
    completed_fragments: list[str] = field(default_factory=list)
    stage_versions: dict[str, int] = field(default_factory=dict)
    stage_progress: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def as_manifest(self, **extra: Any) -> dict[str, Any]:
        manifest = {
            "schema_version": 1,
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "format_versions": sorted(self.format_versions),
            "completed_fragments": list(self.completed_fragments),
            "stage_versions": dict(sorted(self.stage_versions.items())),
            "stage_progress": self.stage_progress,
            "artifacts": sorted(self.artifacts),
            "artifact_count": len(self.artifacts),
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest.update(extra)
        return manifest


class SnapshotManager:
    """Atomic, versioned global stage snapshots.

    Cache payloads are content addressed; a payload is trusted only when its key
    appears in a fully installed snapshot manifest. A process killed before the
    manifest rename cannot make a half-written stage part of recovery state.
    """

    def __init__(self, root: str | Path, *, artifact_exists) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_exists = artifact_exists
        self.state = SnapshotState()
        self.recovery: dict[str, Any] = {
            "used": False,
            "snapshot_version": 0,
            "stage_versions": {},
            "format_versions": [],
            "completed_fragments": [],
            "artifact_count": 0,
            "point": None,
        }
        self._load()

    @property
    def current_pointer(self) -> Path:
        return self.root / "CURRENT.json"

    def _read_pointer(self) -> tuple[int, Path] | None:
        try:
            data = json.loads(self.current_pointer.read_text(encoding="utf-8"))
            version = int(data["global_version"])
            path = self.root / data["manifest"]
            return version, path
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load_manifest(self, path: Path) -> SnapshotState | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            version = int(data["global_version"])
            artifacts = [str(item) for item in data.get("artifacts", [])]
            if not all(self.artifact_exists(key) for key in artifacts):
                return None
            completed = [str(item) for item in data.get("completed_fragments", [])]
            stage_versions = {
                str(stage): int(stage_version)
                for stage, stage_version in data.get("stage_versions", {}).items()
            }
            return SnapshotState(
                global_version=version,
                config_hash=str(data.get("config_hash", "")),
                format_versions=[str(item) for item in data.get("format_versions", [])],
                completed_fragments=completed,
                stage_versions=stage_versions,
                stage_progress=data.get("stage_progress", {}),
                artifacts=artifacts,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load(self) -> None:
        pointer = self._read_pointer()
        candidates: list[Path] = []
        if pointer:
            candidates.append(pointer[1])
        candidates.extend(
            sorted(self.root.glob("v*/manifest.json"), reverse=True)
        )
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            state = self._load_manifest(path)
            if state is None:
                continue
            self.state = state
            self.recovery = {
                "used": True,
                "snapshot_version": state.global_version,
                "stage_versions": dict(sorted(state.stage_versions.items())),
                "format_versions": sorted(state.format_versions),
                "completed_fragments": list(state.completed_fragments),
                "artifact_count": len(state.artifacts),
                "point": str(path),
                "pointer_matches": bool(
                    pointer and pointer[0] == state.global_version
                ),
            }
            return

    def commit(
        self,
        *,
        stage: str,
        config_hash: str,
        format_versions: Iterable[str],
        completed_fragments: Iterable[str],
        new_artifact_keys: Iterable[str] = (),
        stage_progress: dict[str, Any] | None = None,
        snapshot_interval: int | None = None,
    ) -> dict[str, Any]:
        next_version = self.state.global_version + 1
        artifacts = set(self.state.artifacts)
        artifacts.update(new_artifact_keys)
        state = SnapshotState(
            global_version=next_version,
            config_hash=config_hash,
            format_versions=sorted(set(format_versions)),
            completed_fragments=sorted(set(completed_fragments)),
            stage_versions={**self.state.stage_versions, stage: next_version},
            stage_progress={**self.state.stage_progress, stage: stage_progress or {}},
            artifacts=sorted(artifacts),
        )
        manifest = state.as_manifest(snapshot_interval=snapshot_interval)
        temp = self.root / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            manifest_path = temp / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            final_dir = self.root / f"v{next_version:08d}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp, final_dir)
            pointer = {
                "global_version": next_version,
                "manifest": f"v{next_version:08d}/manifest.json",
            }
            pointer_temp = self.root / f".CURRENT.tmp.{os.getpid()}"
            pointer_temp.write_text(
                json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(pointer_temp, self.current_pointer)
        finally:
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
        self.state = state
        return manifest

    def status(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "global_version": self.state.global_version,
            "stage_versions": dict(sorted(self.state.stage_versions.items())),
            "artifact_count": len(self.state.artifacts),
        }

