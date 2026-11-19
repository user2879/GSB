from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import PipelineError
from .util import (
    atomic_write_bytes,
    atomic_write_text,
    read_json,
    sha256_bytes,
    sha256_json,
)


@dataclass(frozen=True)
class SnapshotState:
    state_root: Path
    snapshot_root: Path
    pointer: dict
    manifest: dict

    @property
    def version(self) -> int:
        return int(self.pointer["global_version"])

    def stage_path(self, stage: str) -> Path:
        return self.snapshot_root / stage

    def input_path(self, name: str) -> Path:
        return self.snapshot_root / "inputs" / name

    def summary(self) -> dict:
        diagnostics_path = self.stage_path("fusion") / "diagnostics.json"
        if diagnostics_path.is_file():
            diagnostics = read_json(diagnostics_path)
            return diagnostics.get("summary", {"completed_queries": 0, "result_count": 0, "progress": 0.0})
        return {"completed_queries": 0, "result_count": 0, "progress": 0.0}


class SnapshotManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots = self.root / "snapshots"
        self.work = self.root / "work"
        self.pointer_path = self.root / "CURRENT.json"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> SnapshotState | None:
        if not self.pointer_path.is_file():
            return None
        try:
            pointer = read_json(self.pointer_path)
            snapshot_root = self.root / pointer["snapshot"]
            manifest = read_json(snapshot_root / "manifest.json")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineError("E_INDEX_INVALID", f"invalid CURRENT pointer or manifest: {exc}") from exc
        if int(manifest.get("global_version", -1)) != int(pointer.get("global_version", -2)):
            raise PipelineError("E_CONFIG_MISMATCH", "CURRENT version does not match manifest version")
        if pointer.get("config_hash") != manifest.get("config_hash"):
            raise PipelineError("E_CONFIG_MISMATCH", "CURRENT config hash does not match manifest")
        manifest_sha = sha256_bytes((snapshot_root / "manifest.json").read_bytes())
        if pointer.get("manifest_sha256") not in (None, manifest_sha):
            raise PipelineError("E_INDEX_INVALID", "manifest digest does not match CURRENT pointer")
        if manifest.get("status") != "complete":
            raise PipelineError("E_CONFIG_MISMATCH", "CURRENT snapshot is not marked complete")
        return SnapshotState(self.root, snapshot_root, pointer, manifest)

    def begin_staging(self, global_version: int) -> Path:
        stamp = f"{os.getpid()}-{time.time_ns()}"
        staging = self.work / f"staging-{global_version:08d}-{stamp}"
        staging.mkdir(parents=True, exist_ok=False)
        return staging

    def discard_staging(self, staging: str | Path) -> None:
        path = Path(staging)
        if path.exists():
            shutil.rmtree(path)

    def next_version(self, current: SnapshotState | None) -> int:
        candidate = 1 if current is None else current.version + 1
        while (self.snapshots / f"v{candidate:08d}").exists():
            candidate += 1
        return candidate

    def commit(self, staging: str | Path, global_version: int) -> SnapshotState:
        staging_path = Path(staging)
        target = self.snapshots / f"v{global_version:08d}"
        manifest_path = staging_path / "manifest.json"
        manifest = read_json(manifest_path)
        if int(manifest.get("global_version", -1)) != global_version:
            raise PipelineError("E_INDEX_INVALID", "staging manifest version mismatch")
        if target.exists():
            raise PipelineError("E_INDEX_INVALID", f"snapshot target already exists: {target.name}")
        os.replace(staging_path, target)
        manifest_sha = sha256_bytes((target / "manifest.json").read_bytes())
        pointer = {
            "schema_version": 1,
            "global_version": global_version,
            "snapshot": str(target.relative_to(self.root)).replace(os.sep, "/"),
            "manifest_sha256": manifest_sha,
            "config_hash": manifest.get("config_hash"),
        }
        atomic_write_bytes(
            self.pointer_path,
            (json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return SnapshotState(self.root, target, pointer, manifest)

    def stage_temp(self, staging: str | Path, stage: str) -> Path:
        return Path(staging) / f".{stage}.tmp"

    def commit_stage(self, staging: str | Path, stage: str) -> Path:
        staging_path = Path(staging)
        temporary = self.stage_temp(staging_path, stage)
        target = staging_path / stage
        if target.exists():
            raise PipelineError("E_INDEX_INVALID", f"stage target already exists: {stage}")
        os.replace(temporary, target)
        return target

    def record_budget_event(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        path = self.root / "budget_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    def check_resume(self, expected_config_hash: str | None = None, expected_version: int | None = None) -> SnapshotState:
        current = self.load_current()
        if current is None:
            raise PipelineError("E_INDEX_INVALID", "no completed snapshot to resume")
        if expected_version is not None and current.version != expected_version:
            raise PipelineError(
                "E_CONFIG_MISMATCH",
                "requested version does not match CURRENT",
                {"expected": expected_version, "actual": current.version},
            )
        if expected_config_hash is not None and current.manifest.get("config_hash") != expected_config_hash:
            raise PipelineError(
                "E_CONFIG_MISMATCH",
                "config hash differs from the committed snapshot; refusing to resume",
                {"expected": expected_config_hash, "actual": current.manifest.get("config_hash")},
            )
        return current

    def cleanup_work(self) -> list[str]:
        removed = []
        for path in sorted(self.work.iterdir()):
            if path.is_dir() and path.name.startswith("staging-"):
                shutil.rmtree(path)
                removed.append(path.name)
        return removed

    def cleanup_orphan_snapshots(self, current: SnapshotState) -> list[str]:
        removed = []
        current_name = current.snapshot_root.name
        for path in sorted(self.snapshots.iterdir()):
            if not path.is_dir() or not path.name.startswith("v"):
                continue
            try:
                version = int(path.name[1:])
            except ValueError:
                continue
            if path.name != current_name and version > current.version:
                shutil.rmtree(path)
                removed.append(path.name)
        return removed

    def read_input_bytes(self, current: SnapshotState, name: str) -> bytes:
        return current.input_path(name).read_bytes()


def stage_digest(path: str | Path) -> str:
    root = Path(path)
    entries = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            entries.append([str(item.relative_to(root)).replace(os.sep, "/"), sha256_bytes(item.read_bytes())])
    return sha256_json(entries)


def write_stage_file(path: str | Path, data: bytes) -> None:
    atomic_write_bytes(path, data)


def write_stage_text(path: str | Path, text: str) -> None:
    atomic_write_text(path, text)
