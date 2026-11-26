from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .util import atomic_write_json, canonical_json, json_hash, load_json


@dataclass(frozen=True)
class LoadedSnapshot:
    state: dict[str, Any]
    snapshot_dir: Path
    pointer: dict[str, Any]


class SnapshotStore:
    """Versioned directory snapshots with an atomic CURRENT pointer.

    A snapshot is fully materialized under a temporary directory and renamed
    into place before the single ``CURRENT.json`` pointer is atomically
    replaced.  No stage ever overwrites files inside a live snapshot.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.current_path = self.root / "CURRENT.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_temporaries()

    def _cleanup_stale_temporaries(self) -> None:
        for entry in self.snapshots_dir.iterdir():
            if entry.is_dir() and entry.name.startswith(".tmp-"):
                shutil.rmtree(entry, ignore_errors=True)

    def load(self) -> LoadedSnapshot | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = load_json(self.current_path)
        except Exception:
            return None
        if not isinstance(pointer, dict):
            return None
        relative = pointer.get("snapshot")
        if not isinstance(relative, str):
            return None
        snapshot_dir = (self.root / relative).resolve()
        try:
            snapshot_dir.relative_to(self.root.resolve())
        except ValueError:
            return None
        state_path = snapshot_dir / "state.json"
        if not state_path.exists():
            return None
        try:
            state = load_json(state_path)
        except Exception:
            return None
        if not isinstance(state, dict):
            return None
        return LoadedSnapshot(state=state, snapshot_dir=snapshot_dir, pointer=pointer)

    def commit(
        self,
        state: Mapping[str, Any],
        *,
        report: Mapping[str, Any] | None = None,
        report_text: str | None = None,
    ) -> LoadedSnapshot:
        current = self.load()
        revision = int(current.state.get("snapshot_revision", 0)) + 1 if current is not None else 1
        next_state = dict(state)
        next_state["snapshot_revision"] = revision
        digest = json_hash(next_state)
        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=self.snapshots_dir))
        try:
            state_path = temp_dir / "state.json"
            with state_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(next_state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if report is not None:
                report_path = temp_dir / "report.json"
                with report_path.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if report_text is not None:
                report_text_path = temp_dir / "report.md"
                with report_text_path.open("w", encoding="utf-8") as handle:
                    handle.write(report_text)
                    handle.flush()
                    os.fsync(handle.fileno())
            final_name = f"snapshot-v{revision:08d}-{digest[:12]}-{uuid.uuid4().hex[:8]}"
            final_dir = self.snapshots_dir / final_name
            os.replace(temp_dir, final_dir)
            _fsync_directory(self.snapshots_dir)
            pointer = {
                "schema": 1,
                "snapshot": str(final_dir.relative_to(self.root)),
                "snapshot_revision": revision,
                "state_digest": digest,
                "global_version": next_state.get("global_version"),
                "config_hash": next_state.get("config_hash"),
                "manifest_hash": next_state.get("manifest_hash"),
            }
            atomic_write_json(self.current_path, pointer, indent=2)
            return LoadedSnapshot(state=next_state, snapshot_dir=final_dir, pointer=pointer)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def snapshot_digest(self) -> str | None:
        loaded = self.load()
        if loaded is None:
            return None
        return loaded.pointer.get("state_digest")

    def history(self) -> list[str]:
        return sorted(entry.name for entry in self.snapshots_dir.iterdir() if entry.is_dir() and not entry.name.startswith("."))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
