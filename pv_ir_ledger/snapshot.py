from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .hashing import canonical_json


class SnapshotStore:
    def __init__(self, output_dir: str | Path, workspace_root: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.state_dir = self.output_dir / "state"
        self.snapshot_dir = self.state_dir / "snapshots"
        self.current_path = self.state_dir / "current.json"
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.temp_root = self.workspace_root / ".pv_tmp"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        probe = self.temp_root / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()

    def load(self) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not self.current_path.exists():
            return None, []
        try:
            return json.loads(self.current_path.read_text(encoding="utf-8")), []
        except (OSError, json.JSONDecodeError) as exc:
            return None, [{"code": "E_SNAPSHOT_RECOVERY", "message": "committed snapshot was unreadable", "detail": str(exc)}]

    def commit(self, state: dict[str, Any]) -> Path:
        global_version = int(state.get("manifest", {}).get("global_version", 0))
        temp_dir = Path(tempfile.mkdtemp(prefix=f"snapshot-v{global_version}-", dir=self.temp_root))
        try:
            temp_file = temp_dir / "current.json"
            payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            with temp_file.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            versioned = self.snapshot_dir / f"snapshot-{global_version:08d}.json"
            os.replace(temp_file, versioned)
            current_tmp = temp_dir / "current-pointer.json"
            with current_tmp.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(current_tmp, self.current_path)
            self._fsync_directory(self.state_dir)
            return self.current_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        try:
            descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass


def snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return canonical_json(left) == canonical_json(right)


def ignore_stale_transactions(temp_root: str | Path) -> list[str]:
    root = Path(temp_root)
    if not root.exists():
        return []
    ignored: list[str] = []
    for path in root.glob("snapshot-v*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            ignored.append(str(path))
    return ignored
