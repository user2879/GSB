"""Directory-then-pointer atomic snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .constants import SNAPSHOT_SCHEMA_VERSION


class SnapshotStore:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.snapshot_root = self.state_dir / "snapshots"
        self.latest_path = self.state_dir / "latest.json"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def load_latest(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            candidates = sorted(
                path
                for path in self.snapshot_root.glob("snapshot-*")
                if path.is_dir()
            )
            if not candidates:
                return None
            snapshot_dir = candidates[-1]
        else:
            with self.latest_path.open("r", encoding="utf-8") as handle:
                pointer = json.load(handle)
            snapshot_dir = self.snapshot_root / pointer["snapshot_dir"]
            if not snapshot_dir.exists():
                return None
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        artifacts: dict[str, Any] = {}
        for path in sorted(snapshot_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            with path.open("r", encoding="utf-8") as handle:
                artifacts[path.stem] = json.load(handle)
        return {"snapshot_dir": str(snapshot_dir), "manifest": manifest, **artifacts}

    def commit(
        self,
        manifest: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts = artifacts or {}
        revision = int(manifest.get("snapshot_revision", self._next_revision()))
        full_manifest = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_revision": revision,
            **manifest,
        }
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".staging-", dir=self.snapshot_root)
        )
        try:
            self._write_json(temporary_dir / "manifest.json", full_manifest)
            for name, value in artifacts.items():
                self._write_json(temporary_dir / f"{name}.json", value)

            snapshot_name = f"snapshot-{revision:06d}"
            final_dir = self.snapshot_root / snapshot_name
            if final_dir.exists():
                suffix = 1
                while (self.snapshot_root / f"{snapshot_name}-{suffix}").exists():
                    suffix += 1
                final_dir = self.snapshot_root / f"{snapshot_name}-{suffix}"
            os.replace(temporary_dir, final_dir)

            pointer = {
                "snapshot_dir": final_dir.name,
                "global_version": full_manifest.get("global_version"),
                "policy_config_hash": full_manifest.get("policy_config_hash"),
                "width_gear_fingerprint": full_manifest.get("width_gear_fingerprint"),
                "committed_combo_count": len(
                    full_manifest.get("completed_combo_ids", [])
                ),
            }
            self._atomic_write_json(self.latest_path, pointer)
            return {
                "snapshot_dir": str(final_dir),
                "manifest": full_manifest,
                "pointer": pointer,
            }
        finally:
            if temporary_dir.exists():
                for child in temporary_dir.glob("*"):
                    child.unlink()
                temporary_dir.rmdir()

    def _next_revision(self) -> int:
        return len(list(self.snapshot_root.glob("snapshot-*"))) + 1

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _atomic_write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
