"""Crash-tolerant staged snapshot commit and recovery."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .utils import read_json, stable_hash, write_json


class SnapshotStore:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.current = self.output_root / "current"
        self.backup = self.output_root / ".previous"
        self.recovery_state = self.recover()

    def recover(self) -> str:
        self.output_root.mkdir(parents=True, exist_ok=True)
        state = "clean"
        if not self.current.exists() and self.backup.exists():
            os.replace(self.backup, self.current)
            state = "restored_previous"
        elif self.current.exists() and self.backup.exists():
            shutil.rmtree(self.backup)
            state = "removed_stale_previous"
        for path in sorted(self.output_root.glob(".staging-*")):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        return state

    def load_manifest(self) -> dict[str, Any] | None:
        path = self.current / "manifest.json"
        if not path.is_file():
            return None
        try:
            value = read_json(path)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def load_cache(self) -> dict[str, Any] | None:
        path = self.current / "cache.json"
        if not path.is_file():
            return None
        try:
            value = read_json(path)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def commit(self, documents: dict[str, Any], manifest: dict[str, Any]) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        staging = self.output_root / f".staging-{os.getpid()}-{stable_hash(manifest, length=12)}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            for filename, document in sorted(documents.items()):
                if not filename.endswith(".json"):
                    filename = f"{filename}.json"
                write_json(staging / filename, document)
            write_json(staging / "manifest.json", manifest)
            if self.current.exists():
                if self.backup.exists():
                    shutil.rmtree(self.backup)
                os.replace(self.current, self.backup)
            os.replace(staging, self.current)
            if self.backup.exists():
                shutil.rmtree(self.backup)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if not self.current.exists() and self.backup.exists():
                os.replace(self.backup, self.current)
            raise
        return self.current


def decide_run_mode(
    previous: dict[str, Any] | None,
    *,
    dataset_hash: str,
    config_hash: str,
    effective_hash: str,
    all_images_complete: bool,
    force_rebuild: bool,
) -> dict[str, Any]:
    if force_rebuild or previous is None:
        return {
            "mode": "rebuild",
            "reason": "forced rebuild" if force_rebuild else "no valid previous manifest",
            "resume_from_version": None,
        }
    if previous.get("schema_version") != 1:
        return {"mode": "rebuild", "reason": "snapshot schema changed", "resume_from_version": previous.get("global_version")}
    if previous.get("config_hash") != config_hash or previous.get("effective_hash") != effective_hash:
        return {
            "mode": "incremental_rebuild",
            "reason": "configuration or effective tier changed; unaffected stage keys may still be reused",
            "resume_from_version": previous.get("global_version"),
        }
    if previous.get("dataset_hash") != dataset_hash:
        return {
            "mode": "incremental",
            "reason": "image/truth/annotation content changed",
            "resume_from_version": previous.get("global_version"),
        }
    if not all_images_complete:
        return {
            "mode": "resume",
            "reason": "same source revision with incomplete image set or pending budget work",
            "resume_from_version": previous.get("global_version"),
        }
    return {
        "mode": "resume",
        "reason": "source revision and configuration already complete",
        "resume_from_version": previous.get("global_version"),
    }
