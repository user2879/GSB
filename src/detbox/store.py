"""Staged snapshot store with atomic replacement and recovery."""
from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

from .errors import ConfigMismatchError
from .util import atomic_write_json, read_json, replace_dir_atomic


STAGE_DETECT = "detect"
STAGE_MATCH = "match"
STAGE_SERIALIZE = "serialize"
SNAPSHOT_VERSION = 3


class SnapshotStore:
    def __init__(self, workdir: str | Path):
        self.root = Path(workdir)
        self.cache = self.root / "cache"
        self.snapshot = self.root / "snapshot"
        self.tmp = self.root / "tmp"
        for directory in (self.cache, self.snapshot, self.tmp):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.cleanup_orphans()

    def cleanup_orphans(self) -> None:
        """Remove staging/backup dirs left by a hard kill; committed stages stay."""
        import shutil

        for base in (self.snapshot, self.root):
            if not base.exists():
                continue
            for child in base.iterdir():
                name = child.name
                if ".staging-" in name or ".old-" in name or name.startswith(".tmp-"):
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except FileNotFoundError:
                            pass

    # ------------------------------------------------------------ manifest
    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        return read_json(self.manifest_path)

    def write_manifest(self, manifest: dict) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def verify_resume(
        self,
        global_version: int,
        config_hash: str,
        strict: bool = False,
    ) -> None:
        """Reject resume when identity/version/config no longer match.

        Edits applied through this tool update the stored hashes, so normal
        incremental runs resume; an externally modified/unknown state triggers
        E_CONFIG_MISMATCH and refuses to continue.
        """
        manifest = self.load_manifest()
        if manifest is None:
            return
        if manifest.get("snapshot_version") != global_version:
            raise ConfigMismatchError(
                f"snapshot version {manifest.get('snapshot_version')} != {global_version}"
            )
        if manifest.get("config_hash") != config_hash:
            if strict:
                raise ConfigMismatchError(
                    "config hash changed outside the edit journal; refusing to resume"
                )

    # --------------------------------------------------------------- cache
    def cache_entry(self, key: str) -> Path:
        return self.cache / f"{key}.json"

    def cache_get(self, key: str) -> dict | None:
        path = self.cache_entry(key)
        if not path.exists():
            return None
        try:
            return read_json(path)
        except Exception:
            return None

    def cache_put(self, key: str, payload: dict) -> None:
        atomic_write_json(self.cache_entry(key), payload)

    # ------------------------------------------------------------ snapshot
    def commit_stage(self, stage: str, builder) -> Path:
        target = self.snapshot / stage
        replace_dir_atomic(target, builder)
        return target

    def stage_path(self, stage: str) -> Path:
        return self.snapshot / stage

    def stage_exists(self, stage: str) -> bool:
        return (self.snapshot / stage).exists()

    # ----------------------------------------------------------- signals
    @staticmethod
    def install_signal_handlers(state: dict) -> None:
        def handler(signum, frame):
            state["shutdown_requested"] = True

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
