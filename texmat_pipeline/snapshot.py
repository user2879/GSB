"""Atomic stage snapshots with versioned manifest and resume detection."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_STALE, Failure

MANIFEST_NAME = "manifest.json"


@dataclass
class SnapshotState:
    global_version: int
    config_hash: str
    frac_bits: int
    unit_levels: int
    completed_batches: int
    artifacts: dict[str, str]
    stale: bool
    failure: Failure | None


class SnapshotStore:
    """Manages a symlink-swapped versioned snapshot tree.

    Each commit writes a brand-new version directory under ``versions/`` and
    atomically swaps the ``current`` symlink (or directory on platforms
    without symlink support).  Existing versions are never overwritten, so a
    commit is an identity change rather than a per-file overwrite.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.current_link = self.root / "current"
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def inspect(self, expected_config_hash: str, expected_frac_bits: int) -> SnapshotState:
        manifest = self._read_manifest()
        if manifest is None:
            return SnapshotState(0, "", expected_frac_bits, 0, 0, {}, False, None)
        stale = (
            manifest.get("config_hash") != expected_config_hash
            or int(manifest.get("frac_bits", -1)) != expected_frac_bits
        )
        failure = None
        if stale:
            failure = Failure(
                E_SNAPSHOT_STALE,
                str(self.current_link),
                "snapshot version/config hash does not match current inputs; rebuilding",
            )
        return SnapshotState(
            int(manifest["global_version"]),
            manifest.get("config_hash", ""),
            int(manifest.get("frac_bits", expected_frac_bits)),
            int(manifest.get("unit_levels", 0)),
            int(manifest.get("completed_batches", 0)),
            dict(manifest.get("artifacts", {})),
            stale,
            failure,
        )

    def _read_manifest(self) -> dict[str, Any] | None:
        path = self._current_path() / MANIFEST_NAME if self._current_path() else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _current_path(self) -> Path | None:
        if self.current_link.is_symlink():
            target = os.readlink(self.current_link)
            candidate = Path(target)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            return candidate
        if self.current_link.is_dir():
            return self.current_link
        return None

    def commit(self, manifest: dict[str, Any], write_files: dict[str, bytes | str]) -> Path:
        previous = self._read_manifest()
        next_version = int(previous["global_version"]) + 1 if previous else 1
        version_dir = self.versions_dir / f"v{next_version:06d}"
        staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=str(self.versions_dir)))
        try:
            for rel, content in write_files.items():
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    target.write_bytes(content)
                else:
                    target.write_text(content, encoding="utf-8")
            manifest = dict(manifest)
            manifest["global_version"] = next_version
            (staging / MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
            if version_dir.exists():
                shutil.rmtree(version_dir)
            os.rename(staging, version_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        self._switch_current(version_dir)
        self._prune_versions(keep=version_dir)
        return version_dir

    def _switch_current(self, version_dir: Path) -> None:
        tmp_link = self.root / ".current.tmp"
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        try:
            # Relative link stays valid if the snapshot root is moved; the
            # target lives under the versions/ subdirectory.
            os.symlink(str(Path("versions") / version_dir.name), tmp_link)
            os.replace(tmp_link, self.current_link)
        except OSError:
            # Fallback for filesystems without symlink support: atomic tree swap.
            if self.current_link.exists():
                backup = self.root / ".current.old"
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(self.current_link, backup)
                shutil.copytree(version_dir, self.current_link)
                shutil.rmtree(backup)
            else:
                shutil.copytree(version_dir, self.current_link)

    def _prune_versions(self, keep: Path) -> None:
        for child in self.versions_dir.iterdir():
            if child.is_dir() and child != keep:
                shutil.rmtree(child)

    def current_dir(self) -> Path | None:
        return self._current_path()
