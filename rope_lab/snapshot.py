"""Staging-directory plus atomic symlink snapshot commits."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .codes import E_SNAPSHOT_VERSION
from .hashing import fingerprint


class SnapshotConflict(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.current_link = self.root / "current"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def head_meta(self) -> dict[str, Any] | None:
        target = self._resolve_current()
        if target is None:
            return None
        meta_path = target / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _resolve_current(self) -> Path | None:
        if self.current_link.is_symlink():
            return (self.root / os.readlink(self.current_link)).resolve()
        if self.current_link.exists():
            return self.current_link.resolve()
        return None

    def load_current(self) -> dict[str, Any] | None:
        target = self._resolve_current()
        if target is None or not target.exists():
            return None
        payload: dict[str, Any] = {}
        for path in sorted(target.glob("*.json")):
            payload[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        payload["_snapshot_path"] = str(target)
        return payload

    def commit(
        self,
        sections: dict[str, Any],
        *,
        global_version: int,
        config_hash: str,
        expected_head_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        current_head = self.head_meta()
        current_fp = fingerprint(current_head) if current_head else None
        if expected_head_fingerprint is not None and expected_head_fingerprint != current_fp:
            raise SnapshotConflict(
                "snapshot head changed before commit",
                {"code": E_SNAPSHOT_VERSION, "expected": expected_head_fingerprint, "actual": current_fp},
            )

        staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))
        try:
            commit_name = f"v{int(global_version):06d}-{config_hash[:12]}-{staging.name.split('-')[-1]}"
            meta = {
                "format": "rope-lab-stage-snapshot",
                "global_version": int(global_version),
                "config_hash": config_hash,
                "commit_name": commit_name,
                "head_fingerprint_before": current_fp,
            }
            all_sections = dict(sections)
            all_sections["meta"] = meta
            for name, value in all_sections.items():
                (staging / f"{name}.json").write_text(
                    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            self._fsync_dir(staging)

            final_path = self.snapshot_dir / commit_name
            os.replace(staging, final_path)

            tmp_link = self.root / f".current-tmp-{os.getpid()}"
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
            os.symlink(os.path.relpath(final_path, self.root), tmp_link)
            os.replace(tmp_link, self.current_link)
            self._fsync_dir(self.root)
            return meta
        finally:
            if staging.exists():
                # Staging is normally renamed; this only cleans a failed commit.
                for child in sorted(staging.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                staging.rmdir()

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        flags = os.O_RDONLY
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def recovery_decision(self, global_version: int, config_hash: str) -> dict[str, Any]:
        head = self.head_meta()
        if head is None:
            return {"use_snapshot": False, "action": "rebuild", "reason": "empty_snapshot"}
        if head.get("config_hash") != config_hash:
            return {
                "use_snapshot": False,
                "action": "migrate_cache_then_rebuild",
                "reason": "config_hash_changed",
                "code": E_SNAPSHOT_VERSION,
                "snapshot": head,
            }
        if int(head.get("global_version", -1)) != int(global_version):
            return {
                "use_snapshot": False,
                "action": "rebuild",
                "reason": "global_version_changed",
                "code": E_SNAPSHOT_VERSION,
                "snapshot": head,
            }
        return {"use_snapshot": True, "action": "restore", "reason": "version_and_hash_match", "snapshot": head}

    def prune_commits(self, keep: int = 1) -> None:
        commits = sorted(path for path in self.snapshot_dir.iterdir() if path.is_dir() and path.name.startswith("v"))
        current_name = None
        if self.current_link.is_symlink():
            current_name = Path(os.readlink(self.current_link)).name
        for old_commit in commits:
            if old_commit.name == current_name:
                continue
            shutil.rmtree(old_commit, ignore_errors=True)
