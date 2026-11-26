"""Atomic, generation-based snapshot storage.

A generation directory is written completely before the tiny ``CURRENT``
pointer is atomically replaced.  Restart therefore sees either the old complete
generation or the new complete generation, never a half-written stage.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_text
from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_VERSION_MISMATCH, SnapshotError


STAGE_FILES = (
    "windows",
    "observations",
    "clusters",
    "segments",
    "scheduler",
    "cache",
    "dataset",
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.pointer_path = self.root / "CURRENT"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def current_pointer(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        try:
            return json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(E_SNAPSHOT_CORRUPT, f"cannot read CURRENT: {exc}") from exc

    def already_committed(self, *, version: int, config_hash: str) -> bool:
        pointer = self.current_pointer()
        return bool(
            pointer
            and int(pointer.get("version", -1)) == int(version)
            and pointer.get("config_hash") == config_hash
        )

    def begin_generation(self, version: int, config_hash: str) -> Path:
        token = uuid.uuid4().hex[:12]
        tmp = self.root / f".tmp-{version}-{config_hash[:12]}-{token}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        return tmp

    def write_stage(self, generation_dir: Path, name: str, payload: Any) -> str:
        if name not in STAGE_FILES:
            raise ValueError(f"unknown snapshot stage: {name}")
        path = generation_dir / f"{name}.json"
        text = canonical_json(payload) + "\n"
        path.write_text(text, encoding="utf-8")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        return sha256_text(text)

    def commit(
        self,
        *,
        version: int,
        config_hash: str,
        stages: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        pointer = self.current_pointer()
        if not force and self.already_committed(version=version, config_hash=config_hash):
            return {"committed": False, "reason": "version_and_config_hash_unchanged", "pointer": pointer}

        tmp = self.begin_generation(version, config_hash)
        stage_hashes: dict[str, str] = {}
        try:
            for name in STAGE_FILES:
                if name not in stages:
                    raise ValueError(f"missing snapshot stage: {name}")
                stage_hashes[name] = self.write_stage(tmp, name, stages[name])
            manifest = {
                "manifest_version": 1,
                "version": int(version),
                "config_hash": config_hash,
                "parent": pointer,
                "stages": stage_hashes,
                "committed": True,
            }
            manifest_text = canonical_json(manifest) + "\n"
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(manifest_text, encoding="utf-8")
            with manifest_path.open("rb") as handle:
                os.fsync(handle.fileno())
            _fsync_directory(tmp)
            generation = self.snapshots_dir / f"g-{version:08d}-{config_hash[:12]}-{uuid.uuid4().hex[:8]}"
            os.replace(tmp, generation)
            pointer_payload = {
                "generation": generation.name,
                "version": int(version),
                "config_hash": config_hash,
                "manifest_hash": sha256_text(manifest_text),
            }
            pointer_tmp = self.root / f".CURRENT-{uuid.uuid4().hex[:12]}"
            pointer_tmp.write_text(canonical_json(pointer_payload) + "\n", encoding="utf-8")
            with pointer_tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(pointer_tmp, self.pointer_path)
            _fsync_directory(self.root)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return {"committed": True, "generation": generation.name, "pointer": pointer_payload}

    def write_uncommitted_generation(
        self,
        *,
        version: int,
        config_hash: str,
        stages: dict[str, Any],
    ) -> Path:
        """Write stages but deliberately leave no CURRENT pointer update.

        This is used by the crash test to prove that an uncommitted directory is
        invisible to restart.
        """

        tmp = self.begin_generation(version=f"crash-{version}", config_hash=config_hash)
        for name in STAGE_FILES:
            if name in stages:
                self.write_stage(tmp, name, stages[name])
        (tmp / "manifest.json").write_text(
            canonical_json({"committed": False, "version": version, "config_hash": config_hash}) + "\n",
            encoding="utf-8",
        )
        return tmp

    def load_current(self) -> tuple[dict[str, Any], dict[str, Any]]:
        pointer = self.current_pointer()
        if pointer is None:
            raise SnapshotError(E_SNAPSHOT_CORRUPT, "no committed snapshot exists")
        generation_dir = self.snapshots_dir / str(pointer["generation"])
        if not generation_dir.is_dir():
            raise SnapshotError(E_SNAPSHOT_CORRUPT, "CURRENT points to a missing generation")
        try:
            manifest_text = (generation_dir / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(E_SNAPSHOT_CORRUPT, f"cannot read manifest: {exc}") from exc
        if manifest.get("committed") is not True:
            raise SnapshotError(E_SNAPSHOT_CORRUPT, "CURRENT points to an uncommitted generation")
        if int(manifest.get("version", -1)) != int(pointer["version"]):
            raise SnapshotError(E_SNAPSHOT_VERSION_MISMATCH, "manifest version differs from CURRENT")
        if manifest.get("config_hash") != pointer.get("config_hash"):
            raise SnapshotError(E_SNAPSHOT_VERSION_MISMATCH, "manifest config hash differs from CURRENT")
        stages: dict[str, Any] = {}
        for name in STAGE_FILES:
            path = generation_dir / f"{name}.json"
            try:
                text = path.read_text(encoding="utf-8")
                payload = json.loads(text)
            except (OSError, json.JSONDecodeError) as exc:
                raise SnapshotError(E_SNAPSHOT_CORRUPT, f"cannot read stage {name}: {exc}") from exc
            if sha256_text(text) != manifest["stages"].get(name):
                raise SnapshotError(E_SNAPSHOT_CORRUPT, f"stage digest mismatch: {name}")
            stages[name] = payload
        return pointer, {"manifest": manifest, "manifest_hash": sha256_text(manifest_text), "stages": stages}

    def list_generations(self) -> list[str]:
        return sorted(path.name for path in self.snapshots_dir.iterdir() if path.is_dir())
