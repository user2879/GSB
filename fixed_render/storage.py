from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import canonical_json


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, cache_key: str) -> Path:
        return self.objects / cache_key[:2] / cache_key

    def exists(self, cache_key: str) -> bool:
        directory = self.artifact_dir(cache_key)
        return (directory / "metadata.json").exists() and (directory / "arrays.npz").exists()

    def write(
        self,
        cache_key: str,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> Path:
        final_dir = self.artifact_dir(cache_key)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.exists(cache_key):
            return final_dir
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=final_dir.parent))
        try:
            np.savez_compressed(temporary / "arrays.npz", **arrays)
            atomic_write_json(temporary / "metadata.json", metadata)
            with (temporary / "metadata.json").open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, final_dir)
        except Exception:
            # Temporary artifacts are never visible under their final key.
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final_dir

    def load_arrays(self, cache_key: str) -> dict[str, np.ndarray]:
        with np.load(self.artifact_dir(cache_key) / "arrays.npz", allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def load_metadata(self, cache_key: str) -> dict[str, Any]:
        return json.loads((self.artifact_dir(cache_key) / "metadata.json").read_text("utf-8"))


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / "current.json"

    def commit(self, version: int, manifest: dict[str, Any]) -> Path:
        final_dir = self.root / f"v{version:06d}"
        temporary = Path(tempfile.mkdtemp(prefix=f".v{version:06d}.", dir=self.root))
        try:
            atomic_write_json(temporary / "manifest.json", manifest)
            atomic_write_json(temporary / "cursor.json", manifest.get("cursor", {}))
            pending = manifest.get("pending", [])
            atomic_write_json(temporary / "pending.json", pending)
            budget = manifest.get("budget", {})
            atomic_write_json(temporary / "budget.json", budget)
            os.replace(temporary, final_dir)
        except Exception:
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
            raise
        pointer = {"version": version, "path": final_dir.name}
        atomic_write_json(self.current_path, pointer)
        atomic_write_json(self.root / f"v{version:06d}.json", manifest)
        return final_dir

    def current_pointer(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        return json.loads(self.current_path.read_text("utf-8"))

    def current(self) -> dict[str, Any] | None:
        pointer = self.current_pointer()
        if not pointer:
            return None
        path = self.root / pointer["path"] / "manifest.json"
        if not path.exists():
            fallback = self.root / f"v{int(pointer['version']):06d}.json"
            path = fallback if fallback.exists() else path
        return json.loads(path.read_text("utf-8"))


class LedgerStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "ledgers"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, ledger: dict[str, Any]) -> Path:
        safe_name = name.replace("/", "_").replace(":", "_")
        path = self.root / f"{safe_name}.json"
        return atomic_write_json(path, ledger)

    def load(self, name: str) -> dict[str, Any]:
        safe_name = name.replace("/", "_").replace(":", "_")
        return json.loads((self.root / f"{safe_name}.json").read_text("utf-8"))
