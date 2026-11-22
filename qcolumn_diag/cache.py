"""Persistent whole-snapshot storage with atomic pointer replacement."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .fingerprint import canonical_json


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, canonical_json(value) + "\n")


def empty_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_version": None,
        "config_hash": None,
        "sample_hash": None,
        "complete": False,
        "columns": {},
        "completed_columns": [],
        "pending_columns": [],
        "failures": [],
    }


class SnapshotStore:
    """Store snapshots as immutable generation directories.

    A complete JSON package is first written under ``tmp``.  The staging
    directory is then renamed to a new immutable generation and only after that
    is ``CURRENT`` atomically replaced.  This avoids partial snapshots and works
    on Windows without POSIX directory-replacement semantics.
    """

    def __init__(self, root: str | Path, *, retain: int = 5) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.tmp_dir = self.root / "tmp"
        self.pointer_path = self.root / "CURRENT"
        self.retain = max(1, int(retain))
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._generation = 0

    def _list_generations(self) -> list[tuple[int, Path]]:
        generations: list[tuple[int, Path]] = []
        if not self.snapshots_dir.exists():
            return generations
        for path in self.snapshots_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("v"):
                continue
            try:
                generation = int(path.name[1:])
            except ValueError:
                continue
            if (path / "snapshot.json").is_file():
                generations.append((generation, path))
        return sorted(generations)

    def _current_generation_path(self) -> Path | None:
        if self.pointer_path.is_file():
            try:
                name = self.pointer_path.read_text(encoding="utf-8").strip()
            except OSError:
                name = ""
            candidate = self.snapshots_dir / name
            if name and (candidate / "snapshot.json").is_file():
                return candidate
        generations = self._list_generations()
        return generations[-1][1] if generations else None

    def load(self) -> tuple[dict[str, Any], int]:
        current = self._current_generation_path()
        if current is None:
            self._generation = 0
            return empty_snapshot(), 0
        try:
            generation = int(current.name[1:])
            snapshot = json.loads(
                (current / "snapshot.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            self._generation = 0
            return empty_snapshot(), 0
        if not isinstance(snapshot, dict):
            self._generation = 0
            return empty_snapshot(), 0
        self._generation = max(self._generation, generation)
        base = empty_snapshot()
        base.update(snapshot)
        return base, generation

    def commit(
        self,
        snapshot: dict[str, Any],
        *,
        run_version: str,
        config_hash: str,
        sample_hash: str,
        complete: bool,
    ) -> int:
        payload = dict(snapshot)
        payload.update(
            {
                "run_version": run_version,
                "config_hash": config_hash,
                "sample_hash": sample_hash,
                "complete": bool(complete),
            }
        )
        self._generation += 1
        generation = self._generation
        payload["snapshot_generation"] = generation

        staging = Path(
            tempfile.mkdtemp(prefix="stage-", dir=self.tmp_dir)
        )
        try:
            snapshot_file = staging / "snapshot.json"
            with snapshot_file.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(payload) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            target = self.snapshots_dir / f"v{generation:08d}"
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
            write_text_atomic(self.pointer_path, target.name + "\n")
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        self._cleanup(target)
        return generation

    def _cleanup(self, current: Path) -> None:
        generations = self._list_generations()
        keep_names = {path.name for _, path in generations[-self.retain :]}
        keep_names.add(current.name)
        for _, path in generations:
            if path.name not in keep_names:
                shutil.rmtree(path, ignore_errors=True)
