"""Content-addressed, atomically committed stage artifact cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable

from .config import canonical_hash


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON to a sibling temporary file and atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def content_key(stage: str, inputs: Any, algorithm_version: str) -> str:
    return canonical_hash(
        {
            "stage": stage,
            "algorithm": algorithm_version,
            "inputs": inputs,
        }
    )


@dataclass
class CacheLookup:
    hit: bool
    value: Any | None
    key: str


@dataclass
class CacheStats:
    reuse_count: int = 0
    recompute_count: int = 0
    by_stage: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"reuse": {}, "recompute": {}}
    )

    def record(self, stage: str, action: str) -> None:
        target = self.by_stage.setdefault(action, {})
        target[stage] = target.get(stage, 0) + 1
        if action == "reuse":
            self.reuse_count += 1
        elif action == "recompute":
            self.recompute_count += 1


class ArtifactCache:
    """Simple JSON artifact cache protected by a process-local lock.

    Every value is committed with :func:`atomic_write_json`; readers ignore a
    damaged entry and treat it as a miss. Keys already include the algorithm,
    image content and parameter fingerprints.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def _path(self, stage: str, key: str) -> Path:
        if not all(c.isalnum() or c in "._-" for c in stage):
            raise ValueError("非法缓存阶段名")
        return self.root / stage / key[:2] / f"{key}.json"

    def get(self, stage: str, key: str, record: bool = True) -> CacheLookup:
        path = self._path(stage, key)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            hit = True
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            value = None
            hit = False
        if record:
            with self._lock:
                self.stats.record(stage, "reuse" if hit else "recompute")
        return CacheLookup(hit=hit, value=value, key=key)

    def put(self, stage: str, key: str, value: Any) -> None:
        atomic_write_json(self._path(stage, key), value)

    def compute(
        self,
        stage: str,
        inputs: Any,
        algorithm_version: str,
        factory: Callable[[], Any],
    ) -> tuple[Any, str, bool]:
        key = content_key(stage, inputs, algorithm_version)
        found = self.get(stage, key)
        if found.hit:
            return found.value, key, True
        value = factory()
        self.put(stage, key, value)
        return value, key, False

    def export_stats(self) -> dict[str, Any]:
        by_stage: dict[str, Any] = {"reuse": {}, "recompute": {}}
        for action in ("reuse", "recompute"):
            by_stage[action] = dict(sorted(self.stats.by_stage.get(action, {}).items()))
        return {
            "reuse_count": self.stats.reuse_count,
            "recompute_count": self.stats.recompute_count,
            "by_stage": by_stage,
        }
