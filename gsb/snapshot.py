from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import SNAPSHOT_FORMAT_VERSION
from .canonical import digest
from .errors import GSBError


@dataclass
class SessionSnapshot:
    capacity_version: int
    spec_digest: str
    watermarks: dict[str, Any] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    wait_records: list[dict[str, Any]] = field(default_factory=list)
    generated_monotonic_ms: int = 0
    snapshot_format: int = SNAPSHOT_FORMAT_VERSION
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_format": self.snapshot_format,
            "capacity_version": int(self.capacity_version),
            "spec_digest": self.spec_digest,
            "watermarks": self.watermarks,
            "traces": self.traces,
            "decisions": self.decisions,
            "wait_records": self.wait_records,
            "generated_monotonic_ms": self.generated_monotonic_ms,
            "stale": self.stale,
            "snapshot_digest": digest(self._digest_payload()),
        }

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "snapshot_format": self.snapshot_format,
            "capacity_version": self.capacity_version,
            "spec_digest": self.spec_digest,
            "watermarks": self.watermarks,
            "traces": self.traces,
            "decisions": self.decisions,
            "wait_records": self.wait_records,
            "generated_monotonic_ms": self.generated_monotonic_ms,
            "stale": self.stale,
        }

    @property
    def snapshot_digest(self) -> str:
        return digest(self._digest_payload())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionSnapshot":
        version = int(payload.get("snapshot_format", 0))
        if version != SNAPSHOT_FORMAT_VERSION:
            raise GSBError(
                f"unsupported snapshot format {version}",
                field="snapshot_format",
                position=version,
            )
        expected = payload.get("snapshot_digest")
        snapshot = cls(
            capacity_version=int(payload["capacity_version"]),
            spec_digest=str(payload["spec_digest"]),
            watermarks=payload.get("watermarks", {}),
            traces=payload.get("traces", []),
            decisions=payload.get("decisions", []),
            wait_records=payload.get("wait_records", []),
            generated_monotonic_ms=int(payload.get("generated_monotonic_ms", 0)),
            snapshot_format=version,
            stale=bool(payload.get("stale", False)),
        )
        if expected and expected != snapshot.snapshot_digest:
            raise GSBError("snapshot digest mismatch", field="snapshot_digest")
        return snapshot

    def save_atomic(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SessionSnapshot":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class ViewStore:
    """Whole-snapshot current view; old generations are marked stale."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: SessionSnapshot | None = None
        self.generations: list[SessionSnapshot] = []

    def replace(self, snapshot: SessionSnapshot) -> SessionSnapshot:
        with self._lock:
            for old in self.generations:
                old.stale = True
            snapshot.stale = False
            self.generations.append(snapshot)
            self._current = snapshot
            return snapshot

    @property
    def current(self) -> SessionSnapshot:
        with self._lock:
            if self._current is None:
                raise GSBError("no session snapshot has been installed", field="snapshot")
            return self._current

    def visible_view(self) -> dict[str, Any]:
        with self._lock:
            current = self.current
            if current.stale:
                return {"snapshot_format": SNAPSHOT_FORMAT_VERSION, "stale": True, "generations": []}
            view = current.to_dict()
            view["generations_visible"] = 1
            return view

    def stale_generations(self) -> list[dict[str, Any]]:
        with self._lock:
            return [snapshot.to_dict() for snapshot in self.generations if snapshot.stale]
