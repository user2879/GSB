"""Immutable layout snapshots with atomic commit and restart recovery.

Each commit writes a new immutable file ``snapshot_vNNNN.json`` via
write-temp-then-``os.replace`` and updates ``current.json`` atomically. The
version pointer and the per-snapshot content hash are the only authority for
new/old decisions; stale writes are refused with ``E_STALE_VERSION`` and old
files are never served through the cache. Recovery after a forced termination
reads solely ``current.json`` (no fabricated exit report).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import StaleVersionError
from .layout import LayoutResult
from .models import SubtitleItem, canonical_bytes, content_hash

POINTER_NAME = "current.json"
SNAPSHOT_PREFIX = "snapshot_v"
SNAPSHOT_SUFFIX = ".json"


@dataclass(frozen=True)
class LayoutSnapshot:
    version: int
    items: tuple[SubtitleItem, ...]
    layout: LayoutResult
    items_hash: str
    layout_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "items_hash": self.items_hash,
            "layout_hash": self.layout_hash,
            "items": [item.to_dict() for item in self.items],
            "layout": self.layout.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutSnapshot":
        items = tuple(SubtitleItem.from_dict(row) for row in data["items"])
        layout = _layout_from_dict(data["layout"])
        return cls(int(data["version"]), items, layout,
                   str(data["items_hash"]), str(data["layout_hash"]))


def _layout_from_dict(data: dict[str, Any]) -> LayoutResult:
    # Local import to avoid a circular dependency at module import time.
    from .layout import (Arbitration, LaidOutCue, LayoutConfig, LayoutResult)
    from .wrapping import WrapResult

    config = LayoutConfig(**{
        key: tuple(value) if key == "font_tiers" else value
        for key, value in data["config"].items()
    })
    cues = tuple(
        LaidOutCue(
            item=SubtitleItem.from_dict(row["item"]),
            lane_index=row["lane_index"], x=row["x"], y=row["y"],
            width=row["width"], side_margin=row["side_margin"],
            wrap=WrapResult(tuple(row["wrap"]["lines"]),
                            row["wrap"]["font_size"],
                            row["wrap"]["max_line_width"],
                            row["wrap"]["hard_broken"]),
        )
        for row in data["cues"]
    )
    arbitrations = tuple(Arbitration(**row) for row in data["arbitrations"])
    return LayoutResult(cues, arbitrations, config, data["content_hash"])


class SnapshotStore:
    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, LayoutSnapshot] = {}

    # ------------------------------------------------------------------
    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def commit(self, items: tuple[SubtitleItem, ...],
               layout: LayoutResult, *,
               expected_previous_version: int | None = None) -> LayoutSnapshot:
        current = self.load_current()
        previous_version = current.version if current else 0
        if (expected_previous_version is not None
                and expected_previous_version != previous_version):
            raise StaleVersionError(
                f"expected base version {expected_previous_version}, "
                f"store is at {previous_version}")
        version = previous_version + 1
        items_hash = content_hash([item.to_dict() for item in items])
        if layout.content_hash != content_hash(_layout_signature(layout)):
            raise StaleVersionError("layout hash mismatch; refusing stale cache")
        snapshot = LayoutSnapshot(version, tuple(items), layout, items_hash,
                                  layout.content_hash)
        self._atomic_write(self._snapshot_path(version), snapshot.to_dict())
        pointer = {"version": version,
                   "layout_hash": snapshot.layout_hash,
                   "items_hash": snapshot.items_hash}
        self._atomic_write(self.directory / POINTER_NAME, pointer)
        self._cache[snapshot.layout_hash] = snapshot
        return snapshot

    def load_current(self) -> LayoutSnapshot | None:
        pointer_path = self.directory / POINTER_NAME
        if not pointer_path.exists():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        path = self._snapshot_path(int(pointer["version"]))
        if not path.exists():
            # Pointer exists but immutable file missing: never fabricate state.
            return None
        cached = self._cache.get(pointer["layout_hash"])
        if cached is not None and cached.version == pointer["version"]:
            return cached
        snapshot = LayoutSnapshot.from_dict(
            json.loads(path.read_text(encoding="utf-8")))
        if (snapshot.layout_hash != pointer["layout_hash"]
                or snapshot.items_hash != pointer["items_hash"]):
            raise StaleVersionError("pointer/hash mismatch on recovery")
        self._cache[snapshot.layout_hash] = snapshot
        return snapshot

    def history(self) -> list[int]:
        versions = []
        for path in self.directory.glob(f"{SNAPSHOT_PREFIX}*{SNAPSHOT_SUFFIX}"):
            stem = path.stem[len(SNAPSHOT_PREFIX):]
            if stem.isdigit():
                versions.append(int(stem))
        return sorted(versions)

    def load_version(self, version: int) -> LayoutSnapshot | None:
        path = self._snapshot_path(version)
        if not path.exists():
            return None
        return LayoutSnapshot.from_dict(
            json.loads(path.read_text(encoding="utf-8")))

    def restore(self, snapshot: LayoutSnapshot) -> LayoutSnapshot:
        """Point current at an existing immutable snapshot (undo support).

        No snapshot file is rewritten or deleted; only the atomically
        replaced pointer changes, and the cache is keyed by content hash so
        an older layout can never be served from a stale cached entry.
        """
        if not self._snapshot_path(snapshot.version).exists():
            raise StaleVersionError(
                f"snapshot v{snapshot.version} missing; cannot restore")
        pointer = {"version": snapshot.version,
                   "layout_hash": snapshot.layout_hash,
                   "items_hash": snapshot.items_hash}
        self._atomic_write(self.directory / POINTER_NAME, pointer)
        self._cache.pop(snapshot.layout_hash, None)
        reloaded = self.load_current()
        if reloaded is None or reloaded.version != snapshot.version:
            raise StaleVersionError("restore verification failed")
        return reloaded

    def _snapshot_path(self, version: int) -> Path:
        return self.directory / f"{SNAPSHOT_PREFIX}{version:04d}{SNAPSHOT_SUFFIX}"


def _layout_signature(layout: LayoutResult) -> dict[str, Any]:
    return {
        "config": layout.config.__dict__,
        "cues": [cue.to_dict() for cue in layout.cues],
        "arbitrations": [arb.to_dict() for arb in layout.arbitrations],
    }
