"""Domain models for the subtitle layout workbench.

All model classes are immutable value objects. Deterministic serialisation is
provided through :func:`canonical_json`, which sorts keys and uses fixed
separators so the same logical content always produces identical bytes (and
therefore identical content hashes).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

FPS = 30
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, fixed separators, no whitespace."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def content_hash(obj: Any) -> str:
    """SHA-256 over the canonical JSON representation of ``obj``."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class SubtitleItem:
    """A single subtitle / accessibility-description cue."""

    item_id: str
    start_frame: int
    end_frame: int
    text: str
    speaker: str
    language: str
    priority: int          # 1 = highest reading priority
    is_ad: bool = False    # accessibility description (audio description)

    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleItem":
        return cls(
            item_id=str(data["item_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            text=str(data["text"]),
            speaker=str(data["speaker"]),
            language=str(data["language"]),
            priority=int(data["priority"]),
            is_ad=bool(data["is_ad"]),
        )


@dataclass(frozen=True)
class Shot:
    """A virtual camera shot described purely by frame numbers."""

    shot_id: str
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shot":
        return cls(str(data["shot_id"]), int(data["start_frame"]),
                   int(data["end_frame"]))


@dataclass(frozen=True)
class Timeline:
    """The synthetic 90s 1920x1080 timeline."""

    fps: int
    width: int
    height: int
    total_frames: int
    shots: tuple[Shot, ...] = field(default_factory=tuple)

    def shot_at(self, frame: int) -> Shot | None:
        for shot in self.shots:
            if shot.start_frame <= frame < shot.end_frame:
                return shot
        return None

    def crosses_cut(self, start_frame: int, end_frame: int) -> bool:
        return any(start_frame < shot.start_frame < end_frame
                   for shot in self.shots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "shots": [shot.to_dict() for shot in self.shots],
        }


@dataclass(frozen=True)
class SampleProject:
    timeline: Timeline
    items: tuple[SubtitleItem, ...]
    seed: int
    manifest: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "timeline": self.timeline.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "ground_truth": list(self.manifest),
        }
