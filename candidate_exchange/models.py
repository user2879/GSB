from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .util import quantize_int, scaled_to_float


@dataclass(frozen=True)
class RawCandidate:
    series_name: str
    frame_index: int
    class_code: int
    class_name: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    precision_label: str
    source_version: str
    source_ordinal: int
    missing_fields: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalCandidate:
    series_name: str
    frame_index: int
    class_code: int
    class_name: str
    x1_q: int
    y1_q: int
    x2_q: int
    y2_q: int
    score_micro: int
    precision_label: str
    source_version: str
    source_ordinal: int
    missing_fields: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def score_pm(self) -> int:
        return quantize_int(self.score_micro / 1000.0)

    @property
    def score(self) -> float:
        return self.score_micro / 1_000_000.0

    def box(self, decimal_places: int) -> tuple[float, float, float, float]:
        return (
            scaled_to_float(self.x1_q, decimal_places),
            scaled_to_float(self.y1_q, decimal_places),
            scaled_to_float(self.x2_q, decimal_places),
            scaled_to_float(self.y2_q, decimal_places),
        )

    def integer_box(self, decimal_places: int = 3) -> tuple[int, int, int, int]:
        scale = 10 ** decimal_places
        return (
            quantize_int(self.x1_q / scale),
            quantize_int(self.y1_q / scale),
            quantize_int(self.x2_q / scale),
            quantize_int(self.y2_q / scale),
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            -self.score_micro,
            self.x1_q,
            self.y1_q,
            self.x2_q,
            self.y2_q,
            self.class_code,
            self.source_ordinal,
        )

    def identity_key(self) -> str:
        return (
            f"{self.frame_index}:{self.class_code}:"
            f"{self.x1_q},{self.y1_q},{self.x2_q},{self.y2_q}:{self.score_micro}"
        )

    def to_dict(self, decimal_places: int = 3) -> dict[str, Any]:
        x1, y1, x2, y2 = self.box(decimal_places)
        return {
            "series_name": self.series_name,
            "frame_index": self.frame_index,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "box": [x1, y1, x2, y2],
            "integer_box": list(self.integer_box(decimal_places)),
            "score": self.score,
            "score_pm": self.score_pm,
            "precision_label": self.precision_label,
            "source_version": self.source_version,
            "source_ordinal": self.source_ordinal,
            "missing_fields": list(self.missing_fields),
            "extra": self.extra,
        }


@dataclass
class UnifiedFrame:
    series_name: str
    frame_index: int
    image_path: str
    image_sha256: str
    image_size: tuple[int, int]
    candidates: list[CanonicalCandidate]
    candidate_key: str = ""
    content_fingerprint: str = ""
    status: str = "ok"


@dataclass
class FailureRecord:
    series_name: str
    frame_index: int | None
    stage: str
    error_code: str
    exception_type: str
    message: str
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.series_name,
            -1 if self.frame_index is None else self.frame_index,
            self.stage,
            self.error_code,
            self.injection_id or "",
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_name": self.series_name,
            "frame_index": self.frame_index,
            "stage": self.stage,
            "error_code": self.error_code,
            "exception_type": self.exception_type,
            "message": self.message,
            "injection_id": self.injection_id,
            "details": self.details,
        }


@dataclass
class TrackFrame:
    frame_index: int
    box: list[int]
    score_pm: int
    center_x_q: int
    center_y_q: int
    velocity_x_q: int
    velocity_y_q: int
    candidate_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "box": self.box,
            "score_pm": self.score_pm,
            "center": [self.center_x_q, self.center_y_q],
            "velocity": [self.velocity_x_q, self.velocity_y_q],
            "candidate_key": self.candidate_key,
        }


@dataclass
class Track:
    track_id: str
    series_name: str
    class_code: int
    class_name: str
    frames: list[TrackFrame]
    reentered_pool: bool = False
    eviction_count: int = 0
    cache_key: str = ""
    dependency_frame_keys: dict[int, str] = field(default_factory=dict)
    reused: bool = False

    @property
    def first_frame(self) -> int:
        return self.frames[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.frames[-1].frame_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "series_name": self.series_name,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "frame_start": self.first_frame,
            "frame_end": self.last_frame,
            "frame_count": len(self.frames),
            "reentered_pool": self.reentered_pool,
            "eviction_count": self.eviction_count,
            "cache_key": self.cache_key,
            "reused": self.reused,
            "dependency_frame_keys": {
                str(key): value for key, value in sorted(self.dependency_frame_keys.items())
            },
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Track":
        frames = [
            TrackFrame(
                frame_index=item["frame_index"],
                box=list(item["box"]),
                score_pm=item["score_pm"],
                center_x_q=item["center"][0],
                center_y_q=item["center"][1],
                velocity_x_q=item["velocity"][0],
                velocity_y_q=item["velocity"][1],
                candidate_key=item["candidate_key"],
            )
            for item in value["frames"]
        ]
        return cls(
            track_id=value["track_id"],
            series_name=value["series_name"],
            class_code=value["class_code"],
            class_name=value["class_name"],
            frames=frames,
            reentered_pool=value.get("reentered_pool", False),
            eviction_count=value.get("eviction_count", 0),
            cache_key=value.get("cache_key", ""),
            dependency_frame_keys={
                int(k): v for k, v in value.get("dependency_frame_keys", {}).items()
            },
            reused=value.get("reused", False),
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.series_name,
            self.first_frame,
            self.class_code,
            self.track_id,
        )
