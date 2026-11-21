"""Unified intermediate representation shared by all exchange versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DIMENSIONS
from .util import MICRO_SCALE, micro_to_float, micro_to_permille


@dataclass(frozen=True)
class FrameSummary:
    frame_index: int
    rms_db: float
    zcr: float
    peak: float
    max_zero_run: int
    zero_run_ratio: float
    clipped_samples: int
    frame_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "rms_db": self.rms_db,
            "zcr": self.zcr,
            "peak": self.peak,
            "max_zero_run": self.max_zero_run,
            "zero_run_ratio": self.zero_run_ratio,
            "clipped_samples": self.clipped_samples,
            "frame_fingerprint": self.frame_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrameSummary":
        return cls(
            frame_index=int(value["frame_index"]),
            rms_db=float(value["rms_db"]),
            zcr=float(value["zcr"]),
            peak=float(value["peak"]),
            max_zero_run=int(value["max_zero_run"]),
            zero_run_ratio=float(value["zero_run_ratio"]),
            clipped_samples=int(value["clipped_samples"]),
            frame_fingerprint=str(value.get("frame_fingerprint", "")),
        )


@dataclass
class UnifiedResult:
    file_id: str
    source_version: int
    total_micro: int
    dimensions_micro: dict[str, int]
    level_code: str | None = None
    level_name: str | None = None
    precision_tag: str | None = None
    frame_summaries: list[FrameSummary] = field(default_factory=list)
    bypass: dict[str, Any] = field(default_factory=dict)
    location: str = ""
    source_fixed_bits: int | None = None

    def __post_init__(self) -> None:
        if set(self.dimensions_micro) != set(DIMENSIONS):
            raise ValueError(f"dimensions must be exactly {DIMENSIONS}")
        if not 0 <= self.total_micro <= MICRO_SCALE:
            raise ValueError("total_micro outside 0..1000000")
        for name, value in self.dimensions_micro.items():
            if not 0 <= value <= MICRO_SCALE:
                raise ValueError(f"dimension {name} outside 0..1000000")

    @property
    def total_permille(self) -> int:
        return micro_to_permille(self.total_micro)

    def dimension_permille(self, name: str) -> int:
        return micro_to_permille(self.dimensions_micro[name])

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "source_version": self.source_version,
            "total_micro": self.total_micro,
            "total_score": micro_to_float(self.total_micro),
            "total_permille": self.total_permille,
            "dimensions_micro": dict(self.dimensions_micro),
            "dimensions": {
                name: micro_to_float(self.dimensions_micro[name]) for name in DIMENSIONS
            },
            "dimensions_permille": {
                name: self.dimension_permille(name) for name in DIMENSIONS
            },
            "level_code": self.level_code,
            "level_name": self.level_name,
            "precision_tag": self.precision_tag,
            "frame_summary_count": len(self.frame_summaries),
            "frame_summaries": [frame.to_dict() for frame in self.frame_summaries],
            "bypass": self.bypass,
            "location": self.location,
            "source_fixed_bits": self.source_fixed_bits,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UnifiedResult":
        frames = [
            FrameSummary.from_dict(item) for item in value.get("frame_summaries", [])
        ]
        return cls(
            file_id=str(value["file_id"]),
            source_version=int(value["source_version"]),
            total_micro=int(value["total_micro"]),
            dimensions_micro={
                name: int(value["dimensions_micro"][name]) for name in DIMENSIONS
            },
            level_code=value.get("level_code"),
            level_name=value.get("level_name"),
            precision_tag=value.get("precision_tag"),
            frame_summaries=frames,
            bypass=dict(value.get("bypass", {})),
            location=str(value.get("location", "")),
            source_fixed_bits=value.get("source_fixed_bits"),
        )

    def with_frames(self, frames: list[FrameSummary]) -> "UnifiedResult":
        return UnifiedResult(
            file_id=self.file_id,
            source_version=self.source_version,
            total_micro=self.total_micro,
            dimensions_micro=dict(self.dimensions_micro),
            level_code=self.level_code,
            level_name=self.level_name,
            precision_tag=self.precision_tag,
            frame_summaries=list(frames),
            bypass=dict(self.bypass),
            location=self.location,
            source_fixed_bits=self.source_fixed_bits,
        )


@dataclass(frozen=True)
class Degradation:
    code: str
    file_id: str
    source_version: int
    target_version: int
    message: str
    residual_micro: int = 0
    fields: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.file_id,
            self.source_version,
            self.target_version,
            self.code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "file_id": self.file_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "message": self.message,
            "residual_micro": self.residual_micro,
            "fields": self.fields,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Degradation":
        return cls(
            code=str(value["code"]),
            file_id=str(value["file_id"]),
            source_version=int(value["source_version"]),
            target_version=int(value["target_version"]),
            message=str(value["message"]),
            residual_micro=int(value.get("residual_micro", 0)),
            fields=dict(value.get("fields", {})),
        )


def sorted_degradations(items: list[Degradation]) -> list[Degradation]:
    return sorted(items, key=Degradation.sort_key)


def result_sort_key(result: UnifiedResult) -> tuple[str, int]:
    return result.file_id, result.source_version
