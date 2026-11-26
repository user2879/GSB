"""Data models shared by loaders, stages and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center_x_fixed(self) -> int:
        return self.left + self.right

    @property
    def center_y_fixed(self) -> int:
        return self.top + self.bottom

    def intersection(self, other: "BBox") -> "BBox | None":
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if left >= right or top >= bottom:
            return None
        return BBox(left, top, right, bottom)

    def expanded(self, amount: int) -> "BBox":
        return BBox(
            self.left - amount,
            self.top - amount,
            self.right + amount,
            self.bottom + amount,
        )

    def contains_point(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]

    @classmethod
    def from_list(cls, value: list[int]) -> "BBox":
        return cls(*value)


@dataclass(frozen=True)
class ColumnSpec:
    index: int
    x_min: int
    x_max: int

    def contains(self, x: int) -> bool:
        return self.x_min <= x < self.x_max

    def to_dict(self) -> dict[str, int]:
        return {"index": self.index, "x_min": self.x_min, "x_max": self.x_max}


@dataclass(frozen=True)
class TruthKeypoint:
    keypoint_id: str
    x: int
    y: int
    bbox: BBox
    column: int
    reading_index: int
    text: str
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.keypoint_id,
            "x": self.x,
            "y": self.y,
            "bbox": self.bbox.to_list(),
            "column": self.column,
            "reading_index": self.reading_index,
            "text": self.text,
            "template_id": self.template_id,
        }


@dataclass(frozen=True)
class SampleManifest:
    sample_id: str
    manifest_path: Path
    image_path: Path
    width: int
    height: int
    pixel_mode: str
    image_sha256: str
    template_version: str
    seed: int
    columns: tuple[ColumnSpec, ...]
    keypoints: tuple[TruthKeypoint, ...]
    reading_order: tuple[str, ...]

    def keypoint_by_id(self) -> dict[str, TruthKeypoint]:
        return {item.keypoint_id: item for item in self.keypoints}

    def column_by_index(self) -> dict[int, ColumnSpec]:
        return {item.index: item for item in self.columns}

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image": self.image_path.name,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "pixel_mode": self.pixel_mode,
            "template_version": self.template_version,
            "seed": self.seed,
            "columns": [item.to_dict() for item in self.columns],
            "keypoints": [item.to_dict() for item in self.keypoints],
            "reading_order": list(self.reading_order),
        }


@dataclass(frozen=True)
class AnnotationVersion:
    sample_id: str
    version: str
    path: Path
    content_fingerprint: str
    keypoints: dict[str, tuple[int, int]]

    def coordinate_for(self, keypoint_id: str) -> tuple[int, int] | None:
        return self.keypoints.get(keypoint_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "version": self.version,
            "path": str(self.path),
            "content_fingerprint": self.content_fingerprint,
            "keypoints": {
                keypoint_id: {"x": value[0], "y": value[1]}
                for keypoint_id, value in sorted(self.keypoints.items())
            },
        }


@dataclass
class SampleRecord:
    manifest: SampleManifest
    image_bytes: bytes
    image_fingerprint: str
    annotation: AnnotationVersion | None = None

    @property
    def sample_id(self) -> str:
        return self.manifest.sample_id


@dataclass
class DatasetState:
    dataset_path: Path
    sample_order: list[str] = field(default_factory=list)
    samples: dict[str, SampleRecord] = field(default_factory=dict)
    annotation_versions: dict[str, str] = field(default_factory=dict)
    command_cursor: int = 0

    def ordered_samples(self) -> list[SampleRecord]:
        return [self.samples[sample_id] for sample_id in self.sample_order if sample_id in self.samples]

    def clone(self) -> "DatasetState":
        return DatasetState(
            dataset_path=self.dataset_path,
            sample_order=list(self.sample_order),
            samples=dict(self.samples),
            annotation_versions=dict(self.annotation_versions),
            command_cursor=self.command_cursor,
        )


@dataclass(frozen=True)
class EditCommand:
    seq: int
    op: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "op": self.op, **self.payload}
