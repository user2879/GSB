"""Small immutable records shared by validation, pipeline, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Failure:
    code: str
    scope: str
    subject: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    seq: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "code": self.code,
            "scope": self.scope,
            "subject": self.subject,
            "message": self.message,
            "details": self.details,
        }


class FailureLog:
    """Append-only failure ledger with deterministic sequence numbers."""

    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(
        self,
        code: str,
        scope: str,
        subject: str,
        message: str,
        **details: Any,
    ) -> Failure:
        failure = Failure(
            code=code,
            scope=scope,
            subject=str(subject),
            message=message,
            details=details,
            seq=len(self._items) + 1,
        )
        self._items.append(failure)
        return failure

    def extend(self, failures: list[Failure]) -> None:
        for failure in failures:
            self.add(
                failure.code,
                failure.scope,
                failure.subject,
                failure.message,
                **failure.details,
            )

    @property
    def items(self) -> list[Failure]:
        return list(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self._items]


@dataclass(frozen=True)
class ColumnTruth:
    id: str
    x0: int
    x1: int
    y0: int
    y1: int
    char_count: int
    start_reading_index: int
    order: int

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1 - 1) / 2.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    def contains_x(self, x: int) -> bool:
        return self.x0 <= x < self.x1

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x0": self.x0,
            "x1": self.x1,
            "y0": self.y0,
            "y1": self.y1,
            "char_count": self.char_count,
            "start_reading_index": self.start_reading_index,
            "order": self.order,
        }


@dataclass(frozen=True)
class TruthKeypoint:
    id: str
    column: str
    x: int
    y: int
    reading_index: int
    char: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "column": self.column,
            "x": self.x,
            "y": self.y,
            "reading_index": self.reading_index,
        }
        if self.char:
            result["char"] = self.char
        return result


@dataclass(frozen=True)
class ImageTruth:
    image: str
    columns: tuple[ColumnTruth, ...]
    keypoints: tuple[TruthKeypoint, ...]
    pixel_size: tuple[int, int]
    extra: dict[str, Any] = field(default_factory=dict)

    def column_map(self) -> dict[str, ColumnTruth]:
        return {column.id: column for column in self.columns}

    def reading_sorted(self) -> list[TruthKeypoint]:
        return sorted(self.keypoints, key=lambda point: (point.reading_index, point.id))


@dataclass(frozen=True)
class KeypointVersion:
    version: str
    image: str
    keypoints: tuple[TruthKeypoint, ...]
    path: str


@dataclass(frozen=True)
class ImageEntry:
    name: str
    image_path: str
    truth_path: str
    truth: ImageTruth | None
    keypoint_versions: dict[str, str]
    image_hash: str = ""


@dataclass(frozen=True)
class Dataset:
    root: str
    manifest_path: str
    manifest: dict[str, Any]
    images: tuple[ImageEntry, ...]
    image_truths: dict[str, ImageTruth]
    keypoint_versions: dict[str, dict[str, KeypointVersion]]
    scanned_pngs: tuple[str, ...]
    resolution: tuple[int, int]
    pixel_mode: str
    content_hash: str = ""

    def image_map(self) -> dict[str, ImageEntry]:
        return {entry.name: entry for entry in self.images}

    def stats(self) -> dict[str, int]:
        points = sum(len(truth.keypoints) for truth in self.image_truths.values())
        max_columns = max((len(truth.columns) for truth in self.image_truths.values()), default=0)
        max_chars = max((column.char_count for truth in self.image_truths.values() for column in truth.columns), default=0)
        return {
            "image_count": len(self.images),
            "total_points": points,
            "max_columns": max_columns,
            "max_chars_per_column": max_chars,
        }
