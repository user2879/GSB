"""Small, JSON-friendly value objects shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def to_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_any(cls, value: Any) -> "BBox":
        if isinstance(value, BBox):
            return value
        if isinstance(value, dict):
            return cls(
                int(value["x0"]),
                int(value["y0"]),
                int(value["x1"]),
                int(value["y1"]),
            )
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return cls(*(int(part) for part in value))
        raise ValueError(f"invalid bounding box: {value!r}")


@dataclass
class Block:
    block_id: str
    bbox: BBox
    ink_density: float
    projection_periodicity: float
    role: str | None = None
    role_evidence: dict[str, Any] = field(default_factory=dict)
    column_id: int | None = None
    is_spanning: bool = False
    reading_order: int | None = None
    reading_order_rule: str | None = None

    def geometry(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "bbox": self.bbox.to_list(),
            "width": self.bbox.width,
            "height": self.bbox.height,
            "ink_density": self.ink_density,
            "projection_periodicity": self.projection_periodicity,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.geometry()
        result.update(
            {
                "role": self.role,
                "role_evidence": self.role_evidence,
                "column_id": self.column_id,
                "is_spanning": self.is_spanning,
                "reading_order": self.reading_order,
                "reading_order_rule": self.reading_order_rule,
            }
        )
        return result


def round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)

