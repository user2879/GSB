from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable

from .errors import BusinessFailure, E_STRUCTURE_INVALID, WorkbenchError
from .util import dec, dec_str, round_half_up, sha256_json


TARGET_ATTRIBUTES = ("warm", "cool", "vivid", "mono", "solar")
REPAIR_METHODS = ("mean", "edge_extend", "noise_guard")

ATTRIBUTE_NAME_TO_CODE = {name: idx + 1 for idx, name in enumerate(TARGET_ATTRIBUTES)}
ATTRIBUTE_CODE_TO_NAME = {code: name for name, code in ATTRIBUTE_NAME_TO_CODE.items()}
REPAIR_NAME_TO_CODE = {name: idx + 1 for idx, name in enumerate(REPAIR_METHODS)}
REPAIR_CODE_TO_NAME = {code: name for name, code in REPAIR_NAME_TO_CODE.items()}


@dataclass(frozen=True)
class Rect:
    x: Decimal
    y: Decimal
    w: Decimal
    h: Decimal

    @classmethod
    def from_values(cls, x: Any, y: Any, w: Any, h: Any) -> "Rect":
        return cls(dec(x), dec(y), dec(w), dec(h))

    def to_dict(self) -> dict[str, str]:
        return {"x": dec_str(self.x), "y": dec_str(self.y), "w": dec_str(self.w), "h": dec_str(self.h)}

    def integer_half_up(self) -> tuple[int, int, int, int]:
        return (round_half_up(self.x), round_half_up(self.y), round_half_up(self.w), round_half_up(self.h))

    def violates(self, image_width: int, image_height: int) -> str | None:
        if self.w <= 0 or self.h <= 0:
            return "rect width and height must be positive"
        if self.x < 0 or self.y < 0:
            return "rect origin is negative"
        if self.x + self.w > Decimal(image_width) or self.y + self.h > Decimal(image_height):
            return "rect extends beyond image bounds"
        return None


@dataclass(frozen=True)
class RegionEdit:
    exchange_version: int
    image_name: str
    region_index: int
    rect: Rect
    style_strength: Decimal
    target_attribute: str
    repair_method: str
    precision_label: str
    rotation_deg: Decimal = Decimal("0")
    feather_radius: Decimal = Decimal("0")
    sidecar_applied: bool = False
    source_format: str = "internal"
    source_fingerprint: str = ""

    @property
    def region_key(self) -> str:
        return f"{self.image_name}#{self.region_index:03d}"

    @property
    def style_permille(self) -> int:
        return round_half_up(self.style_strength * Decimal(1000))

    def with_fields(self, **changes: Any) -> "RegionEdit":
        return RegionEdit(**{**self.__dict__, **changes})

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "exchange_version": self.exchange_version,
            "image": self.image_name,
            "region_index": self.region_index,
            "rect": self.rect.to_dict(),
            "style_strength": dec_str(self.style_strength),
            "style_permille": self.style_permille,
            "target_attribute": self.target_attribute,
            "repair_method": self.repair_method,
            "precision_label": self.precision_label,
            "rotation_deg": dec_str(self.rotation_deg),
            "feather_radius": dec_str(self.feather_radius),
            "sidecar_applied": self.sidecar_applied,
            "source_format": self.source_format,
        }

    def content_fingerprint(self) -> str:
        return sha256_json(self.canonical_dict())

    def validate_semantics(self) -> None:
        if self.region_index < 1:
            raise WorkbenchError(E_STRUCTURE_INVALID, "region_index must be >= 1", stage="parse")
        if not Decimal(0) <= self.style_strength <= Decimal(1):
            raise WorkbenchError(E_STRUCTURE_INVALID, "style_strength must be within [0, 1]", stage="parse")
        if self.target_attribute not in TARGET_ATTRIBUTES:
            raise WorkbenchError(E_STRUCTURE_INVALID, f"unknown target_attribute: {self.target_attribute}", stage="parse")
        if self.repair_method not in REPAIR_METHODS:
            raise WorkbenchError(E_STRUCTURE_INVALID, f"unknown repair_method: {self.repair_method}", stage="parse")
        if not -Decimal(180) <= self.rotation_deg <= Decimal(180):
            raise WorkbenchError(E_STRUCTURE_INVALID, "rotation_deg must be within [-180, 180]", stage="parse")
        if self.feather_radius < 0:
            raise WorkbenchError(E_STRUCTURE_INVALID, "feather_radius must be >= 0", stage="parse")


@dataclass(frozen=True)
class ImageInfo:
    name: str
    path: str
    width: int
    height: int
    source_fingerprint: str


@dataclass
class ParseOutcome:
    regions: list[RegionEdit] = field(default_factory=list)
    failures: list[BusinessFailure] = field(default_factory=list)
    sidecar_hits: int = 0
    sidecar_misses: int = 0
    document_fingerprint: str = ""
    exchange_version: int = 0

    def merge(self, other: "ParseOutcome") -> None:
        self.regions.extend(other.regions)
        self.failures.extend(other.failures)
        self.sidecar_hits += other.sidecar_hits
        self.sidecar_misses += other.sidecar_misses


def stable_region_sort(regions: Iterable[RegionEdit]) -> list[RegionEdit]:
    return sorted(regions, key=lambda r: (r.image_name, r.region_index))

