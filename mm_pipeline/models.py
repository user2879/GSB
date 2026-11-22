"""Typed data models shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

from .config import STAGE_FEATURE, STAGE_FUSION

RELATION_LINK = "link"
RELATION_CONFLICT = "conflict"
RELATION_SUPPLEMENT = "supplement"
RELATION_TYPES = (RELATION_LINK, RELATION_CONFLICT, RELATION_SUPPLEMENT)

CATEGORY_SUPPORT = "support"
CATEGORY_REJECT = "reject"
CATEGORY_NEUTRAL = "neutral"


@dataclass(frozen=True)
class Assertion:
    assertion_id: str
    entity: str
    attribute: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Relation:
    relation_id: str
    relation_type: str
    assertion_id: str
    region: int
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Sample:
    sample_id: str
    image_path: str
    assertions_path: str
    relations_path: str
    image_bytes: bytes
    width: int
    height: int
    assertions: list[Assertion]
    relations: list[Relation]
    image_hash: str
    assertion_hash: str
    relation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "assertions": [a.to_dict() for a in self.assertions],
            "relations": [r.to_dict() for r in self.relations],
            "image_hash": self.image_hash,
            "assertion_hash": self.assertion_hash,
            "relation_hash": self.relation_hash,
        }


@dataclass
class RegionFeature:
    sample_id: str
    region: int
    visual_vector: list[int]
    text_vector: list[int]
    backfilled_assertions: list[str]
    referenced_assertions: list[str]
    referenced_regions: list[int]
    cache_key: str
    reused: bool = False

    @property
    def sort_key(self):
        return (self.sample_id, self.region, STAGE_FEATURE)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Contribution:
    modality: str            # "visual" or "text"
    relation_type: str       # link / conflict / supplement / intrinsic
    assertion_id: Optional[str]
    weight_fixed: int
    score_fixed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegionResult:
    sample_id: str
    region: int
    category: str
    confidence_fixed: int
    visual_score_fixed: int
    text_score_fixed: int
    visual_weight_fixed: int
    text_weight_fixed: int
    visual_quantized: int
    text_quantized: int
    contributions: list[Contribution]
    boundary: bool
    boundary_reasons: list[str]
    referenced_assertions: list[str]
    referenced_regions: list[int]
    feature_cache_key: str
    cache_key: str
    reused: bool = False

    @property
    def sort_key(self):
        return (self.sample_id, self.region, STAGE_FUSION)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # asdict already recurses into Contribution dataclasses.
        return payload


@dataclass
class LedgerEntry:
    sample_id: str
    region: int
    category: str
    confidence_fixed: int
    boundary: bool
    boundary_reasons: list[str]
    cache_key: str

    @property
    def sort_key(self):
        return self.sample_id, self.region

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
