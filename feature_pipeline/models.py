"""Unified intermediate representation shared by the three parsers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


V1 = "v1"
V2 = "v2"
V3 = "v3"
SUPPORTED_VERSIONS = (V1, V2, V3)

# Precision labels allowed unless the metadata table restricts a feature.
GLOBAL_PRECISION_LABELS = ("per_mille", "float32", "float64")


def stable_sort_key(entity_id: str, stage: str) -> tuple:
    """Deterministic task ordering key derived from entity id and stage."""
    return (str(entity_id), str(stage))


@dataclass
class Cell:
    """One (entity, feature) observation retained with provenance.

    ``value`` is the parsed numeric value in the canonical float64 domain and
    may be NaN only when ``missing`` is true. ``raw_text`` keeps the original
    fixed-width/JSON token while ``raw_bytes`` keeps the source bytes for v3.
    """

    entity_id: str
    feature: str
    source_version: str
    value: float
    missing: bool
    precision_label: Optional[str] = None
    raw_text: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    partition_id: Optional[int] = None
    filled: bool = False

    def clone(self, **changes) -> "Cell":
        data = {
            "entity_id": self.entity_id,
            "feature": self.feature,
            "source_version": self.source_version,
            "value": self.value,
            "missing": self.missing,
            "precision_label": self.precision_label,
            "raw_text": self.raw_text,
            "raw_bytes": self.raw_bytes,
            "partition_id": self.partition_id,
            "filled": self.filled,
        }
        data.update(changes)
        return Cell(**data)

    def value_json(self) -> Optional[float]:
        if self.missing:
            return None
        value = float(self.value)
        return value if math.isfinite(value) else None


@dataclass
class EntityTable:
    """Entity-level feature table produced by one source parser."""

    source_version: str
    cells: list[Cell] = field(default_factory=list)
    feature_order: list[str] = field(default_factory=list)
    partition_boundaries: dict[int, list[str]] = field(default_factory=dict)

    def entity_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for cell in self.cells:
            seen.setdefault(cell.entity_id, None)
        return list(seen.keys())

    def feature_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in self.feature_order:
            seen.setdefault(name, None)
        for cell in self.cells:
            seen.setdefault(cell.feature, None)
        return list(seen.keys())

    def cell_map(self) -> dict[tuple[str, str], Cell]:
        return {(cell.entity_id, cell.feature): cell for cell in self.cells}


@dataclass(frozen=True)
class FeatureMeta:
    """One row of the feature metadata table."""

    feature: str
    semantic_type: str
    allowed_precisions: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict) -> "FeatureMeta":
        allowed = tuple(
            part.strip()
            for part in str(row.get("allowed_precisions", "")).split("|")
            if part.strip()
        )
        return cls(
            feature=str(row["feature"]).strip(),
            semantic_type=str(row["semantic_type"]).strip(),
            allowed_precisions=allowed,
        )


@dataclass
class MergeConfig:
    """Validated merge configuration."""

    feature_whitelist: list[str]
    scaling_method: str  # raw | percentile | zscore
    missing_fill: dict
    conflict_priority: str  # source_version | precision_label | abs_value
    source_order: list[str]
    precision_rank: list[str]
    numeric_tolerance: float
    max_workers: int
    batch_size: int
    percentile_tie_rule: str = "average_rank"

    def to_dict(self) -> dict:
        return {
            "feature_whitelist": list(self.feature_whitelist),
            "scaling_method": self.scaling_method,
            "missing_fill": dict(self.missing_fill),
            "conflict_priority": self.conflict_priority,
            "source_order": list(self.source_order),
            "precision_rank": list(self.precision_rank),
            "numeric_tolerance": self.numeric_tolerance,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "percentile_tie_rule": self.percentile_tie_rule,
        }
