"""Typed data models shared by every pipeline stage."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

LAYER_REQUEST = "request"
LAYER_ITEM = "item"
LAYER_CROSS = "cross"
VALID_LAYERS = (LAYER_REQUEST, LAYER_ITEM, LAYER_CROSS)


@dataclass(frozen=True)
class SlotSpec:
    key: str
    layer: str
    dim: int
    retention: int
    weight: int
    default_vector: tuple
    default_vector_source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Receipt:
    batch_seq: int
    rev_seq: int
    slot_key: str
    effective_ms: int
    points: int
    vector: tuple
    item_id: str | None
    digest: str
    group_key: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadResult:
    config: dict
    slots: dict
    batches: dict
    receipts: list
    failures: list = field(default_factory=list)


@dataclass
class AuthorityRecord:
    group_key: str
    slot_key: str
    item_id: str | None
    receipt: Receipt
    authority_fp: str
    tie_resolved: bool
    history: list = field(default_factory=list)
    pruned: list = field(default_factory=list)


@dataclass
class RejectionRecord:
    digest: str
    batch_seq: int
    rev_seq: int
    slot_key: str
    group_key: str
    reason: str
    conflicting_digest: str | None
    conflicting_rev_seq: int | None
    conflicting_batch_seq: int | None


@dataclass
class SlotResolution:
    slot_key: str
    group_key: str
    item_id: str | None
    rank_ms: int
    used_digest: str | None
    used_rev_seq: int | None
    used_batch_seq: int | None
    effective_ms: int | None
    staleness_ms: int | None
    state: str
    vector: tuple
    vector_source: str


@dataclass
class ScoredCandidate:
    request_id: str
    item_id: str
    total_score: int
    slot_breakdown: list
    resolutions: list
    ranking_cache_key: str
