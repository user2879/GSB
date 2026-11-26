"""Core domain models for corpus, trigger revisions and routing config."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class TextItem:
    """A single short text made of controlled-vocabulary subword tokens."""

    text_id: str
    tokens: tuple[str, ...]
    channel: str
    gold_label: str

    def joined(self) -> str:
        return "".join(self.tokens)

    def content_key(self) -> tuple[str, ...]:
        return self.tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_id": self.text_id,
            "tokens": list(self.tokens),
            "channel": self.channel,
            "gold_label": self.gold_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextItem":
        return cls(
            text_id=str(data["text_id"]),
            tokens=tuple(str(t) for t in data["tokens"]),
            channel=str(data["channel"]),
            gold_label=str(data["gold_label"]),
        )


@dataclass(frozen=True)
class TriggerOp:
    """A single revision operation on one trigger entry.

    ``op`` is one of ``upsert`` / ``delete``.  ``term`` is the subword token
    sequence of the trigger; weight and intent ownership may be reassigned.
    """

    op: str
    entry_id: str
    term: tuple[str, ...] = ()
    weight: int = 0
    intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "entry_id": self.entry_id,
            "term": list(self.term),
            "weight": self.weight,
            "intent": self.intent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TriggerOp":
        return cls(
            op=str(data["op"]),
            entry_id=str(data["entry_id"]),
            term=tuple(str(t) for t in data.get("term", ())),
            weight=int(data.get("weight", 0)),
            intent=str(data.get("intent", "")),
        )


@dataclass(frozen=True)
class Revision:
    """A trigger-library revision explicitly referencing a parent revision."""

    revision_id: str
    parent_id: str
    sequence: int
    ops: tuple[TriggerOp, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "parent_id": self.parent_id,
            "sequence": self.sequence,
            "ops": [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Revision":
        return cls(
            revision_id=str(data["revision_id"]),
            parent_id=str(data["parent_id"]),
            sequence=int(data["sequence"]),
            ops=tuple(TriggerOp.from_dict(op) for op in data.get("ops", [])),
        )


@dataclass(frozen=True)
class EffectiveEntry:
    """A trigger entry after revision-chain synthesis with provenance."""

    entry_id: str
    term: tuple[str, ...]
    weight: int
    intent: str
    born_revision: str
    last_revision: str
    overrides: tuple[str, ...]

    def term_key(self) -> tuple[str, ...]:
        return self.term

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "term": list(self.term),
            "weight": self.weight,
            "intent": self.intent,
            "born_revision": self.born_revision,
            "last_revision": self.last_revision,
            "overrides": list(self.overrides),
        }


@dataclass(frozen=True)
class RoutingConfig:
    """Routing, fixed-point and scheduling configuration."""

    fraction_bits: int
    gear: int
    intents: tuple[str, ...]
    intent_thresholds: dict[str, int]
    dedup_threshold_permille: int
    max_workers: int
    global_match_budget: int
    channel_quotas: dict[str, int]
    per_batch_recompute_limit: int
    fallback_intent: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingConfig":
        return cls(
            fraction_bits=int(data["fraction_bits"]),
            gear=int(data["gear"]),
            intents=tuple(str(i) for i in data["intents"]),
            intent_thresholds={
                str(k): int(v) for k, v in data["intent_thresholds"].items()
            },
            dedup_threshold_permille=int(data["dedup_threshold_permille"]),
            max_workers=int(data["max_workers"]),
            global_match_budget=int(data["global_match_budget"]),
            channel_quotas={
                str(k): int(v) for k, v in data["channel_quotas"].items()
            },
            per_batch_recompute_limit=int(data["per_batch_recompute_limit"]),
            fallback_intent=str(data["fallback_intent"]),
        )


@dataclass(frozen=True)
class Corpus:
    """A loaded corpus batch snapshot."""

    texts: tuple[TextItem, ...]
    vocab: tuple[str, ...]

    def by_id(self) -> dict[str, TextItem]:
        return {item.text_id: item for item in self.texts}


@dataclass(frozen=True)
class RevisionChain:
    root_id: str
    head_id: str
    revisions: tuple[Revision, ...]

    def ordered(self) -> list[Revision]:
        by_id = {rev.revision_id: rev for rev in self.revisions}
        order: list[Revision] = []
        cursor = self.head_id
        seen: set[str] = set()
        while cursor != self.root_id:
            if cursor in seen:
                raise ValueError(f"revision cycle at {cursor}")
            seen.add(cursor)
            rev = by_id[cursor]
            order.append(rev)
            cursor = rev.parent_id
        order.reverse()
        return order
