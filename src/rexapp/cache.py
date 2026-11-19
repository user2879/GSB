"""Content-fingerprint stage cache with auditable reuse and invalidation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RuleConfig
from .documents import Document
from .extraction import Entity, EventMention, ExcludedPair, RelationCandidate
from .pipeline import BoundaryRecord, STAGES, compute_stage_keys
from .pipeline import (
    run_confidence_stage,
    run_entity_stage,
    run_event_stage,
    run_relation_stage,
)

DOWNSTREAM = {
    "entity": ("relation", "event", "confidence"),
    "relation": ("confidence",),
    "event": ("confidence",),
    "confidence": (),
}


@dataclass
class StageEntry:
    key: str
    value: object
    refs: set[str] = field(default_factory=set)


@dataclass
class AuditRecord:
    doc_id: str
    stage: str
    action: str
    key: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "stage": self.stage,
            "action": self.action,
            "key": self.key,
            "reason": self.reason,
        }


class IncrementalCache:
    """Per-document, per-stage cache keyed by content fingerprints."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, StageEntry]] = {}
        self.audit: list[AuditRecord] = []
        self.reused_count = 0
        self.recomputed_count = 0
        self.invalidated_count = 0

    def invalidate_document(self, doc_id: str, reason: str) -> list[str]:
        removed: list[str] = []
        if doc_id in self._entries:
            for stage in STAGES:
                entry = self._entries[doc_id].pop(stage, None)
                if entry is not None:
                    self.invalidated_count += 1
                    removed.append(stage)
                    self.audit.append(
                        AuditRecord(doc_id, stage, "invalidate", entry.key, reason)
                    )
            del self._entries[doc_id]
        return removed

    def invalidate_stage_and_downstream(self, doc_id: str, stage: str, reason: str) -> None:
        bucket = self._entries.get(doc_id)
        if not bucket:
            return
        targets = (stage, *DOWNSTREAM[stage])
        for target in targets:
            entry = bucket.pop(target, None)
            if entry is not None:
                self.invalidated_count += 1
                self.audit.append(AuditRecord(doc_id, target, "invalidate", entry.key, reason))

    def _lookup(self, doc_id: str, stage: str, key: str) -> StageEntry | None:
        return self._entries.get(doc_id, {}).get(stage) if key else None

    def _store(self, doc_id: str, stage: str, key: str, value: object, refs: set[str]) -> object:
        self._entries.setdefault(doc_id, {})[stage] = StageEntry(key=key, value=value, refs=set(refs))
        self.recomputed_count += 1
        self.audit.append(AuditRecord(doc_id, stage, "recompute", key))
        return value

    def process_document(
        self,
        document: Document,
        config: RuleConfig,
        trigger_refs: set[str],
    ) -> dict:
        doc_id = document.doc_id
        keys = compute_stage_keys(document, config)
        result: dict = {}

        entity_entry = self._lookup(doc_id, "entity", keys["entity"])
        if entity_entry is not None and entity_entry.key == keys["entity"]:
            entities: list[Entity] = entity_entry.value  # type: ignore[assignment]
            self.reused_count += 1
            self.audit.append(AuditRecord(doc_id, "entity", "reuse", keys["entity"]))
        else:
            entities = run_entity_stage(document, config)
            self._store(doc_id, "entity", keys["entity"], entities, trigger_refs)
        result["entities"] = entities

        relation_entry = self._lookup(doc_id, "relation", keys["relation"])
        if relation_entry is not None and relation_entry.key == keys["relation"]:
            relation_payload = relation_entry.value
            relations: list[RelationCandidate] = relation_payload[0]  # type: ignore[index]
            excluded: list[ExcludedPair] = relation_payload[1]  # type: ignore[index]
            self.reused_count += 1
            self.audit.append(AuditRecord(doc_id, "relation", "reuse", keys["relation"]))
        else:
            relations, excluded = run_relation_stage(document, config, entities)
            self._store(
                doc_id,
                "relation",
                keys["relation"],
                (relations, excluded),
                trigger_refs,
            )
        result["relations"] = relations
        result["excluded_pairs"] = excluded

        event_entry = self._lookup(doc_id, "event", keys["event"])
        if event_entry is not None and event_entry.key == keys["event"]:
            events: list[EventMention] = event_entry.value  # type: ignore[assignment]
            self.reused_count += 1
            self.audit.append(AuditRecord(doc_id, "event", "reuse", keys["event"]))
        else:
            events = run_event_stage(document, config, entities)
            self._store(doc_id, "event", keys["event"], events, trigger_refs)
        result["events"] = events

        confidence_entry = self._lookup(doc_id, "confidence", keys["confidence"])
        if confidence_entry is not None and confidence_entry.key == keys["confidence"]:
            boundaries: list[BoundaryRecord] = confidence_entry.value  # type: ignore[assignment]
            run_confidence_stage(document, config, relations, events)
            self.reused_count += 1
            self.audit.append(AuditRecord(doc_id, "confidence", "reuse", keys["confidence"]))
        else:
            boundaries = run_confidence_stage(document, config, relations, events)
            self._store(doc_id, "confidence", keys["confidence"], boundaries, trigger_refs)
        result["boundaries"] = boundaries
        result["stage_keys"] = keys
        return result

    def snapshot(self) -> dict:
        return {
            doc_id: {stage: entry.key for stage, entry in bucket.items()}
            for doc_id, bucket in self._entries.items()
        }
