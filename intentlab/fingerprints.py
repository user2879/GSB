"""Canonical content fingerprints and cache-key derivation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Protocol

from .models import Query, RuleView, TriggerEntry


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def query_content_fingerprint(query: Query) -> str:
    return digest(
        {
            "query_id": query.query_id,
            "terms": list(query.terms),
            "session_id": query.session_id,
            "turn": query.turn,
            "source": query.source,
            "candidates": list(query.candidates),
            "slots": [slot.to_dict() for slot in query.slots],
            "priority": query.priority,
        }
    )


def trigger_entry_fingerprint(entry: TriggerEntry) -> str:
    return digest(
        {
            "trigger_id": entry.trigger_id,
            "terms": list(entry.terms),
            "intent_id": entry.intent_id,
            "weight": entry.weight,
            "slot_pattern": list(entry.slot_pattern),
            "entry_order": entry.entry_order,
            "source_revision": entry.source_revision,
        }
    )


def trigger_match_fingerprint(entry: TriggerEntry) -> str:
    return digest(
        {
            "trigger_id": entry.trigger_id,
            "terms": list(entry.terms),
            "entry_order": entry.entry_order,
        }
    )


def rule_view_fingerprint(view: RuleView) -> str:
    return digest(
        {
            "active_revision_ids": list(view.active_revision_ids),
            "lineage_revision_ids": list(view.lineage_revision_ids),
            "chain_version": view.chain_version,
            "entries": [
                {
                    "trigger_id": entry.trigger_id,
                    "fingerprint": trigger_entry_fingerprint(entry),
                }
                for entry in sorted(view.entries, key=lambda item: item.trigger_id)
            ],
        }
    )


def match_cache_key(
    query: Query,
    considered_entries: Iterable[TriggerEntry],
    *,
    match_rule_version: str,
) -> str:
    entries = sorted(considered_entries, key=lambda entry: entry.trigger_id)
    return digest(
        {
            "stage": "match",
            "query": query_content_fingerprint(query),
            "entries": [
                {
                    "trigger_id": entry.trigger_id,
                    "fingerprint": trigger_match_fingerprint(entry),
                }
                for entry in entries
            ],
            "match_rule_version": match_rule_version,
        }
    )


class ConfigLike(Protocol):
    fraction_bits: int
    quantum: int
    tie_rule: str
    default_threshold: int

    def threshold_for(self, intent_id: str) -> int: ...

    def to_dict(self) -> dict[str, Any]: ...


def score_cache_key(
    query: Query,
    match_key: str,
    referenced_entries: Iterable[TriggerEntry],
    config: ConfigLike,
) -> str:
    entries = sorted(referenced_entries, key=lambda entry: entry.trigger_id)
    relevant_thresholds = {
        intent_id: config.threshold_for(intent_id)
        for intent_id in sorted(set(query.candidates) | {entry.intent_id for entry in entries})
    }
    return digest(
        {
            "stage": "score",
            "query": query_content_fingerprint(query),
            "match_key": match_key,
            "entries": [
                {
                    "trigger_id": entry.trigger_id,
                    "fingerprint": trigger_entry_fingerprint(entry),
                }
                for entry in entries
            ],
            "fraction_bits": config.fraction_bits,
            "quantum": config.quantum,
            "tie_rule": config.tie_rule,
            "default_threshold": config.default_threshold,
            "thresholds": relevant_thresholds,
        }
    )


def segment_cache_key(
    *,
    session_id: str,
    intent: str,
    score_keys: Iterable[str],
    query_fingerprints: Iterable[str],
    merge_config: dict[str, Any],
) -> str:
    return digest(
        {
            "stage": "merge",
            "session_id": session_id,
            "intent": intent,
            "score_keys": sorted(score_keys),
            "query_fingerprints": list(query_fingerprints),
            "merge_config": merge_config,
        }
    )


def state_fingerprint(
    *,
    queries: Iterable[Query],
    revisions: Iterable[Any],
    active_revision_ids: Iterable[str],
    config: ConfigLike,
) -> str:
    return digest(
        {
            "queries": [
                query_content_fingerprint(query)
                for query in sorted(queries, key=lambda item: item.query_id)
            ],
            "revisions": [revision.to_dict() for revision in revisions],
            "active_revision_ids": sorted(active_revision_ids),
            "config": config.to_dict(),
        }
    )

