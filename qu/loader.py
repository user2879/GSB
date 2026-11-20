"""JSONL loaders with schema validation and stable failure records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import fingerprint
from .errors import (
    E_DUPLICATE_QUERY_ID,
    E_EMPTY_QUERY,
    E_INVALID_JSON,
    E_INVALID_WEIGHT,
    E_RULE_DANGLING,
)
from .normalization import normalize_query


def _failure(query_id: Any, stage: str, code: str, message: str,
             exception_type: str, line_no: Optional[int] = None) -> dict:
    record = {
        "query_id": query_id,
        "stage": stage,
        "code": code,
        "message": message,
        "exception_type": exception_type,
    }
    if line_no is not None:
        record["line_no"] = line_no
    return record


def _is_finite_weight(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class QuerySample:
    query_id: str
    query: str
    context: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    line_no: int = 0


@dataclass
class VocabEntry:
    intent: str
    term: str
    weight: float
    line_no: int = 0


@dataclass
class EntityEntry:
    entity_type: str
    term: str
    entity_id: str
    weight: float
    line_no: int = 0


@dataclass
class RewriteRule:
    rule_id: str
    rule_type: str
    trigger_term: str
    replacement: str
    weight: float
    ref_terms: Tuple[str, ...]
    line_no: int = 0


@dataclass
class LoadedBundle:
    samples: List[QuerySample]
    vocab: List[VocabEntry]
    entities: List[EntityEntry]
    rules: List[RewriteRule]
    failures: List[dict]
    table_fingerprints: Dict[str, str]


def load_samples(path: str) -> Tuple[List[QuerySample], List[dict]]:
    samples: List[QuerySample] = []
    failures: List[dict] = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    f"invalid JSON: {exc.msg}", "JSONDecodeError", line_no))
                continue
            if not isinstance(obj, dict):
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    "JSON line is not an object", "TypeError", line_no))
                continue
            query_id = obj.get("query_id")
            raw_query = obj.get("query", obj.get("text"))
            if query_id is None:
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    "missing query_id", "KeyError", line_no))
                continue
            query_id = str(query_id)
            if query_id in seen_ids:
                failures.append(_failure(
                    query_id, "load", E_DUPLICATE_QUERY_ID,
                    "duplicate query_id", "DuplicateQueryIdError", line_no))
                continue
            if not isinstance(raw_query, str) or raw_query.strip() == "":
                failures.append(_failure(
                    query_id, "load", E_EMPTY_QUERY,
                    "query text is empty", "EmptyQueryError", line_no))
                seen_ids.add(query_id)
                continue
            # Reject punctuation-only / whitespace-only queries early.
            try:
                normalize_query(raw_query)
            except Exception as exc:  # EmptyQueryError by construction
                failures.append(_failure(
                    query_id, "load", E_EMPTY_QUERY,
                    str(exc), type(exc).__name__, line_no))
                seen_ids.add(query_id)
                continue
            seen_ids.add(query_id)
            samples.append(QuerySample(
                query_id=query_id,
                query=raw_query,
                context=obj.get("context") if isinstance(obj.get("context"), dict) else {},
                labels=obj.get("labels") if isinstance(obj.get("labels"), dict) else {},
                line_no=line_no,
            ))
    return samples, failures


def load_intent_vocab(path: str) -> Tuple[List[VocabEntry], List[dict]]:
    entries: List[VocabEntry] = []
    failures: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    f"invalid JSON: {exc.msg}", "JSONDecodeError", line_no))
                continue
            weight = obj.get("weight", 1.0)
            if not _is_finite_weight(weight):
                failures.append(_failure(
                    obj.get("intent", f"<line:{line_no}>"), "load",
                    E_INVALID_WEIGHT, f"non-finite weight: {weight!r}",
                    "InvalidWeightError", line_no))
                continue
            term = obj.get("term")
            intent = obj.get("intent")
            if not isinstance(term, str) or not term or not isinstance(intent, str):
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    "vocab entry requires non-empty intent and term",
                    "ValueError", line_no))
                continue
            entries.append(VocabEntry(intent=intent, term=term,
                                      weight=float(weight), line_no=line_no))
    return entries, failures


def load_entity_dict(path: str) -> Tuple[List[EntityEntry], List[dict]]:
    entries: List[EntityEntry] = []
    failures: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    f"invalid JSON: {exc.msg}", "JSONDecodeError", line_no))
                continue
            weight = obj.get("weight", 1.0)
            if not _is_finite_weight(weight):
                failures.append(_failure(
                    obj.get("entity_id", f"<line:{line_no}>"), "load",
                    E_INVALID_WEIGHT, f"non-finite weight: {weight!r}",
                    "InvalidWeightError", line_no))
                continue
            entity_type = obj.get("entity_type")
            term = obj.get("term")
            entity_id = obj.get("entity_id")
            if entity_type not in ("brand", "category", "attribute") or \
                    not isinstance(term, str) or not term or \
                    not isinstance(entity_id, str) or not entity_id:
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    "entity entry requires entity_type/term/entity_id",
                    "ValueError", line_no))
                continue
            entries.append(EntityEntry(
                entity_type=entity_type, term=term, entity_id=entity_id,
                weight=float(weight), line_no=line_no))
    return entries, failures


def load_rewrite_rules(path: str, known_terms: Sequence[str]) -> Tuple[List[RewriteRule], List[dict]]:
    rules: List[RewriteRule] = []
    failures: List[dict] = []
    known = set(known_terms)
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(_failure(
                    f"<line:{line_no}>", "load", E_INVALID_JSON,
                    f"invalid JSON: {exc.msg}", "JSONDecodeError", line_no))
                continue
            rule_id = str(obj.get("rule_id", f"<line:{line_no}>"))
            weight = obj.get("weight", 1.0)
            if not _is_finite_weight(weight):
                failures.append(_failure(
                    rule_id, "load", E_INVALID_WEIGHT,
                    f"non-finite weight: {weight!r}", "InvalidWeightError", line_no))
                continue
            trigger = obj.get("trigger_term", "")
            ref_terms = tuple(obj.get("ref_terms") or [trigger])
            dangling = [term for term in ref_terms if term not in known]
            if dangling:
                failures.append(_failure(
                    rule_id, "load", E_RULE_DANGLING,
                    f"rule references unknown terms: {dangling}",
                    "RuleDanglingError", line_no))
                continue
            rules.append(RewriteRule(
                rule_id=rule_id,
                rule_type=obj.get("type", "synonym"),
                trigger_term=trigger,
                replacement=obj.get("replacement", ""),
                weight=float(weight),
                ref_terms=ref_terms,
                line_no=line_no,
            ))
    return rules, failures


def _table_fingerprint(rows: Sequence[Any], attrs: Sequence[str]) -> str:
    payload = [[getattr(row, attr) for attr in attrs] for row in rows]
    return fingerprint(payload)


def load_bundle(samples_path: str, vocab_path: str, entity_path: str,
                rules_path: str) -> LoadedBundle:
    samples, sample_failures = load_samples(samples_path)
    vocab, vocab_failures = load_intent_vocab(vocab_path)
    entities, entity_failures = load_entity_dict(entity_path)
    known_terms = [entry.term for entry in vocab] + [entry.term for entry in entities]
    rules, rule_failures = load_rewrite_rules(rules_path, known_terms)
    failures = sample_failures + vocab_failures + entity_failures + rule_failures
    failures.sort(key=lambda item: (str(item["query_id"]), item["stage"], item["code"]))
    return LoadedBundle(
        samples=samples,
        vocab=vocab,
        entities=entities,
        rules=rules,
        failures=failures,
        table_fingerprints={
            "samples": fingerprint([s.query_id for s in samples]),
            "sample_content": fingerprint(
                [[s.query_id, s.query, s.context, s.labels] for s in samples]),
            "vocab": _table_fingerprint(vocab, ("intent", "term", "weight")),
            "entities": _table_fingerprint(
                entities, ("entity_type", "term", "entity_id", "weight")),
            "rules": _table_fingerprint(
                rules, ("rule_id", "rule_type", "trigger_term", "replacement", "weight")),
        },
    )
