"""Loading, validation, and sequential edit application."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import Codes, IssueCollector
from .models import Query, Revision, RevisionOp, RuntimeConfig, Slot


DEFAULT_ALLOWED_FRACTION_BITS = (8, 12, 16, 20)
DEFAULT_ALLOWED_QUANTUMS = (1, 4, 16, 64)
DEFAULT_ALLOWED_WEIGHTS = (600, 700, 800, 900, 1000, 1100, 1200)


@dataclass(frozen=True)
class LoadedInputs:
    config: RuntimeConfig
    vocabulary: tuple[str, ...]
    queries: tuple[Query, ...]
    revisions: tuple[Revision, ...]
    active_revision_ids: tuple[str, ...]


def load_json(path: str | Path, issues: IssueCollector, *, scope: str) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        issues.add(
            Codes.JSON_INVALID,
            f"file not found: {path}",
            scope=scope,
            retained=False,
            details={"path": str(path)},
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.add(
            Codes.JSON_INVALID,
            f"cannot parse JSON: {exc}",
            scope=scope,
            retained=False,
            details={"path": str(path)},
        )
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[int] = []
    for item in value:
        if not _is_int(item) or item <= 0:
            return None
        result.append(item)
    return tuple(sorted(set(result)))


def _as_positive_int(
    value: Any,
    *,
    fallback: int,
    code: str,
    field: str,
    issues: IssueCollector,
    allow_zero: bool = False,
) -> int:
    minimum = 0 if allow_zero else 1
    if not _is_int(value) or value < minimum:
        issues.add(
            code,
            f"{field} must be an integer >= {minimum}",
            scope="config",
            details={"field": field, "value": value, "fallback": fallback},
        )
        return fallback
    return value


def parse_config(data: Any, issues: IssueCollector) -> RuntimeConfig:
    if not isinstance(data, dict):
        issues.add(
            Codes.CONFIG_INVALID,
            "config root must be an object; defaults used",
            scope="config",
            retained=False,
        )
        data = {}

    allowed_section = data.get("allowed", {})
    if not isinstance(allowed_section, dict):
        issues.add(Codes.CONFIG_INVALID, "config.allowed must be an object", scope="config")
        allowed_section = {}

    parsed_bits = _int_tuple(allowed_section.get("fraction_bits"))
    allowed_bits = parsed_bits or DEFAULT_ALLOWED_FRACTION_BITS
    if parsed_bits is None and "fraction_bits" in allowed_section:
        issues.add(
            Codes.FRACTION_BITS_ENUM,
            "allowed fraction bits must be a non-empty positive integer enum",
            scope="config",
            details={"fallback": list(allowed_bits)},
        )

    parsed_quantums = _int_tuple(allowed_section.get("quantums"))
    allowed_quantums = parsed_quantums or DEFAULT_ALLOWED_QUANTUMS
    if parsed_quantums is None and "quantums" in allowed_section:
        issues.add(
            Codes.QUANTUM_ENUM,
            "allowed quantums must be a non-empty positive integer enum",
            scope="config",
            details={"fallback": list(allowed_quantums)},
        )

    parsed_weights = _int_tuple(allowed_section.get("weights"))
    allowed_weights = parsed_weights or DEFAULT_ALLOWED_WEIGHTS
    if parsed_weights is None and "weights" in allowed_section:
        issues.add(
            Codes.TRIGGER_WEIGHT_ENUM,
            "allowed weights must be a non-empty positive integer enum",
            scope="config",
            details={"fallback": list(allowed_weights)},
        )

    fixed = data.get("fixed_point", {})
    if not isinstance(fixed, dict):
        issues.add(Codes.CONFIG_INVALID, "config.fixed_point must be an object", scope="config")
        fixed = {}

    fraction_bits = fixed.get("fraction_bits", allowed_bits[0])
    if not _is_int(fraction_bits) or fraction_bits not in allowed_bits:
        issues.add(
            Codes.FRACTION_BITS_ENUM,
            "fraction_bits is outside the allowed enum",
            scope="config",
            entity_id="fixed_point",
            details={"value": fraction_bits, "allowed": list(allowed_bits)},
        )
        fraction_bits = allowed_bits[0]

    quantum = fixed.get("quantum", allowed_quantums[0])
    if not _is_int(quantum) or quantum not in allowed_quantums:
        issues.add(
            Codes.QUANTUM_ENUM,
            "quantum is outside the allowed enum",
            scope="config",
            entity_id="fixed_point",
            details={"value": quantum, "allowed": list(allowed_quantums)},
        )
        quantum = allowed_quantums[0]

    threshold_section = data.get("thresholds", {})
    if not isinstance(threshold_section, dict):
        issues.add(Codes.CONFIG_INVALID, "config.thresholds must be an object", scope="config")
        threshold_section = {}

    default_threshold = _as_positive_int(
        threshold_section.get("default", 120),
        fallback=120,
        code=Codes.THRESHOLD_INVALID,
        field="thresholds.default",
        issues=issues,
        allow_zero=True,
    )
    raw_thresholds = threshold_section.get("intents", {})
    thresholds: dict[str, int] = {}
    if not isinstance(raw_thresholds, dict):
        issues.add(
            Codes.THRESHOLD_INVALID,
            "thresholds.intents must be an object",
            scope="config",
        )
    else:
        for intent_id in sorted(raw_thresholds):
            value = raw_thresholds[intent_id]
            if not isinstance(intent_id, str) or not intent_id:
                issues.add(
                    Codes.THRESHOLD_INVALID,
                    "threshold intent id must be a non-empty string",
                    scope="config",
                )
            elif not _is_int(value) or value < 0:
                issues.add(
                    Codes.THRESHOLD_INVALID,
                    "intent threshold must be a non-negative integer",
                    scope="config",
                    entity_id=intent_id,
                    details={"value": value},
                )
            else:
                thresholds[intent_id] = value

    tie_rule = threshold_section.get("tie_rule", "accept_on_tie")
    if tie_rule not in {"accept_on_tie", "reject_on_tie"}:
        issues.add(
            Codes.CONFIG_INVALID,
            "thresholds.tie_rule must be accept_on_tie or reject_on_tie",
            scope="config",
            details={"value": tie_rule},
        )
        tie_rule = "accept_on_tie"

    diagnostics = data.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        issues.add(Codes.CONFIG_INVALID, "config.diagnostics must be an object", scope="config")
        diagnostics = {}
    tolerance = diagnostics.get("tolerance", 0.001)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance <= 0:
        issues.add(
            Codes.CONFIG_INVALID,
            "diagnostics.tolerance must be a positive number",
            scope="config",
            details={"value": tolerance},
        )
        tolerance = 0.001
    reference_extra_bits = _as_positive_int(
        diagnostics.get("reference_extra_bits", 4),
        fallback=4,
        code=Codes.CONFIG_INVALID,
        field="diagnostics.reference_extra_bits",
        issues=issues,
    )

    scheduler = data.get("scheduler", {})
    if not isinstance(scheduler, dict):
        issues.add(Codes.CONFIG_INVALID, "config.scheduler must be an object", scope="config")
        scheduler = {}
    workers = _as_positive_int(
        scheduler.get("workers", 4),
        fallback=4,
        code=Codes.CONFIG_INVALID,
        field="scheduler.workers",
        issues=issues,
    )
    if workers > 256:
        issues.add(
            Codes.CONFIG_INVALID,
            "scheduler.workers is capped at 256",
            scope="config",
            details={"value": workers},
        )
        workers = 256
    match_budget_points = _as_positive_int(
        scheduler.get("match_budget_points", 100_000),
        fallback=100_000,
        code=Codes.CONFIG_INVALID,
        field="scheduler.match_budget_points",
        issues=issues,
        allow_zero=True,
    )
    batch_query_limit = _as_positive_int(
        scheduler.get("batch_query_limit", 10_000),
        fallback=10_000,
        code=Codes.CONFIG_INVALID,
        field="scheduler.batch_query_limit",
        issues=issues,
    )
    raw_quotas = scheduler.get("source_quotas", {})
    source_quotas: dict[str, int] = {}
    if not isinstance(raw_quotas, dict):
        issues.add(
            Codes.CONFIG_INVALID,
            "scheduler.source_quotas must be an object",
            scope="config",
        )
    else:
        for source in sorted(raw_quotas):
            value = raw_quotas[source]
            if not isinstance(source, str) or not source:
                issues.add(
                    Codes.CONFIG_INVALID,
                    "source quota key must be a non-empty string",
                    scope="config",
                )
            elif not _is_int(value) or value < 0:
                issues.add(
                    Codes.CONFIG_INVALID,
                    "source quota must be a non-negative integer",
                    scope="config",
                    entity_id=source,
                )
            else:
                source_quotas[source] = value

    merge = data.get("merge", {})
    if not isinstance(merge, dict):
        issues.add(Codes.CONFIG_INVALID, "config.merge must be an object", scope="config")
        merge = {}
    max_turn_gap = _as_positive_int(
        merge.get("max_turn_gap", 1),
        fallback=1,
        code=Codes.CONFIG_INVALID,
        field="merge.max_turn_gap",
        issues=issues,
    )
    term_gap = _as_positive_int(
        merge.get("term_gap", 1),
        fallback=1,
        code=Codes.CONFIG_INVALID,
        field="merge.term_gap",
        issues=issues,
        allow_zero=True,
    )

    global_version = _as_positive_int(
        data.get("global_version", 1),
        fallback=1,
        code=Codes.CONFIG_INVALID,
        field="global_version",
        issues=issues,
        allow_zero=True,
    )
    fallback_intent = data.get("fallback_intent", "unknown")
    if not isinstance(fallback_intent, str) or not fallback_intent:
        issues.add(
            Codes.CONFIG_INVALID,
            "fallback_intent must be a non-empty string",
            scope="config",
        )
        fallback_intent = "unknown"

    return RuntimeConfig(
        fraction_bits=fraction_bits,
        quantum=quantum,
        allowed_fraction_bits=allowed_bits,
        allowed_quantums=allowed_quantums,
        allowed_weights=allowed_weights,
        thresholds=thresholds,
        default_threshold=default_threshold,
        tie_rule=tie_rule,
        fallback_intent=fallback_intent,
        tolerance=float(tolerance),
        reference_extra_bits=reference_extra_bits,
        workers=workers,
        match_budget_points=match_budget_points,
        source_quotas=source_quotas,
        batch_query_limit=batch_query_limit,
        max_turn_gap=max_turn_gap,
        term_gap=term_gap,
        global_version=global_version,
        raw=copy.deepcopy(data),
    )


def parse_vocabulary(data: Any, issues: IssueCollector) -> tuple[str, ...]:
    if isinstance(data, dict):
        values = data.get("terms")
    else:
        values = data
    if not isinstance(values, list):
        issues.add(
            Codes.SCHEMA_INVALID,
            "controlled vocabulary must be a list or an object with terms",
            scope="vocabulary",
            retained=False,
        )
        return ()
    seen: dict[str, None] = {}
    for index, term in enumerate(values):
        if not isinstance(term, str) or not term.strip():
            issues.add(
                Codes.SCHEMA_INVALID,
                "controlled vocabulary term must be a non-empty string",
                scope="vocabulary",
                entity_id=str(index),
            )
            continue
        seen.setdefault(term, None)
    return tuple(sorted(seen))


def _parse_slots(value: Any, term_count: int, query_id: str, issues: IssueCollector) -> tuple[Slot, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        issues.add(
            Codes.SCHEMA_INVALID,
            "slots must be a list",
            scope="query",
            entity_id=query_id,
        )
        return ()
    slots: list[Slot] = []
    for index, raw in enumerate(value):
        slot_ref = f"{query_id}:slot[{index}]"
        if not isinstance(raw, dict):
            issues.add(Codes.SCHEMA_INVALID, "slot must be an object", scope="query", entity_id=query_id)
            continue
        name = raw.get("name")
        start = raw.get("start")
        end = raw.get("end")
        if not isinstance(name, str) or not name:
            issues.add(
                Codes.SLOT_NAME_INVALID,
                "slot name must be a non-empty string",
                scope="query",
                entity_id=slot_ref,
            )
            continue
        if not _is_int(start) or not _is_int(end):
            issues.add(
                Codes.SLOT_TYPE,
                "slot start/end must be integers",
                scope="query",
                entity_id=slot_ref,
                details={"start": start, "end": end},
            )
            continue
        if start < 0 or end > term_count or start >= end:
            issues.add(
                Codes.SLOT_RANGE,
                "slot must satisfy 0 <= start < end <= term_count",
                scope="query",
                entity_id=slot_ref,
                details={"start": start, "end": end, "term_count": term_count},
            )
            continue
        slots.append(Slot(name=name, start=start, end=end))
    return tuple(slots)


def parse_query(
    raw: Any,
    vocabulary: set[str] | frozenset[str],
    issues: IssueCollector,
    *,
    existing_ids: set[str] | None = None,
    reject_duplicate: bool = True,
) -> Query | None:
    if not isinstance(raw, dict):
        issues.add(Codes.SCHEMA_INVALID, "query must be an object", scope="query", retained=False)
        return None
    query_id = raw.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        issues.add(
            Codes.QUERY_ID_INVALID,
            "query_id must be a non-empty string",
            scope="query",
            retained=False,
        )
        return None
    if reject_duplicate and existing_ids is not None and query_id in existing_ids:
        issues.add(
            Codes.QUERY_ID_DUPLICATE,
            "duplicate query_id; later occurrence rejected",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None
    terms_raw = raw.get("terms")
    if not isinstance(terms_raw, list) or not terms_raw:
        issues.add(
            Codes.QUERY_TERMS_INVALID,
            "terms must be a non-empty list",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None
    terms: list[str] = []
    term_error = False
    for index, term in enumerate(terms_raw):
        if not isinstance(term, str) or not term or term not in vocabulary:
            issues.add(
                Codes.QUERY_TERM_UNKNOWN,
                "query term is not a string in the controlled vocabulary",
                scope="query",
                entity_id=query_id,
                details={"index": index, "term": term},
            )
            term_error = True
        else:
            terms.append(term)
    if term_error:
        return None

    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        issues.add(
            Codes.CONTEXT_INVALID,
            "session_id must be a non-empty string",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None
    turn = raw.get("turn")
    if not _is_int(turn) or turn < 0:
        issues.add(
            Codes.CONTEXT_INVALID,
            "turn must be a non-negative integer",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None
    source = raw.get("source", "default")
    if not isinstance(source, str) or not source:
        issues.add(
            Codes.CONTEXT_INVALID,
            "source must be a non-empty string",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None

    ground_truth = raw.get("ground_truth_intent", raw.get("intent"))
    if not isinstance(ground_truth, str) or not ground_truth:
        issues.add(
            Codes.SCHEMA_INVALID,
            "ground_truth_intent must be a non-empty string",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        issues.add(
            Codes.CANDIDATES_INVALID,
            "candidates must be a non-empty list",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None
    candidates: list[str] = []
    for candidate in candidates_raw:
        if not isinstance(candidate, str) or not candidate:
            issues.add(
                Codes.CANDIDATES_INVALID,
                "candidate intent ids must be non-empty strings",
                scope="query",
                entity_id=query_id,
            )
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        issues.add(
            Codes.CANDIDATES_INVALID,
            "at least one valid candidate intent is required",
            scope="query",
            entity_id=query_id,
            retained=False,
        )
        return None

    priority = raw.get("priority", 0)
    if not _is_int(priority):
        issues.add(
            Codes.SCHEMA_INVALID,
            "priority must be an integer",
            scope="query",
            entity_id=query_id,
        )
        priority = 0
    slots = _parse_slots(raw.get("slots", []), len(terms), query_id, issues)
    text = " ".join(terms)
    provided_text = raw.get("text")
    if provided_text is not None and provided_text != text:
        issues.add(
            Codes.SCHEMA_INVALID,
            "text is derived from terms; provided text was ignored",
            scope="query",
            entity_id=query_id,
            retained=True,
            details={"provided": provided_text, "derived": text},
        )
    return Query(
        query_id=query_id,
        terms=tuple(terms),
        session_id=session_id,
        turn=turn,
        source=source,
        ground_truth_intent=ground_truth,
        candidates=tuple(candidates),
        slots=slots,
        priority=priority,
        text=text,
    )


def parse_queries(
    data: Any,
    vocabulary: Sequence[str],
    issues: IssueCollector,
) -> tuple[Query, ...]:
    if isinstance(data, dict):
        values = data.get("queries")
    else:
        values = data
    if not isinstance(values, list):
        issues.add(
            Codes.SCHEMA_INVALID,
            "queries must be a list or an object with queries",
            scope="query",
            retained=False,
        )
        return ()
    vocabulary_set = frozenset(vocabulary)
    seen: set[str] = set()
    result: list[Query] = []
    for raw in values:
        query = parse_query(raw, vocabulary_set, issues, existing_ids=seen)
        if query is not None:
            seen.add(query.query_id)
            result.append(query)
    return tuple(sorted(result, key=lambda item: (item.session_id, item.turn, item.query_id)))


def _parse_slot_pattern(
    value: Any,
    *,
    revision_id: str,
    trigger_id: str,
    issues: IssueCollector,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        issues.add(
            Codes.TRIGGER_SLOT_PATTERN_INVALID,
            "slot_pattern must be a list of non-empty strings",
            scope="revision",
            entity_id=f"{revision_id}:{trigger_id}",
        )
        return None
    return tuple(value)


def parse_revision(
    raw: Any,
    vocabulary: set[str] | frozenset[str],
    allowed_weights: Sequence[int],
    issues: IssueCollector,
) -> Revision | None:
    if not isinstance(raw, dict):
        issues.add(Codes.SCHEMA_INVALID, "revision must be an object", scope="revision")
        return None
    revision_id = raw.get("revision_id")
    if not isinstance(revision_id, str) or not revision_id:
        issues.add(
            Codes.SCHEMA_INVALID,
            "revision_id must be a non-empty string",
            scope="revision",
            retained=False,
        )
        return None
    parent_id = raw.get("parent_id")
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        issues.add(
            Codes.PARENT_INVALID,
            "parent_id must be null or a non-empty string",
            scope="revision",
            entity_id=revision_id,
            retained=False,
        )
        return None
    seq = raw.get("seq", 0)
    if not _is_int(seq) or seq < 0:
        issues.add(
            Codes.SCHEMA_INVALID,
            "revision seq must be a non-negative integer",
            scope="revision",
            entity_id=revision_id,
        )
        seq = 0
    label = raw.get("label", revision_id)
    if not isinstance(label, str):
        label = revision_id
    ops_raw = raw.get("ops", [])
    if not isinstance(ops_raw, list):
        issues.add(
            Codes.REVISION_OP_INVALID,
            "revision ops must be a list",
            scope="revision",
            entity_id=revision_id,
        )
        ops_raw = []
    ops: list[RevisionOp] = []
    for index, op_raw in enumerate(ops_raw):
        op_ref = f"{revision_id}:op[{index}]"
        if not isinstance(op_raw, dict):
            issues.add(Codes.REVISION_OP_INVALID, "op must be an object", scope="revision", entity_id=op_ref)
            continue
        op = op_raw.get("op")
        trigger_id = op_raw.get("trigger_id")
        if op not in {"add", "update", "delete"}:
            issues.add(
                Codes.REVISION_OP_INVALID,
                "op must be add, update, or delete",
                scope="revision",
                entity_id=op_ref,
                details={"op": op},
            )
            continue
        if not isinstance(trigger_id, str) or not trigger_id:
            issues.add(
                Codes.TRIGGER_ID_INVALID,
                "trigger_id must be a non-empty string",
                scope="revision",
                entity_id=op_ref,
            )
            continue

        terms: tuple[str, ...] | None = None
        if "terms" in op_raw:
            terms_raw = op_raw.get("terms")
            if not isinstance(terms_raw, list) or not terms_raw:
                issues.add(
                    Codes.REVISION_OP_INVALID,
                    "terms must be a non-empty list when provided",
                    scope="revision",
                    entity_id=op_ref,
                )
                continue
            valid_terms: list[str] = []
            unknown = False
            for term in terms_raw:
                if not isinstance(term, str) or not term or term not in vocabulary:
                    issues.add(
                        Codes.TRIGGER_TERM_UNKNOWN,
                        "trigger term is outside the controlled vocabulary",
                        scope="revision",
                        entity_id=op_ref,
                        details={"term": term},
                    )
                    unknown = True
                else:
                    valid_terms.append(term)
            if unknown or not valid_terms:
                continue
            terms = tuple(valid_terms)

        intent_id = op_raw.get("intent_id")
        if intent_id is not None and (not isinstance(intent_id, str) or not intent_id):
            issues.add(
                Codes.REVISION_OP_INVALID,
                "intent_id must be a non-empty string when provided",
                scope="revision",
                entity_id=op_ref,
            )
            continue
        weight = op_raw.get("weight")
        if weight is not None:
            if not _is_int(weight):
                issues.add(
                    Codes.TRIGGER_WEIGHT_INVALID,
                    "trigger weight must be an integer",
                    scope="revision",
                    entity_id=op_ref,
                )
                continue
            if weight not in allowed_weights:
                issues.add(
                    Codes.TRIGGER_WEIGHT_ENUM,
                    "trigger weight is outside the allowed enum",
                    scope="revision",
                    entity_id=op_ref,
                    details={"weight": weight, "allowed": list(allowed_weights)},
                )
                continue
        slot_pattern = _parse_slot_pattern(
            op_raw.get("slot_pattern"),
            revision_id=revision_id,
            trigger_id=trigger_id,
            issues=issues,
        )
        if "slot_pattern" in op_raw and slot_pattern is None:
            continue

        if op == "add":
            if terms is None or intent_id is None or weight is None:
                issues.add(
                    Codes.REVISION_OP_INVALID,
                    "add requires terms, intent_id, and weight",
                    scope="revision",
                    entity_id=op_ref,
                )
                continue
        elif op == "update":
            if terms is None and intent_id is None and weight is None and slot_pattern is None:
                issues.add(
                    Codes.REVISION_OP_INVALID,
                    "update requires at least one changed field",
                    scope="revision",
                    entity_id=op_ref,
                )
                continue
        else:
            if any(field is not None for field in (terms, intent_id, weight, slot_pattern)):
                issues.add(
                    Codes.REVISION_OP_INVALID,
                    "delete does not accept payload fields",
                    scope="revision",
                    entity_id=op_ref,
                )
                continue
        ops.append(
            RevisionOp(
                op=op,
                trigger_id=trigger_id,
                terms=terms,
                intent_id=intent_id,
                weight=weight,
                slot_pattern=slot_pattern,
            )
        )
    return Revision(
        revision_id=revision_id,
        parent_id=parent_id,
        seq=seq,
        label=label,
        ops=tuple(ops),
    )


def _prune_invalid_revision_graph(
    revisions: dict[str, Revision], issues: IssueCollector
) -> dict[str, Revision]:
    accepted = dict(revisions)
    changed = True
    while changed:
        changed = False
        for revision_id in sorted(list(accepted)):
            revision = accepted[revision_id]
            if revision.parent_id is not None and revision.parent_id not in accepted:
                issues.add(
                    Codes.PARENT_MISSING,
                    "revision parent does not exist; revision rejected",
                    scope="revision",
                    entity_id=revision_id,
                    retained=False,
                    details={"parent_id": revision.parent_id},
                )
                del accepted[revision_id]
                changed = True

    def cycle_nodes() -> set[str]:
        nodes: set[str] = set()
        for start in accepted:
            path: list[str] = []
            seen_in_path: dict[str, int] = {}
            current: str | None = start
            while current is not None and current in accepted:
                if current in seen_in_path:
                    nodes.update(path[seen_in_path[current] :])
                    break
                seen_in_path[current] = len(path)
                path.append(current)
                current = accepted[current].parent_id
        return nodes

    cyclic = cycle_nodes()
    for revision_id in sorted(cyclic):
        issues.add(
            Codes.REVISION_CYCLE,
            "revision participates in a parent cycle; revision rejected",
            scope="revision",
            entity_id=revision_id,
            retained=False,
        )
        accepted.pop(revision_id, None)
    if cyclic:
        return _prune_invalid_revision_graph(accepted, issues)
    return accepted


def parse_rules(
    data: Any,
    vocabulary: Sequence[str],
    config: RuntimeConfig,
    issues: IssueCollector,
) -> tuple[tuple[Revision, ...], tuple[str, ...]]:
    if not isinstance(data, dict):
        issues.add(
            Codes.SCHEMA_INVALID,
            "rules root must be an object",
            scope="revision",
            retained=False,
        )
        return (), ()
    values = data.get("revisions")
    if not isinstance(values, list):
        issues.add(
            Codes.SCHEMA_INVALID,
            "rules.revisions must be a list",
            scope="revision",
            retained=False,
        )
        return (), ()
    vocabulary_set = frozenset(vocabulary)
    parsed: dict[str, Revision] = {}
    for raw in values:
        revision = parse_revision(raw, vocabulary_set, config.allowed_weights, issues)
        if revision is None:
            continue
        if revision.revision_id in parsed:
            issues.add(
                Codes.REVISION_DUPLICATE,
                "duplicate revision_id; later occurrence rejected",
                scope="revision",
                entity_id=revision.revision_id,
                retained=False,
            )
            continue
        parsed[revision.revision_id] = revision
    accepted = _prune_invalid_revision_graph(parsed, issues)
    roots = [item for item in accepted.values() if item.parent_id is None]
    if not roots and accepted:
        issues.add(
            Codes.REVISION_ROOT_INVALID,
            "no root revision remains after validation",
            scope="revision",
            retained=False,
        )
    ordered = tuple(sorted(accepted.values(), key=lambda item: (item.seq, item.revision_id)))

    raw_active = data.get("active_revision_ids")
    if raw_active is None:
        children = {revision.parent_id for revision in accepted.values() if revision.parent_id is not None}
        active = sorted(rid for rid in accepted if rid not in children)
    elif isinstance(raw_active, list):
        active = []
        for revision_id in raw_active:
            if not isinstance(revision_id, str) or revision_id not in accepted:
                issues.add(
                    Codes.ACTIVE_REVISION_MISSING,
                    "active revision does not exist; entry rejected",
                    scope="revision",
                    entity_id=str(revision_id),
                )
                continue
            if revision_id not in active:
                active.append(revision_id)
    else:
        issues.add(
            Codes.ACTIVE_REVISION_INVALID,
            "active_revision_ids must be a list",
            scope="revision",
        )
        active = []
    if not active and accepted:
        children = {revision.parent_id for revision in accepted.values() if revision.parent_id is not None}
        active = sorted(rid for rid in accepted if rid not in children)
        issues.add(
            Codes.ACTIVE_REVISION_INVALID,
            "no valid active revision; leaves were selected",
            scope="revision",
            details={"selected": active},
        )
    return ordered, tuple(active)


def _validate_revision_addition(
    revision: Revision,
    revisions: Mapping[str, Revision],
    issues: IssueCollector,
) -> bool:
    if revision.revision_id in revisions:
        issues.add(
            Codes.REVISION_DUPLICATE,
            "edit tried to add an existing revision",
            scope="edit",
            entity_id=revision.revision_id,
            retained=False,
        )
        return False
    if revision.parent_id is not None and revision.parent_id not in revisions:
        issues.add(
            Codes.PARENT_MISSING,
            "edit revision parent does not exist",
            scope="edit",
            entity_id=revision.revision_id,
            retained=False,
        )
        return False
    candidate = dict(revisions)
    candidate[revision.revision_id] = revision
    pruned = _prune_invalid_revision_graph(candidate, IssueCollector())
    if revision.revision_id not in pruned:
        issues.add(
            Codes.REVISION_CYCLE,
            "edit revision would create a cycle",
            scope="edit",
            entity_id=revision.revision_id,
            retained=False,
        )
        return False
    return True


def apply_edits(
    data: Any,
    *,
    queries: Sequence[Query],
    revisions: Sequence[Revision],
    active_revision_ids: Sequence[str],
    config: RuntimeConfig,
    vocabulary: Sequence[str],
    issues: IssueCollector,
) -> tuple[tuple[Query, ...], tuple[Revision, ...], tuple[str, ...], RuntimeConfig]:
    if data is None:
        return tuple(queries), tuple(revisions), tuple(active_revision_ids), config
    if isinstance(data, dict):
        commands = data.get("commands", [])
    else:
        commands = data
    if not isinstance(commands, list):
        issues.add(
            Codes.SCHEMA_INVALID,
            "edits must be a list or an object with commands",
            scope="edit",
            retained=False,
        )
        return tuple(queries), tuple(revisions), tuple(active_revision_ids), config

    query_map = {query.query_id: query for query in queries}
    revision_map = {revision.revision_id: revision for revision in revisions}
    active = list(active_revision_ids)
    current_config = config
    vocabulary_set = frozenset(vocabulary)

    for index, command in enumerate(commands):
        command_ref = f"cmd[{index}]"
        if not isinstance(command, dict):
            issues.add(
                Codes.EDIT_PAYLOAD_INVALID,
                "edit command must be an object",
                scope="edit",
                entity_id=command_ref,
                retained=False,
            )
            continue
        op = command.get("op")
        if op == "replace_query":
            raw_query = command.get("query")
            if not isinstance(raw_query, dict):
                issues.add(
                    Codes.EDIT_PAYLOAD_INVALID,
                    "replace_query requires a query object",
                    scope="edit",
                    entity_id=command_ref,
                    retained=False,
                )
                continue
            query_id = raw_query.get("query_id")
            if query_id not in query_map:
                issues.add(
                    Codes.EDIT_TARGET_MISSING,
                    "replace_query target does not exist",
                    scope="edit",
                    entity_id=str(query_id),
                    retained=False,
                )
                continue
            other_ids = set(query_map) - {query_id}
            parsed = parse_query(
                raw_query,
                vocabulary_set,
                issues,
                existing_ids=other_ids,
                reject_duplicate=True,
            )
            if parsed is not None:
                query_map[query_id] = parsed
        elif op == "add_revision":
            raw_revision = command.get("revision")
            parsed_revision = parse_revision(
                raw_revision,
                vocabulary_set,
                current_config.allowed_weights,
                issues,
            )
            if parsed_revision is not None and _validate_revision_addition(
                parsed_revision, revision_map, issues
            ):
                revision_map[parsed_revision.revision_id] = parsed_revision
        elif op == "set_active_revisions":
            raw_ids = command.get("revision_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                issues.add(
                    Codes.EDIT_PAYLOAD_INVALID,
                    "set_active_revisions requires a non-empty revision_ids list",
                    scope="edit",
                    entity_id=command_ref,
                    retained=False,
                )
                continue
            next_active: list[str] = []
            valid = True
            for revision_id in raw_ids:
                if not isinstance(revision_id, str) or revision_id not in revision_map:
                    issues.add(
                        Codes.ACTIVE_REVISION_MISSING,
                        "active revision does not exist; command rejected",
                        scope="edit",
                        entity_id=str(revision_id),
                        retained=False,
                    )
                    valid = False
                    break
                if revision_id not in next_active:
                    next_active.append(revision_id)
            if valid:
                active = next_active
        elif op == "set_fixed_point":
            next_bits = command.get("fraction_bits", current_config.fraction_bits)
            next_quantum = command.get("quantum", current_config.quantum)
            valid = True
            if not _is_int(next_bits) or next_bits not in current_config.allowed_fraction_bits:
                issues.add(
                    Codes.FRACTION_BITS_ENUM,
                    "edited fraction_bits is outside the allowed enum",
                    scope="edit",
                    entity_id=command_ref,
                    retained=False,
                    details={"allowed": list(current_config.allowed_fraction_bits)},
                )
                valid = False
            if not _is_int(next_quantum) or next_quantum not in current_config.allowed_quantums:
                issues.add(
                    Codes.QUANTUM_ENUM,
                    "edited quantum is outside the allowed enum",
                    scope="edit",
                    entity_id=command_ref,
                    retained=False,
                    details={"allowed": list(current_config.allowed_quantums)},
                )
                valid = False
            if valid:
                current_config = replace(
                    current_config,
                    fraction_bits=next_bits,
                    quantum=next_quantum,
                )
        elif op == "set_threshold":
            intent_id = command.get("intent_id")
            value = command.get("value")
            if not isinstance(intent_id, str) or not intent_id or not _is_int(value) or value < 0:
                issues.add(
                    Codes.THRESHOLD_INVALID,
                    "set_threshold requires intent_id and a non-negative integer value",
                    scope="edit",
                    entity_id=command_ref,
                    retained=False,
                )
                continue
            thresholds = dict(current_config.thresholds)
            if intent_id == "default":
                current_config = replace(current_config, default_threshold=value)
            else:
                thresholds[intent_id] = value
                current_config = replace(current_config, thresholds=thresholds)
        else:
            issues.add(
                Codes.EDIT_OP_UNKNOWN,
                "unknown edit operation; command rejected",
                scope="edit",
                entity_id=command_ref,
                retained=False,
                details={"op": op},
            )

    ordered_queries = tuple(sorted(query_map.values(), key=lambda item: (item.session_id, item.turn, item.query_id)))
    ordered_revisions = tuple(sorted(revision_map.values(), key=lambda item: (item.seq, item.revision_id)))
    return ordered_queries, ordered_revisions, tuple(active), current_config


def load_inputs(
    *,
    config_path: str | Path,
    vocabulary_path: str | Path,
    queries_path: str | Path,
    rules_path: str | Path,
    edits_path: str | Path | None = None,
) -> tuple[LoadedInputs, IssueCollector]:
    issues = IssueCollector()
    config_data = load_json(config_path, issues, scope="config")
    config = parse_config(config_data, issues)
    vocab_data = load_json(vocabulary_path, issues, scope="vocabulary")
    vocabulary = parse_vocabulary(vocab_data, issues)
    queries_data = load_json(queries_path, issues, scope="query")
    queries = parse_queries(queries_data, vocabulary, issues)
    rules_data = load_json(rules_path, issues, scope="revision")
    revisions, active = parse_rules(rules_data, vocabulary, config, issues)
    if edits_path is not None:
        edit_data = load_json(edits_path, issues, scope="edit")
        queries, revisions, active, config = apply_edits(
            edit_data,
            queries=queries,
            revisions=revisions,
            active_revision_ids=active,
            config=config,
            vocabulary=vocabulary,
            issues=issues,
        )
    return (
        LoadedInputs(
            config=config,
            vocabulary=vocabulary,
            queries=queries,
            revisions=revisions,
            active_revision_ids=active,
        ),
        issues,
    )
