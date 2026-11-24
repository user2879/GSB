"""Strict loading and validation for evidence, rules, config and injections."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .canonical import decimal_text, parse_decimal, sha256_obj
from .codes import (
    BusinessFault,
    E_CONFIG_BATCH_BUDGET,
    E_CONFIG_CONCURRENCY,
    E_CONFIG_FIXED_WIDTH,
    E_CONFIG_JSON,
    E_CONFIG_NUMERIC,
    E_CONFIG_QUANTIZATION,
    E_CONFIG_SCHEMA,
    E_CONFIG_SNAPSHOT_INTERVAL,
    E_CONFIG_WORKORDER_QUOTA,
    E_CONFIDENCE_RANGE,
    E_DUPLICATE_EVIDENCE_KEY,
    E_EVENT_TIME_INVALID,
    E_EVIDENCE_JSON,
    E_EVIDENCE_SCHEMA,
    E_EVIDENCE_SEQ,
    E_HALF_LIFE_INVALID,
    E_HALF_LIFE_ZERO,
    E_INJECTION_SCHEMA,
    E_INPUT_IO,
    E_RELATION_DUPLICATE_ID,
    E_RELATION_SCHEMA,
    E_RELATION_UNKNOWN_TYPE,
    E_ROLE_RANGE,
    E_RULE_JSON,
    E_RULE_SCHEMA,
    E_TERMS_EMPTY,
    E_THRESHOLD_INVALID,
    E_UNKNOWN_EVIDENCE_TYPE,
    E_UNKNOWN_SOURCE_ROLE,
    E_WEIGHT_RANGE,
    E_WORKORDER_ID,
    failure_sort_key,
)
from .models import (
    ArbitrationConfig,
    Evidence,
    Injection,
    LoadedInputs,
    Relation,
    RulePackage,
    Threshold,
)


ALLOWED_DECIMAL_WIDTHS = frozenset({2, 3, 4, 6})
ALLOWED_QUANTIZATION_STEPS = frozenset({1, 2, 5, 10})
ALLOWED_FUSION_MODES = frozenset({"SUM", "MEAN"})
ALLOWED_INJECTION_STAGES = frozenset({"load", "decay", "relations", "aggregation"})
ALLOWED_INJECTION_FAULTS = frozenset(
    {
        "unknown_evidence_type",
        "confidence_out_of_range",
        "negative_timestamp",
        "zero_half_life",
        "relation_unknown_type",
    }
)
INJECTION_FAULT_STAGE = {
    "unknown_evidence_type": "load",
    "confidence_out_of_range": "load",
    "negative_timestamp": "load",
    "zero_half_life": "decay",
    "relation_unknown_type": "relations",
}


def _read_json(path: Path, code: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessFault(code, f"cannot read {path.name}: {exc.__class__.__name__}", "global") from exc
    try:
        return json.loads(text, parse_float=Decimal, parse_int=int)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BusinessFault(code, f"invalid JSON in {path.name}", "global") from exc


def _failure(
    code: str,
    message: str,
    stage: str,
    *,
    workorder_id: str | None = None,
    evidence_seq: int | None = None,
    relation_id: str | None = None,
    injection_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return BusinessFault(
        code=code,
        message=message,
        stage=stage,
        workorder_id=workorder_id,
        evidence_seq=evidence_seq,
        relation_id=relation_id,
        injection_id=injection_id,
        details=details or {},
    ).as_failure()


def _value(raw: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def load_config_file(path: Path) -> ArbitrationConfig:
    raw = _read_json(path, E_CONFIG_JSON)
    if not isinstance(raw, dict):
        raise BusinessFault(E_CONFIG_SCHEMA, "config root must be an object", "config")

    concurrency = _value(raw, ("concurrency", "thread_pool_concurrency"))
    if not _is_positive_int(concurrency):
        raise BusinessFault(E_CONFIG_CONCURRENCY, "concurrency must be a positive integer", "config")

    budget = _value(raw, ("batch_evidence_budget", "evidence_budget"))
    if not _is_positive_int(budget):
        raise BusinessFault(E_CONFIG_BATCH_BUDGET, "batch_evidence_budget must be a positive integer", "config")

    quota = _value(raw, ("workorder_evidence_quota", "workorder_quota"))
    if not _is_positive_int(quota):
        raise BusinessFault(E_CONFIG_WORKORDER_QUOTA, "workorder_evidence_quota must be a positive integer", "config")

    width = _value(raw, ("fixed_decimal_width", "decimal_width"))
    if width not in ALLOWED_DECIMAL_WIDTHS or isinstance(width, bool):
        raise BusinessFault(
            E_CONFIG_FIXED_WIDTH,
            f"fixed_decimal_width must be one of {sorted(ALLOWED_DECIMAL_WIDTHS)}",
            "config",
        )

    quant_raw = _value(raw, ("quantization_level", "quantization_step"))
    quant_step: int | None = None
    if isinstance(quant_raw, str):
        label = quant_raw.strip().upper()
        if label.startswith("Q"):
            label = label[1:]
        if label.isdigit():
            quant_step = int(label)
    elif isinstance(quant_raw, int) and not isinstance(quant_raw, bool):
        quant_step = quant_raw
    if quant_step not in ALLOWED_QUANTIZATION_STEPS:
        raise BusinessFault(
            E_CONFIG_QUANTIZATION,
            f"quantization_level must be one of Q1, Q2, Q5, Q10 or 1, 2, 5, 10",
            "config",
        )
    if quant_step > 10**width:
        raise BusinessFault(E_CONFIG_QUANTIZATION, "quantization step exceeds fixed-point scale", "config")

    interval = _value(raw, ("snapshot_commit_interval", "snapshot_interval"))
    if not _is_positive_int(interval):
        raise BusinessFault(
            E_CONFIG_SNAPSHOT_INTERVAL,
            "snapshot_commit_interval must be a positive integer",
            "config",
        )

    try:
        exclusion = parse_decimal(_value(raw, ("exclusion_strength", "mutual_exclusion_strength"), "0.10"), field="exclusion_strength")
        support = parse_decimal(_value(raw, ("support_strength", "support_bonus_strength"), "0.05"), field="support_strength")
        zero = parse_decimal(_value(raw, ("zero_lower_bound", "score_floor"), "0"), field="zero_lower_bound")
    except ValueError as exc:
        raise BusinessFault(E_CONFIG_NUMERIC, str(exc), "config") from exc
    if exclusion < 0 or support < 0 or zero < 0:
        raise BusinessFault(E_CONFIG_NUMERIC, "strengths and zero_lower_bound must be non-negative", "config")

    fusion = str(_value(raw, ("fusion_mode", "aggregation_mode"), "SUM")).strip().upper()
    if fusion not in ALLOWED_FUSION_MODES:
        raise BusinessFault(E_CONFIG_SCHEMA, "fusion_mode must be SUM or MEAN", "config")

    normalized = {
        "schema_version": int(raw.get("schema_version", 1)),
        "concurrency": concurrency,
        "batch_evidence_budget": budget,
        "workorder_evidence_quota": quota,
        "fixed_decimal_width": width,
        "quantization_step": quant_step,
        "snapshot_commit_interval": interval,
        "exclusion_strength": decimal_text(exclusion),
        "support_strength": decimal_text(support),
        "zero_lower_bound": decimal_text(zero),
        "fusion_mode": fusion,
    }
    return ArbitrationConfig(
        concurrency=concurrency,
        batch_evidence_budget=budget,
        workorder_evidence_quota=quota,
        fixed_decimal_width=width,
        quantization_step=quant_step,
        snapshot_commit_interval=interval,
        exclusion_strength=exclusion,
        support_strength=support,
        zero_lower_bound=zero,
        fusion_mode=fusion,
        normalized=normalized,
        config_hash=sha256_obj(normalized),
    )


def _parse_relations(raw_relations: Any, source_name: str) -> list[Relation]:
    if raw_relations is None:
        return []
    if not isinstance(raw_relations, list):
        raise BusinessFault(E_RULE_SCHEMA, f"{source_name} must be an array", "rules")
    result: list[Relation] = []
    for index, item in enumerate(raw_relations):
        if not isinstance(item, dict):
            raise BusinessFault(E_RELATION_SCHEMA, f"{source_name}[{index}] must be an object", "rules")
        relation_id = item.get("id", item.get("relation_id"))
        if not isinstance(relation_id, str) or not relation_id.strip():
            raise BusinessFault(E_RELATION_SCHEMA, f"{source_name}[{index}] has no stable id", "rules")
        left = item.get("left_type")
        right = item.get("right_type")
        if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
            raise BusinessFault(
                E_RELATION_SCHEMA,
                f"relation {relation_id} requires non-empty left_type and right_type",
                "rules",
                relation_id=relation_id,
            )
        strength_raw = item.get("strength")
        try:
            strength = parse_decimal(strength_raw, field="relation strength") if strength_raw is not None else None
        except ValueError as exc:
            raise BusinessFault(E_RELATION_SCHEMA, str(exc), "rules", relation_id=relation_id) from exc
        if strength is not None and strength < 0:
            raise BusinessFault(E_RELATION_SCHEMA, "relation strength must be non-negative", "rules", relation_id=relation_id)
        explicit_kind = item.get("kind")
        if explicit_kind is None:
            kind = "mutual_exclusion" if "mutual" in source_name.lower() else "support"
        else:
            kind = str(explicit_kind).strip().lower()
            if kind in {"mutex", "exclusion", "mutual-exclusion"}:
                kind = "mutual_exclusion"
            elif kind in {"bonus", "support_bonus"}:
                kind = "support"
        if kind not in {"mutual_exclusion", "support"}:
            raise BusinessFault(E_RELATION_SCHEMA, f"relation {relation_id} has invalid kind", "rules", relation_id=relation_id)
        result.append(
            Relation(
                relation_id=relation_id,
                kind=kind,
                left_type=left,
                right_type=right,
                strength=strength,
            )
        )
    return result


def _parse_thresholds(raw_thresholds: Any) -> list[Threshold]:
    if isinstance(raw_thresholds, dict):
        items = [
            {"priority": str(priority), "min_score": score}
            for priority, score in raw_thresholds.items()
        ]
    elif isinstance(raw_thresholds, list):
        items = raw_thresholds
    else:
        raise BusinessFault(E_THRESHOLD_INVALID, "priority_thresholds must be an array or object", "rules")
    if not items:
        raise BusinessFault(E_THRESHOLD_INVALID, "priority_thresholds must not be empty", "rules")
    result: list[Threshold] = []
    seen_priorities: set[str] = set()
    seen_scores: set[Decimal] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise BusinessFault(E_THRESHOLD_INVALID, f"threshold[{index}] must be an object", "rules")
        priority = item.get("priority")
        if not isinstance(priority, str) or not priority.strip():
            raise BusinessFault(E_THRESHOLD_INVALID, f"threshold[{index}] has no priority", "rules")
        try:
            minimum = parse_decimal(item.get("min_score"), field="min_score")
        except ValueError as exc:
            raise BusinessFault(E_THRESHOLD_INVALID, str(exc), "rules") from exc
        if priority in seen_priorities:
            raise BusinessFault(E_THRESHOLD_INVALID, "priority values must be unique", "rules")
        if minimum in seen_scores:
            raise BusinessFault(E_THRESHOLD_INVALID, "threshold values must be unique", "rules")
        if minimum < 0:
            raise BusinessFault(E_THRESHOLD_INVALID, "thresholds must be non-negative", "rules")
        seen_priorities.add(priority)
        seen_scores.add(minimum)
        result.append(Threshold(priority=priority, min_score=minimum))
    result.sort(key=lambda threshold: (threshold.min_score, threshold.priority))
    if result[0].min_score > 0:
        raise BusinessFault(E_THRESHOLD_INVALID, "lowest threshold must cover score zero", "rules")
    return result


def load_rules_file(path: Path) -> RulePackage:
    raw = _read_json(path, E_RULE_JSON)
    if not isinstance(raw, dict):
        raise BusinessFault(E_RULE_SCHEMA, "rules root must be an object", "rules")

    raw_weights = raw.get("evidence_type_weights", raw.get("weights"))
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise BusinessFault(E_RULE_SCHEMA, "evidence_type_weights must be a non-empty object", "rules")
    weights: dict[str, Decimal] = {}
    for evidence_type, value in raw_weights.items():
        if not isinstance(evidence_type, str) or not evidence_type:
            raise BusinessFault(E_RULE_SCHEMA, "evidence type keys must be non-empty strings", "rules")
        try:
            weight = parse_decimal(value, field=f"weight[{evidence_type}]")
        except ValueError as exc:
            raise BusinessFault(E_RULE_SCHEMA, str(exc), "rules") from exc
        if weight < 0:
            raise BusinessFault(E_WEIGHT_RANGE, f"weight for {evidence_type} must be non-negative", "rules")
        weights[evidence_type] = weight

    raw_roles = raw.get("source_role_credibility", raw.get("role_credibility"))
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise BusinessFault(E_RULE_SCHEMA, "source_role_credibility must be a non-empty object", "rules")
    roles: dict[str, Decimal] = {}
    for role, value in raw_roles.items():
        if not isinstance(role, str) or not role:
            raise BusinessFault(E_RULE_SCHEMA, "role keys must be non-empty strings", "rules")
        try:
            credibility = parse_decimal(value, field=f"credibility[{role}]")
        except ValueError as exc:
            raise BusinessFault(E_RULE_SCHEMA, str(exc), "rules") from exc
        if credibility < 0 or credibility > 1:
            raise BusinessFault(E_ROLE_RANGE, f"credibility for {role} must be in [0, 1]", "rules")
        roles[role] = credibility

    relations = _parse_relations(raw.get("mutual_exclusions"), "mutual_exclusions")
    relations.extend(_parse_relations(raw.get("supports"), "supports"))
    relations.extend(_parse_relations(raw.get("relations"), "relations"))
    relation_ids: set[str] = set()
    for relation in relations:
        if relation.relation_id in relation_ids:
            raise BusinessFault(
                E_RELATION_DUPLICATE_ID,
                f"duplicate relation id {relation.relation_id}",
                "rules",
                relation_id=relation.relation_id,
            )
        relation_ids.add(relation.relation_id)
        for endpoint, endpoint_type in (("left", relation.left_type), ("right", relation.right_type)):
            if endpoint_type not in weights:
                raise BusinessFault(
                    E_RELATION_UNKNOWN_TYPE,
                    f"relation {relation.relation_id} {endpoint}_type is not in weight table",
                    "rules",
                    relation_id=relation.relation_id,
                    details={"endpoint_type": endpoint_type},
                )
    relations.sort(key=lambda relation: relation.relation_id)

    half_life = raw.get("half_life_ms")
    if isinstance(half_life, bool) or not isinstance(half_life, int):
        raise BusinessFault(E_HALF_LIFE_INVALID, "half_life_ms must be an integer", "rules")
    if half_life == 0:
        raise BusinessFault(E_HALF_LIFE_ZERO, "half_life_ms must be greater than zero", "rules")
    if half_life < 0:
        raise BusinessFault(E_HALF_LIFE_INVALID, "half_life_ms must not be negative", "rules")

    thresholds = _parse_thresholds(raw.get("priority_thresholds", raw.get("thresholds")))

    normalized = {
        "schema_version": int(raw.get("schema_version", 1)),
        "evidence_type_weights": {key: decimal_text(weights[key]) for key in sorted(weights)},
        "source_role_credibility": {key: decimal_text(roles[key]) for key in sorted(roles)},
        "relations": [relation.canonical() for relation in relations],
        "half_life_ms": half_life,
        "priority_thresholds": [threshold.canonical() for threshold in thresholds],
        "boundary_policy": "INCLUSIVE_UPPER",
        "relation_application_order": "EXCLUSIONS_THEN_SUPPORTS",
    }
    return RulePackage(
        weights=weights,
        role_credibility=roles,
        relations=tuple(relations),
        half_life_ms=half_life,
        thresholds=tuple(thresholds),
        normalized=normalized,
        fingerprint=sha256_obj(normalized),
    )


def _line_failure(
    code: str,
    message: str,
    line_no: int,
    *,
    workorder_id: str | None = None,
    evidence_seq: int | None = None,
) -> dict[str, Any]:
    return _failure(
        code,
        message,
        "load",
        workorder_id=workorder_id,
        evidence_seq=evidence_seq,
        details={"line_no": line_no},
    )


def load_evidence_file(path: Path) -> tuple[list[Evidence], list[dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BusinessFault(E_INPUT_IO, f"cannot read {path.name}: {exc.__class__.__name__}", "global") from exc

    evidence: list[Evidence] = []
    failures: list[dict[str, Any]] = []
    seen_keys: dict[tuple[str, int], int] = {}
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            failures.append(_line_failure(E_EVIDENCE_JSON, "blank evidence line is not allowed", line_no))
            continue
        try:
            raw = json.loads(line, parse_float=Decimal, parse_int=int)
        except (json.JSONDecodeError, ValueError):
            failures.append(_line_failure(E_EVIDENCE_JSON, "invalid evidence JSON", line_no))
            continue
        if not isinstance(raw, dict):
            failures.append(_line_failure(E_EVIDENCE_SCHEMA, "evidence record must be an object", line_no))
            continue
        workorder_id = raw.get("workorder_id")
        if not isinstance(workorder_id, str) or not workorder_id.strip():
            failures.append(_line_failure(E_WORKORDER_ID, "workorder_id must be a non-empty string", line_no))
            continue
        evidence_seq = raw.get("evidence_seq")
        if not _is_positive_int(evidence_seq):
            failures.append(
                _line_failure(
                    E_EVIDENCE_SEQ,
                    "evidence_seq must be a positive integer",
                    line_no,
                    workorder_id=workorder_id,
                )
            )
            continue
        evidence_type = raw.get("evidence_type")
        source_role = raw.get("source_role")
        if not isinstance(evidence_type, str) or not evidence_type:
            failures.append(
                _line_failure(
                    E_EVIDENCE_SCHEMA,
                    "evidence_type must be a non-empty string",
                    line_no,
                    workorder_id=workorder_id,
                    evidence_seq=evidence_seq,
                )
            )
            continue
        if not isinstance(source_role, str) or not source_role:
            failures.append(
                _line_failure(
                    E_EVIDENCE_SCHEMA,
                    "source_role must be a non-empty string",
                    line_no,
                    workorder_id=workorder_id,
                    evidence_seq=evidence_seq,
                )
            )
            continue
        if "event_ms" not in raw:
            failures.append(
                _line_failure(
                    E_EVENT_TIME_INVALID,
                    "event_ms is required",
                    line_no,
                    workorder_id=workorder_id,
                    evidence_seq=evidence_seq,
                )
            )
            continue
        if "terms" not in raw:
            failures.append(
                _line_failure(
                    E_TERMS_EMPTY,
                    "terms is required",
                    line_no,
                    workorder_id=workorder_id,
                    evidence_seq=evidence_seq,
                )
            )
            continue
        try:
            confidence = parse_decimal(raw.get("confidence"), field="confidence")
        except ValueError:
            failures.append(
                _line_failure(
                    E_CONFIDENCE_RANGE,
                    "confidence must be numeric",
                    line_no,
                    workorder_id=workorder_id,
                    evidence_seq=evidence_seq,
                )
            )
            continue

        key = (workorder_id, evidence_seq)
        if key in seen_keys:
            failures.append(
                _failure(
                    E_DUPLICATE_EVIDENCE_KEY,
                    "workorder_id/evidence_seq combination must be unique",
                    "load",
                    workorder_id=workorder_id,
                    evidence_seq=evidence_seq,
                    details={"first_line_no": seen_keys[key], "duplicate_line_no": line_no},
                )
            )
        else:
            seen_keys[key] = line_no
        evidence.append(
            Evidence(
                workorder_id=workorder_id,
                evidence_seq=evidence_seq,
                evidence_type=evidence_type,
                source_role=source_role,
                event_ms=raw.get("event_ms"),
                terms=raw.get("terms"),
                confidence=confidence,
                confidence_text=decimal_text(confidence),
                line_no=line_no,
            )
        )

    evidence.sort(key=lambda item: (item.workorder_id, item.evidence_seq, item.line_no))
    failures.sort(key=failure_sort_key)
    return evidence, failures


def validate_workorder_evidence(
    records: list[Evidence],
    rules: RulePackage,
    *,
    injection_by_seq: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Perform semantic validation after any load-stage fault injection."""

    injection_by_seq = injection_by_seq or {}
    failures: list[dict[str, Any]] = []
    for record in records:
        injection_id = injection_by_seq.get(record.evidence_seq)
        base = {
            "workorder_id": record.workorder_id,
            "evidence_seq": record.evidence_seq,
            "injection_id": injection_id,
        }
        if isinstance(record.event_ms, bool) or not isinstance(record.event_ms, int) or record.event_ms < 0:
            failures.append(
                _failure(
                    E_EVENT_TIME_INVALID,
                    "event_ms must be a non-negative integer",
                    "load",
                    **base,
                )
            )
        if record.confidence < 0 or record.confidence > 1:
            failures.append(
                _failure(
                    E_CONFIDENCE_RANGE,
                    "confidence must be in the declared [0, 1] interval",
                    "load",
                    **base,
                )
            )
        if not isinstance(record.terms, list) or not record.terms or not all(
            isinstance(term, str) and bool(term.strip()) for term in record.terms
        ):
            failures.append(
                _failure(
                    E_TERMS_EMPTY,
                    "terms must be a non-empty array of non-empty strings",
                    "load",
                    **base,
                )
            )
        if record.evidence_type not in rules.weights:
            failures.append(
                _failure(
                    E_UNKNOWN_EVIDENCE_TYPE,
                    f"evidence_type {record.evidence_type} is not in weight table",
                    "load",
                    **base,
                )
            )
        if record.source_role not in rules.role_credibility:
            failures.append(
                _failure(
                    E_UNKNOWN_SOURCE_ROLE,
                    f"source_role {record.source_role} is not in credibility table",
                    "load",
                    **base,
                )
            )
    return failures


def load_inputs(data_dir: Path) -> LoadedInputs:
    config = load_config_file(data_dir / "config.json")
    rules = load_rules_file(data_dir / "rules.json")
    evidence, failures = load_evidence_file(data_dir / "evidence.jsonl")
    return LoadedInputs(
        config=config,
        rules=rules,
        evidence=tuple(evidence),
        evidence_failures=tuple(failures),
        input_version=load_input_version(data_dir),
    )


def load_input_version(data_dir: Path) -> int:
    path = data_dir / "input_meta.json"
    if not path.exists():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessFault(E_INPUT_IO, "input_meta.json is unreadable", "global") from exc
    version = raw.get("version") if isinstance(raw, dict) else None
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise BusinessFault(E_INPUT_IO, "input_meta.json version must be a non-negative integer", "global")
    return version


def load_injections(path: Path | None) -> list[Injection]:
    if path is None:
        return []
    raw = _read_json(path, E_INJECTION_SCHEMA)
    if not isinstance(raw, dict):
        raise BusinessFault(E_INJECTION_SCHEMA, "injection root must be an object", "global")
    items = raw.get("injections", [])
    if not isinstance(items, list):
        raise BusinessFault(E_INJECTION_SCHEMA, "injections must be an array", "global")
    result: list[Injection] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise BusinessFault(E_INJECTION_SCHEMA, f"injections[{index}] must be an object", "global")
        injection_id = item.get("id", item.get("injection_id"))
        workorder_id = item.get("workorder_id")
        stage = item.get("stage")
        fault = item.get("fault")
        if not isinstance(injection_id, str) or not injection_id:
            raise BusinessFault(E_INJECTION_SCHEMA, f"injections[{index}] has no id", "global")
        if injection_id in seen_ids:
            raise BusinessFault(E_INJECTION_SCHEMA, f"duplicate injection id {injection_id}", "global")
        if not isinstance(workorder_id, str) or not workorder_id:
            raise BusinessFault(E_INJECTION_SCHEMA, f"injection {injection_id} has no workorder_id", "global")
        if stage not in ALLOWED_INJECTION_STAGES:
            raise BusinessFault(E_INJECTION_SCHEMA, f"injection {injection_id} has invalid stage", "global")
        if fault not in ALLOWED_INJECTION_FAULTS:
            raise BusinessFault(E_INJECTION_SCHEMA, f"injection {injection_id} has invalid fault", "global")
        if stage != INJECTION_FAULT_STAGE[fault]:
            raise BusinessFault(
                E_INJECTION_SCHEMA,
                f"injection {injection_id} fault {fault} must target stage {INJECTION_FAULT_STAGE[fault]}",
                "global",
            )
        evidence_seq = item.get("evidence_seq")
        if evidence_seq is not None and not _is_positive_int(evidence_seq):
            raise BusinessFault(E_INJECTION_SCHEMA, f"injection {injection_id} has invalid evidence_seq", "global")
        seen_ids.add(injection_id)
        result.append(
            Injection(
                injection_id=injection_id,
                workorder_id=workorder_id,
                stage=stage,
                fault=fault,
                evidence_seq=evidence_seq,
            )
        )
    result.sort(key=lambda injection: injection.injection_id)
    return result


def group_evidence(evidence: Iterable[Evidence]) -> dict[str, list[Evidence]]:
    result: dict[str, list[Evidence]] = {}
    for record in evidence:
        result.setdefault(record.workorder_id, []).append(record)
    for records in result.values():
        records.sort(key=lambda item: (item.evidence_seq, item.line_no))
    return result
