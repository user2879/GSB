"""Rule-pack parsing, validation and selector matching."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .constants import ERROR_EVIDENCE_TYPE_UNKNOWN, ERROR_JSON_INVALID, ERROR_ROLE_UNKNOWN
from .errors import RuleValidationError
from .model import Evidence, PriorityBand, Relation, Rules, Selector
from .numeric import ONE, ZERO, as_decimal, fingerprint


def _selector(raw: Any, *, relation_id: str, side: str) -> Selector:
    if not isinstance(raw, dict):
        raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} {side} must be an object")
    evidence_type = raw.get("evidence_type") or raw.get("type")
    if not isinstance(evidence_type, str) or not evidence_type:
        raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} {side} lacks evidence_type")
    roles_raw = raw.get("source_roles", raw.get("roles", ()))
    if isinstance(roles_raw, str):
        roles = (roles_raw,)
    elif roles_raw is None:
        roles = ()
    elif isinstance(roles_raw, (list, tuple)) and all(isinstance(role, str) and role for role in roles_raw):
        roles = tuple(sorted(set(roles_raw)))
    else:
        raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} {side} source_roles invalid")
    return Selector(evidence_type=evidence_type, source_roles=roles)


def parse_rules(raw: Any) -> Rules:
    if not isinstance(raw, dict):
        raise RuleValidationError(ERROR_JSON_INVALID, "rules package must be a JSON object")

    weights_raw = raw.get("evidence_type_weights")
    trusts_raw = raw.get("role_trust")
    if not isinstance(weights_raw, dict) or not weights_raw:
        raise RuleValidationError(ERROR_JSON_INVALID, "evidence_type_weights must be a non-empty object")
    if not isinstance(trusts_raw, dict) or not trusts_raw:
        raise RuleValidationError(ERROR_JSON_INVALID, "role_trust must be a non-empty object")

    weights: dict[str, Decimal] = {}
    trusts: dict[str, Decimal] = {}
    try:
        for key, value in weights_raw.items():
            if not isinstance(key, str) or not key:
                raise ValueError("evidence type names must be non-empty strings")
            dec = as_decimal(value, field=f"evidence_type_weights.{key}")
            if dec < ZERO or dec > ONE:
                raise ValueError(f"evidence_type_weights.{key} must be in [0, 1]")
            weights[key] = dec
        for key, value in trusts_raw.items():
            if not isinstance(key, str) or not key:
                raise ValueError("role names must be non-empty strings")
            dec = as_decimal(value, field=f"role_trust.{key}")
            if dec < ZERO or dec > ONE:
                raise ValueError(f"role_trust.{key} must be in [0, 1]")
            trusts[key] = dec
    except ValueError as exc:
        raise RuleValidationError(ERROR_JSON_INVALID, str(exc)) from exc

    relations_raw = raw.get("relations", [])
    if not isinstance(relations_raw, list):
        raise RuleValidationError(ERROR_JSON_INVALID, "relations must be an array")
    relations: list[Relation] = []
    relation_ids: set[str] = set()
    for item in relations_raw:
        if not isinstance(item, dict):
            raise RuleValidationError(ERROR_JSON_INVALID, "each relation must be an object")
        relation_id = item.get("id") or item.get("relation_id")
        if not isinstance(relation_id, str) or not relation_id:
            raise RuleValidationError(ERROR_JSON_INVALID, "each relation needs a non-empty id")
        if relation_id in relation_ids:
            raise RuleValidationError(ERROR_JSON_INVALID, f"duplicate relation id {relation_id}")
        relation_ids.add(relation_id)
        kind_raw = str(item.get("kind", "")).lower()
        if kind_raw in {"exclusive", "exclusion", "conflict", "mutex"}:
            kind = "exclusive"
        elif kind_raw in {"support", "supports", "boost", "reinforce"}:
            kind = "support"
        else:
            raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} has unknown kind {kind_raw!r}")
        left = _selector(item.get("left"), relation_id=relation_id, side="left")
        right = _selector(item.get("right"), relation_id=relation_id, side="right")
        if left == right:
            raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} is self-reflexive")
        if left.evidence_type not in weights:
            raise RuleValidationError(
                ERROR_EVIDENCE_TYPE_UNKNOWN,
                f"relation {relation_id} references unknown evidence type {left.evidence_type}",
            )
        if right.evidence_type not in weights:
            raise RuleValidationError(
                ERROR_EVIDENCE_TYPE_UNKNOWN,
                f"relation {relation_id} references unknown evidence type {right.evidence_type}",
            )
        for role in left.source_roles + right.source_roles:
            if role not in trusts:
                raise RuleValidationError(
                    ERROR_ROLE_UNKNOWN,
                    f"relation {relation_id} references unknown role {role}",
                )
        bonus: Decimal | None = None
        if kind == "support":
            try:
                bonus = as_decimal(item.get("bonus", raw.get("default_support_bonus", "0")), field="support bonus")
            except ValueError as exc:
                raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} bonus invalid: {exc}") from exc
            if bonus < ZERO:
                raise RuleValidationError(ERROR_JSON_INVALID, f"relation {relation_id} bonus must be non-negative")
        relations.append(Relation(relation_id, kind, left, right, bonus))
    relations.sort(key=lambda relation: relation.relation_id)

    bands_raw = raw.get("priority_bands")
    if not isinstance(bands_raw, list) or not bands_raw:
        raise RuleValidationError(ERROR_JSON_INVALID, "priority_bands must be a non-empty array")
    bands: list[PriorityBand] = []
    levels: set[int] = set()
    names: set[str] = set()
    for item in bands_raw:
        if not isinstance(item, dict):
            raise RuleValidationError(ERROR_JSON_INVALID, "each priority band must be an object")
        level = item.get("level")
        name = item.get("name")
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            raise RuleValidationError(ERROR_JSON_INVALID, "priority band level must be a non-negative integer")
        if not isinstance(name, str) or not name:
            raise RuleValidationError(ERROR_JSON_INVALID, "priority band name must be non-empty")
        if level in levels or name in names:
            raise RuleValidationError(ERROR_JSON_INVALID, "priority band levels and names must be unique")
        levels.add(level)
        names.add(name)
        try:
            min_score = as_decimal(item.get("min_score"), field="priority band min_score")
        except ValueError as exc:
            raise RuleValidationError(ERROR_JSON_INVALID, str(exc)) from exc
        if min_score < ZERO:
            raise RuleValidationError(ERROR_JSON_INVALID, "priority band min_score must be non-negative")
        bands.append(PriorityBand(level=level, name=name, min_score=min_score))
    bands.sort(key=lambda band: (band.min_score, band.level, band.name))
    return Rules(
        schema=str(raw.get("schema", "arbitration-rules/v1")),
        version=str(raw.get("version", "1")),
        weights=weights,
        trusts=trusts,
        relations=tuple(relations),
        priority_bands=tuple(bands),
    )


def rules_fingerprint(rules: Rules) -> str:
    return fingerprint(rules.as_dict())


def matched_pairs(relation: Relation, evidences: tuple[Evidence, ...]) -> list[tuple[Evidence, Evidence]]:
    left = [item for item in evidences if relation.left.matches(item)]
    right = [item for item in evidences if relation.right.matches(item)]
    pairs: list[tuple[Evidence, Evidence]] = []
    for first in left:
        for second in right:
            if first.evidence_seq == second.evidence_seq:
                continue
            pairs.append((first, second))
    pairs.sort(key=lambda pair: (pair[0].evidence_seq, pair[1].evidence_seq))
    return pairs


def relation_references(
    rules: Rules, evidence_by_ticket: dict[str, tuple[Evidence, ...]]
) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}
    for relation in rules.relations:
        refs = [ticket_id for ticket_id, rows in evidence_by_ticket.items() if matched_pairs(relation, rows)]
        references[relation.relation_id] = sorted(refs)
    return references
