"""Stage-scoped content fingerprints, cache store and reference index."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .constants import CODEC_VERSION, FIXED_ALGORITHM_VERSION, STAGE_ORDER
from .model import ArbitrationConfig, Evidence, Rules
from .numeric import fingerprint, jsonable
from .rules import matched_pairs, relation_references


def _ticket_evidence_fingerprint(evidences: tuple[Evidence, ...]) -> str:
    return fingerprint(
        {
            "schema": "evidence-ticket/v1",
            "rows": [evidence.business_dict() for evidence in sorted(evidences, key=lambda row: row.evidence_seq)],
        }
    )


def _relevant_relation_fingerprint(
    evidences: tuple[Evidence, ...],
    rules: Rules,
) -> tuple[str, list[str]]:
    relevant = [relation for relation in rules.relations if matched_pairs(relation, evidences)]
    relevant_ids = [relation.relation_id for relation in relevant]
    return fingerprint([relation.as_dict() for relation in relevant]), relevant_ids


def score_cache_key(
    ticket_id: str,
    evidences: tuple[Evidence, ...],
    rules: Rules,
    config: ArbitrationConfig,
) -> str:
    evidence_fp = _ticket_evidence_fingerprint(evidences)
    types = sorted({evidence.evidence_type for evidence in evidences})
    roles = sorted({evidence.source_role for evidence in evidences})
    rule_params_fp = fingerprint(
        {
            "weights": {name: rules.weights[name] for name in types},
            "trusts": {name: rules.trusts[name] for name in roles},
            "half_life": config.decay_half_life,
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_units": config.quantization_units,
        }
    )
    resource_pack_fp = fingerprint(
        {
            "schema": "score-resource-pack/v1",
            "types": types,
            "roles": roles,
            "evidence_fp": evidence_fp,
        }
    )
    return fingerprint(
        {
            "algorithm": FIXED_ALGORITHM_VERSION,
            "stage": "score",
            "ticket_id": ticket_id,
            "upstream_fp": evidence_fp,
            "rule_params_fp": rule_params_fp,
            "resource_pack_fp": resource_pack_fp,
        }
    )


def relations_cache_key(
    ticket_id: str,
    evidences: tuple[Evidence, ...],
    score_result: dict,
    upstream_cache_key: str,
    rules: Rules,
    config: ArbitrationConfig,
) -> str:
    score_content_fp = score_result["score_fingerprint"]
    upstream_fp = fingerprint({"cache_key": upstream_cache_key, "content_fp": score_content_fp})
    relation_fp, relevant_ids = _relevant_relation_fingerprint(evidences, rules)
    rule_params_fp = fingerprint(
        {
            "relevant_relations": relation_fp,
            "relevant_relation_ids": relevant_ids,
            "exclusion_strength": config.exclusion_strength,
            "support_bonus": config.support_bonus,
            "fusion": config.fusion,
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_units": config.quantization_units,
        }
    )
    resource_pack_fp = fingerprint(
        {
            "schema": "relation-resource-pack/v1",
            "upstream_fp": upstream_fp,
            "relevant_relation_ids": relevant_ids,
        }
    )
    return fingerprint(
        {
            "algorithm": FIXED_ALGORITHM_VERSION,
            "stage": "relations",
            "ticket_id": ticket_id,
            "upstream_fp": upstream_fp,
            "rule_params_fp": rule_params_fp,
            "resource_pack_fp": resource_pack_fp,
        }
    )


def decision_cache_key(
    ticket_id: str,
    relation_result: dict,
    upstream_cache_key: str,
    rules: Rules,
    config: ArbitrationConfig,
) -> str:
    relation_content_fp = relation_result["relation_fingerprint"]
    upstream_fp = fingerprint({"cache_key": upstream_cache_key, "content_fp": relation_content_fp})
    rule_params_fp = fingerprint(
        {
            "decision_threshold": config.decision_threshold,
            "fusion": config.fusion,
            "threshold_equality_upgrades": config.threshold_equality_upgrades,
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_units": config.quantization_units,
            "priority_bands": [band.as_dict() for band in rules.priority_bands],
        }
    )
    resource_pack_fp = fingerprint(
        {
            "schema": "decision-resource-pack/v1",
            "upstream_fp": upstream_fp,
            "reason_schema": "reason-codes/v1",
        }
    )
    return fingerprint(
        {
            "algorithm": FIXED_ALGORITHM_VERSION,
            "stage": "decision",
            "ticket_id": ticket_id,
            "upstream_fp": upstream_fp,
            "rule_params_fp": rule_params_fp,
            "resource_pack_fp": resource_pack_fp,
        }
    )


def convert_cache_key(
    ticket_id: str,
    decision: dict,
    upstream_cache_key: str,
    config: ArbitrationConfig,
) -> str:
    decision_content_fp = decision["decision_fingerprint"]
    upstream_fp = fingerprint({"cache_key": upstream_cache_key, "content_fp": decision_content_fp})
    rule_params_fp = fingerprint(
        {
            "target_version": config.target_version,
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_units": config.quantization_units,
        }
    )
    resource_pack_fp = fingerprint(
        {
            "schema": "exchange-resource-pack/v1",
            "codec_version": CODEC_VERSION,
            "upstream_fp": upstream_fp,
            "target_version": config.target_version,
        }
    )
    return fingerprint(
        {
            "algorithm": CODEC_VERSION,
            "stage": "convert",
            "ticket_id": ticket_id,
            "upstream_fp": upstream_fp,
            "rule_params_fp": rule_params_fp,
            "resource_pack_fp": resource_pack_fp,
        }
    )


def build_reference_index(
    evidence_by_ticket: dict[str, tuple[Evidence, ...]], rules: Rules
) -> dict[str, Any]:
    types: dict[str, list[str]] = {name: [] for name in sorted(rules.weights)}
    roles: dict[str, list[str]] = {name: [] for name in sorted(rules.trusts)}
    for ticket_id in sorted(evidence_by_ticket):
        for evidence in evidence_by_ticket[ticket_id]:
            types.setdefault(evidence.evidence_type, []).append(ticket_id)
            roles.setdefault(evidence.source_role, []).append(ticket_id)
    types = {name: sorted(set(ticket_ids)) for name, ticket_ids in sorted(types.items())}
    roles = {name: sorted(set(ticket_ids)) for name, ticket_ids in sorted(roles.items())}
    relations = relation_references(rules, evidence_by_ticket)
    return {
        "tickets": sorted(evidence_by_ticket),
        "evidence_types": types,
        "roles": roles,
        "relations": relations,
    }


class CacheStore:
    """JSON-serializable persisted stage cache keyed by ticket id."""

    def __init__(self, stages: dict[str, dict[str, dict]] | None = None):
        self.stages = deepcopy(stages or {stage: {} for stage in STAGE_ORDER})
        for stage in STAGE_ORDER:
            self.stages.setdefault(stage, {})

    @classmethod
    def from_dict(cls, raw: Any) -> "CacheStore":
        if not isinstance(raw, dict):
            return cls()
        stages = raw.get("stages", raw)
        if not isinstance(stages, dict):
            return cls()
        cleaned: dict[str, dict[str, dict]] = {}
        for stage in STAGE_ORDER:
            stage_data = stages.get(stage, {})
            cleaned[stage] = stage_data if isinstance(stage_data, dict) else {}
        return cls(cleaned)

    def as_dict(self) -> dict:
        return {
            "schema": "stage-cache/v1",
            "stages": {
                stage: {ticket_id: self.stages[stage][ticket_id] for ticket_id in sorted(self.stages[stage])}
                for stage in STAGE_ORDER
            },
        }

    def lookup(self, stage: str, ticket_id: str, key: str) -> dict | None:
        entry = self.stages.get(stage, {}).get(ticket_id)
        if isinstance(entry, dict) and entry.get("key") == key and isinstance(entry.get("value"), dict):
            return deepcopy(entry["value"])
        return None

    def put(self, stage: str, ticket_id: str, key: str, value: dict) -> None:
        self.stages.setdefault(stage, {})[ticket_id] = {"key": key, "value": jsonable(value)}

    def entry(self, stage: str, ticket_id: str) -> dict | None:
        entry = self.stages.get(stage, {}).get(ticket_id)
        return deepcopy(entry) if isinstance(entry, dict) else None

    def prune(self, ticket_ids: set[str]) -> None:
        for stage in STAGE_ORDER:
            self.stages[stage] = {
                ticket_id: entry
                for ticket_id, entry in self.stages[stage].items()
                if ticket_id in ticket_ids
            }

    def stage_version(self, stage: str) -> str:
        data = self.stages.get(stage, {})
        return fingerprint({ticket_id: data[ticket_id] for ticket_id in sorted(data)})

    def counts(self) -> dict[str, int]:
        return {stage: len(self.stages.get(stage, {})) for stage in STAGE_ORDER}
