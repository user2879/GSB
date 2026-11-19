"""Deterministic multi-node observation merge and arbitration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .observer import FetchBatch, FetchOutcome, Observation
from .spec import Spec


ATTRIBUTION_ORDER = (
    "OBSERVATION_MISSING",
    "OBSERVATION_TIMEOUT",
    "VALUE_DIVERGENCE",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_value(value: int | str) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid effective value")
    if isinstance(value, int):
        return canonical_json({"type": "int", "value": value})
    if isinstance(value, str):
        return canonical_json({"type": "string", "value": value})
    raise ValueError(f"unsupported effective value type: {type(value).__name__}")


def _observation_sort_key(observation: Observation) -> tuple[int, int, str, str]:
    return (
        -observation.observation_seq,
        -observation.timestamp_ms,
        observation.node_id,
        canonical_value(observation.value),
    )


def _strict_equal(left: int | str | None, right: int | str) -> bool:
    if left is None:
        return False
    return type(left) is type(right) and left == right


@dataclass(frozen=True)
class MergeResult:
    decisions: tuple[dict[str, Any], ...]
    comparisons: tuple[dict[str, Any], ...]
    divergence_keys: tuple[str, ...]

    def decisions_by_key(self) -> dict[str, dict[str, Any]]:
        return {decision["key"]: decision for decision in self.decisions}

    def comparisons_by_key_node(self) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (row["key"], row["node_id"]): row
            for row in self.comparisons
        }

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "content_version": 1,
            "decisions": list(self.decisions),
            "comparisons": list(self.comparisons),
            "divergence_keys": list(self.divergence_keys),
        }

    def deterministic_bytes(self) -> bytes:
        return canonical_json(self.deterministic_payload()).encode("utf-8")

    def deterministic_digest(self) -> str:
        return hashlib.sha256(self.deterministic_bytes()).hexdigest()


def _latest_per_node(node_id: str, outcome: FetchOutcome) -> Observation | None:
    if not outcome.observations:
        return None
    records = sorted(outcome.observations, key=_observation_sort_key)
    return records[0]


def _arbitrate(latest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not latest_rows:
        return {
            "winner": None,
            "winner_value": None,
            "winner_basis": "NO_VALID_OBSERVATION",
            "majority_count": 0,
            "tied_values": [],
            "basis_text": "no valid observation exists; no arbitration is possible",
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in latest_rows:
        groups.setdefault(row["_value_token"], []).append(row)
    tokens = sorted(groups)
    if len(tokens) == 1:
        token = tokens[0]
        winner_row = sorted(groups[token], key=_latest_row_sort_key)[0]
        return {
            "winner": winner_row,
            "winner_value": winner_row["value"],
            "winner_basis": "ONLY_VALUE",
            "majority_count": len(groups[token]),
            "tied_values": [winner_row["value"]],
            "basis_text": "all valid nodes report one value; that value wins",
        }

    max_count = max(len(groups[token]) for token in tokens)
    top_tokens = [token for token in tokens if len(groups[token]) == max_count]
    if len(top_tokens) == 1:
        token = top_tokens[0]
        winner_row = sorted(groups[token], key=_latest_row_sort_key)[0]
        return {
            "winner": winner_row,
            "winner_value": winner_row["value"],
            "winner_basis": "MAJORITY",
            "majority_count": max_count,
            "tied_values": [row["value"] for row in sorted(groups[token], key=_latest_row_sort_key)],
            "basis_text": (
                f"majority value wins: {max_count} of {len(latest_rows)} valid nodes"
            ),
        }

    tied_rows = [row for token in top_tokens for row in groups[token]]
    latest_row = sorted(tied_rows, key=_latest_row_sort_key)[0]
    return {
        "winner": latest_row,
        "winner_value": latest_row["value"],
        "winner_basis": "LATEST_TIE_BREAK",
        "majority_count": max_count,
        "tied_values": [
            row["value"] for row in sorted(tied_rows, key=lambda item: item["_value_token"])
        ],
        "basis_text": (
            "value counts are tied; the highest observation sequence, then highest "
            "timestamp, then node id breaks the tie deterministically"
        ),
    }


def _latest_row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        -int(row["observation_seq"]),
        -int(row["timestamp_ms"]),
        str(row["node_id"]),
        str(row["_value_token"]),
    )


def _missing_outcome(spec: Spec, node_id: str, key: str) -> FetchOutcome:
    return FetchOutcome(
        node_id=node_id,
        key=key,
        status="NOT_FETCHED",
        error_reason="no request outcome was supplied to the merger",
    )


def merge_observations(spec: Spec, batch: FetchBatch) -> MergeResult:
    outcomes = {(item.node_id, item.key): item for item in batch.outcomes}
    decisions: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    divergence_keys: list[str] = []
    node_ids = list(spec.sorted_node_ids)
    total_nodes = len(node_ids)

    for key in spec.sorted_keys:
        key_spec = spec.keys[key]
        per_node: list[dict[str, Any]] = []
        outcome_rows: dict[str, FetchOutcome] = {}
        missing_nodes: list[dict[str, Any]] = []
        timeout_nodes: list[dict[str, Any]] = []

        for node_id in node_ids:
            outcome = outcomes.get((node_id, key)) or _missing_outcome(spec, node_id, key)
            outcome_rows[node_id] = outcome
            latest = _latest_per_node(node_id, outcome)
            if latest is None:
                detail = {
                    "node_id": node_id,
                    "status": outcome.status,
                    "reason": outcome.error_reason,
                    "business_code": "E_OBSERVE",
                    "diagnostic_code": outcome.diagnostic_code,
                }
                if outcome.status == "TIMEOUT":
                    timeout_nodes.append(detail)
                else:
                    missing_nodes.append(detail)
            else:
                per_node.append(
                    {
                        "node_id": node_id,
                        "value": latest.value,
                        "observation_seq": latest.observation_seq,
                        "timestamp_ms": latest.timestamp_ms,
                        "duplicate_count": len(outcome.observations),
                        "_value_token": canonical_value(latest.value),
                    }
                )

        per_node_sorted = sorted(per_node, key=lambda row: row["node_id"])
        valid_node_count = len(per_node_sorted)
        quorum_met = (
            valid_node_count * 10000 >= key_spec.quorum_bp * total_nodes
        )
        unique_tokens = sorted({row["_value_token"] for row in per_node_sorted})
        divergent = len(unique_tokens) > 1

        candidate_row = (
            sorted(per_node_sorted, key=_latest_row_sort_key)[0]
            if per_node_sorted
            else None
        )
        arbitration = _arbitrate(per_node_sorted)
        winner_row = arbitration["winner"]
        arbitrated_value = arbitration["winner_value"]
        effective_value = arbitrated_value if quorum_met else None
        expected_matches = (
            _strict_equal(effective_value, key_spec.expected)
            if effective_value is not None
            else None
        )

        attributions: list[str] = []
        if missing_nodes:
            attributions.append("OBSERVATION_MISSING")
        if timeout_nodes:
            attributions.append("OBSERVATION_TIMEOUT")
        if divergent:
            attributions.append("VALUE_DIVERGENCE")
        attributions = [item for item in ATTRIBUTION_ORDER if item in set(attributions)]

        if valid_node_count == 0:
            status = "NO_VALUE"
        elif not quorum_met:
            status = "INSUFFICIENT_OBSERVATIONS"
        elif divergent:
            status = "DIVERGED"
        elif not expected_matches:
            status = "EXPECTED_MISMATCH"
        else:
            status = "OK"

        decision = {
            "key": key,
            "status": status,
            "expected": key_spec.expected,
            "quorum_bp": key_spec.quorum_bp,
            "total_nodes": total_nodes,
            "valid_node_count": valid_node_count,
            "quorum_met": quorum_met,
            "divergent": divergent,
            "candidate": (
                {
                    "node_id": candidate_row["node_id"],
                    "value": candidate_row["value"],
                    "observation_seq": candidate_row["observation_seq"],
                    "timestamp_ms": candidate_row["timestamp_ms"],
                }
                if candidate_row is not None
                else None
            ),
            "arbitrated_value": arbitrated_value,
            "effective_value": effective_value,
            "winner_basis": arbitration["winner_basis"],
            "majority_count": arbitration["majority_count"],
            "tied_values": arbitration["tied_values"],
            "arbitration_rule": spec.arbitration_rule,
            "basis_text": arbitration["basis_text"],
            "expected_matches": expected_matches,
            "attributions": attributions,
            "attribution_details": {
                "missing_nodes": missing_nodes,
                "timeout_nodes": timeout_nodes,
                "divergent_values": [
                    row["value"]
                    for row in sorted(per_node_sorted, key=_latest_row_sort_key)
                    if row["_value_token"] in unique_tokens
                ] if divergent else [],
            },
        }
        decisions.append(decision)

        if divergent:
            divergence_keys.append(key)

        for row in per_node_sorted:
            outcome = outcome_rows[row["node_id"]]
            adopted = (
                winner_row is not None
                and row["_value_token"] == canonical_value(winner_row["value"])
            )
            if adopted:
                reason = f"ADOPTED:{arbitration['winner_basis']}"
                if not quorum_met:
                    reason += ":TENTATIVE_QUORUM_NOT_MET"
                if arbitration["winner_basis"] == "MAJORITY":
                    reason += f":{arbitration['majority_count']}_OF_{valid_node_count}"
            else:
                reason = "NOT_ADOPTED:VALUE_OUTLIER"
            comparisons.append(
                {
                    "key": key,
                    "node_id": row["node_id"],
                    "request_status": outcome.status,
                    "http_status": outcome.http_status,
                    "error_reason": outcome.error_reason,
                    "value": row["value"],
                    "observation_seq": row["observation_seq"],
                    "timestamp_ms": row["timestamp_ms"],
                    "duplicate_count": row["duplicate_count"],
                    "adopted": adopted,
                    "reason": reason,
                    "latest_candidate": (
                        candidate_row is not None
                        and row["node_id"] == candidate_row["node_id"]
                        and row["_value_token"] == candidate_row["_value_token"]
                    ),
                }
            )

        for node_id in node_ids:
            outcome = outcome_rows[node_id]
            has_latest = any(row["node_id"] == node_id for row in per_node_sorted)
            if has_latest:
                continue
            comparisons.append(
                {
                    "key": key,
                    "node_id": node_id,
                    "request_status": outcome.status,
                    "http_status": outcome.http_status,
                    "error_reason": outcome.error_reason,
                    "value": None,
                    "observation_seq": None,
                    "timestamp_ms": None,
                    "duplicate_count": 0,
                    "adopted": False,
                    "reason": f"NOT_ADOPTED:OBSERVATION_MISSING:{outcome.status}",
                    "latest_candidate": False,
                }
            )

    decisions.sort(key=lambda item: item["key"])
    comparisons.sort(key=lambda item: (item["key"], item["node_id"]))
    return MergeResult(
        decisions=tuple(decisions),
        comparisons=tuple(comparisons),
        divergence_keys=tuple(sorted(divergence_keys)),
    )
