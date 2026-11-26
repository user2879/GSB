from __future__ import annotations

from fractions import Fraction
from typing import Any

from .canonical import fingerprint
from .quantization import boundary_information


def _score(candidate: dict[str, Any]) -> Fraction:
    return Fraction(
        candidate["quantized_score_numerator"],
        candidate["quantized_score_denominator"],
    )


def _exact_score(candidate: dict[str, Any]) -> Fraction:
    return Fraction(candidate["score_numerator"], candidate["score_denominator"])


def decision_key_parts(
    event: dict[str, Any],
    candidates: list[dict[str, Any]],
    schema_event_type: dict[str, Any],
    revision: dict[str, Any],
    config: dict[str, Any],
    scope: dict[str, Any] | None = None,
) -> dict[str, str]:
    threshold = int(config.get("decision", {}).get("accept_threshold_permille", 600))
    required_roles = sorted(schema_event_type.get("required_roles", []))
    rule_fingerprint = fingerprint(
        {
            "required_roles": required_roles,
            "mutex_groups": schema_event_type.get("mutex_groups", []),
            "threshold_permille": threshold,
            "adjudication_rule_revision_id": (
                scope["adjudication_rule_revision_id"]
                if scope
                else revision["revision_id"]
            ),
            "adjudication_rule_fingerprint": (
                scope["adjudication_rule_fingerprint"]
                if scope
                else fingerprint(
                    {
                        "required_roles": required_roles,
                        "mutex_groups": schema_event_type.get("mutex_groups", []),
                    },
                    16,
                )
            ),
        },
        16,
    )
    cache_key = fingerprint(
        {
            "artifact": "role_decision",
            "event_id": event["event_id"],
            "all_candidate_keys": sorted(candidate["cache_key"] for candidate in candidates),
            "decision_rule_fingerprint": rule_fingerprint,
        }
    )
    return {"cache_key": cache_key, "rule_pack_fingerprint": rule_fingerprint}


def adjudicate_event(
    event: dict[str, Any],
    argument_result: dict[str, Any],
    schema_event_type: dict[str, Any],
    revision: dict[str, Any],
    config: dict[str, Any],
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = event["event_id"]
    threshold = int(config.get("decision", {}).get("accept_threshold_permille", 600))
    candidates = argument_result.get("candidates", [])
    accepted: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []

    for candidate in candidates:
        boundary = boundary_information(
            _exact_score(candidate), threshold, candidate["quantization"]
        )
        if boundary["boundary"]:
            boundary_records.append(
                {
                    "event_id": event_id,
                    "candidate_id": candidate["artifact_id"],
                    "arg_id": candidate["arg_id"],
                    "role": candidate["role"],
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "score_fixed": candidate["score_fixed"],
                    **boundary,
                }
            )
        if candidate.get("eligible") and _score(candidate) >= Fraction(threshold, 1000):
            accepted.append(candidate)

    accepted_roles = {candidate["role"] for candidate in accepted}
    required_roles = sorted(schema_event_type.get("required_roles", []))
    missing_roles = sorted(set(required_roles) - accepted_roles)

    suppressed: list[dict[str, Any]] = []
    mutex_decisions: list[dict[str, Any]] = []
    suppressed_ids: set[str] = set()
    quant_config = config.get("quantization", {})
    digits = int(quant_config.get("frac_digits", 3))
    level = str(quant_config.get("level", "standard"))
    multipliers = quant_config.get(
        "level_multipliers", {"fine": 1, "standard": 5, "coarse": 25}
    )
    quantum = int(quant_config.get("quantum_multiplier", multipliers.get(level, 5)))
    half_step = Fraction(quantum, 2 * (10**digits))

    for group_index, group in enumerate(schema_event_type.get("mutex_groups", [])):
        group_roles = list(group)
        best_by_role: dict[str, dict[str, Any]] = {}
        for candidate in accepted:
            if candidate["role"] not in group_roles or candidate["artifact_id"] in suppressed_ids:
                continue
            current = best_by_role.get(candidate["role"])
            if current is None or (
                _score(candidate),
                -candidate["start"],
                candidate["arg_id"],
            ) > (
                _score(current),
                -current["start"],
                current["arg_id"],
            ):
                best_by_role[candidate["role"]] = candidate

        if len(best_by_role) < 2:
            continue

        ranked = sorted(
            best_by_role.values(),
            key=lambda candidate: (
                -_score(candidate),
                candidate["role"],
                candidate["start"],
                candidate["arg_id"],
            ),
        )
        winner = ranked[0]
        loser_roles = {candidate["role"] for candidate in ranked[1:]}
        group_suppressed = [
            candidate
            for candidate in accepted
            if candidate["role"] in loser_roles
            and candidate["artifact_id"] != winner["artifact_id"]
        ]
        for candidate in group_suppressed:
            if candidate["artifact_id"] in suppressed_ids:
                continue
            suppressed_ids.add(candidate["artifact_id"])
            margin = _score(winner) - _score(candidate)
            near_tie = margin <= half_step
            record = {
                "event_id": event_id,
                "candidate_id": candidate["artifact_id"],
                "arg_id": candidate["arg_id"],
                "role": candidate["role"],
                "start": candidate["start"],
                "winner_candidate_id": winner["artifact_id"],
                "winner_role": winner["role"],
                "mutex_group_index": group_index,
                "suppression_reason": "role_mutex_confidence_order",
                "tie_rule": "confidence_desc,role_asc,start_asc",
                "score_margin_numerator": margin.numerator,
                "score_margin_denominator": margin.denominator,
                "near_mutex_tie_within_half_step": near_tie,
                "high_width_reference_value": candidate["quantization"][
                    "high_width_reference_value"
                ],
            }
            suppressed.append(record)

        mutex_decisions.append(
            {
                "event_id": event_id,
                "mutex_group": group_roles,
                "winner_candidate_id": winner["artifact_id"],
                "winner_role": winner["role"],
                "ranked_candidate_ids": [item["artifact_id"] for item in ranked],
                "suppressed_candidate_ids": [
                    item["candidate_id"] for item in suppressed if item["mutex_group_index"] == group_index
                ],
                "basis": "accepted mutex-role candidates sorted by quantized confidence desc then role asc",
            }
        )

    accepted_ids = sorted(
        candidate["artifact_id"]
        for candidate in accepted
        if candidate["artifact_id"] not in suppressed_ids
    )
    status = "complete" if not missing_roles else "incomplete"
    reason_codes = []
    if missing_roles:
        reason_codes.append("E_REQUIRED_ROLE_MISSING")

    active_candidate_keys = sorted(
        candidate["cache_key"]
        for candidate in candidates
        if candidate["artifact_id"] in accepted_ids
    )
    key_parts = decision_key_parts(
        event, candidates, schema_event_type, revision, config, scope
    )
    rule_fingerprint = key_parts["rule_pack_fingerprint"]
    cache_key = key_parts["cache_key"]
    return {
        "artifact_id": f"decision:{event_id}",
        "cache_key": cache_key,
        "event_id": event_id,
        "status": status,
        "complete": not missing_roles,
        "missing_required_roles": missing_roles,
        "reason_codes": reason_codes,
        "required_roles": required_roles,
        "accepted_candidate_ids": accepted_ids,
        "suppressed_candidates": suppressed,
        "boundary_candidates": sorted(
            boundary_records, key=lambda item: (item["start"], item["role"], item["candidate_id"])
        ),
        "mutex_decisions": mutex_decisions,
        "rule_pack_fingerprint": rule_fingerprint,
        "candidate_cache_keys": sorted(candidate["cache_key"] for candidate in candidates),
    }
