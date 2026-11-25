"""Hierarchical fixed-point error bound derivation (requirement 4).

All bounds are conservative integer ULP bounds in Q units.

* Source level: one residual ULP per raw channel/topic/relevance quantization,
  hence three ULP for every adopted source evidence chain.
* Evidence level: the bounded sum of per-evidence source contributions over
  all adopted chains, plus one ULP for the round mean truncation.
* Round level: one mean-truncation ULP per round plus one ULP for the session
  view mean truncation.

The dominant level is the largest non-zero contribution; ties at the explicit
 50% threshold are broken deterministically as source > evidence > round.
"""

from .fixedpoint import div_trunc

SOURCE_PER_EVIDENCE_ULP = 3
DOMINANT_THRESHOLD_RATIO = 50


def _basis_points(part, total):
    if total <= 0:
        return 0
    quotient, _ = div_trunc(part * 10000, total)
    return quotient


def round_error_attribution(record):
    count = record["count"]
    source_bound = SOURCE_PER_EVIDENCE_ULP * count
    evidence_bound = source_bound + (1 if count else 0)
    round_bound = evidence_bound + 1 if count else 0
    total = source_bound + evidence_bound + round_bound
    contributions = [
        ("source", source_bound),
        ("evidence", evidence_bound),
        ("round", round_bound),
    ]
    shares = {
        level: _basis_points(bound, total) for level, bound in contributions
    }
    dominant_level, dominant_bound = max(
        contributions, key=lambda item: (item[1], -_level_priority(item[0]))
    )
    threshold_bp = _basis_points(dominant_bound, total)
    threshold_met = threshold_bp >= DOMINANT_THRESHOLD_RATIO * 100
    result = {
        "session_id": record["session_id"],
        "round": record["round"],
        "ulp_note": "one ULP equals 1/scale in Q-unit integer representation",
        "bounds_ulp": {
            "single_evidence_quantization": SOURCE_PER_EVIDENCE_ULP,
            "source_total": source_bound,
            "evidence_arbitration_total": evidence_bound,
            "round_arbitration_total": round_bound,
        },
        "shares_basis_points": shares,
        "dominant_level": dominant_level,
        "dominant_threshold_basis_points": DOMINANT_THRESHOLD_RATIO * 100,
        "dominant_observed_basis_points": threshold_bp,
        "dominant_threshold_met": threshold_met,
        "explicit_threshold": (
            "dominant level must account for at least "
            f"{DOMINANT_THRESHOLD_RATIO}% of total ULP bound"
        ),
    }
    tie = record.get("tie")
    if tie is not None:
        within = tie["quantized_delta"] <= round_bound
        tie["within_error_bound"] = within
        tie["error_affected"] = within
        tie["bound_ulp"] = round_bound
        tie["stable_rule"] = (
            "select max quantized_weight; tie -> min channel; tie -> "
            "min fragment_id; never use wall-clock or thread order"
        )
        result["tie_break"] = tie
    return result


def _level_priority(level):
    return {"source": 0, "evidence": 1, "round": 2}[level]


def session_error_attribution(session_id, records, round_attributions):
    ordered_records = sorted(records, key=lambda item: item["round"])
    ordered_attrs = sorted(round_attributions, key=lambda item: item["round"])
    source = sum(item["bounds_ulp"]["source_total"] for item in ordered_attrs)
    evidence = sum(
        item["bounds_ulp"]["evidence_arbitration_total"] for item in ordered_attrs
    )
    round_component = sum(
        1 if item["count"] else 0 for item in ordered_records
    ) + (1 if ordered_records else 0)
    total = source + evidence + round_component
    contributions = [
        ("source", source),
        ("evidence", evidence),
        ("round", round_component),
    ]
    shares = {
        level: _basis_points(bound, total) for level, bound in contributions
    }
    dominant_level, dominant_bound = max(
        contributions, key=lambda item: (item[1], -_level_priority(item[0]))
    )
    observed_bp = _basis_points(dominant_bound, total)
    threshold_met = observed_bp >= DOMINANT_THRESHOLD_RATIO * 100
    return {
        "session_id": session_id,
        "bounds_ulp": {
            "source_contribution": source,
            "evidence_contribution": evidence,
            "round_contribution": round_component,
            "session_view_aggregation_total": total,
            "view_mean_truncation_ulp": 1 if ordered_records else 0,
        },
        "shares_basis_points": shares,
        "dominant_level": dominant_level,
        "dominant_threshold_basis_points": DOMINANT_THRESHOLD_RATIO * 100,
        "dominant_observed_basis_points": observed_bp,
        "dominant_threshold_met": threshold_met,
        "explicit_threshold": (
            "dominant level must account for at least "
            f"{DOMINANT_THRESHOLD_RATIO}% of total session ULP bound"
        ),
    }
