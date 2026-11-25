from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Sequence

from .fingerprint import (
    evidence_content_fingerprint,
)
from .fixedpoint import FixedDomain
from .models import AppConfig, Evidence, Turn
from .similarity import combined_topic_score_bp, jaccard_bp
from .util import sha256_json, unique_sorted


def compute_recall(
    turn: Turn,
    config: AppConfig,
    evidence_map: Mapping[str, Evidence],
    domain: FixedDomain,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[int]]] = {}
    for channel, evidence_ids in sorted(turn.channels.items()):
        for evidence_id in evidence_ids:
            grouped.setdefault(evidence_id, {}).setdefault(channel, []).append(1)

    results: list[dict[str, Any]] = []
    for evidence_id in sorted(grouped):
        channels = grouped[evidence_id]
        breakdown: list[dict[str, Any]] = []
        for channel, occurrences in sorted(channels.items()):
            weight_units = domain.value_to_units(config.channel_weights[channel])
            breakdown.append(
                {
                    "channel": channel,
                    "channel_weight_units": weight_units,
                    "occurrence_count": len(occurrences),
                    "score_units": weight_units,
                    "score": domain.units_text(weight_units),
                }
            )
        authority = sorted(
            breakdown,
            key=lambda item: (-item["score_units"], item["channel"]),
        )[0]
        source_channels = sorted(channels)
        results.append(
            {
                "evidence_id": evidence_id,
                "evidence_fingerprint": evidence_content_fingerprint(
                    evidence_map[evidence_id]
                ),
                "authority_channel": authority["channel"],
                "authority_score_units": authority["score_units"],
                "authority_score": domain.units_text(authority["score_units"]),
                "source_channels": source_channels,
                "covered_channels": [
                    channel
                    for channel in source_channels
                    if channel != authority["channel"]
                ],
                "reference_count": sum(
                    item["occurrence_count"] for item in breakdown
                ),
                "score_breakdown": breakdown,
            }
        )
    return {
        "domain": domain.describe(),
        "evidence": results,
        "ordering": "evidence_id ascending; authority by score desc then channel asc",
    }


def compute_dedupe(recall: Mapping[str, Any]) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    for item in recall["evidence"]:
        merged.append(
            {
                "evidence_id": item["evidence_id"],
                "evidence_fingerprint": item["evidence_fingerprint"],
                "authority_channel": item["authority_channel"],
                "merged_score_units": item["authority_score_units"],
                "merged_score": item["authority_score"],
                "source_channels": list(item["source_channels"]),
                "covered_channels": list(item["covered_channels"]),
                "reference_count": item["reference_count"],
                "score_breakdown": list(item["score_breakdown"]),
            }
        )
    return {
        "domain": recall["domain"],
        "evidence": merged,
        "merge_order": [item["evidence_id"] for item in merged],
        "rule": "highest score wins; ties by channel name ascending",
    }


def compute_boundary(
    left: Turn,
    right: Turn,
    left_dedupe: Mapping[str, Any],
    right_dedupe: Mapping[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    query_similarity_bp = jaccard_bp(left.query, right.query)
    left_ids = {item["evidence_id"] for item in left_dedupe["evidence"]}
    right_ids = {item["evidence_id"] for item in right_dedupe["evidence"]}
    evidence_similarity_bp = jaccard_bp(left_ids, right_ids)
    combined_bp = combined_topic_score_bp(
        query_similarity_bp, evidence_similarity_bp
    )
    threshold_bp = config.topic.switch_threshold_bp
    exact_equal = combined_bp == threshold_bp
    split = combined_bp < threshold_bp
    return {
        "query_similarity_bp": query_similarity_bp,
        "evidence_similarity_bp": evidence_similarity_bp,
        "combined_score_bp": combined_bp,
        "threshold_bp": threshold_bp,
        "exact_equal": exact_equal,
        "split": split,
        "decision": "split" if split else "continue",
        "equal_rule": "continue_on_equal",
    }


def compute_topics(
    turns: Sequence[Turn],
    dedupe_results: Mapping[int, Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    evidence_map: Mapping[str, Evidence],
) -> tuple[list[dict[str, Any]], dict[int, str], dict[int, dict[str, Any]]]:
    if not turns:
        return [], {}, {}
    starts = [0]
    for index, boundary in enumerate(boundaries):
        if boundary["split"]:
            starts.append(index + 1)
    topics: list[dict[str, Any]] = []
    turn_topics: dict[int, str] = {}
    turn_topic_records: dict[int, dict[str, Any]] = {}
    used_ids: set[str] = set()
    for topic_index, start_index in enumerate(starts):
        end_index = (
            starts[topic_index + 1] - 1
            if topic_index + 1 < len(starts)
            else len(turns) - 1
        )
        start_turn = turns[start_index]
        identity_evidence = [
            {
                "evidence_id": item["evidence_id"],
                "content_fingerprint": evidence_content_fingerprint(
                    evidence_map[item["evidence_id"]]
                ),
            }
            for item in dedupe_results[start_turn.turn_id]["evidence"]
        ]
        base_id = "topic-" + sha256_json(
            {
                "start_query": start_turn.query,
                "start_channel_declarations": {
                    channel: list(ids)
                    for channel, ids in sorted(start_turn.channels.items())
                },
                "start_evidence_distribution": identity_evidence,
            }
        )[:12]
        topic_id = base_id
        suffix = 2
        while topic_id in used_ids:
            topic_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(topic_id)
        topic_turns = turns[start_index : end_index + 1]
        topic = {
            "topic_id": topic_id,
            "start_turn_id": start_turn.turn_id,
            "end_turn_id": turns[end_index].turn_id,
            "start_index": start_index + 1,
            "end_index": end_index + 1,
            "boundary_topic": topic_index > 0,
            "boundary_start": topic_index > 0,
            "boundary_exact_equal": (
                boundaries[start_index - 1]["exact_equal"]
                if topic_index > 0 and boundaries
                else False
            ),
            "turn_ids": [turn.turn_id for turn in topic_turns],
        }
        topics.append(topic)
        for offset, turn in enumerate(topic_turns):
            turn_topics[turn.turn_id] = topic_id
            turn_topic_records[turn.turn_id] = {
                "topic_id": topic_id,
                "start_turn_id": start_turn.turn_id,
                "end_turn_id": turns[end_index].turn_id,
                "offset": offset,
                "length": len(topic_turns),
                "topic_start_content_fingerprint": sha256_json(
                    {
                        "query": start_turn.query,
                        "channels": {
                            channel: list(ids)
                            for channel, ids in sorted(start_turn.channels.items())
                        },
                    }
                ),
                "boundary_topic": topic_index > 0,
            }
    return topics, turn_topics, turn_topic_records


def compute_evidence_attribution(
    turns: Sequence[Turn],
    dedupe_results: Mapping[int, Mapping[str, Any]],
    turn_topics: Mapping[int, str],
    evidence_map: Mapping[str, Evidence],
) -> dict[str, dict[str, Any]]:
    cited: dict[str, list[int]] = {}
    for turn in turns:
        for item in dedupe_results[turn.turn_id]["evidence"]:
            cited.setdefault(item["evidence_id"], []).append(turn.turn_id)
    result: dict[str, dict[str, Any]] = {}
    for evidence_id in sorted(cited):
        turn_ids = sorted(set(cited[evidence_id]))
        topic_ids = unique_sorted(turn_topics[turn_id] for turn_id in turn_ids)
        result[evidence_id] = {
            "evidence_id": evidence_id,
            "evidence_fingerprint": evidence_content_fingerprint(
                evidence_map[evidence_id]
            ),
            "citing_turns": turn_ids,
            "topic_ids": topic_ids,
            "cross_topic": len(topic_ids) > 1,
        }
    return result


def compute_confidence(
    turn: Turn,
    dedupe: Mapping[str, Any],
    config: AppConfig,
    domain: FixedDomain,
    evidence_map: Mapping[str, Evidence],
    evidence_attribution: Mapping[str, Mapping[str, Any]],
    topic_record: Mapping[str, Any],
) -> dict[str, Any]:
    decay_units = domain.value_to_units(config.confidence.turn_decay)
    penalty_factor_units = domain.value_to_units(
        Decimal(1) - config.confidence.cross_topic_penalty
    )
    answer_threshold_units = domain.value_to_units(
        config.confidence.answer_threshold
    )
    sum_contributions = 0
    sum_base_weights = 0
    citation_chain: list[dict[str, Any]] = []
    ordered = sorted(dedupe["evidence"], key=lambda item: item["evidence_id"])
    max_authority_score = max(
        (int(item["merged_score_units"]) for item in ordered), default=0
    )
    topic_start = int(topic_record["start_turn_id"])
    topic_end = int(topic_record["end_turn_id"])
    for item in ordered:
        evidence_id = item["evidence_id"]
        evidence = evidence_map[evidence_id]
        attribution = evidence_attribution[evidence_id]
        prior_in_topic = [
            turn_id
            for turn_id in attribution["citing_turns"]
            if topic_start <= turn_id <= topic_end
        ]
        first_turn = min(prior_in_topic) if prior_in_topic else turn.turn_id
        distance = max(0, turn.turn_id - first_turn)
        decay_units_for_evidence = domain.pow_ratio(decay_units, distance)
        base_weight_units = domain.value_to_units(evidence.weight)
        reliability_units = (
            domain.div(int(item["merged_score_units"]), max_authority_score)
            if max_authority_score
            else 0
        )
        effective_base_units = domain.mul(base_weight_units, reliability_units)
        cross_topic = bool(attribution["cross_topic"])
        cross_factor_units = penalty_factor_units if cross_topic else domain.one
        distance_weighted = domain.mul(
            effective_base_units, decay_units_for_evidence
        )
        contribution_units = domain.mul(distance_weighted, cross_factor_units)
        sum_base_weights = domain.add(sum_base_weights, effective_base_units)
        sum_contributions = domain.add(sum_contributions, contribution_units)
        citation_chain.append(
            {
                "evidence_id": evidence_id,
                "evidence_fingerprint": evidence_content_fingerprint(evidence),
                "authority_channel": item["authority_channel"],
                "source_channels": list(item["source_channels"]),
                "covered_channels": list(item["covered_channels"]),
                "reference_count": item["reference_count"],
                "base_weight_units": base_weight_units,
                "base_weight": domain.units_text(base_weight_units),
                "authority_score_units": int(item["merged_score_units"]),
                "authority_reliability_units": reliability_units,
                "authority_reliability": domain.units_text(reliability_units),
                "effective_base_weight_units": effective_base_units,
                "effective_base_weight": domain.units_text(effective_base_units),
                "first_turn_in_topic": first_turn,
                "turn_distance": distance,
                "decay_units": decay_units_for_evidence,
                "decay_factor": domain.units_text(decay_units_for_evidence),
                "cross_topic": cross_topic,
                "topic_ids": list(attribution["topic_ids"]),
                "cross_topic_factor_units": cross_factor_units,
                "cross_topic_factor": domain.units_text(cross_factor_units),
                "contribution_units": contribution_units,
                "contribution": domain.units_text(contribution_units),
            }
        )
    confidence_units = domain.div(sum_contributions, sum_base_weights)
    conclusion = confidence_units >= answer_threshold_units
    return {
        "domain": domain.describe(),
        "confidence_units": confidence_units,
        "confidence": domain.units_text(confidence_units),
        "answer_threshold_units": answer_threshold_units,
        "answer_threshold": domain.units_text(answer_threshold_units),
        "conclusion": "answer" if conclusion else "abstain",
        "conclusion_bool": conclusion,
        "sum_contribution_units": sum_contributions,
        "sum_contribution": domain.units_text(sum_contributions),
        "sum_base_weight_units": sum_base_weights,
        "sum_base_weight": domain.units_text(sum_base_weights),
        "accumulation_order": [item["evidence_id"] for item in ordered],
        "citation_chain": citation_chain,
    }
