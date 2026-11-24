"""Evidence-topic attribution and ordered fixed-point confidence synthesis."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .fixedpoint import FixedProfile, as_fraction, difference_record, half_up_div, fraction_float
from .hashing import cache_key, fingerprint
from .model import RuntimeState


def build_scoring_cache_key(
    state: RuntimeState,
    session_id: str,
    target_turn_id: str,
    recall_results: dict[tuple[str, str], dict],
    topic_results: dict[str, dict],
    attribution: dict[str, dict],
    profile: FixedProfile,
) -> tuple[str, int]:
    target = state.find_turn(session_id, target_turn_id)
    assert target is not None
    topic_map = topic_results[session_id]["turn_topic_map"]
    target_topic = topic_map[target_turn_id]
    fingerprints: list[str] = []
    point_count = 0
    order_index = 0
    for source in state.sorted_turns(session_id):
        if source.seq > target.seq:
            break
        recall = recall_results.get((session_id, source.turn_id))
        if recall is None:
            continue
        source_topic = topic_map[source.turn_id]
        for item in sorted(recall["evidence"], key=lambda evidence: evidence["evidence_id"]):
            order_index += 1
            point_count += 1
            evidence_id = item["evidence_id"]
            evidence = state.evidence[evidence_id]
            info = attribution[evidence_id]
            distance = target.seq - source.seq
            reasons = []
            if info["cross_topic"]:
                reasons.append("evidence_cross_topic")
            if source_topic != target_topic:
                reasons.append("source_topic_differs")
            fingerprints.append(
                fingerprint(
                    {
                        "order_index": order_index,
                        "source_turn_fingerprint": source.content_fingerprint,
                        "source_seq": source.seq,
                        "target_seq": target.seq,
                        "evidence_content_fingerprint": evidence.content_fingerprint,
                        "evidence_weight_fingerprint": evidence.weight_fingerprint,
                        "dedup_output_fingerprint": recall["output_fingerprint"],
                        "authority_channel_id": item["authority_channel_id"],
                        "citation_count": item["citation_count"],
                        "distance": distance,
                        "decay": state.distance_decay,
                        "topic_penalty": state.cross_topic_penalty,
                        "topic_reasons": reasons,
                        "attribution_fingerprint": info["attribution_fingerprint"],
                        "source_topic": source_topic,
                        "target_topic": target_topic,
                    }
                )
            )
    key = cache_key(
        "scoring",
        {
            "target_turn_fingerprint": target.content_fingerprint,
            "target_topic": target_topic,
            "chain_fingerprints": fingerprints,
            "normalization_weight": state.normalization_weight,
            "answer_threshold": state.answer_threshold,
            "quant_fingerprint": profile.fingerprint,
        },
    )
    return key, point_count


def attribute_evidence(
    state: RuntimeState,
    recall_results: dict[tuple[str, str], dict],
    topic_results: dict[str, dict],
) -> dict[str, dict]:
    attribution: dict[str, dict] = {}
    for evidence_id in sorted(state.evidence):
        citations: list[dict[str, Any]] = []
        for session_id in state.session_order:
            topic_result = topic_results[session_id]
            topic_map = topic_result["turn_topic_map"]
            for turn in state.sorted_turns(session_id):
                recall = recall_results.get((session_id, turn.turn_id))
                if recall is None:
                    continue
                for item in recall["evidence"]:
                    if item["evidence_id"] != evidence_id:
                        continue
                    citations.append(
                        {
                            "session_id": session_id,
                            "turn_id": turn.turn_id,
                            "seq": turn.seq,
                            "topic_id": topic_map[turn.turn_id],
                            "global_topic_key": f"{session_id}/{topic_map[turn.turn_id]}",
                            "citation_count": item["citation_count"],
                            "authority_channel_id": item["authority_channel_id"],
                        }
                    )
        citations.sort(key=lambda item: (item["session_id"], item["seq"], item["turn_id"]))
        topic_refs = sorted({(item["session_id"], item["topic_id"], item["global_topic_key"]) for item in citations})
        topic_ids = [item[1] for item in topic_refs]
        global_topic_keys = [item[2] for item in topic_refs]
        cross_topic = len(global_topic_keys) > 1
        key = cache_key(
            "evidence-attribution",
            {
                "evidence_content_fingerprint": state.evidence[evidence_id].content_fingerprint,
                "citations": [
                    fingerprint(
                        {
                            "session_id": item["session_id"],
                            "turn_id": item["turn_id"],
                            "seq": item["seq"],
                            "topic_key": item["global_topic_key"],
                            "citation_count": item["citation_count"],
                        }
                    )
                    for item in citations
                ],
            },
        )
        attribution[evidence_id] = {
            "stage": "evidence_attribution",
            "evidence_id": evidence_id,
            "cache_key": key,
            "citations": citations,
            "topic_ids": topic_ids,
            "global_topic_keys": global_topic_keys,
            "topic_count": len(global_topic_keys),
            "cross_topic": cross_topic,
            "attribution_fingerprint": fingerprint(
                [(item["turn_id"], item["global_topic_key"], item["citation_count"]) for item in citations]
            ),
        }
    return attribution


def score_turn(
    state: RuntimeState,
    session_id: str,
    target_turn_id: str,
    recall_results: dict[tuple[str, str], dict],
    topic_results: dict[str, dict],
    attribution: dict[str, dict],
    profile: FixedProfile,
    contribution_cache: dict[str, dict] | None = None,
) -> dict:
    target = state.find_turn(session_id, target_turn_id)
    assert target is not None
    topic_map = topic_results[session_id]["turn_topic_map"]
    target_topic = topic_map[target_turn_id]
    decay_base = as_fraction(state.distance_decay)
    penalty_base = as_fraction(state.cross_topic_penalty)
    normalization = as_fraction(state.normalization_weight)
    threshold = as_fraction(state.answer_threshold)
    tolerance = as_fraction(state.error_tolerance)

    chain: list[dict[str, Any]] = []
    order_index = 0
    for source in state.sorted_turns(session_id):
        if source.seq > target.seq:
            break
        recall = recall_results.get((session_id, source.turn_id))
        if recall is None:
            continue
        source_topic = topic_map[source.turn_id]
        evidence_items = sorted(recall["evidence"], key=lambda item: item["evidence_id"])
        for item in evidence_items:
            order_index += 1
            evidence_id = item["evidence_id"]
            evidence = state.evidence[evidence_id]
            info = attribution[evidence_id]
            distance = target.seq - source.seq
            decay_real = decay_base ** distance
            source_topic_differs = source_topic != target_topic
            penalty_reasons = []
            if info["cross_topic"]:
                penalty_reasons.append("evidence_cross_topic")
            if source_topic_differs:
                penalty_reasons.append("source_topic_differs")
            topic_penalty_real = penalty_base if penalty_reasons else Fraction(1)
            count = int(item["citation_count"])
            weight_real = as_fraction(evidence.weight)
            contribution_real = weight_real * count * decay_real * topic_penalty_real

            chain_payload_fingerprint = fingerprint(
                {
                    "order_index": order_index,
                    "source_turn_fingerprint": source.content_fingerprint,
                    "source_seq": source.seq,
                    "target_seq": target.seq,
                    "evidence_content_fingerprint": evidence.content_fingerprint,
                    "evidence_weight_fingerprint": evidence.weight_fingerprint,
                    "dedup_output_fingerprint": recall["output_fingerprint"],
                    "authority_channel_id": item["authority_channel_id"],
                    "citation_count": count,
                    "distance": distance,
                    "decay": state.distance_decay,
                    "topic_penalty": state.cross_topic_penalty,
                    "topic_reasons": penalty_reasons,
                    "attribution_fingerprint": info["attribution_fingerprint"],
                    "source_topic": source_topic,
                    "target_topic": target_topic,
                }
            )
            contribution_key = cache_key(
                "fixed-contribution",
                {"input_fingerprint": chain_payload_fingerprint, "quant_fingerprint": profile.fingerprint},
            )
            cached_contribution = contribution_cache.get(contribution_key) if contribution_cache else None
            if cached_contribution is not None:
                weight_int = int(cached_contribution["weight_int"])
                decay_int = int(cached_contribution["decay_int"])
                penalty_int = int(cached_contribution["penalty_int"])
                contribution_int = int(cached_contribution["contribution_int"])
                contribution_status = "reused"
            else:
                weight_int = profile.quantize(weight_real)
                decay_int = profile.quantize(decay_real)
                penalty_int = profile.quantize(topic_penalty_real)
                contribution_int = profile.multiply(weight_int, decay_int)
                contribution_int = profile.multiply(contribution_int, penalty_int)
                contribution_int *= count
                contribution_status = "recomputed"
            chain.append(
                {
                    "order_index": order_index,
                    "source_session_id": session_id,
                    "source_turn_id": source.turn_id,
                    "source_seq": source.seq,
                    "target_turn_id": target_turn_id,
                    "evidence_id": evidence_id,
                    "authority_channel_id": item["authority_channel_id"],
                    "source_channels": item["source_channels"],
                    "citation_count": count,
                    "distance": distance,
                    "source_topic_id": source_topic,
                    "target_topic_id": target_topic,
                    "evidence_cross_topic": info["cross_topic"],
                    "evidence_topic_ids": info["global_topic_keys"],
                    "topic_penalty_reasons": penalty_reasons,
                    "factors": {
                        "evidence_weight": {
                            "real": weight_real,
                            "fixed_int": weight_int,
                            "fixed_value": Fraction(weight_int, profile.scale),
                        },
                        "turn_distance_decay": {
                            "distance": distance,
                            "base": decay_base,
                            "real": decay_real,
                            "fixed_int": decay_int,
                            "fixed_value": Fraction(decay_int, profile.scale),
                        },
                        "cross_topic_penalty": {
                            "real": topic_penalty_real,
                            "fixed_int": penalty_int,
                            "fixed_value": Fraction(penalty_int, profile.scale),
                        },
                        "citation_count": count,
                    },
                    "contribution_real": contribution_real,
                    "contribution_int": contribution_int,
                    "contribution_fixed": Fraction(contribution_int, profile.scale),
                    "contribution_error": difference_record(
                        contribution_real, contribution_int, profile, tolerance
                    ),
                    "input_fingerprint": chain_payload_fingerprint,
                    "contribution_cache_key": contribution_key,
                    "cache_status": contribution_status,
                }
            )

    total_real = sum((item["contribution_real"] for item in chain), Fraction(0))
    total_int = 0
    for item in chain:
        total_int += item["contribution_int"]
    confidence_real = min(Fraction(1), max(Fraction(0), total_real / normalization))
    normalization_int = profile.quantize(normalization)
    confidence_int = half_up_div(total_int * profile.scale, normalization_int) if normalization_int else 0
    confidence_int = max(0, min(profile.scale, confidence_int))
    gear_step = profile.gear_step
    if gear_step > 1:
        confidence_int = ((confidence_int + gear_step // 2) // gear_step) * gear_step
        confidence_int = max(0, min(profile.scale, confidence_int))
    confidence_fixed = Fraction(confidence_int, profile.scale)
    conclusion = "supported" if confidence_fixed >= threshold else "uncertain"
    reference_conclusion = "supported" if confidence_real >= threshold else "uncertain"

    scoring_key = cache_key(
        "scoring",
        {
            "target_turn_fingerprint": target.content_fingerprint,
            "target_topic": target_topic,
            "chain_fingerprints": [item["input_fingerprint"] for item in chain],
            "normalization_weight": state.normalization_weight,
            "answer_threshold": state.answer_threshold,
            "quant_fingerprint": profile.fingerprint,
        },
    )
    confidence_error = difference_record(confidence_real, confidence_int, profile, tolerance)
    margins = {
        "candidate_margin_real": confidence_fixed - threshold,
        "reference_margin_real": confidence_real - threshold,
        "threshold_real": threshold,
    }
    flipped = conclusion != reference_conclusion
    flip_reason = None
    if flipped:
        reference_margin = margins["reference_margin_real"]
        candidate_margin = margins["candidate_margin_real"]
        if reference_margin >= 0 and candidate_margin < 0:
            flip_reason = {
                "classification": "quantization_rounding",
                "direction": "supported_to_uncertain",
                "criterion": "reference_confidence >= threshold but fixed_confidence < threshold",
                "reference_margin": fraction_float(reference_margin, 12),
                "candidate_margin": fraction_float(candidate_margin, 12),
                "rounding_error": confidence_error["abs_difference"],
            }
        else:
            flip_reason = {
                "classification": "quantization_rounding",
                "direction": "uncertain_to_supported",
                "criterion": "reference_confidence < threshold but fixed_confidence >= threshold",
                "reference_margin": fraction_float(reference_margin, 12),
                "candidate_margin": fraction_float(candidate_margin, 12),
                "rounding_error": confidence_error["abs_difference"],
            }

    topic_errors = []
    for item in recall_results_key_evidence(recall_results, session_id, target_turn_id):
        evidence_id = item["evidence_id"]
        info = attribution[evidence_id]
        penalty_real = penalty_base if info["cross_topic"] else Fraction(1)
        penalty_int = profile.quantize(penalty_real)
        topic_errors.append(
            {
                "session_id": session_id,
                "turn_id": target_turn_id,
                "evidence_id": evidence_id,
                "stage": "topic_attribution",
                "topic_id": target_topic,
                "evidence_topic_count": info["topic_count"],
                "cross_topic": info["cross_topic"],
                "attribution_count_delta": 0,
                "cross_topic_delta": 0,
                **difference_record(penalty_real, penalty_int, profile, tolerance),
            }
        )

    return {
        "stage": "confidence_synthesis",
        "session_id": session_id,
        "turn_id": target_turn_id,
        "turn_seq": target.seq,
        "scoring_cache_key": scoring_key,
        "topic_id": target_topic,
        "budget_points": len(chain),
        "evidence_chain": [
            {
                "order_index": item["order_index"],
                "source_turn_id": item["source_turn_id"],
                "source_seq": item["source_seq"],
                "evidence_id": item["evidence_id"],
                "citation_count": item["citation_count"],
                "distance": item["distance"],
                "authority_channel_id": item["authority_channel_id"],
            }
            for item in chain
        ],
        "chain_details": chain,
        "confidence_int": confidence_int,
        "confidence_scale": profile.scale,
        "confidence_decimal": profile.decimal_text(confidence_int),
        "confidence": fraction_float(confidence_fixed, 9),
        "total_weighted_int": total_int,
        "normalization_int": normalization_int,
        "conclusion": conclusion,
        "reference_confidence": fraction_float(confidence_real, 12),
        "reference_confidence_real": confidence_real,
        "reference_conclusion": reference_conclusion,
        "conclusion_flipped_vs_reference": flipped,
        "flip_reason": flip_reason,
        "threshold": fraction_float(threshold, 12),
        "margins": {
            "candidate_margin": fraction_float(margins["candidate_margin_real"], 12),
            "reference_margin": fraction_float(margins["reference_margin_real"], 12),
        },
        "confidence_error": confidence_error,
        "topic_errors": topic_errors,
    }


def recall_results_key_evidence(recall_results, session_id, turn_id):
    recall = recall_results.get((session_id, turn_id))
    if recall is None:
        return []
    return sorted(recall["evidence"], key=lambda item: item["evidence_id"])
