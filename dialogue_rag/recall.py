"""Channel-weighted recall and deterministic per-turn deduplication."""

from __future__ import annotations

from fractions import Fraction

from .fixedpoint import FixedProfile, as_fraction, difference_record
from .hashing import cache_key, fingerprint
from .model import RuntimeState


def build_recall_cache_key(state: RuntimeState, session_id: str, turn_id: str, profile: FixedProfile) -> str:
    turn = state.find_turn(session_id, turn_id)
    turn_fingerprint = turn.content_fingerprint if turn else fingerprint({"missing": turn_id})
    hits = state.hits.get((session_id, turn_id), [])
    declaration_fingerprint = fingerprint(
        [
            {
                "channel_id": hit.channel_id,
                "evidence_id": hit.evidence_id,
                "raw_score": hit.raw_score,
                "mentions": hit.mentions,
            }
            for hit in sorted(hits, key=lambda item: (item.channel_id, item.evidence_id, item.raw_score))
        ]
    )
    relevant_channels = {
        hit.channel_id: state.channels[hit.channel_id].weight_fingerprint
        for hit in hits
        if hit.channel_id in state.channels
    }
    return cache_key(
        "recall",
        {
            "turn_fingerprint": turn_fingerprint,
            "declaration_fingerprint": declaration_fingerprint,
            "channel_weight_fingerprints": relevant_channels,
            "quant_fingerprint": profile.fingerprint,
        },
    )


def compute_turn_recall(state: RuntimeState, session_id: str, turn_id: str, profile: FixedProfile) -> dict:
    key = (session_id, turn_id)
    hits = state.hits.get(key, [])
    candidates: list[dict] = []
    relevant_channels: dict[str, str] = {}

    for input_index, hit in enumerate(hits):
        channel = state.channels[hit.channel_id]
        relevant_channels[hit.channel_id] = channel.weight_fingerprint
        raw_score = as_fraction(hit.raw_score)
        channel_weight = as_fraction(channel.weight)
        weighted_score = raw_score * channel_weight
        score_int = profile.quantize(weighted_score)
        candidates.append(
            {
                "input_index": input_index,
                "channel_id": hit.channel_id,
                "evidence_id": hit.evidence_id,
                "mentions": hit.mentions,
                "raw_score_real": raw_score,
                "channel_weight_real": channel_weight,
                "weighted_score_real": weighted_score,
                "weighted_score_int": score_int,
                "weighted_score_fixed": Fraction(score_int, profile.scale),
            }
        )

    evidence_ids = sorted({item["evidence_id"] for item in candidates})
    merged: list[dict] = []
    recall_errors: list[dict] = []
    dedup_errors: list[dict] = []
    tolerance = as_fraction(state.error_tolerance)

    for evidence_id in evidence_ids:
        group = [item for item in candidates if item["evidence_id"] == evidence_id]
        ordered = sorted(
            group,
            key=lambda item: (
                -item["weighted_score_real"],
                item["channel_id"],
                -item["raw_score_real"],
                item["input_index"],
            ),
        )
        authority = ordered[0]
        source_channels = sorted({item["channel_id"] for item in group})
        covered = []
        for item in ordered[1:]:
            covered.append(
                {
                    "channel_id": item["channel_id"],
                    "input_index": item["input_index"],
                    "weighted_score_real": item["weighted_score_real"],
                    "authority_channel_id": authority["channel_id"],
                    "reason": "weighted_score_lower"
                    if item["weighted_score_real"] < authority["weighted_score_real"]
                    else "channel_name_later",
                }
            )
        citation_count = sum(item["mentions"] for item in group)
        authority_diff = difference_record(
            authority["weighted_score_real"], authority["weighted_score_int"], profile, tolerance
        )
        recall_errors.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "evidence_id": evidence_id,
                "channel_id": authority["channel_id"],
                "stage": "recall_scoring",
                **authority_diff,
            }
        )
        merged_score_int = profile.quantize(authority["weighted_score_real"])
        dedup_diff = difference_record(
            authority["weighted_score_real"], merged_score_int, profile, tolerance
        )
        dedup_errors.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "evidence_id": evidence_id,
                "stage": "dedup_merge",
                "citation_count": citation_count,
                "citation_count_delta": 0,
                "included": True,
                **dedup_diff,
            }
        )
        merged.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "evidence_id": evidence_id,
                "authority_channel_id": authority["channel_id"],
                "authority_input_index": authority["input_index"],
                "source_channels": source_channels,
                "covered_channels": covered,
                "citation_count": citation_count,
                "merged_score_real": authority["weighted_score_real"],
                "merged_score_int": merged_score_int,
                "merge_basis": {
                    "sort_keys": ["weighted_score_desc", "channel_id_asc", "raw_score_desc", "input_index_asc"],
                    "authority_channel_id": authority["channel_id"],
                    "authority_weighted_score": authority["weighted_score_real"],
                    "candidate_count": len(group),
                    "covered_count": len(covered),
                },
            }
        )

    merged.sort(
        key=lambda item: (
            -item["merged_score_real"],
            item["authority_channel_id"],
            item["evidence_id"],
        )
    )
    selected: list[dict] = []
    excluded: list[dict] = []
    for rank, item in enumerate(merged, start=1):
        ranked = dict(item)
        ranked["rank"] = rank
        if rank <= state.max_evidence_per_turn:
            ranked["included"] = True
            selected.append(ranked)
        else:
            ranked["included"] = False
            excluded.append(ranked)
            for error in dedup_errors:
                if error["evidence_id"] == item["evidence_id"]:
                    error["included"] = False

    turn = state.find_turn(session_id, turn_id)
    recall_key = build_recall_cache_key(state, session_id, turn_id, profile)
    dedup_key = cache_key(
        "dedup",
        {
            "recall_cache_key": recall_key,
            "merged_fingerprint": fingerprint(
                [(item["evidence_id"], item["authority_channel_id"], item["citation_count"], item["merged_score_real"]) for item in merged]
            ),
            "max_evidence_per_turn": state.max_evidence_per_turn,
            "quant_fingerprint": profile.fingerprint,
        },
    )
    output_fingerprint = fingerprint(
        [(item["evidence_id"], item["authority_channel_id"], item["source_channels"], item["citation_count"]) for item in selected]
    )
    return {
        "stage": "recall_dedup",
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_seq": turn.seq if turn else None,
        "recall_cache_key": recall_key,
        "dedup_cache_key": dedup_key,
        "output_fingerprint": output_fingerprint,
        "candidates": candidates,
        "evidence": selected,
        "excluded_by_limit": excluded,
        "recall_errors": recall_errors,
        "dedup_errors": dedup_errors,
    }
