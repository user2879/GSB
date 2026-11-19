"""Speaker attribution scoring and trajectory reconstruction."""

from __future__ import annotations

from typing import Any

from .constants import E_SEGMENT_TOO_SHORT, STAGE_ARTIFACT_VERSION
from .util import clamp_int, decimal_to_fixed, json_hash, mul_fixed, round_div


def attribution_cache_key(
    calibration_key: str,
    extract_key: str,
    turn: dict[str, Any],
    previous_attribution_key: str | None,
    config: dict[str, Any],
    injection_fingerprint: list[dict[str, Any]] | None = None,
) -> str:
    fixed = config["fixed_point"]
    return json_hash(
        {
            "stage": "attribution",
            "artifact_version": STAGE_ARTIFACT_VERSION,
            "calibration_key": calibration_key,
            "extract_key": extract_key,
            "turn_id": turn["turn_id"],
            "conversation_id": turn["conversation_id"],
            "conversation_position": turn["conversation_position"],
            "speaker_id": turn["speaker_id"],
            "speaker_candidates": turn["speaker_candidates"],
            "gap_ms": turn["gap_ms"],
            "previous_attribution_key": previous_attribution_key,
            "attribution": config["attribution"],
            "decimal_places": fixed["decimal_places"],
            "quantization_levels": fixed["quantization_levels"],
            "injections": injection_fingerprint or [],
        }
    )


def _speaker_score(
    speaker_id: str,
    turn: dict[str, Any],
    calibration: dict[str, Any],
    extract_record: dict[str, Any],
    config: dict[str, Any],
) -> tuple[int, dict[str, int]]:
    fixed = config["fixed_point"]
    scale = int(fixed["scale"])
    dp = int(fixed["decimal_places"])
    profiles = config["attribution"]["speaker_profiles"]
    profile = profiles.get(speaker_id)
    if profile is None:
        affinity_i = 0
        pitch_match = 0
    else:
        affinity_sum = 0
        for emotion in config["evidence"]["emotions"]:
            affinity_i_emotion = decimal_to_fixed(profile["emotion_affinity"].get(emotion, 0), dp)
            affinity_sum += mul_fixed(calibration["calibrated_evidence_fixed"][emotion], affinity_i_emotion, scale)
        affinity_i = clamp_int(round_div(affinity_sum, scale), 0, scale)
        f0_mean = float(extract_record["aggregate_features"]["f0_mean_hz"])
        difference = abs(f0_mean - float(profile["pitch_center_hz"]))
        pitch_match = clamp_int(int(round((1.0 - difference / float(profile["pitch_tolerance_hz"])) * scale)), 0, scale)
    script_prior = scale if turn["speaker_id"] == speaker_id else 0
    weights = config["attribution"]["score_weights"]
    score = 0
    components = {"emotion_affinity": affinity_i, "pitch_match": pitch_match, "script_prior": script_prior}
    for key in ("emotion_affinity", "pitch_match", "script_prior"):
        weight_i = decimal_to_fixed(weights[key], dp)
        score += mul_fixed(components[key], weight_i, scale)
    return quantize_score(score, scale), components


def quantize_score(value: int, scale: int) -> int:
    return clamp_int(value, 0, scale)


def attribute_turn(
    turn: dict[str, Any],
    calibration: dict[str, Any],
    extract_record: dict[str, Any],
    previous_attribution: dict[str, Any] | None,
    trajectory_context: dict[str, Any] | None,
    config: dict[str, Any],
    all_turns: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    previous_key = previous_attribution["cache_key"] if previous_attribution is not None else None
    cache_key = attribution_cache_key(
        calibration["cache_key"], extract_record["cache_key"], turn, previous_key, config
    )
    duration_ms = float(extract_record["aggregate_features"]["duration_ms"])
    min_segment = float(config["attribution"]["min_segment_ms"])
    if duration_ms < min_segment:
        return cache_key, {
            "cache_key": cache_key,
            "status": "failed",
            "failure": {
                "code": E_SEGMENT_TOO_SHORT,
                "message": "segment is shorter than attribution.min_segment_ms",
                "duration_ms": duration_ms,
                "min_segment_ms": min_segment,
            },
        }

    scores: dict[str, int] = {}
    components: dict[str, dict[str, int]] = {}
    for speaker_id in sorted(turn["speaker_candidates"]):
        score, detail = _speaker_score(speaker_id, turn, calibration, extract_record, config)
        scores[speaker_id] = score
        components[speaker_id] = detail
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    speaker = ranked[0][0]
    tied = sum(1 for _, score in ranked if score == ranked[0][1])

    break_reason: str | None = None
    gap_turns = 0
    gap_ms = float(turn["gap_ms"])
    trajectory_id: str
    turn_ids: list[str]
    turn_indices: list[int]
    if previous_attribution is None or trajectory_context is None:
        break_reason = "conversation_start"
    else:
        previous_turn = trajectory_context["turn"]
        previous_speaker = trajectory_context["speaker_id"]
        gap_turns = int(turn["conversation_position"]) - int(previous_turn["conversation_position"]) - 1
        gap_ms = 0.0
        for candidate in all_turns:
            if (
                candidate["conversation_id"] == turn["conversation_id"]
                and int(previous_turn["conversation_position"]) < int(candidate["conversation_position"])
                <= int(turn["conversation_position"])
            ):
                gap_ms += float(candidate["gap_ms"])
        if previous_turn["conversation_id"] != turn["conversation_id"]:
            break_reason = "conversation_boundary"
        elif previous_speaker != speaker:
            break_reason = "speaker_change"
        elif gap_turns > int(config["attribution"]["max_gap_turns"]):
            break_reason = "gap_turns_exceeded"
        elif gap_ms > float(config["attribution"]["min_silence_ms"]):
            break_reason = "silence_gap"
    if break_reason is None:
        trajectory_id = trajectory_context["trajectory_id"]
        turn_ids = list(trajectory_context["turn_ids"]) + [turn["turn_id"]]
        turn_indices = list(trajectory_context["turn_indices"]) + [turn["turn_index"]]
    else:
        trajectory_id = f"{turn['conversation_id']}:{turn['turn_id']}"
        turn_ids = [turn["turn_id"]]
        turn_indices = [turn["turn_index"]]
    record = {
        "cache_key": cache_key,
        "status": "ok",
        "speaker_id": speaker,
        "speaker_scores_fixed": scores,
        "speaker_score_components_fixed": components,
        "score_tie_break": "speaker_id_lexicographic" if tied > 1 else "unique_max",
        "trajectory_id": trajectory_id,
        "trajectory_turn_ids": turn_ids,
        "trajectory_turn_indices": turn_indices,
        "trajectory_break_reason": break_reason,
        "gap_turns": gap_turns,
        "gap_ms": round(gap_ms, 6),
        "continued": break_reason is None,
    }
    return cache_key, record
