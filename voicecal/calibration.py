"""Deterministic fixed-point turn calibration over an evidence chain."""

from __future__ import annotations

from typing import Any

from .constants import STAGE_ARTIFACT_VERSION
from .util import decimal_to_fixed, json_hash, mul_fixed, quantize_levels, clamp_int, round_div


def effective_gain(config: dict[str, Any], overrides: dict[str, Any], turn_id: str) -> float:
    turn_gain = overrides.get("turn_gain", {}).get(turn_id)
    if turn_gain is not None:
        return float(turn_gain)
    global_gain = overrides.get("gain")
    if global_gain is not None:
        return float(global_gain)
    return float(config["calibration"]["gain"])


def calibration_cache_key(
    initial_key: str,
    previous_calibration_key: str | None,
    turn: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    gain: float,
    config: dict[str, Any],
    injection_fingerprint: list[dict[str, Any]] | None = None,
) -> str:
    fixed = config["fixed_point"]
    previous_speaker = previous_turn["speaker_id"] if previous_turn is not None else None
    return json_hash(
        {
            "stage": "calibration",
            "artifact_version": STAGE_ARTIFACT_VERSION,
            "initial_key": initial_key,
            "previous_calibration_key": previous_calibration_key,
            "turn_id": turn["turn_id"],
            "conversation_id": turn["conversation_id"],
            "speaker_id": turn["speaker_id"],
            "previous_speaker_id": previous_speaker,
            "gain": gain,
            "decay": config["calibration"]["decay"],
            "same_speaker_factor": config["calibration"]["same_speaker_factor"],
            "different_speaker_factor": config["calibration"]["different_speaker_factor"],
            "speaker_switch_penalty": config["calibration"]["speaker_switch_penalty"],
            "decimal_places": fixed["decimal_places"],
            "quantization_levels": fixed["quantization_levels"],
            "injections": injection_fingerprint or [],
        }
    )


def calibrate_turn(
    initial_record: dict[str, Any],
    turn: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    previous_calibration: dict[str, Any] | None,
    chain_sum_before: dict[str, int],
    gain: float,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    fixed = config["fixed_point"]
    scale = int(fixed["scale"])
    levels = int(fixed["quantization_levels"])
    dp = int(fixed["decimal_places"])
    previous_key = previous_calibration["cache_key"] if previous_calibration is not None else None
    cache_key = calibration_cache_key(
        initial_record["cache_key"],
        previous_key,
        turn,
        previous_turn,
        gain,
        config,
    )
    gain_i = decimal_to_fixed(gain, dp)
    decay_i = decimal_to_fixed(config["calibration"]["decay"], dp)
    same_i = decimal_to_fixed(config["calibration"]["same_speaker_factor"], dp)
    different_i = decimal_to_fixed(config["calibration"]["different_speaker_factor"], dp)
    penalty_i = decimal_to_fixed(config["calibration"]["speaker_switch_penalty"], dp)
    speaker_changed = previous_turn is not None and previous_turn["speaker_id"] != turn["speaker_id"]
    factor_i = different_i if speaker_changed else same_i

    calibrated: dict[str, int] = {}
    previous_values = previous_calibration["calibrated_evidence_fixed"] if previous_calibration is not None else None
    chain_sum = dict(chain_sum_before)
    terms: dict[str, dict[str, int]] = {}
    for emotion in config["evidence"]["emotions"]:
        initial = int(initial_record["intensities_fixed"][emotion])
        previous = int(previous_values[emotion]) if previous_values is not None else 0
        gain_term = mul_fixed(previous, gain_i, scale)
        decayed = mul_fixed(gain_term, decay_i, scale)
        base = initial + decayed
        factored = mul_fixed(base, factor_i, scale)
        penalty = penalty_i if speaker_changed else 0
        value = clamp_int(factored - penalty, 0, scale)
        value = quantize_levels(value, scale, levels)
        calibrated[emotion] = value
        chain_sum[emotion] = chain_sum.get(emotion, 0) + value
        terms[emotion] = {
            "initial": initial,
            "previous_calibrated": previous,
            "gain_term": gain_term,
            "decayed_previous": decayed,
            "base_before_factor": base,
            "transition_factor": factor_i,
            "switch_penalty_applied": penalty,
            "final_quantized": value,
        }
    ranked = sorted(calibrated.items(), key=lambda item: (-item[1], item[0]))
    emotion_decision = ranked[0][0]
    record = {
        "cache_key": cache_key,
        "status": "ok",
        "initial_key": initial_record["cache_key"],
        "previous_calibration_key": previous_key,
        "previous_turn_id": previous_turn["turn_id"] if previous_turn is not None else None,
        "calibrated_evidence_fixed": calibrated,
        "emotion_decision": emotion_decision,
        "emotion_decision_score": calibrated[emotion_decision],
        "emotion_tie_break": "lexicographic_min_label",
        "chain_sum_fixed": chain_sum,
        "coefficients": {
            "gain_fixed": gain_i,
            "decay_fixed": decay_i,
            "same_speaker_factor_fixed": same_i,
            "different_speaker_factor_fixed": different_i,
            "speaker_switch_penalty_fixed": penalty_i,
            "transition_factor_fixed": factor_i,
            "speaker_changed": speaker_changed,
            "switch_penalty_applied": speaker_changed,
        },
        "terms_fixed": terms,
        "algorithm": "explicit_emotion_loop_initial_plus_decayed_gain_then_transition",
    }
    return cache_key, record
