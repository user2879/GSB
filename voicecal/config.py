"""Calibration configuration loading and strict validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .constants import CONFIG_SCHEMA_VERSION, E_CONFIG_INVALID
from .errors import VoicecalError
from .util import json_hash, read_json

EMOTIONS_DEFAULT = ["angry", "happy", "neutral", "sad"]
FEATURE_KEYS = {
    "f0_mean",
    "f0_std",
    "energy_mean",
    "energy_dynamic",
    "speech_rate",
    "rhythm_variability",
}


def _invalid(message: str, **details: Any) -> VoicecalError:
    return VoicecalError(E_CONFIG_INVALID, message, **details)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{field} must be an object", field=field)
    return value


def _require_number(value: Any, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{field} must be numeric", field=field)
    number = float(value)
    if not math.isfinite(number):
        raise _invalid(f"{field} must be finite", field=field)
    if minimum is not None and number < minimum:
        raise _invalid(f"{field} must be >= {minimum}", field=field)
    if maximum is not None and number > maximum:
        raise _invalid(f"{field} must be <= {maximum}", field=field)
    return number


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _invalid("configuration root must be an object")
    if int(raw.get("schema_version", -1)) != CONFIG_SCHEMA_VERSION:
        raise _invalid("unsupported configuration schema version", expected=CONFIG_SCHEMA_VERSION)

    frame = _require_mapping(raw.get("frame"), "frame")
    length_ms = _require_number(frame.get("length_ms"), "frame.length_ms", 1)
    shift_ms = _require_number(frame.get("shift_ms"), "frame.shift_ms", 1)
    if shift_ms > length_ms:
        raise _invalid("frame.shift_ms must not exceed frame.length_ms")
    sample_rate = int(_require_number(frame.get("sample_rate", 16000), "frame.sample_rate", 1))
    if sample_rate != 16000:
        raise _invalid("frame.sample_rate must be 16000", sample_rate=sample_rate)
    frame_length_samples = int(round(length_ms * sample_rate / 1000.0))
    frame_shift_samples = int(round(shift_ms * sample_rate / 1000.0))
    if frame_length_samples < 2 or frame_shift_samples < 1:
        raise _invalid("frame length/shift produce invalid sample counts")
    pitch_min = _require_number(frame.get("pitch_min_hz"), "frame.pitch_min_hz", 20)
    pitch_max = _require_number(frame.get("pitch_max_hz"), "frame.pitch_max_hz", 20)
    if pitch_max <= pitch_min:
        raise _invalid("frame.pitch_max_hz must exceed frame.pitch_min_hz")
    if sample_rate / pitch_max < 2:
        raise _invalid("frame length is too short for configured maximum pitch")
    frame_config = {
        "sample_rate": sample_rate,
        "length_ms": length_ms,
        "shift_ms": shift_ms,
        "frame_length_samples": frame_length_samples,
        "frame_shift_samples": frame_shift_samples,
        "frame_length_ms": length_ms,
        "frame_shift_ms": shift_ms,
        "pitch_min_hz": pitch_min,
        "pitch_max_hz": pitch_max,
        "voiced_correlation": _require_number(frame.get("voiced_correlation"), "frame.voiced_correlation", 0, 1),
        "silence_floor_dbfs": _require_number(frame.get("silence_floor_dbfs"), "frame.silence_floor_dbfs", -160, 0),
        "onset_margin_db": _require_number(frame.get("onset_margin_db"), "frame.onset_margin_db", 0, 100),
        "min_onset_gap_ms": _require_number(frame.get("min_onset_gap_ms"), "frame.min_onset_gap_ms", 1),
    }

    evidence = _require_mapping(raw.get("evidence"), "evidence")
    emotions_raw = evidence.get("emotions", EMOTIONS_DEFAULT)
    if not isinstance(emotions_raw, list) or not emotions_raw or not all(isinstance(item, str) and item for item in emotions_raw):
        raise _invalid("evidence.emotions must be a non-empty string array", field="evidence.emotions")
    emotions = sorted(set(emotions_raw))
    if len(emotions) != len(emotions_raw):
        raise _invalid("evidence.emotions must not contain duplicates", field="evidence.emotions")
    weights = _require_mapping(evidence.get("initial_weights"), "evidence.initial_weights")
    if set(weights) != set(emotions):
        raise _invalid("evidence.initial_weights must define every emotion exactly once")
    clean_weights: dict[str, dict[str, float]] = {}
    for emotion in emotions:
        per_emotion = _require_mapping(weights[emotion], f"evidence.initial_weights.{emotion}")
        missing = FEATURE_KEYS - set(per_emotion)
        extra = set(per_emotion) - FEATURE_KEYS
        if missing or extra:
            raise _invalid(
                "emotion weight feature set mismatch",
                emotion=emotion,
                missing=sorted(missing),
                extra=sorted(extra),
            )
        clean_weights[emotion] = {
            key: _require_number(per_emotion[key], f"evidence.initial_weights.{emotion}.{key}", 0, 10)
            for key in sorted(FEATURE_KEYS)
        }
    ranges = _require_mapping(evidence.get("feature_ranges"), "evidence.feature_ranges")
    clean_ranges: dict[str, list[float]] = {}
    for key in sorted(FEATURE_KEYS):
        pair = ranges.get(key)
        if not isinstance(pair, list) or len(pair) != 2:
            raise _invalid("feature range must be a two-element array", feature=key)
        low = _require_number(pair[0], f"evidence.feature_ranges.{key}[0]")
        high = _require_number(pair[1], f"evidence.feature_ranges.{key}[1]")
        if high <= low:
            raise _invalid("feature range high must exceed low", feature=key)
        clean_ranges[key] = [low, high]
    if evidence.get("normalization", "stable_softmax") != "stable_softmax":
        raise _invalid("evidence.normalization must be 'stable_softmax'")

    calibration = _require_mapping(raw.get("calibration"), "calibration")
    clean_calibration = {
        "gain": _require_number(calibration.get("gain"), "calibration.gain", 0, 1),
        "decay": _require_number(calibration.get("decay"), "calibration.decay", 0, 1),
        "same_speaker_factor": _require_number(calibration.get("same_speaker_factor"), "calibration.same_speaker_factor", 0, 1),
        "different_speaker_factor": _require_number(
            calibration.get("different_speaker_factor"), "calibration.different_speaker_factor", 0, 1
        ),
        "speaker_switch_penalty": _require_number(
            calibration.get("speaker_switch_penalty"), "calibration.speaker_switch_penalty", 0, 1
        ),
    }

    attribution = _require_mapping(raw.get("attribution"), "attribution")
    score_weights = _require_mapping(attribution.get("score_weights"), "attribution.score_weights")
    clean_score_weights = {
        "emotion_affinity": _require_number(score_weights.get("emotion_affinity"), "attribution.score_weights.emotion_affinity", 0, 1),
        "pitch_match": _require_number(score_weights.get("pitch_match"), "attribution.score_weights.pitch_match", 0, 1),
        "script_prior": _require_number(score_weights.get("script_prior"), "attribution.score_weights.script_prior", 0, 1),
    }
    profile_weights_total = sum(clean_score_weights.values())
    if abs(profile_weights_total - 1.0) > 1e-9:
        raise _invalid("attribution.score_weights must sum to 1.0", total=profile_weights_total)
    profiles = _require_mapping(attribution.get("speaker_profiles"), "attribution.speaker_profiles")
    if not profiles:
        raise _invalid("attribution.speaker_profiles must not be empty")
    clean_profiles: dict[str, dict[str, Any]] = {}
    for speaker_id, profile_raw in sorted(profiles.items()):
        profile = _require_mapping(profile_raw, f"attribution.speaker_profiles.{speaker_id}")
        affinity = _require_mapping(
            profile.get("emotion_affinity", {}), f"attribution.speaker_profiles.{speaker_id}.emotion_affinity"
        )
        clean_affinity = {emotion: _require_number(affinity.get(emotion, 0), f"speaker_affinity.{speaker_id}.{emotion}", 0, 1) for emotion in emotions}
        clean_profiles[speaker_id] = {
            "pitch_center_hz": _require_number(profile.get("pitch_center_hz"), f"speaker_profile.{speaker_id}.pitch_center_hz", 20, 1000),
            "pitch_tolerance_hz": _require_number(profile.get("pitch_tolerance_hz"), f"speaker_profile.{speaker_id}.pitch_tolerance_hz", 1),
            "emotion_affinity": clean_affinity,
        }
    max_gap_turns = int(_require_number(attribution.get("max_gap_turns"), "attribution.max_gap_turns", 0))
    clean_attribution = {
        "min_segment_ms": _require_number(attribution.get("min_segment_ms"), "attribution.min_segment_ms", 1),
        "min_silence_ms": _require_number(attribution.get("min_silence_ms"), "attribution.min_silence_ms", 0),
        "max_gap_turns": max_gap_turns,
        "score_weights": clean_score_weights,
        "speaker_profiles": clean_profiles,
    }

    fixed_raw = _require_mapping(raw.get("fixed_point"), "fixed_point")
    decimal_places = int(_require_number(fixed_raw.get("decimal_places"), "fixed_point.decimal_places", 0, 9))
    bit_width = int(_require_number(fixed_raw.get("bit_width", 32), "fixed_point.bit_width", 8, 64))
    levels = int(_require_number(fixed_raw.get("quantization_levels"), "fixed_point.quantization_levels", 2))
    if levels > 1_000_001:
        raise _invalid("fixed_point.quantization_levels is unreasonably large", levels=levels)
    fixed = {
        "decimal_places": decimal_places,
        "bit_width": bit_width,
        "quantization_levels": levels,
        "scale": 10**decimal_places,
    }

    concurrency = int(_require_number(raw.get("concurrency"), "concurrency", 1, 64))
    snapshot_interval = int(_require_number(raw.get("snapshot_commit_interval"), "snapshot_commit_interval", 1))
    batch_budget = int(_require_number(raw.get("batch_budget"), "batch_budget", 1))

    clean = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "frame": frame_config,
        "evidence": {
            "emotions": emotions,
            "initial_weights": clean_weights,
            "feature_ranges": clean_ranges,
            "normalization": "stable_softmax",
        },
        "calibration": clean_calibration,
        "attribution": clean_attribution,
        "fixed_point": fixed,
        "concurrency": concurrency,
        "snapshot_commit_interval": snapshot_interval,
        "batch_budget": batch_budget,
    }
    clean["config_hash"] = json_hash(clean)
    return clean


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = read_json(path)
    except FileNotFoundError as exc:
        raise _invalid("configuration file is missing", file=str(path)) from exc
    except ValueError as exc:
        raise _invalid("configuration file is not valid JSON", file=str(path)) from exc
    return validate_config(raw)
