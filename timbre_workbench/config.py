"""Configuration loading, validation and stage fingerprints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, WorkbenchError
from .util import canonical_json, sha256_obj


ENGINE_SCHEMA_VERSION = 1


def _invalid(message: str, **context: Any) -> WorkbenchError:
    return WorkbenchError(E_CONFIG_INVALID, message, context or None)


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": ENGINE_SCHEMA_VERSION,
    "audio": {"sample_rate": 16000, "channels": 1, "bits_per_sample": 16},
    "segmentation": {
        "frame_length_ms": 25.0,
        "frame_shift_ms": 10.0,
        "start_threshold_dbfs": -30.0,
        "continue_threshold_dbfs": -42.0,
        "energy_floor_dbfs": -70.0,
        "min_segment_ms": 80.0,
        "min_gap_ms": 50.0,
    },
    "features": {
        "f0_min_hz": 70.0,
        "f0_max_hz": 400.0,
        "voicing_threshold": 0.28,
        "valid_frame_dbfs": -48.0,
        "formant_count": 3,
        "formant_min_hz": 250.0,
        "formant_max_hz": 3800.0,
        "formant_smoothing_bins": 5,
        "lpc_order": 18,
        "lpc_preemphasis": 0.0,
        "quantiles": [0.1, 0.5, 0.9],
        "quantile_interpolation": "linear",
        "syllable_min_hz": 1.5,
        "syllable_max_hz": 8.0,
    },
    "quantization": {
        "dimensions": [
            {"name": "f0_mean", "source": "f0_mean_hz", "candidate_tiers": [4, 6, 8, 10], "max_step": 2.0},
            {"name": "f0_std", "source": "f0_std_hz", "candidate_tiers": [4, 6, 8], "max_step": 1.0},
            {"name": "f1_mean", "source": "f1_mean_hz", "candidate_tiers": [4, 6, 8], "max_step": 10.0},
            {"name": "f2_mean", "source": "f2_mean_hz", "candidate_tiers": [4, 6, 8], "max_step": 12.0},
            {"name": "f3_mean", "source": "f3_mean_hz", "candidate_tiers": [4, 6, 8], "max_step": 15.0},
            {"name": "speech_rate", "source": "speech_rate_sps", "candidate_tiers": [4, 6, 8], "max_step": 0.25},
        ],
        "minimum_range": 1e-9,
        "tie_break": "lowest_bits_then_source_name",
    },
    "assignment": {
        "metric": "rbf",
        "rbf_gamma": 8.0,
        "similarity_threshold": 0.92,
        "tie_break": "speaker_id_lexicographic",
        "annotation_policy": "record_conflict_keep_prediction",
    },
    "aggregation": {
        "fixed_scale": 1_000_000_000,
        "rounding": "half_away_from_zero",
    },
    "runtime": {
        "first_response_budget_ms": 0,
        "first_response_scheduling_tolerance_ms": 20.0,
        "max_workers": 4,
        "snapshot_after_each_command": True,
        "snapshot_keep": 5,
    },
    "snapshot": {"schema_version": ENGINE_SCHEMA_VERSION},
    "engine": {"schema_version": ENGINE_SCHEMA_VERSION},
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path | None = None, overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if path is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise _invalid("unable to read config", path=str(path), os_error=str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise _invalid("config is not valid JSON", path=str(path), json_error=str(exc)) from exc
    if overlay:
        raw = _deep_merge(raw, overlay)
    config = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    audio = config.get("audio", {})
    if audio.get("sample_rate") != 16000:
        raise _invalid("audio.sample_rate must be 16000")
    if audio.get("channels") != 1 or audio.get("bits_per_sample") != 16:
        raise _invalid("only mono, 16-bit PCM configuration is supported")

    seg = config.get("segmentation", {})
    for key in ("frame_length_ms", "frame_shift_ms", "min_segment_ms", "min_gap_ms"):
        if float(seg.get(key, 0)) <= 0:
            raise _invalid(f"segmentation.{key} must be positive")
    if float(seg["frame_shift_ms"]) > float(seg["frame_length_ms"]):
        raise _invalid("frame shift must not exceed frame length")
    if float(seg["continue_threshold_dbfs"]) >= float(seg["start_threshold_dbfs"]):
        raise _invalid("continue threshold must be below start threshold")

    features = config.get("features", {})
    if float(features.get("f0_min_hz", 0)) >= float(features.get("f0_max_hz", 0)):
        raise _invalid("f0_min_hz must be below f0_max_hz")
    if features.get("quantile_interpolation") not in {"linear", "lower", "higher", "midpoint"}:
        raise _invalid("unsupported quantile interpolation")

    q = config.get("quantization", {})
    dimensions = q.get("dimensions", [])
    if not dimensions:
        raise _invalid("at least one quantization dimension is required")
    names: set[str] = set()
    for dimension in dimensions:
        name = str(dimension.get("name", ""))
        if not name or name in names:
            raise _invalid("quantization dimension names must be non-empty and unique", name=name)
        names.add(name)
        tiers = dimension.get("candidate_tiers", [])
        if not tiers or any(int(tier) <= 0 or int(tier) > 24 for tier in tiers):
            raise _invalid("candidate_tiers must contain bit widths in 1..24", dimension=name)
        if sorted(set(int(tier) for tier in tiers)) != [int(tier) for tier in tiers]:
            raise _invalid("candidate_tiers must be strictly increasing", dimension=name)
        if float(dimension.get("max_step", 0)) <= 0:
            raise _invalid("max_step must be positive", dimension=name)

    assignment = config.get("assignment", {})
    if assignment.get("metric") not in {"rbf", "cosine", "normalized_l1"}:
        raise _invalid("unsupported assignment metric", metric=assignment.get("metric"))
    if not 0.0 <= float(assignment.get("similarity_threshold", -1)) <= 1.0:
        raise _invalid("similarity_threshold must be in [0, 1]")
    if assignment.get("annotation_policy") != "record_conflict_keep_prediction":
        raise _invalid("unsupported annotation policy")

    aggregation = config.get("aggregation", {})
    if int(aggregation.get("fixed_scale", 0)) <= 0:
        raise _invalid("aggregation.fixed_scale must be positive")
    if aggregation.get("rounding") != "half_away_from_zero":
        raise _invalid("only deterministic half_away_from_zero rounding is supported")

    runtime = config.get("runtime", {})
    if int(runtime.get("max_workers", 0)) <= 0:
        raise _invalid("runtime.max_workers must be positive")
    if int(runtime.get("first_response_budget_ms", -1)) < 0:
        raise _invalid("first_response_budget_ms must be non-negative")


def semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in {"runtime", "snapshot"}
    }


def config_hash(config: dict[str, Any]) -> str:
    return sha256_obj(semantic_config(config))


def stage_fingerprint(config: dict[str, Any], *sections: str, extra: Any = None) -> str:
    payload = {section: copy.deepcopy(config.get(section, {})) for section in sections}
    if extra is not None:
        payload["extra"] = extra
    return sha256_obj(payload)


def load_labels(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"files": {}}
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _invalid("unable to read labels", path=str(path), os_error=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise _invalid("labels are not valid JSON", path=str(path), json_error=str(exc)) from exc
    files = labels.get("files", {})
    if not isinstance(files, dict):
        raise _invalid("labels.files must be an object")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for filename, ranges in files.items():
        if not isinstance(ranges, list):
            raise _invalid("label ranges must be arrays", file=filename)
        converted: list[dict[str, Any]] = []
        for row in ranges:
            try:
                start = int(row["start_sample"])
                end = int(row["end_sample"])
                speaker_id = str(row["speaker_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise _invalid("invalid label range", file=filename, row=row) from exc
            if start < 0 or end <= start or not speaker_id:
                raise _invalid("invalid label range bounds", file=filename, row=row)
            converted.append({"start_sample": start, "end_sample": end, "speaker_id": speaker_id})
        converted.sort(key=lambda row: (row["start_sample"], row["end_sample"], row["speaker_id"]))
        normalized[str(filename).replace("\\", "/").lstrip("./")] = converted
    return {"files": normalized}


def labels_hash(labels: dict[str, Any]) -> str:
    return sha256_obj(labels)


def config_json(config: dict[str, Any]) -> str:
    return canonical_json(config)
