"""Global configuration and local sample metadata loading."""

from __future__ import annotations

import copy
import math
import wave
from pathlib import Path
from typing import Any

from .constants import CONFIG_SCHEMA_VERSION, SCORE_TYPES
from .util import read_json, sha256_json


def _require_int(config: dict[str, Any], name: str, minimum: int) -> None:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_number(config: dict[str, Any], name: str, lower: float, upper: float) -> None:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not (lower <= float(value) <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}]")


def load_config(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")

    classes = config.get("classes")
    if not isinstance(classes, list) or len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("classes must contain four unique class names")
    if not all(isinstance(item, str) and item for item in classes):
        raise ValueError("every class name must be a non-empty string")

    priority = config.get("class_priority")
    if not isinstance(priority, list) or sorted(priority) != sorted(classes):
        raise ValueError("class_priority must be a permutation of classes")

    _require_int(config, "frame_shift_ms", 1)
    _require_int(config, "thread_pool_workers", 1)
    _require_int(config, "snapshot_commit_interval", 1)
    _require_int(config, "interaction_budget_ms", 1)
    _require_number(config, "decision_threshold", 0.0, 1.0)
    _require_number(config, "probability_quantum", 1e-12, 1.0)
    _require_number(config, "drift_alert_threshold", 0.0, 1e9)
    _require_number(config, "calibration_tolerance", 1e-15, 1e-1)

    quantiles = config.get("calibration_quantiles")
    if (
        not isinstance(quantiles, list)
        or len(quantiles) != 2
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in quantiles)
    ):
        raise ValueError("calibration_quantiles must contain two numeric values")
    low, high = float(quantiles[0]), float(quantiles[1])
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("calibration_quantiles must satisfy 0 <= low < high <= 1")

    codebook_version = config.get("class_codebook_version")
    codebook = config.get("class_codebook")
    if not isinstance(codebook_version, str) or not codebook_version:
        raise ValueError("class_codebook_version must be a non-empty string")
    if not isinstance(codebook, list) or len(codebook) != len(classes):
        raise ValueError("class_codebook must contain one entry per class")
    codebook_names = [entry.get("name") for entry in codebook if isinstance(entry, dict)]
    codebook_codes = [entry.get("code") for entry in codebook if isinstance(entry, dict)]
    if len(codebook_names) != len(classes) or sorted(codebook_names) != sorted(classes):
        raise ValueError("class_codebook entries must cover exactly the global classes")
    if len(set(codebook_codes)) != len(codebook_codes) or not all(
        isinstance(code, int) and not isinstance(code, bool) for code in codebook_codes
    ):
        raise ValueError("class_codebook codes must be unique integers")

    if not isinstance(config.get("global_version"), str) or not config["global_version"]:
        raise ValueError("global_version must be a non-empty string")

    # These fields are retained for documentation in the report, but only the
    # two canonical markers are accepted so malformed inputs fail loudly.
    marker = config.get("score_type_markers", list(SCORE_TYPES))
    if set(marker) != set(SCORE_TYPES):
        raise ValueError("score_type_markers must contain logit and probability")

    return copy.deepcopy(config)


def config_hash(config: dict[str, Any]) -> str:
    return sha256_json({"kind": "config", "config": config})


def load_codebooks(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("codebooks"), dict):
        raise ValueError("codebooks.json must contain a codebooks object")
    return value


def codebook_hash(codebooks: dict[str, Any]) -> str:
    return sha256_json({"kind": "codebooks", "codebooks": codebooks})


def load_baseline(path: str | Path) -> list[str]:
    value = read_json(path)
    ids = value.get("baseline_packet_ids") if isinstance(value, dict) else None
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("baseline_packet_ids must be a list of strings")
    if len(set(ids)) != len(ids):
        raise ValueError("baseline_packet_ids must be unique")
    return list(ids)


def load_injections(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    value = read_json(path)
    items = value.get("injections") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise ValueError("injections must be a list")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each injection must be an object")
        injection_id = item.get("injection_id")
        kind = item.get("kind")
        if not isinstance(injection_id, str) or not injection_id or injection_id in seen:
            raise ValueError("injection_id values must be unique non-empty strings")
        if kind not in {
            "truncated_packet",
            "frame_count_mismatch",
            "missing_score_type",
            "nonfinite_logit",
            "unmappable_codebook",
        }:
            raise ValueError(f"unsupported injection kind: {kind}")
        seen.add(injection_id)
        result.append(copy.deepcopy(item))
    return result


def load_truth(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {"clips": {}, "background_label": "none"}
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("clips"), dict):
        raise ValueError("truth.json must contain a clips object")
    return value


def load_audio_metadata(audio_dir: str | Path) -> dict[str, dict[str, Any]]:
    audio_dir = Path(audio_dir)
    result: dict[str, dict[str, Any]] = {}
    if not audio_dir.exists():
        return result
    for path in sorted(audio_dir.glob("*.wav")):
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            frame_count = handle.getnframes()
        if channels != 1 or sample_width != 2 or rate != 16_000:
            raise ValueError(
                f"{path.name} must be 16 kHz mono 16-bit PCM "
                f"(got {rate} Hz, {channels} channel(s), {sample_width * 8} bit)"
            )
        duration_ms = frame_count * 1_000.0 / rate
        result[path.name] = {
            "audio_file": path.name,
            "sample_rate": rate,
            "channels": channels,
            "bits_per_sample": sample_width * 8,
            "sample_count": frame_count,
            "duration_ms": duration_ms,
            "byte_size": path.stat().st_size,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
    return result
