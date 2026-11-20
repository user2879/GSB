"""Resource loading and validation for phoneme map, acoustic and stream config."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from .errors import BusinessError, E_CONFIG_INVALID

PRECISION_LEVELS = {
    "low": 4,
    "medium": 6,
    "high": 10,
}


def round_sig(value, significant_digits: int):
    """Round to N significant digits, element-wise.

    Zero stays zero. Everything is computed in float64; rounding is only
    applied when a value is written into an artifact.
    """
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        if arr == 0.0 or not np.isfinite(float(arr)):
            return float(arr)
        factor = 10.0 ** (significant_digits - 1 - int(math.floor(math.log10(abs(float(arr))))))
        return float(np.round(arr * factor) / factor)
    out = arr.copy()
    finite = np.isfinite(arr) & (arr != 0.0)
    if finite.any():
        magnitude = np.floor(np.log10(np.abs(arr[finite])))
        factor = np.power(10.0, significant_digits - 1 - magnitude.astype(np.float64))
        out[finite] = np.round(arr[finite] * factor) / factor
    return out


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise BusinessError(E_CONFIG_INVALID, f"resource not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BusinessError(E_CONFIG_INVALID, f"invalid JSON in {path}: {exc}") from exc


@dataclass(frozen=True)
class PhonemeMap:
    # phoneme -> word, each phoneme maps to exactly one word (unique keys)
    mapping: Dict[str, str]
    words: List[str]

    @property
    def phonemes(self) -> List[str]:
        return list(self.mapping.keys())

    def word_of(self, phoneme: str) -> str:
        return self.mapping[phoneme]


@dataclass(frozen=True)
class AcousticConfig:
    frame_length_ms: int
    frame_shift_ms: int
    mel_bands: int
    # emission_table: phoneme -> {word: logprob}; missing word falls back to smooth
    emission_table: Dict[str, Dict[str, float]]
    # bigram probabilities in [0, 1], key "<word_a>|<word_b>"
    bigram: Dict[str, float]
    # word -> backoff weight in [0, 1]
    backoff: Dict[str, float]
    log_smooth: float
    # synthetic centroids (log-mel vectors) keyed by phoneme
    centroids: Dict[str, List[float]]


@dataclass(frozen=True)
class StreamConfig:
    chunk_ms: int
    overlap_ms: int
    min_consistent_frames: int
    redecode_budget_ms: int
    concurrency: int
    concurrency_limit: int
    snapshot_commit_interval_chunks: int
    precision: str
    simulated_task_cost_ms: int = 1

    @property
    def significant_digits(self) -> int:
        return PRECISION_LEVELS[self.precision]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BusinessError(E_CONFIG_INVALID, message)


def load_phoneme_map(path: str) -> PhonemeMap:
    raw = _load_json(path)
    _require(isinstance(raw, dict), "phoneme map must be a JSON object")
    mapping_raw = raw.get("mapping", raw)
    _require(isinstance(mapping_raw, dict), "phoneme map mapping must be an object")
    _require(len(mapping_raw) > 0, "phoneme map is empty")
    mapping: Dict[str, str] = {}
    for index, (phoneme, word) in enumerate(mapping_raw.items()):
        _require(isinstance(phoneme, str) and isinstance(word, str),
                 "phoneme and word entries must be strings")
        # JSON object keys are inherently unique; reject duplicate semantics defensively
        _require(phoneme not in mapping, f"duplicate phoneme entry: {phoneme}")
        mapping[phoneme] = word
    words = [mapping[phoneme] for phoneme in sorted(mapping)]
    return PhonemeMap(mapping=mapping, words=words)


def _validate_probability_table(table: Any, name: str) -> None:
    _require(isinstance(table, dict), f"{name} must be an object")
    for key, value in table.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                _require(isinstance(inner_value, (int, float)) and 0.0 <= float(inner_value) <= 1.0,
                         f"{name}[{key}][{inner_key}] not in [0,1]")
        else:
            _require(isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0,
                     f"{name}[{key}] not in [0,1]")


def load_acoustic_config(path: str, phoneme_map: PhonemeMap) -> AcousticConfig:
    raw = _load_json(path)
    _require(isinstance(raw, dict), "acoustic config must be an object")
    frame_length_ms = raw.get("frame_length_ms")
    frame_shift_ms = raw.get("frame_shift_ms")
    mel_bands = raw.get("mel_bands")
    _require(isinstance(frame_length_ms, int) and frame_length_ms > 0,
             "frame_length_ms must be a positive integer")
    _require(isinstance(frame_shift_ms, int) and 0 < frame_shift_ms <= frame_length_ms,
             "frame_shift_ms must be a positive integer <= frame_length_ms")
    _require(isinstance(mel_bands, int) and mel_bands > 0,
             "mel_bands must be a positive integer")
    emission_table = raw.get("emission_table", {})
    _require(isinstance(emission_table, dict), "emission_table must be an object")
    for phoneme, rows in emission_table.items():
        _require(phoneme in phoneme_map.mapping, f"emission table has unknown phoneme {phoneme}")
        _require(isinstance(rows, dict), "emission rows must be objects")
        for word, score in rows.items():
            _require(isinstance(score, (int, float)), "emission scores must be numeric")
    bigram = raw.get("bigram", {})
    backoff = raw.get("backoff", {})
    _validate_probability_table(bigram, "bigram")
    _validate_probability_table(backoff, "backoff")
    for word in backoff:
        _require(word in phoneme_map.words, f"backoff has unknown word {word}")
    log_smooth = raw.get("log_smooth")
    _require(isinstance(log_smooth, (int, float)), "log_smooth must be numeric")
    _require(log_smooth < 0.0, "log_smooth must be a negative log-domain value")
    centroids = raw.get("centroids", {})
    _require(isinstance(centroids, dict), "centroids must be an object")
    for phoneme, vector in centroids.items():
        _require(phoneme in phoneme_map.mapping, f"centroid has unknown phoneme {phoneme}")
        _require(isinstance(vector, list) and len(vector) == mel_bands,
                 f"centroid for {phoneme} must have {mel_bands} bands")
        for component in vector:
            _require(isinstance(component, (int, float)), "centroid components must be numeric")
    return AcousticConfig(
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        mel_bands=mel_bands,
        emission_table=emission_table,
        bigram=bigram,
        backoff=backoff,
        log_smooth=float(log_smooth),
        centroids={key: [float(v) for v in value] for key, value in centroids.items()},
    )


def load_stream_config(path: str) -> StreamConfig:
    raw = _load_json(path)
    _require(isinstance(raw, dict), "stream config must be an object")
    chunk_ms = raw.get("chunk_ms")
    overlap_ms = raw.get("overlap_ms")
    min_consistent_frames = raw.get("min_consistent_frames")
    budget = raw.get("redecode_budget_ms")
    concurrency = raw.get("concurrency")
    concurrency_limit = raw.get("concurrency_limit", concurrency)
    interval = raw.get("snapshot_commit_interval_chunks")
    precision = raw.get("precision")
    _require(isinstance(chunk_ms, int) and chunk_ms > 0, "chunk_ms must be a positive integer")
    _require(isinstance(overlap_ms, int) and overlap_ms > 0, "overlap_ms must be a positive integer")
    _require(overlap_ms < chunk_ms, "overlap_ms must be smaller than chunk_ms")
    _require(isinstance(min_consistent_frames, int) and min_consistent_frames > 0,
             "min_consistent_frames must be a positive integer")
    _require(isinstance(budget, int) and budget > 0, "redecode_budget_ms must be a positive integer")
    _require(isinstance(concurrency, int) and concurrency > 0, "concurrency must be a positive integer")
    _require(isinstance(concurrency_limit, int) and concurrency_limit > 0,
             "concurrency_limit must be a positive integer")
    _require(concurrency <= concurrency_limit, "concurrency must not exceed concurrency_limit")
    _require(isinstance(interval, int) and interval > 0,
             "snapshot_commit_interval_chunks must be a positive integer")
    _require(precision in PRECISION_LEVELS, f"precision must be one of {sorted(PRECISION_LEVELS)}")
    cost = raw.get("simulated_task_cost_ms", 1)
    _require(isinstance(cost, int) and cost >= 0, "simulated_task_cost_ms must be a non-negative integer")
    return StreamConfig(
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        min_consistent_frames=min_consistent_frames,
        redecode_budget_ms=budget,
        concurrency=concurrency,
        concurrency_limit=concurrency_limit,
        snapshot_commit_interval_chunks=interval,
        precision=precision,
        simulated_task_cost_ms=cost,
    )


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(phoneme_map: PhonemeMap, acoustic: AcousticConfig,
                stream: StreamConfig, extra: Dict[str, Any] | None = None) -> str:
    payload = {
        "phoneme_map": phoneme_map.mapping,
        "acoustic": {
            "frame_length_ms": acoustic.frame_length_ms,
            "frame_shift_ms": acoustic.frame_shift_ms,
            "mel_bands": acoustic.mel_bands,
            "emission_table": acoustic.emission_table,
            "bigram": acoustic.bigram,
            "backoff": acoustic.backoff,
            "log_smooth": acoustic.log_smooth,
            "centroids": acoustic.centroids,
        },
        "stream": {
            "chunk_ms": stream.chunk_ms,
            "overlap_ms": stream.overlap_ms,
            "min_consistent_frames": stream.min_consistent_frames,
            "redecode_budget_ms": stream.redecode_budget_ms,
            "concurrency": stream.concurrency,
            "concurrency_limit": stream.concurrency_limit,
            "snapshot_commit_interval_chunks": stream.snapshot_commit_interval_chunks,
            "precision": stream.precision,
            "simulated_task_cost_ms": stream.simulated_task_cost_ms,
        },
        "extra": extra or {},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def list_wav_files(input_dir: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".wav"):
            paths.append(os.path.join(input_dir, name))
    return paths
