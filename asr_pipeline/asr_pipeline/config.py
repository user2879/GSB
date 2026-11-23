"""Configuration loading, strict validation and canonical fingerprinting."""

import json
import math

from .errors import E_CONFIG_INVALID, PipelineError

ALLOWED_QUANT_MODES = ("fp64", "q16", "q32")
ALLOWED_FRAC_BITS = (8, 16)
MAX_CONCURRENCY = 64
SNAPSHOT_VERSION = 1


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_int_positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PipelineError(E_CONFIG_INVALID, f"{name} must be a positive integer, got {value!r}")
    return value


def _require_finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(E_CONFIG_INVALID, f"{name} must be a finite number, got {value!r}")
    if not math.isfinite(float(value)):
        raise PipelineError(E_CONFIG_INVALID, f"{name} must be finite, got {value!r}")
    return float(value)


def load_phoneme_map(path):
    data = _load_json(path)
    if not isinstance(data, dict) or not data:
        raise PipelineError(E_CONFIG_INVALID, "phoneme_map must be a non-empty object")
    mapping = {}
    for phoneme, char in data.items():
        if not isinstance(phoneme, str) or not isinstance(char, str) or not char:
            raise PipelineError(E_CONFIG_INVALID, f"invalid phoneme map entry {phoneme!r}")
        mapping[phoneme] = char
    return mapping


def load_score_params(path):
    data = _load_json(path)
    required = [
        "frame_length_ms", "frame_shift_ms", "mel_bands", "vocab",
        "emission_table", "bigram_probs", "backoff_weight", "unigram_probs",
        "log_smooth", "significant_digits",
    ]
    for key in required:
        if key not in data:
            raise PipelineError(E_CONFIG_INVALID, f"score params missing key: {key}")

    frame_length_ms = _require_int_positive(data["frame_length_ms"], "frame_length_ms")
    frame_shift_ms = _require_int_positive(data["frame_shift_ms"], "frame_shift_ms")
    if frame_shift_ms > frame_length_ms:
        raise PipelineError(E_CONFIG_INVALID, "frame_shift_ms must be <= frame_length_ms")
    mel_bands = _require_int_positive(data["mel_bands"], "mel_bands")

    vocab = data["vocab"]
    if not isinstance(vocab, list) or not vocab or any(not isinstance(c, str) for c in vocab):
        raise PipelineError(E_CONFIG_INVALID, "vocab must be a non-empty list of strings")
    if len(set(vocab)) != len(vocab):
        raise PipelineError(E_CONFIG_INVALID, "vocab entries must be unique")

    table = data["emission_table"]
    if not isinstance(table, list) or len(table) != mel_bands:
        raise PipelineError(
            E_CONFIG_INVALID,
            f"emission_table rows ({_safe_len(table)}) must equal mel_bands ({mel_bands})",
        )
    for row_idx, row in enumerate(table):
        if not isinstance(row, list) or len(row) != len(vocab):
            raise PipelineError(
                E_CONFIG_INVALID,
                f"emission_table row {row_idx} width {_safe_len(row)} != vocab size {len(vocab)}",
            )
        for value in row:
            _require_finite_number(value, f"emission_table[{row_idx}]")

    bigram = data["bigram_probs"]
    if not isinstance(bigram, dict):
        raise PipelineError(E_CONFIG_INVALID, "bigram_probs must be an object")
    for key, probs in bigram.items():
        if not isinstance(probs, dict):
            raise PipelineError(E_CONFIG_INVALID, f"bigram_probs[{key}] must be an object")
        for next_char, prob in probs.items():
            _require_finite_number(prob, f"bigram_probs[{key}][{next_char}]")

    unigram = data["unigram_probs"]
    if not isinstance(unigram, dict) or set(unigram.keys()) != set(vocab):
        raise PipelineError(E_CONFIG_INVALID, "unigram_probs must cover every vocab char")
    for char, prob in unigram.items():
        _require_finite_number(prob, f"unigram_probs[{char}]")

    backoff_weight = _require_finite_number(data["backoff_weight"], "backoff_weight")
    log_smooth = _require_finite_number(data["log_smooth"], "log_smooth")
    significant_digits = _require_int_positive(data["significant_digits"], "significant_digits")
    if not 1 <= significant_digits <= 17:
        raise PipelineError(E_CONFIG_INVALID, "significant_digits must be within [1, 17]")

    return {
        "frame_length_ms": frame_length_ms,
        "frame_shift_ms": frame_shift_ms,
        "mel_bands": mel_bands,
        "vocab": list(vocab),
        "emission_table": [[float(v) for v in row] for row in table],
        "bigram_probs": {k: {n: float(v) for n, v in row.items()} for k, row in bigram.items()},
        "unigram_probs": {k: float(v) for k, v in unigram.items()},
        "backoff_weight": backoff_weight,
        "log_smooth": log_smooth,
        "significant_digits": significant_digits,
    }


def _safe_len(value):
    return len(value) if isinstance(value, list) else "non-list"


def load_schedule_config(path):
    data = _load_json(path)
    required = [
        "max_concurrency", "global_point_budget", "shard_point_quota",
        "shards_per_batch", "snapshot_commit_shards", "quant_mode",
        "quant_frac_bits", "shard_duration_ms", "overlap_ms",
    ]
    for key in required:
        if key not in data:
            raise PipelineError(E_CONFIG_INVALID, f"schedule config missing key: {key}")

    max_concurrency = _require_int_positive(data["max_concurrency"], "max_concurrency")
    if max_concurrency > MAX_CONCURRENCY:
        raise PipelineError(E_CONFIG_INVALID, f"max_concurrency must be <= {MAX_CONCURRENCY}")
    global_budget = _require_int_positive(data["global_point_budget"], "global_point_budget")
    shard_quota = _require_int_positive(data["shard_point_quota"], "shard_point_quota")
    shards_per_batch = _require_int_positive(data["shards_per_batch"], "shards_per_batch")
    snapshot_commit_shards = _require_int_positive(
        data["snapshot_commit_shards"], "snapshot_commit_shards"
    )
    shard_duration_ms = _require_int_positive(data["shard_duration_ms"], "shard_duration_ms")
    overlap_ms = _require_int_positive(data["overlap_ms"], "overlap_ms")
    if overlap_ms >= shard_duration_ms:
        raise PipelineError(E_CONFIG_INVALID, "overlap_ms must be < shard_duration_ms")

    quant_mode = data["quant_mode"]
    if quant_mode not in ALLOWED_QUANT_MODES:
        raise PipelineError(E_CONFIG_INVALID, f"quant_mode must be one of {ALLOWED_QUANT_MODES}")
    quant_frac_bits = data["quant_frac_bits"]
    if quant_frac_bits not in ALLOWED_FRAC_BITS:
        raise PipelineError(E_CONFIG_INVALID, f"quant_frac_bits must be one of {ALLOWED_FRAC_BITS}")

    return {
        "max_concurrency": max_concurrency,
        "global_point_budget": global_budget,
        "shard_point_quota": shard_quota,
        "shards_per_batch": shards_per_batch,
        "snapshot_commit_shards": snapshot_commit_shards,
        "quant_mode": quant_mode,
        "quant_frac_bits": quant_frac_bits,
        "shard_duration_ms": shard_duration_ms,
        "overlap_ms": overlap_ms,
    }


def load_injections(path):
    """Return {injection_id: injection spec}; empty dict when path is None."""
    if not path:
        return {}
    data = _load_json(path)
    if not isinstance(data, list):
        raise PipelineError(E_CONFIG_INVALID, "injection config must be a list")
    result = {}
    allowed_types = {
        "truncate_wav", "non_finite_samples", "emission_dim_mismatch", "quota_below_min_points",
    }
    for item in data:
        for key in ("id", "type", "audio", "shard_index"):
            if key not in item:
                raise PipelineError(E_CONFIG_INVALID, f"injection entry missing {key}")
        if item["type"] not in allowed_types:
            raise PipelineError(E_CONFIG_INVALID, f"unknown injection type {item['type']}")
        if not isinstance(item["id"], str) or item["id"] in result:
            raise PipelineError(E_CONFIG_INVALID, "injection id must be a unique string")
        if not isinstance(item["shard_index"], int) or item["shard_index"] < 0:
            raise PipelineError(E_CONFIG_INVALID, "injection shard_index must be >= 0")
        result[item["id"]] = dict(item)
    return result


def canonical_json(obj):
    """Deterministic JSON text used for fingerprints."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
