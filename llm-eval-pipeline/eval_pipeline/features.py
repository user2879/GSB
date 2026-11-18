"""Deterministic sub-word tokenization, hashed bucket vectors and reward scoring."""

from __future__ import annotations

import math
import re

import numpy as np

from .errors import E_NON_FINITE, PipelineError
from .util import double_hex, stable_hash

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str, precision_level: int) -> list:
    """Tokenize text into words and sub-word pieces.

    precision 1: lowercased word tokens only (fastest, coarsest)
    precision 2: words plus adjacent-character bigrams (default)
    precision 3: words, bigrams and trigrams (richest, most expensive)
    """
    words = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    tokens = list(words)
    if precision_level >= 2:
        for word in words:
            for index in range(len(word) - 1):
                tokens.append("2g:" + word[index:index + 2])
    if precision_level >= 3:
        for word in words:
            for index in range(len(word) - 2):
                tokens.append("3g:" + word[index:index + 3])
    return tokens


class FeatureExtractor:
    """Hashed bucket-count feature extractor with a fixed parameter fingerprint."""

    def __init__(self, precision_level: int, hash_buckets: int, tokenizer_seed: int = 0):
        self.precision_level = precision_level
        self.hash_buckets = hash_buckets
        self.tokenizer_seed = tokenizer_seed
        self.params = {
            "precision_level": precision_level,
            "hash_buckets": hash_buckets,
            "tokenizer_seed": tokenizer_seed,
            "version": 1,
        }

    def vector(self, text: str) -> np.ndarray:
        counts = np.zeros(self.hash_buckets, dtype=np.int64)
        for token in tokenize(text, self.precision_level):
            bucket = stable_hash(token, self.tokenizer_seed) % self.hash_buckets
            counts[bucket] += 1
        return counts

    def unique_buckets(self, texts: list) -> int:
        touched = set()
        for text in texts:
            for token in tokenize(text, self.precision_level):
                touched.add(stable_hash(token, self.tokenizer_seed) % self.hash_buckets)
        return len(touched)


def high_cardinality_ratio(vectors: list) -> float:
    """Fraction of non-empty buckets whose global count is exactly one."""
    if not vectors:
        return 0.0
    totals = np.zeros(vectors[0].shape[0], dtype=np.int64)
    for vector in vectors:
        totals = totals + vector
    used = totals > 0
    if int(used.sum()) == 0:
        return 0.0
    return float(np.count_nonzero(totals == 1)) / float(int(used.sum()))


def _bucket_weight(bucket: int) -> float:
    """Fixed per-bucket scoring weight derived from the bucket index (no RNG)."""
    raw = stable_hash(f"weight:{bucket}", seed=20240101)
    return (raw % 10_000) / 10_000.0


def score_text(vector: np.ndarray) -> float:
    """Score in a fixed bucket order so float summation is reproducible."""
    total = 0.0
    for bucket in range(vector.shape[0]):
        total += float(vector[bucket]) * _bucket_weight(bucket)
    return total


def score_pair(pair: dict, extractor: FeatureExtractor, boundary_threshold: float) -> dict:
    """Compute chosen/rejected features, scores and the double-precision diff."""
    prompt_vector = extractor.vector(pair["prompt"])
    chosen_vector = extractor.vector(pair["chosen"])
    rejected_vector = extractor.vector(pair["rejected"])

    chosen_score = score_text(chosen_vector + prompt_vector)
    rejected_score = score_text(rejected_vector + prompt_vector)
    diff = float(chosen_score - rejected_score)
    if not math.isfinite(diff):
        raise PipelineError(E_NON_FINITE, "non-finite reward difference", sample_id=pair["pair_id"], stage="scoring")

    return {
        "pair_id": pair["pair_id"],
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "reward_diff": diff,
        "reward_diff_ref": double_hex(diff),
        "boundary": abs(diff) < boundary_threshold,
        "chosen_buckets": _nonzero_buckets(chosen_vector),
        "rejected_buckets": _nonzero_buckets(rejected_vector),
        "prompt_buckets": _nonzero_buckets(prompt_vector),
    }


def _nonzero_buckets(vector: np.ndarray) -> list:
    return [int(index) for index in np.nonzero(vector)[0].tolist()]


def inject_non_finite(result: dict, injection_no: int) -> dict:
    """Fault injection: replace the reward diff with NaN to prove isolation."""
    corrupted = dict(result)
    corrupted["reward_diff"] = float("nan")
    corrupted["reward_diff_ref"] = "nan"
    corrupted["injection_no"] = injection_no
    return corrupted


def reward_distribution(scored: list) -> dict:
    """Distribution statistics over finite reward diffs (sorted, quantile based)."""
    values = sorted(
        item["reward_diff"]
        for item in scored
        if isinstance(item.get("reward_diff"), (int, float)) and math.isfinite(item["reward_diff"])
    )
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}

    def quantile(ratio: float) -> float:
        index = min(len(values) - 1, int(ratio * (len(values) - 1)))
        return float(values[index])

    return {
        "count": len(values),
        "min": float(values[0]),
        "max": float(values[-1]),
        "mean": float(sum(values) / len(values)),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "boundary_count": sum(1 for item in scored if item.get("boundary")),
    }
