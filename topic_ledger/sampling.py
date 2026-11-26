from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .canonical import fingerprint
from .keys import round_cache_key, shard_cache_key, slice_cache_key, topic_parameter_fingerprint


def aggregate_term_counts(term_ids: list[int]) -> list[list[int]]:
    if not term_ids:
        return []
    values = np.asarray(term_ids, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    return [[int(term_id), int(count)] for term_id, count in zip(unique, counts)]


def topic_weights(config: dict[str, Any], vocab_size: int) -> np.ndarray:
    seed = int(config["topic_parameter_seed"]) + 1009 * int(config["topic_parameter_version"])
    rng = np.random.default_rng(seed)
    topic_count = int(config["topic_count"])
    tier = config["fixed_point"]["bit_width_tier"]
    dtype = np.int32 if tier == "I32" else np.int64
    weights = rng.integers(2, 64, size=(topic_count, vocab_size), dtype=np.int64)
    return weights.astype(dtype)


def fixed_soft_weights(topic_counts: np.ndarray, scale: int) -> list[int]:
    total = int(topic_counts.sum())
    topic_count = int(topic_counts.shape[0])
    if total == 0:
        weights = [0] * topic_count
        weights[0] = scale
        return weights
    raw = [(int(count) * scale) // total for count in topic_counts]
    remainder = scale - sum(raw)
    order = sorted(range(topic_count), key=lambda index: (-int(topic_counts[index]), index))
    cursor = 0
    while remainder > 0 and order:
        raw[order[cursor % len(order)]] += 1
        remainder -= 1
        cursor += 1
    return raw


def _stable_rng_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _integer_categorical(
    rng: np.random.Generator, scores: np.ndarray, size: int
) -> np.ndarray:
    cumulative = np.cumsum(scores.astype(np.int64), dtype=np.int64)
    total = int(cumulative[-1])
    if total <= 0:
        return np.zeros(size, dtype=np.int64)
    draws = rng.integers(0, total, size=size, dtype=np.int64)
    return np.searchsorted(cumulative, draws, side="right").astype(np.int64)


def compute_shard_receipt(request: dict[str, Any]) -> dict[str, Any]:
    shard_id = str(request["shard_id"])
    node_id = str(request["node_id"])
    config = request["config"]
    vocab_size = int(request["vocab_size"])
    term_counts_raw = request.get("term_counts") or []
    document_refs = request.get("document_refs") or []

    term_ids = [int(item[0]) for item in term_counts_raw]
    counts = np.asarray([int(item[1]) for item in term_counts_raw], dtype=np.int64)
    total_tokens = int(counts.sum()) if counts.size else 0
    topic_count = int(config["topic_count"])
    rounds = int(config["sample_rounds"])
    scale = int(config["fixed_point"]["scale"])
    tier = str(config["fixed_point"]["bit_width_tier"])
    topic_version = int(config["topic_parameter_version"])
    topic_fp = topic_parameter_fingerprint(config, vocab_size)
    slice_key = slice_cache_key(
        shard_id=shard_id,
        document_refs=document_refs,
        term_counts=[[term_id, int(count)] for term_id, count in zip(term_ids, counts)],
    )

    weights = topic_weights(config, vocab_size)
    shard_weights = weights[:, term_ids].astype(np.int64) if term_ids else np.zeros((topic_count, 0), dtype=np.int64)
    rng = np.random.default_rng(_stable_rng_seed("sampler", slice_key, topic_fp, tier, str(rounds)))

    assignment_counts = np.zeros((len(term_ids), topic_count), dtype=np.int64)
    for term_index, token_count in enumerate(counts):
        assignment_counts[term_index] = np.bincount(
            _integer_categorical(rng, shard_weights[:, term_index], int(token_count)),
            minlength=topic_count,
        )

    round_keys: list[str] = []
    round_fingerprints: list[str] = []
    for round_index in range(1, rounds + 1):
        next_counts = np.zeros_like(assignment_counts)
        for term_index, token_count in enumerate(counts):
            current = assignment_counts[term_index]
            scores = shard_weights[:, term_index] * (current + 1)
            assignments = _integer_categorical(rng, scores, int(token_count))
            next_counts[term_index] = np.bincount(assignments, minlength=topic_count)
        assignment_counts = next_counts
        round_fp = fingerprint(assignment_counts.tolist())
        round_fingerprints.append(round_fp)
        round_keys.append(
            round_cache_key(
                slice_key=slice_key,
                topic_fp=topic_fp,
                bit_width_tier=tier,
                rounds=rounds,
                round_index=round_index,
                round_fingerprint=round_fp,
            )
        )

    topic_totals = assignment_counts.sum(axis=0).astype(np.int64)
    soft_weights = fixed_soft_weights(topic_totals, scale)
    entries: list[dict[str, int]] = []
    for term_index, term_id in enumerate(term_ids):
        for topic_id in range(topic_count):
            entries.append(
                {
                    "term_id": term_id,
                    "topic_id": topic_id,
                    "count": int(assignment_counts[term_index, topic_id]),
                }
            )

    final_fp = fingerprint(
        {
            "counts": entries,
            "topic_totals": [int(value) for value in topic_totals],
            "soft_weights": soft_weights,
        }
    )
    shard_key = shard_cache_key(
        slice_key=slice_key,
        topic_fp=topic_fp,
        bit_width_tier=tier,
        rounds=rounds,
        final_fingerprint=final_fp,
    )
    work_points = total_tokens * rounds
    monotonic_ms = int(request.get("monotonic_clock_ms", 10_000 + work_points))

    receipt = {
        "schema": "topic-shard-receipt/v1",
        "shard_id": shard_id,
        "node_id": node_id,
        "topic_parameter_version": topic_version,
        "topic_parameter_fingerprint": topic_fp,
        "monotonic_ms": monotonic_ms,
        "work_points": work_points,
        "total_tokens": total_tokens,
        "rounds": rounds,
        "soft_summary": {
            "topic_counts": [int(value) for value in topic_totals],
            "soft_weights_fixed": soft_weights,
            "fixed_point_scale": scale,
        },
        "term_topic_counts": entries,
        "document_refs": sorted(document_refs, key=lambda item: item["doc_id"]),
        "cache_keys": {
            "slice": slice_key,
            "rounds": round_keys,
            "shard": shard_key,
        },
        "round_fingerprints": round_fingerprints,
    }
    receipt["receipt_digest"] = fingerprint(receipt)
    return receipt
