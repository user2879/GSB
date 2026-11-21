"""The deterministic NumPy forward pass shared by server and verification code."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .util import canonical_json, scalar_to_u64, sha256_hex


MASK = np.uint64((1 << 63) - 1)
MULTIPLIER = np.uint64(6364136223846793005)
INCREMENT = np.uint64(1442695040888963407)


def _u64_words(data: bytes, count: int) -> np.ndarray:
    needed = count * 8
    repeats = (needed + len(data) - 1) // len(data)
    raw = (data * repeats)[:needed]
    return np.frombuffer(raw, dtype=">u8").astype(np.uint64)


def deterministic_input(seed: int, input_size: int, shard_index: int) -> np.ndarray:
    material = hashlib.sha256(f"input:{seed}:{input_size}:{shard_index}".encode("ascii")).digest()
    return _u64_words(material, input_size) & MASK


def deterministic_weights(seed: int, config_version: int, shard_index: int, input_size: int) -> np.ndarray:
    material = hashlib.sha256(
        f"weights:{seed}:{config_version}:{shard_index}:{input_size}".encode("ascii")
    ).digest()
    return _u64_words(material, input_size * input_size).reshape((input_size, input_size)) & MASK


def forward_summary(
    *,
    seed: int,
    config_version: int,
    shard_index: int,
    shard_count: int,
    input_size: int,
    input_scale: int,
    items: dict[str, int | float | bool | str],
) -> tuple[str, list[str]]:
    """Run a small integer-only NumPy forward pass and return a stable digest."""

    keys = sorted(items)
    x = deterministic_input(seed, input_size, shard_index)
    weights = deterministic_weights(seed, config_version, shard_index, input_size)
    shard_word = np.uint64((shard_index + 1) * 0x9E3779B97F4A7C15 & ((1 << 64) - 1))
    for key in keys:
        scalar = np.uint64(scalar_to_u64(items[key]) & ((1 << 64) - 1))
        x = (x * MULTIPLIER + scalar + shard_word + INCREMENT) & MASK
    y = (x @ weights + x) & MASK
    y = ((y + np.uint64(input_scale)) * MULTIPLIER + INCREMENT) & MASK

    envelope = {
        "version": config_version,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "input_size": input_size,
        "input_scale": input_scale,
        "keys": keys,
    }
    digest = hashlib.sha256()
    digest.update(canonical_json(envelope).encode("utf-8"))
    digest.update(b"\0")
    digest.update(y.astype("<u8", copy=False).tobytes())
    return digest.hexdigest(), keys


def aggregate_output_summaries(shard_summaries: list[str]) -> str:
    return sha256_hex(canonical_json({"shards": shard_summaries}))


def request_semantic_digest(payload: dict[str, Any]) -> str:
    semantic = {
        "protocol": payload.get("protocol"),
        "op": payload.get("op"),
        "request_id": payload.get("request_id"),
        "task_id": payload.get("task_id"),
        "node_id": payload.get("node_id"),
        "config_version": payload.get("config_version"),
        "shard_index": payload.get("shard_index"),
        "shard_count": payload.get("shard_count"),
        "items": payload.get("items"),
        "input": payload.get("input"),
    }
    return sha256_hex(canonical_json(semantic))
