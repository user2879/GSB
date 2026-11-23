"""Windowed MinHash, signed random projection buckets, and similarities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .canonical import stable_key
from .config import FingerprintConfig
from .features import NormalizedText
from .fixedpoint import format_fixed, similarity_units


def _u64(seed_material: str) -> int:
    return int.from_bytes(hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest(), "big")


@dataclass(frozen=True)
class SignatureResult:
    signature: tuple[int, ...]
    layout: tuple[tuple[int, int], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "signature": list(self.signature),
            "layout": [{"window": window, "lane": lane} for window, lane in self.layout],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SignatureResult":
        return cls(
            signature=tuple(int(value) for value in payload["signature"]),
            layout=tuple((int(item["window"]), int(item["lane"])) for item in payload["layout"]),
        )


def _window_lane_counts(window_count: int, dimension: int) -> list[int]:
    base, remainder = divmod(dimension, window_count)
    return [base + (1 if index < remainder else 0) for index in range(window_count)]


def compute_minhash(normalized: NormalizedText, config: FingerprintConfig) -> SignatureResult:
    dimension = config.signature_dimension
    counts = _window_lane_counts(config.window_count, dimension)
    signature: list[int] = []
    layout: list[tuple[int, int]] = []
    global_lane = 0
    for window_index, (window, lane_count) in enumerate(zip(normalized.windows, counts)):
        for _ in range(lane_count):
            minimum: int | None = None
            for token in window:
                value = _u64(f"{config.hash_seed}|{config.algorithm_fingerprint}|minhash|{global_lane}|{token}")
                if minimum is None or value < minimum:
                    minimum = value
            if minimum is None:
                minimum = (1 << 64) - 1
            signature.append(minimum)
            layout.append((window_index, global_lane))
            global_lane += 1
    if len(signature) != dimension:
        raise AssertionError("signature layout must match configured dimension")
    return SignatureResult(tuple(signature), tuple(layout))


def signed_projection_bucket(signature: tuple[int, ...], config: FingerprintConfig) -> str:
    bits: list[str] = []
    for bit_index in range(config.bucket_bits):
        projection = 0
        for component_index, component in enumerate(signature):
            weight = 1 if _u64(
                f"{config.hash_seed}|{config.bucket_fingerprint}|projection|{bit_index}|{component_index}"
            ) & 1 else -1
            projection += weight * component
        bits.append("1" if projection >= 0 else "0")
    return "".join(bits)


def pair_similarity(
    signature_a: tuple[int, ...],
    signature_b: tuple[int, ...],
    config: FingerprintConfig,
) -> dict[str, Any]:
    if len(signature_a) != len(signature_b) or len(signature_a) != config.signature_dimension:
        raise ValueError("signature vectors must share the configured dimension")
    matches = sum(1 for left, right in zip(signature_a, signature_b) if left == right)
    units = similarity_units(
        matches,
        config.signature_dimension,
        config.scale,
        config.quantization_levels,
    )
    return {
        "matches": matches,
        "dimension": config.signature_dimension,
        "similarity_units": units,
        "similarity_fixed": format_fixed(units, config.fixed_decimal_places),
        "boundary": units == config.threshold_units,
        "qualifies": units >= config.threshold_units,
    }


def signature_task_key(text_id: str) -> str:
    return stable_key(text_id, "signature")


def similarity_task_key(text_id: str, representative_id: str) -> str:
    return stable_key(text_id, "similarity", representative_id)
