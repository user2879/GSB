"""Deterministic term normalization, permutation, and window partitioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from .canonical import sha256_json, sha256_text, stable_key
from .config import FingerprintConfig
from .normalization_rules import normalize_token
from .text_store import TextRecord


class StableRandom:
    """SHA-256 counter DRBG; no process-global random state is consulted."""

    def __init__(self, seed_material: str):
        self._seed = seed_material.encode("utf-8")
        self._counter = 0

    def next_u64(self) -> int:
        digest = hashlib.sha256(self._seed + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        return int.from_bytes(digest[:8], "big")

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def stable_permutation(items: Sequence[str], seed_material: str) -> list[str]:
    result = list(items)
    rng = StableRandom(seed_material)
    for index in range(len(result) - 1, 0, -1):
        swap = rng.randbelow(index + 1)
        result[index], result[swap] = result[swap], result[index]
    return result


@dataclass(frozen=True)
class NormalizedText:
    text_id: str
    normalized_terms: tuple[str, ...]
    permuted_terms: tuple[str, ...]
    windows: tuple[tuple[str, ...], ...]
    padding_positions: tuple[tuple[int, int, str], ...]
    window_size: int
    permutation_seed: str

    def payload(self) -> dict[str, Any]:
        return {
            "text_id": self.text_id,
            "normalized_terms": list(self.normalized_terms),
            "permuted_terms": list(self.permuted_terms),
            "windows": [list(window) for window in self.windows],
            "padding_positions": [
                {"window": window, "position": position, "token": token}
                for window, position, token in self.padding_positions
            ],
            "window_size": self.window_size,
            "permutation_seed": self.permutation_seed,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NormalizedText":
        return cls(
            text_id=payload["text_id"],
            normalized_terms=tuple(payload["normalized_terms"]),
            permuted_terms=tuple(payload["permuted_terms"]),
            windows=tuple(tuple(window) for window in payload["windows"]),
            padding_positions=tuple(
                (item["window"], item["position"], item["token"]) for item in payload["padding_positions"]
            ),
            window_size=int(payload["window_size"]),
            permutation_seed=str(payload["permutation_seed"]),
        )

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.payload())


def normalize_and_window(record: TextRecord, config: FingerprintConfig) -> NormalizedText:
    normalized_terms = [normalize_token(term) for term in record.terms]
    if not normalized_terms or any(not term for term in normalized_terms):
        raise ValueError(f"text {record.text_id!r} has an empty normalized term")
    permutation_seed = sha256_text(
        stable_key(record.text_id, config.algorithm_fingerprint, "permutation")
    )
    permuted = stable_permutation(normalized_terms, permutation_seed)
    window_count = config.window_count
    window_size = max(1, (len(permuted) + window_count - 1) // window_count)
    padded_total = window_count * window_size
    padded = list(permuted)
    padding_positions: list[tuple[int, int, str]] = []
    for flat_index in range(len(permuted), padded_total):
        window_index, position = divmod(flat_index, window_size)
        token = f"<PAD:{window_index}:{position}>"
        padded.append(token)
        padding_positions.append((window_index, position, token))
    windows = tuple(
        tuple(padded[start : start + window_size])
        for start in range(0, len(padded), window_size)
    )
    return NormalizedText(
        text_id=record.text_id,
        normalized_terms=tuple(normalized_terms),
        permuted_terms=tuple(permuted),
        windows=windows,
        padding_positions=tuple(padding_positions),
        window_size=window_size,
        permutation_seed=permutation_seed,
    )
