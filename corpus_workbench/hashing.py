"""Deterministic PRNG, tokenisation, shingles and MinHash signatures."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from .config import DedupConfig
from .util import canonical_json, sha256_json, sha256_text

MASK64 = (1 << 64) - 1
MINHASH_PRIME = (1 << 61) - 1
ALGORITHM_VERSION = "minhash-affine-v1"
TOKENIZER_VERSION = "unicode-word-casefold-v1"

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class StableRandom:
    """SplitMix64 with platform-independent output."""

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def stable_u64(value: str | bytes, namespace: str = "") -> int:
    digest = hashlib.blake2b(
        value.encode("utf-8") if isinstance(value, str) else value,
        digest_size=8,
        person=b"cw-u64-v1",
    ).digest()
    if namespace:
        digest = hashlib.blake2b(digest, digest_size=8, person=namespace.encode("utf-8")[:16]).digest()
    return int.from_bytes(digest, "big")


def source_seed(source: str, salt: str) -> int:
    return stable_u64(f"{source}\x00{salt}", namespace="cw-seed")


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def shingle_tokens(tokens: list[str], shingle_length: int) -> set[tuple[str, ...]]:
    if shingle_length <= 0:
        raise ValueError("shingle_length must be positive")
    if not tokens:
        return set()
    if len(tokens) < shingle_length:
        return {tuple(tokens)}
    return {
        tuple(tokens[index : index + shingle_length])
        for index in range(0, len(tokens) - shingle_length + 1)
    }


def shingle_hashes(text: str, shingle_length: int) -> frozenset[int]:
    tokens = tokenize(text)
    shingles = shingle_tokens(tokens, shingle_length)
    if not shingles:
        raise ValueError("document has no tokens after tokenisation")
    return frozenset(stable_u64("\x1f".join(shingle), namespace="cw-sh") for shingle in shingles)


@lru_cache(maxsize=128)
def permutation_family(signature_dim: int) -> tuple[tuple[int, int], ...]:
    if signature_dim <= 0:
        raise ValueError("signature_dim must be positive")
    coefficients: list[tuple[int, int]] = []
    for index in range(signature_dim):
        raw_a = hashlib.blake2b(
            f"{ALGORITHM_VERSION}\x00a\x00{index}".encode("utf-8"),
            digest_size=16,
            person=b"cw-perm-v1",
        ).digest()
        raw_b = hashlib.blake2b(
            f"{ALGORITHM_VERSION}\x00b\x00{index}".encode("utf-8"),
            digest_size=16,
            person=b"cw-perm-v1",
        ).digest()
        a = int.from_bytes(raw_a, "big") % (MINHASH_PRIME - 1) + 1
        b = int.from_bytes(raw_b, "big") % MINHASH_PRIME
        coefficients.append((a, b))
    return tuple(coefficients)


def compute_signature_from_shingles(shingles: frozenset[int] | set[int], signature_dim: int) -> tuple[int, ...]:
    if not shingles:
        raise ValueError("cannot compute a signature from an empty shingle set")
    values = [value % MINHASH_PRIME for value in shingles]
    signature: list[int] = []
    for a, b in permutation_family(signature_dim):
        signature.append(min((a * value + b) % MINHASH_PRIME for value in values))
    return tuple(signature)


def compute_signature(text: str, config: DedupConfig) -> tuple[tuple[int, ...], int]:
    hashes = shingle_hashes(text, config.shingle_length)
    return compute_signature_from_shingles(hashes, config.signature_dim), len(hashes)


def signature_cache_key(text: str, config: DedupConfig) -> str:
    return sha256_json(
        {
            "algorithm": ALGORITHM_VERSION,
            "tokenizer": TOKENIZER_VERSION,
            "content_fingerprint": sha256_text(text),
            "shingle_length": config.shingle_length,
            "signature_dim": config.signature_dim,
        }
    )


def group_decision_cache_key(
    member_fingerprints: list[str],
    weights: dict[str, float],
    config: DedupConfig,
) -> str:
    return sha256_json(
        {
            "algorithm": "representative-decision-v1",
            "member_fingerprints": member_fingerprints,
            "weights": {name: weights[name] for name in sorted(weights)},
            "shingle_length": config.shingle_length,
            "signature_dim": config.signature_dim,
            "jaccard_threshold": config.jaccard_threshold,
            "representative_rule": config.representative_rule,
        }
    )


def minhash_similarity(left: tuple[int, ...] | list[int], right: tuple[int, ...] | list[int]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def canonical_cache_material(value: object) -> str:
    return canonical_json(value)
