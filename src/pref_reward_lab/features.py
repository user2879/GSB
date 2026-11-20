from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .canonical import fingerprint
from .config import LabConfig
from .errors import LabError
from .models import FeatureResult, RawPair

WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class FeatureParams:
    tokenizer_version: str
    precision_tier: str
    hash_buckets: int
    feature_dim: int
    normalize_text: bool

    @classmethod
    def from_config(
        cls, config: LabConfig, *, precision_tier: str, hash_buckets: int, feature_dim: int | None = None
    ) -> "FeatureParams":
        return cls(
            tokenizer_version=config.tokenizer_version,
            precision_tier=precision_tier,
            hash_buckets=hash_buckets,
            feature_dim=config.feature_dim if feature_dim is None else feature_dim,
            normalize_text=config.normalize_text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_version": self.tokenizer_version,
            "precision_tier": self.precision_tier,
            "hash_buckets": self.hash_buckets,
            "feature_dim": self.feature_dim,
            "normalize_text": self.normalize_text,
        }


def normalize_text(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def tokenize_text(text: str, params: FeatureParams) -> list[str]:
    normalized = normalize_text(text, params.normalize_text)
    words = WORD_PATTERN.findall(normalized)
    tokens: list[str] = []
    for word in words:
        tokens.append(word)
        if params.tokenizer_version == "subword_v2" and len(word) > 2:
            tokens.extend(f"@{word[index:index + 2]}" for index in range(len(word) - 1))
        if params.precision_tier in {"balanced", "fine"} and len(word) > 2:
            tokens.extend(f"^{word[index:index + 2]}" for index in range(len(word) - 1))
        if params.precision_tier == "fine" and len(word) > 3:
            tokens.extend(f"#{word[index:index + 3]}" for index in range(len(word) - 2))
    return tokens[: params.feature_dim]


def _bucket_for_token(segment: str, token: str, params: FeatureParams) -> int:
    material = (
        f"{params.tokenizer_version}|{params.precision_tier}|{segment}|{token}"
    ).encode("utf-8")
    digest = hashlib.blake2b(
        material,
        digest_size=8,
        person=b"rewardlab",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % params.hash_buckets


def _count_tokens(tokens: list[str], segment: str, params: FeatureParams) -> dict[int, int]:
    counts: dict[int, int] = {}
    for token in tokens:
        bucket = _bucket_for_token(segment, token, params)
        counts[bucket] = counts.get(bucket, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def extract_pair_features(pair: RawPair, params: FeatureParams) -> FeatureResult:
    if params.hash_buckets <= 0:
        raise LabError(
            "E_HASH_BUCKETS_ZERO",
            "hash bucket count must be positive during feature extraction",
            {"pair_id": pair.pair_id, "stage": "features"},
        )
    prompt_tokens = tokenize_text(pair.prompt, params)
    chosen_tokens = tokenize_text(pair.chosen, params)
    rejected_tokens = tokenize_text(pair.rejected, params)
    prompt_counts = _count_tokens(prompt_tokens, "prompt", params)
    chosen_counts = _count_tokens(chosen_tokens, "chosen", params)
    rejected_counts = _count_tokens(rejected_tokens, "rejected", params)
    combined: dict[int, int] = {}
    for counts in (prompt_counts, chosen_counts, rejected_counts):
        for bucket, count in counts.items():
            combined[bucket] = combined.get(bucket, 0) + count
    precision_fingerprint = fingerprint(
        {
            "feature_params": params.to_dict(),
            "tokenizer_contract": 2,
        },
        prefix="feature-params:",
    )
    return FeatureResult(
        pair_id=pair.pair_id,
        buckets={key: combined[key] for key in sorted(combined)},
        prompt_buckets=prompt_counts,
        chosen_buckets=chosen_counts,
        rejected_buckets=rejected_counts,
        prompt_tokens=len(prompt_tokens),
        chosen_tokens=len(chosen_tokens),
        rejected_tokens=len(rejected_tokens),
        precision_fingerprint=precision_fingerprint,
    )


def feature_cache_key(
    pair: RawPair,
    params: FeatureParams,
    *,
    source_version: str,
    injection_fingerprint: str = "",
) -> str:
    return fingerprint(
        {
            "pair": pair.content_dict(),
            "feature_params": params.to_dict(),
            "source_version": source_version,
            "injection": injection_fingerprint,
            "extractor_schema": 2,
        },
        prefix="features:",
    )

