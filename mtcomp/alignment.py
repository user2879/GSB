from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any

from .cache import SentenceKeys
from .canonical import (
    fixed_div,
    fixed_from_float,
    fixed_log2,
    fixed_mul,
    sha256_value,
)
from .model import SourceSentence, TranslationState


@dataclass(frozen=True)
class CandidateSpan:
    start: int
    length: int
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    used_entry_keys: tuple[str, ...]
    confidence_sum_fixed: int
    frequency_sum: float
    alignment_score_fixed: int
    order_score_fixed: int
    length_score_fixed: int
    score_fixed: int
    source_chars: int
    target_chars: int

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def signature(self) -> str:
        return hashlib.sha256(
            (
                f"{self.start}:{self.length}:"
                + "\x1f".join(self.target_tokens)
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "length": self.length,
            "source_tokens": list(self.source_tokens),
            "target_tokens": list(self.target_tokens),
            "used_entry_keys": list(self.used_entry_keys),
            "confidence_sum_fixed": self.confidence_sum_fixed,
            "frequency_sum": self.frequency_sum,
            "alignment_score_fixed": self.alignment_score_fixed,
            "order_score_fixed": self.order_score_fixed,
            "length_score_fixed": self.length_score_fixed,
            "score_fixed": self.score_fixed,
            "source_chars": self.source_chars,
            "target_chars": self.target_chars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateSpan":
        return cls(
            start=int(data["start"]),
            length=int(data["length"]),
            source_tokens=tuple(data["source_tokens"]),
            target_tokens=tuple(data["target_tokens"]),
            used_entry_keys=tuple(data["used_entry_keys"]),
            confidence_sum_fixed=int(data["confidence_sum_fixed"]),
            frequency_sum=float(data["frequency_sum"]),
            alignment_score_fixed=int(data["alignment_score_fixed"]),
            order_score_fixed=int(data["order_score_fixed"]),
            length_score_fixed=int(data["length_score_fixed"]),
            score_fixed=int(data["score_fixed"]),
            source_chars=int(data["source_chars"]),
            target_chars=int(data["target_chars"]),
        )


@dataclass(frozen=True)
class AlignmentResult:
    sentence_id: str
    alignment_key: str
    tokens: tuple[str, ...]
    candidates: tuple[CandidateSpan, ...]
    referenced_entry_keys: tuple[str, ...]
    referenced_sources: tuple[str, ...]
    candidate_signature: str
    missing_tokens: tuple[str, ...]

    def candidates_from(self, start: int) -> tuple[CandidateSpan, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.start == start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "alignment_key": self.alignment_key,
            "tokens": list(self.tokens),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "referenced_entry_keys": list(self.referenced_entry_keys),
            "referenced_sources": list(self.referenced_sources),
            "candidate_signature": self.candidate_signature,
            "missing_tokens": list(self.missing_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentResult":
        return cls(
            sentence_id=data["sentence_id"],
            alignment_key=data["alignment_key"],
            tokens=tuple(data["tokens"]),
            candidates=tuple(
                CandidateSpan.from_dict(item) for item in data["candidates"]
            ),
            referenced_entry_keys=tuple(data["referenced_entry_keys"]),
            referenced_sources=tuple(data["referenced_sources"]),
            candidate_signature=data["candidate_signature"],
            missing_tokens=tuple(data.get("missing_tokens", [])),
        )


def _quantized_weights(state: TranslationState) -> tuple[int, int, int, int, int]:
    assert state.params is not None
    params = state.params
    bits = params.fixed_point_bits
    tier = params.quantization_tier
    return (
        1 << bits,
        fixed_from_float(params.alignment_weight, bits, tier),
        fixed_from_float(params.order_weight, bits, tier),
        fixed_from_float(params.length_penalty, bits, tier),
        fixed_from_float(params.fusion_temperature, bits, tier),
    )


def _candidate_sort_key(candidate: CandidateSpan) -> tuple[Any, ...]:
    return (
        candidate.start,
        -candidate.confidence_sum_fixed,
        -candidate.frequency_sum,
        candidate.target_tokens,
        candidate.length,
        candidate.used_entry_keys,
    )


def build_alignment(
    state: TranslationState,
    sentence: SourceSentence,
    keys: SentenceKeys,
) -> AlignmentResult:
    assert state.params is not None and state.config is not None
    params = state.params
    config = state.config
    bits = params.fixed_point_bits
    tier = params.quantization_tier
    scale, align_weight, order_weight, length_penalty, temperature = _quantized_weights(
        state
    )
    buckets: dict[str, tuple[Any, ...]] = {
        token: state.lex_bucket(token) for token in set(sentence.tokens)
    }
    missing_tokens = tuple(sorted(token for token, bucket in buckets.items() if not bucket))
    candidates: list[CandidateSpan] = []
    source_len = len(sentence.tokens)
    log_ngram_denominator = max(1, fixed_log2(config.ngram_order + 1, bits, tier))

    for start in range(source_len):
        max_length = min(config.ngram_order, source_len - start)
        for span_length in range(1, max_length + 1):
            span_tokens = sentence.tokens[start : start + span_length]
            span_buckets = [buckets[token] for token in span_tokens]
            if any(not bucket for bucket in span_buckets):
                continue
            per_bucket_limit = config.candidate_limit if span_length <= 2 else min(2, config.candidate_limit)
            combo_buckets = [bucket[:per_bucket_limit] for bucket in span_buckets]
            seen_targets: set[tuple[str, ...]] = set()
            combinations = itertools.product(*combo_buckets)
            for combination_index, combination in enumerate(combinations):
                if combination_index >= max(512, config.candidate_limit * 16):
                    break
                target_tokens = tuple(entry.target for entry in combination)
                if target_tokens in seen_targets:
                    continue
                seen_targets.add(target_tokens)
                confidence_sum = sum(
                    fixed_from_float(entry.alignment_confidence, bits, tier)
                    for entry in combination
                )
                frequency_sum = 0.0
                for entry in combination:
                    frequency_sum = min(1.0e300, frequency_sum + entry.frequency)
                log_conf_sum = 0
                for entry in combination:
                    log_conf_sum += fixed_log2(
                        max(entry.alignment_confidence, 1.0 / scale), bits, tier
                    )
                mean_log_conf = fixed_div(log_conf_sum, span_length, bits)
                alignment_score = fixed_mul(align_weight, mean_log_conf, bits)

                span_ratio = fixed_div(
                    fixed_log2(span_length, bits, tier),
                    log_ngram_denominator,
                    bits,
                )
                order_score = fixed_mul(order_weight, span_ratio, bits)

                source_chars = sum(len(token) for token in span_tokens)
                target_chars = sum(len(token) for token in target_tokens)
                char_gap = abs(target_chars - source_chars)
                char_ratio = fixed_div(
                    fixed_from_float(
                        char_gap / max(1, source_chars), bits, tier
                    ),
                    scale,
                    bits,
                )
                length_score = -fixed_mul(length_penalty, char_ratio, bits)

                total = 0
                total += alignment_score
                total += order_score
                total += length_score
                fused = fixed_div(total, max(1, temperature), bits)
                candidate = CandidateSpan(
                    start=start,
                    length=span_length,
                    source_tokens=span_tokens,
                    target_tokens=target_tokens,
                    used_entry_keys=tuple(entry.key for entry in combination),
                    confidence_sum_fixed=confidence_sum,
                    frequency_sum=frequency_sum,
                    alignment_score_fixed=alignment_score,
                    order_score_fixed=order_score,
                    length_score_fixed=length_score,
                    score_fixed=fused,
                    source_chars=source_chars,
                    target_chars=target_chars,
                )
                candidates.append(candidate)

    candidates.sort(key=_candidate_sort_key)
    candidate_signature = sha256_value(
        [
            {
                "start": item.start,
                "length": item.length,
                "target": list(item.target_tokens),
                "score": item.score_fixed,
            }
            for item in candidates
        ]
    )
    return AlignmentResult(
        sentence_id=sentence.sentence_id,
        alignment_key=keys.alignment_key,
        tokens=sentence.tokens,
        candidates=tuple(candidates),
        referenced_entry_keys=tuple(key for key, _ in keys.referenced_entries),
        referenced_sources=keys.referenced_sources,
        candidate_signature=candidate_signature,
        missing_tokens=missing_tokens,
    )
