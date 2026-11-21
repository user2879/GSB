from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_json, number_token


ENTRY_SEP = "\x1f"
STABLE_SCHEMA_VERSION = 1


def entry_key(source: str, target: str) -> str:
    return f"{source}{ENTRY_SEP}{target}"


def split_entry_key(key: str) -> tuple[str, str]:
    source, target = key.split(ENTRY_SEP, 1)
    return source, target


@dataclass(frozen=True)
class SourceSentence:
    sentence_id: str
    tokens: tuple[str, ...]
    batch: int

    @property
    def stable_key(self) -> tuple[int, str]:
        return (self.batch, self.sentence_id)

    @property
    def content_hash(self) -> str:
        payload = {"tokens": list(self.tokens)}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sentence_id,
            "tokens": list(self.tokens),
            "batch": self.batch,
        }


@dataclass(frozen=True)
class LexiconEntry:
    source: str
    target: str
    frequency: float
    alignment_confidence: float

    @property
    def key(self) -> str:
        return entry_key(self.source, self.target)

    def fingerprint(self) -> str:
        payload = {
            "source": self.source,
            "target": self.target,
            "frequency": number_token(self.frequency),
            "alignment_confidence": number_token(self.alignment_confidence),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "frequency": self.frequency,
            "alignment_confidence": self.alignment_confidence,
            "key": self.key,
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True)
class ScoreParams:
    alignment_weight: float
    order_weight: float
    length_penalty: float
    fusion_temperature: float
    confidence_threshold: float
    fixed_point_bits: int
    quantization_tier: int
    allowed_fixed_bits: tuple[int, ...] = (8, 12, 16, 20, 24)
    allowed_quantization_tiers: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    @property
    def scale(self) -> int:
        return 1 << self.fixed_point_bits

    @property
    def quant_step(self) -> int:
        return max(1, self.scale // self.quantization_tier)

    def fingerprint(self) -> str:
        payload = {
            "alignment_weight": number_token(self.alignment_weight),
            "order_weight": number_token(self.order_weight),
            "length_penalty": number_token(self.length_penalty),
            "fusion_temperature": number_token(self.fusion_temperature),
            "confidence_threshold": number_token(self.confidence_threshold),
            "fixed_point_bits": self.fixed_point_bits,
            "quantization_tier": self.quantization_tier,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def with_field(self, name: str, value: float | int) -> "ScoreParams":
        data = self.to_dict(include_enums=True)
        data[name] = value
        return ScoreParams.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ScoreParams":
        names = {
            "alignment_weight": ("alignment_weight", "word_alignment_weight"),
            "order_weight": ("order_weight", "word_order_weight"),
            "length_penalty": ("length_penalty", "length_penalty_coefficient"),
            "fusion_temperature": ("fusion_temperature", "temperature"),
            "confidence_threshold": ("confidence_threshold", "threshold"),
            "fixed_point_bits": ("fixed_point_bits", "fixed_bits", "fractional_bits"),
            "quantization_tier": ("quantization_tier", "quant_tier"),
        }

        def pick(name: str) -> Any:
            for key in names[name]:
                if key in data:
                    return data[key]
            raise KeyError(name)

        allowed_bits = tuple(data.get("allowed_fixed_bits", (8, 12, 16, 20, 24)))
        allowed_tiers = tuple(data.get("allowed_quantization_tiers", (1, 2, 4, 8, 16, 32)))
        return cls(
            alignment_weight=float(pick("alignment_weight")),
            order_weight=float(pick("order_weight")),
            length_penalty=float(pick("length_penalty")),
            fusion_temperature=float(pick("fusion_temperature")),
            confidence_threshold=float(pick("confidence_threshold")),
            fixed_point_bits=int(pick("fixed_point_bits")),
            quantization_tier=int(pick("quantization_tier")),
            allowed_fixed_bits=tuple(int(x) for x in allowed_bits),
            allowed_quantization_tiers=tuple(int(x) for x in allowed_tiers),
        )

    def to_dict(self, include_enums: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "alignment_weight": self.alignment_weight,
            "order_weight": self.order_weight,
            "length_penalty": self.length_penalty,
            "fusion_temperature": self.fusion_temperature,
            "confidence_threshold": self.confidence_threshold,
            "fixed_point_bits": self.fixed_point_bits,
            "quantization_tier": self.quantization_tier,
        }
        if include_enums:
            payload["allowed_fixed_bits"] = list(self.allowed_fixed_bits)
            payload["allowed_quantization_tiers"] = list(self.allowed_quantization_tiers)
        return payload

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in ("alignment_weight", "order_weight", "length_penalty", "fusion_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value):
                errors.append(f"{name} must be finite")
        if self.alignment_weight < 0:
            errors.append("alignment_weight must be non-negative")
        if self.order_weight < 0:
            errors.append("order_weight must be non-negative")
        if self.length_penalty < 0:
            errors.append("length_penalty must be non-negative")
        if self.fusion_temperature <= 0:
            errors.append("fusion_temperature must be positive")
        if not math.isfinite(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            errors.append("confidence_threshold must be finite and in [0, 1]")
        if self.fixed_point_bits not in self.allowed_fixed_bits:
            errors.append(
                f"fixed_point_bits must be one of {sorted(self.allowed_fixed_bits)}"
            )
        if self.quantization_tier not in self.allowed_quantization_tiers:
            errors.append(
                "quantization_tier must be one of "
                f"{sorted(self.allowed_quantization_tiers)}"
            )
        return errors


@dataclass(frozen=True)
class CompileConfig:
    ngram_order: int
    max_sentences_per_batch: int
    thread_pool_concurrency: int
    decode_points_per_batch: int
    snapshot_commit_interval_batches: int
    max_sentence_tokens: int
    vocabulary: tuple[str, ...]
    candidate_limit: int = 8

    def fingerprint(self) -> str:
        payload = {
            "ngram_order": self.ngram_order,
            "max_sentences_per_batch": self.max_sentences_per_batch,
            "thread_pool_concurrency": self.thread_pool_concurrency,
            "decode_points_per_batch": self.decode_points_per_batch,
            "snapshot_commit_interval_batches": self.snapshot_commit_interval_batches,
            "max_sentence_tokens": self.max_sentence_tokens,
            "vocabulary": list(self.vocabulary),
            "candidate_limit": self.candidate_limit,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def result_fingerprint(self) -> str:
        """Compilation fields that can change a sentence result.

        Concurrency, batch budgets, snapshot intervals and batch size limits
        affect scheduling/validation only and therefore never enter cache keys.
        """
        payload = {
            "ngram_order": self.ngram_order,
            "candidate_limit": self.candidate_limit,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ngram_order": self.ngram_order,
            "max_sentences_per_batch": self.max_sentences_per_batch,
            "thread_pool_concurrency": self.thread_pool_concurrency,
            "decode_points_per_batch": self.decode_points_per_batch,
            "snapshot_commit_interval_batches": self.snapshot_commit_interval_batches,
            "max_sentence_tokens": self.max_sentence_tokens,
            "vocabulary": list(self.vocabulary),
            "candidate_limit": self.candidate_limit,
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 1 <= self.ngram_order <= 6:
            errors.append("ngram_order must be in [1, 6]")
        if self.max_sentences_per_batch <= 0:
            errors.append("max_sentences_per_batch must be positive")
        if not 1 <= self.thread_pool_concurrency <= 64:
            errors.append("thread_pool_concurrency must be in [1, 64]")
        if self.decode_points_per_batch <= 0:
            errors.append("decode_points_per_batch must be positive")
        if self.snapshot_commit_interval_batches <= 0:
            errors.append("snapshot_commit_interval_batches must be positive")
        if self.max_sentence_tokens <= 0:
            errors.append("max_sentence_tokens must be positive")
        if not self.vocabulary:
            errors.append("vocabulary must not be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            errors.append("vocabulary contains duplicates")
        if self.candidate_limit <= 0:
            errors.append("candidate_limit must be positive")
        return errors


@dataclass
class TranslationState:
    sentences: dict[str, SourceSentence] = field(default_factory=dict)
    lexicon: dict[str, LexiconEntry] = field(default_factory=dict)
    params: ScoreParams | None = None
    config: CompileConfig | None = None

    def clone(self) -> "TranslationState":
        return copy.deepcopy(self)

    def lex_bucket(self, source: str) -> tuple[LexiconEntry, ...]:
        entries = [entry for entry in self.lexicon.values() if entry.source == source]
        entries.sort(
            key=lambda entry: (
                -entry.alignment_confidence,
                -entry.frequency,
                entry.target,
            )
        )
        return tuple(entries)

    def referenced_entry_keys(self, tokens: Iterable[str]) -> tuple[str, ...]:
        keys: set[str] = set()
        for token in tokens:
            for entry in self.lex_bucket(token):
                keys.add(entry.key)
        return tuple(sorted(keys))

    def referenced_sources(self, tokens: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted({token for token in tokens if self.lex_bucket(token)}))

    def lexicon_hash(self) -> str:
        payload = [
            {
                "source": entry.source,
                "target": entry.target,
                "frequency": number_token(entry.frequency),
                "alignment_confidence": number_token(entry.alignment_confidence),
            }
            for entry in sorted(self.lexicon.values(), key=lambda item: item.key)
        ]
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def source_hash(self) -> str:
        payload = [self.sentences[key].to_dict() for key in sorted(self.sentences)]
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def state_fingerprint(self) -> str:
        payload = {
            "sentences": [self.sentences[key].to_dict() for key in sorted(self.sentences)],
            "lexicon": [
                {
                    "source": entry.source,
                    "target": entry.target,
                    "frequency": number_token(entry.frequency),
                    "alignment_confidence": number_token(entry.alignment_confidence),
                }
                for entry in sorted(self.lexicon.values(), key=lambda item: item.key)
            ],
            "params": self.params.to_dict(include_enums=True) if self.params else None,
            "config": self.config.to_dict() if self.config else None,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentences": [
                self.sentences[key].to_dict() for key in sorted(self.sentences)
            ],
            "lexicon": [
                {
                    "source": entry.source,
                    "target": entry.target,
                    "frequency": entry.frequency,
                    "alignment_confidence": entry.alignment_confidence,
                }
                for entry in sorted(self.lexicon.values(), key=lambda item: item.key)
            ],
            "params": self.params.to_dict(include_enums=True) if self.params else None,
            "config": self.config.to_dict() if self.config else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationState":
        sentences = {
            item["id"]: SourceSentence(
                sentence_id=item["id"],
                tokens=tuple(item["tokens"]),
                batch=int(item["batch"]),
            )
            for item in data.get("sentences", [])
        }
        lexicon = {
            entry_key(item["source"], item["target"]): LexiconEntry(
                source=item["source"],
                target=item["target"],
                frequency=float(item["frequency"]),
                alignment_confidence=float(item["alignment_confidence"]),
            )
            for item in data.get("lexicon", [])
        }
        params = ScoreParams.from_mapping(data["params"]) if data.get("params") else None
        config_data = data.get("config")
        config = None
        if config_data:
            config = CompileConfig(
                ngram_order=int(config_data["ngram_order"]),
                max_sentences_per_batch=int(config_data["max_sentences_per_batch"]),
                thread_pool_concurrency=int(config_data["thread_pool_concurrency"]),
                decode_points_per_batch=int(config_data["decode_points_per_batch"]),
                snapshot_commit_interval_batches=int(
                    config_data["snapshot_commit_interval_batches"]
                ),
                max_sentence_tokens=int(config_data["max_sentence_tokens"]),
                vocabulary=tuple(config_data["vocabulary"]),
                candidate_limit=int(config_data.get("candidate_limit", 8)),
            )
        return cls(sentences=sentences, lexicon=lexicon, params=params, config=config)


@dataclass
class Failure:
    code: str
    phase: str
    message: str
    file: str = ""
    line: int = 0
    sentence_id: str | None = None
    entry_key: str | None = None
    injection_id: str | None = None
    command_id: str | None = None
    run_id: int | None = None
    active: bool = True
    resolved_run_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    order: tuple[Any, ...] = field(default_factory=tuple)

    def sort_key(self) -> tuple[Any, ...]:
        if self.order:
            return self.order + (
                self.code,
                self.sentence_id or "",
                self.entry_key or "",
                self.run_id if self.run_id is not None else -1,
                self.command_id or "",
                self.message,
            )
        return (
            self.code,
            self.sentence_id or "",
            self.entry_key or "",
            self.file,
            self.line,
            self.run_id if self.run_id is not None else -1,
            self.command_id or "",
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "phase": self.phase,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "sentence_id": self.sentence_id,
            "entry_key": self.entry_key,
            "injection_id": self.injection_id,
            "command_id": self.command_id,
            "run_id": self.run_id,
            "active": self.active,
        }
        if self.resolved_run_id is not None:
            payload["resolved_run_id"] = self.resolved_run_id
        if self.details:
            payload["details"] = self.details
        if self.order:
            payload["order"] = list(self.order)
        return payload


def stable_failures(failures: Iterable[Failure]) -> list[Failure]:
    return sorted(failures, key=lambda item: item.sort_key())
