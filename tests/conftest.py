from __future__ import annotations

from typing import Any, Iterable

import pytest

from mtcomp.engine import Engine, EngineOptions, canonical_sentence_payload
from mtcomp.model import (
    CompileConfig,
    LexiconEntry,
    ScoreParams,
    SourceSentence,
    TranslationState,
    entry_key,
)


DEFAULT_TARGETS = {
    "a": "甲",
    "b": "乙",
    "c": "丙",
    "d": "丁",
    "e": "戊",
    "f": "己",
    "the": "这",
    "red": "红",
    "cat": "猫",
    "runs": "跑",
    "fast": "快",
}


def make_state(
    sentence_specs: Iterable[tuple[str, str, int]],
    lexicon_specs: Iterable[tuple[str, str, float, float]] | None = None,
    *,
    concurrency: int = 4,
    budget: int = 1000,
    ngram: int = 2,
    max_batch: int = 10,
    max_tokens: int = 32,
    candidate_limit: int = 4,
    threshold: float = 0.5,
    commit_interval: int = 2,
) -> TranslationState:
    sentences = {
        sentence_id: SourceSentence(
            sentence_id=sentence_id,
            tokens=tuple(tokens.split()),
            batch=batch,
        )
        for sentence_id, tokens, batch in sentence_specs
    }
    if lexicon_specs is None:
        tokens = sorted({token for sentence in sentences.values() for token in sentence.tokens})
        lexicon_specs = [
            (
                token,
                DEFAULT_TARGETS.get(token, f"t_{token}"),
                10.0 + index,
                0.90,
            )
            for index, token in enumerate(tokens)
        ]
    lexicon = {
        entry_key(source, target): LexiconEntry(
            source=source,
            target=target,
            frequency=float(frequency),
            alignment_confidence=float(confidence),
        )
        for source, target, frequency, confidence in lexicon_specs
    }
    vocabulary = tuple(
        sorted({token for sentence in sentences.values() for token in sentence.tokens})
    )
    config = CompileConfig(
        ngram_order=ngram,
        max_sentences_per_batch=max_batch,
        thread_pool_concurrency=concurrency,
        decode_points_per_batch=budget,
        snapshot_commit_interval_batches=commit_interval,
        max_sentence_tokens=max_tokens,
        vocabulary=vocabulary,
        candidate_limit=candidate_limit,
    )
    params = ScoreParams(
        alignment_weight=1.1,
        order_weight=0.7,
        length_penalty=0.1,
        fusion_temperature=1.0,
        confidence_threshold=threshold,
        fixed_point_bits=12,
        quantization_tier=16,
        allowed_fixed_bits=(8, 12, 16),
        allowed_quantization_tiers=(1, 2, 4, 8, 16),
    )
    return TranslationState(
        sentences=sentences, lexicon=lexicon, params=params, config=config
    )


def run_engine(state: TranslationState, **option_kwargs: Any) -> Engine:
    engine = Engine(state, options=EngineOptions(**option_kwargs))
    engine.run_to_idle()
    return engine


def payloads(engine: Engine) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: canonical_sentence_payload(row) for row in engine.sentence_rows()
    }


@pytest.fixture
def simple_state() -> TranslationState:
    return make_state(
        [
            ("A", "the red cat", 0),
            ("B", "a red cat", 0),
            ("C", "the cat runs", 1),
        ]
    )
