"""Answer extraction and citation assembly stages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import E_CITATION_MISSING
from .models import Failure
from .normalize import tokenize
from .retrieval import BM25Index, RetrievalResult
from .rules import RuleTable
from .models import RetrievalConfig

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


@dataclass(frozen=True)
class Snippet:
    doc_id: str
    source: str
    sentence_index: int
    text: str
    score: float
    doc_score: float
    backfilled: bool
    backfill_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "sentence_index": self.sentence_index,
            "text": self.text,
            "score": round(self.score, 6),
            "doc_score": round(self.doc_score, 6),
            "backfilled": self.backfilled,
            "backfill_reason": self.backfill_reason,
        }


@dataclass(frozen=True)
class ExtractionResult:
    key: str
    snippets: tuple[Snippet, ...]
    referenced_doc_set: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snippets": [snippet.to_dict() for snippet in self.snippets],
            "referenced_doc_set": list(self.referenced_doc_set),
        }


@dataclass(frozen=True)
class CitationResult:
    key: str
    answer: str
    answer_available: bool
    citations: tuple[Snippet, ...]
    backfilled: tuple[dict[str, Any], ...]
    referenced_doc_set: tuple[str, ...]
    failure: Failure | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_available": self.answer_available,
            "citations": [snippet.to_dict() for snippet in self.citations],
            "backfilled": list(self.backfilled),
            "referenced_doc_set": list(self.referenced_doc_set),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


def _sentence_score(
    sentence: str,
    query_terms: tuple[str, ...],
    doc_score: float,
    max_doc_score: float,
    config: RetrievalConfig,
    rules: RuleTable,
) -> float:
    tokens = set(
        rules.apply_synonyms(
            tokenize(sentence, config.normalization)
        )
    )
    if not query_terms:
        term_overlap = 0.0
    else:
        term_overlap = sum(1 for term in query_terms if term in tokens) / len(
            query_terms
        )
    normalized_doc = doc_score / max_doc_score if max_doc_score > 0 else 0.0
    sentence_weight = float(
        config.fusion_weights.get("sentence_weight", 1.0)
    )
    doc_balance = float(config.fusion_weights.get("doc_balance_weight", 0.2))
    return sentence_weight * term_overlap + doc_balance * normalized_doc


def extract_snippets(
    key: str,
    index: BM25Index,
    retrieval: RetrievalResult,
    config: RetrievalConfig,
    rules: RuleTable,
) -> ExtractionResult:
    window = max(int(config.answer_window), 1)
    max_doc_score = (
        max(candidate.score for candidate in retrieval.candidates)
        if retrieval.candidates
        else 0.0
    )
    ranked: list[Snippet] = []
    for candidate in retrieval.candidates:
        doc = index.get_doc(candidate.doc_id)
        sentences = split_sentences(doc.body)
        for sentence_index, sentence in enumerate(sentences):
            score = _sentence_score(
                sentence,
                retrieval.query_terms,
                candidate.score,
                max_doc_score,
                config,
                rules,
            )
            ranked.append(
                Snippet(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    sentence_index=sentence_index,
                    text=sentence,
                    score=score,
                    doc_score=candidate.score,
                    backfilled=False,
                    backfill_reason="",
                )
            )
    ranked.sort(
        key=lambda snippet: (
            -snippet.score,
            snippet.doc_id,
            snippet.sentence_index,
        )
    )
    chosen = tuple(ranked[:window])
    referenced = tuple(
        dict.fromkeys(snippet.doc_id for snippet in chosen)
    )
    return ExtractionResult(
        key=key,
        snippets=chosen,
        referenced_doc_set=referenced,
    )


def assemble_citations(
    key: str,
    extraction: ExtractionResult,
    retrieval: RetrievalResult,
    index: BM25Index,
    config: RetrievalConfig,
    rules: RuleTable,
    template_name: str,
    turn_scope: str,
) -> CitationResult:
    chosen = list(extraction.snippets)
    chosen_ids = {
        (snippet.doc_id, snippet.sentence_index) for snippet in chosen
    }
    backfilled: list[dict[str, Any]] = []
    min_citations = int(config.min_citations)
    if len(chosen) < min_citations:
        # Backfill from remaining sentences in score-descending order.
        max_doc_score = (
            max(candidate.score for candidate in retrieval.candidates)
            if retrieval.candidates
            else 0.0
        )
        remaining: list[Snippet] = []
        for candidate in retrieval.candidates:
            doc = index.get_doc(candidate.doc_id)
            for sentence_index, sentence in enumerate(split_sentences(doc.body)):
                if (doc.doc_id, sentence_index) in chosen_ids:
                    continue
                score = _sentence_score(
                    sentence,
                    retrieval.query_terms,
                    candidate.score,
                    max_doc_score,
                    config,
                    rules,
                )
                remaining.append(
                    Snippet(
                        doc_id=doc.doc_id,
                        source=doc.source,
                        sentence_index=sentence_index,
                        text=sentence,
                        score=score,
                        doc_score=candidate.score,
                        backfilled=True,
                        backfill_reason="score_desc_backfill",
                    )
                )
        remaining.sort(
            key=lambda snippet: (
                -snippet.score,
                snippet.doc_id,
                snippet.sentence_index,
            )
        )
        for snippet in remaining:
            if len(chosen) >= min_citations:
                break
            chosen.append(snippet)
            chosen_ids.add((snippet.doc_id, snippet.sentence_index))
            backfilled.append(
                {
                    "doc_id": snippet.doc_id,
                    "sentence_index": snippet.sentence_index,
                    "reason": snippet.backfill_reason,
                }
            )
    if not chosen:
        failure = Failure(
            code=E_CITATION_MISSING,
            scope=turn_scope,
            message=(
                "no citable snippet produced from retrieval candidates"
            ),
            sort_key=turn_scope,
        )
        return CitationResult(
            key=key,
            answer="",
            answer_available=False,
            citations=(),
            backfilled=(),
            referenced_doc_set=(),
            failure=failure,
        )
    top = chosen[0]
    sources = list(dict.fromkeys(snippet.source for snippet in chosen))
    answer = rules.render_answer(
        template_name,
        query=retrieval.effective_query,
        top_snippet=top.text,
        citation_count=len(chosen),
        top_doc_id=top.doc_id,
        sources=sources,
    )
    referenced = tuple(
        dict.fromkeys(snippet.doc_id for snippet in chosen)
    )
    return CitationResult(
        key=key,
        answer=answer,
        answer_available=True,
        citations=tuple(chosen),
        backfilled=tuple(backfilled),
        referenced_doc_set=referenced,
        failure=None,
    )
