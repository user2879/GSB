"""BM25 retrieval with NumPy vectorized scoring and deterministic ordering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fingerprint import fingerprint
from .models import Document, RetrievalConfig
from .normalize import tokenize
from .rules import RuleTable


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    score: float
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class RetrievalResult:
    key: str
    effective_query: str
    query_terms: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    doc_set: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_query": self.effective_query,
            "query_terms": list(self.query_terms),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "doc_set": list(self.doc_set),
        }


class BM25Index:
    """Immutable index built from the normalized document corpus."""

    def __init__(
        self,
        docs: list[Document],
        config: RetrievalConfig,
        rules: RuleTable,
    ):
        self.docs = list(docs)
        self.config = config
        self.rules = rules
        self.doc_ids = [doc.doc_id for doc in docs]
        norm_rules = config.normalization
        self.doc_terms: list[list[str]] = []
        self.title_terms: list[set[str]] = []
        vocab: dict[str, int] = {}
        for doc in docs:
            body_tokens = rules.apply_synonyms(
                tokenize(f"{doc.title} {doc.body}", norm_rules)
            )
            title_tokens = set(
                rules.apply_synonyms(tokenize(doc.title, norm_rules))
            )
            terms = rules.expand_categories(body_tokens)
            self.doc_terms.append(terms)
            self.title_terms.append(title_tokens)
            for term in terms:
                if term not in vocab:
                    vocab[term] = len(vocab)
        self.vocab = vocab
        n_docs = max(len(docs), 1)
        tf = np.zeros((n_docs, len(vocab)), dtype=np.float64)
        df = np.zeros(len(vocab), dtype=np.float64)
        doc_lengths = np.zeros(n_docs, dtype=np.float64)
        for row, terms in enumerate(self.doc_terms):
            doc_lengths[row] = len(terms)
            seen: set[str] = set()
            for term in terms:
                col = vocab[term]
                tf[row, col] += 1.0
                if term not in seen:
                    df[col] += 1.0
                    seen.add(term)
        k1 = float(config.bm25["k1"])
        b = float(config.bm25["b"])
        # Robust idf that stays non-negative for terms in every document.
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.avgdl = float(doc_lengths.mean()) if docs else 0.0
        denom = tf + k1 * (
            1.0 - b + b * doc_lengths[:, None] / max(self.avgdl, 1e-9)
        )
        self.tf_component = tf * (k1 + 1.0) / np.maximum(denom, 1e-12)
        self.bm25_weight = float(config.fusion_weights.get("bm25_weight", 1.0))
        self.title_weight = float(config.fusion_weights.get("title_weight", 0.0))
        self.overlap_weight = float(
            config.fusion_weights.get("overlap_weight", 0.0)
        )
        self._fingerprint = fingerprint(
            {
                "docs": [doc.to_dict() for doc in docs],
                "normalization": norm_rules,
                "synonyms": list(rules.synonyms.items()),
                "category_expand": [
                    [key, values]
                    for key, values in sorted(rules.category_expand.items())
                ],
                "rules_version": rules.version,
            }
        )

    @property
    def content_fingerprint(self) -> str:
        return self._fingerprint

    def query_terms(self, effective_query: str) -> list[str]:
        tokens = self.rules.apply_synonyms(
            tokenize(effective_query, self.config.normalization)
        )
        return self.rules.expand_categories(tokens)

    def search(
        self,
        key: str,
        effective_query: str,
        top_k: int,
    ) -> RetrievalResult:
        terms = self.query_terms(effective_query)
        unique_terms = list(dict.fromkeys(terms))
        cols = [self.vocab[term] for term in unique_terms if term in self.vocab]
        if not cols or not self.docs:
            return RetrievalResult(
                key=key,
                effective_query=effective_query,
                query_terms=tuple(unique_terms),
                candidates=(),
                doc_set=(),
            )
        col_array = np.asarray(cols, dtype=np.int64)
        bm25_scores = (
            self.idf[col_array] * self.tf_component[:, col_array]
        ).sum(axis=1)
        # Title and overlap fusion features, fully deterministic.
        title_hit = np.zeros(len(self.docs), dtype=np.float64)
        query_vocab = set(unique_terms)
        for row, title in enumerate(self.title_terms):
            hit_terms = query_vocab & title
            if hit_terms:
                title_hit[row] = len(hit_terms) / max(len(query_vocab), 1)
        matched_mask = self.tf_component[:, col_array] > 0.0
        overlap = matched_mask.sum(axis=1) / max(len(query_vocab), 1)
        raw_scores = (
            self.bm25_weight * bm25_scores
            + self.title_weight * title_hit
            + self.overlap_weight * overlap
        )
        order = sorted(
            range(len(self.docs)),
            key=lambda row: (-raw_scores[row], self.doc_ids[row]),
        )
        candidates: list[Candidate] = []
        for row in order[: max(int(top_k), 0)]:
            if raw_scores[row] <= 0.0:
                break
            matched = tuple(
                term
                for term, col in zip(unique_terms, col_array)
                if self.tf_component[row, col] > 0.0
            )
            candidates.append(
                Candidate(
                    doc_id=self.doc_ids[row],
                    score=float(raw_scores[row]),
                    matched_terms=matched,
                )
            )
        return RetrievalResult(
            key=key,
            effective_query=effective_query,
            query_terms=tuple(unique_terms),
            candidates=tuple(candidates),
            doc_set=tuple(candidate.doc_id for candidate in candidates),
        )

    def get_doc(self, doc_id: str) -> Document:
        for doc in self.docs:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(doc_id)


def bm25_param_fingerprint(config: RetrievalConfig) -> str:
    return fingerprint(
        {
            "bm25": config.bm25,
            "fusion_weights": config.fusion_weights,
            "recall_top_k": config.recall_top_k,
        }
    )


def is_close_score(left: float, right: float, tol: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tol, abs_tol=tol)
