from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .data import Document
from .normalization import token_script_language


def _weighted_tokens(title_tokens: Iterable[str], body_tokens: Iterable[str], title_repeat: int) -> list[str]:
    weighted: list[str] = []
    for _ in range(title_repeat):
        weighted.extend(title_tokens)
    weighted.extend(body_tokens)
    return weighted


class BM25Index:
    def __init__(self, *, k1: float, b: float, title_repeat: int, languages: Iterable[str]) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.title_repeat = int(title_repeat)
        self.languages = tuple(sorted(languages))
        self.postings: dict[str, dict[str, dict[str, int]]] = {lang: {} for lang in self.languages}
        self.doc_lengths: dict[str, dict[str, int]] = {lang: {} for lang in self.languages}
        self.document_languages: dict[str, str] = {}

    def add_document(self, document: Document, *languages: str) -> None:
        effective_languages = languages or (document.declared_language,)
        if len(effective_languages) == 1 and effective_languages[0] in self.languages:
            weighted = _weighted_tokens(document.title_tokens, document.body_tokens, self.title_repeat)
            targets = {effective_languages[0]: weighted}
        else:
            targets: dict[str, list[str]] = defaultdict(list)
            for token in _weighted_tokens(document.title_tokens, document.body_tokens, self.title_repeat):
                lang = token_script_language(token)
                if lang in self.languages:
                    targets[lang].append(token)
        for lang in sorted(targets):
            tokens = targets[lang]
            if not tokens:
                continue
            counts = Counter(tokens)
            self.doc_lengths[lang][document.doc_id] = len(tokens)
            for term, frequency in sorted(counts.items()):
                self.postings[lang].setdefault(term, {})[document.doc_id] = int(frequency)
            if lang not in self.document_languages:
                self.document_languages[lang] = document.doc_id

    def posting_map(self, language: str, term: str) -> dict[str, int]:
        return self.postings.get(language, {}).get(term, {})

    def document_frequency(self, language: str, term: str) -> int:
        return len(self.posting_map(language, term))

    def document_count(self, language: str) -> int:
        return len(self.doc_lengths.get(language, {}))

    def average_document_length(self, language: str) -> float:
        lengths = self.doc_lengths.get(language, {})
        if not lengths:
            return 0.0
        return sum(lengths.values()) / len(lengths)

    def score(self, language: str, term: str, doc_id: str) -> float:
        posting = self.postings.get(language, {}).get(term)
        if not posting or doc_id not in posting:
            return 0.0
        n_docs = self.document_count(language)
        df = len(posting)
        if n_docs <= 0 or df <= 0:
            return 0.0
        avgdl = self.average_document_length(language)
        dl = self.doc_lengths[language][doc_id]
        tf = float(posting[doc_id])
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl if avgdl else 0.0))
        return idf * (tf * (self.k1 + 1.0)) / denominator if denominator else 0.0

    def summary(self) -> dict:
        result = {}
        for lang in self.languages:
            postings = self.postings.get(lang, {})
            result[lang] = {
                "documents": self.document_count(lang),
                "unique_terms": len(postings),
                "postings": sum(len(value) for value in postings.values()),
                "average_document_length": round(self.average_document_length(lang), 12),
                "df_terms": {term: len(value) for term, value in sorted(postings.items())},
            }
        return result

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "title_repeat": self.title_repeat,
            "languages": list(self.languages),
            "postings": {
                lang: {
                    term: {doc_id: int(freq) for doc_id, freq in sorted(docs.items())}
                    for term, docs in sorted(terms.items())
                }
                for lang, terms in sorted(self.postings.items())
            },
            "doc_lengths": {
                lang: {doc_id: int(length) for doc_id, length in sorted(lengths.items())}
                for lang, lengths in sorted(self.doc_lengths.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict) -> "BM25Index":
        index = cls(k1=value["k1"], b=value["b"], title_repeat=value["title_repeat"], languages=value["languages"])
        index.postings = {
            lang: {
                term: {doc_id: int(freq) for doc_id, freq in docs.items()}
                for term, docs in terms.items()
            }
            for lang, terms in value.get("postings", {}).items()
        }
        index.doc_lengths = {
            lang: {doc_id: int(length) for doc_id, length in lengths.items()}
            for lang, lengths in value.get("doc_lengths", {}).items()
        }
        return index

