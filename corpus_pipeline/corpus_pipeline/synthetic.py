from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import Code, fail
from .models import Document, DomainSpec


def derive_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big", signed=False)


def domain_vocabulary(domain: DomainSpec) -> list[str]:
    # Vocabulary is a pure function of the declaration and is stable across
    # process runs, Python hash seeds, and append operations.
    return [f"{domain.id}_subword_{index:04d}" for index in range(domain.vocab_size)]


def subwords_from_terms(terms: Iterable[str]) -> frozenset[str]:
    """Model an external subword tokenizer with stable lexical units."""

    return frozenset(terms)


def _counts_from_payload(payload: Mapping[str, Any], doc_id: str) -> tuple[dict[str, int], int]:
    raw_counts = payload.get("term_counts", payload.get("terms"))
    if raw_counts is not None and not isinstance(raw_counts, Mapping):
        fail(Code.INVALID_EDIT, "replacement term_counts must be an object", doc_id=doc_id)
    if isinstance(raw_counts, Mapping):
        counts: dict[str, int] = {}
        for term, count in raw_counts.items():
            if not isinstance(term, str) or not term:
                fail(Code.INVALID_EDIT, "replacement term must be a non-empty string", doc_id=doc_id)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                fail(Code.INVALID_EDIT, "replacement term count must be a positive integer", doc_id=doc_id, term=term)
            counts[term] = count
        if not counts:
            fail(Code.INVALID_EDIT, "replacement term_counts must be non-empty", doc_id=doc_id)
        return counts, sum(counts.values())

    tokens: Any = payload.get("tokens")
    if tokens is None and isinstance(payload.get("text"), str):
        tokens = payload["text"].split()
    if not isinstance(tokens, (list, tuple)) or not tokens:
        fail(
            Code.INVALID_EDIT,
            "replace_document requires term_counts, tokens, or non-empty text",
            doc_id=doc_id,
        )
    counts: dict[str, int] = {}
    for token in tokens:
        if not isinstance(token, str) or not token:
            fail(Code.INVALID_EDIT, "replacement token must be a non-empty string", doc_id=doc_id)
        counts[token] = counts.get(token, 0) + 1
    return counts, len(tokens)


def replacement_document(domain: DomainSpec, payload: Mapping[str, Any]) -> Document:
    doc_id = payload.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        fail(Code.INVALID_EDIT, "replace_document requires doc_id", domain=domain.id)
    counts, length = _counts_from_payload(payload, doc_id)
    supplied_subwords = payload.get("subwords")
    if supplied_subwords is None:
        subwords = subwords_from_terms(counts.keys())
    else:
        if not isinstance(supplied_subwords, (list, tuple, set, frozenset)):
            fail(Code.INVALID_EDIT, "replacement subwords must be a list", doc_id=doc_id)
        subwords = frozenset(str(item) for item in supplied_subwords if isinstance(item, str) and item)
        if not subwords:
            fail(Code.INVALID_EDIT, "replacement subwords must be non-empty", doc_id=doc_id)
    return Document(
        domain=domain.id,
        doc_id=doc_id,
        term_counts=counts,
        length=length,
        subwords=subwords,
        batch_index=int(payload.get("batch_index", -1)),
        generator_seed=int(payload.get("generator_seed", -1)),
        source="edited",
    ).with_content_hash()


def _allocate_counts(rng: np.random.Generator, length: int, unique_count: int) -> list[int]:
    if unique_count >= length:
        return [1] * length
    remaining = length - unique_count
    if remaining == 0:
        return [1] * unique_count
    concentration = rng.uniform(0.35, 1.25, size=unique_count)
    probabilities = concentration / concentration.sum()
    extras = rng.multinomial(remaining, probabilities)
    return [1 + int(value) for value in extras]


def _new_document_counts(
    domain: DomainSpec,
    rng: np.random.Generator,
    vocabulary: Sequence[str],
    *,
    source_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    length = int(rng.integers(domain.length_min, domain.length_max + 1))
    if source_counts:
        source = dict(source_counts)
        source_terms = list(source.keys())
        mutate = max(1, min(len(source_terms) // 7, len(source_terms) - 2))
        rng.shuffle(source_terms)
        removed = source_terms[:mutate]
        kept = [term for term in source_terms[mutate:]]
        candidates = [term for term in vocabulary if term not in source]
        if len(candidates) < mutate:
            mutate = len(candidates)
            removed = removed[:mutate]
        added = list(rng.choice(candidates, size=mutate, replace=False)) if mutate else []
        new_counts: dict[str, int] = {term: source[term] for term in kept}
        for term in added:
            new_counts[term] = 1
        delta = sum(source.values()) - sum(new_counts.values())
        if delta > 0 and new_counts:
            target = kept[int(rng.integers(0, len(kept)))] if kept else next(iter(new_counts))
            new_counts[target] += delta
        elif delta < 0:
            # Removing noisy terms can leave a short document.  Restore exact
            # length with deterministic repetition at a surviving term.
            target = sorted(new_counts.keys())[0]
            new_counts[target] += -delta
        return {term: int(count) for term, count in sorted(new_counts.items()) if count > 0}

    unique_count = int(rng.integers(max(4, min(8, length)), min(domain.vocab_size, length) + 1))
    terms = list(rng.choice(vocabulary, size=unique_count, replace=False))
    amounts = _allocate_counts(rng, length, unique_count)
    return {term: amount for term, amount in zip(sorted(terms), amounts, strict=True)}


def generate_documents(
    domain: DomainSpec,
    count: int,
    *,
    start_index: int = 0,
    batch_index: int = 0,
    existing_documents: Sequence[Document] = (),
    append_seed: int | None = None,
) -> list[Document]:
    if count < 0:
        fail(Code.INVALID_EDIT, "batch count must be non-negative", domain=domain.id, count=count)
    if count == 0:
        return []
    vocabulary = domain_vocabulary(domain)
    effective_seed = domain.seed if append_seed is None else append_seed
    seed = derive_seed(domain.seed, domain.id, "batch", batch_index, effective_seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    history: list[Document] = list(existing_documents)
    docs: list[Document] = []
    for offset in range(count):
        global_index = start_index + offset
        source_counts: Mapping[str, int] | None = None
        if (
            domain.duplicate_every > 0
            and global_index > 0
            and global_index % domain.duplicate_every == 0
            and history
        ):
            source_counts = history[-1].term_counts
        counts = _new_document_counts(domain, rng, vocabulary, source_counts=source_counts)
        length = sum(counts.values())
        doc = Document(
            domain=domain.id,
            doc_id=f"{domain.id}-{global_index + 1:05d}",
            term_counts=counts,
            length=length,
            subwords=subwords_from_terms(counts.keys()),
            batch_index=batch_index,
            generator_seed=effective_seed,
            source="generated",
        ).with_content_hash()
        history.append(doc)
        docs.append(doc)
    return docs


def generate_domain_streams(domain_specs: Iterable[DomainSpec]) -> dict[str, list[Document]]:
    return {
        domain.id: generate_documents(domain, domain.batch_size, batch_index=0)
        for domain in domain_specs
    }


def append_batch(domain: DomainSpec, documents: Sequence[Document], count: int, append_seed: int) -> list[Document]:
    return generate_documents(
        domain,
        count,
        start_index=len(documents),
        batch_index=max((doc.batch_index for doc in documents), default=-1) + 1,
        existing_documents=documents,
        append_seed=append_seed,
    )


def document_stream_hash(documents: Mapping[str, Sequence[Document]]) -> str:
    payload = {
        domain: [
            {
                "doc_id": doc.doc_id,
                "content_hash": doc.content_hash,
                "batch_index": doc.batch_index,
                "generator_seed": doc.generator_seed,
            }
            for doc in sorted(items, key=lambda item: item.doc_id)
        ]
        for domain, items in sorted(documents.items())
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
