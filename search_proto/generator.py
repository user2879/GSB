from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .model import FIELD_ORDER, ControlledVocabulary, Document, QuerySpec, StopwordTable
from .util import canonical_json, sha256_json, write_json


VOCABULARY_TERMS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "index",
    "search",
    "vector",
    "score",
    "field",
    "query",
    "the",
    "of",
    "and",
    "to",
)


def default_config() -> dict[str, Any]:
    return {
        "schema_version": "search-proto-v1",
        "fields": {
            "title": {"weight": 2.0},
            "body": {"weight": 1.0},
        },
        "bm25": {"k1": 1.2, "b": 0.75},
        "fixed": {"decimal_places": 3, "quantum": 1},
        "allowed_decimal_places": [2, 3, 4, 5],
        "allowed_quantums": [1, 2, 5, 10],
        "top_k": 5,
        "max_workers": 4,
        "batch_point_budget": 64,
        "per_query_point_quota": 192,
        "stage_tolerance_units": 0.5,
    }


def _term(rng: np.random.Generator, terms: tuple[str, ...]) -> str:
    return terms[int(rng.integers(0, len(terms)))]


def _make_document(
    rng: np.random.Generator,
    index: int,
    terms: tuple[str, ...],
) -> Document:
    title_length = 3 + int(rng.integers(0, 4))
    body_length = 8 + int(rng.integers(0, 9))
    title = [_term(rng, terms) for _ in range(title_length)]
    body = [_term(rng, terms) for _ in range(body_length)]

    if index % 2 == 0:
        title[min(1, len(title) - 1)] = "of"
    if index % 3 == 0:
        body[min(2, len(body) - 1)] = "the"
    if index % 4 == 0:
        body[min(5, len(body) - 1)] = "and"
    if index % 5 == 0:
        body[-1] = "to"
    if index % 7 == 0:
        title[min(2, len(title) - 1)] = "alpha"

    marks: dict[str, list[int]] = {}
    if index % 6 == 0:
        marks["body"] = [1]
    if index % 11 == 0:
        marks["title"] = [1]

    return Document.create(
        doc_id=f"D{index + 1:04d}",
        version=index % 3,
        fields={"title": title, "body": body},
        stopword_marks=marks,
    )


def _make_query(
    rng: np.random.Generator,
    index: int,
    terms: tuple[str, ...],
) -> QuerySpec:
    count = 2 + int(rng.integers(0, 3))
    indices = sorted(
        {int(value) for value in rng.integers(0, len(terms), size=count)}
    )
    while len(indices) < count:
        value = int(rng.integers(0, len(terms)))
        if value not in indices:
            indices.append(value)
    query_terms = [terms[value] for value in indices]
    preference_mode = index % 4
    if preference_mode == 0:
        preferences = ["title"]
    elif preference_mode == 1:
        preferences = ["body"]
    else:
        preferences = list(FIELD_ORDER)
    return QuerySpec.create(f"Q{index + 1:04d}", query_terms, preferences)


def _doc_payload(document: Document) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "version": document.version,
        "fields": {
            field_name: list(document.fields.get(field_name, ()))
            for field_name in FIELD_ORDER
            if field_name in document.fields
        },
        "field_stopword_marks": {
            field_name: list(document.stopword_marks.get(field_name, ()))
            for field_name in FIELD_ORDER
            if document.stopword_marks.get(field_name)
        },
    }


def generate_samples(
    out_dir: Path,
    *,
    seed: int = 20261126,
    document_count: int = 24,
    query_count: int = 8,
) -> dict[str, Any]:
    if document_count <= 0 or query_count <= 0:
        raise ValueError("document_count and query_count must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    terms = VOCABULARY_TERMS
    documents = [
        _make_document(rng, index, terms) for index in range(document_count)
    ]
    queries = [_make_query(rng, index, terms) for index in range(query_count)]

    stopwords = StopwordTable.create(
        "sw-v1",
        global_terms=(),
        by_field={
            "title": ["of", "the"],
            "body": ["the", "and", "to"],
        },
    )
    vocabulary = ControlledVocabulary.create("vocab-v1", terms)
    write_json(out_dir / "config.json", default_config())
    write_json(out_dir / "controlled_vocab.json", vocabulary.to_dict())
    write_json(out_dir / "stopwords.json", stopwords.to_dict())

    with (out_dir / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(canonical_json(_doc_payload(document)) + "\n")
    with (out_dir / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            handle.write(canonical_json(query.to_dict()) + "\n")

    replacement_source = documents[min(1, document_count - 1)]
    replacement = Document.create(
        doc_id=replacement_source.doc_id,
        version=replacement_source.version + 1,
        fields={
            "title": ["alpha", "beta", "search", "of", "vector"],
            "body": ["alpha", "score", "index", "and", "query", "phi", "the"],
        },
        stopword_marks={"body": [2]},
    )
    appended_one = Document.create(
        "D0101",
        0,
        {
            "title": ["search", "index", "alpha", "of"],
            "body": ["vector", "score", "search", "and", "beta", "the", "query"],
        },
    )
    appended_two = Document.create(
        "D0102",
        2,
        {
            "title": ["alpha", "search", "vector"],
            "body": ["index", "query", "score", "theta", "the", "phi"],
        },
    )
    deleted_doc_id = documents[min(6, document_count - 1)].doc_id
    edits = [
        {"op": "append_doc", "doc": _doc_payload(appended_one)},
        {"op": "append_doc", "doc": _doc_payload(appended_two)},
        {"op": "replace_doc", "doc": _doc_payload(replacement)},
        {"op": "delete_doc", "doc_id": deleted_doc_id},
        {"op": "set_field_weights", "weights": {"title": 2.5, "body": 0.8}},
        {"op": "set_bm25", "k1": 1.6, "b": 0.65},
        {"op": "add_stopword", "field": "title", "term": "zeta"},
        {"op": "remove_stopword", "field": "body", "term": "and"},
        {"op": "set_fixed", "decimal_places": 4, "quantum": 2},
        {"op": "set_fixed", "decimal_places": 2, "quantum": 10},
        {"op": "set_top_k", "top_k": 6},
    ]
    with (out_dir / "edits.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for edit in edits:
            handle.write(canonical_json(edit) + "\n")

    manifest = {
        "seed": seed,
        "document_count": document_count,
        "query_count": query_count,
        "vocabulary_version": vocabulary.version,
        "stopword_version": stopwords.version,
        "files": {
            name: sha256_json(
                (out_dir / name).read_text(encoding="utf-8").splitlines()
            )
            for name in (
                "config.json",
                "controlled_vocab.json",
                "stopwords.json",
                "documents.jsonl",
                "queries.jsonl",
                "edits.jsonl",
            )
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
