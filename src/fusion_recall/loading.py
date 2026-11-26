from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import is_nonneg_int, rational_pair
from .errors import E_INPUT_INVALID, Failure
from .hashing import canonical_hash
from .vocab import Vocabulary


def _read_lines(path: str | Path | None = None, lines: Iterable[str] | None = None) -> list[str]:
    if lines is not None:
        return list(lines)
    if path is None:
        raise ValueError("either path or lines is required")
    return Path(path).read_text(encoding="utf-8").splitlines()


def _failure(source: str, line_number: int | None, message: str, item_id: str | None = None) -> Failure:
    return Failure(E_INPUT_INVALID, source, line_number, message, item_id)


def validate_document(
    raw: Any,
    config: dict[str, Any],
    vocab: Vocabulary,
    *,
    source: str = "documents",
    line_number: int | None = None,
    seen_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, Failure | None]:
    item_id: str | None = None
    if not isinstance(raw, dict):
        return None, _failure(source, line_number, "document must be a JSON object")
    maybe_id = raw.get("doc_id")
    item_id = maybe_id if isinstance(maybe_id, str) else None
    if not isinstance(maybe_id, str) or not maybe_id:
        return None, _failure(source, line_number, "doc_id must be a non-empty string")
    if seen_ids is not None and maybe_id in seen_ids:
        return None, _failure(source, line_number, f"duplicate doc_id: {maybe_id}", maybe_id)
    version = raw.get("version")
    if not is_nonneg_int(version):
        return None, _failure(source, line_number, "version must be a non-negative integer", maybe_id)

    raw_views = raw.get("views")
    if not isinstance(raw_views, dict):
        return None, _failure(source, line_number, "views must be an object", maybe_id)

    normalized_views: dict[str, Any] = {}
    view_names = config["view_names"]
    allowed_languages = set(config["allowed_languages"])
    for view_name in view_names:
        raw_view = raw_views.get(view_name)
        if not isinstance(raw_view, dict):
            return None, _failure(source, line_number, f"missing view: {view_name}", maybe_id)
        language = raw_view.get("language")
        if language not in allowed_languages:
            return None, _failure(
                source,
                line_number,
                f"language {language!r} is not in declared set {sorted(allowed_languages)}",
                maybe_id,
            )
        raw_fields = raw_view.get("fields")
        if not isinstance(raw_fields, dict):
            return None, _failure(source, line_number, f"view {view_name}.fields must be an object", maybe_id)
        fields: dict[str, list[str]] = {}
        for field_name in config["field_order"]:
            raw_terms = raw_fields.get(field_name, [])
            if not isinstance(raw_terms, list) or not all(isinstance(term, str) for term in raw_terms):
                return None, _failure(
                    source,
                    line_number,
                    f"field {field_name} must be a list of term strings",
                    maybe_id,
                )
            concepts: list[str] = []
            for term in raw_terms:
                concept = vocab.resolve(term)
                if concept is None:
                    return None, _failure(
                        source,
                        line_number,
                        f"term {term!r} is not in controlled vocabulary",
                        maybe_id,
                    )
                concepts.append(concept)
            fields[field_name] = concepts
        unknown_fields = set(raw_fields) - set(config["field_order"])
        if unknown_fields:
            return None, _failure(
                source,
                line_number,
                f"unknown fields: {sorted(unknown_fields)}",
                maybe_id,
            )
        normalized_views[view_name] = {"language": language, "fields": fields}

    extra_views = set(raw_views) - set(view_names)
    if extra_views:
        return None, _failure(source, line_number, f"unknown views: {sorted(extra_views)}", maybe_id)

    document = {
        "doc_id": maybe_id,
        "version": version,
        "views": normalized_views,
    }
    document["document_fingerprint"] = canonical_hash(document)
    if seen_ids is not None:
        seen_ids.add(maybe_id)
    return document, None


def load_documents(
    config: dict[str, Any],
    vocab: Vocabulary,
    *,
    path: str | Path | None = None,
    lines: Iterable[str] | None = None,
    source: str = "documents",
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(_read_lines(path, lines), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(_failure(source, line_number, f"invalid JSON: {exc.msg}").to_dict())
            continue
        document, failure = validate_document(
            raw,
            config,
            vocab,
            source=source,
            line_number=line_number,
            seen_ids=seen_ids,
        )
        if failure is not None:
            failures.append(failure.to_dict())
        else:
            documents.append(document)
    documents.sort(key=lambda item: item["doc_id"])
    return {
        "documents": documents,
        "failures": failures,
        "loaded": len(documents),
        "invalid": len(failures),
    }


def validate_query(
    raw: Any,
    config: dict[str, Any],
    vocab: Vocabulary,
    *,
    source: str = "queries",
    line_number: int | None = None,
    seen_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, Failure | None]:
    if not isinstance(raw, dict):
        return None, _failure(source, line_number, "query must be a JSON object")
    query_id = raw.get("query_id")
    item_id = query_id if isinstance(query_id, str) else None
    if not isinstance(query_id, str) or not query_id:
        return None, _failure(source, line_number, "query_id must be a non-empty string", item_id)
    if seen_ids is not None and query_id in seen_ids:
        return None, _failure(source, line_number, f"duplicate query_id: {query_id}", query_id)
    terms = raw.get("terms")
    if not isinstance(terms, list) or not terms or not all(isinstance(term, str) for term in terms):
        return None, _failure(source, line_number, "query terms must be a non-empty string list", query_id)
    normalized_terms: list[str] = []
    for term in terms:
        concept = vocab.resolve(term)
        if concept is None:
            return None, _failure(
                source,
                line_number,
                f"query term {term!r} is not in controlled vocabulary or is empty",
                query_id,
            )
        normalized_terms.append(concept)

    preference_raw = raw.get("language_preference", {})
    if not isinstance(preference_raw, dict):
        return None, _failure(source, line_number, "language_preference must be an object", query_id)
    preference: dict[str, str] = {}
    preference_rational: dict[str, list[int]] = {}
    for view_name in config["view_names"]:
        numerator, denominator = rational_pair(preference_raw.get(view_name, 1))
        preference[view_name] = f"{numerator}/{denominator}"
        preference_rational[view_name] = [numerator, denominator]
    extra = set(preference_raw) - set(config["view_names"])
    if extra:
        return None, _failure(source, line_number, f"unknown preference views: {sorted(extra)}", query_id)

    query = {
        "query_id": query_id,
        "terms": normalized_terms,
        "language_preference": preference,
        "language_preference_rational": preference_rational,
    }
    query["query_fingerprint"] = canonical_hash(
        {"query_id": query_id, "terms": normalized_terms, "preference": preference_rational}
    )
    if seen_ids is not None:
        seen_ids.add(query_id)
    return query, None


def load_queries(
    config: dict[str, Any],
    vocab: Vocabulary,
    *,
    path: str | Path | None = None,
    lines: Iterable[str] | None = None,
    source: str = "queries",
) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(_read_lines(path, lines), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(_failure(source, line_number, f"invalid JSON: {exc.msg}").to_dict())
            continue
        query, failure = validate_query(
            raw,
            config,
            vocab,
            source=source,
            line_number=line_number,
            seen_ids=seen_ids,
        )
        if failure is not None:
            failures.append(failure.to_dict())
        else:
            queries.append(query)
    queries.sort(key=lambda item: item["query_id"])
    return {
        "queries": queries,
        "failures": failures,
        "loaded": len(queries),
        "invalid": len(failures),
    }
