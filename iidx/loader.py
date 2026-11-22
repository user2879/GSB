from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import CacheStore
from .config import NORMALIZATION_VERSION, AppConfig
from .util import failure, is_int, sha256_file, sha256_json, sort_failures


_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_term(value: str) -> str:
    """NFKC full/half-width folding, Unicode case folding and whitespace collapse."""

    folded = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE.sub(" ", folded).strip()


def normalize_shard(value: str) -> str:
    return normalize_term(value)


class DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


@dataclass(frozen=True)
class NormalizedDoc:
    doc_id: str
    shard: str
    title_terms: tuple[str, ...]
    body_terms: tuple[str, ...]
    token_terms: tuple[str, ...]
    token_count: int
    raw_fingerprint: str
    normalized_fingerprint: str
    norm_cache_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "shard": self.shard,
            "title_terms": list(self.title_terms),
            "body_terms": list(self.body_terms),
            "token_terms": list(self.token_terms),
            "token_count": self.token_count,
            "raw_fingerprint": self.raw_fingerprint,
            "normalized_fingerprint": self.normalized_fingerprint,
            "norm_cache_key": self.norm_cache_key,
        }


@dataclass(frozen=True)
class QueryDef:
    query_id: str
    terms: tuple[str, ...]
    version: str
    top_k: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "terms": list(self.terms),
            "version": self.version,
            "top_k": self.top_k,
        }


@dataclass
class InputPaths:
    data_dir: Path
    documents: Path
    queries: Path
    vocabulary: Path
    query_versions: Path
    config: Path

    def all_source_files(self) -> list[Path]:
        return [self.documents, self.queries, self.vocabulary, self.query_versions]


@dataclass
class LoadResult:
    paths: InputPaths
    documents: dict[str, NormalizedDoc]
    queries: dict[str, QueryDef]
    vocabulary: tuple[str, ...]
    declared_query_versions: tuple[str, ...]
    documents_by_shard: dict[str, tuple[NormalizedDoc, ...]]
    shard_fingerprints: dict[str, str]
    source_fingerprint: str
    input_fingerprints: dict[str, str]
    failures: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_document_count(self) -> int:
        return len(self.documents)

    @property
    def accepted_query_count(self) -> int:
        return len(self.queries)


def discover_inputs(data_dir: Path) -> InputPaths:
    data_dir = data_dir.resolve()
    document_candidates = ["documents.jsonl", "docs.jsonl", "document_stream.jsonl"]
    query_candidates = ["queries.jsonl", "query_set.jsonl", "query.jsonl"]
    vocab_candidates = ["vocabulary.txt", "vocab.txt", "vocabulary.json", "vocab.json"]
    version_candidates = [
        "query_versions.json",
        "query_version_compatibility.json",
        "versions.json",
        "compatibility.json",
    ]

    def choose(candidates: list[str], fallback: str) -> Path:
        for candidate in candidates:
            path = data_dir / candidate
            if path.exists():
                return path
        return data_dir / fallback

    return InputPaths(
        data_dir=data_dir,
        documents=choose(document_candidates, "documents.jsonl"),
        queries=choose(query_candidates, "queries.jsonl"),
        vocabulary=choose(vocab_candidates, "vocabulary.txt"),
        query_versions=choose(version_candidates, "query_versions.json"),
        config=data_dir / "config.json",
    )


def _load_vocabulary(path: Path) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    terms: list[str] = []
    if not path.exists():
        failures.append(
            failure(
                "loading",
                "E_INPUT_FILE_MISSING",
                "input:vocabulary",
                "controlled vocabulary file is missing",
                path=str(path),
            )
        )
        return (), failures
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        failures.append(
            failure(
                "loading",
                "E_INPUT_FILE_UNREADABLE",
                "input:vocabulary",
                f"controlled vocabulary could not be read: {exc}",
                path=str(path),
            )
        )
        return (), failures
    values: list[Any]
    stripped = raw.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(raw, object_pairs_hook=_strict_object)
        except (json.JSONDecodeError, DuplicateKeyError) as exc:
            failures.append(
                failure(
                    "loading",
                    "E_VOCABULARY_JSON_INVALID",
                    "vocabulary:root",
                    f"vocabulary JSON is invalid: {exc}",
                )
            )
            return (), failures
        if isinstance(parsed, dict):
            values = parsed.get("terms", parsed.get("vocabulary", []))
        else:
            values = parsed
        if not isinstance(values, list):
            failures.append(
                failure(
                    "loading",
                    "E_VOCABULARY_JSON_SHAPE",
                    "vocabulary:root",
                    "vocabulary JSON must be a list or contain a terms list",
                )
            )
            return (), failures
    else:
        values = [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for index, value in enumerate(values, start=1):
        if not isinstance(value, str):
            failures.append(
                failure(
                    "loading",
                    "E_VOCABULARY_TERM_INVALID",
                    f"vocabulary:{index:08d}",
                    "vocabulary terms must be strings",
                    value=value,
                )
            )
            continue
        normalized = normalize_term(value)
        if not normalized:
            failures.append(
                failure(
                    "loading",
                    "E_VOCABULARY_TERM_EMPTY",
                    f"vocabulary:{index:08d}",
                    "vocabulary term is empty after normalization",
                    raw=value,
                )
            )
            continue
        terms.append(normalized)
    unique_terms = tuple(sorted(set(terms)))
    if not unique_terms:
        failures.append(
            failure(
                "loading",
                "E_VOCABULARY_EMPTY",
                "vocabulary:root",
                "controlled vocabulary has no usable terms",
            )
        )
    return unique_terms, failures


def _version_names_from(value: Any) -> set[str]:
    """Return declared version names without treating arbitrary metadata as versions."""

    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {str(key) for key in value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if isinstance(item, str)}
    return set()


def _load_query_versions(path: Path) -> tuple[tuple[str, ...], list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not path.exists():
        failures.append(
            failure(
                "loading",
                "E_INPUT_FILE_MISSING",
                "input:query_versions",
                "query-stream compatibility table is missing",
                path=str(path),
            )
        )
        return (), failures, {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError, DuplicateKeyError) as exc:
        failures.append(
            failure(
                "loading",
                "E_QUERY_VERSIONS_INVALID",
                "query_versions:root",
                f"query-stream compatibility table is invalid: {exc}",
                path=str(path),
            )
        )
        return (), failures, {}
    if isinstance(raw, list):
        declared = {str(item) for item in raw}
    elif isinstance(raw, dict):
        declared: set[str] = set()
        for key in (
            "supported_versions",
            "compatible_versions",
            "declared_versions",
            "versions",
            "query_stream_versions",
            "compatibility",
        ):
            if key in raw:
                declared.update(_version_names_from(raw[key]))
        if not declared:
            declared = _version_names_from(raw)
    else:
        failures.append(
            failure(
                "loading",
                "E_QUERY_VERSIONS_SHAPE",
                "query_versions:root",
                "query-stream compatibility table must be a list or object",
            )
        )
        return (), failures, {}
    declared = {item.strip() for item in declared if item.strip()}
    if not declared:
        failures.append(
            failure(
                "loading",
                "E_QUERY_VERSIONS_EMPTY",
                "query_versions:root",
                "query-stream compatibility table declares no versions",
            )
        )
    return tuple(sorted(declared)), failures, raw


def _read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    if not path.exists():
        failures.append(
            failure(
                "loading",
                "E_INPUT_FILE_MISSING",
                f"input:{path.name}",
                f"input file is missing: {path.name}",
                path=str(path),
            )
        )
        return rows, failures
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        failures.append(
            failure(
                "loading",
                "E_INPUT_FILE_UNREADABLE",
                f"input:{path.name}",
                f"input file could not be read: {exc}",
                path=str(path),
            )
        )
        return rows, failures
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_strict_object)
        except (json.JSONDecodeError, DuplicateKeyError) as exc:
            failures.append(
                failure(
                    "loading",
                    "E_JSONL_INVALID",
                    f"{path.name}:{line_number:08d}",
                    f"invalid JSONL record: {exc}",
                    line=line_number,
                )
            )
            continue
        if not isinstance(value, dict):
            failures.append(
                failure(
                    "loading",
                    "E_JSONL_NOT_OBJECT",
                    f"{path.name}:{line_number:08d}",
                    "JSONL record must be an object",
                    line=line_number,
                )
            )
            continue
        rows.append((line_number, value))
    return rows, failures


def _terms_from_field(value: Any, field_name: str) -> tuple[list[str], str | None]:
    if not isinstance(value, list):
        return [], f"{field_name} must be a list of term strings"
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return [], f"{field_name} contains a non-string term"
        result.append(item)
    return result, None


def _normalize_document(
    row: dict[str, Any],
    vocabulary: set[str],
    config: AppConfig,
    cache: CacheStore | None,
    vocabulary_fingerprint: str,
) -> tuple[NormalizedDoc | None, list[dict[str, Any]], bool]:
    failures: list[dict[str, Any]] = []
    raw_id = row.get("doc_id", row.get("document_id", row.get("id")))
    id_aliases = [
        row[key] for key in ("doc_id", "document_id", "id") if key in row and row[key] is not None
    ]
    if len({str(value) for value in id_aliases}) > 1:
        failures.append(
            failure(
                "loading",
                "E_DOCUMENT_ID_AMBIGUOUS",
                "document:id",
                "document defines conflicting ID aliases",
                aliases=id_aliases,
            )
        )
        return None, failures, False
    if isinstance(raw_id, str):
        doc_id = raw_id.strip()
    elif is_int(raw_id):
        doc_id = str(raw_id)
    else:
        doc_id = ""
    if not doc_id:
        failures.append(
            failure(
                "loading",
                "E_DOCUMENT_ID_INVALID",
                "document:id",
                "doc_id must be a non-empty string or integer",
            )
        )
        return None, failures, False
    raw_shard = row.get("shard", row.get("shard_label", row.get("shard_tag")))
    shard_aliases = [
        row[key] for key in ("shard", "shard_label", "shard_tag") if key in row and row[key] is not None
    ]
    if len({normalize_shard(str(value)) for value in shard_aliases}) > 1:
        failures.append(
            failure(
                "loading",
                "E_SHARD_TAG_AMBIGUOUS",
                f"document:{raw_id}",
                "document defines conflicting shard labels; exactly one stable shard label is required",
                aliases=shard_aliases,
            )
        )
        return None, failures, False
    if not isinstance(raw_shard, str) or not raw_shard.strip():
        failures.append(
            failure(
                "loading",
                "E_SHARD_TAG_INVALID",
                f"document:{doc_id}",
                "each document must declare exactly one non-empty shard tag",
                doc_id=doc_id,
            )
        )
        return None, failures, False
    shard = normalize_shard(raw_shard)
    if not shard:
        failures.append(
            failure(
                "loading",
                "E_SHARD_TAG_INVALID",
                f"document:{doc_id}",
                "shard tag is empty after normalization",
                doc_id=doc_id,
            )
        )
        return None, failures, False
    raw_title, error = _terms_from_field(
        row.get("title_terms", row.get("title", [])), "title_terms"
    )
    if error:
        failures.append(
            failure(
                "loading",
                "E_DOCUMENT_TERMS_INVALID",
                f"document:{doc_id}",
                error,
                doc_id=doc_id,
            )
        )
        return None, failures, False
    raw_body, error = _terms_from_field(row.get("body_terms", row.get("body", [])), "body_terms")
    if error:
        failures.append(
            failure(
                "loading",
                "E_DOCUMENT_TERMS_INVALID",
                f"document:{doc_id}",
                error,
                doc_id=doc_id,
            )
        )
        return None, failures, False
    if not raw_title and not raw_body:
        failures.append(
            failure(
                "loading",
                "E_DOCUMENT_EMPTY",
                f"document:{doc_id}",
                "document must contain at least one title or body term",
                doc_id=doc_id,
            )
        )
        return None, failures, False

    raw_fingerprint = sha256_json(
        {
            "doc_id": doc_id,
            "shard": raw_shard,
            "title_terms": raw_title,
            "body_terms": raw_body,
        }
    )
    norm_key = cache.normalized_key(raw_fingerprint, vocabulary_fingerprint, NORMALIZATION_VERSION) if cache else sha256_json(
        {"raw": raw_fingerprint, "vocab": vocabulary_fingerprint, "norm": NORMALIZATION_VERSION}
    )
    cached = cache.lookup_normalized(norm_key) if cache else None
    cache_hit = False
    if cached is not None:
        normalized_payload = cached.get("normalized")
        if isinstance(normalized_payload, dict) and normalized_payload.get("doc_id") == doc_id:
            try:
                title = tuple(str(term) for term in normalized_payload["title_terms"])
                body = tuple(str(term) for term in normalized_payload["body_terms"])
                token_terms = tuple(str(term) for term in normalized_payload["token_terms"])
                token_count = int(normalized_payload["token_count"])
                normalized_fingerprint = str(normalized_payload["normalized_fingerprint"])
                all_terms = set(title) | set(body)
                if token_count == len(token_terms) and all_terms <= vocabulary:
                    cache_hit = True
                    doc = NormalizedDoc(
                        doc_id=doc_id,
                        shard=shard,
                        title_terms=title,
                        body_terms=body,
                        token_terms=token_terms,
                        token_count=token_count,
                        raw_fingerprint=raw_fingerprint,
                        normalized_fingerprint=normalized_fingerprint,
                        norm_cache_key=norm_key,
                    )
                    return doc, [], True
            except (KeyError, TypeError, ValueError):
                cache_hit = False
    title: list[str] = []
    body: list[str] = []
    for field_name, raw_terms, target in (
        ("title_terms", raw_title, title),
        ("body_terms", raw_body, body),
    ):
        for term_index, raw_term in enumerate(raw_terms):
            normalized = normalize_term(raw_term)
            if not normalized:
                failures.append(
                    failure(
                        "loading",
                        "E_DOCUMENT_TERM_EMPTY",
                        f"document:{doc_id}:{field_name}:{term_index}",
                        "term is empty after normalization",
                        doc_id=doc_id,
                        field=field_name,
                        term_index=term_index,
                        raw=raw_term,
                    )
                )
                continue
            if normalized not in vocabulary:
                failures.append(
                    failure(
                        "loading",
                        "E_DOCUMENT_TERM_OUT_OF_VOCAB",
                        f"document:{doc_id}:{field_name}:{term_index}",
                        "term is not present in the controlled vocabulary",
                        doc_id=doc_id,
                        field=field_name,
                        term_index=term_index,
                        term=normalized,
                    )
                )
                continue
            target.append(normalized)
    if failures:
        return None, failures, cache_hit
    if not title and not body:
        failures.append(
            failure(
                "loading",
                "E_DOCUMENT_EMPTY",
                f"document:{doc_id}",
                "document has no valid terms after normalization",
                doc_id=doc_id,
            )
        )
        return None, failures, cache_hit
    token_terms = tuple(title + body)
    normalized_fingerprint = sha256_json(
        {
            "doc_id": doc_id,
            "shard": shard,
            "title_terms": title,
            "body_terms": body,
            "token_terms": list(token_terms),
        }
    )
    doc = NormalizedDoc(
        doc_id=doc_id,
        shard=shard,
        title_terms=tuple(title),
        body_terms=tuple(body),
        token_terms=token_terms,
        token_count=len(token_terms),
        raw_fingerprint=raw_fingerprint,
        normalized_fingerprint=normalized_fingerprint,
        norm_cache_key=norm_key,
    )
    if cache:
        cache.store_normalized(
            norm_key,
            {
                "raw_fingerprint": raw_fingerprint,
                "normalized": {
                    "doc_id": doc_id,
                    "shard": shard,
                    "title_terms": list(doc.title_terms),
                    "body_terms": list(doc.body_terms),
                    "token_terms": list(doc.token_terms),
                    "token_count": doc.token_count,
                    "normalized_fingerprint": normalized_fingerprint,
                },
            },
        )
    return doc, [], cache_hit


def _normalize_query(
    row: dict[str, Any],
    vocabulary: set[str],
    declared_versions: set[str],
    config: AppConfig,
) -> tuple[QueryDef | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    raw_id = row.get("query_id", row.get("id"))
    if isinstance(raw_id, str):
        query_id = raw_id.strip()
    elif is_int(raw_id):
        query_id = str(raw_id)
    else:
        query_id = ""
    if not query_id:
        failures.append(
            failure(
                "loading",
                "E_QUERY_ID_INVALID",
                "query:id",
                "query_id must be a non-empty string or integer",
            )
        )
        return None, failures
    raw_terms, error = _terms_from_field(row.get("query_terms", row.get("terms", [])), "query_terms")
    if error:
        failures.append(
            failure(
                "loading",
                "E_QUERY_TERMS_INVALID",
                f"query:{query_id}",
                error,
                query_id=query_id,
            )
        )
        return None, failures
    if not raw_terms:
        failures.append(
            failure(
                "loading",
                "E_QUERY_EMPTY",
                f"query:{query_id}",
                "query must contain at least one term",
                query_id=query_id,
            )
        )
        return None, failures
    normalized_terms: list[str] = []
    for term_index, raw_term in enumerate(raw_terms):
        if not isinstance(raw_term, str):
            failures.append(
                failure(
                    "loading",
                    "E_QUERY_TERMS_INVALID",
                    f"query:{query_id}:term:{term_index}",
                    "query terms must be strings",
                    query_id=query_id,
                    term_index=term_index,
                )
            )
            continue
        normalized = normalize_term(raw_term)
        if not normalized:
            failures.append(
                failure(
                    "loading",
                    "E_QUERY_TERM_EMPTY",
                    f"query:{query_id}:term:{term_index}",
                    "query term is empty after normalization",
                    query_id=query_id,
                    term_index=term_index,
                )
            )
            continue
        if normalized not in vocabulary:
            failures.append(
                failure(
                    "loading",
                    "E_QUERY_TERM_OUT_OF_VOCAB",
                    f"query:{query_id}:term:{term_index}",
                    "query term is not present in the controlled vocabulary",
                    query_id=query_id,
                    term_index=term_index,
                    term=normalized,
                )
            )
            continue
        normalized_terms.append(normalized)
    if failures:
        return None, failures
    unique_terms = tuple(sorted(set(normalized_terms)))
    if not unique_terms:
        failures.append(
            failure(
                "loading",
                "E_QUERY_EMPTY",
                f"query:{query_id}",
                "query has no valid terms after normalization",
                query_id=query_id,
            )
        )
        return None, failures
    raw_version = row.get("query_stream_version", row.get("version"))
    if not isinstance(raw_version, str) or not raw_version.strip():
        failures.append(
            failure(
                "loading",
                "E_QUERY_VERSION_INVALID",
                f"query:{query_id}",
                "query stream version must be a non-empty string",
                query_id=query_id,
            )
        )
        return None, failures
    version = raw_version.strip()
    if version not in declared_versions:
        failures.append(
            failure(
                "loading",
                "E_QUERY_VERSION_UNSUPPORTED",
                f"query:{query_id}",
                "query stream version is not declared in the compatibility table",
                query_id=query_id,
                version=version,
                declared_versions=sorted(declared_versions),
            )
        )
        return None, failures
    raw_top_k = row.get("top_k", config.query_top_k)
    if not is_int(raw_top_k) or raw_top_k < 1:
        failures.append(
            failure(
                "loading",
                "E_QUERY_TOP_K_INVALID",
                f"query:{query_id}",
                "query top_k must be a positive integer",
                query_id=query_id,
                value=raw_top_k,
            )
        )
        return None, failures
    return QueryDef(query_id=query_id, terms=unique_terms, version=version, top_k=raw_top_k), []


def load_inputs(data_dir: Path, config: AppConfig, cache: CacheStore | None = None) -> LoadResult:
    paths = discover_inputs(data_dir)
    failures: list[dict[str, Any]] = []
    input_fingerprints: dict[str, str] = {}
    for path in paths.all_source_files():
        if path.exists():
            try:
                input_fingerprints[path.name] = sha256_file(path)
            except OSError:
                input_fingerprints[path.name] = "unreadable"
        else:
            input_fingerprints[path.name] = "missing"
    source_fingerprint = sha256_json({"schema": 1, "inputs": input_fingerprints})

    vocabulary, vocab_failures = _load_vocabulary(paths.vocabulary)
    failures.extend(vocab_failures)
    vocabulary_set = set(vocabulary)
    vocabulary_fingerprint = sha256_json(list(vocabulary))
    declared_versions, version_failures, version_table = _load_query_versions(paths.query_versions)
    failures.extend(version_failures)
    declared_version_set = set(declared_versions)

    document_rows, document_parse_failures = _read_jsonl(paths.documents)
    query_rows, query_parse_failures = _read_jsonl(paths.queries)
    failures.extend(document_parse_failures)
    failures.extend(query_parse_failures)

    documents: dict[str, NormalizedDoc] = {}
    documents_by_shard: dict[str, list[NormalizedDoc]] = {}
    normalization_hits = 0
    for line_number, row in document_rows:
        doc, doc_failures, cache_hit = _normalize_document(
            row, vocabulary_set, config, cache, vocabulary_fingerprint
        )
        normalization_hits += int(cache_hit)
        if doc is None:
            for item in doc_failures:
                item.setdefault("line", line_number)
                item["stable_key"] = f"{paths.documents.name}:{line_number:08d}:{item.get('stable_key', '')}"
            failures.extend(doc_failures)
            continue
        if doc.doc_id in documents:
            failures.append(
                failure(
                    "loading",
                    "E_DOCUMENT_ID_DUPLICATE",
                    f"{paths.documents.name}:{line_number:08d}",
                    "document ID is duplicated; the first occurrence is accepted",
                    line=line_number,
                    doc_id=doc.doc_id,
                )
            )
            continue
        documents[doc.doc_id] = doc
        documents_by_shard.setdefault(doc.shard, []).append(doc)

    queries: dict[str, QueryDef] = {}
    for line_number, row in query_rows:
        query, query_failures = _normalize_query(row, vocabulary_set, declared_version_set, config)
        if query is None:
            for item in query_failures:
                item.setdefault("line", line_number)
                item["stable_key"] = f"{paths.queries.name}:{line_number:08d}:{item.get('stable_key', '')}"
            failures.extend(query_failures)
            continue
        if query.query_id in queries:
            failures.append(
                failure(
                    "loading",
                    "E_QUERY_ID_DUPLICATE",
                    f"{paths.queries.name}:{line_number:08d}",
                    "query ID is duplicated; the first occurrence is accepted",
                    line=line_number,
                    query_id=query.query_id,
                )
            )
            continue
        queries[query.query_id] = query

    sorted_shards: dict[str, tuple[NormalizedDoc, ...]] = {}
    shard_fingerprints: dict[str, str] = {}
    for shard in sorted(documents_by_shard):
        ordered_docs = tuple(sorted(documents_by_shard[shard], key=lambda item: item.doc_id))
        sorted_shards[shard] = ordered_docs
        shard_fingerprints[shard] = sha256_json(
            [
                {
                    "doc_id": doc.doc_id,
                    "normalized_fingerprint": doc.normalized_fingerprint,
                    "token_terms": list(doc.token_terms),
                }
                for doc in ordered_docs
            ]
        )

    result = LoadResult(
        paths=paths,
        documents=documents,
        queries=queries,
        vocabulary=vocabulary,
        declared_query_versions=declared_versions,
        documents_by_shard=sorted_shards,
        shard_fingerprints=shard_fingerprints,
        source_fingerprint=source_fingerprint,
        input_fingerprints=input_fingerprints,
        failures=sort_failures(failures),
        stats={
            "document_rows": len(document_rows),
            "query_rows": len(query_rows),
            "accepted_documents": len(documents),
            "accepted_queries": len(queries),
            "rejected_documents": len(document_rows) - len(documents),
            "rejected_queries": len(query_rows) - len(queries),
            "vocabulary_size": len(vocabulary),
            "declared_query_versions": list(declared_versions),
            "normalization_cache_hits": normalization_hits,
            "shards": len(sorted_shards),
        },
    )
    return result


def document_line_payload(doc: NormalizedDoc | dict[str, Any]) -> dict[str, Any]:
    if isinstance(doc, NormalizedDoc):
        return {
            "doc_id": doc.doc_id,
            "title_terms": list(doc.title_terms),
            "body_terms": list(doc.body_terms),
            "shard": doc.shard,
        }
    return {
        "doc_id": doc.get("doc_id", doc.get("id")),
        "title_terms": doc.get("title_terms", doc.get("title", [])),
        "body_terms": doc.get("body_terms", doc.get("body", [])),
        "shard": doc.get("shard", doc.get("shard_label")),
    }
