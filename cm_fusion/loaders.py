from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import FailureRecord, ValidationError, sort_failures
from .normalization import normalize_term
from .util import sha256_file


@dataclass(frozen=True)
class Document:
    doc_id: str
    title_terms: tuple[str, ...]
    body_terms: tuple[str, ...]
    shard: str
    line_no: int

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title_terms": list(self.title_terms),
            "body_terms": list(self.body_terms),
            "shard": self.shard,
            "line_no": self.line_no,
        }


@dataclass(frozen=True)
class Query:
    query_id: str
    terms: tuple[str, ...]
    stream_version: str
    line_no: int

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "terms": list(self.terms),
            "stream_version": self.stream_version,
            "line_no": self.line_no,
        }


@dataclass(frozen=True)
class LoadedCorpus:
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    failures: tuple[FailureRecord, ...]
    document_lines: tuple[str, ...]
    query_lines: tuple[str, ...]
    docs_hash: str
    queries_hash: str

    def to_dict(self) -> dict:
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "queries": [query.to_dict() for query in self.queries],
            "failures": [failure.to_dict() for failure in self.failures],
            "docs_hash": self.docs_hash,
            "queries_hash": self.queries_hash,
        }


def _read_lines(path: str | Path) -> list[str]:
    data = Path(path).read_bytes()
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return normalized.decode("utf-8", errors="surrogateescape").split("\n")


def _decode_json_line(text: str) -> tuple[dict | None, str | None]:
    if not text.strip():
        return None, "blank JSONL line"
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        return None, f"invalid JSON: {exc.msg if isinstance(exc, json.JSONDecodeError) else exc}"
    if not isinstance(value, dict):
        return None, "JSONL value must be an object"
    return value, None


def _validate_doc_terms(value: object, field_name: str, config: Config) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError("E_DOC_INVALID", f"{field_name} must be a non-empty term array")
    vocabulary = set(config.controlled_vocabulary)
    normalized: list[str] = []
    for term in value:
        try:
            item = normalize_term(term, config.normalization)
        except ValidationError as exc:
            raise ValidationError("E_DOC_INVALID", f"{field_name}: {exc.message}") from exc
        if item not in vocabulary:
            raise ValidationError("E_DOC_INVALID", f"{field_name} term outside controlled vocabulary: {item}")
        normalized.append(item)
    return tuple(normalized)


def _validate_query_terms(value: object, config: Config) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError("E_INDEX_INVALID", "query terms must be a non-empty array")
    vocabulary = set(config.controlled_vocabulary)
    normalized: list[str] = []
    for term in value:
        try:
            item = normalize_term(term, config.normalization)
        except ValidationError as exc:
            raise ValidationError("E_INDEX_INVALID", f"query term invalid: {exc.message}") from exc
        if item not in vocabulary:
            raise ValidationError("E_INDEX_INVALID", f"query term outside controlled vocabulary: {item}")
        normalized.append(item)
    return tuple(normalized)


def load_documents(path: str | Path, config: Config) -> tuple[list[Document], list[FailureRecord], list[str]]:
    source = Path(path).name
    lines = _read_lines(path)
    documents: list[Document] = []
    failures: list[FailureRecord] = []
    seen_ids: set[str] = set()
    for line_no, text in enumerate(lines, start=1):
        if not text.strip():
            if line_no != len(lines):
                failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", "", "blank JSONL line"))
            continue
        raw, error = _decode_json_line(text)
        if error is not None:
            failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", "", error))
            continue
        assert raw is not None
        doc_id = raw.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            failures.append(FailureRecord(source, line_no, "E_DOC_INVALID", "", "doc_id must be a non-empty string"))
            continue
        if doc_id in seen_ids:
            failures.append(FailureRecord(source, line_no, "E_DOC_INVALID", doc_id, "duplicate document ID"))
            continue
        try:
            title_terms = _validate_doc_terms(raw.get("title_terms"), "title_terms", config)
            body_raw = raw.get("body_terms")
            if not isinstance(body_raw, list) or not body_raw:
                raise ValidationError("E_DOC_INVALID", "body_terms must not be empty")
            body_terms = _validate_doc_terms(body_raw, "body_terms", config)
            shard = raw.get("shard")
            if not isinstance(shard, str) or not shard:
                raise ValidationError("E_DOC_INVALID", "shard must be a single non-empty string")
            if shard not in config.allowed_shards:
                raise ValidationError("E_INDEX_INVALID", f"unknown shard: {shard}")
        except ValidationError as exc:
            failures.append(FailureRecord(source, line_no, exc.code, doc_id, exc.message))
            continue
        seen_ids.add(doc_id)
        documents.append(Document(doc_id, title_terms, body_terms, shard, line_no))
    return documents, failures, lines


def load_queries(path: str | Path, config: Config) -> tuple[list[Query], list[FailureRecord], list[str]]:
    source = Path(path).name
    lines = _read_lines(path)
    queries: list[Query] = []
    failures: list[FailureRecord] = []
    seen_ids: set[str] = set()
    for line_no, text in enumerate(lines, start=1):
        if not text.strip():
            if line_no != len(lines):
                failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", "", "blank JSONL line"))
            continue
        raw, error = _decode_json_line(text)
        if error is not None:
            failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", "", error))
            continue
        assert raw is not None
        query_id = raw.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", "", "query_id must be a non-empty string"))
            continue
        if query_id in seen_ids:
            failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", query_id, "duplicate query ID"))
            continue
        stream_version = raw.get("stream_version")
        if not isinstance(stream_version, str) or not stream_version:
            failures.append(FailureRecord(source, line_no, "E_INDEX_INVALID", query_id, "stream_version must be a non-empty string"))
            continue
        try:
            config.compat.require_query_stream(stream_version)
            terms = _validate_query_terms(raw.get("terms", raw.get("query_terms")), config)
        except ValidationError as exc:
            failures.append(FailureRecord(source, line_no, exc.code, query_id, exc.message))
            continue
        seen_ids.add(query_id)
        queries.append(Query(query_id, terms, stream_version, line_no))
    return queries, failures, lines


def load_corpus(docs_path: str | Path, queries_path: str | Path, config: Config) -> LoadedCorpus:
    documents, doc_failures, doc_lines = load_documents(docs_path, config)
    queries, query_failures, query_lines = load_queries(queries_path, config)
    failures = sort_failures(doc_failures + query_failures)
    return LoadedCorpus(
        documents=tuple(documents),
        queries=tuple(queries),
        failures=tuple(failures),
        document_lines=tuple(doc_lines),
        query_lines=tuple(query_lines),
        docs_hash=sha256_file(docs_path),
        queries_hash=sha256_file(queries_path),
    )


def format_document_line(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def replace_document_line(lines: list[str], doc_id: str, replacement: dict) -> list[str]:
    result = list(lines)
    for index, line in enumerate(result):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("doc_id") == doc_id:
            result[index] = format_document_line(replacement)
            return result
    raise ValidationError("E_EDIT_INVALID", f"document to fix was not found: {doc_id}")

