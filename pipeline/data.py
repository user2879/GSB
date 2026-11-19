from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import sha256_json
from .errors import Failure, RecordInvalidError
from .normalization import normalize_text, token_script_language, tokenize


SOURCE_FILES = {
    "documents": "documents.jsonl",
    "queries": "queries.jsonl",
    "translations": "translations.jsonl",
    "qrels": "qrels.jsonl",
}


@dataclass(frozen=True)
class RawItem:
    source: str
    line: int
    value: Optional[dict]
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "line": self.line,
            "value": self.value,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RawItem":
        return cls(
            source=value["source"],
            line=int(value["line"]),
            value=value.get("value"),
            error_type=value.get("error_type"),
            error_message=value.get("error_message"),
        )


@dataclass(frozen=True)
class Document:
    doc_id: str
    declared_language: str
    title: str
    body: str
    title_tokens: tuple[str, ...]
    body_tokens: tuple[str, ...]

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.title_tokens + self.body_tokens

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "declared_language": self.declared_language,
            "title": self.title,
            "body": self.body,
            "title_tokens": list(self.title_tokens),
            "body_tokens": list(self.body_tokens),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Document":
        return cls(
            doc_id=value["doc_id"],
            declared_language=value["declared_language"],
            title=value["title"],
            body=value["body"],
            title_tokens=tuple(value["title_tokens"]),
            body_tokens=tuple(value["body_tokens"]),
        )


@dataclass(frozen=True)
class Query:
    query_id: str
    declared_language: str
    text: str
    tokens: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "declared_language": self.declared_language,
            "text": self.text,
            "tokens": list(self.tokens),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Query":
        return cls(
            query_id=value["query_id"],
            declared_language=value["declared_language"],
            text=value["text"],
            tokens=tuple(value["tokens"]),
        )


@dataclass(frozen=True)
class TranslationEntry:
    source_lang: str
    source_term: str
    target_lang: str
    target_term: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "source_lang": self.source_lang,
            "source_term": self.source_term,
            "target_lang": self.target_lang,
            "target_term": self.target_term,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TranslationEntry":
        return cls(
            source_lang=value["source_lang"],
            source_term=value["source_term"],
            target_lang=value["target_lang"],
            target_term=value["target_term"],
            confidence=float(value["confidence"]),
        )


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    grade: int

    def to_dict(self) -> dict:
        return {"query_id": self.query_id, "doc_id": self.doc_id, "grade": self.grade}

    @classmethod
    def from_dict(cls, value: dict) -> "Qrel":
        return cls(query_id=value["query_id"], doc_id=value["doc_id"], grade=int(value["grade"]))


@dataclass
class NormalizedState:
    documents: list[Document] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    translations: list[TranslationEntry] = field(default_factory=list)
    qrels: list[Qrel] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    seen_doc_ids: set[str] = field(default_factory=set)
    seen_query_ids: set[str] = field(default_factory=set)
    accepted_doc_ids: set[str] = field(default_factory=set)
    accepted_query_ids: set[str] = field(default_factory=set)
    qrel_keys: set[tuple[str, str]] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "documents": [item.to_dict() for item in self.documents],
            "queries": [item.to_dict() for item in self.queries],
            "translations": [item.to_dict() for item in self.translations],
            "qrels": [item.to_dict() for item in self.qrels],
            "failures": [item.to_dict() for item in self.failures],
            "seen_doc_ids": sorted(self.seen_doc_ids),
            "seen_query_ids": sorted(self.seen_query_ids),
            "accepted_doc_ids": sorted(self.accepted_doc_ids),
            "accepted_query_ids": sorted(self.accepted_query_ids),
            "qrel_keys": [list(key) for key in sorted(self.qrel_keys)],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "NormalizedState":
        return cls(
            documents=[Document.from_dict(item) for item in value.get("documents", [])],
            queries=[Query.from_dict(item) for item in value.get("queries", [])],
            translations=[TranslationEntry.from_dict(item) for item in value.get("translations", [])],
            qrels=[Qrel.from_dict(item) for item in value.get("qrels", [])],
            failures=[Failure(**item) for item in value.get("failures", [])],
            seen_doc_ids=set(value.get("seen_doc_ids", [])),
            seen_query_ids=set(value.get("seen_query_ids", [])),
            accepted_doc_ids=set(value.get("accepted_doc_ids", [])),
            accepted_query_ids=set(value.get("accepted_query_ids", [])),
            qrel_keys={tuple(item) for item in value.get("qrel_keys", [])},
        )


class TranslationResolver:
    def __init__(self, entries: Iterable[TranslationEntry]) -> None:
        ordered = sorted(
            entries,
            key=lambda item: (item.source_lang, item.source_term, -item.confidence, item.target_lang, item.target_term),
        )
        self.entries = ordered
        self._by_key: dict[tuple[str, str], list[TranslationEntry]] = {}
        for entry in ordered:
            self._by_key.setdefault((entry.source_lang, entry.source_term), []).append(entry)

    def lookup(self, source_lang: str, source_term: str, target_languages: Iterable[str] | None = None) -> list[TranslationEntry]:
        allowed = None if target_languages is None else set(target_languages)
        candidates = self._by_key.get((source_lang, source_term), [])
        if allowed is not None:
            candidates = [item for item in candidates if item.target_lang in allowed]
        return sorted(candidates, key=lambda item: (-item.confidence, item.target_term, item.target_lang))

    def has_translation(self, source_lang: str, source_term: str) -> bool:
        return bool(self._by_key.get((source_lang, source_term)))

    def hash(self) -> str:
        return sha256_json([item.to_dict() for item in self.entries])


def read_raw_sources(sample_dir: str | Path) -> list[RawItem]:
    sample_dir = Path(sample_dir)
    items: list[RawItem] = []
    for source, filename in SOURCE_FILES.items():
        path = sample_dir / filename
        if not path.exists():
            items.append(RawItem(source=source, line=0, value=None, error_type="FileNotFoundError", error_message=str(path)))
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    items.append(RawItem(source=source, line=line_number, value=None, error_type="JSONDecodeError", error_message="empty line"))
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    items.append(RawItem(source=source, line=line_number, value=None, error_type=type(exc).__name__, error_message=str(exc)))
                    continue
                if not isinstance(value, dict):
                    items.append(RawItem(source=source, line=line_number, value=None, error_type="TypeError", error_message="record must be a JSON object"))
                    continue
                items.append(RawItem(source=source, line=line_number, value=value))
    return items


def _pick(record: dict, *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _as_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordInvalidError(f"{field_name} must be a non-empty string")
    return value


def _record_identifier(source: str, value: Optional[dict]) -> str:
    if isinstance(value, dict):
        candidate = _pick(value, "doc_id", "document_id", "query_id", "source_term", "id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return f"line:{source}"


def _failure_for(item: RawItem, exc: BaseException) -> Failure:
    code = getattr(exc, "code", "E_RECORD_INVALID")
    identifier = _record_identifier(item.source, item.value)
    return Failure(
        identifier=identifier,
        stage="normalize",
        code=code,
        exception_type=type(exc).__name__,
        message=str(exc),
        source=item.source,
        line=item.line,
    )


def normalize_raw_item(state: NormalizedState, item: RawItem, config: dict) -> None:
    if item.value is None:
        failure = Failure(
            identifier=_record_identifier(item.source, None),
            stage="normalize",
            code="E_RECORD_INVALID",
            exception_type=item.error_type or "RecordInvalidError",
            message=item.error_message or "invalid record",
            source=item.source,
            line=item.line,
        )
        state.failures.append(failure)
        return
    try:
        rules = config["normalization"]
        allowed = set(config["allowed_languages"])
        index_languages = set(config["index_languages"])
        record = item.value
        if item.source == "documents":
            doc_id_raw = _as_nonempty_string(_pick(record, "doc_id", "document_id", "id"), "doc_id")
            doc_id = doc_id_raw.strip()
            if doc_id in state.seen_doc_ids:
                raise RecordInvalidError(f"duplicate document ID: {doc_id}")
            state.seen_doc_ids.add(doc_id)
            language = _as_nonempty_string(_pick(record, "lang", "language"), "language")
            if language not in allowed:
                raise RecordInvalidError(f"unsupported document language: {language}")
            title = normalize_text(_as_nonempty_string(_pick(record, "title"), "title"), rules)
            body = normalize_text(_as_nonempty_string(_pick(record, "body"), "body"), rules)
            title_tokens = tokenize(title, rules)
            body_tokens = tokenize(body, rules)
            if not title_tokens or not body_tokens:
                raise RecordInvalidError("title/body must contain at least one token")
            state.documents.append(Document(doc_id, language, title, body, title_tokens, body_tokens))
            state.accepted_doc_ids.add(doc_id)
        elif item.source == "queries":
            query_id_raw = _as_nonempty_string(_pick(record, "query_id", "id"), "query_id")
            query_id = query_id_raw.strip()
            if query_id in state.seen_query_ids:
                raise RecordInvalidError(f"duplicate query ID: {query_id}")
            state.seen_query_ids.add(query_id)
            language = _as_nonempty_string(_pick(record, "lang", "language"), "language")
            if language not in allowed:
                raise RecordInvalidError(f"unsupported query language: {language}")
            text = normalize_text(_as_nonempty_string(_pick(record, "query", "text"), "query"), rules)
            tokens = tokenize(text, rules)
            if not tokens:
                raise RecordInvalidError("query must contain at least one token")
            state.queries.append(Query(query_id, language, text, tokens))
            state.accepted_query_ids.add(query_id)
        elif item.source == "translations":
            source_term = normalize_text(_as_nonempty_string(_pick(record, "source_term", "source"), "source_term"), rules)
            target_term = normalize_text(_as_nonempty_string(_pick(record, "target_term", "target"), "target_term"), rules)
            if not source_term or not target_term:
                raise RecordInvalidError("translation terms must be non-empty after normalization")
            source_lang = _pick(record, "source_lang", "source_language")
            target_lang = _pick(record, "target_lang", "target_language")
            if source_lang is None or target_lang is None:
                source_script = token_script_language(source_term)
                target_script = token_script_language(target_term)
                if source_script == "zh" and target_script == "en":
                    source_lang, target_lang = "zh", "en"
                elif source_script == "en" and target_script == "zh":
                    source_lang, target_lang = "en", "zh"
            if source_lang not in index_languages or target_lang not in index_languages:
                raise RecordInvalidError("translation languages must be in index_languages")
            if source_lang == target_lang:
                raise RecordInvalidError("translation source and target languages must differ")
            confidence = _pick(record, "confidence", "alignment_confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                raise RecordInvalidError("confidence must be numeric")
            confidence = float(confidence)
            if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
                raise RecordInvalidError("confidence must be finite and in [0, 1]")
            state.translations.append(TranslationEntry(source_lang, source_term, target_lang, target_term, confidence))
        elif item.source == "qrels":
            query_id = _as_nonempty_string(_pick(record, "query_id", "qid"), "query_id").strip()
            doc_id = _as_nonempty_string(_pick(record, "doc_id", "did"), "doc_id").strip()
            key = (query_id, doc_id)
            if key in state.qrel_keys:
                raise RecordInvalidError(f"duplicate qrel pair: {query_id}/{doc_id}")
            state.qrel_keys.add(key)
            grade = _pick(record, "relevance", "relevance_grade", "grade")
            if not isinstance(grade, int) or isinstance(grade, bool):
                raise RecordInvalidError("relevance grade must be an integer")
            if grade < 0 or grade > int(config["max_relevance_grade"]):
                raise RecordInvalidError(f"relevance grade must be in [0, {config['max_relevance_grade']}]")
            if query_id not in state.accepted_query_ids:
                raise RecordInvalidError(f"qrel references unknown query ID: {query_id}")
            if doc_id not in state.accepted_doc_ids:
                raise RecordInvalidError(f"qrel references unknown document ID: {doc_id}")
            state.qrels.append(Qrel(query_id, doc_id, grade))
        else:
            raise RecordInvalidError(f"unsupported source: {item.source}")
    except Exception as exc:
        if not isinstance(exc, RecordInvalidError):
            exc = RecordInvalidError(str(exc))
        state.failures.append(_failure_for(item, exc))


def apply_normalization_batch(state: NormalizedState, items: Iterable[RawItem], config: dict) -> None:
    for item in items:
        normalize_raw_item(state, item, config)


def valid_translation_hash(state: NormalizedState) -> str:
    entries = sorted(
        state.translations,
        key=lambda item: (item.source_lang, item.source_term, -item.confidence, item.target_lang, item.target_term),
    )
    return sha256_json([item.to_dict() for item in entries])


def sorted_failures(failures: Iterable[Failure]) -> list[Failure]:
    return sorted(failures, key=lambda item: item.stable_key)


def input_hashes(sample_dir: str | Path) -> dict[str, str]:
    sample_dir = Path(sample_dir)
    result = {}
    for source, filename in SOURCE_FILES.items():
        path = sample_dir / filename
        if path.exists():
            import hashlib

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result[source] = digest
        else:
            result[source] = "missing"
    return result


def state_counts(state: NormalizedState) -> dict:
    return {
        "documents": len(state.documents),
        "queries": len(state.queries),
        "translations": len(state.translations),
        "qrels": len(state.qrels),
        "invalid_records": len(state.failures),
    }
