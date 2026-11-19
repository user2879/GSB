"""Document model and deterministic JSONL loading with validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ErrorCode, Failure


@dataclass(frozen=True)
class Document:
    doc_id: str
    words: tuple[str, ...]
    positions: tuple[tuple[int, int], ...]
    text: str
    entity_hints: tuple[dict, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "words": list(self.words),
            "positions": [list(span) for span in self.positions],
            "text": self.text,
            "entity_hints": list(self.entity_hints),
        }


@dataclass
class LoadResult:
    documents: list[Document]
    failures: list[Failure]


def _synthetic_positions(words: list[str]) -> tuple[tuple[int, int], ...]:
    positions: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        positions.append((cursor, cursor + len(word)))
        cursor += len(word) + 1
    return tuple(positions)


def parse_document(raw: dict, known_entity_types: set[str], line_no: int) -> tuple[Document | None, Failure | None]:
    doc_id = str(raw.get("doc_id") or f"line-{line_no}")
    words_raw = raw.get("words")
    if isinstance(words_raw, str):
        words = words_raw.split(" ")
    elif isinstance(words_raw, list):
        words = [str(word) for word in words_raw]
    else:
        return None, Failure(doc_id, "load", ErrorCode.E_DOC_INVALID, "missing or invalid words field")

    if not words or any(not isinstance(word, str) or not word for word in words):
        return None, Failure(doc_id, "load", ErrorCode.E_DOC_INVALID, "word sequence must be non-empty")

    positions_raw = raw.get("positions")
    if positions_raw is None:
        positions = _synthetic_positions(words)
    else:
        if not isinstance(positions_raw, list) or len(positions_raw) != len(words):
            return None, Failure(doc_id, "load", ErrorCode.E_DOC_INVALID, "positions length must match words")
        positions = tuple(tuple(int(v) for v in span) for span in positions_raw)  # type: ignore[arg-type]

    text = raw.get("text") or " ".join(words)
    limit = len(text)
    cursor = 0
    for start, end in positions:
        if start < 0 or end < 0 or start > end or end > limit or start < cursor:
            return None, Failure(
                doc_id,
                "load",
                ErrorCode.E_POSITION_OUT_OF_RANGE,
                f"position [{start},{end}) out of range or non-contiguous in text length {limit}",
            )
        cursor = end

    hints = tuple(raw.get("entity_hints", []))
    for hint in hints:
        if hint.get("entity_type") not in known_entity_types:
            return None, Failure(
                doc_id,
                "load",
                ErrorCode.E_ENTITY_TYPE_UNKNOWN,
                f"unknown entity type {hint.get('entity_type')!r} in entity_hints",
            )

    document = Document(
        doc_id=doc_id,
        words=tuple(words),
        positions=positions,
        text=text,
        entity_hints=hints,
    )
    return document, None


def load_documents(path: str | Path, known_entity_types: set[str]) -> LoadResult:
    documents: list[Document] = []
    failures: list[Failure] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                failures.append(Failure(f"line-{line_no}", "load", ErrorCode.E_JSON_PARSE, str(exc)))
                continue
            if not isinstance(raw, dict):
                failures.append(Failure(f"line-{line_no}", "load", ErrorCode.E_JSON_PARSE, "line is not a JSON object"))
                continue
            document, failure = parse_document(raw, known_entity_types, line_no)
            if failure is not None:
                failures.append(failure)
                continue
            if document.doc_id in seen_ids:
                failures.append(
                    Failure(document.doc_id, "load", ErrorCode.E_DOC_INVALID, "duplicate document id")
                )
                continue
            seen_ids.add(document.doc_id)
            documents.append(document)

    documents.sort(key=lambda doc: doc.doc_id)
    failures.sort(key=lambda item: (item.doc_id, item.code))
    return LoadResult(documents=documents, failures=failures)
