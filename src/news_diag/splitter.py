from __future__ import annotations

from decimal import Decimal

from .errors import WorkbenchError, E_STAGE_FAILED
from .models import Document, Sentence


SENTENCE_DELIMITERS = frozenset("。！？!?；;")
CLOSING_MARKS = frozenset("”’」』）》】〕〉\"'")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_paragraph(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(text)
    while index < length:
        start = index
        while index < length and text[index] not in SENTENCE_DELIMITERS:
            index += 1
        if index < length:
            while index < length and text[index] in SENTENCE_DELIMITERS:
                index += 1
            while index < length and text[index] in CLOSING_MARKS:
                index += 1
        end = index
        trimmed_start, trimmed_end = _trim_span(text, start, end)
        if trimmed_end > trimmed_start:
            spans.append((trimmed_start, trimmed_end))
        if index == start:
            index += 1
    return spans


def document_lexicon(document: Document) -> tuple[str, ...]:
    terms: set[str] = set()
    for entity in document.entities:
        terms.add(entity.name)
        terms.update(entity.aliases)
    for event in document.events:
        terms.add(event.predicate)
    return tuple(sorted((term for term in terms if term), key=lambda item: (-len(item), item)))


def count_non_overlapping(text: str, term: str) -> int:
    if not term:
        return 0
    count = 0
    cursor = 0
    while True:
        found = text.find(term, cursor)
        if found < 0:
            break
        count += 1
        cursor = found + len(term)
    return count


def document_frequency_total(document: Document) -> Decimal:
    total = 0
    for term in document_lexicon(document):
        total += count_non_overlapping(document.text, term)
    return Decimal(total)


def _sentence_entity_ids(document: Document, text: str) -> tuple[str, ...]:
    entity_ids: list[str] = []
    for entity in document.entities:
        if any(term and term in text for term in entity.surface_terms()):
            entity_ids.append(entity.entity_id)
    return tuple(entity_ids)


def _sentence_frequency_hits(document: Document, text: str) -> Decimal:
    hits = 0
    for term in document_lexicon(document):
        hits += count_non_overlapping(text, term)
    return Decimal(hits)


def split_document(document: Document) -> list[Sentence]:
    sentences: list[Sentence] = []
    for paragraph_index, (paragraph_start, paragraph_end) in enumerate(document.paragraph_spans):
        paragraph_text = document.text[paragraph_start:paragraph_end]
        spans = split_paragraph(paragraph_text)
        if not spans and paragraph_text.strip():
            spans = [(0, len(paragraph_text))]
        for local_start, local_end in spans:
            global_start = paragraph_start + local_start
            global_end = paragraph_start + local_end
            sentence_text = document.text[global_start:global_end]
            sentence_index = len(sentences)
            sentences.append(
                Sentence(
                    sentence_index=sentence_index,
                    paragraph_index=paragraph_index,
                    start=global_start,
                    end=global_end,
                    text=sentence_text,
                    sort_key=f"{document.doc_id}|{sentence_index:08d}|sentence",
                    entity_ids=_sentence_entity_ids(document, sentence_text),
                    frequency_hits=_sentence_frequency_hits(document, sentence_text),
                )
            )
    if not sentences:
        raise WorkbenchError(
            E_STAGE_FAILED,
            "split",
            "document produced no sentences",
            details={"doc_id": document.doc_id},
        )
    return sentences
