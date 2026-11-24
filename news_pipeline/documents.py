"""Structured news-document loading and per-document validation."""

import json

from .errors import E_INPUT_INVALID, PipelineError
from .hashing import fingerprint


REQUIRED_FIELDS = ("doc_id", "title", "paragraphs", "entities", "events")


def _validate_document(payload, line_no):
    if not isinstance(payload, dict):
        raise PipelineError(E_INPUT_INVALID, "line %d: document must be a JSON object" % line_no)
    for field_name in REQUIRED_FIELDS:
        if field_name not in payload:
            raise PipelineError(
                E_INPUT_INVALID,
                "line %d: missing field %r" % (line_no, field_name),
                object_id=str(payload.get("doc_id", "unknown")),
            )
    doc_id = payload["doc_id"]
    if not isinstance(doc_id, str) or not doc_id:
        raise PipelineError(E_INPUT_INVALID, "line %d: doc_id must be a non-empty string" % line_no)
    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise PipelineError(E_INPUT_INVALID, "doc %s: title must be non-empty" % doc_id, doc_id)

    paragraphs = payload["paragraphs"]
    if not isinstance(paragraphs, list) or not paragraphs:
        raise PipelineError(E_INPUT_INVALID, "doc %s: paragraphs must be a non-empty list" % doc_id, doc_id)
    norm_paragraphs = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict) or "pid" not in paragraph or "text" not in paragraph:
            raise PipelineError(
                E_INPUT_INVALID,
                "doc %s: paragraph %d must contain pid and text" % (doc_id, index),
                doc_id,
            )
        if not isinstance(paragraph["text"], str) or not paragraph["text"].strip():
            raise PipelineError(
                E_INPUT_INVALID,
                "doc %s: paragraph %r text is empty" % (doc_id, paragraph["pid"]),
                doc_id,
            )
        norm_paragraphs.append({"pid": str(paragraph["pid"]), "text": paragraph["text"]})

    entities = payload["entities"]
    if not isinstance(entities, list) or not entities:
        raise PipelineError(E_INPUT_INVALID, "doc %s: entities must be a non-empty list" % doc_id, doc_id)
    norm_entities = []
    seen_entities = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict) or "text" not in entity or "type" not in entity:
            raise PipelineError(
                E_INPUT_INVALID,
                "doc %s: entity %d must contain text and type" % (doc_id, index),
                doc_id,
            )
        text = entity["text"]
        etype = entity["type"]
        if not isinstance(text, str) or not text.strip() or not isinstance(etype, str) or not etype.strip():
            raise PipelineError(E_INPUT_INVALID, "doc %s: entity %d is malformed" % (doc_id, index), doc_id)
        if text in seen_entities:
            continue
        seen_entities.add(text)
        norm_entities.append({"text": text, "type": etype})

    events = payload["events"]
    if not isinstance(events, list):
        raise PipelineError(E_INPUT_INVALID, "doc %s: events must be a list" % doc_id, doc_id)
    norm_events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise PipelineError(E_INPUT_INVALID, "doc %s: event %d must be an object" % (doc_id, index), doc_id)
        triple = event.get("triple")
        if not isinstance(triple, list) or len(triple) != 3 or not all(isinstance(x, str) and x for x in triple):
            raise PipelineError(
                E_INPUT_INVALID,
                "doc %s: event %d triple must be a 3-element string list" % (doc_id, index),
                doc_id,
            )
        norm_events.append({"triple": [triple[0], triple[1], triple[2]]})

    document = {
        "doc_id": doc_id,
        "title": payload["title"],
        "paragraphs": norm_paragraphs,
        "entities": norm_entities,
        "events": norm_events,
    }
    document["content_fingerprint"] = fingerprint(
        {
            "doc_id": doc_id,
            "title": payload["title"],
            "paragraphs": norm_paragraphs,
            "entities": norm_entities,
            "events": norm_events,
        }
    )
    return document


def load_documents(path):
    """Return (documents, failures). Failures never stop remaining documents."""
    documents = []
    failures = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {
                        "code": E_INPUT_INVALID,
                        "object_id": "line:%d" % line_no,
                        "message": "invalid JSON: %s" % exc.msg,
                    }
                )
                continue
            try:
                document = _validate_document(payload, line_no)
            except PipelineError as exc:
                failures.append(exc.to_record())
                continue
            if document["doc_id"] in seen_ids:
                failures.append(
                    {
                        "code": E_INPUT_INVALID,
                        "object_id": document["doc_id"],
                        "message": "duplicate doc_id",
                    }
                )
                continue
            seen_ids.add(document["doc_id"])
            documents.append(document)
    documents.sort(key=lambda item: item["doc_id"])
    return documents, failures
