"""Line-oriented input readers and structural validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import DocumentInvalid
from .util import content_fingerprint


@dataclass
class ParsedRecord:
    index: int
    value: Any = None
    error: str | None = None


@dataclass
class DocumentLoadResult:
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    base_documents_hash: str = ""


def read_json_records(path: str | Path) -> list[ParsedRecord]:
    """Read JSONL, or one JSON array containing records.

    Blank lines are ignored. A malformed line becomes a ParsedRecord error so
    callers can emit the appropriate business code and continue.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            value = json.loads(text)
        except Exception as exc:
            return [ParsedRecord(0, error=f"JSON 数组解析失败: {exc}")]
        if not isinstance(value, list):
            return [ParsedRecord(0, error="JSON 根节点必须是数组")]
        return [ParsedRecord(index, value=item) for index, item in enumerate(value)]

    records: list[ParsedRecord] = []
    record_index = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            records.append(ParsedRecord(record_index, error=f"第 {line_number} 行 JSON 解析失败: {exc}"))
        else:
            records.append(ParsedRecord(record_index, value=value))
        record_index += 1
    return records


def normalize_document(raw: Any, *, stage: str = "load") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DocumentInvalid("新闻条目必须是 JSON 对象", stage=stage)

    document_id = raw.get("document_id", raw.get("doc_id", raw.get("id")))
    if not isinstance(document_id, str) or not document_id.strip():
        raise DocumentInvalid("document_id 必须是非空字符串", stage=stage)
    document_id = document_id.strip()

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise DocumentInvalid("title 必须是非空字符串", stage=stage, document_id=document_id,
                              details={"field": "title"})

    paragraphs = raw.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise DocumentInvalid("paragraphs 必须是非空数组", stage=stage, document_id=document_id,
                              details={"field": "paragraphs"})
    normalized_paragraphs: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise DocumentInvalid(f"paragraphs[{index}] 必须是非空字符串", stage=stage,
                                  document_id=document_id,
                                  details={"field": "paragraphs", "index": index})
        normalized_paragraphs.append(paragraph)

    entities_raw = raw.get("entities")
    if not isinstance(entities_raw, list):
        raise DocumentInvalid("entities 必须是数组", stage=stage, document_id=document_id,
                              details={"field": "entities"})
    entities: list[dict[str, str]] = []
    seen_entity_ids: set[str] = set()
    for index, entity in enumerate(entities_raw):
        if isinstance(entity, str):
            if not entity.strip():
                raise DocumentInvalid(f"entities[{index}] 不能为空", stage=stage,
                                      document_id=document_id,
                                      details={"field": "entities", "index": index})
            normalized = {"id": f"entity-{index}", "text": entity, "type": "ENTITY"}
        elif isinstance(entity, dict):
            entity_id = entity.get("id", entity.get("entity_id"))
            entity_text = entity.get("text", entity.get("name", entity.get("label")))
            entity_type = entity.get("type", "ENTITY")
            if not isinstance(entity_id, str) or not entity_id.strip():
                raise DocumentInvalid(f"entities[{index}].id 必须是非空字符串", stage=stage,
                                      document_id=document_id,
                                      details={"field": "entities", "index": index})
            if not isinstance(entity_text, str) or not entity_text.strip():
                raise DocumentInvalid(f"entities[{index}].text 必须是非空字符串", stage=stage,
                                      document_id=document_id,
                                      details={"field": "entities", "index": index})
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise DocumentInvalid(f"entities[{index}].type 必须是非空字符串", stage=stage,
                                      document_id=document_id,
                                      details={"field": "entities", "index": index})
            normalized = {"id": entity_id.strip(), "text": entity_text.strip(),
                          "type": entity_type.strip()}
        else:
            raise DocumentInvalid(f"entities[{index}] 必须是字符串或对象", stage=stage,
                                  document_id=document_id,
                                  details={"field": "entities", "index": index})
        if normalized["id"] in seen_entity_ids:
            raise DocumentInvalid(f"实体 ID 重复: {normalized['id']}", stage=stage,
                                  document_id=document_id,
                                  details={"field": "entities", "index": index})
        seen_entity_ids.add(normalized["id"])
        entities.append(normalized)

    events_raw = raw.get("events")
    if not isinstance(events_raw, list):
        raise DocumentInvalid("events 必须是数组", stage=stage, document_id=document_id,
                              details={"field": "events"})
    events: list[dict[str, str]] = []
    for index, event in enumerate(events_raw):
        if isinstance(event, (list, tuple)):
            if len(event) != 3:
                raise DocumentInvalid(f"events[{index}] 三元组必须包含三元素", stage=stage,
                                      document_id=document_id,
                                      details={"field": "events", "index": index})
            subject, predicate, object_ = event
            event = {"subject": subject, "predicate": predicate, "object": object_}
        if not isinstance(event, dict):
            raise DocumentInvalid(f"events[{index}] 必须是三元组对象", stage=stage,
                                  document_id=document_id,
                                  details={"field": "events", "index": index})
        triple: dict[str, str] = {}
        for field in ("subject", "predicate", "object"):
            value = event.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DocumentInvalid(f"events[{index}].{field} 必须是非空字符串", stage=stage,
                                      document_id=document_id,
                                      details={"field": "events", "index": index})
            triple[field] = value.strip()
        events.append(triple)

    return {
        "document_id": document_id,
        "title": title,
        "paragraphs": normalized_paragraphs,
        "entities": entities,
        "events": events,
    }


def load_documents(path: str | Path) -> DocumentLoadResult:
    result = DocumentLoadResult()
    seen_ids: set[str] = set()
    for record in read_json_records(path):
        if record.error is not None:
            result.failures.append({
                "document_id": "",
                "stage": "load",
                "code": "E_DOC_INVALID",
                "exception_type": "DocumentInvalid",
                "message": record.error,
                "record_index": record.index,
            })
            continue
        try:
            document = normalize_document(record.value, stage="load")
        except DocumentInvalid as exc:
            result.failures.append(exc.to_failure(record_index=record.index))
            continue
        document_id = document["document_id"]
        if document_id in seen_ids:
            result.failures.append({
                "document_id": document_id,
                "stage": "load",
                "code": "E_DOC_INVALID",
                "exception_type": "DocumentInvalid",
                "message": f"文档 ID 重复: {document_id}",
                "record_index": record.index,
                "details": {"field": "document_id"},
            })
            continue
        seen_ids.add(document_id)
        result.documents[document_id] = document

    ordered_documents = [result.documents[key] for key in sorted(result.documents)]
    result.base_documents_hash = content_fingerprint(ordered_documents)
    return result


def load_commands(path: str | Path) -> list[ParsedRecord]:
    return read_json_records(path)


def load_injections(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    injection_path = Path(path)
    with injection_path.open("r", encoding="utf-8") as handle:
        if injection_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("injections", [])
    else:
        return []
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            entry = dict(item)
            entry.setdefault("id", f"INJ-{index + 1:03d}")
            normalized.append(entry)
    return normalized
