from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .util import atomic_write_json, atomic_write_text


def read_document_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("document JSONL row is not an object")
            rows.append(value)
    return rows


def write_document_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    atomic_write_text(path, text)


def _doc_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("doc_id", row.get("document_id", row.get("id")))
    if value is None:
        return None
    return str(value)


def append_document(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_document_rows(path)
    doc_id = _doc_id(payload)
    if doc_id is None:
        raise ValueError("document payload needs doc_id")
    if any(_doc_id(row) == doc_id for row in rows):
        raise ValueError(f"document ID already exists: {doc_id}")
    rows.append(dict(payload))
    write_document_rows(path, rows)
    return {"action": "append", "doc_id": doc_id, "document_count": len(rows)}


def update_document(path: Path, doc_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_document_rows(path)
    matches = [index for index, row in enumerate(rows) if _doc_id(row) == doc_id]
    if not matches:
        raise ValueError(f"document ID not found: {doc_id}")
    if len(matches) > 1:
        raise ValueError(f"document ID is duplicated: {doc_id}")
    index = matches[0]
    merged = dict(rows[index])
    merged.update(payload)
    merged["doc_id"] = doc_id
    rows[index] = merged
    write_document_rows(path, rows)
    return {"action": "update", "doc_id": doc_id, "document_count": len(rows)}


def delete_document(path: Path, doc_id: str) -> dict[str, Any]:
    rows = read_document_rows(path)
    kept = [row for row in rows if _doc_id(row) != doc_id]
    if len(kept) == len(rows):
        raise ValueError(f"document ID not found: {doc_id}")
    write_document_rows(path, kept)
    return {"action": "delete", "doc_id": doc_id, "document_count": len(kept)}


def patch_config(path: Path, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("configuration root must be an object")
    else:
        value = {}
    value.update(overrides)
    atomic_write_json(path, value)
    return value
