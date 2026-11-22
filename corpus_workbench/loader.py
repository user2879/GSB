"""Deterministic JSONL corpus loading with injectable-fault tracing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codes import E_CONFIG_INVALID, E_DUP_DOC_ID, E_ENCODING_INVALID, E_SOURCE_UNKNOWN
from .config import InjectionItem
from .models import Document, Failure, content_fingerprint
from .util import file_sha256, safe_relative, sha256_json

_SOURCE_RE = re.compile(rb'"source"\s*:\s*"([^"\\]*)"')
_DOC_ID_RE = re.compile(rb'"doc_id"\s*:\s*"([^"\\]*)"')
_INJECTION_RE = re.compile(rb'"injection_id"\s*:\s*"([^"\\]*)"')


@dataclass
class LoadResult:
    documents: list[Document]
    failures: list[Failure]
    corpus_digest: str
    file_digests: dict[str, str]
    injection_traces: dict[str, dict[str, Any]]
    fatal: bool = False


def _extract_byte_field(pattern: re.Pattern[bytes], raw: bytes) -> str:
    match = pattern.search(raw)
    if not match:
        return ""
    try:
        return match.group(1).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return match.group(1).decode("utf-8", errors="replace")


def _extract_trace(raw: bytes) -> tuple[str, str, str | None]:
    return (
        _extract_byte_field(_SOURCE_RE, raw),
        _extract_byte_field(_DOC_ID_RE, raw),
        _extract_byte_field(_INJECTION_RE, raw) or None,
    )


def _norm_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None
    return value


def _valid_text(text: Any) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


def _configured_injection_map(
    injections: list[InjectionItem] | None,
) -> dict[tuple[str, str, str], str]:
    if not injections:
        return {}
    fault_codes = {
        "empty_text": E_ENCODING_INVALID,
        "non_utf8": E_ENCODING_INVALID,
        "unknown_source": E_SOURCE_UNKNOWN,
        "duplicate_doc_id": E_DUP_DOC_ID,
    }
    result: dict[tuple[str, str, str], str] = {}
    for item in injections:
        code = fault_codes.get(item.fault)
        if code is not None:
            result.setdefault((item.source, item.doc_id, code), item.injection_id)
    return result


def load_documents(
    data_path: str | Path,
    source_weights: dict[str, float],
    injections: list[InjectionItem] | None = None,
) -> LoadResult:
    root = Path(data_path)
    if root.is_dir():
        files = sorted((path for path in root.rglob("*.jsonl") if path.is_file()), key=lambda p: p.as_posix())
    elif root.is_file():
        files = [root]
    else:
        failure = Failure(
            source="",
            doc_id="",
            stage="load_validate",
            code=E_CONFIG_INVALID,
            message=f"corpus path not found: {root}",
            file=str(root),
        )
        return LoadResult([], [failure], sha256_json({"missing": str(root)}), {}, {}, fatal=True)

    documents: list[Document] = []
    failures: list[Failure] = []
    seen_ids: dict[str, str] = {}
    file_digests: dict[str, str] = {}
    injection_traces: dict[str, dict[str, Any]] = {}
    configured_injections = _configured_injection_map(injections)
    base = root.parent if root.is_file() else root

    for file_path in files:
        relative = safe_relative(file_path, base)
        file_digests[relative] = file_sha256(file_path)
        raw_bytes = file_path.read_bytes()
        for line_index, raw_line in enumerate(raw_bytes.splitlines(), start=1):
            if not raw_line.strip():
                continue
            trace_source, trace_doc_id, trace_injection = _extract_trace(raw_line)
            try:
                text_line = raw_line.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                injection_id = trace_injection or configured_injections.get(
                    (trace_source, trace_doc_id, E_ENCODING_INVALID)
                )
                failures.append(
                    Failure(
                        source=trace_source,
                        doc_id=trace_doc_id,
                        stage="load_validate",
                        code=E_ENCODING_INVALID,
                        message=f"line is not valid UTF-8: {exc}",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                if injection_id:
                    injection_traces[injection_id] = {
                        "source": trace_source,
                        "doc_id": trace_doc_id,
                        "file": relative,
                        "line_no": line_index,
                        "code": E_ENCODING_INVALID,
                    }
                continue
            if line_index == 1:
                text_line = text_line.lstrip("\ufeff")

            try:
                parsed = json.loads(text_line)
            except json.JSONDecodeError as exc:
                injection_id = trace_injection or configured_injections.get(
                    (trace_source, trace_doc_id, E_ENCODING_INVALID)
                )
                failures.append(
                    Failure(
                        source=trace_source,
                        doc_id=trace_doc_id,
                        stage="load_validate",
                        code=E_ENCODING_INVALID,
                        message=f"line is not valid JSON: {exc.msg}",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                continue

            if not isinstance(parsed, dict):
                failures.append(
                    Failure(
                        source=trace_source,
                        doc_id=trace_doc_id,
                        stage="load_validate",
                        code=E_CONFIG_INVALID,
                        message="document record must be a JSON object",
                        injection_id=trace_injection,
                        file=relative,
                        line_no=line_index,
                    )
                )
                continue

            source = parsed.get("source")
            doc_id = parsed.get("doc_id")
            text = parsed.get("text")
            metadata = _norm_metadata(parsed.get("metadata", {}))
            injection_id = (
                metadata.get("injection_id")
                if isinstance(metadata, dict) and isinstance(metadata.get("injection_id"), str)
                else trace_injection
            )
            if not isinstance(source, str) or not source or not isinstance(doc_id, str) or not doc_id:
                failures.append(
                    Failure(
                        source=source if isinstance(source, str) else "",
                        doc_id=doc_id if isinstance(doc_id, str) else "",
                        stage="load_validate",
                        code=E_CONFIG_INVALID,
                        message="document must contain non-empty string source and doc_id fields",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                continue

            if source not in source_weights:
                injection_id = injection_id or configured_injections.get(
                    (source, doc_id, E_SOURCE_UNKNOWN)
                )
                failures.append(
                    Failure(
                        source=source,
                        doc_id=doc_id,
                        stage="load_validate",
                        code=E_SOURCE_UNKNOWN,
                        message=f"source {source!r} is absent from source_weights",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                if injection_id:
                    injection_traces[injection_id] = {
                        "source": source,
                        "doc_id": doc_id,
                        "file": relative,
                        "line_no": line_index,
                        "code": E_SOURCE_UNKNOWN,
                    }
                continue

            if doc_id in seen_ids:
                injection_id = injection_id or configured_injections.get(
                    (source, doc_id, E_DUP_DOC_ID)
                )
                failures.append(
                    Failure(
                        source=source,
                        doc_id=doc_id,
                        stage="load_validate",
                        code=E_DUP_DOC_ID,
                        message=f"duplicate doc_id; first seen in {seen_ids[doc_id]}",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                if injection_id:
                    injection_traces[injection_id] = {
                        "source": source,
                        "doc_id": doc_id,
                        "file": relative,
                        "line_no": line_index,
                        "code": E_DUP_DOC_ID,
                    }
                continue
            seen_ids[doc_id] = f"{relative}:{line_index}"

            if metadata is None:
                failures.append(
                    Failure(
                        source=source,
                        doc_id=doc_id,
                        stage="load_validate",
                        code=E_CONFIG_INVALID,
                        message="metadata must be a JSON object when present",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                continue
            if not _valid_text(text):
                injection_id = injection_id or configured_injections.get(
                    (source, doc_id, E_ENCODING_INVALID)
                )
                failures.append(
                    Failure(
                        source=source,
                        doc_id=doc_id,
                        stage="load_validate",
                        code=E_ENCODING_INVALID,
                        message="text must be a non-empty valid UTF-8 string",
                        injection_id=injection_id,
                        file=relative,
                        line_no=line_index,
                    )
                )
                if injection_id:
                    injection_traces[injection_id] = {
                        "source": source,
                        "doc_id": doc_id,
                        "file": relative,
                        "line_no": line_index,
                        "code": E_ENCODING_INVALID,
                    }
                continue

            documents.append(
                Document(
                    source=source,
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                    fingerprint=content_fingerprint(text),
                    file=relative,
                    line_no=line_index,
                    injection_id=injection_id,
                )
            )
            if injection_id:
                injection_traces[injection_id] = {
                    "source": source,
                    "doc_id": doc_id,
                    "file": relative,
                    "line_no": line_index,
                    "code": None,
                }

    documents.sort(key=lambda doc: (doc.source, doc.doc_id))
    failures.sort(key=Failure.sort_key)
    corpus_digest = sha256_json({"files": file_digests})
    return LoadResult(documents, failures, corpus_digest, file_digests, injection_traces)
