"""Content-based v1 JSONL and v2 batch JSON ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import make_sync_marker, normalize_event
from .errors import (
    E_EMPTY_INPUT,
    E_ENCODING,
    E_FORMAT_UNKNOWN,
    E_IO,
    E_MALFORMED_EVENT,
    E_MALFORMED_JSON,
    business_error,
)


@dataclass
class ParsedOccurrence:
    core: dict[str, Any]
    extension: dict[str, Any]
    source_format: str
    source: str
    position: str
    source_order: int
    position_order: int
    sync_marker: str
    raw_event: dict[str, Any] = field(repr=False)

    def as_provenance(self) -> dict[str, Any]:
        return {
            "sync_marker": self.sync_marker,
            "source_format": self.source_format,
            "source": self.source,
            "position": self.position,
            "source_order": self.source_order,
            "position_order": self.position_order,
            "original_ts": self.extension.get("original_ts"),
            "severity": self.extension.get("severity"),
            "source_name": self.extension.get("source"),
            "event_core": self.core,
        }


@dataclass
class ParsedFile:
    path: str
    source_order: int
    detected_format: str | None
    sha256: str
    size_bytes: int
    occurrences: list[ParsedOccurrence] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.errors)


def detect_format(text: str) -> str | None:
    """Detect input structure without consulting path names or extensions."""

    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped:
        return None
    first = stripped[0]
    if first == "[":
        return "v2"
    if first == "{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return "v1"
        if isinstance(parsed, dict):
            return "v1"
        return None
    return None


def _decode_error(path: Path, source_order: int, exc: UnicodeDecodeError) -> ParsedFile:
    return ParsedFile(
        path=str(path),
        source_order=source_order,
        detected_format=None,
        sha256="",
        size_bytes=path.stat().st_size if path.exists() else 0,
        errors=[
            business_error(
                E_ENCODING,
                f"input is not valid UTF-8: {exc}",
                source=str(path),
                position="file:1",
            )
        ],
    )


def parse_file(path_like: str | Path, source_order: int = 0) -> ParsedFile:
    path = Path(path_like)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ParsedFile(
            path=str(path),
            source_order=source_order,
            detected_format=None,
            sha256="",
            size_bytes=0,
            errors=[
                business_error(
                    E_IO,
                    f"cannot read input: {exc}",
                    source=str(path),
                    position="file:1",
                )
            ],
        )

    digest = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        result = _decode_error(path, source_order, exc)
        result.sha256 = digest
        result.size_bytes = len(data)
        return result

    if not text.strip():
        return ParsedFile(
            path=str(path),
            source_order=source_order,
            detected_format=None,
            sha256=digest,
            size_bytes=len(data),
            errors=[
                business_error(
                    E_EMPTY_INPUT,
                    "input contains no JSON content",
                    source=str(path),
                    position="file:1",
                )
            ],
        )

    source_format = detect_format(text)
    if source_format is None:
        return ParsedFile(
            path=str(path),
            source_order=source_order,
            detected_format=None,
            sha256=digest,
            size_bytes=len(data),
            errors=[
                business_error(
                    E_FORMAT_UNKNOWN,
                    "content is neither a JSON object stream nor a JSON array",
                    source=str(path),
                    position="file:1",
                )
            ],
        )

    result = ParsedFile(
        path=str(path),
        source_order=source_order,
        detected_format=source_format,
        sha256=digest,
        size_bytes=len(data),
    )
    if source_format == "v2":
        _parse_v2(text, result)
    else:
        _parse_v1(text, result)
    return result


def _parse_v1(text: str, result: ParsedFile) -> None:
    # A single pretty-printed object is also valid v1 content. A multi-event
    # JSONL stream cannot be parsed as one JSON value and falls through below.
    try:
        single = json.loads(text.strip())
    except json.JSONDecodeError:
        single = None
    if isinstance(single, dict):
        _append_occurrence(single, "v1", result, "line:1", 0)
        return

    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        position = f"line:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            result.errors.append(
                business_error(
                    E_MALFORMED_JSON,
                    f"invalid JSONL line: {exc.msg}",
                    source_format="v1",
                    source=result.path,
                    position=position,
                    details={"line": line_number, "column": exc.colno},
                )
            )
            continue
        if not isinstance(raw, dict):
            result.errors.append(
                business_error(
                    E_MALFORMED_EVENT,
                    "JSONL value must be an object",
                    source_format="v1",
                    source=result.path,
                    position=position,
                )
            )
            continue
        _append_occurrence(raw, "v1", result, position, line_number - 1)


def _parse_v2(text: str, result: ParsedFile) -> None:
    try:
        raw_batch = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(
            business_error(
                E_MALFORMED_JSON,
                f"invalid batch JSON: {exc.msg}",
                source_format="v2",
                source=result.path,
                position="file:1",
                details={"line": exc.lineno, "column": exc.colno},
            )
        )
        return
    if not isinstance(raw_batch, list):
        result.errors.append(
            business_error(
                E_MALFORMED_EVENT,
                "v2 batch root must be an array",
                source_format="v2",
                source=result.path,
                position="file:1",
            )
        )
        return
    for index, raw in enumerate(raw_batch):
        position = f"element:{index}"
        if not isinstance(raw, dict):
            result.errors.append(
                business_error(
                    E_MALFORMED_EVENT,
                    "batch array element must be an object",
                    source_format="v2",
                    source=result.path,
                    position=position,
                    details={"index": index},
                )
            )
            continue
        _append_occurrence(raw, "v2", result, position, index)


def _append_occurrence(
    raw: dict[str, Any],
    source_format: str,
    result: ParsedFile,
    position: str,
    position_order: int,
) -> None:
    core, extension, error = normalize_event(
        raw,
        source_format=source_format,
        source=result.path,
        position=position,
    )
    if error is not None:
        result.errors.append(error)
        return
    marker = make_sync_marker(
        event_id=core["event_id"],
        source_format=source_format,
        position=position,
        original_ts=extension.get("original_ts"),
        source=Path(result.path).name,
    )
    result.occurrences.append(
        ParsedOccurrence(
            core=core,
            extension=extension,
            source_format=source_format,
            source=result.path,
            position=position,
            source_order=result.source_order,
            position_order=position_order,
            sync_marker=marker,
            raw_event=raw,
        )
    )


def parse_files(
    paths: Iterable[str | Path],
    *,
    workers: int = 1,
) -> list[ParsedFile]:
    """Parse streams concurrently and return results in deterministic input order."""

    path_list = [str(Path(path)) for path in paths]
    if workers <= 1 or len(path_list) <= 1:
        return [parse_file(path, index) for index, path in enumerate(path_list)]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(path_list)))) as pool:
        futures = [
            pool.submit(parse_file, path, index)
            for index, path in enumerate(path_list)
        ]
        return [future.result() for future in futures]
