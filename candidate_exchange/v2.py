from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .errors import E_FIELD_TRUNCATED, E_VERSION_MISMATCH, BusinessError
from .formats import ParsedExchange
from .models import RawCandidate
from .util import file_sha256, stable_hash


V2_SCHEMA = "candidate-exchange-v2-json"
V2_FORMAT_VERSION = "v2"
_REQUIRED_KEYS = (
    "series_name",
    "frame_index",
    "class_code",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "precision_label",
)


def format_fingerprint() -> str:
    return stable_hash(
        {
            "schema": V2_SCHEMA,
            "format_version": V2_FORMAT_VERSION,
            "required_keys": _REQUIRED_KEYS,
            "numeric_semantics": "finite_float",
        }
    )


def _iter_frames(document: dict[str, Any]) -> Iterable[dict[str, Any]]:
    frames = document.get("frames")
    if isinstance(frames, list):
        for item in frames:
            if isinstance(item, dict):
                yield item
        return
    series = document.get("series")
    if isinstance(series, list):
        for group in series:
            if not isinstance(group, dict):
                continue
            series_name = group.get("series_name")
            group_frames = group.get("frames")
            if not isinstance(group_frames, list):
                continue
            for item in group_frames:
                if not isinstance(item, dict):
                    continue
                frame = dict(item)
                frame.setdefault("series_name", series_name)
                yield frame


def _frame_location(frame: dict[str, Any]) -> tuple[str | None, int | None]:
    series_name = frame.get("series_name")
    frame_index = frame.get("frame_index")
    return (
        series_name if isinstance(series_name, str) else None,
        frame_index if isinstance(frame_index, int) and not isinstance(frame_index, bool) else None,
    )


def parse_v2(path: str | Path) -> ParsedExchange:
    file_path = Path(path)
    result = ParsedExchange(
        version="v2",
        path=str(file_path),
        file_sha256=file_sha256(file_path),
        format_fingerprint=format_fingerprint(),
    )
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception as exc:
        result.failures.append(
            BusinessError(
                E_FIELD_TRUNCATED,
                f"v2 JSON cannot be parsed: {exc}",
                "parsing_v2",
                series_name="<file>",
                details={"exception_type": exc.__class__.__name__},
            )
        )
        return result
    if not isinstance(document, dict):
        result.failures.append(
            BusinessError(
                E_VERSION_MISMATCH,
                "v2 JSON root must be an object",
                "parsing_v2",
                series_name="<file>",
            )
        )
        return result
    if document.get("format_version") != V2_FORMAT_VERSION:
        result.failures.append(
            BusinessError(
                E_VERSION_MISMATCH,
                f"unexpected v2 format_version: {document.get('format_version')!r}",
                "parsing_v2",
                series_name="<file>",
            )
        )
        return result
    ordinal = 0
    for frame in _iter_frames(document):
        ordinal += 1
        series_name, frame_index = _frame_location(frame)
        candidates = frame.get("candidates", [])
        if not isinstance(candidates, list):
            result.failures.append(
                BusinessError(
                    E_FIELD_TRUNCATED,
                    "v2 frame candidates must be a list",
                    "parsing_v2",
                    series_name=series_name,
                    frame_index=frame_index,
                )
            )
            continue
        for candidate_index, item in enumerate(candidates):
            if not isinstance(item, dict):
                result.failures.append(
                    BusinessError(
                        E_FIELD_TRUNCATED,
                        "v2 candidate must be an object",
                        "parsing_v2",
                        series_name=series_name,
                        frame_index=frame_index,
                        details={"candidate_index": candidate_index},
                    )
                )
                continue
            candidate = dict(item)
            candidate.setdefault("series_name", series_name)
            candidate.setdefault("frame_index", frame_index)
            missing = [
                key for key in _REQUIRED_KEYS if key not in candidate or candidate[key] is None
            ]
            if missing:
                result.failures.append(
                    BusinessError(
                        E_FIELD_TRUNCATED,
                        f"v2 candidate missing required fields: {missing}",
                        "parsing_v2",
                        series_name=series_name,
                        frame_index=frame_index,
                        details={"candidate_index": candidate_index, "missing_fields": missing},
                    )
                )
                continue
            if (
                not isinstance(candidate["series_name"], str)
                or not candidate["series_name"]
                or not isinstance(candidate["frame_index"], int)
                or isinstance(candidate["frame_index"], bool)
                or not isinstance(candidate["class_code"], int)
                or isinstance(candidate["class_code"], bool)
                or not isinstance(candidate["class_name"], str)
                or not candidate["class_name"]
                or not isinstance(candidate["precision_label"], str)
                or not candidate["precision_label"]
            ):
                result.failures.append(
                    BusinessError(
                        E_VERSION_MISMATCH,
                        "v2 candidate scalar types are inconsistent",
                        "parsing_v2",
                        series_name=series_name,
                        frame_index=frame_index,
                        details={"candidate_index": candidate_index},
                    )
                )
                continue
            numeric: dict[str, float] = {}
            invalid_number = False
            for name in ("x1", "y1", "x2", "y2", "score"):
                value = candidate[name]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    invalid_number = True
                    break
                numeric[name] = float(value)
            if invalid_number:
                result.failures.append(
                    BusinessError(
                        E_VERSION_MISMATCH,
                        "v2 candidate contains a non-finite or non-numeric value",
                        "parsing_v2",
                        series_name=candidate["series_name"],
                        frame_index=candidate["frame_index"],
                        details={"candidate_index": candidate_index},
                    )
                )
                continue
            result.add_candidate(
                RawCandidate(
                    series_name=candidate["series_name"],
                    frame_index=candidate["frame_index"],
                    class_code=candidate["class_code"],
                    class_name=candidate["class_name"],
                    x1=numeric["x1"],
                    y1=numeric["y1"],
                    x2=numeric["x2"],
                    y2=numeric["y2"],
                    score=numeric["score"],
                    precision_label=candidate["precision_label"],
                    source_version="v2",
                    source_ordinal=ordinal * 1000 + candidate_index,
                    missing_fields=(),
                    extra=dict(candidate["extra"])
                    if isinstance(candidate.get("extra"), dict)
                    else {},
                )
            )
    return result
