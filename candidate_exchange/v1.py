from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .errors import E_FIELD_TRUNCATED, BusinessError
from .formats import ParsedExchange
from .models import RawCandidate
from .util import file_sha256, stable_hash


V1_SCHEMA = "candidate-exchange-v1-fixed"
V1_FIELDS: tuple[tuple[str, int], ...] = (
    ("series_name", 16),
    ("frame_index", 8),
    ("class_code", 6),
    ("x1", 8),
    ("y1", 8),
    ("x2", 8),
    ("y2", 8),
    ("score_pm", 5),
    ("missing", 8),
)
V1_ROW_WIDTH = sum(width for _, width in V1_FIELDS)
V1_HEADER = (
    "#V1 fixed-width | "
    + "|".join(f"{name}({width})" for name, width in V1_FIELDS)
    + " | numeric fields are right aligned; missing uses C/X/Y/S flags"
)
_INTEGER = re.compile(r"^[0-9]+$")
_SIGNED_INTEGER = re.compile(r"^-?[0-9]+$")


def format_v1_row(
    series_name: str,
    frame_index: int,
    class_code: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    score_pm: int,
    missing: str = "--------",
) -> str:
    if len(series_name) > 16:
        raise ValueError("series_name exceeds v1 field width")
    fields = (
        f"{series_name:<16}",
        f"{frame_index:08d}",
        f"{class_code:>6d}",
        f"{x1:>8d}",
        f"{y1:>8d}",
        f"{x2:>8d}",
        f"{y2:>8d}",
        f"{score_pm:>5d}",
        f"{missing:<8}",
    )
    row = "".join(fields)
    if len(row) != V1_ROW_WIDTH:
        raise AssertionError("internal v1 row width error")
    return row


def serialize_v1_rows(rows: Iterable[RawCandidate]) -> str:
    lines = [V1_HEADER]
    for candidate in rows:
        lines.append(
            format_v1_row(
                candidate.series_name,
                candidate.frame_index,
                candidate.class_code,
                int(candidate.x1),
                int(candidate.y1),
                int(candidate.x2),
                int(candidate.y2),
                int(round(candidate.score * 1000)),
            )
        )
    return "\n".join(lines) + "\n"


def format_fingerprint() -> str:
    return stable_hash(
        {
            "schema": V1_SCHEMA,
            "fields": V1_FIELDS,
            "row_width": V1_ROW_WIDTH,
            "numeric_semantics": "integer_pixels_per_mille",
        }
    )


def _partial_location(line: str) -> tuple[str | None, int | None]:
    series = line[:16].strip() or None
    frame_text = line[16:24].strip()
    frame_index: int | None = None
    if frame_text.isdigit():
        frame_index = int(frame_text)
    return series, frame_index


def _parse_row(line: str, line_number: int) -> RawCandidate:
    series, frame_index = _partial_location(line)
    if len(line) != V1_ROW_WIDTH:
        raise BusinessError(
            E_FIELD_TRUNCATED,
            f"v1 line {line_number} has {len(line)} bytes, expected {V1_ROW_WIDTH}",
            "parsing_v1",
            series_name=series,
            frame_index=frame_index,
            details={
                "line_number": line_number,
                "actual_width": len(line),
                "expected_width": V1_ROW_WIDTH,
            },
        )
    values: dict[str, str] = {}
    offset = 0
    for name, width in V1_FIELDS:
        values[name] = line[offset : offset + width]
        offset += width
    missing = values["missing"].strip()
    missing_flags: list[str] = []
    if missing and missing != "--------":
        for marker, field_name in (
            ("C", "class_code"),
            ("X", "x"),
            ("Y", "y"),
            ("S", "score"),
        ):
            if marker in missing[:4]:
                missing_flags.append(field_name)
    if not values["series_name"].strip():
        missing_flags.append("series_name")
    for field_name in ("frame_index", "class_code", "x1", "y1", "x2", "y2", "score_pm"):
        if not values[field_name].strip():
            missing_flags.append(field_name)
    if missing_flags:
        raise BusinessError(
            E_FIELD_TRUNCATED,
            f"v1 line {line_number} is missing required fields: {sorted(set(missing_flags))}",
            "parsing_v1",
            series_name=series,
            frame_index=frame_index,
            details={"line_number": line_number, "missing_fields": sorted(set(missing_flags))},
        )
    if not _INTEGER.fullmatch(values["frame_index"].strip()):
        raise BusinessError(
            E_FIELD_TRUNCATED,
            f"v1 frame field is misaligned on line {line_number}",
            "parsing_v1",
            series_name=series,
            frame_index=frame_index,
            details={"line_number": line_number, "field": "frame_index"},
        )
    numeric_fields: dict[str, int] = {}
    for name in ("class_code", "x1", "y1", "x2", "y2", "score_pm"):
        text = values[name].strip()
        pattern = _SIGNED_INTEGER if name in {"x1", "y1", "x2", "y2"} else _INTEGER
        if not pattern.fullmatch(text):
            raise BusinessError(
                E_FIELD_TRUNCATED,
                f"v1 field {name} is misaligned on line {line_number}",
                "parsing_v1",
                series_name=series,
                frame_index=frame_index,
                details={"line_number": line_number, "field": name, "value": text},
            )
        numeric_fields[name] = int(text)
    return RawCandidate(
        series_name=values["series_name"].strip(),
        frame_index=int(values["frame_index"].strip()),
        class_code=numeric_fields["class_code"],
        class_name=None,
        x1=float(numeric_fields["x1"]),
        y1=float(numeric_fields["y1"]),
        x2=float(numeric_fields["x2"]),
        y2=float(numeric_fields["y2"]),
        score=numeric_fields["score_pm"] / 1000.0,
        precision_label="per_mille_exact",
        source_version="v1",
        source_ordinal=line_number,
        missing_fields=tuple(sorted(set(missing_flags))),
        extra={"score_pm": numeric_fields["score_pm"], "line_number": line_number},
    )


def parse_v1(path: str | Path) -> ParsedExchange:
    file_path = Path(path)
    result = ParsedExchange(
        version="v1",
        path=str(file_path),
        file_sha256=file_sha256(file_path),
        format_fingerprint=format_fingerprint(),
    )
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            try:
                candidate = _parse_row(line, line_number)
            except BusinessError as exc:
                result.failures.append(exc)
                continue
            result.add_candidate(candidate)
    return result
