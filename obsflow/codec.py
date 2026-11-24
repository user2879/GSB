"""Encoders / decoders and strong validation for the three historical formats.

On-disk layout (JSON Lines, UTF-8):

* line 1 is a header object ``{"obsflow": "record-stream", "version": 1|2|3}``;
* every subsequent line is one record object.

Record fields by version:

* v1: ``seq``, ``value``
* v2: v1 + ``source_id``, ``ts_ms``
* v3: v2 + ``metric_key``, ``scale``, ``source_format``

Missing historical fields are *filled* from the source declaration and the
fill is recorded on the normalised record; original values are never
overwritten.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any, Optional

from .errors import FormatError, IntegrityError, OrderError
from .models import (
    SUPPORTED_FORMATS,
    SUPPORTED_SCALES,
    NormalizedRecord,
    SourceDeclaration,
)

HEADER_KEY = "obsflow"
HEADER_VALUE = "record-stream"

# Required raw fields per format version.
REQUIRED_FIELDS: dict[int, tuple[str, ...]] = {
    1: ("seq", "value"),
    2: ("seq", "value", "source_id", "ts_ms"),
    3: ("seq", "value", "source_id", "ts_ms", "metric_key", "scale",
        "source_format"),
}

_INT_FIELDS = ("seq", "value", "ts_ms", "scale", "source_format")
_STR_FIELDS = ("source_id", "metric_key")


def encode_header(version: int) -> str:
    if version not in SUPPORTED_FORMATS:
        raise FormatError(f"unsupported format version {version}",
                          field="version")
    return json.dumps({HEADER_KEY: HEADER_VALUE, "version": version},
                      sort_keys=True)


def encode_record(version: int, normalized: NormalizedRecord) -> str:
    """Serialise a normalised record into the requested historical version."""
    if version not in SUPPORTED_FORMATS:
        raise FormatError(f"unsupported format version {version}",
                          field="version")
    raw: dict[str, Any] = {"seq": normalized.seq, "value": normalized.raw_value}
    if version >= 2:
        raw["source_id"] = normalized.source_id
        raw["ts_ms"] = normalized.ts_ms
    if version >= 3:
        raw["metric_key"] = normalized.metric_key
        raw["scale"] = normalized.record_scale
        raw["source_format"] = normalized.source_format
    return json.dumps(raw, sort_keys=True)


def write_stream(path: str, version: int,
                 records: Iterable[NormalizedRecord]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(encode_header(version) + "\n")
        for record in records:
            handle.write(encode_record(version, record) + "\n")


def _fail(field: str, message: str, line_no: int,
          code: type[ObsFlowErrorT] = FormatError) -> "ObsFlowErrorT":
    return code(message, field=field, line_no=line_no)


# Type alias used only for the helper above.
ObsFlowErrorT = FormatError


def _read_header(lines: list[str], path: str) -> tuple[int, int]:
    if not lines:
        raise IntegrityError(f"empty stream file: {path}", field="file",
                             line_no=1)
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise IntegrityError(
            f"header is not valid JSON in {path}: {exc.msg}",
            field="header", line_no=1) from exc
    if not isinstance(header, dict) or header.get(HEADER_KEY) != HEADER_VALUE:
        raise FormatError(f"missing or invalid stream header in {path}",
                          field="header", line_no=1)
    version = header.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise FormatError("header version must be an integer",
                          field="version", line_no=1)
    if version not in SUPPORTED_FORMATS:
        raise FormatError(f"unknown format version {version}",
                          field="version", line_no=1)
    return version, 1


def _validate_raw(raw: Any, version: int, line_no: int,
                  declaration: SourceDeclaration) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _fail("record", "record line must be a JSON object", line_no)

    for name in REQUIRED_FIELDS[version]:
        if name not in raw:
            raise _fail(name, f"required field '{name}' is missing", line_no)

    for name in _INT_FIELDS:
        if name in raw:
            value = raw[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise _fail(name,
                            f"field '{name}' must be an integer", line_no)
    for name in _STR_FIELDS:
        if name in raw:
            value = raw[name]
            if not isinstance(value, str) or not value:
                raise _fail(name,
                            f"field '{name}' must be a non-empty string",
                            line_no)

    if raw["seq"] < 0:
        raise _fail("seq", "observation sequence must be non-negative",
                    line_no)
    if version >= 2 and raw["ts_ms"] < 0:
        raise _fail("ts_ms", "collection timestamp must not be negative",
                    line_no)
    if version >= 3 and raw["scale"] not in SUPPORTED_SCALES:
        raise _fail("scale",
                    f"scale {raw['scale']} is not in supported set "
                    f"{list(SUPPORTED_SCALES)}", line_no)
    if version >= 3 and raw["source_format"] not in SUPPORTED_FORMATS:
        raise _fail("source_format",
                    f"source_format {raw['source_format']} is unsupported",
                    line_no)
    if version >= 3 and raw["source_format"] != version:
        raise _fail("source_format",
                    "record source_format does not match stream version",
                    line_no)
    if version >= 2 and raw["source_id"] != declaration.source_id:
        raise _fail("source_id",
                    f"record source_id {raw['source_id']!r} does not match "
                    f"declaration {declaration.source_id!r}", line_no)
    return raw


def _normalize(raw: dict[str, Any], version: int, line_no: int,
               declaration: SourceDeclaration) -> NormalizedRecord:
    filled: list[str] = []
    basis: dict[str, str] = {}

    if version >= 2:
        source_id = raw["source_id"]
        ts_ms = raw["ts_ms"]
    else:
        source_id = declaration.source_id
        ts_ms = 0
        filled.extend(("source_id", "ts_ms"))
        basis["source_id"] = (
            f"filled from source declaration {declaration.source_id!r} "
            "(v1 has no source_id)")
        basis["ts_ms"] = ("filled with epoch 0 because v1 carries no "
                          "collection timestamp")

    if version >= 3:
        metric_key_raw = raw["metric_key"]
        record_scale = raw["scale"]
        source_format = raw["source_format"]
    else:
        metric_key_raw = declaration.default_metric_key
        record_scale = declaration.default_scale
        source_format = version
        filled.extend(("metric_key", "scale", "source_format"))
        basis["metric_key"] = (
            f"filled from declaration default_metric_key "
            f"{declaration.default_metric_key!r} (v{version} has no key)")
        basis["scale"] = (
            f"filled from declaration default_scale "
            f"{declaration.default_scale} (v{version} has no scale)")
        basis["source_format"] = (
            f"filled with stream header version {version}")

    metric_key = declaration.metric_mapping.get(metric_key_raw, metric_key_raw)
    if metric_key != metric_key_raw and "metric_key" not in filled:
        # Mapping is a normalisation step, not a fill; record the basis only.
        basis["metric_key_mapping"] = (
            f"mapped raw key {metric_key_raw!r} -> {metric_key!r} via "
            "declaration metric_mapping")

    return NormalizedRecord(
        source_id=source_id,
        seq=raw["seq"],
        ts_ms=ts_ms,
        metric_key=metric_key,
        raw_value=raw["value"],
        record_scale=record_scale,
        source_format=source_format,
        filled_fields=tuple(filled),
        fill_basis=basis,
        line_no=line_no,
    )


def decode_stream(
    path: str,
    declaration: SourceDeclaration,
    *,
    order_check: bool = True,
) -> Iterator[NormalizedRecord]:
    """Lazily decode and validate one stream file.

    ``order_check`` enables per-file strictly increasing sequence validation.
    It is always performed during load-time validation (E_FORMAT hierarchy,
    specifically E_ORDER) and can be reused by the injection pipeline.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise IntegrityError(f"cannot read stream file {path}: {exc}",
                             field="file", line_no=1) from exc

    version, header_line = _read_header(lines, path)
    if version != declaration.format_version:
        raise FormatError(
            f"stream version {version} does not match declared format "
            f"version {declaration.format_version}",
            field="version", line_no=header_line)

    previous_seq: Optional[int] = None
    for offset, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntegrityError(
                f"malformed JSON record in {path}: {exc.msg}",
                field="record", line_no=offset) from exc
        raw = _validate_raw(raw, version, offset, declaration)
        if order_check and previous_seq is not None and \
                raw["seq"] <= previous_seq:
            raise OrderError(
                f"observation sequence regressed: {previous_seq} -> "
                f"{raw['seq']}",
                field="seq", line_no=offset)
        previous_seq = raw["seq"]
        yield _normalize(raw, version, offset, declaration)


def load_stream(path: str,
                declaration: SourceDeclaration) -> list[NormalizedRecord]:
    """Eagerly load, validate and normalise one stream file."""
    return list(decode_stream(path, declaration, order_check=True))


def resolve_stream_path(base_dir: str, file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    return os.path.normpath(os.path.join(base_dir, file_path))
