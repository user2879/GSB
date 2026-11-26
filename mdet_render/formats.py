"""Bidirectional v1 fixed-width and v2 JSON intermediate record formats.

v1 is deliberately small and line-oriented.  v2 is lossless JSON.  A v2 -> v1
conversion emits a sidecar containing the exact original v2 document and the
truncation residuals; re-upgrading with that sidecar restores the original
fields exactly.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import BusinessError
from .util import canonical_json, read_json_file, sha256_hex

NOISE_CODE_TO_NAME: dict[int, str] = {
    0: "gaussian",
    1: "uniform",
    2: "perlin",
    3: "none",
}
NOISE_NAME_TO_CODE: dict[str, int] = {name: code for code, name in NOISE_CODE_TO_NAME.items()}

# name, width, alignment
V1_COLUMNS: tuple[tuple[str, int, str], ...] = (
    ("stage", 20, "left"),
    ("frame", 8, "right"),
    ("step", 8, "right"),
    ("block", 8, "right"),
    ("guide_permille", 12, "right"),
    ("noise_code", 8, "right"),
    ("quant_bits", 8, "right"),
    ("quant_level", 8, "right"),
    ("residual", 16, "right"),
    ("digest", 64, "left"),
)
V1_SEPARATOR = "|"
V1_LINE_WIDTH = sum(width for _, width, _ in V1_COLUMNS) + len(V1_COLUMNS) - 1


@dataclass
class StageRecord:
    stage: str
    frame: int = 0
    step: int = 0
    block: int = 0
    guide_strength: float | None = 0.0
    noise_scheme: str | None = "none"
    precision_tag: str | None = ""
    quantization_bits: int = 0
    quantization_level: int = 0
    residual: int = 0
    digest: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    source_version: int = 2
    missing_fields: list[str] = field(default_factory=list)
    restored_from_sidecar: bool = False

    def as_v2_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"stage": self.stage}
        optional = {
            "frame": self.frame,
            "step": self.step,
            "block": self.block,
            "guide_strength": self.guide_strength,
            "noise_scheme": self.noise_scheme,
            "precision_tag": self.precision_tag,
            "quantization_bits": self.quantization_bits,
            "quantization_level": self.quantization_level,
            "residual": self.residual,
            "digest": self.digest,
        }
        for key, value in optional.items():
            if key not in self.missing_fields:
                record[key] = value
        record.update(self.extra)
        return record


def _format_cell(value: Any, width: int, alignment: str) -> str:
    text = "" if value is None else str(value)
    if len(text) > width:
        raise BusinessError("E_FORMAT_V1_OVERFLOW", "v1 field exceeds its fixed width",
                            {"value": text, "width": width})
    return text.ljust(width) if alignment == "left" else text.rjust(width)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BusinessError("E_FORMAT_V1", f"v1 {name} must be an integer", {"value": value})
    return value


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def _precision_tag(guide_permille: int, noise_code: int) -> str:
    return (f"permille={guide_permille};noise_code={noise_code};"
            f"noise_scheme={NOISE_CODE_TO_NAME[noise_code]}")


def to_v1_lines(records: Iterable[StageRecord]) -> str:
    lines: list[str] = []
    for record in records:
        permille: int | None
        if record.guide_strength is None:
            permille = None
        else:
            permille = _round_half_away(float(record.guide_strength) * 1000.0)
        if record.noise_scheme is None:
            noise_code: int | None = None
        else:
            try:
                noise_code = NOISE_NAME_TO_CODE[record.noise_scheme]
            except KeyError as exc:
                raise BusinessError("E_FORMAT_NOISE_SCHEME", "noise scheme has no v1 code",
                                    {"noise_scheme": record.noise_scheme,
                                     "known": sorted(NOISE_NAME_TO_CODE)}) from exc
        cells = [
            _format_cell(record.stage, 20, "left"),
            _format_cell(record.frame if "frame" not in record.missing_fields else None, 8, "right"),
            _format_cell(record.step if "step" not in record.missing_fields else None, 8, "right"),
            _format_cell(record.block if "block" not in record.missing_fields else None, 8, "right"),
            _format_cell(permille, 12, "right"),
            _format_cell(noise_code, 8, "right"),
            _format_cell(record.quantization_bits if "quantization_bits" not in record.missing_fields else None,
                         8, "right"),
            _format_cell(record.quantization_level if "quantization_level" not in record.missing_fields else None,
                         8, "right"),
            _format_cell(record.residual if "residual" not in record.missing_fields else None, 16, "right"),
            _format_cell(record.digest if "digest" not in record.missing_fields else None, 64, "left"),
        ]
        lines.append(V1_SEPARATOR.join(cells))
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_v1_line(line: str, line_number: int) -> StageRecord:
    if len(line) > V1_LINE_WIDTH:
        raise BusinessError("E_FORMAT_V1_WIDTH", "v1 line is wider than the fixed schema",
                            {"line": line_number, "width": len(line), "expected": V1_LINE_WIDTH})
    padded = line.ljust(V1_LINE_WIDTH)
    raw_cells = padded.split(V1_SEPARATOR)
    if len(raw_cells) != len(V1_COLUMNS):
        raise BusinessError("E_FORMAT_V1_WIDTH", "v1 line has the wrong number of fixed columns",
                            {"line": line_number, "columns": len(raw_cells), "expected": len(V1_COLUMNS)})
    cells: dict[str, str] = {}
    offset = 0
    for name, width, _ in V1_COLUMNS:
        cells[name] = padded[offset:offset + width].strip()
        offset += width + 1
    stage = cells["stage"]
    if not stage:
        raise BusinessError("E_FORMAT_V1", "v1 stage field is required", {"line": line_number})
    missing: list[str] = []

    def optional_int(name: str, default: int) -> int:
        if not cells[name]:
            missing.append(name)
            return default
        try:
            return int(cells[name], 10)
        except ValueError as exc:
            raise BusinessError("E_FORMAT_V1", f"v1 {name} is not an integer",
                                {"line": line_number, "value": cells[name]}) from exc

    frame = optional_int("frame", 0)
    step = optional_int("step", 0)
    block = optional_int("block", 0)
    if cells["guide_permille"]:
        try:
            guide_permille = int(cells["guide_permille"], 10)
        except ValueError as exc:
            raise BusinessError("E_FORMAT_V1", "v1 guide_permille is not an integer",
                                {"line": line_number, "value": cells["guide_permille"]}) from exc
        guide: float | None = guide_permille / 1000.0
    else:
        guide_permille = 0
        guide = None
        missing.append("guide_strength")
    if cells["noise_code"]:
        noise_code = optional_int("noise_code", 0)
        if noise_code not in NOISE_CODE_TO_NAME:
            raise BusinessError("E_FORMAT_NOISE_CODE", "v1 noise code is outside the mapping table",
                                {"line": line_number, "noise_code": noise_code,
                                 "known": sorted(NOISE_CODE_TO_NAME)})
        noise_scheme: str | None = NOISE_CODE_TO_NAME[noise_code]
    else:
        noise_code = 0
        noise_scheme = None
        missing.append("noise_scheme")
    quant_bits = optional_int("quant_bits", 0)
    quant_level = optional_int("quant_level", 0)
    residual = optional_int("residual", 0)
    digest = cells["digest"]
    if not digest:
        missing.append("digest")
    precision = None if guide is None and noise_scheme is None else _precision_tag(guide_permille, noise_code)
    return StageRecord(
        stage=stage,
        frame=frame,
        step=step,
        block=block,
        guide_strength=guide,
        noise_scheme=noise_scheme,
        precision_tag=precision,
        quantization_bits=quant_bits,
        quantization_level=quant_level,
        residual=residual,
        digest=digest,
        source_version=1,
        missing_fields=sorted(set(missing)),
    )


def parse_v1(text: str) -> list[StageRecord]:
    records: list[StageRecord] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        records.append(_parse_v1_line(line.rstrip(), index))
    return records


def _validate_v2_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise BusinessError("E_FORMAT_V2", "v2 record must be an object", {"index": index})
    if not isinstance(record.get("stage"), str) or not record["stage"]:
        raise BusinessError("E_FORMAT_V2", "v2 record.stage is required", {"index": index})
    for field_name in ("frame", "step", "block", "quantization_bits", "quantization_level",
                       "residual"):
        if field_name in record and (isinstance(record[field_name], bool)
                                     or not isinstance(record[field_name], int)):
            raise BusinessError("E_FORMAT_V2", f"{field_name} must be an integer",
                                {"index": index, "field": field_name, "value": record[field_name]})
    if "guide_strength" in record:
        value = record["guide_strength"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise BusinessError("E_FORMAT_V2", "guide_strength must be a finite number",
                                {"index": index, "value": value})
    for field_name in ("noise_scheme", "precision_tag"):
        if field_name in record and record[field_name] is not None and not isinstance(
                record[field_name], str):
            raise BusinessError("E_FORMAT_V2", f"{field_name} must be a string or null",
                                {"index": index, "field": field_name})
    if "digest" in record and not isinstance(record["digest"], str):
        raise BusinessError("E_FORMAT_V2", "digest must be a string", {"index": index})
    return record


def parse_v2(document: dict[str, Any]) -> list[StageRecord]:
    if not isinstance(document, dict):
        raise BusinessError("E_FORMAT_V2", "v2 document must be an object")
    if document.get("format_version") != 2:
        raise BusinessError("E_FORMAT_V2", "v2 format_version must be 2",
                            {"format_version": document.get("format_version")})
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise BusinessError("E_FORMAT_V2", "v2 records must be an array")
    known = {
        "stage", "frame", "step", "block", "guide_strength", "noise_scheme", "precision_tag",
        "quantization_bits", "quantization_level", "residual", "digest",
    }
    records: list[StageRecord] = []
    for index, raw in enumerate(raw_records):
        raw = _validate_v2_record(raw, index)
        missing = [name for name in (
            "frame", "step", "block", "guide_strength", "noise_scheme", "precision_tag",
            "quantization_bits", "quantization_level", "residual", "digest",
        ) if name not in raw]
        records.append(
            StageRecord(
                stage=raw["stage"],
                frame=raw.get("frame", 0),
                step=raw.get("step", 0),
                block=raw.get("block", 0),
                guide_strength=raw.get("guide_strength"),
                noise_scheme=raw.get("noise_scheme"),
                precision_tag=raw.get("precision_tag"),
                quantization_bits=raw.get("quantization_bits", 0),
                quantization_level=raw.get("quantization_level", 0),
                residual=raw.get("residual", 0),
                digest=raw.get("digest", ""),
                extra={key: copy.deepcopy(value) for key, value in raw.items() if key not in known},
                source_version=2,
                missing_fields=missing,
            )
        )
    return records


def to_v2_document(records: Iterable[StageRecord]) -> dict[str, Any]:
    return {"format_version": 2, "records": [record.as_v2_dict() for record in records]}


def v2_to_v1(document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert a v2 document to v1 and exact-restoration sidecar metadata."""
    records = parse_v2(document)
    v1_text = to_v1_lines(records)
    lines = v1_text.splitlines()
    sidecar_records: list[dict[str, Any]] = []
    for index, (record, line) in enumerate(zip(records, lines)):
        permille = None if record.guide_strength is None else _round_half_away(float(record.guide_strength) * 1000.0)
        rounding_residual = None if permille is None else float(record.guide_strength) - permille / 1000.0
        precision = record.precision_tag
        if precision is None and permille is not None and record.noise_scheme is not None:
            precision = _precision_tag(permille, NOISE_NAME_TO_CODE[record.noise_scheme])
        overwritten: dict[str, Any] = {
            "guide_strength": record.guide_strength,
            "noise_scheme": record.noise_scheme,
            "precision_tag": record.precision_tag,
            "extra": copy.deepcopy(record.extra),
            "missing_fields": list(record.missing_fields),
        }
        sidecar_records.append({
            "record_index": index,
            "stage": record.stage,
            "frame": record.frame,
            "step": record.step,
            "block": record.block,
            "v1_line_sha256": sha256_hex(line),
            "precision_tag": precision,
            "guide_rounding_residual": rounding_residual,
            "truncated_residual": rounding_residual,
            "overwritten_fields": overwritten,
            "original_v2_record": copy.deepcopy(document["records"][index]),
        })
    sidecar = {
        "schema_version": 1,
        "direction": "v2-to-v1",
        "record_count": len(records),
        "v1_document_sha256": sha256_hex(v1_text),
        "records": sidecar_records,
        "original_v2_document": copy.deepcopy(document),
    }
    return v1_text, sidecar


def v1_to_v2(v1_text: str, sidecar: dict[str, Any] | None = None) -> dict[str, Any]:
    """Upgrade v1 to v2, restoring exact v2 fields from a matching sidecar."""
    if sidecar is not None:
        if not isinstance(sidecar, dict):
            raise BusinessError("E_FORMAT_SIDECAR", "sidecar must be a JSON object")
        original = sidecar.get("original_v2_document")
        if (isinstance(original, dict) and sidecar.get("v1_document_sha256") == sha256_hex(v1_text)):
            return copy.deepcopy(original)
        raw_records = sidecar.get("records")
        if raw_records is not None and not isinstance(raw_records, list):
            raise BusinessError("E_FORMAT_SIDECAR", "sidecar records must be an array")
    else:
        raw_records = None
    records = parse_v1(v1_text)
    lines = [line for line in v1_text.splitlines() if line.strip()]
    restored: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        restored_record: dict[str, Any] | None = None
        if raw_records is not None and index < len(raw_records):
            item = raw_records[index]
            if isinstance(item, dict):
                line_sha = sha256_hex(lines[index].rstrip()) if index < len(lines) else None
                # The sidecar's line hash was computed before trailing whitespace was stripped.
                original_text_line = lines[index]
                sidecar_line_sha = item.get("v1_line_sha256")
                original_record = item.get("original_v2_record")
                if isinstance(original_record, dict) and (
                    sidecar_line_sha == sha256_hex(original_text_line)
                    or sidecar_line_sha == sha256_hex(original_text_line.rstrip())
                    or sidecar_line_sha == line_sha
                ):
                    restored_record = copy.deepcopy(original_record)
        if restored_record is not None:
            restored.append(restored_record)
            continue
        permille = None if record.guide_strength is None else _round_half_away(record.guide_strength * 1000.0)
        if record.noise_scheme is not None and record.precision_tag is None:
            code = NOISE_NAME_TO_CODE.get(record.noise_scheme)
            if code is None:
                raise BusinessError("E_FORMAT_NOISE_SCHEME", "noise scheme has no v1 mapping",
                                    {"noise_scheme": record.noise_scheme})
            record.precision_tag = _precision_tag(permille or 0, code)
        upgraded = record.as_v2_dict()
        upgraded["_source_version"] = 1
        if record.missing_fields:
            upgraded["_missing_fields"] = list(record.missing_fields)
        restored.append(upgraded)
    return {"format_version": 2, "records": restored}


def load_v1(path) -> str:
    from pathlib import Path
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot read v1 file {path}", {"path": str(path)}) from exc


def load_sidecar(path) -> dict[str, Any]:
    from pathlib import Path
    if path is None:
        return None  # type: ignore[return-value]
    raw = read_json_file(Path(path), code="E_FORMAT_SIDECAR")
    if not isinstance(raw, dict):
        raise BusinessError("E_FORMAT_SIDECAR", "sidecar must be a JSON object")
    return raw
