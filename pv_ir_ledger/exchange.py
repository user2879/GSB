from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from . import EXCHANGE_V1_SCHEMA, EXCHANGE_V2_SCHEMA
from .errors import ErrorCode, make_issue
from .recognition import GRADE_CODE_TO_NAME, GRADE_NAME_TO_CODE


V1_FIELDS: list[tuple[str, int]] = [
    ("image", 32),
    ("component_id", 6),
    ("x1", 6),
    ("y1", 6),
    ("x2", 6),
    ("y2", 6),
    ("intensity_permille", 6),
    ("grade_code", 4),
]
V1_TOTAL_WIDTH = sum(width for _, width in V1_FIELDS)
V1_NUMERIC_FIELDS = {"component_id", "x1", "y1", "x2", "y2", "intensity_permille", "grade_code"}


def logical_key(image: str, component_id: int | str | None) -> str:
    return f"{image}#{component_id}"


def decimal_value(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "-":
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None
    return None


def canonical_decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return "0" if text == "-0" else text


def _empty_judgment(source_version: str) -> dict[str, Any]:
    return {
        "schema": EXCHANGE_V1_SCHEMA if source_version == "v1" else EXCHANGE_V2_SCHEMA,
        "source_version": source_version,
        "image": None,
        "component_id": None,
        "bbox": [None, None, None, None],
        "bbox_decimal": [None, None, None, None],
        "intensity_permille": None,
        "intensity_decimal": None,
        "grade_code": None,
        "grade_name": None,
        "quantization_scheme": None,
        "missing_fields": [],
        "raw_fields": {},
        "precision_labels": [],
        "sidecar_key": None,
        "issues": [],
    }


def parse_v1_line(line: str, line_number: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    judgment = _empty_judgment("v1")
    issues: list[dict[str, Any]] = []
    if line_number is not None:
        judgment["line_number"] = line_number
    if not line.strip():
        issue = make_issue(
            ErrorCode.FIELD_TRUNCATED,
            "v1 record is empty",
            line_number=line_number,
            reason="EMPTY_RECORD",
        )
        issues.append(issue)
        judgment["issues"].append(issue)
        return judgment, issues

    if len(line) < V1_TOTAL_WIDTH:
        issue = make_issue(
            ErrorCode.FIELD_TRUNCATED,
            "v1 fixed-width record is shorter than the declared field map",
            line_number=line_number,
            expected_width=V1_TOTAL_WIDTH,
            actual_width=len(line),
            reason="LINE_WIDTH_INSUFFICIENT",
        )
        issues.append(issue)
        judgment["issues"].append(issue)

    offset = 0
    values: dict[str, str | None] = {}
    for field, width in V1_FIELDS:
        end = offset + width
        if offset >= len(line):
            raw = ""
        else:
            raw = line[offset : min(end, len(line))]
        judgment["raw_fields"][field] = raw
        if len(raw) < width:
            judgment["missing_fields"].append(field)
        stripped = raw.strip()
        if not stripped or stripped == "-":
            values[field] = None
            if field not in judgment["missing_fields"]:
                judgment["missing_fields"].append(field)
        else:
            values[field] = stripped
        offset = end

    judgment["image"] = values["image"]
    if judgment["image"] is not None:
        judgment["sidecar_key"] = logical_key(judgment["image"], values.get("component_id") or "?")

    for field in V1_NUMERIC_FIELDS:
        text = values.get(field)
        if text is None:
            continue
        try:
            if not all(ch.isdigit() or ch in "+-" for ch in text) or text in {"+", "-"}:
                raise ValueError("illegal numeric character")
            number = int(text)
        except ValueError:
            issue = make_issue(
                ErrorCode.FIELD_TRUNCATED,
                "v1 numeric field contains illegal characters",
                line_number=line_number,
                field=field,
                raw_value=text,
                reason="ILLEGAL_CHARACTER",
            )
            issues.append(issue)
            judgment["issues"].append(issue)
            judgment["missing_fields"].append(field)
            continue
        if field == "component_id":
            judgment["component_id"] = number
        elif field == "intensity_permille":
            judgment["intensity_permille"] = number
            judgment["intensity_decimal"] = str(Decimal(number) / Decimal(1000))
        elif field == "grade_code":
            judgment["grade_code"] = number
            judgment["grade_name"] = GRADE_CODE_TO_NAME.get(number)
        else:
            index = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}[field]
            judgment["bbox"][index] = number
            judgment["bbox_decimal"][index] = str(number)

    if judgment["image"] and judgment["component_id"] is not None:
        judgment["sidecar_key"] = logical_key(judgment["image"], judgment["component_id"])
    return judgment, issues


def parse_v1_text(text: str, sidecar: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judgments: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    entries = {entry["key"]: entry for entry in (sidecar or {}).get("entries", []) if "key" in entry}
    for line_number, line in enumerate(text.splitlines(), start=1):
        judgment, line_issues = parse_v1_line(line, line_number)
        if not line and not line_issues:
            continue
        if judgment.get("sidecar_key") in entries:
            judgment["sidecar_entry"] = entries[judgment["sidecar_key"]]
        judgments.append(judgment)
        issues.extend(line_issues)
    return judgments, issues


def parse_v2_document(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return [], [make_issue(ErrorCode.FIELD_TRUNCATED, "v2 result document must be an object", reason="NOT_OBJECT")]
    records = raw.get("judgments")
    if not isinstance(records, list):
        return [], [make_issue(ErrorCode.FIELD_TRUNCATED, "v2 document must contain judgments list", reason="MISSING_JUDGMENTS")]

    judgments: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        judgment = _empty_judgment("v2")
        judgment["record_index"] = index
        if not isinstance(record, dict):
            issue = make_issue(ErrorCode.FIELD_TRUNCATED, "v2 judgment must be an object", index=index)
            issues.append(issue)
            judgment["issues"].append(issue)
            judgments.append(judgment)
            continue
        judgment["raw_fields"] = dict(record)
        judgment["image"] = record.get("image") if isinstance(record.get("image"), str) else None
        if record.get("image") is None:
            judgment["missing_fields"].append("image")

        component = record.get("component_id")
        if isinstance(component, int) and not isinstance(component, bool):
            judgment["component_id"] = component
        else:
            judgment["missing_fields"].append("component_id")

        bbox = record.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            judgment["missing_fields"].append("bbox")
            issue = make_issue(ErrorCode.FIELD_TRUNCATED, "v2 bbox must contain four floats", index=index)
            issues.append(issue)
            judgment["issues"].append(issue)
        else:
            for coordinate, value in enumerate(bbox):
                decoded = decimal_value(value)
                if decoded is None:
                    judgment["missing_fields"].append(f"bbox[{coordinate}]")
                    judgment["bbox_decimal"][coordinate] = None
                else:
                    judgment["bbox_decimal"][coordinate] = canonical_decimal_text(decoded)
                    judgment["bbox"][coordinate] = float(decoded)

        intensity = decimal_value(record.get("intensity"))
        if intensity is None:
            judgment["missing_fields"].append("intensity")
        else:
            judgment["intensity_decimal"] = canonical_decimal_text(intensity)
            permille = (intensity * Decimal(1000)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
            judgment["intensity_permille"] = int(permille)

        grade_name = record.get("grade")
        if isinstance(grade_name, str) and grade_name in GRADE_NAME_TO_CODE:
            judgment["grade_name"] = grade_name
            judgment["grade_code"] = GRADE_NAME_TO_CODE[grade_name]
        else:
            judgment["missing_fields"].append("grade")
            issue = make_issue(
                ErrorCode.FIELD_TRUNCATED,
                "v2 named grade is not in the code table",
                index=index,
                grade=grade_name,
            )
            issues.append(issue)
            judgment["issues"].append(issue)

        scheme = record.get("quantization_scheme", raw.get("quantization_scheme"))
        if isinstance(scheme, str):
            judgment["quantization_scheme"] = scheme
        else:
            judgment["missing_fields"].append("quantization_scheme")
        labels = record.get("precision_labels")
        if isinstance(labels, list):
            judgment["precision_labels"] = [str(label) for label in labels]

        if judgment["image"] and judgment["component_id"] is not None:
            judgment["sidecar_key"] = logical_key(judgment["image"], judgment["component_id"])
        judgments.append(judgment)
    return judgments, issues


def load_v2_file(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_float=str)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [make_issue(ErrorCode.FIELD_TRUNCATED, "v2 result file could not be parsed", path=str(path), detail=str(exc))]
    return parse_v2_document(raw)


def format_v1_field(value: Any, width: int, align: str = "right") -> str:
    if value is None:
        text = "-"
    else:
        text = str(value)
    if len(text) > width:
        if width >= 3:
            text = text[: width - 3] + "..."
        else:
            text = text[:width]
    return text.rjust(width) if align == "right" else text.ljust(width)


def judgment_to_v1_line(judgment: dict[str, Any]) -> str:
    values = {
        "image": judgment.get("image"),
        "component_id": judgment.get("component_id"),
        "x1": judgment["bbox"][0] if len(judgment.get("bbox", [])) == 4 else None,
        "y1": judgment["bbox"][1] if len(judgment.get("bbox", [])) == 4 else None,
        "x2": judgment["bbox"][2] if len(judgment.get("bbox", [])) == 4 else None,
        "y2": judgment["bbox"][3] if len(judgment.get("bbox", [])) == 4 else None,
        "intensity_permille": judgment.get("intensity_permille"),
        "grade_code": judgment.get("grade_code"),
    }
    parts: list[str] = []
    for field, width in V1_FIELDS:
        parts.append(format_v1_field(values[field], width, "left" if field == "image" else "right"))
    return "".join(parts)


def write_v1_text(judgments: list[dict[str, Any]]) -> str:
    return "\n".join(judgment_to_v1_line(judgment) for judgment in judgments) + "\n"


def judgment_to_v2_record(judgment: dict[str, Any]) -> dict[str, Any]:
    bbox_decimal = judgment.get("bbox_decimal") or [None, None, None, None]
    bbox = []
    for index, value in enumerate(bbox_decimal):
        if value is not None:
            bbox.append(float(Decimal(value)))
        else:
            bbox.append(judgment.get("bbox", [None] * 4)[index])
    intensity_text = judgment.get("intensity_decimal")
    intensity = float(Decimal(intensity_text)) if intensity_text is not None else (
        judgment.get("intensity_permille") / 1000 if judgment.get("intensity_permille") is not None else None
    )
    return {
        "image": judgment.get("image"),
        "component_id": judgment.get("component_id"),
        "bbox": bbox,
        "intensity": intensity,
        "grade": judgment.get("grade_name"),
        "quantization_scheme": judgment.get("quantization_scheme"),
        "precision_labels": judgment.get("precision_labels", []),
    }


def write_v2_document(judgments: list[dict[str, Any]], scheme: str | None = None) -> dict[str, Any]:
    return {
        "schema": EXCHANGE_V2_SCHEMA,
        "quantization_scheme": scheme or "MIXED",
        "judgments": [judgment_to_v2_record(item) for item in judgments],
    }


def downgrade_v2_to_v1(judgment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if judgment.get("source_version") != "v2":
        raise ValueError("downgrade expects a v2 unified judgment")
    v1 = _empty_judgment("v1")
    for key in ("image", "component_id", "grade_code", "grade_name", "missing_fields"):
        v1[key] = judgment.get(key, [] if key == "missing_fields" else None)
    v1["sidecar_key"] = judgment.get("sidecar_key") or logical_key(judgment.get("image"), judgment.get("component_id"))

    original_bbox = [decimal_value(v) for v in judgment.get("bbox_decimal", [])]
    rounded_bbox: list[int | None] = []
    bbox_residuals: list[str | None] = []
    for value in original_bbox:
        if value is None:
            rounded_bbox.append(None)
            bbox_residuals.append(None)
        else:
            rounded = int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            rounded_bbox.append(rounded)
            bbox_residuals.append(format(value - Decimal(rounded), "f"))
    v1["bbox"] = rounded_bbox
    v1["bbox_decimal"] = [str(v) if v is not None else None for v in rounded_bbox]

    intensity = decimal_value(judgment.get("intensity_decimal"))
    if intensity is None and judgment.get("intensity_permille") is not None:
        intensity = Decimal(judgment["intensity_permille"]) / Decimal(1000)
    if intensity is None:
        v1["intensity_permille"] = None
        intensity_residual = None
    else:
        permille = int((intensity * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        v1["intensity_permille"] = permille
        v1["intensity_decimal"] = str(Decimal(permille) / Decimal(1000))
        intensity_residual = format(intensity - Decimal(permille) / Decimal(1000), "f")

    v1["quantization_scheme"] = "PERMILLE_Q3_V1"
    v1["precision_labels"] = [
        "V2_FLOAT_BBOX_ROUNDED_HALF_UP",
        "V2_FLOAT_INTENSITY_ROUNDED_TO_PERMILLE",
    ]
    v1["raw_fields"] = {field: "" for field, _ in V1_FIELDS}

    sidecar_entry = {
        "key": v1["sidecar_key"],
        "image": v1["image"],
        "component_id": v1["component_id"],
        "rounding": {
            "bbox_original_decimal": [format(v, "f") if v is not None else None for v in original_bbox],
            "bbox_rounded_integer": rounded_bbox,
            "bbox_residual_decimal": bbox_residuals,
            "intensity_original_decimal": format(intensity, "f") if intensity is not None else None,
            "intensity_rounded_permille": v1["intensity_permille"],
            "intensity_residual_decimal": intensity_residual,
        },
        "precision_labels": list(judgment.get("precision_labels", [])) + v1["precision_labels"],
        "discarded_quantization_scheme": judgment.get("quantization_scheme"),
        "original_v2": judgment_to_v2_record(judgment),
        "original_raw_fields": deepcopy(judgment.get("raw_fields", {})),
        "record_index": judgment.get("record_index"),
    }
    sidecar_entry["original_v2"]["quantization_scheme"] = judgment.get("quantization_scheme")
    return v1, sidecar_entry


def upgrade_v1_to_v2(
    judgment: dict[str, Any], sidecar: dict[str, Any] | None = None
) -> dict[str, Any]:
    if judgment.get("source_version") != "v1":
        raise ValueError("upgrade expects a v1 unified judgment")
    key = judgment.get("sidecar_key") or logical_key(judgment.get("image"), judgment.get("component_id"))
    entries = {entry["key"]: entry for entry in (sidecar or {}).get("entries", [])}
    attached = judgment.get("sidecar_entry") or entries.get(key)
    result = _empty_judgment("v2")
    for field in ("image", "component_id", "grade_code", "grade_name", "missing_fields", "raw_fields"):
        result[field] = judgment.get(field, [] if field in {"missing_fields"} else None)
    result["sidecar_key"] = key

    if attached and attached.get("original_v2"):
        original = attached["original_v2"]
        restored, issues = parse_v2_document({"judgments": [original]})
        if restored:
            restored_judgment = restored[0]
            restored_judgment["source_version"] = "v2"
            restored_judgment["sidecar_key"] = key
            if attached.get("original_raw_fields") is not None:
                restored_judgment["raw_fields"] = deepcopy(attached["original_raw_fields"])
            if "record_index" in attached:
                restored_judgment["record_index"] = attached["record_index"]
            restored_judgment["sidecar_entry"] = attached
            return restored_judgment
        result["issues"].extend(issues)

    result["bbox"] = [float(v) if v is not None else None for v in judgment.get("bbox", [])]
    result["bbox_decimal"] = [str(v) if v is not None else None for v in judgment.get("bbox_decimal", [])]
    result["intensity_permille"] = judgment.get("intensity_permille")
    if judgment.get("intensity_permille") is not None:
        result["intensity_decimal"] = format(Decimal(judgment["intensity_permille"]) / Decimal(1000), "f")
    result["grade_name"] = judgment.get("grade_name") or GRADE_CODE_TO_NAME.get(judgment.get("grade_code"))
    result["quantization_scheme"] = "PERMILLE_Q3_V1"
    result["precision_labels"] = ["EXACT_INTEGER_TO_FLOAT"]
    return result


def build_sidecar_document(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "pv-ir-exchange-sidecar-v1", "entries": sorted(entries, key=lambda e: e["key"])}


def logical_comparison_view(judgment: dict[str, Any]) -> dict[str, Any]:
    bbox = []
    for value in judgment.get("bbox_decimal", []):
        decoded = decimal_value(value)
        if decoded is None:
            bbox.append(None)
        else:
            bbox.append(canonical_decimal_text(decoded))
    intensity = decimal_value(judgment.get("intensity_decimal"))
    intensity_text = None
    if intensity is not None:
        intensity_text = canonical_decimal_text(intensity)
    return {
        "image": judgment.get("image"),
        "component_id": judgment.get("component_id"),
        "bbox_decimal": bbox,
        "intensity_decimal": intensity_text,
        "intensity_permille": judgment.get("intensity_permille"),
        "grade_code": judgment.get("grade_code"),
        "grade_name": judgment.get("grade_name"),
        "quantization_scheme": judgment.get("quantization_scheme"),
        "missing_fields": sorted(judgment.get("missing_fields", [])),
    }


def strict_internal_view(judgment: dict[str, Any]) -> dict[str, Any]:
    view = deepcopy(judgment)
    view.pop("sidecar_entry", None)
    view.pop("restored_from_sidecar", None)
    return view
