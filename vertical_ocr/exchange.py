from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from .errors import FieldTruncated
from .ledger import JsonlLedger, SidecarStore
from .model import (
    MISSING,
    ORDER_CODE_TO_NAME,
    ORDER_NAME_TO_CODE,
    ExchangeRecord,
    is_missing,
    jsonable,
)


V1_FIELDS: list[tuple[str, int, str]] = [
    ("image_no", 4, "int"),
    ("char_no", 4, "int"),
    ("x1", 5, "int"),
    ("y1", 5, "int"),
    ("x2", 5, "int"),
    ("y2", 5, "int"),
    ("confidence_permille", 5, "int"),
    ("order_code", 4, "int"),
]
V1_WIDTH = sum(width for _, width, _ in V1_FIELDS)
V3_HEADER = "# vertical-ocr/v3"
V3_REQUIRED_KEYS = [
    "image_no",
    "char_no",
    "bbox",
    "confidence",
    "reading_order",
    "char",
    "keypoint_version",
    "keypoint_x",
    "keypoint_y",
    "annotation_version",
]


def _business_key(image_no: Any, char_no: Any) -> str:
    return f"img-{image_no}:char-{char_no}"


def sidecar_id_for(business_key: str) -> str:
    return "SC-" + hashlib.sha256(business_key.encode("utf-8")).hexdigest()[:16]


def _missing_or_value(text: str) -> Any:
    return MISSING if text == "" or text == "__MISSING__" else text


def _parse_int_field(version: str, field_name: str, raw: str) -> int | object:
    stripped = raw.strip()
    if not stripped:
        return MISSING
    try:
        if not all(ch.isdigit() or ch in "+-" for ch in stripped):
            raise ValueError("illegal character")
        return int(stripped)
    except ValueError as exc:
        raise FieldTruncated(version, field_name, f"illegal integer field {field_name}: {raw!r}") from exc


def _format_fixed_int(value: Any, width: int) -> str:
    if is_missing(value):
        return " " * width
    text = str(int(value))
    if len(text) > width:
        raise FieldTruncated("v1", "fixed_width", f"value {text!r} exceeds width {width}")
    if value >= 0:
        return text.zfill(width)
    return "-" + str(abs(int(value))).zfill(width - 1)


def round_half_up_int(value: float) -> int:
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _as_float(value: Any, version: str, field_name: str) -> float:
    if is_missing(value):
        raise FieldTruncated(version, field_name, "required numeric field is explicitly missing")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FieldTruncated(version, field_name, f"illegal float {value!r}") from exc
    if not math.isfinite(result):
        raise FieldTruncated(version, field_name, "float must be finite")
    return result


def parse_v1(text: str) -> list[ExchangeRecord]:
    records: list[ExchangeRecord] = []
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        if len(line) < V1_WIDTH:
            raise FieldTruncated(
                "v1",
                f"line_{lineno}",
                f"fixed-width line has {len(line)} chars, expected at least {V1_WIDTH}",
            )
        pos = 0
        values: dict[str, Any] = {}
        for name, width, kind in V1_FIELDS:
            chunk = line[pos:pos + width]
            pos += width
            values[name] = _parse_int_field("v1", name, chunk)
        bbox = [values[name] for name in ("x1", "y1", "x2", "y2")]
        missing = {name for name, value in values.items() if is_missing(value)}
        if any(is_missing(v) for v in bbox):
            bbox_value = MISSING
        else:
            bbox_value = bbox
        confidence = MISSING
        if not is_missing(values["confidence_permille"]):
            confidence = values["confidence_permille"] / 1000.0
        reading_order = MISSING
        if not is_missing(values["order_code"]):
            reading_order = ORDER_CODE_TO_NAME.get(values["order_code"], values["order_code"])
        rec = ExchangeRecord(
            image_no=values["image_no"],
            char_no=values["char_no"],
            bbox=bbox_value,
            confidence=confidence,
            reading_order=reading_order,
            source_version="v1",
            present_fields={name for name, value in values.items() if not is_missing(value)},
            missing_fields=missing | {"char", "keypoint_x", "keypoint_y", "annotation_version", "keypoint_version"},
            precision_labels=["integer_bbox_exact_float", "permille_confidence_exact_scale_1000"],
            completeness_labels=[
                "char_missing",
                "keypoint_coordinates_missing",
                "annotation_version_missing",
                "keypoint_version_missing",
            ],
            source_values={"v1_fields": values, "line_no": lineno},
        )
        if is_missing(rec.confidence):
            rec.missing_fields.add("confidence")
        if is_missing(rec.bbox):
            rec.missing_fields.add("bbox")
        if is_missing(rec.reading_order):
            rec.missing_fields.add("reading_order")
        rec.char = MISSING
        rec.keypoint_x = MISSING
        rec.keypoint_y = MISSING
        rec.annotation_version = MISSING
        rec.keypoint_version = MISSING
        records.append(rec)
    return records


def _record_from_v2_dict(obj: dict[str, Any], index: int) -> ExchangeRecord:
    required = ["image_no", "char_no", "bbox", "confidence", "reading_order", "keypoint_version", "char"]
    for key in required:
        if key not in obj:
            raise FieldTruncated("v2", key, f"record {index} lacks key {key}")

    def val(key: str) -> Any:
        value = obj[key]
        return MISSING if value == "__MISSING__" else value

    bbox = val("bbox")
    if not is_missing(bbox) and (not isinstance(bbox, list) or len(bbox) != 4):
        raise FieldTruncated("v2", "bbox", "bbox must be a four-element array")
    rec = ExchangeRecord(
        image_no=val("image_no"),
        char_no=val("char_no"),
        bbox=bbox,
        confidence=val("confidence"),
        reading_order=val("reading_order"),
        char=val("char"),
        keypoint_version=val("keypoint_version"),
        keypoint_x=MISSING,
        keypoint_y=MISSING,
        annotation_version=val("annotation_version") if "annotation_version" in obj else MISSING,
        source_version="v2",
        source_values={"v2": obj},
    )
    for key in ("keypoint_x", "keypoint_y", "annotation_version"):
        if is_missing(getattr(rec, key)):
            rec.mark_missing(key)
    rec.present_fields = {k for k in required if not is_missing(val(k))}
    return rec


def parse_v2(text: str | dict[str, Any]) -> list[ExchangeRecord]:
    try:
        data = text if isinstance(text, dict) else json.loads(text)
    except json.JSONDecodeError as exc:
        raise FieldTruncated("v2", "json", f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise FieldTruncated("v2", "records", "v2 must contain a records array")
    return [_record_from_v2_dict(obj, i) for i, obj in enumerate(data["records"])]


def _coerce_kv_value(key: str, raw: str, block_index: int) -> Any:
    value = _missing_or_value(raw.strip())
    if is_missing(value):
        return MISSING
    int_keys = {"image_no", "char_no", "keypoint_x", "keypoint_y"}
    float_json_keys = {"bbox", "confidence"}
    if key in int_keys:
        try:
            return int(value)
        except ValueError as exc:
            raise FieldTruncated("v3", key, f"illegal integer {value!r} in block {block_index}") from exc
    if key in float_json_keys:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FieldTruncated("v3", key, f"illegal JSON value {value!r}") from exc
        if key == "bbox" and (not isinstance(parsed, list) or len(parsed) != 4):
            raise FieldTruncated("v3", "bbox", "bbox must be a four-element JSON array")
        if key == "confidence":
            _as_float(parsed, "v3", "confidence")
        return parsed
    return value


def parse_v3(text: str) -> list[ExchangeRecord]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != V3_HEADER:
        raise FieldTruncated("v3", "header", f"missing delimiter header {V3_HEADER}")
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "---":
            if current:
                blocks.append(current)
                current = {}
            continue
        if "=" not in stripped:
            raise FieldTruncated("v3", "kv", f"expected key=value, got {line!r}")
        key, value = stripped.split("=", 1)
        current[key.strip()] = value.strip()
    if current:
        blocks.append(current)

    records: list[ExchangeRecord] = []
    for index, kv in enumerate(blocks):
        for key in V3_REQUIRED_KEYS:
            if key not in kv:
                raise FieldTruncated("v3", key, f"block {index} lacks key {key}")
        parsed = {key: _coerce_kv_value(key, kv[key], index) for key in V3_REQUIRED_KEYS}
        sidecar_id = kv.get("sidecar_id", MISSING)
        rec = ExchangeRecord(source_version="v3", sidecar_id=sidecar_id, source_values={"v3": kv})
        for key, value in parsed.items():
            setattr(rec, key, value)
            if is_missing(value):
                rec.missing_fields.add(key)
        if "precision_labels" in kv and kv["precision_labels"]:
            rec.precision_labels = json.loads(kv["precision_labels"])
        if "completeness_labels" in kv and kv["completeness_labels"]:
            rec.completeness_labels = json.loads(kv["completeness_labels"])
        if "residuals" in kv and kv["residuals"]:
            rec.residuals = json.loads(kv["residuals"])
        records.append(rec)
    return records


def parse_version(version: str, text: str) -> list[ExchangeRecord]:
    if version == "v1":
        return parse_v1(text)
    if version == "v2":
        return parse_v2(text)
    if version == "v3":
        return parse_v3(text)
    raise ValueError(f"unknown exchange version {version}")


def record_business_key(rec: ExchangeRecord) -> str:
    return _business_key(rec.image_no, rec.char_no)


def downgrade_v3_to_v1(
    records: Iterable[ExchangeRecord],
    sidecars: SidecarStore | None = None,
    ledger: JsonlLedger | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    sidecar_entries: list[dict[str, Any]] = []
    for rec in records:
        key = record_business_key(rec)
        sid = sidecar_id_for(key)
        rec.sidecar_id = sid
        if is_missing(rec.bbox):
            bbox = [MISSING] * 4
            int_bbox = [MISSING] * 4
            bbox_residual = [MISSING] * 4
        else:
            bbox = [float(v) for v in rec.bbox]
            int_bbox = [math.floor(v) for v in bbox]
            bbox_residual = [original - integer for original, integer in zip(bbox, int_bbox)]
        if is_missing(rec.confidence):
            confidence = MISSING
            permille = MISSING
            confidence_residual = MISSING
        else:
            confidence = _as_float(rec.confidence, "v3", "confidence")
            permille = round_half_up_int(confidence * 1000)
            confidence_residual = confidence - permille / 1000.0
        order_name = rec.reading_order
        order_code = MISSING if is_missing(order_name) else ORDER_NAME_TO_CODE.get(order_name, 10)
        payload = {
            "business_key": key,
            "source_version": rec.source_version,
            "original_bbox": None if any(is_missing(v) for v in bbox) else bbox,
            "integer_bbox": None if any(is_missing(v) for v in int_bbox) else int_bbox,
            "bbox_truncation_residual": None if any(is_missing(v) for v in bbox_residual) else bbox_residual,
            "original_confidence": None if is_missing(confidence) else confidence,
            "permille_confidence": None if is_missing(permille) else permille,
            "confidence_round_residual": None if is_missing(confidence_residual) else confidence_residual,
            "reading_order_name": None if is_missing(order_name) else order_name,
            "order_code": None if is_missing(order_code) else order_code,
            "keypoint_x": None if is_missing(rec.keypoint_x) else rec.keypoint_x,
            "keypoint_y": None if is_missing(rec.keypoint_y) else rec.keypoint_y,
            "annotation_version": None if is_missing(rec.annotation_version) else rec.annotation_version,
            "keypoint_version": None if is_missing(rec.keypoint_version) else rec.keypoint_version,
            "char": None if is_missing(rec.char) else rec.char,
            "precision_labels": rec.precision_labels,
            "completeness_labels": rec.completeness_labels,
            "residuals": rec.residuals,
        }
        if sidecars is not None:
            stored = sidecars.put(sid, key, payload)
            sidecar_entries.append(stored)
        if ledger is not None:
            ledger.append(
                "downgrade_v3_to_v1",
                {
                    "sidecar_id": sid,
                    "business_key": key,
                    "permille_confidence": jsonable(permille),
                    "bbox_truncation_residual": jsonable(bbox_residual),
                    "confidence_round_residual": jsonable(confidence_residual),
                },
            )
        values = [
            rec.image_no if not is_missing(rec.image_no) else MISSING,
            rec.char_no if not is_missing(rec.char_no) else MISSING,
            *int_bbox,
            permille,
            order_code,
        ]
        line = "".join(_format_fixed_int(value, width) for value, (_, width, _) in zip(values, V1_FIELDS))
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else ""), sidecar_entries


def upgrade_v1_to_v3(
    records: Iterable[ExchangeRecord],
    sidecars: SidecarStore | None = None,
    ledger: JsonlLedger | None = None,
) -> list[ExchangeRecord]:
    upgraded: list[ExchangeRecord] = []
    for rec in records:
        key = record_business_key(rec)
        sid = sidecar_id_for(key)
        restored = sidecars.get(sid) if sidecars is not None else None
        if restored is None and sidecars is not None:
            restored = sidecars.find_latest_for_business_key(key)
        if restored:
            if restored.get("original_bbox") is not None:
                rec.bbox = [float(v) for v in restored["original_bbox"]]
            if restored.get("original_confidence") is None:
                rec.confidence = MISSING
            else:
                rec.confidence = float(restored["original_confidence"])
            rec.reading_order = restored.get("reading_order_name") or MISSING
            rec.char = restored.get("char", MISSING) or MISSING
            rec.keypoint_version = restored.get("keypoint_version") or MISSING
            rec.keypoint_x = restored.get("keypoint_x") if restored.get("keypoint_x") is not None else MISSING
            rec.keypoint_y = restored.get("keypoint_y") if restored.get("keypoint_y") is not None else MISSING
            rec.annotation_version = restored.get("annotation_version") or MISSING
            rec.sidecar_id = restored["sidecar_id"]
            rec.residuals = {
                "bbox_truncation_residual": restored.get("bbox_truncation_residual"),
                "confidence_round_residual": restored.get("confidence_round_residual"),
            }
            for label in ("sidecar_restored", "precision_restored", "completeness_restored"):
                if label not in rec.precision_labels:
                    rec.precision_labels.append(label)
            if ledger is not None:
                ledger.append("sidecar_restore", {"sidecar_id": restored["sidecar_id"], "business_key": key})
        else:
            if not is_missing(rec.bbox):
                rec.bbox = [float(v) for v in rec.bbox]
            if not is_missing(rec.reading_order) and isinstance(rec.reading_order, int):
                rec.reading_order = ORDER_CODE_TO_NAME.get(rec.reading_order, f"unknown_code_{rec.reading_order}")
            rec.sidecar_id = MISSING
            for field_name in ("keypoint_x", "keypoint_y", "annotation_version", "keypoint_version"):
                if not is_missing(getattr(rec, field_name)):
                    continue
                if f"{field_name}_missing" not in rec.completeness_labels:
                    rec.mark_missing(field_name)
        rec.source_version = "v1+sidecar" if restored else "v1"
        upgraded.append(rec)
    return upgraded


def to_v2_dict(records: Iterable[ExchangeRecord]) -> dict[str, Any]:
    out = []
    for rec in records:
        out.append({
            "image_no": jsonable_scalar(rec.image_no),
            "char_no": jsonable_scalar(rec.char_no),
            "bbox": "__MISSING__" if is_missing(rec.bbox) else rec.bbox,
            "confidence": "__MISSING__" if is_missing(rec.confidence) else float(rec.confidence),
            "reading_order": "__MISSING__" if is_missing(rec.reading_order) else rec.reading_order,
            "char": "__MISSING__" if is_missing(rec.char) else rec.char,
            "keypoint_version": "__MISSING__" if is_missing(rec.keypoint_version) else rec.keypoint_version,
            "annotation_version": "__MISSING__" if is_missing(rec.annotation_version) else rec.annotation_version,
            "missing_fields": sorted(rec.missing_fields),
            "precision_labels": rec.precision_labels,
            "completeness_labels": rec.completeness_labels,
        })
    return {"schema": "vertical-ocr/v2", "records": out}


def jsonable_scalar(value: Any) -> Any:
    return "__MISSING__" if is_missing(value) else value


def to_v2(records: Iterable[ExchangeRecord]) -> str:
    return json.dumps(to_v2_dict(records), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def to_v3(records: Iterable[ExchangeRecord]) -> str:
    lines = [V3_HEADER]
    for rec in records:
        values = {
            "image_no": rec.image_no,
            "char_no": rec.char_no,
            "bbox": json.dumps(rec.bbox, ensure_ascii=False) if not is_missing(rec.bbox) else "__MISSING__",
            "confidence": "__MISSING__" if is_missing(rec.confidence) else repr(float(rec.confidence)),
            "reading_order": rec.reading_order,
            "char": rec.char,
            "keypoint_version": rec.keypoint_version,
            "keypoint_x": rec.keypoint_x,
            "keypoint_y": rec.keypoint_y,
            "annotation_version": rec.annotation_version,
            "sidecar_id": rec.sidecar_id,
            "precision_labels": json.dumps(rec.precision_labels, ensure_ascii=False),
            "completeness_labels": json.dumps(rec.completeness_labels, ensure_ascii=False),
            "residuals": json.dumps(rec.residuals, ensure_ascii=False),
        }
        for key, value in values.items():
            if is_missing(value):
                value = "__MISSING__"
            lines.append(f"{key}={value}")
        lines.append("---")
    return "\n".join(lines).rstrip("-") + "\n"


def convert_text(
    source_version: str,
    target_version: str,
    text: str,
    *,
    sidecars: SidecarStore | None = None,
    ledger: JsonlLedger | None = None,
) -> str:
    records = parse_version(source_version, text)
    if source_version == "v1" and target_version == "v3":
        records = upgrade_v1_to_v3(records, sidecars, ledger)
        return to_v3(records)
    if source_version == "v3" and target_version == "v1":
        output, _ = downgrade_v3_to_v1(records, sidecars, ledger)
        return output
    if target_version == source_version:
        return text
    if target_version == "v2":
        return to_v2(records)
    if target_version == "v3":
        return to_v3(records)
    if target_version == "v1":
        output, _ = downgrade_v3_to_v1(records, sidecars, ledger)
        return output
    raise ValueError(f"unsupported conversion {source_version}->{target_version}")
