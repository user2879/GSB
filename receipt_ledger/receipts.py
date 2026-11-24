from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import failure as schema_failure
from .constants import MISSING, Code


WIDTHS: dict[str, list[tuple[str, int, str]]] = {
    "r1": [
        ("batch_id", 12, "str"),
        ("request_id", 16, "str"),
        ("output_digest", 20, "str"),
    ],
    "r2": [
        ("batch_id", 12, "str"),
        ("request_id", 16, "str"),
        ("output_digest", 20, "str"),
        ("param_version", 12, "int"),
        ("latency_ms", 12, "int"),
        ("worker_id", 12, "str"),
    ],
    "r3": [
        ("batch_id", 12, "str"),
        ("request_id", 16, "str"),
        ("output_digest", 20, "str"),
        ("param_version", 12, "int"),
        ("latency_ms", 12, "int"),
        ("worker_id", 12, "str"),
        ("rank", 12, "int"),
        ("overrides", 24, "ref"),
        ("param_digest", 20, "str"),
    ],
}

MARKERS = {"R1": "r1", "R2": "r2", "R3": "r3"}
VERSION_MARKER = {"r1": "R1", "r2": "R2", "r3": "R3"}


@dataclass(frozen=True)
class ParsedLine:
    receipt: dict[str, Any] | None
    failure: dict[str, Any] | None


def _pad(value: Any, width: int) -> str:
    text = MISSING if value is None else str(value)
    if len(text) > width:
        raise ValueError(f"field value {text!r} exceeds fixed width {width}")
    return text.ljust(width)


def encode_line(receipt: dict[str, Any], version: str | None = None) -> str:
    version = version or receipt.get("record_version") or "r3"
    fields = WIDTHS[version]
    values: list[str] = [VERSION_MARKER[version]]
    for name, width, _kind in fields:
        value = receipt.get(name)
        if name == "overrides":
            state = receipt.get("overrides_state")
            if state == "explicit_none" and not value:
                value = "-"
            elif state == "missing" and not value:
                value = MISSING
        values.append(_pad(value, width))
    return "|".join(values)


def _parse_int(raw: str, *, nonfinite_code: str | None = None) -> tuple[int | None, str | None, str | None]:
    text = raw.strip()
    if text == MISSING:
        return None, "missing", None
    try:
        if nonfinite_code and text.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
            return None, "nonfinite", nonfinite_code
        return int(text), "present", None
    except ValueError:
        try:
            number = float(text)
            if not math.isfinite(number) and nonfinite_code:
                return None, "nonfinite", nonfinite_code
        except ValueError:
            pass
        return None, "invalid", Code.RECEIPT_FIELD_INVALID


def parse_line(line: str, *, source: str, line_no: int) -> ParsedLine:
    raw_line = line.rstrip("\r\n")
    if not raw_line.strip():
        return ParsedLine(None, None)
    marker = raw_line[:2]
    version = MARKERS.get(marker)
    if version is None:
        return ParsedLine(None, schema_failure(
            source, "unknown receipt marker or truncated record", index=line_no,
            code=Code.RECEIPT_TRUNCATED,
        ))
    specs = WIDTHS[version]
    parts = raw_line.split("|")
    expected_count = len(specs) + 1
    if len(parts) < expected_count:
        return ParsedLine(None, schema_failure(
            source, f"truncated {version} record: expected {expected_count} columns, got {len(parts)}",
            index=line_no, code=Code.RECEIPT_TRUNCATED,
        ))
    if len(parts) > expected_count:
        return ParsedLine(None, schema_failure(
            source, f"malformed {version} record: too many columns", index=line_no,
            code=Code.RECEIPT_FIELD_INVALID,
        ))
    if parts[0] != marker:
        return ParsedLine(None, schema_failure(source, "bad receipt marker", index=line_no,
                                              code=Code.RECEIPT_FIELD_INVALID))

    receipt: dict[str, Any] = {
        "record_version": version,
        "source_file": source,
        "line_no": line_no,
        "receipt_uid": f"{version}:{source}:{line_no}",
        "missing_fields": [],
        "precision_labels": [],
        "completeness_labels": [],
    }
    for index, ((name, width, kind), column) in enumerate(zip(specs, parts[1:]), start=1):
        if len(column) != width:
            return ParsedLine(None, schema_failure(
                source, f"fixed-width field {name} has width {len(column)}, expected {width}",
                index=line_no, item_id=name, field=name, code=Code.RECEIPT_TRUNCATED,
            ))
        value_text = column.strip()
        if kind == "int":
            nonfinite_code = Code.RECEIPT_NONFINITE_LATENCY if name == "latency_ms" else None
            value, state, error_code = _parse_int(value_text, nonfinite_code=nonfinite_code)
            if error_code:
                message = "non-finite latency_ms" if error_code == Code.RECEIPT_NONFINITE_LATENCY else f"invalid integer field {name}"
                return ParsedLine(None, schema_failure(source, message, index=line_no, field=name,
                                                      code=error_code))
            receipt[name] = value
            receipt[f"{name}_state"] = state
            if state == "missing":
                receipt["missing_fields"].append(name)
        else:
            if value_text == MISSING or value_text == "":
                receipt[name] = None
                receipt[f"{name}_state"] = "missing"
                receipt["missing_fields"].append(name)
            elif name == "overrides":
                if value_text == "-":
                    receipt[name] = None
                    receipt["overrides_state"] = "explicit_none"
                else:
                    receipt[name] = value_text
                    receipt["overrides_state"] = "reference"
            else:
                receipt[name] = value_text
                receipt[f"{name}_state"] = "present"

    if version in {"r1", "r2"}:
        receipt["rank"] = None
        receipt["rank_source"] = "lexicographic_request_id"
        receipt["rank_basis"] = "missing stable rank; ordered by request_id lexicographic ascending"
        receipt["precision_labels"].append("rank:lexicographic_request_id")
    else:
        receipt["rank_source"] = "r3.stable_rank"
        receipt["rank_basis"] = "r3 stable rank"

    if version == "r1":
        for name in ("param_version", "latency_ms", "worker_id", "param_digest"):
            if name not in receipt:
                receipt[name] = None
                receipt[f"{name}_state"] = "missing"
                receipt["missing_fields"].append(name)
        receipt.setdefault("overrides", None)
        receipt.setdefault("overrides_state", "missing")
        receipt["completeness_labels"].append("r1_to_r3:minimal_identity_only")
    elif version == "r2":
        for name in ("param_digest",):
            receipt[name] = None
            receipt[f"{name}_state"] = "missing"
            receipt["missing_fields"].append(name)
        receipt.setdefault("overrides", None)
        receipt.setdefault("overrides_state", "missing")
        receipt["completeness_labels"].append("r2_to_r3:resource_fields_present")
    else:
        receipt["completeness_labels"].append("r3:complete_wire_shape")

    return ParsedLine(receipt, None)


def parse_receipt_text(text: str, *, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parsed = parse_line(line, source=source, line_no=line_no)
        if parsed.failure:
            failures.append(parsed.failure)
        if parsed.receipt:
            receipts.append(parsed.receipt)
    return receipts, failures


def rank_batch_receipts(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return receipts ordered by r3 rank, with lexicographic fallback."""
    def key(receipt: dict[str, Any]) -> tuple[int, int, str, str]:
        rank = receipt.get("rank")
        rank_key = rank if isinstance(rank, int) else 10**9
        version_order = {"r3": 0, "r2": 1, "r1": 2}.get(receipt.get("record_version"), 3)
        return (rank_key, 0 if isinstance(rank, int) else 1, str(receipt.get("request_id", "")), version_order)

    return sorted(receipts, key=key)


def upgrade_to_r3(receipt: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(receipt)
    upgraded["target_version"] = "r3"
    added_missing = sorted(set([
        "latency_ms", "param_version", "worker_id", "rank", "overrides", "param_digest", "description",
    ]) - set(k for k, v in receipt.items() if v is not None and not k.endswith("_state")))
    upgraded["upgrade_missing_fields"] = added_missing
    upgraded.setdefault("description", MISSING)
    upgraded["description_state"] = "missing"
    labels = list(upgraded.get("precision_labels", []))
    labels.append("numeric_and_description_fields:explicit_missing")
    upgraded["precision_labels"] = sorted(set(labels))
    completeness = list(upgraded.get("completeness_labels", []))
    completeness.append("upgraded_to_r3_with_explicit_missing_markers")
    upgraded["completeness_labels"] = sorted(set(completeness))
    upgraded["r3_line"] = encode_line(upgraded, "r3") if _encodable(upgraded) else None
    return upgraded


def _encodable(receipt: dict[str, Any]) -> bool:
    try:
        encode_line(receipt, "r3")
        return True
    except ValueError:
        return False


def build_sidecar(receipt: dict[str, Any]) -> dict[str, Any]:
    truncated = truncated_fields_for_r1(receipt)
    return {
        "sidecar_version": "receipt-sidecar.v1",
        "receipt_uid": receipt["receipt_uid"],
        "source_record_version": receipt["record_version"],
        "stable_rank": receipt.get("rank"),
        "rank_source": receipt.get("rank_source"),
        "overridden_reference": receipt.get("overrides"),
        "overrides_state": receipt.get("overrides_state"),
        "param_digest": receipt.get("param_digest"),
        "param_version": receipt.get("param_version"),
        "latency_ms": receipt.get("latency_ms"),
        "worker_id": receipt.get("worker_id"),
        "truncated_fields": truncated,
    }


def truncated_fields_for_r1(receipt: dict[str, Any]) -> list[str]:
    extra = ["param_version", "latency_ms", "worker_id", "rank", "overrides", "param_digest"]
    return [name for name in extra if name in receipt]


def downgrade_to_r1(receipt: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    sidecar = build_sidecar(receipt)
    truncated = sorted(set(truncated_fields_for_r1(receipt) + sidecar["truncated_fields"]))
    r1_receipt = {
        "batch_id": receipt["batch_id"],
        "request_id": receipt["request_id"],
        "output_digest": receipt["output_digest"],
    }
    line = encode_line(r1_receipt, "r1")
    return line, sidecar, truncated
