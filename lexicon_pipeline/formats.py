"""jv1 fixed-width text and jv2 JSON receipt conversion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_FIELD_TRUNCATED, E_FORMAT_INVALID, E_SIDECAR_MISMATCH, Issue
from .fixedpoint import ROUNDING_CODES, quantize_int, quantization_residual
from .hashing import fingerprint


DEFAULT_WIDTHS: dict[str, int] = {
    "doc_seq": 4,
    "fragment_seq": 4,
    "score_permille": 5,
    "pos_code": 4,
}
FIELD_ORDER = ("doc_seq", "fragment_seq", "score_permille", "pos_code")
_DIGITS = re.compile(r"[0-9]+")


@dataclass
class FormatResult:
    records: list[dict[str, Any]]
    issues: list[Issue]
    text: str | None = None
    sidecar: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.issues


def receipt_key(doc_seq: int, fragment_seq: int) -> str:
    return f"{int(doc_seq):04d}:{int(fragment_seq):04d}"


def encode_jv1_record(record: dict[str, Any], widths: dict[str, int] | None = None, line_number: int | None = None) -> str:
    widths = widths or DEFAULT_WIDTHS
    parts: list[str] = []
    for field in FIELD_ORDER:
        width = widths[field]
        if field not in record:
            raise _field_issue(field, f"missing integer field {field}", line_number)
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise _field_issue(field, f"{field} must be an integer", line_number)
        text = str(value)
        if value < 0 or len(text) > width:
            issue = Issue(
                E_FIELD_TRUNCATED,
                f"value {value!r} does not fit fixed width {width} for {field}",
                line=line_number,
                field=field,
            )
            raise FormatIssueError(issue)
        parts.append(text.zfill(width))
    return "".join(parts)


class FormatIssueError(Exception):
    def __init__(self, issue: Issue):
        self.issue = issue
        super().__init__(issue.message)


def _field_issue(field: str, message: str, line: int | None) -> FormatIssueError:
    return FormatIssueError(Issue(E_FIELD_TRUNCATED, message, line=line, field=field))


def encode_jv1(records: list[dict[str, Any]], widths: dict[str, int] | None = None) -> FormatResult:
    lines: list[str] = []
    issues: list[Issue] = []
    for index, record in enumerate(records, start=1):
        try:
            lines.append(encode_jv1_record(record, widths, index))
        except FormatIssueError as exc:
            issues.append(exc.issue)
    return FormatResult(records=[], issues=issues, text="\n".join(lines) + ("\n" if lines else ""))


def parse_jv1(text: str, widths: dict[str, int] | None = None) -> FormatResult:
    widths = widths or DEFAULT_WIDTHS
    offsets: list[tuple[str, int, int]] = []
    offset = 0
    for field in FIELD_ORDER:
        width = widths[field]
        offsets.append((field, offset, width))
        offset += width
    line_width = offset

    records: list[dict[str, Any]] = []
    issues: list[Issue] = []
    raw_lines = text.splitlines()
    for line_number, line in enumerate(raw_lines, start=1):
        if not line:
            issues.append(Issue(E_FIELD_TRUNCATED, "empty fixed-width line", line=line_number, field="doc_seq"))
            continue
        if len(line) < line_width:
            for field, start, width in offsets:
                if len(line) < start + width:
                    issues.append(
                        Issue(
                            E_FIELD_TRUNCATED,
                            f"line length {len(line)} is shorter than field {field} ending at {start + width}",
                            line=line_number,
                            field=field,
                        )
                    )
                    break
            continue
        if len(line) > line_width:
            issues.append(
                Issue(
                    E_FIELD_TRUNCATED,
                    f"line length {len(line)} exceeds fixed record width {line_width}",
                    line=line_number,
                    field="trailing_characters",
                )
            )
            continue

        record: dict[str, Any] = {"_line": line_number}
        valid = True
        for field, start, width in offsets:
            chunk = line[start : start + width]
            if not _DIGITS.fullmatch(chunk):
                issues.append(
                    Issue(
                        E_FIELD_TRUNCATED,
                        f"field {field} contains illegal character(s): {chunk!r}",
                        line=line_number,
                        field=field,
                    )
                )
                valid = False
                continue
            record[field] = int(chunk)
        if valid:
            record["key"] = receipt_key(record["doc_seq"], record["fragment_seq"])
            records.append(record)
    return FormatResult(records=records, issues=issues)


def load_jv1_file(path: str | Path, widths: dict[str, int] | None = None) -> FormatResult:
    return parse_jv1(Path(path).read_text(encoding="utf-8"), widths)


def validate_jv2(data: Any, pos_codes: dict[str, int] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    if not isinstance(data, dict):
        return [Issue(E_FORMAT_INVALID, "jv2 root must be an object", path="$")]
    if data.get("format") != "jv2":
        issues.append(Issue(E_FORMAT_INVALID, "format must equal jv2", path="format"))
    if not isinstance(data.get("version"), str):
        issues.append(Issue(E_FORMAT_INVALID, "version must be a string", path="version"))
    receipts = data.get("receipts")
    if not isinstance(receipts, list):
        issues.append(Issue(E_FORMAT_INVALID, "receipts must be a list", path="receipts"))
        return issues

    for index, receipt in enumerate(receipts):
        path = f"receipts[{index}]"
        if not isinstance(receipt, dict):
            issues.append(Issue(E_FORMAT_INVALID, "receipt must be an object", path=path))
            continue
        for field in ("doc_seq", "fragment_seq"):
            if not isinstance(receipt.get(field), int) or isinstance(receipt.get(field), bool):
                issues.append(Issue(E_FORMAT_INVALID, f"{field} must be an integer", path=f"{path}.{field}"))
        score = receipt.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            issues.append(Issue(E_FORMAT_INVALID, "score must be a floating-point number", path=f"{path}.score"))
        pos = receipt.get("pos")
        if not isinstance(pos, str) or not pos:
            issues.append(Issue(E_FORMAT_INVALID, "pos must be a named POS string", path=f"{path}.pos"))
        elif pos_codes is not None and pos not in pos_codes:
            issues.append(Issue(E_FORMAT_INVALID, f"pos {pos!r} is not in code table", path=f"{path}.pos"))
        mode = receipt.get("rounding_mode")
        if not isinstance(mode, str) or mode not in ROUNDING_CODES:
            issues.append(Issue(E_FORMAT_INVALID, "rounding_mode is missing or invalid", path=f"{path}.rounding_mode"))
    return issues


def parse_jv2(text: str, pos_codes: dict[str, int] | None = None) -> FormatResult:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        issue = Issue(E_FORMAT_INVALID, f"invalid JSON: {exc.msg}", line=exc.lineno, path="$")
        return FormatResult(records=[], issues=[issue])
    issues = validate_jv2(data, pos_codes)
    records = data.get("receipts", []) if isinstance(data, dict) else []
    return FormatResult(records=records, issues=issues)


def jv1_to_jv2(
    jv1_records: list[dict[str, Any]],
    code_to_pos: dict[int, str],
    sidecar: dict[str, Any] | None = None,
) -> FormatResult:
    issues: list[Issue] = []
    sidecar_records = (sidecar or {}).get("records", {})
    receipts: list[dict[str, Any]] = []
    restored = 0
    for source in jv1_records:
        key = source.get("key") or receipt_key(source["doc_seq"], source["fragment_seq"])
        pos_name = code_to_pos.get(source["pos_code"])
        if pos_name is None:
            issues.append(Issue(E_FORMAT_INVALID, f"unknown POS code {source['pos_code']}", field="pos_code", line=source.get("_line")))
            continue
        entry = sidecar_records.get(key)
        if entry and "jv2_receipt" in entry:
            if entry.get("pos_code") != source["pos_code"] or entry.get("score_permille") != source["score_permille"]:
                issues.append(Issue(E_SIDECAR_MISMATCH, "sidecar core fields do not match jv1 record", field="sidecar", line=source.get("_line")))
                continue
            receipt = json.loads(json.dumps(entry["jv2_receipt"]))
            receipt["restored_from_sidecar"] = True
            restored += 1
        else:
            receipt = {
                "doc_seq": source["doc_seq"],
                "fragment_seq": source["fragment_seq"],
                "score": source["score_permille"] / 1000.0,
                "pos": pos_name,
                "rounding_mode": "EXACT",
                "precision_label": "jv1:permille:exact:P3",
                "precision_labels": ["jv1:permille", "exact-rational", "P3"],
                "restored_from_sidecar": False,
            }
        receipt["key"] = key
        receipts.append(receipt)
    result = FormatResult(records=receipts, issues=issues)
    result.sidecar = {"restored": restored, "available": len(sidecar_records)} if sidecar else None
    return result


def jv2_to_jv1(
    receipts: list[dict[str, Any]],
    pos_codes: dict[str, int],
    widths: dict[str, int] | None = None,
    line_offset: int = 0,
) -> FormatResult:
    widths = widths or DEFAULT_WIDTHS
    jv1_records: list[dict[str, Any]] = []
    sidecar_records: dict[str, Any] = {}
    issues: list[Issue] = []
    downgrade_entries: list[dict[str, Any]] = []

    for index, receipt in enumerate(receipts, start=1):
        path = f"receipts[{index - 1}]"
        try:
            doc_seq = receipt["doc_seq"]
            fragment_seq = receipt["fragment_seq"]
            score = float(receipt["score"])
            pos_name = receipt["pos"]
            mode = receipt.get("rounding_mode", "ROUND_HALF_UP")
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(Issue(E_FORMAT_INVALID, f"cannot downgrade receipt: {exc}", path=path))
            continue
        if pos_name not in pos_codes:
            issues.append(Issue(E_FORMAT_INVALID, f"unknown named POS {pos_name!r}", path=f"{path}.pos"))
            continue
        if mode not in ROUNDING_CODES:
            issues.append(Issue(E_FORMAT_INVALID, f"unknown rounding mode {mode!r}", path=f"{path}.rounding_mode"))
            continue
        score_permille = quantize_int(score, 1000, mode)
        if score_permille < 0 or score_permille > 1000:
            issues.append(Issue(E_FIELD_TRUNCATED, f"permille score {score_permille} is outside 0..1000", field="score_permille"))
            continue
        pos_code = pos_codes[pos_name]
        key = receipt_key(doc_seq, fragment_seq)
        jv1_record = {
            "doc_seq": doc_seq,
            "fragment_seq": fragment_seq,
            "score_permille": score_permille,
            "pos_code": pos_code,
            "key": key,
        }
        line_number = line_offset + index
        try:
            line = encode_jv1_record(jv1_record, widths, line_number)
        except FormatIssueError as exc:
            issues.append(exc.issue)
            continue
        jv1_record["line"] = line
        jv1_records.append(jv1_record)

        residual = quantization_residual(score, score_permille, 1000)
        precision_labels = list(receipt.get("precision_labels") or [])
        label = receipt.get("precision_label") or "jv2:float64:rounded-permille"
        if label not in precision_labels:
            precision_labels.append(label)
        sidecar_records[key] = {
            "key": key,
            "doc_seq": doc_seq,
            "fragment_seq": fragment_seq,
            "score_permille": score_permille,
            "pos_code": pos_code,
            "original_score": score,
            "truncation_residual": residual,
            "precision_label": label,
            "precision_labels": precision_labels,
            "named_pos": pos_name,
            "suppressed_rounding_mode": mode,
            "rounding_code": ROUNDING_CODES[mode],
            "jv2_receipt": _clean_receipt_for_restore(receipt),
            "sidecar_fingerprint": fingerprint(
                {"key": key, "score": score, "pos": pos_name, "mode": mode, "residual": residual},
                20,
            ),
        }
        downgrade_entries.append(
            {
                "key": key,
                "doc_seq": doc_seq,
                "fragment_seq": fragment_seq,
                "score_permille": score_permille,
                "pos_code": pos_code,
                "named_pos": pos_name,
                "rounding_mode": mode,
                "rounding_code": ROUNDING_CODES[mode],
                "residual": residual,
                "precision_label": label,
                "precision_labels": precision_labels,
            }
        )

    text = "\n".join(record["line"] for record in jv1_records)
    if text:
        text += "\n"
    sidecar = {
        "format": "jv1-downgrade-sidecar",
        "version": "1.0",
        "records": sidecar_records,
        "downgrade_entries": downgrade_entries,
    }
    return FormatResult(records=jv1_records, issues=issues, text=text, sidecar=sidecar)


def _clean_receipt_for_restore(receipt: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(receipt))
    cleaned.pop("key", None)
    cleaned.pop("restored_from_sidecar", None)
    return cleaned


def write_jv2_document(receipts: list[dict[str, Any]], rounding_mode: str = "ROUND_HALF_UP") -> dict[str, Any]:
    return {
        "format": "jv2",
        "version": "2.0",
        "rounding_mode": rounding_mode,
        "receipts": sorted(receipts, key=lambda item: (item["doc_seq"], item["fragment_seq"])),
    }
