"""Fixed width receipt parsers, normalization and bidirectional conversion.

Receipt history comes in three line based formats.  Every field has a fixed
text width so old files remain parseable without a schema service.

==========  ================================================================
Version     Fields (header defines exact widths)
==========  ================================================================
v1          ARTIFACT_ID(24) HIT_PERMILLE(5)                       total 29
v2          PARAM_VERSION(6) ARTIFACT_ID(24) HIT_PERMILLE(5)
            ELAPSED_MS(8)                                          total 43
v3          PARAM_VERSION(6) ARTIFACT_ID(24) HIT_PERMILLE(5)
            ELAPSED_MS(8) DIGEST(16) DUP_REFS(24)                 total 83
==========  ================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .errors import E_INPUT_INVALID, InputInvalidError, Violation
from .models import ReceiptRecord

V1_WIDTHS = (24, 5)
V2_WIDTHS = (6, 24, 5, 8)
V3_WIDTHS = (6, 24, 5, 8, 16, 24)
V1_HEADER = "ARTIFACT_ID             " + "HIT  "
V2_HEADER = "PARAM " + "ARTIFACT_ID            " + "HIT  " + "ELAPSED  "
V3_HEADER = V2_HEADER + "DIGEST16        " + "DUP_REFS                "
IRRECOVERABLE = "IRRECOVERABLE###"
FIELD_NAMES = {
    1: ("artifact_id", "hit_permille"),
    2: ("param_version", "artifact_id", "hit_permille", "elapsed_ms"),
    3: (
        "param_version",
        "artifact_id",
        "hit_permille",
        "elapsed_ms",
        "digest",
        "duplicate_refs",
    ),
}
PRESENT_BY_VERSION = {
    1: {"param_version": False, "elapsed_ms": False, "digest": False, "duplicate_refs": False},
    2: {"param_version": True, "elapsed_ms": True, "digest": False, "duplicate_refs": False},
    3: {"param_version": True, "elapsed_ms": True, "digest": True, "duplicate_refs": True},
}


def widths_for(version: int) -> Tuple[int, ...]:
    return {1: V1_WIDTHS, 2: V2_WIDTHS, 3: V3_WIDTHS}[version]


def header_for(version: int) -> str:
    return {1: V1_HEADER, 2: V2_HEADER, 3: V3_HEADER}[version]


def _slice_fields(line: str, widths: Tuple[int, ...]) -> List[str]:
    fields = []
    start = 0
    for width in widths:
        fields.append(line[start : start + width])
        start += width
    return fields


def parse_receipts(version: int, text: str, strict: bool = True) -> Tuple["ReceiptRecord", ...]:
    """Parse a receipt document and normalize it to ReceiptRecord tuples."""

    if version not in (1, 2, 3):
        raise InputInvalidError(
            [Violation(E_INPUT_INVALID, f"receipts.v{version}", "unsupported receipt version")]
        )
    widths = widths_for(version)
    expected_width = sum(widths)
    violations: List[Violation] = []
    records: List[ReceiptRecord] = []
    seen: Dict[str, int] = {}
    lines = text.splitlines()
    if not lines:
        raise InputInvalidError(
            [Violation(E_INPUT_INVALID, f"receipts.v{version}", "empty receipt document")]
        )
    expected_header = header_for(version)
    if lines[0] != expected_header:
        violations.append(
            Violation(
                E_INPUT_INVALID,
                f"receipts.v{version}:L1",
                "header width/text mismatch; fixed width header is required",
            )
        )
    start_line = 1 if lines[0] == expected_header else 0
    for offset, raw in enumerate(lines[start_line:], start=start_line + 1):
        if not raw.strip():
            continue
        loc = f"receipts.v{version}:L{offset}"
        if len(raw) != expected_width:
            violations.append(
                Violation(
                    E_INPUT_INVALID,
                    loc,
                    f"field width violation: expected {expected_width} chars, got {len(raw)}",
                )
            )
            continue
        fields = [part.strip() for part in _slice_fields(raw, widths)]
        values: Dict[str, Any] = dict(zip(FIELD_NAMES[version], fields))
        artifact_id = values["artifact_id"]
        if not artifact_id:
            violations.append(Violation(E_INPUT_INVALID, loc, "artifact_id is required"))
        elif artifact_id in seen:
            violations.append(
                Violation(
                    E_INPUT_INVALID,
                    loc,
                    f"duplicate artifact_id {artifact_id!r}, first seen at L{seen[artifact_id]}",
                )
            )
        else:
            seen[artifact_id] = offset
        hit_permille = _parse_int(values["hit_permille"], "hit_permille", loc, violations)
        if hit_permille is not None and not 0 <= hit_permille <= 1000:
            violations.append(
                Violation(E_INPUT_INVALID, loc, "hit_permille must be in [0,1000]")
            )
        param_version = None
        elapsed_ms = None
        digest = None
        duplicate_refs: Tuple[str, ...] = ()
        if version >= 2:
            param_version = _parse_optional_int(
                values["param_version"], "param_version", loc, violations, strict
            )
            elapsed_ms = _parse_optional_int(
                values["elapsed_ms"], "elapsed_ms", loc, violations, strict
            )
            if param_version is not None and param_version < 0:
                violations.append(Violation(E_INPUT_INVALID, loc, "param_version must be non-negative"))
            if elapsed_ms is not None and elapsed_ms < 0:
                violations.append(Violation(E_INPUT_INVALID, loc, "elapsed_ms must be non-negative"))
        if version == 3:
            digest = values["digest"] or None
            if digest is None and strict:
                violations.append(Violation(E_INPUT_INVALID, loc, "v3 digest is a required field"))
            ref_text = values["duplicate_refs"]
            duplicate_refs = tuple(item for item in ref_text.split(",") if item)
        if artifact_id and hit_permille is not None and 0 <= hit_permille <= 1000:
            records.append(
                ReceiptRecord(
                    source_version=version,
                    wire_version=version,
                    line_no=offset,
                    artifact_id=artifact_id,
                    hit_permille=hit_permille,
                    present=dict(PRESENT_BY_VERSION[version]),
                    param_version=param_version,
                    elapsed_ms=elapsed_ms,
                    digest=digest,
                    duplicate_refs=duplicate_refs,
                )
            )
    if violations:
        raise InputInvalidError(violations)
    return tuple(records)


def _parse_int(text: str, field: str, loc: str, violations: List[Violation]) -> Optional[int]:
    try:
        return int(text)
    except ValueError:
        violations.append(Violation(E_INPUT_INVALID, loc, f"{field} must be an integer"))
        return None


def _parse_optional_int(
    text: str,
    field: str,
    loc: str,
    violations: List[Violation],
    strict: bool,
) -> Optional[int]:
    if text == "":
        if strict:
            violations.append(Violation(E_INPUT_INVALID, loc, f"{field} is required"))
        return None
    return _parse_int(text, field, loc, violations)


def render_receipts(version: int, records: Tuple["ReceiptRecord", ...]) -> str:
    """Serialize normalized records back to a fixed width document."""

    widths = widths_for(version)
    lines = [header_for(version)]
    for record in records:
        if version == 1:
            values = [record.artifact_id, str(record.hit_permille)]
        elif version == 2:
            values = [
                _fixed_blank(record.param_version, widths[0]),
                record.artifact_id,
                str(record.hit_permille),
                _fixed_blank(record.elapsed_ms, widths[3]),
            ]
        else:
            refs = ",".join(record.duplicate_refs)
            values = [
                _fixed_blank(record.param_version, widths[0]),
                record.artifact_id,
                str(record.hit_permille),
                _fixed_blank(record.elapsed_ms, widths[3]),
                _required(record.digest, "digest"),
                refs,
            ]
        lines.append("".join(str(value).ljust(width) for value, width in zip(values, widths)))
    return "\n".join(lines) + "\n"


def _required(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"cannot render missing field {name}")
    return value


def _fixed_blank(value: Optional[int], width: int) -> str:
    if value is None:
        return " " * width
    return str(value)


@dataclass(frozen=True)
class ConversionResult:
    version: int
    records: Tuple["ReceiptRecord", ...]
    text: str
    sidecar: Dict[str, dict]
    precision_tags: Dict[str, str]
    diffs: Tuple[dict, ...]


def hit_rate_float(hit_permille: int) -> float:
    """Exact permille -> rate conversion (denominator 1000)."""

    return hit_permille / 1000.0


def precision_label(hit_permille: int) -> str:
    """Label explaining the exact decimal precision supplied by v1."""

    return f"permille:{hit_permille};rate={hit_permille:03d}/1000"


def upgrade_v1_to_v2(
    records: Tuple["ReceiptRecord", ...],
    param_version: int,
    elapsed_default: Optional[int] = None,
) -> ConversionResult:
    upgraded = []
    tags: Dict[str, str] = {}
    sidecar: Dict[str, dict] = {}
    for record in records:
        tags[record.artifact_id] = precision_label(record.hit_permille)
        upgraded.append(
            ReceiptRecord(
                source_version=1,
                wire_version=2,
                line_no=record.line_no,
                artifact_id=record.artifact_id,
                hit_permille=record.hit_permille,
                present=dict(PRESENT_BY_VERSION[2]),
                param_version=param_version,
                elapsed_ms=elapsed_default,
            )
        )
        sidecar[record.artifact_id] = {
            "upgraded_from": 1,
            "hit_rate_float": hit_rate_float(record.hit_permille),
            "precision_tag": tags[record.artifact_id],
            "elapsed_ms_origin": "missing-in-v1",
        }
    upgraded_tuple = tuple(upgraded)
    text = render_receipts(2, upgraded_tuple)
    reparsed = parse_receipts(2, text, strict=False)
    diffs = conversion_diffs(records, reparsed, sidecar)
    return ConversionResult(2, upgraded_tuple, text, sidecar, tags, diffs)


def downgrade_v2_to_v1(records: Tuple["ReceiptRecord", ...]) -> ConversionResult:
    downgraded = []
    sidecar: Dict[str, dict] = {}
    tags: Dict[str, str] = {}
    for record in records:
        label = precision_label(record.hit_permille)
        tags[record.artifact_id] = label
        # Permille is already integral; record the residual of any float rate
        # representation relative to its 3-decimal truncation.
        rate = hit_rate_float(record.hit_permille)
        truncated = round(rate, 3)
        residual = rate - truncated
        downgraded.append(
            ReceiptRecord(
                source_version=record.source_version,
                wire_version=1,
                line_no=record.line_no,
                artifact_id=record.artifact_id,
                hit_permille=record.hit_permille,
                present=dict(PRESENT_BY_VERSION[1]),
            )
        )
        sidecar[record.artifact_id] = {
            "downgraded_from": 2,
            "truncation_residual": residual,
            "precision_tag": label,
            "param_version": record.param_version,
            "elapsed_ms": record.elapsed_ms,
        }
    downgraded_tuple = tuple(downgraded)
    text = render_receipts(1, downgraded_tuple)
    reparsed = parse_receipts(1, text, strict=False)
    diffs = conversion_diffs(records, reparsed, sidecar)
    return ConversionResult(1, downgraded_tuple, text, sidecar, tags, diffs)


def downgrade_v3_to_v2(records: Tuple["ReceiptRecord", ...]) -> ConversionResult:
    downgraded = []
    sidecar: Dict[str, dict] = {}
    for record in records:
        downgraded.append(
            ReceiptRecord(
                source_version=record.source_version,
                wire_version=2,
                line_no=record.line_no,
                artifact_id=record.artifact_id,
                hit_permille=record.hit_permille,
                present=dict(PRESENT_BY_VERSION[2]),
                param_version=record.param_version,
                elapsed_ms=record.elapsed_ms,
            )
        )
        sidecar[record.artifact_id] = {
            "downgraded_from": 3,
            "digest": record.digest,
            "duplicate_refs": list(record.duplicate_refs),
        }
    downgraded_tuple = tuple(downgraded)
    text = render_receipts(2, downgraded_tuple)
    reparsed = parse_receipts(2, text, strict=False)
    diffs = conversion_diffs(records, reparsed, sidecar)
    return ConversionResult(2, downgraded_tuple, text, sidecar, {}, diffs)


def upgrade_v2_to_v3(
    records: Tuple["ReceiptRecord", ...],
    sidecar: Optional[Dict[str, dict]] = None,
) -> ConversionResult:
    sidecar = sidecar or {}
    upgraded = []
    restored: Dict[str, dict] = {}
    for record in records:
        entry = sidecar.get(record.artifact_id, {})
        digest = entry.get("digest")
        refs = tuple(entry.get("duplicate_refs", ()))
        recoverable = digest is not None
        if not recoverable:
            digest = IRRECOVERABLE
        restored[record.artifact_id] = {
            "digest_recoverable": recoverable,
            "duplicate_refs_recoverable": bool(refs),
        }
        upgraded.append(
            ReceiptRecord(
                source_version=record.source_version,
                wire_version=3,
                line_no=record.line_no,
                artifact_id=record.artifact_id,
                hit_permille=record.hit_permille,
                present=dict(PRESENT_BY_VERSION[3]),
                param_version=record.param_version,
                elapsed_ms=record.elapsed_ms,
                digest=digest,
                duplicate_refs=refs,
            )
        )
    upgraded_tuple = tuple(upgraded)
    text = render_receipts(3, upgraded_tuple)
    reparsed = parse_receipts(3, text, strict=False)
    diffs = conversion_diffs(records, reparsed, restored)
    return ConversionResult(3, upgraded_tuple, text, sidecar, {}, diffs)


def conversion_diffs(
    before: Tuple["ReceiptRecord", ...],
    after: Tuple["ReceiptRecord", ...],
    sidecar: Dict[str, dict],
) -> Tuple[dict, ...]:
    """Field-by-field differences produced by a format conversion."""

    before_by_id = {record.artifact_id: record for record in before}
    diffs: List[dict] = []
    for record in after:
        old = before_by_id.get(record.artifact_id)
        if old is None:
            diffs.append(
                {
                    "artifact_id": record.artifact_id,
                    "field": "*",
                    "before": None,
                    "after": record.to_dict(),
                    "disposition": "added",
                }
            )
            continue
        for field_name in (
            "source_version",
            "wire_version",
            "line_no",
            "artifact_id",
            "hit_permille",
            "param_version",
            "elapsed_ms",
            "digest",
            "duplicate_refs",
            "present",
        ):
            old_value = getattr(old, field_name)
            new_value = getattr(record, field_name)
            if old_value != new_value:
                disposition = "moved-to-sidecar" if field_name in sidecar.get(record.artifact_id, {}) else "converted"
                diffs.append(
                    {
                        "artifact_id": record.artifact_id,
                        "field": field_name,
                        "before": old_value,
                        "after": new_value,
                        "disposition": disposition,
                    }
                )
    return tuple(diffs)
