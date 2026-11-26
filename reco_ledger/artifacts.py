"""v1 fixed-width, v2 JSON and v3 JSON artifact compatibility layer."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .codes import Codes
from .config import AppConfig, QUANT_SCHEME_DIGITS, canonical_hash
from .diagnostics import DiagnosticBag


V1_WIDTHS = {
    "request_seq": 6,
    "candidate_seq": 6,
    "ctr_permille": 4,
    "quant_width": 1,
}
V1_LINE_WIDTH = sum(V1_WIDTHS.values())
V1_DIGITS = re.compile(r"^[0-9]+$")
IDENTIFIER_DIGITS = re.compile(r"^[rc]?([0-9]{1,6})$")

V2_FIELDS = {
    "request_id",
    "candidate_id",
    "ctr",
    "cvr",
    "param_version",
    "elapsed_ms",
}
V3_EXTRA_FIELDS = {
    "content_digest",
    "covered_estimates",
    "conversion_visible",
    "quant_scheme",
    "request_seq",
    "candidate_seq",
    "ctr_permille",
    "quant_width",
    "completeness",
    "missing_fields",
    "bypass_id",
    "provenance",
}
COVERED_REF_FIELDS = {
    "source_file",
    "line_no",
    "request_id",
    "candidate_id",
    "source_version",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def request_id_from_seq(sequence: int) -> str:
    return f"r{sequence:06d}"


def candidate_id_from_seq(sequence: int) -> str:
    return f"c{sequence:06d}"


def identifier_to_seq(identifier: str) -> int | None:
    match = IDENTIFIER_DIGITS.fullmatch(identifier)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class InternalEstimate:
    """Lossless-enough union of the three exchange schemas."""

    source_version: int
    source_file: str
    line_no: int
    request_id: str
    candidate_id: str
    ctr: float | None = None
    cvr: float | None = None
    ctr_permille: int | None = None
    quant_width: int | None = None
    quant_scheme: str | None = None
    param_version: int | None = None
    elapsed_ms: int | None = None
    content_digest: str | None = None
    covered_estimates: tuple[Any, ...] = ()
    conversion_visible: bool | None = None
    request_seq: int | None = None
    candidate_seq: int | None = None
    missing_fields: tuple[str, ...] = ()
    completeness: str | None = None
    bypass_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.candidate_id)

    @property
    def origin_ref(self) -> str:
        return f"{self.source_file}:{self.line_no}"

    @property
    def record_id(self) -> str:
        return canonical_hash(
            {
                "source_version": self.source_version,
                "source_file": self.source_file,
                "line_no": self.line_no,
                "key": list(self.key),
            }
        )[:16]

    def to_dict(self) -> dict[str, Any]:
        covered = [item if isinstance(item, str) else dict(item) for item in self.covered_estimates]
        return {
            "source_version": self.source_version,
            "source_file": self.source_file,
            "line_no": self.line_no,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "ctr": self.ctr,
            "cvr": self.cvr,
            "ctr_permille": self.ctr_permille,
            "quant_width": self.quant_width,
            "quant_scheme": self.quant_scheme,
            "param_version": self.param_version,
            "elapsed_ms": self.elapsed_ms,
            "content_digest": self.content_digest,
            "covered_estimates": covered,
            "conversion_visible": self.conversion_visible,
            "request_seq": self.request_seq,
            "candidate_seq": self.candidate_seq,
            "missing_fields": list(self.missing_fields),
            "completeness": self.completeness,
            "bypass_id": self.bypass_id,
            "provenance": dict(self.provenance),
        }

    def v3_payload(self) -> dict[str, Any]:
        """Serialize as a v3 interchange row, including explicit null markers."""

        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "ctr": self.ctr,
            "cvr": self.cvr,
            "param_version": self.param_version,
            "elapsed_ms": self.elapsed_ms,
            "content_digest": self.content_digest,
            "covered_estimates": [
                item if isinstance(item, str) else dict(item)
                for item in self.covered_estimates
            ],
            "conversion_visible": self.conversion_visible,
            "quant_scheme": self.quant_scheme,
            "request_seq": self.request_seq,
            "candidate_seq": self.candidate_seq,
            "ctr_permille": self.ctr_permille,
            "quant_width": self.quant_width,
            "completeness": self.completeness,
            "missing_fields": list(self.missing_fields),
            "bypass_id": self.bypass_id,
            "provenance": dict(self.provenance),
        }
        return payload


@dataclass
class ArtifactLoadResult:
    estimates: list[InternalEstimate]
    diagnostics: DiagnosticBag


def _allowed_quant_widths(config: AppConfig) -> set[int]:
    widths: set[int] = set()
    for scheme in config.quantization.allowed_schemes:
        if scheme in QUANT_SCHEME_DIGITS:
            widths.add(QUANT_SCHEME_DIGITS[scheme])
    return widths or set(QUANT_SCHEME_DIGITS.values())


def _make_v1_line(
    request_seq: int,
    candidate_seq: int,
    ctr_permille: int,
    quant_width: int,
) -> str:
    if request_seq < 0 or request_seq >= 10 ** V1_WIDTHS["request_seq"]:
        raise ValueError("request_seq exceeds v1 field width")
    if candidate_seq < 0 or candidate_seq >= 10 ** V1_WIDTHS["candidate_seq"]:
        raise ValueError("candidate_seq exceeds v1 field width")
    if ctr_permille < 0 or ctr_permille > 1000:
        raise ValueError("ctr_permille outside [0, 1000]")
    if quant_width < 1 or quant_width > 9:
        raise ValueError("quant_width outside [1, 9]")
    return (
        f"{request_seq:0{V1_WIDTHS['request_seq']}d}"
        f"{candidate_seq:0{V1_WIDTHS['candidate_seq']}d}"
        f"{ctr_permille:0{V1_WIDTHS['ctr_permille']}d}"
        f"{quant_width:0{V1_WIDTHS['quant_width']}d}"
    )


def estimate_to_v1_line(estimate: InternalEstimate) -> str:
    request_seq = estimate.request_seq
    if request_seq is None:
        request_seq = identifier_to_seq(estimate.request_id)
    candidate_seq = estimate.candidate_seq
    if candidate_seq is None:
        candidate_seq = identifier_to_seq(estimate.candidate_id)
    if request_seq is None or candidate_seq is None:
        raise ValueError("v1 conversion requires numeric sequence identifiers")
    if estimate.ctr_permille is not None:
        permille = estimate.ctr_permille
    elif estimate.ctr is not None:
        scaled = Decimal(str(estimate.ctr)) * Decimal(1000)
        permille = int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    else:
        raise ValueError("v1 conversion requires ctr or ctr_permille")
    permille = max(0, min(1000, permille))
    quant_width = estimate.quant_width
    if quant_width is None and estimate.quant_scheme is not None:
        quant_width = QUANT_SCHEME_DIGITS.get(estimate.quant_scheme)
    if quant_width is None:
        quant_width = 1
    return _make_v1_line(request_seq, candidate_seq, permille, quant_width)


def parse_v1_text(
    text: str,
    *,
    config: AppConfig,
    source_file: str = "<v1>",
    injection_id: str | None = None,
) -> ArtifactLoadResult:
    bag = DiagnosticBag()
    estimates: list[InternalEstimate] = []
    allowed_widths = _allowed_quant_widths(config)
    for line_index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if len(line) != V1_LINE_WIDTH or not V1_DIGITS.fullmatch(line):
            bag.add(
                Codes.FIELD_TRUNCATED,
                "v1 line does not match fixed field widths or contains illegal characters",
                source=source_file,
                entry_index=line_index,
                injection_id=injection_id,
                details={"expected_width": V1_LINE_WIDTH, "actual_width": len(line)},
            )
            continue
        offset = 0
        request_seq = int(line[offset : offset + V1_WIDTHS["request_seq"]])
        offset += V1_WIDTHS["request_seq"]
        candidate_seq = int(line[offset : offset + V1_WIDTHS["candidate_seq"]])
        offset += V1_WIDTHS["candidate_seq"]
        ctr_permille = int(line[offset : offset + V1_WIDTHS["ctr_permille"]])
        offset += V1_WIDTHS["ctr_permille"]
        quant_width = int(line[offset : offset + V1_WIDTHS["quant_width"]])
        if ctr_permille < 0 or ctr_permille > 1000:
            bag.add(
                Codes.VALUE_RANGE,
                "v1 ctr_permille must be in [0, 1000]",
                source=source_file,
                entry_index=line_index,
                field_name="ctr_permille",
                injection_id=injection_id,
            )
            continue
        if quant_width not in allowed_widths:
            bag.add(
                Codes.VALUE_RANGE,
                f"v1 quant_width {quant_width} is outside allowed enum {sorted(allowed_widths)}",
                source=source_file,
                entry_index=line_index,
                field_name="quant_width",
                injection_id=injection_id,
            )
            continue
        estimates.append(
            InternalEstimate(
                source_version=1,
                source_file=source_file,
                line_no=line_index,
                request_id=request_id_from_seq(request_seq),
                candidate_id=candidate_id_from_seq(candidate_seq),
                ctr=ctr_permille / 1000.0,
                ctr_permille=ctr_permille,
                quant_width=quant_width,
                quant_scheme=f"q{quant_width}",
                request_seq=request_seq,
                candidate_seq=candidate_seq,
                missing_fields=(
                    "cvr",
                    "param_version",
                    "elapsed_ms",
                    "content_digest",
                    "covered_estimates",
                    "conversion_visible",
                ),
                completeness="partial_v1",
            )
        )
    return ArtifactLoadResult(estimates, bag)


def _decode_json_records(
    text: str,
    *,
    source_file: str,
    bag: DiagnosticBag,
) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[tuple[int, Mapping[str, Any]]] = []
        for line_index, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                bag.add(
                    Codes.TYPE_MISMATCH,
                    f"invalid JSON record: {exc.msg}",
                    source=source_file,
                    entry_index=line_index,
                )
                continue
            records.append((line_index, item))
        return records

    if isinstance(value, Mapping):
        if isinstance(value.get("records"), list):
            return [(index, item) for index, item in enumerate(value["records"], start=1)]
        if isinstance(value.get("estimates"), list):
            return [(index, item) for index, item in enumerate(value["estimates"], start=1)]
        return [(1, value)]
    if isinstance(value, list):
        return [(index, item) for index, item in enumerate(value, start=1)]
    bag.add(Codes.TYPE_MISMATCH, "JSON root must be an object or list", source=source_file)
    return []


def _validate_common_json_fields(
    record: Mapping[str, Any],
    *,
    source_file: str,
    line_no: int,
    bag: DiagnosticBag,
    allowed: set[str],
    injection_id: str | None,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    bad = False
    for key in record:
        if key not in allowed:
            bag.add(
                Codes.FIELD_UNKNOWN,
                f"unknown artifact field {key}",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
    missing: list[str] = []
    for key in V2_FIELDS:
        if key not in record:
            bag.add(
                Codes.FIELD_MISSING,
                f"missing artifact field {key}",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            missing.append(key)
            bad = True
    if bad:
        return None, tuple(missing)

    validated: dict[str, Any] = {}
    for key in ("request_id", "candidate_id"):
        value = record[key]
        if not isinstance(value, str) or not value:
            bag.add(
                Codes.TYPE_MISMATCH,
                f"{key} must be a non-empty string",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
        else:
            validated[key] = value
    for key in ("ctr", "cvr"):
        value = record[key]
        if not _is_number(value):
            bag.add(
                Codes.TYPE_MISMATCH,
                f"{key} must be numeric",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            bag.add(
                Codes.VALUE_RANGE,
                f"{key} must be finite and within [0, 1]",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
        else:
            validated[key] = number
    for key in ("param_version", "elapsed_ms"):
        value = record[key]
        if not _is_int(value):
            bag.add(
                Codes.TYPE_MISMATCH,
                f"{key} must be an integer",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
        elif value < 0:
            bag.add(
                Codes.VALUE_RANGE,
                f"{key} must be non-negative",
                source=source_file,
                entry_index=line_no,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
        else:
            validated[key] = value
    return (validated if not bad else None), tuple(missing)


def _validate_covered_estimates(
    value: Any,
    *,
    source_file: str,
    line_no: int,
    bag: DiagnosticBag,
    injection_id: str | None,
) -> tuple[Any, ...] | None:
    if value is None:
        return ()
    if not isinstance(value, list):
        bag.add(
            Codes.TYPE_MISMATCH,
            "covered_estimates must be a list or null",
            source=source_file,
            entry_index=line_no,
            field_name="covered_estimates",
            injection_id=injection_id,
        )
        return None
    result: list[Any] = []
    bad = False
    for ref_index, ref in enumerate(value):
        if isinstance(ref, str):
            if not ref:
                bag.add(
                    Codes.VALUE_RANGE,
                    f"covered_estimates[{ref_index}] must not be empty",
                    source=source_file,
                    entry_index=line_no,
                    field_name=f"covered_estimates[{ref_index}]",
                    injection_id=injection_id,
                )
                bad = True
            result.append(ref)
            continue
        if not isinstance(ref, Mapping):
            bag.add(
                Codes.TYPE_MISMATCH,
                f"covered_estimates[{ref_index}] must be a string or object",
                source=source_file,
                entry_index=line_no,
                field_name=f"covered_estimates[{ref_index}]",
                injection_id=injection_id,
            )
            bad = True
            continue
        unknown = set(ref) - COVERED_REF_FIELDS
        if unknown:
            for key in sorted(unknown):
                bag.add(
                    Codes.FIELD_UNKNOWN,
                    f"unknown covered reference field {key}",
                    source=source_file,
                    entry_index=line_no,
                    field_name=f"covered_estimates[{ref_index}].{key}",
                    injection_id=injection_id,
                )
            bad = True
            continue
        if "line_no" in ref and (not _is_int(ref["line_no"]) or ref["line_no"] < 1):
            bag.add(
                Codes.VALUE_RANGE,
                "covered reference line_no must be a positive integer",
                source=source_file,
                entry_index=line_no,
                field_name=f"covered_estimates[{ref_index}].line_no",
                injection_id=injection_id,
            )
            bad = True
            continue
        if "source_file" in ref and not isinstance(ref["source_file"], str):
            bag.add(
                Codes.TYPE_MISMATCH,
                "covered reference source_file must be a string",
                source=source_file,
                entry_index=line_no,
                field_name=f"covered_estimates[{ref_index}].source_file",
                injection_id=injection_id,
            )
            bad = True
            continue
        if not any(
            name in ref for name in ("source_file", "line_no", "request_id", "candidate_id")
        ):
            bag.add(
                Codes.VALUE_RANGE,
                "covered reference needs at least one identifying field",
                source=source_file,
                entry_index=line_no,
                field_name=f"covered_estimates[{ref_index}]",
                injection_id=injection_id,
            )
            bad = True
            continue
        result.append(dict(ref))
    return tuple(result) if not bad else None


def parse_v2_text(
    text: str,
    *,
    config: AppConfig,
    source_file: str = "<v2>",
    injection_id: str | None = None,
) -> ArtifactLoadResult:
    bag = DiagnosticBag()
    estimates: list[InternalEstimate] = []
    for line_no, record in _decode_json_records(text, source_file=source_file, bag=bag):
        if not isinstance(record, Mapping):
            bag.add(
                Codes.TYPE_MISMATCH,
                "v2 record must be an object",
                source=source_file,
                entry_index=line_no,
                injection_id=injection_id,
            )
            continue
        validated, missing = _validate_common_json_fields(
            record,
            source_file=source_file,
            line_no=line_no,
            bag=bag,
            allowed=V2_FIELDS,
            injection_id=injection_id,
        )
        if validated is None:
            continue
        estimates.append(
            InternalEstimate(
                source_version=2,
                source_file=source_file,
                line_no=line_no,
                request_id=validated["request_id"],
                candidate_id=validated["candidate_id"],
                ctr=validated["ctr"],
                cvr=validated["cvr"],
                param_version=validated["param_version"],
                elapsed_ms=validated["elapsed_ms"],
                missing_fields=(
                    "request_seq",
                    "candidate_seq",
                    "ctr_permille",
                    "quant_width",
                    "quant_scheme",
                    "content_digest",
                    "covered_estimates",
                    "conversion_visible",
                ),
                completeness="v2_core_complete",
            )
        )
    return ArtifactLoadResult(estimates, bag)


def parse_v3_text(
    text: str,
    *,
    config: AppConfig,
    source_file: str = "<v3>",
    injection_id: str | None = None,
) -> ArtifactLoadResult:
    bag = DiagnosticBag()
    estimates: list[InternalEstimate] = []
    allowed = V2_FIELDS | V3_EXTRA_FIELDS
    for line_no, record in _decode_json_records(text, source_file=source_file, bag=bag):
        if not isinstance(record, Mapping):
            bag.add(
                Codes.TYPE_MISMATCH,
                "v3 record must be an object",
                source=source_file,
                entry_index=line_no,
                injection_id=injection_id,
            )
            continue
        validated, missing = _validate_common_json_fields(
            record,
            source_file=source_file,
            line_no=line_no,
            bag=bag,
            allowed=allowed,
            injection_id=injection_id,
        )
        if validated is None:
            continue
        bad = False
        missing_fields = list(missing)
        for required_extra in ("content_digest", "covered_estimates", "conversion_visible"):
            if required_extra not in record:
                bag.add(
                    Codes.FIELD_MISSING,
                    f"v3 requires field {required_extra}",
                    source=source_file,
                    entry_index=line_no,
                    field_name=required_extra,
                    injection_id=injection_id,
                )
                bad = True

        content_digest = record.get("content_digest")
        if content_digest is not None and not isinstance(content_digest, str):
            bag.add(
                Codes.TYPE_MISMATCH,
                "content_digest must be a string or null",
                source=source_file,
                entry_index=line_no,
                field_name="content_digest",
                injection_id=injection_id,
            )
            bad = True
        if content_digest is None:
            missing_fields.append("content_digest")
        covered = _validate_covered_estimates(
            record.get("covered_estimates"),
            source_file=source_file,
            line_no=line_no,
            bag=bag,
            injection_id=injection_id,
        )
        if covered is None:
            bad = True
        elif not covered:
            missing_fields.append("covered_estimates")
        conversion_visible = record.get("conversion_visible")
        if conversion_visible is not None and not isinstance(conversion_visible, bool):
            bag.add(
                Codes.TYPE_MISMATCH,
                "conversion_visible must be boolean or null",
                source=source_file,
                entry_index=line_no,
                field_name="conversion_visible",
                injection_id=injection_id,
            )
            bad = True
        if conversion_visible is None:
            missing_fields.append("conversion_visible")

        quant_scheme = record.get("quant_scheme")
        if quant_scheme is not None:
            if not isinstance(quant_scheme, str):
                bag.add(
                    Codes.TYPE_MISMATCH,
                    "quant_scheme must be a string when present",
                    source=source_file,
                    entry_index=line_no,
                    field_name="quant_scheme",
                    injection_id=injection_id,
                )
                bad = True
            elif quant_scheme not in config.quantization.allowed_schemes:
                bag.add(
                    Codes.VALUE_RANGE,
                    f"quant_scheme {quant_scheme!r} is outside allowed enum "
                    f"{list(config.quantization.allowed_schemes)}",
                    source=source_file,
                    entry_index=line_no,
                    field_name="quant_scheme",
                    injection_id=injection_id,
                )
                bad = True
        else:
            missing_fields.append("quant_scheme")

        request_seq = record.get("request_seq")
        candidate_seq = record.get("candidate_seq")
        ctr_permille = record.get("ctr_permille")
        quant_width = record.get("quant_width")
        for name, value in (
            ("request_seq", request_seq),
            ("candidate_seq", candidate_seq),
            ("ctr_permille", ctr_permille),
            ("quant_width", quant_width),
        ):
            if value is not None and (not _is_int(value) or value < 0):
                bag.add(
                    Codes.TYPE_MISMATCH
                    if not _is_int(value)
                    else Codes.VALUE_RANGE,
                    f"{name} must be a non-negative integer or null",
                    source=source_file,
                    entry_index=line_no,
                    field_name=name,
                    injection_id=injection_id,
                )
                bad = True
            if value is None:
                missing_fields.append(name)
        if ctr_permille is not None and ctr_permille > 1000:
            bag.add(
                Codes.VALUE_RANGE,
                "ctr_permille must be within [0, 1000]",
                source=source_file,
                entry_index=line_no,
                field_name="ctr_permille",
                injection_id=injection_id,
            )
            bad = True
        if quant_width is not None and quant_width not in _allowed_quant_widths(config):
            bag.add(
                Codes.VALUE_RANGE,
                f"quant_width {quant_width} is outside allowed enum",
                source=source_file,
                entry_index=line_no,
                field_name="quant_width",
                injection_id=injection_id,
            )
            bad = True

        completeness = record.get("completeness")
        if completeness is not None and not isinstance(completeness, str):
            bag.add(
                Codes.TYPE_MISMATCH,
                "completeness must be a string or null",
                source=source_file,
                entry_index=line_no,
                field_name="completeness",
                injection_id=injection_id,
            )
            bad = True
        declared_missing = record.get("missing_fields")
        if declared_missing is not None:
            if not isinstance(declared_missing, list) or not all(
                isinstance(item, str) for item in declared_missing
            ):
                bag.add(
                    Codes.TYPE_MISMATCH,
                    "missing_fields must be a list of strings or null",
                    source=source_file,
                    entry_index=line_no,
                    field_name="missing_fields",
                    injection_id=injection_id,
                )
                bad = True
            else:
                missing_fields.extend(declared_missing)
        bypass_id = record.get("bypass_id")
        if bypass_id is not None and not isinstance(bypass_id, str):
            bag.add(
                Codes.TYPE_MISMATCH,
                "bypass_id must be a string or null",
                source=source_file,
                entry_index=line_no,
                field_name="bypass_id",
                injection_id=injection_id,
            )
            bad = True
        provenance = record.get("provenance")
        if provenance is not None and not isinstance(provenance, Mapping):
            bag.add(
                Codes.TYPE_MISMATCH,
                "provenance must be an object or null",
                source=source_file,
                entry_index=line_no,
                field_name="provenance",
                injection_id=injection_id,
            )
            bad = True
        if bad:
            continue
        estimates.append(
            InternalEstimate(
                source_version=3,
                source_file=source_file,
                line_no=line_no,
                request_id=validated["request_id"],
                candidate_id=validated["candidate_id"],
                ctr=validated["ctr"],
                cvr=validated["cvr"],
                param_version=validated["param_version"],
                elapsed_ms=validated["elapsed_ms"],
                content_digest=content_digest,
                covered_estimates=covered or (),
                conversion_visible=conversion_visible,
                quant_scheme=quant_scheme,
                request_seq=request_seq,
                candidate_seq=candidate_seq,
                ctr_permille=ctr_permille,
                quant_width=quant_width,
                missing_fields=tuple(dict.fromkeys(missing_fields)),
                completeness=completeness,
                bypass_id=bypass_id,
                provenance=dict(provenance or {}),
            )
        )
    return ArtifactLoadResult(estimates, bag)


def parse_artifact_file(
    path: str | Path,
    *,
    config: AppConfig,
    version: int | None = None,
    injection_id: str | None = None,
) -> ArtifactLoadResult:
    source_file = str(path)
    text = Path(path).read_text(encoding="utf-8")
    if version is None:
        name = Path(path).name.lower()
        if "v3" in name:
            version = 3
        elif "v2" in name:
            version = 2
        elif "v1" in name:
            version = 1
        else:
            stripped = text.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                version = 3 if (
                    "content_digest" in stripped
                    or "covered_estimates" in stripped
                    or "conversion_visible" in stripped
                ) else 2
            else:
                version = 1
    if version == 1:
        return parse_v1_text(
            text, config=config, source_file=source_file, injection_id=injection_id
        )
    if version == 2:
        return parse_v2_text(
            text, config=config, source_file=source_file, injection_id=injection_id
        )
    if version == 3:
        return parse_v3_text(
            text, config=config, source_file=source_file, injection_id=injection_id
        )
    bag = DiagnosticBag()
    bag.add(Codes.VALUE_RANGE, f"unsupported artifact version {version}", source=source_file)
    return ArtifactLoadResult([], bag)


def serialize_v1(estimates: Sequence[InternalEstimate]) -> str:
    return "\n".join(estimate_to_v1_line(estimate) for estimate in estimates) + "\n"


def serialize_v2(estimates: Sequence[InternalEstimate]) -> str:
    payloads = [
        {
            "request_id": estimate.request_id,
            "candidate_id": estimate.candidate_id,
            "ctr": estimate.ctr,
            "cvr": estimate.cvr,
            "param_version": estimate.param_version,
            "elapsed_ms": estimate.elapsed_ms,
        }
        for estimate in estimates
    ]
    return json.dumps(payloads, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def serialize_v3(estimates: Sequence[InternalEstimate]) -> str:
    return (
        json.dumps(
            [estimate.v3_payload() for estimate in estimates],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


@dataclass
class ConversionBypassLedger:
    """Per-row metadata that cannot be represented in v1."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entries)

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "count": self.count,
            "entries": self.entries,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConversionBypassLedger":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data.get("entries", []) if isinstance(data, Mapping) else []
        return cls(entries=[dict(item) for item in entries])


def downgrade_to_v1(
    estimates: Sequence[InternalEstimate],
    *,
    config: AppConfig,
) -> tuple[list[str], ConversionBypassLedger]:
    lines: list[str] = []
    ledger = ConversionBypassLedger()
    allowed_widths = _allowed_quant_widths(config)
    for estimate in estimates:
        if estimate.request_seq is None:
            request_seq = identifier_to_seq(estimate.request_id)
        else:
            request_seq = estimate.request_seq
        if estimate.candidate_seq is None:
            candidate_seq = identifier_to_seq(estimate.candidate_id)
        else:
            candidate_seq = estimate.candidate_seq
        if request_seq is None or candidate_seq is None:
            raise ValueError("cannot downgrade a record without numeric identifiers")
        if estimate.ctr is None:
            if estimate.ctr_permille is None:
                raise ValueError("cannot downgrade a record without ctr")
            exact_permille = Decimal(estimate.ctr_permille)
        else:
            exact_permille = Decimal(str(estimate.ctr)) * Decimal(1000)
        rounded_permille = int(
            exact_permille.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        rounded_permille = max(0, min(1000, rounded_permille))
        residual = exact_permille - Decimal(rounded_permille)
        precision_label = (
            "exact_permille"
            if residual == 0
            else ("clamped" if rounded_permille in (0, 1000) else "rounded_half_up")
        )
        quant_width = estimate.quant_width
        if quant_width is None and estimate.quant_scheme is not None:
            quant_width = QUANT_SCHEME_DIGITS.get(estimate.quant_scheme)
        if quant_width is None:
            quant_width = config.quantization.decimal_digits
        if quant_width not in allowed_widths:
            raise ValueError(f"quant_width {quant_width} is outside allowed enum")
        line = _make_v1_line(
            request_seq, candidate_seq, rounded_permille, quant_width
        )
        v1_line_no = len(lines) + 1
        lines.append(line)
        bypass = {
            "bypass_id": canonical_hash(
                {
                    "record_id": estimate.record_id,
                    "v1_line_no": v1_line_no,
                    "key": list(estimate.key),
                }
            )[:20],
            "record_id": estimate.record_id,
            "record_key": {"request_id": estimate.request_id, "candidate_id": estimate.candidate_id},
            "origin": {
                "source_version": estimate.source_version,
                "source_file": estimate.source_file,
                "line_no": estimate.line_no,
            },
            "v1_line_no": v1_line_no,
            "v1_line": line,
            "exact_ctr_permille": str(exact_permille),
            "rounded_ctr_permille": rounded_permille,
            "truncation_residual_permille": str(residual),
            "precision_label": precision_label,
            "covered_estimates": [
                item if isinstance(item, str) else dict(item)
                for item in estimate.covered_estimates
            ],
            "conversion_visible": estimate.conversion_visible,
            "quant_scheme": estimate.quant_scheme,
            "original": estimate.to_dict(),
        }
        ledger.append(bypass)
    return lines, ledger


def _find_bypass(
    estimate: InternalEstimate,
    *,
    line_text: str,
    line_no: int,
    entries: Sequence[Mapping[str, Any]],
    used: set[int],
) -> Mapping[str, Any] | None:
    key = {"request_id": estimate.request_id, "candidate_id": estimate.candidate_id}
    for index, entry in enumerate(entries):
        if index in used:
            continue
        if entry.get("v1_line") == line_text:
            used.add(index)
            return entry
    for index, entry in enumerate(entries):
        if index in used:
            continue
        if entry.get("record_key") == key:
            used.add(index)
            return entry
    return None


def upgrade_v1_to_v3(
    estimates: Sequence[InternalEstimate],
    *,
    config: AppConfig,
    bypass: ConversionBypassLedger | None = None,
    v1_lines: Sequence[str] | None = None,
) -> list[InternalEstimate]:
    """Upgrade v1 rows. Bypass metadata restores fields lost in v1."""

    entries = bypass.entries if bypass is not None else []
    used: set[int] = set()
    upgraded: list[InternalEstimate] = []
    for index, estimate in enumerate(estimates):
        if estimate.source_version != 1:
            raise ValueError("upgrade_v1_to_v3 only accepts v1-normalized rows")
        line_text = v1_lines[index] if v1_lines is not None else estimate_to_v1_line(estimate)
        matched = _find_bypass(
            estimate,
            line_text=line_text,
            line_no=estimate.line_no,
            entries=entries,
            used=used,
        )
        if estimate.ctr_permille is None:
            raise ValueError("v1 row is missing ctr_permille")
        ctr = Decimal(estimate.ctr_permille) / Decimal(1000)
        ctr_float = float(ctr)
        quant_width = estimate.quant_width or config.quantization.decimal_digits
        quant_scheme = f"q{quant_width}"
        if quant_scheme not in config.quantization.allowed_schemes:
            raise ValueError(f"mapped quant scheme {quant_scheme} is not allowed")
        cvr: float | None = None
        param_version: int | None = None
        elapsed_ms: int | None = None
        content_digest: str | None = None
        covered: tuple[Any, ...] = ()
        conversion_visible: bool | None = None
        bypass_id: str | None = None
        provenance: dict[str, Any] = {"upgraded_from": "v1"}
        if matched is not None:
            original = matched.get("original", {})
            cvr = original.get("cvr")
            param_version = original.get("param_version")
            elapsed_ms = original.get("elapsed_ms")
            content_digest = original.get("content_digest")
            covered_raw = matched.get("covered_estimates", [])
            covered = tuple(
                item if isinstance(item, str) else dict(item) for item in covered_raw
            )
            conversion_visible = matched.get("conversion_visible")
            bypass_id = matched.get("bypass_id")
            provenance["bypass"] = {
                "bypass_id": bypass_id,
                "precision_label": matched.get("precision_label"),
                "truncation_residual_permille": matched.get(
                    "truncation_residual_permille"
                ),
            }
        missing_fields: list[str] = []
        if cvr is None:
            missing_fields.append("cvr")
        if param_version is None:
            missing_fields.append("param_version")
        if elapsed_ms is None:
            missing_fields.append("elapsed_ms")
        if content_digest is None:
            missing_fields.append("content_digest")
        if not covered:
            missing_fields.append("covered_estimates")
        if conversion_visible is None:
            missing_fields.append("conversion_visible")
        completeness = (
            "complete_via_bypass"
            if matched is not None and not missing_fields
            else "partial_v1_missing_" + "_".join(missing_fields)
            if missing_fields
            else "complete_v1_semantics"
        )
        upgraded.append(
            replace(
                estimate,
                source_version=3,
                ctr=ctr_float,
                cvr=cvr,
                quant_scheme=quant_scheme,
                param_version=param_version,
                elapsed_ms=elapsed_ms,
                content_digest=content_digest,
                covered_estimates=covered,
                conversion_visible=conversion_visible,
                missing_fields=tuple(dict.fromkeys(missing_fields)),
                completeness=completeness,
                bypass_id=bypass_id,
                provenance=provenance,
            )
        )
    return upgraded


def load_artifact_directory(
    directory: str | Path,
    *,
    config: AppConfig,
    suffixes: Iterable[str] = (".txt", ".json", ".jsonl"),
) -> ArtifactLoadResult:
    bag = DiagnosticBag()
    estimates: list[InternalEstimate] = []
    root = Path(directory)
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in set(suffixes)
    )
    for path in files:
        result = parse_artifact_file(path, config=config)
        estimates.extend(result.estimates)
        bag.extend(result.diagnostics.items)
    return ArtifactLoadResult(estimates, bag)

