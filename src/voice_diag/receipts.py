from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import FieldTruncatedError, SchemaError
from .util import canonical_json, json_hash, round_half_away_from_zero


_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
_DECIMAL_PATTERN = re.compile(r"^[0-9]+$")
_REQUIRED_A1_FIELDS = ("interval_id", "speaker_id", "confidence", "command_code")


@dataclass(frozen=True)
class A1Layout:
    """The physical layout of one fixed-width a1 evidence line.

    The separator is intentionally structural, not part of a field.  Numeric
    fields are zero padded; identifier fields are space padded on the right.
    """

    interval_id: int = 12
    speaker_id: int = 8
    confidence: int = 6
    command_code: int = 5
    separator: str = "|"

    def __post_init__(self) -> None:
        for name in _REQUIRED_A1_FIELDS:
            width = getattr(self, name)
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                raise SchemaError(f"a1 width for {name} must be a positive integer")
        if not self.separator or len(self.separator) > 1:
            raise SchemaError("a1 separator must be exactly one character")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, separator: str = "|") -> "A1Layout":
        try:
            return cls(
                interval_id=int(value["interval_id"]),
                speaker_id=int(value["speaker_id"]),
                confidence=int(value["confidence"]),
                command_code=int(value["command_code"]),
                separator=separator,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError("invalid a1 field width mapping") from exc

    @property
    def widths(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _REQUIRED_A1_FIELDS}

    @property
    def line_width(self) -> int:
        return sum(self.widths.values()) + len(self.separator) * (len(self.widths) - 1)


@dataclass(frozen=True)
class ReceiptRecord:
    interval_id: str
    speaker_id: str
    endpoint_confidence: float
    command_name: str | None
    command_code: int | None
    quant_scheme: str
    source_version: str
    missing_fields: tuple[str, ...] = ()
    storage: dict[str, Any] = field(default_factory=dict, compare=False)
    preserved_fields: tuple[str, ...] = ()
    restored_from: str | None = None

    def canonical_fields(self) -> dict[str, Any]:
        return {
            "interval_id": self.interval_id,
            "speaker_id": self.speaker_id,
            "endpoint_confidence": self.endpoint_confidence,
            "command_name": self.command_name,
            "command_code": self.command_code,
            "quant_scheme": self.quant_scheme,
            "source_version": self.source_version,
            "missing_fields": list(self.missing_fields),
        }

    def as_dict(self) -> dict[str, Any]:
        result = self.canonical_fields()
        result["storage"] = self.storage
        result["preserved_fields"] = list(self.preserved_fields)
        if self.restored_from is not None:
            result["restored_from"] = self.restored_from
        return result


@dataclass(frozen=True)
class ReceiptDocument:
    records: tuple[ReceiptRecord, ...]
    source_version: str
    sidecar: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    restored_from: str | None = None

    @property
    def by_interval(self) -> dict[str, ReceiptRecord]:
        return {record.interval_id: record for record in self.records}

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_version": self.source_version,
            "restored_from": self.restored_from,
            "warnings": list(self.warnings),
            "records": [record.as_dict() for record in self.records],
            "sidecar": self.sidecar,
        }


@dataclass(frozen=True)
class ReceiptConversion:
    version: str
    text: str
    sidecar: dict[str, Any]
    document: ReceiptDocument

    def write(self, evidence_path: str | Path, sidecar_path: str | Path | None = None) -> None:
        from .util import atomic_write_json, atomic_write_text

        target = Path(evidence_path)
        atomic_write_text(target, self.text)
        if sidecar_path is not None:
            atomic_write_json(Path(sidecar_path), self.sidecar)


class _Vocabulary:
    def __init__(self, value: Mapping[int, str] | Iterable[tuple[int, str]] | None) -> None:
        self.code_to_name: dict[int, str] = {}
        self.name_to_code: dict[str, int] = {}
        if value is None:
            return
        items = value.items() if isinstance(value, Mapping) else value
        for code, name in items:
            if isinstance(code, bool) or not isinstance(code, int):
                raise SchemaError("command vocabulary codes must be integers")
            if not isinstance(name, str) or not name:
                raise SchemaError("command vocabulary names must be non-empty strings")
            if code in self.code_to_name or name in self.name_to_code:
                raise SchemaError("command vocabulary contains duplicate code or name")
            self.code_to_name[code] = name
            self.name_to_code[name] = code

    def name_for(self, code: int) -> str | None:
        return self.code_to_name.get(code)

    def code_for(self, name: str) -> int | None:
        return self.name_to_code.get(name)


def _validate_id_field(raw: str, name: str, *, where: str) -> str:
    value = raw.rstrip(" ")
    if not value:
        raise FieldTruncatedError(f"a1 field {name} is empty", where=where)
    if any(char.isspace() for char in value) or not _ID_PATTERN.fullmatch(value):
        raise FieldTruncatedError(
            f"a1 field {name} contains illegal characters",
            where=where,
            details={"field": name, "value": raw},
        )
    return value


def _parse_numeric_field(raw: str, name: str, *, where: str) -> int:
    value = raw.lstrip(" ")
    if not value or not _DECIMAL_PATTERN.fullmatch(value):
        raise FieldTruncatedError(
            f"a1 field {name} is not an ASCII decimal integer",
            where=where,
            details={"field": name, "value": raw},
        )
    return int(value)


def _checked_confidence(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError("endpoint confidence must be numeric", where=where)
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise SchemaError("endpoint confidence must be finite and in 0..1", where=where)
    return confidence


def _checked_identifier(value: Any, name: str, *, where: str) -> str:
    if not isinstance(value, str) or not value or not _ID_PATTERN.fullmatch(value):
        raise FieldTruncatedError(
            f"{name} is not a serializable identifier",
            where=where,
            details={"field": name, "value": value},
        )
    return value


def _checked_command_code(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError("command code must be a positive integer", where=where)
    return value


def _sidecar_record(sidecar: Mapping[str, Any] | None, interval_id: str) -> Mapping[str, Any] | None:
    if not sidecar:
        return None
    records = sidecar.get("records")
    if not isinstance(records, dict):
        return None
    entry = records.get(interval_id)
    return entry if isinstance(entry, dict) else None


def _rendered_matches(rendered: Any, actual: Mapping[str, Any]) -> bool:
    if not isinstance(rendered, dict):
        return False
    for key, expected in rendered.items():
        if key not in actual:
            return False
        if isinstance(expected, float) or isinstance(actual[key], float):
            if not math.isclose(float(expected), float(actual[key]), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif expected != actual[key]:
            return False
    return True


def _restore_from_sidecar(
    canonical: dict[str, Any],
    storage: Mapping[str, Any],
    entry: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, ...], str | None, str | None]:
    if entry is None:
        return canonical, (), None, None
    rendered = entry.get("rendered")
    before = entry.get("before")
    if not _rendered_matches(rendered, storage) or not isinstance(before, dict):
        return canonical, (), None, "sidecar_rendered_values_mismatch"
    restored = dict(canonical)
    preserved: list[str] = []
    declared_overwrites = entry.get("overwritten_fields")
    declared_set = set(declared_overwrites) if isinstance(declared_overwrites, list) else set()
    for key in (
        "interval_id",
        "speaker_id",
        "endpoint_confidence",
        "command_name",
        "command_code",
        "quant_scheme",
    ):
        if key in before and (before[key] != restored.get(key) or key in declared_set):
            restored[key] = before[key]
            preserved.append(key)
    restored_from = entry.get("before_version")
    if not isinstance(restored_from, str):
        restored_from = None
    return restored, tuple(preserved), restored_from, None


def _build_record(
    *,
    canonical: dict[str, Any],
    storage: dict[str, Any],
    source_version: str,
    missing_fields: tuple[str, ...],
    entry: Mapping[str, Any] | None,
) -> tuple[ReceiptRecord, str | None]:
    restored, preserved, restored_from, warning = _restore_from_sidecar(canonical, storage, entry)
    warnings: list[str] = []
    if warning:
        warnings.append(f"{canonical.get('interval_id', '?')}:{warning}")
    confidence = _checked_confidence(restored["endpoint_confidence"], where="restored receipt")
    command_code = restored.get("command_code")
    if command_code is not None:
        command_code = _checked_command_code(command_code, where="restored receipt")
    quant_scheme = restored.get("quant_scheme")
    if not isinstance(quant_scheme, str) or not quant_scheme:
        raise SchemaError("restored receipt quant_scheme must be a non-empty string")
    record = ReceiptRecord(
        interval_id=str(restored["interval_id"]),
        speaker_id=str(restored["speaker_id"]),
        endpoint_confidence=confidence,
        command_name=restored.get("command_name"),
        command_code=command_code,
        quant_scheme=quant_scheme,
        source_version=source_version,
        missing_fields=tuple(missing_fields),
        storage=storage,
        preserved_fields=preserved,
        restored_from=restored_from,
    )
    return record, warning


def parse_a1(
    text: str,
    *,
    layout: A1Layout | None = None,
    vocabulary: Mapping[int, str] | Iterable[tuple[int, str]] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    quant_scheme: str = "permille",
) -> ReceiptDocument:
    layout = layout or A1Layout()
    vocab = _Vocabulary(vocabulary)
    records: list[ReceiptRecord] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line:
            continue
        where = f"a1:line:{line_number}"
        if len(line) != layout.line_width:
            raise FieldTruncatedError(
                "a1 line width is not the declared fixed width",
                where=where,
                details={"expected": layout.line_width, "actual": len(line)},
            )
        parts = line.split(layout.separator)
        if len(parts) != len(_REQUIRED_A1_FIELDS):
            raise FieldTruncatedError("a1 field count is not four", where=where)
        widths = layout.widths
        for name, value in zip(_REQUIRED_A1_FIELDS, parts):
            if len(value) != widths[name]:
                raise FieldTruncatedError(
                    "a1 field width does not match layout",
                    where=where,
                    details={"field": name, "expected": widths[name], "actual": len(value)},
                )
        interval_id = _validate_id_field(parts[0], "interval_id", where=where)
        speaker_id = _validate_id_field(parts[1], "speaker_id", where=where)
        confidence_permille = _parse_numeric_field(parts[2], "confidence", where=where)
        command_code = _parse_numeric_field(parts[3], "command_code", where=where)
        if interval_id in seen:
            raise SchemaError("duplicate interval_id in a1 receipt", where=where, details={"interval_id": interval_id})
        seen.add(interval_id)
        if not 0 <= confidence_permille <= 1000:
            raise SchemaError(
                "a1 confidence must be in permille range 0..1000",
                where=where,
                details={"confidence": confidence_permille},
            )
        if command_code <= 0:
            raise SchemaError("a1 command_code must be positive", where=where)
        command_name = vocab.name_for(command_code)
        if vocabulary is not None and command_name is None:
            raise SchemaError(
                "a1 command_code is absent from the declared command vocabulary",
                where=where,
                details={"command_code": command_code},
            )
        storage = {
            "interval_id": interval_id,
            "speaker_id": speaker_id,
            "confidence_permille": confidence_permille,
            "command_code": command_code,
        }
        canonical = {
            "interval_id": interval_id,
            "speaker_id": speaker_id,
            "endpoint_confidence": confidence_permille / 1000.0,
            "command_name": command_name,
            "command_code": command_code,
            "quant_scheme": quant_scheme,
        }
        record, warning = _build_record(
            canonical=canonical,
            storage=storage,
            source_version="a1",
            missing_fields=("command_name", "quant_scheme") if command_name is None else ("quant_scheme",),
            entry=_sidecar_record(sidecar, interval_id),
        )
        if warning:
            warnings.append(f"{interval_id}:{warning}")
        records.append(record)
    restored_from = sidecar.get("source_version") if isinstance(sidecar, dict) else None
    return ReceiptDocument(
        records=tuple(records),
        source_version="a1",
        sidecar=dict(sidecar or {}),
        warnings=tuple(warnings),
        restored_from=restored_from if isinstance(restored_from, str) else None,
    )


def _a2_payload(value: Any) -> tuple[Any, str | None]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid a2 JSON: {exc}") from exc
    if isinstance(value, list):
        return value, None
    if not isinstance(value, dict):
        raise SchemaError("a2 payload must be a JSON object or array")
    version = value.get("version")
    if version is not None and version != "a2":
        raise SchemaError("a2 version field must be 'a2'")
    records = value.get("records")
    if not isinstance(records, list):
        raise SchemaError("a2 records must be an array")
    default_scheme = value.get("quant_scheme")
    if default_scheme is not None and (not isinstance(default_scheme, str) or not default_scheme):
        raise SchemaError("a2 default quant_scheme must be a non-empty string")
    return records, default_scheme


def parse_a2(
    value: str | Mapping[str, Any] | list[Any],
    *,
    vocabulary: Mapping[int, str] | Iterable[tuple[int, str]] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    allowed_quant_schemes: Iterable[str] | None = None,
) -> ReceiptDocument:
    records_raw, default_scheme = _a2_payload(value)
    vocab = _Vocabulary(vocabulary)
    allowed = set(allowed_quant_schemes) if allowed_quant_schemes is not None else None
    records: list[ReceiptRecord] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(records_raw):
        where = f"a2:record:{index}"
        if not isinstance(raw, dict):
            raise SchemaError("a2 record must be an object", where=where)
        interval_id = raw.get("interval_id")
        speaker_id = raw.get("speaker_id")
        confidence = raw.get("endpoint_confidence")
        command_name = raw.get("command")
        scheme = raw.get("quant_scheme", default_scheme)
        if not isinstance(interval_id, str) or not _ID_PATTERN.fullmatch(interval_id):
            raise SchemaError("a2 interval_id must be a valid identifier", where=where)
        if not isinstance(speaker_id, str) or not _ID_PATTERN.fullmatch(speaker_id):
            raise SchemaError("a2 speaker_id must be a valid identifier", where=where)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise SchemaError("a2 endpoint_confidence must be numeric", where=where)
        confidence_float = float(confidence)
        if not math.isfinite(confidence_float) or not 0.0 <= confidence_float <= 1.0:
            raise SchemaError("a2 endpoint_confidence must be finite and in 0..1", where=where)
        if not isinstance(command_name, str) or not command_name:
            raise SchemaError("a2 command must be a non-empty string", where=where)
        if not isinstance(scheme, str) or not scheme:
            raise SchemaError("a2 quant_scheme must be a named string", where=where)
        if allowed is not None and scheme not in allowed:
            raise SchemaError(
                "a2 quant_scheme is not in the allowed enumeration",
                where=where,
                details={"quant_scheme": scheme, "allowed": sorted(allowed)},
            )
        if interval_id in seen:
            raise SchemaError("duplicate interval_id in a2 receipt", where=where)
        seen.add(interval_id)
        command_code: int | None = None
        if vocab.code_to_name:
            command_code = vocab.code_for(command_name)
            if command_code is None:
                raise SchemaError(
                    "a2 command is absent from the declared command vocabulary",
                    where=where,
                    details={"command": command_name},
                )
        else:
            supplied_code = raw.get("command_code")
            if isinstance(supplied_code, int) and not isinstance(supplied_code, bool):
                command_code = supplied_code
        missing = () if command_code is not None else ("command_code",)
        storage = {
            "interval_id": interval_id,
            "speaker_id": speaker_id,
            "endpoint_confidence": confidence_float,
            "command": command_name,
            "quant_scheme": scheme,
        }
        canonical = {
            "interval_id": interval_id,
            "speaker_id": speaker_id,
            "endpoint_confidence": confidence_float,
            "command_name": command_name,
            "command_code": command_code,
            "quant_scheme": scheme,
        }
        record, warning = _build_record(
            canonical=canonical,
            storage=storage,
            source_version="a2",
            missing_fields=missing,
            entry=_sidecar_record(sidecar, interval_id),
        )
        if warning:
            warnings.append(f"{interval_id}:{warning}")
        records.append(record)
    restored_from = sidecar.get("source_version") if isinstance(sidecar, dict) else None
    return ReceiptDocument(
        records=tuple(records),
        source_version="a2",
        sidecar=dict(sidecar or {}),
        warnings=tuple(warnings),
        restored_from=restored_from if isinstance(restored_from, str) else None,
    )


def _resolve_command(
    record: ReceiptRecord,
    vocab: _Vocabulary,
) -> tuple[int, str | None]:
    code = record.command_code
    name = record.command_name
    if code is None and name is not None:
        code = vocab.code_for(name)
        if code is None and vocab.code_to_name:
            raise SchemaError(
                "cannot serialize receipt with a command name outside the vocabulary",
                details={"interval_id": record.interval_id, "command_name": name},
            )
    elif code is not None:
        code = _checked_command_code(code, where=f"receipt:{record.interval_id}")
        vocabulary_name = vocab.name_for(code)
        if vocabulary_name is not None:
            if name is not None and name != vocabulary_name:
                raise SchemaError(
                    "receipt command name and integer code disagree with the vocabulary",
                    details={
                        "interval_id": record.interval_id,
                        "command_name": name,
                        "command_code": code,
                        "vocabulary_name": vocabulary_name,
                    },
                )
            name = vocabulary_name
        elif vocab.code_to_name:
            raise SchemaError(
                "cannot serialize receipt with a command code outside the vocabulary",
                details={"interval_id": record.interval_id, "command_code": code},
            )
    if name is None and code is not None:
        name = vocab.name_for(code)
    if code is None:
        raise SchemaError(
            "cannot serialize receipt without a command code",
            details={"interval_id": record.interval_id},
        )
    return _checked_command_code(code, where=f"receipt:{record.interval_id}"), name


def _before_fields(record: ReceiptRecord) -> dict[str, Any]:
    return {
        "interval_id": record.interval_id,
        "speaker_id": record.speaker_id,
        "endpoint_confidence": record.endpoint_confidence,
        "command_name": record.command_name,
        "command_code": record.command_code,
        "quant_scheme": record.quant_scheme,
    }


def to_a1(
    document: ReceiptDocument,
    *,
    layout: A1Layout | None = None,
    vocabulary: Mapping[int, str] | Iterable[tuple[int, str]] | None = None,
) -> ReceiptConversion:
    layout = layout or A1Layout()
    vocab = _Vocabulary(vocabulary)
    lines: list[str] = []
    sidecar_records: dict[str, Any] = {}
    widths = layout.widths
    seen_interval_ids: set[str] = set()
    for record in document.records:
        interval_id = _checked_identifier(record.interval_id, "interval_id", where="to_a1")
        speaker_id = _checked_identifier(record.speaker_id, "speaker_id", where="to_a1")
        if interval_id in seen_interval_ids:
            raise SchemaError("duplicate interval_id in receipt document", details={"interval_id": interval_id})
        seen_interval_ids.add(interval_id)
        confidence = _checked_confidence(record.endpoint_confidence, where=f"to_a1:{interval_id}")
        command_code, command_name = _resolve_command(record, vocab)
        confidence_permille = round_half_away_from_zero(confidence * 1000.0)
        if not 0 <= confidence_permille <= 1000:
            raise SchemaError("endpoint confidence cannot be represented in permille", details={"interval_id": interval_id})
        try:
            interval_field = f"{interval_id:<{widths['interval_id']}}"
            speaker_field = f"{speaker_id:<{widths['speaker_id']}}"
            confidence_field = f"{confidence_permille:0{widths['confidence']}d}"
            command_field = f"{command_code:0{widths['command_code']}d}"
        except (ValueError, TypeError) as exc:
            raise FieldTruncatedError("value cannot be formatted into a1 widths") from exc
        fields = (
            ("interval_id", interval_id, interval_field),
            ("speaker_id", speaker_id, speaker_field),
            ("confidence", str(confidence_permille), confidence_field),
            ("command_code", str(command_code), command_field),
        )
        for name, value, rendered in fields:
            if len(rendered) > widths[name]:
                raise FieldTruncatedError(
                    "a1 value exceeds the declared field width",
                    details={
                        "interval_id": record.interval_id,
                        "field": name,
                        "width": widths[name],
                        "value": value,
                    },
                )
        line = layout.separator.join((interval_field, speaker_field, confidence_field, command_field))
        if len(line) != layout.line_width:
            raise FieldTruncatedError(
                "serialized a1 line disagrees with declared layout",
                details={"expected": layout.line_width, "actual": len(line)},
            )
        lines.append(line)
        rendered_storage = {
            "interval_id": interval_id,
            "speaker_id": speaker_id,
            "confidence_permille": confidence_permille,
            "command_code": command_code,
        }
        before = _before_fields(record)
        residual = confidence - confidence_permille / 1000.0
        overwritten: list[str] = []
        if confidence != confidence_permille / 1000.0:
            overwritten.append("endpoint_confidence")
        if record.quant_scheme != "permille":
            overwritten.append("quant_scheme")
        if record.command_name is not None:
            # a1 carries the integer code only; the named command remains in
            # the sidecar so a later up-conversion can restore it exactly.
            overwritten.append("command_name")
        sidecar_records[interval_id] = {
            "rendered_version": "a1",
            "before_version": record.source_version,
            "rendered": rendered_storage,
            "before": before,
            "residual": residual,
            "precision_label": "float_to_permille_round_half_away",
            "overwritten_fields": overwritten,
            "missing_fields": list(record.missing_fields),
        }
    text = "\n".join(lines)
    if text:
        text += "\n"
    sidecar = {
        "schema": 1,
        "source_version": document.source_version,
        "rendered_version": "a1",
        "precision_label": "permille",
        "records": sidecar_records,
    }
    canonical_records = tuple(
        ReceiptRecord(
            interval_id=_checked_identifier(record.interval_id, "interval_id", where="to_a1"),
            speaker_id=_checked_identifier(record.speaker_id, "speaker_id", where="to_a1"),
            endpoint_confidence=sidecar_records[record.interval_id]["rendered"]["confidence_permille"] / 1000.0,
            command_name=vocab.name_for(sidecar_records[record.interval_id]["rendered"]["command_code"]),
            command_code=sidecar_records[record.interval_id]["rendered"]["command_code"],
            quant_scheme="permille",
            source_version="a1",
            missing_fields=("command_name", "quant_scheme"),
            storage=dict(sidecar_records[record.interval_id]["rendered"]),
        )
        for record in document.records
    )
    rendered_doc = ReceiptDocument(canonical_records, "a1", sidecar)
    return ReceiptConversion("a1", text, sidecar, rendered_doc)


def to_a2(
    document: ReceiptDocument,
    *,
    vocabulary: Mapping[int, str] | Iterable[tuple[int, str]] | None = None,
) -> ReceiptConversion:
    vocab = _Vocabulary(vocabulary)
    payload_records: list[dict[str, Any]] = []
    sidecar_records: dict[str, Any] = {}
    resolved_codes: list[int] = []
    seen_interval_ids: set[str] = set()
    for record in document.records:
        interval_id = _checked_identifier(record.interval_id, "interval_id", where="to_a2")
        speaker_id = _checked_identifier(record.speaker_id, "speaker_id", where="to_a2")
        if interval_id in seen_interval_ids:
            raise SchemaError("duplicate interval_id in receipt document", details={"interval_id": interval_id})
        seen_interval_ids.add(interval_id)
        confidence = _checked_confidence(record.endpoint_confidence, where=f"to_a2:{interval_id}")
        if not isinstance(record.quant_scheme, str) or not record.quant_scheme:
            raise SchemaError(
                "a2 quant_scheme must be a non-empty named string",
                details={"interval_id": interval_id},
            )
        command_code, command_name = _resolve_command(record, vocab)
        if command_name is None:
            raise SchemaError(
                "cannot serialize a2 without a named command",
                details={"interval_id": interval_id},
            )
        resolved_codes.append(command_code)
        rendered = {
            "interval_id": interval_id,
            "speaker_id": speaker_id,
            "endpoint_confidence": confidence,
            "command": command_name,
            "quant_scheme": record.quant_scheme,
        }
        payload_records.append(rendered)
        before = _before_fields(record)
        overwritten: list[str] = []
        if record.command_code is not None:
            # a2's public schema names commands; the integer code is retained in
            # the sidecar as an overwritten/omitted field.
            overwritten.append("command_code")
        if record.command_name is not None and record.command_name != command_name:
            overwritten.append("command_name")
        if record.quant_scheme != rendered["quant_scheme"]:
            overwritten.append("quant_scheme")
        sidecar_records[interval_id] = {
            "rendered_version": "a2",
            "before_version": record.source_version,
            "rendered": rendered,
            "before": before,
            "residual": 0.0,
            "precision_label": "float_preserved_json",
            "overwritten_fields": overwritten,
            "missing_fields": list(record.missing_fields),
        }
    payload = {
        "version": "a2",
        "records": payload_records,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    sidecar = {
        "schema": 1,
        "source_version": document.source_version,
        "rendered_version": "a2",
        "precision_label": "float64_json",
        "records": sidecar_records,
    }
    canonical_records = tuple(
        ReceiptRecord(
            interval_id=item["interval_id"],
            speaker_id=item["speaker_id"],
            endpoint_confidence=float(item["endpoint_confidence"]),
            command_name=item["command"],
            command_code=resolved_codes[index],
            quant_scheme=item["quant_scheme"],
            source_version="a2",
            missing_fields=(),
            storage=dict(item),
        )
        for index, item in enumerate(payload_records)
    )
    rendered_doc = ReceiptDocument(canonical_records, "a2", sidecar)
    return ReceiptConversion("a2", text, sidecar, rendered_doc)


def compare_normalized_receipts(
    a1_document: ReceiptDocument,
    a2_document: ReceiptDocument,
    *,
    confidence_tolerance: float = 0.0005,
) -> dict[str, Any]:
    """Compare two parsed versions after normalization.

    The a1 round trip necessarily carries at most permille precision, so the
    default tolerance is half a permille unit.  Mismatches are returned rather
    than silently choosing one version.
    """

    left = a1_document.by_interval
    right = a2_document.by_interval
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    mismatches: list[dict[str, Any]] = []
    for interval_id in sorted(set(left) & set(right)):
        one = left[interval_id]
        two = right[interval_id]
        reasons: list[str] = []
        if one.speaker_id != two.speaker_id:
            reasons.append("speaker_id")
        if abs(one.endpoint_confidence - two.endpoint_confidence) > confidence_tolerance + 1e-12:
            reasons.append("endpoint_confidence")
        if one.command_code is not None and two.command_code is not None and one.command_code != two.command_code:
            reasons.append("command_code")
        if reasons:
            mismatches.append(
                {
                    "interval_id": interval_id,
                    "fields": reasons,
                    "a1": one.canonical_fields(),
                    "a2": two.canonical_fields(),
                }
            )
    digest_payload = {
        "a1": [record.canonical_fields() for record in sorted(a1_document.records, key=lambda item: item.interval_id)],
        "a2": [record.canonical_fields() for record in sorted(a2_document.records, key=lambda item: item.interval_id)],
    }
    return {
        "ok": not missing_left and not missing_right and not mismatches,
        "missing_in_a1": missing_left,
        "missing_in_a2": missing_right,
        "mismatches": mismatches,
        "normalized_digest": json_hash(digest_payload),
    }


def document_digest(document: ReceiptDocument) -> str:
    return json_hash([record.canonical_fields() for record in document.records])


def load_receipt_a1(path: str | Path, **kwargs: Any) -> ReceiptDocument:
    receipt_path = Path(path)
    try:
        text = receipt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SchemaError("cannot read a1 receipt", where=str(receipt_path)) from exc
    return parse_a1(text, **kwargs)


def load_receipt_a2(path: str | Path, **kwargs: Any) -> ReceiptDocument:
    receipt_path = Path(path)
    try:
        text = receipt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SchemaError("cannot read a2 receipt", where=str(receipt_path)) from exc
    return parse_a2(text, **kwargs)
