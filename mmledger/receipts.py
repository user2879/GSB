"""Loading and validation for coexisting r1/r2/r3 receipt files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import (
    E_DUPLICATE_BATCH_ID,
    E_DURATION_NONFINITE,
    E_RECEIPT_ENCODING,
    E_RECEIPT_FIELD_TYPE,
    E_RECEIPT_JSON,
    E_RECEIPT_LAYOUT,
    E_RECEIPT_VERSION,
    E_TRUNCATED_RECEIPT,
    E_UNKNOWN_PARAMETER_VERSION,
    Failure,
    sort_failures,
)
from .jsonutil import MISSING, decode_missing, hash_file, is_missing
from .models import BASE_FIELDS, R1, R2, R3, VERSION_FIELDS, VERSIONS, Receipt


VERSION_KEYS = ("version", "schema_version", "receipt_version")
ENVELOPE_KEYS = {"version", "schema_version", "receipt_version", "receipts", "notes"}


@dataclass(frozen=True)
class ParseResult:
    receipts: tuple[Receipt, ...]
    failures: tuple[Failure, ...]
    file_fingerprints: Mapping[str, str]
    file_versions: Mapping[str, str]
    record_counts: Mapping[str, int]
    injection_hits: tuple[Mapping[str, Any], ...]

    @property
    def ok_count(self) -> int:
        return len(self.receipts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipts": [receipt.as_ledger_dict() for receipt in self.receipts],
            "failures": [failure.to_dict() for failure in self.failures],
            "file_fingerprints": dict(sorted(self.file_fingerprints.items())),
            "file_versions": dict(sorted(self.file_versions.items())),
            "record_counts": dict(sorted(self.record_counts.items())),
            "injection_hits": [dict(item) for item in self.injection_hits],
            "ok_count": self.ok_count,
        }


def normalize_version(value: Any) -> str | None:
    if value in (R1, R2, R3):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool) and value in (1, 2, 3):
        return f"r{value}"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in VERSIONS:
            return lowered
        if lowered in {"1", "2", "3"}:
            return f"r{lowered}"
        if lowered.startswith("receipt-") and lowered[8:] in VERSIONS:
            return lowered[8:]
    return None


def _version_from_mapping(value: Mapping[str, Any]) -> str | None:
    for key in VERSION_KEYS:
        if key in value:
            version = normalize_version(value[key])
            if version is not None:
                return version
    return None


def _strip_version_keys(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in VERSION_KEYS}


def _record_injection_id(record: Mapping[str, Any], file_injection: str | None) -> str | None:
    value = record.get("injection_id")
    if isinstance(value, str) and value:
        return value
    return file_injection


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_record(
    raw: Mapping[str, Any],
    version: str,
    *,
    source: str,
    ordinal: int,
    uid: str,
    injection_id: str | None,
    known_versions: frozenset[int],
) -> tuple[Receipt | None, Failure | None]:
    body = _strip_version_keys(raw)
    body.pop("injection_id", None)
    expected = VERSION_FIELDS[version]
    expected_set = set(expected)
    extra = sorted(set(body) - expected_set)
    missing_keys = sorted(expected_set - set(body))
    if extra or missing_keys:
        details: dict[str, Any] = {}
        if missing_keys:
            details["missing_keys"] = missing_keys
        if extra:
            details["unexpected_keys"] = extra
        return None, Failure(
            E_RECEIPT_LAYOUT,
            f"{version} receipt field layout is inconsistent",
            source,
            ordinal,
            item=uid,
            injection_id=injection_id,
            details=details,
        )

    values: dict[str, Any] = {name: decode_missing(body[name]) for name in expected}
    for name in BASE_FIELDS:
        if is_missing(values[name]) or not isinstance(values[name], str) or not values[name].strip():
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                f"{name} must be a non-empty string",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )

    duration = values.get("duration_ms", MISSING)
    if isinstance(duration, float) and not math.isfinite(duration):
        return None, Failure(
            E_DURATION_NONFINITE,
            "duration_ms is not finite",
            source,
            ordinal,
            item=uid,
            injection_id=injection_id,
            details={"value": str(duration)},
        )

    parameter_version = values.get("parameter_version", MISSING)
    if not is_missing(parameter_version):
        if not (_is_int(parameter_version) and parameter_version > 0):
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                "parameter_version must be a positive integer",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )
        if parameter_version not in known_versions:
            return None, Failure(
                E_UNKNOWN_PARAMETER_VERSION,
                f"unknown parameter_version {parameter_version}",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
                details={"parameter_version": parameter_version},
            )

    if "duration_ms" in values and not is_missing(values["duration_ms"]):
        if not (_is_int(values["duration_ms"]) and values["duration_ms"] >= 0):
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                "duration_ms must be a non-negative integer",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )

    if "worker_id" in values and not is_missing(values["worker_id"]):
        if not isinstance(values["worker_id"], str) or not values["worker_id"].strip():
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                "worker_id must be a non-empty string",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )

    if "stable_rank" in values and not is_missing(values["stable_rank"]):
        if not (_is_int(values["stable_rank"]) and values["stable_rank"] > 0):
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                "stable_rank must be a positive integer",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )

    if "overwritten_ref" in values and not is_missing(values["overwritten_ref"]):
        reference = values["overwritten_ref"]
        if reference is not None and (not isinstance(reference, str) or not reference.strip()):
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                "overwritten_ref must be null or a non-empty string",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )

    if "parameter_digest" in values and not is_missing(values["parameter_digest"]):
        digest = values["parameter_digest"]
        if not isinstance(digest, str) or not digest.strip():
            return None, Failure(
                E_RECEIPT_FIELD_TYPE,
                "parameter_digest must be a non-empty string",
                source,
                ordinal,
                item=uid,
                injection_id=injection_id,
            )

    receipt = Receipt(
        batch_id=str(values["batch_id"]),
        request_id=str(values["request_id"]),
        output_digest=str(values["output_digest"]),
        parameter_version=values.get("parameter_version", MISSING),
        duration_ms=values.get("duration_ms", MISSING),
        worker_id=values.get("worker_id", MISSING),
        stable_rank=values.get("stable_rank", MISSING),
        overwritten_ref=values.get("overwritten_ref", MISSING),
        parameter_digest=values.get("parameter_digest", MISSING),
        format_version=version,
        source_version=version,
        uid=uid,
        source_file=source,
        source_index=ordinal,
        provenance={"loaded_from": source},
    )
    return receipt, None


def _file_records(
    raw: Any,
    *,
    source: str,
    file_injection: str | None,
    known_versions: frozenset[int],
) -> tuple[list[Receipt], list[Failure], str | None, int]:
    receipts: list[Receipt] = []
    failures: list[Failure] = []
    file_version: str | None = None

    if isinstance(raw, dict) and "receipts" in raw:
        envelope_extra = sorted(set(raw) - ENVELOPE_KEYS)
        if envelope_extra:
            failures.append(
                Failure(
                    E_RECEIPT_LAYOUT,
                    "receipt envelope contains unexpected keys",
                    source,
                    injection_id=file_injection,
                    details={"unexpected_keys": envelope_extra},
                )
            )
            return [], failures, None, 0
        file_version = _version_from_mapping(raw)
        records_value = raw.get("receipts")
        if file_version is None:
            failures.append(
                Failure(
                    E_RECEIPT_VERSION,
                    "receipt envelope has no valid version marker",
                    source,
                    injection_id=file_injection,
                )
            )
            return [], failures, None, 0
        if not isinstance(records_value, list):
            failures.append(
                Failure(
                    E_RECEIPT_LAYOUT,
                    "receipt envelope receipts must be a list",
                    source,
                    injection_id=file_injection,
                )
            )
            return [], failures, file_version, 0
        records = records_value
    elif isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = [raw]
    else:
        failures.append(
            Failure(
                E_RECEIPT_LAYOUT,
                "receipt file must contain an object or array",
                source,
                injection_id=file_injection,
            )
        )
        return [], failures, None, 0

    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(
                Failure(
                    E_RECEIPT_LAYOUT,
                    "receipt record must be an object",
                    source,
                    ordinal,
                    injection_id=file_injection,
                )
            )
            continue
        record_version = _version_from_mapping(record)
        if file_version is None:
            if record_version is None:
                failures.append(
                    Failure(
                        E_RECEIPT_VERSION,
                        "receipt has no valid version marker",
                        source,
                        ordinal,
                        injection_id=_record_injection_id(record, file_injection),
                    )
                )
                continue
            if ordinal == 0:
                file_version = record_version
        elif record_version is not None and record_version != file_version:
            failures.append(
                Failure(
                    E_RECEIPT_VERSION,
                    f"record version {record_version} does not match file version {file_version}",
                    source,
                    ordinal,
                    injection_id=_record_injection_id(record, file_injection),
                )
            )
            continue
        version = record_version or file_version
        if version is None:
            failures.append(
                Failure(
                    E_RECEIPT_VERSION,
                    "receipt has no valid version marker",
                    source,
                    ordinal,
                    injection_id=_record_injection_id(record, file_injection),
                )
            )
            continue
        if file_version is not None and version != file_version:
            failures.append(
                Failure(
                    E_RECEIPT_VERSION,
                    f"mixed receipt versions in one file: {file_version} and {version}",
                    source,
                    ordinal,
                    injection_id=_record_injection_id(record, file_injection),
                )
            )
            continue
        uid = f"{source}#{ordinal}"
        receipt, failure = _validate_record(
            record,
            version,
            source=source,
            ordinal=ordinal,
            uid=uid,
            injection_id=_record_injection_id(record, file_injection),
            known_versions=known_versions,
        )
        if failure is not None:
            failures.append(failure)
        elif receipt is not None:
            receipts.append(receipt)
    return receipts, failures, file_version, len(records)


def parse_receipt_paths(
    paths: Iterable[Path],
    *,
    base_dir: Path,
    known_versions: frozenset[int],
    injections: Mapping[str, str] | None = None,
) -> ParseResult:
    injection_lookup = dict(injections or {})
    receipts: list[Receipt] = []
    failures: list[Failure] = []
    fingerprints: dict[str, str] = {}
    versions: dict[str, str] = {}
    counts: dict[str, int] = {}
    hits: list[dict[str, Any]] = []
    seen_identity: dict[tuple[Any, ...], str] = {}

    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            relative = path.as_posix()
        injection_id = injection_lookup.get(relative)
        try:
            payload = path.read_bytes()
            fingerprints[relative] = hash_file(path)
        except OSError as exc:
            failures.append(
                Failure(E_RECEIPT_JSON, f"cannot read receipt file: {exc}", relative, injection_id=injection_id)
            )
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(
                Failure(E_RECEIPT_ENCODING, f"receipt is not UTF-8: {exc}", relative, injection_id=injection_id)
            )
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            truncated = exc.pos >= max(0, len(text.rstrip()) - 1) or exc.msg.startswith("Unterminated string")
            code = E_TRUNCATED_RECEIPT if truncated else E_RECEIPT_JSON
            failures.append(
                Failure(
                    code,
                    f"invalid JSON receipt: {exc.msg}",
                    relative,
                    ordinal=int(exc.lineno),
                    injection_id=injection_id,
                    details={"line": exc.lineno, "column": exc.colno, "position": exc.pos},
                )
            )
            continue

        file_receipts, file_failures, file_version, count = _file_records(
            raw,
            source=relative,
            file_injection=injection_id,
            known_versions=known_versions,
        )
        failures.extend(file_failures)
        counts[relative] = count
        if file_version is not None:
            versions[relative] = file_version
        for receipt in file_receipts:
            identity = (
                receipt.batch_id,
                receipt.request_id,
                "-" if is_missing(receipt.worker_id) else receipt.worker_id,
                "-" if is_missing(receipt.stable_rank) else receipt.stable_rank,
            )
            if identity in seen_identity:
                failures.append(
                    Failure(
                        E_DUPLICATE_BATCH_ID,
                        f"duplicate delivery identity for batch {receipt.batch_id}",
                        receipt.source_file,
                        receipt.source_index,
                        item=receipt.uid,
                        injection_id=_record_injection_id(raw if isinstance(raw, dict) else {}, injection_id),
                        details={"first_uid": seen_identity[identity]},
                    )
                )
                continue
            seen_identity[identity] = receipt.uid
            receipts.append(receipt)

    ordered_failures = tuple(sort_failures(failures))
    hit_map: dict[str, dict[str, Any]] = {}
    for failure in ordered_failures:
        if failure.injection_id:
            hit_map[failure.injection_id] = {
                "injection_id": failure.injection_id,
                "hit": True,
                "code": failure.code,
                "source": failure.source,
                "ordinal": failure.ordinal,
            }
    for injection_id in sorted(set(injection_lookup.values())):
        hit_map.setdefault(
            injection_id,
            {"injection_id": injection_id, "hit": False, "code": None, "source": None, "ordinal": None},
        )
    hits = [hit_map[key] for key in sorted(hit_map)]
    receipts.sort(key=lambda receipt: receipt.log_key)
    return ParseResult(
        receipts=tuple(receipts),
        failures=ordered_failures,
        file_fingerprints=dict(sorted(fingerprints.items())),
        file_versions=dict(sorted(versions.items())),
        record_counts=dict(sorted(counts.items())),
        injection_hits=tuple(hits),
    )
