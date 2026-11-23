"""JSONL text library loading and text-record validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import sha256_json
from .config import FingerprintConfig
from .errors import E_TEXT_INVALID, Failure
from .normalization_rules import normalize_token


@dataclass(frozen=True)
class TextRecord:
    text_id: str
    terms: tuple[str, ...]
    source: str
    version: int

    def payload(self) -> dict[str, Any]:
        return {
            "text_id": self.text_id,
            "terms": list(self.terms),
            "source": self.source,
            "version": self.version,
        }

    @property
    def content_fingerprint(self) -> str:
        return sha256_json(self.payload())


def _text_failure(message: str, *, source: str, line: int | None, item_id: str | None = None) -> Failure:
    return Failure(
        E_TEXT_INVALID,
        "text_validation",
        message,
        item_id=item_id,
        source=source,
        line=line,
    )


def validate_text_payload(
    payload: Any,
    config: FingerprintConfig,
    *,
    source: str,
    line: int | None = None,
) -> tuple[TextRecord | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(payload, dict):
        return None, [_text_failure("text record must be an object", source=source, line=line)]

    text_id = payload.get("text_id")
    item_id = text_id if isinstance(text_id, str) and text_id else None
    if not isinstance(text_id, str) or not text_id.strip():
        failures.append(_text_failure("text_id must be a non-empty string", source=source, line=line))

    terms_raw = payload.get("terms")
    terms: list[str] | None = None
    if not isinstance(terms_raw, list) or not terms_raw:
        failures.append(_text_failure("terms must be a non-empty list", source=source, line=line, item_id=item_id))
    elif not all(isinstance(term, str) and term.strip() for term in terms_raw):
        failures.append(
            _text_failure("every term must be a non-empty string", source=source, line=line, item_id=item_id)
        )
    else:
        normalized_terms = [normalize_token(term) for term in terms_raw]
        vocabulary = config.normalized_vocabulary
        unknown = sorted({term for term in normalized_terms if term not in vocabulary})
        if any(not term for term in normalized_terms):
            failures.append(
                _text_failure("a term normalizes to an empty token", source=source, line=line, item_id=item_id)
            )
        elif unknown:
            failures.append(
                _text_failure(
                    "terms fall outside the controlled vocabulary",
                    source=source,
                    line=line,
                    item_id=item_id,
                )
            )
        else:
            terms = list(terms_raw)

    source_label = payload.get("source")
    if not isinstance(source_label, str) or source_label not in config.source_labels:
        failures.append(
            _text_failure("source is not in the configured enumeration", source=source, line=line, item_id=item_id)
        )

    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        failures.append(
            _text_failure("version must be a non-negative integer", source=source, line=line, item_id=item_id)
        )

    if failures:
        return None, failures
    return TextRecord(text_id=text_id, terms=tuple(terms or ()), source=source_label, version=version), []


def load_text_jsonl(
    path: str,
    config: FingerprintConfig,
) -> tuple[list[TextRecord], list[Failure]]:
    records: list[TextRecord] = []
    failures: list[Failure] = []
    seen_ids: set[str] = set()
    last_version: int | None = None
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except Exception as exc:
                failures.append(_text_failure(f"invalid JSON: {exc}", source=path, line=line_number))
                continue
            raw_id = payload.get("text_id") if isinstance(payload, dict) else None
            raw_version = payload.get("version") if isinstance(payload, dict) else None
            version_failure: Failure | None = None
            if isinstance(raw_version, int) and not isinstance(raw_version, bool) and raw_version >= 0:
                if last_version is not None and raw_version < last_version:
                    version_failure = _text_failure(
                        "version markers must be monotonically non-decreasing",
                        source=path,
                        line=line_number,
                        item_id=raw_id if isinstance(raw_id, str) else None,
                    )
                last_version = raw_version if last_version is None else max(last_version, raw_version)
            if isinstance(raw_id, str):
                if raw_id in seen_ids:
                    failures.append(
                        _text_failure("duplicate text_id", source=path, line=line_number, item_id=raw_id)
                    )
                    if version_failure is not None:
                        failures.append(version_failure)
                    continue
                seen_ids.add(raw_id)
            record, record_failures = validate_text_payload(payload, config, source=path, line=line_number)
            if version_failure is not None:
                record_failures = [*record_failures, version_failure]
                record = None
            if record is not None:
                records.append(record)
            failures.extend(record_failures)
    return records, failures


def records_by_id(records: Iterable[TextRecord]) -> dict[str, TextRecord]:
    return {record.text_id: record for record in records}


def validate_record_collection(
    payloads: Iterable[dict[str, Any]],
    config: FingerprintConfig,
    *,
    source: str,
) -> tuple[list[TextRecord], list[Failure]]:
    records: list[TextRecord] = []
    failures: list[Failure] = []
    seen: set[str] = set()
    last_version: int | None = None
    for index, payload in enumerate(payloads):
        raw_id = payload.get("text_id") if isinstance(payload, dict) else None
        raw_version = payload.get("version") if isinstance(payload, dict) else None
        version_failure: Failure | None = None
        if isinstance(raw_version, int) and not isinstance(raw_version, bool) and raw_version >= 0:
            if last_version is not None and raw_version < last_version:
                version_failure = _text_failure(
                    "version markers must be monotonically non-decreasing",
                    source=source,
                    line=index + 1,
                    item_id=raw_id if isinstance(raw_id, str) else None,
                )
            last_version = raw_version if last_version is None else max(last_version, raw_version)
        if isinstance(raw_id, str):
            if raw_id in seen:
                failures.append(_text_failure("duplicate text_id", source=source, line=index + 1, item_id=raw_id))
                if version_failure is not None:
                    failures.append(version_failure)
                continue
            seen.add(raw_id)
        record, record_failures = validate_text_payload(payload, config, source=source, line=index + 1)
        if version_failure is not None:
            record_failures = [*record_failures, version_failure]
            record = None
        if record is not None:
            records.append(record)
        failures.extend(record_failures)
    return records, failures
