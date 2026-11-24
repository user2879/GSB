from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import SAMPLE_INVALID, Failure
from .hashing import fingerprint


@dataclass(frozen=True)
class Sample:
    seq: int
    batch: int
    values: tuple[Decimal | None, ...]
    missing: tuple[bool, ...]
    line_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "batch": self.batch,
            "values": [None if v is None else format(v, "f") for v in self.values],
            "missing": list(self.missing),
        }


def load_and_validate_samples(
    path: str | Path,
    channel_count: int,
    injection_id: str | None = None,
    channel_names: list[str] | None = None,
) -> tuple[list[Sample], list[Failure], str]:
    path = Path(path)
    failures: list[Failure] = []
    samples: list[Sample] = []
    seen_seq: set[int] = set()
    last_batch = -1

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return [], [
            Failure(
                SAMPLE_INVALID,
                f"cannot read samples file: {exc}",
                path=str(path),
                injection_id=injection_id,
            )
        ], ""

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                parse_constant=_reject_constant,
                parse_float=Decimal,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(
                Failure(
                    SAMPLE_INVALID,
                    f"line {line_number}: invalid JSON: {exc}",
                    seq=_extract_seq_safely(line),
                    path=f"{path}:{line_number}",
                    injection_id=injection_id,
                )
            )
            continue

        row_errors = validate_sample_row(
            row, line_number, channel_count, seen_seq, last_batch, channel_names
        )
        for message, seq, batch in row_errors:
            failures.append(
                Failure(
                    SAMPLE_INVALID,
                    message,
                    batch=batch,
                    seq=seq,
                    path=f"{path}:{line_number}",
                    injection_id=injection_id,
                )
            )

        if row_errors:
            seq = row.get("seq") if isinstance(row, dict) else None
            batch = row.get("batch") if isinstance(row, dict) else None
            if isinstance(seq, int) and not isinstance(seq, bool):
                seen_seq.add(seq)
            if isinstance(batch, int) and not isinstance(batch, bool) and batch >= 0:
                last_batch = max(last_batch, batch)
            continue

        seq = int(row["seq"])
        batch = int(row["batch"])
        values, missing = _normalize_readings(row, channel_count, channel_names)
        samples.append(
            Sample(
                seq=seq,
                batch=batch,
                values=tuple(values),
                missing=tuple(missing),
                line_number=line_number,
            )
        )
        seen_seq.add(seq)
        last_batch = batch

    samples.sort(key=lambda s: s.seq)
    if not samples and not failures:
        failures.append(Failure(SAMPLE_INVALID, "sample file contains no rows", path=str(path)))
    data_fingerprint = fingerprint([s.to_dict() for s in samples])
    return samples, failures, data_fingerprint


def validate_sample_row(
    row: Any,
    line_number: int,
    channel_count: int,
    seen_seq: set[int],
    last_batch: int,
    channel_names: list[str] | None = None,
) -> list[tuple[str, int | None, int | None]]:
    errors: list[tuple[str, int | None, int | None]] = []
    if not isinstance(row, dict):
        return [(f"line {line_number}: row must be a JSON object", None, None)]

    seq = row.get("seq", row.get("sample_seq", row.get("sample_index")))
    batch = row.get("batch", row.get("batch_no"))
    if isinstance(seq, bool) or not isinstance(seq, int):
        errors.append((f"line {line_number}: seq must be an integer", None, _safe_int(batch)))
    elif seq <= 0:
        errors.append((f"line {line_number}: seq must be positive", seq, _safe_int(batch)))
    elif seq in seen_seq:
        errors.append((f"line {line_number}: duplicate seq {seq}", seq, _safe_int(batch)))

    if isinstance(batch, bool) or not isinstance(batch, int):
        errors.append((f"line {line_number}: batch must be a non-negative integer", _safe_int(seq), None))
    elif batch < 0:
        errors.append((f"line {line_number}: batch must be non-negative", _safe_int(seq), batch))
    elif batch < last_batch:
        errors.append(
            (
                f"line {line_number}: batch regressed from {last_batch} to {batch}",
                _safe_int(seq),
                batch,
            )
        )

    values, missing, reading_errors = extract_readings(row, channel_count, channel_names)
    for message in reading_errors:
        errors.append((f"line {line_number}: {message}", _safe_int(seq), _safe_int(batch)))
    return errors


def extract_readings(
    row: dict[str, Any],
    channel_count: int,
    channel_names: list[str] | None = None,
) -> tuple[list[Decimal | None], list[bool], list[str]]:
    errors: list[str] = []
    raw_missing = row.get("missing", row.get("missing_mask", row.get("is_missing")))
    raw_values = row.get("values", row.get("readings", row.get("channels")))

    if isinstance(raw_values, dict):
        names = channel_names or []
        if names and all(name in raw_values for name in names):
            raw_values = [raw_values[name] for name in names]
        else:
            raw_values = [raw_values.get(str(i), raw_values.get(i)) for i in range(channel_count)]

    if not isinstance(raw_values, list) or len(raw_values) != channel_count:
        errors.append(f"values/readings must be a list of length {channel_count}")
        return [], [], errors

    if isinstance(raw_missing, dict):
        names = channel_names or []
        if names and all(name in raw_missing for name in names):
            raw_missing = [raw_missing[name] for name in names]
        else:
            raw_missing = [raw_missing.get(str(i), raw_missing.get(i)) for i in range(channel_count)]
    elif raw_missing is None:
        missing = [v is None for v in raw_values]
    elif isinstance(raw_missing, list) and len(raw_missing) == channel_count:
        missing = []
        for flag in raw_missing:
            if not isinstance(flag, bool):
                errors.append("missing flags must be booleans")
                missing.append(bool(flag))
            else:
                missing.append(flag)
    elif isinstance(raw_missing, bool):
        missing = [raw_missing] * channel_count
    else:
        errors.append(f"missing must be a boolean list of length {channel_count}")
        missing = [v is None for v in raw_values]

    values: list[Decimal | None] = []
    for i, (value, is_missing) in enumerate(zip(raw_values, missing)):
        if is_missing:
            if value is not None:
                errors.append(f"channel {i}: missing marker requires null reading")
            values.append(None)
            continue
        if value is None:
            errors.append(f"channel {i}: null reading requires missing=true")
            values.append(None)
            continue
        try:
            decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        except Exception:
            errors.append(f"channel {i}: reading is not numeric")
            values.append(None)
            continue
        if not decimal_value.is_finite():
            errors.append(f"channel {i}: reading must be finite")
        values.append(decimal_value)
    return values, missing, errors


def _normalize_readings(
    row: dict[str, Any],
    channel_count: int,
    channel_names: list[str] | None = None,
) -> tuple[list[Decimal | None], list[bool]]:
    values, missing, _ = extract_readings(row, channel_count, channel_names)
    return values, missing


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _extract_seq_safely(line: str) -> int | None:
    try:
        row = json.loads(line)
        value = row.get("seq") if isinstance(row, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    except Exception:
        return None


def samples_fingerprint(samples: list[Sample]) -> str:
    return fingerprint([s.to_dict() for s in samples])
