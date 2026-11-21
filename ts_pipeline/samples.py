"""Local JSONL sample loading and structural validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .errors import E_SAMPLE_INVALID, Failure


_MISSING_MARKERS = {"MISSING", "__MISSING__", "NA"}


@dataclass(frozen=True)
class SequenceSample:
    sequence_id: str
    timestamps_ms: tuple[int, ...]
    values: tuple[float | None, ...]
    missing: tuple[bool, ...]
    source_line: int
    content_fingerprint: str

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def interval_ms(self) -> int:
        if len(self.timestamps_ms) < 2:
            return 0
        return self.timestamps_ms[1] - self.timestamps_ms[0]

    def to_summary(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "length": self.length,
            "interval_ms": self.interval_ms,
            "missing_count": sum(self.missing),
            "content_fingerprint": self.content_fingerprint,
            "source_line": self.source_line,
        }


@dataclass
class LoadedSamples:
    samples: dict[str, SequenceSample]
    failures: list[Failure]
    fatal: bool


def _line_failure(line_no: int, message: str, sequence_id: str | None = None) -> Failure:
    return Failure(
        code=E_SAMPLE_INVALID,
        stage="load_validation",
        message=message,
        sequence_id=sequence_id,
        detail={"line": line_no},
    )


def _validate_sequence(line_no: int, obj: Any) -> tuple[SequenceSample | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(obj, dict):
        return None, [_line_failure(line_no, "sample line must be a JSON object")]
    sequence_id = obj.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        return None, [_line_failure(line_no, "sequence_id must be a non-empty string")]
    timestamps = obj.get("timestamps_ms")
    values = obj.get("values")
    missing_flags = obj.get("missing")
    if not isinstance(timestamps, list) or not timestamps:
        return None, [_line_failure(line_no, "timestamps_ms must be a non-empty list", sequence_id)]
    if not isinstance(values, list):
        return None, [_line_failure(line_no, "values must be a list", sequence_id)]
    if len(timestamps) != len(values):
        return None, [_line_failure(line_no, "timestamps_ms and values must have equal length", sequence_id)]
    if missing_flags is None:
        missing_flags = [False] * len(values)
    if not isinstance(missing_flags, list) or len(missing_flags) != len(values):
        return None, [_line_failure(line_no, "missing must be a list matching values length", sequence_id)]
    if any(not isinstance(flag, bool) for flag in missing_flags):
        return None, [_line_failure(line_no, "missing entries must be booleans", sequence_id)]

    clean_timestamps: list[int] = []
    for index, stamp in enumerate(timestamps):
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            failures.append(_line_failure(line_no, f"timestamp at position {index} is not an integer", sequence_id))
            return None, failures
        clean_timestamps.append(stamp)
    for index in range(1, len(clean_timestamps)):
        if clean_timestamps[index] <= clean_timestamps[index - 1]:
            failures.append(_line_failure(line_no, "timestamps must be strictly increasing", sequence_id))
            return None, failures
    if len(clean_timestamps) >= 2:
        interval = clean_timestamps[1] - clean_timestamps[0]
        for index in range(2, len(clean_timestamps)):
            if clean_timestamps[index] - clean_timestamps[index - 1] != interval:
                failures.append(_line_failure(line_no, "timestamps must be equally spaced", sequence_id))
                return None, failures

    clean_values: list[float | None] = []
    clean_missing: list[bool] = []
    for index, (value, flag) in enumerate(zip(values, missing_flags)):
        if value is None or (isinstance(value, str) and value in _MISSING_MARKERS):
            if flag is False and value is None:
                clean_values.append(None)
                clean_missing.append(True)
                continue
            clean_values.append(None)
            clean_missing.append(True)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(
                _line_failure(line_no, f"value at position {index} is neither finite nor a missing marker", sequence_id)
            )
            return None, failures
        numeric = float(value)
        if not math.isfinite(numeric):
            failures.append(_line_failure(line_no, f"value at position {index} is not finite", sequence_id))
            return None, failures
        clean_values.append(numeric)
        clean_missing.append(bool(flag))

    sample_payload = {
        "sequence_id": sequence_id,
        "timestamps_ms": clean_timestamps,
        "values": clean_values,
        "missing": clean_missing,
    }
    sample = SequenceSample(
        sequence_id=sequence_id,
        timestamps_ms=tuple(clean_timestamps),
        values=tuple(clean_values),
        missing=tuple(clean_missing),
        source_line=line_no,
        content_fingerprint=fingerprint({"schema": "sample-sequence-v1", **sample_payload}),
    )
    return sample, failures


def load_samples(path: Path) -> LoadedSamples:
    failures: list[Failure] = []
    samples: dict[str, SequenceSample] = {}
    if not path.exists():
        failures.append(
            Failure(E_SAMPLE_INVALID, "load_validation", f"sample file not found: {path}")
        )
        return LoadedSamples({}, failures, True)
    seen: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                failures.append(_line_failure(line_no, f"invalid JSON: {exc.msg}"))
                continue
            sample, item_failures = _validate_sequence(line_no, obj)
            failures.extend(item_failures)
            if sample is None:
                continue
            if sample.sequence_id in seen:
                failures.append(
                    _line_failure(
                        line_no,
                        f"duplicate sequence_id; first seen at line {seen[sample.sequence_id]}",
                        sample.sequence_id,
                    )
                )
                continue
            seen[sample.sequence_id] = line_no
            samples[sample.sequence_id] = sample
    return LoadedSamples(samples, failures, False)


def sample_file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
