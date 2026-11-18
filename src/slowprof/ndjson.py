"""Strict NDJSON loading and deterministic sample-stream metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json_bytes
from .errors import E_DEPTH, E_IO, E_SAMPLE, SampleValidationError, SlowProfError
from .models import Interval, ProfilePolicy, Sample, frame_is_valid, is_int

_REQUIRED_FIELDS = (
    "request_id",
    "timestamp_ms",
    "duration_ms",
    "service_id",
    "frames",
    "weight",
)


@dataclass(frozen=True, slots=True)
class NDJSONMetadata:
    path: str
    file_sha256: str
    selected_samples_sha256: str
    line_count: int
    selected_row_count: int
    selected_total_weight: int
    unique_stack_count: int
    frame_names: tuple[str, ...]
    frame_name_bytes: int
    selected_total_depth: int
    selected_max_depth: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_sha256": self.file_sha256,
            "selected_samples_sha256": self.selected_samples_sha256,
            "line_count": self.line_count,
            "selected_row_count": self.selected_row_count,
            "selected_total_weight": self.selected_total_weight,
            "unique_stack_count": self.unique_stack_count,
            "frame_count": len(self.frame_names),
            "frame_name_bytes": self.frame_name_bytes,
            "selected_total_depth": self.selected_total_depth,
            "selected_max_depth": self.selected_max_depth,
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
        }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_object(raw: bytes, line_no: int) -> dict[str, Any]:
    payload = raw[:-1] if raw.endswith(b"\n") else raw
    if payload.endswith(b"\r"):
        payload = payload[:-1]
    if line_no == 1 and payload.startswith(b"\xef\xbb\xbf"):
        payload = payload[3:]
    if not payload.strip():
        raise SampleValidationError(
            "blank lines are not valid NDJSON records",
            line_no=line_no,
            field="line",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SampleValidationError(
            "line is not valid UTF-8",
            line_no=line_no,
            field="line",
            details={"error": str(exc)},
        ) from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SampleValidationError(
            "line is not valid JSON",
            line_no=line_no,
            field="line",
            details={"error": str(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise SampleValidationError(
            "each NDJSON line must be a JSON object",
            line_no=line_no,
            field="line",
        )
    return value


def parse_sample_object(value: dict[str, Any], *, line_no: int, max_depth: int) -> Sample:
    """Validate one already-decoded object and return an immutable sample."""

    for field in _REQUIRED_FIELDS:
        if field not in value:
            raise SampleValidationError(
                f"missing required field {field}",
                line_no=line_no,
                field=field,
            )

    request_id = value["request_id"]
    if not isinstance(request_id, str) or not request_id:
        raise SampleValidationError(
            "request_id must be a non-empty string",
            line_no=line_no,
            field="request_id",
        )

    service_id = value["service_id"]
    if not isinstance(service_id, str) or not service_id:
        raise SampleValidationError(
            "service_id must be a non-empty string",
            line_no=line_no,
            field="service_id",
        )

    timestamp_ms = value["timestamp_ms"]
    if not is_int(timestamp_ms):
        raise SampleValidationError(
            "timestamp_ms must be an integer",
            line_no=line_no,
            field="timestamp_ms",
        )

    duration_ms = value["duration_ms"]
    if not is_int(duration_ms) or duration_ms <= 0:
        raise SampleValidationError(
            "duration_ms must be a positive integer",
            line_no=line_no,
            field="duration_ms",
            details={"actual": duration_ms},
        )

    weight = value["weight"]
    if not is_int(weight) or weight <= 0:
        raise SampleValidationError(
            "weight must be a positive integer",
            line_no=line_no,
            field="weight",
            details={"actual": weight},
        )

    frames_value = value["frames"]
    if not isinstance(frames_value, list) or not frames_value:
        raise SampleValidationError(
            "frames must be a non-empty list of frame names",
            line_no=line_no,
            field="frames",
        )
    if len(frames_value) > max_depth:
        raise SampleValidationError(
            "stack depth exceeds max_depth",
            line_no=line_no,
            field="frames",
            code=E_DEPTH,
            details={"depth": len(frames_value), "max_depth": max_depth},
        )
    frames: list[str] = []
    for index, frame in enumerate(frames_value):
        if not frame_is_valid(frame):
            raise SampleValidationError(
                "frame must have the exact form svc.method",
                line_no=line_no,
                field=f"frames[{index}]",
                details={"actual": frame},
            )
        frames.append(frame)

    return Sample(
        request_id=request_id,
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        service_id=service_id,
        frames=tuple(frames),
        weight=weight,
        line_no=line_no,
    )


def parse_sample_line(raw: bytes, *, line_no: int, max_depth: int) -> Sample:
    return parse_sample_object(_decode_object(raw, line_no), line_no=line_no, max_depth=max_depth)


def _sample_digest_record(sample: Sample) -> dict[str, Any]:
    return {
        "request_id": sample.request_id,
        "timestamp_ms": sample.timestamp_ms,
        "duration_ms": sample.duration_ms,
        "service_id": sample.service_id,
        "frames": list(sample.frames),
        "weight": sample.weight,
    }


def _stack_digest(sample: Sample) -> bytes:
    payload = canonical_json_bytes(list(sample.frames))
    return hashlib.blake2b(payload, digest_size=16).digest()


def _iter_file_bytes(path: str | Path) -> Iterator[tuple[int, bytes]]:
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise SlowProfError(
            E_IO,
            "cannot open NDJSON sample file",
            field="samples_path",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    with handle:
        for line_no, raw in enumerate(handle, start=1):
            yield line_no, raw


def scan_ndjson(path: str | Path, policy: ProfilePolicy, interval: Interval) -> NDJSONMetadata:
    """Validate the complete stream and return selection metadata.

    No aggregate is returned from this function.  Every line, including lines
    outside the selected interval, is validated before metadata is exposed.
    """

    file_digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    selected_frame_names: set[str] = set()
    unique_stack_digests: set[bytes] = set()
    line_count = 0
    selected_row_count = 0
    selected_total_weight = 0
    selected_total_depth = 0
    selected_max_depth = 0
    frame_name_bytes = 0
    first_timestamp: int | None = None
    last_timestamp: int | None = None

    for line_no, raw in _iter_file_bytes(path):
        file_digest.update(raw)
        line_count = line_no
        sample = parse_sample_line(raw, line_no=line_no, max_depth=policy.max_depth)
        if last_timestamp is not None and sample.timestamp_ms < last_timestamp:
            raise SampleValidationError(
                "timestamp_ms must be monotonically non-decreasing",
                line_no=line_no,
                field="timestamp_ms",
                details={"previous": last_timestamp, "actual": sample.timestamp_ms},
            )
        if first_timestamp is None:
            first_timestamp = sample.timestamp_ms
        last_timestamp = sample.timestamp_ms
        if not interval.includes(sample.timestamp_ms):
            continue
        selected_row_count += 1
        selected_total_weight += sample.weight
        selected_total_depth += len(sample.frames)
        selected_max_depth = max(selected_max_depth, len(sample.frames))
        unique_stack_digests.add(_stack_digest(sample))
        for frame in sample.frames:
            selected_frame_names.add(frame)
            frame_name_bytes += len(frame.encode("utf-8"))
        selected_digest.update(canonical_json_bytes(_sample_digest_record(sample)))
        selected_digest.update(b"\n")

    return NDJSONMetadata(
        path=str(Path(path)),
        file_sha256=file_digest.hexdigest(),
        selected_samples_sha256=selected_digest.hexdigest(),
        line_count=line_count,
        selected_row_count=selected_row_count,
        selected_total_weight=selected_total_weight,
        unique_stack_count=len(unique_stack_digests),
        frame_names=tuple(sorted(selected_frame_names)),
        frame_name_bytes=frame_name_bytes,
        selected_total_depth=selected_total_depth,
        selected_max_depth=selected_max_depth,
        first_timestamp_ms=first_timestamp,
        last_timestamp_ms=last_timestamp,
    )


def iter_validated_samples(
    path: str | Path,
    policy: ProfilePolicy,
    interval: Interval,
    *,
    expected: NDJSONMetadata | None = None,
) -> Iterator[Sample]:
    """Re-validate and yield selected samples without retaining them.

    If ``expected`` is supplied, the complete file and selected-content hashes
    are checked at end-of-stream.  A changed file therefore invalidates the
    entire aggregation rather than producing a mixed partial result.
    """

    file_digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    last_timestamp: int | None = None
    count = 0
    selected_count = 0
    for line_no, raw in _iter_file_bytes(path):
        count = line_no
        file_digest.update(raw)
        sample = parse_sample_line(raw, line_no=line_no, max_depth=policy.max_depth)
        if last_timestamp is not None and sample.timestamp_ms < last_timestamp:
            raise SampleValidationError(
                "timestamp_ms must be monotonically non-decreasing",
                line_no=line_no,
                field="timestamp_ms",
                details={"previous": last_timestamp, "actual": sample.timestamp_ms},
            )
        last_timestamp = sample.timestamp_ms
        if not interval.includes(sample.timestamp_ms):
            continue
        selected_count += 1
        selected_digest.update(canonical_json_bytes(_sample_digest_record(sample)))
        selected_digest.update(b"\n")
        yield sample

    if expected is not None:
        actual_file_sha = file_digest.hexdigest()
        actual_selected_sha = selected_digest.hexdigest()
        if count != expected.line_count or actual_file_sha != expected.file_sha256:
            raise SampleValidationError(
                "sample file changed between validation and aggregation",
                line_no=count if count else 1,
                field="$file",
                details={
                    "expected_line_count": expected.line_count,
                    "actual_line_count": count,
                    "expected_file_sha256": expected.file_sha256,
                    "actual_file_sha256": actual_file_sha,
                },
            )
        if (
            selected_count != expected.selected_row_count
            or actual_selected_sha != expected.selected_samples_sha256
        ):
            raise SampleValidationError(
                "selected sample content changed between validation and aggregation",
                line_no=count if count else 1,
                field="$file",
                details={
                    "expected_selected_row_count": expected.selected_row_count,
                    "actual_selected_row_count": selected_count,
                    "expected_selected_samples_sha256": expected.selected_samples_sha256,
                    "actual_selected_samples_sha256": actual_selected_sha,
                },
            )


def load_samples(
    path: str | Path,
    policy: ProfilePolicy,
    interval: Interval,
) -> tuple[tuple[Sample, ...], NDJSONMetadata]:
    """Materialize a fully validated selected stream (convenience API)."""

    metadata = scan_ndjson(path, policy, interval)
    samples = tuple(
        iter_validated_samples(path, policy, interval, expected=metadata)
    )
    return samples, metadata
