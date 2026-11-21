"""Deterministic NDJSON sample loading and strict validation.

Input contract (one JSON object per line, a request-aggregated sample)::

    {
      "request_id": "req-0001",
      "timestamp_ms": 1700000000123,   # integer milliseconds
      "duration_ms": 42,               # positive integer milliseconds
      "service_id": "billing",
      "weight": 3,                     # positive integer sampling weight
      "frames": ["billing.charge", "gateway.http", "tcp.send"]
    }

The stack is ordered root-first; the *last* frame is the on-CPU / leaf frame.
Validation runs in file order and every error is reported with its 1-based
physical line number and the offending field name under :class:`SampleError`
(``E_SAMPLE``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .errors import RangeError, SampleError

# svc.method: two non-empty dot-separated segments, letters/digits/_/- only.
FRAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")

REQUIRED_FIELDS = (
    "request_id",
    "timestamp_ms",
    "duration_ms",
    "service_id",
    "weight",
    "frames",
)


@dataclass(frozen=True)
class Sample:
    """One validated, request-aggregated stack sample."""

    request_id: str
    timestamp_ms: int
    duration_ms: int
    service_id: str
    weight: int
    frames: Tuple[str, ...]
    line_number: int


@dataclass(frozen=True)
class Interval:
    """Inclusive observation window in integer-millisecond coordinates."""

    start_ms: Optional[int]
    end_ms: Optional[int]

    def contains(self, timestamp_ms: int) -> bool:
        if self.start_ms is not None and timestamp_ms < self.start_ms:
            return False
        if self.end_ms is not None and timestamp_ms > self.end_ms:
            return False
        return True

    def canonical(self) -> str:
        return f"{self.start_ms if self.start_ms is not None else '-'}" \
               f"..{self.end_ms if self.end_ms is not None else '-'}"


def make_interval(start_ms: Optional[int],
                  end_ms: Optional[int]) -> Interval:
    """Validate interval parameters (``E_RANGE`` on bad bounds)."""

    if start_ms is not None and not isinstance(start_ms, int):
        raise RangeError("start_ms must be an integer or null",
                         field="start_ms")
    if end_ms is not None and not isinstance(end_ms, int):
        raise RangeError("end_ms must be an integer or null",
                         field="end_ms")
    if (start_ms is not None and end_ms is not None
            and start_ms > end_ms):
        raise RangeError(
            f"start_ms={start_ms} is greater than end_ms={end_ms}",
            field="start_ms",
        )
    return Interval(start_ms, end_ms)


def _require_int(value: object, field: str, line_number: int,
                 *, positive: bool = False) -> int:
    # bool is a subclass of int but never a valid timestamp/duration/weight.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SampleError(
            f"{field} must be an integer, got {type(value).__name__}",
            field=field, line_number=line_number,
        )
    if positive and value <= 0:
        raise SampleError(
            f"{field} must be a positive integer, got {value}",
            field=field, line_number=line_number,
        )
    return value


def _validate_frames(value: object,
                     line_number: int) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SampleError(
            "frames must be a non-empty list of frame names",
            field="frames", line_number=line_number,
        )
    frames: List[str] = []
    for index, frame in enumerate(value):
        field = f"frames[{index}]"
        if not isinstance(frame, str):
            raise SampleError(
                f"{field} must be a string",
                field=field, line_number=line_number,
            )
        if not frame or not FRAME_RE.match(frame):
            raise SampleError(
                f"{field}={frame!r} must match svc.method with no empty "
                "segments",
                field=field, line_number=line_number,
            )
        frames.append(frame)
    return tuple(frames)


def parse_line(raw_line: str, line_number: int) -> Sample:
    """Parse and validate one physical NDJSON line."""

    try:
        obj = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise SampleError(
            f"invalid JSON: {exc.msg}", field="<json>",
            line_number=line_number,
        ) from exc
    if not isinstance(obj, dict):
        raise SampleError(
            "sample row must be a JSON object", field="<json>",
            line_number=line_number,
        )
    for field in REQUIRED_FIELDS:
        if field not in obj:
            raise SampleError(
                f"missing required field {field!r}", field=field,
                line_number=line_number,
            )

    request_id = obj["request_id"]
    if not isinstance(request_id, str) or not request_id:
        raise SampleError(
            "request_id must be a non-empty string",
            field="request_id", line_number=line_number,
        )
    service_id = obj["service_id"]
    if not isinstance(service_id, str) or not service_id:
        raise SampleError(
            "service_id must be a non-empty string",
            field="service_id", line_number=line_number,
        )
    timestamp_ms = _require_int(
        obj["timestamp_ms"], "timestamp_ms", line_number)
    duration_ms = _require_int(
        obj["duration_ms"], "duration_ms", line_number, positive=True)
    weight = _require_int(
        obj["weight"], "weight", line_number, positive=True)
    frames = _validate_frames(obj["frames"], line_number)

    return Sample(
        request_id=request_id,
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        service_id=service_id,
        weight=weight,
        frames=frames,
        line_number=line_number,
    )


def iter_samples(path: str | Path) -> Iterator[Sample]:
    """Yield validated samples in physical file order.

    Per-service timestamps are checked for the monotone-non-decreasing
    invariant as samples are seen.
    """

    last_timestamp: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8", newline="\n") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue  # blank lines are tolerated, not numbered as rows
            sample = parse_line(raw_line, line_number)
            previous = last_timestamp.get(sample.service_id)
            if previous is not None and sample.timestamp_ms < previous:
                raise SampleError(
                    f"service {sample.service_id!r} timestamp regressed: "
                    f"{sample.timestamp_ms} < previous {previous}",
                    field="timestamp_ms", line_number=line_number,
                )
            last_timestamp[sample.service_id] = sample.timestamp_ms
            yield sample


def load_samples(path: str | Path,
                 interval: Optional[Interval] = None) -> List[Sample]:
    """Eagerly load samples inside ``interval`` (all samples if ``None``)."""

    samples: List[Sample] = []
    for sample in iter_samples(path):
        if interval is None or interval.contains(sample.timestamp_ms):
            samples.append(sample)
    return samples


def sample_stream_digest(path: str | Path,
                         interval: Optional[Interval] = None) -> str:
    """SHA-256 over the canonical per-row bytes of the in-window stream.

    Hashing the canonical re-serialization (not the raw text) makes the
    digest insensitive to whitespace/key ordering while still binding the
    full logical sample content and its physical ordering.
    """

    digest = hashlib.sha256()
    for sample in iter_samples(path):
        if interval is not None and not interval.contains(
                sample.timestamp_ms):
            continue
        canonical = json.dumps(
            {
                "request_id": sample.request_id,
                "timestamp_ms": sample.timestamp_ms,
                "duration_ms": sample.duration_ms,
                "service_id": sample.service_id,
                "weight": sample.weight,
                "frames": list(sample.frames),
            },
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(canonical).to_bytes(8, "big"))
        digest.update(canonical)
    return digest.hexdigest()


def summarize_input(path: str | Path,
                    interval: Optional[Interval] = None) -> dict:
    """Deterministic input summary used by generator/demo output."""

    total_rows = 0
    total_weight = 0
    services: set[str] = set()
    depth_sum = 0
    digest = hashlib.sha256()
    for sample in iter_samples(path):
        if interval is not None and not interval.contains(
                sample.timestamp_ms):
            continue
        total_rows += 1
        total_weight += sample.weight
        services.add(sample.service_id)
        depth_sum += len(sample.frames)
        digest.update(
            json.dumps(
                [sample.request_id, sample.timestamp_ms,
                 sample.duration_ms, sample.service_id,
                 sample.weight, list(sample.frames)],
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return {
        "rows_in_window": total_rows,
        "total_weight": total_weight,
        "service_count": len(services),
        "services": sorted(services),
        "avg_depth": (depth_sum / total_rows) if total_rows else 0.0,
        "digest": digest.hexdigest(),
        "interval": interval.canonical() if interval is not None else "-..-",
    }
