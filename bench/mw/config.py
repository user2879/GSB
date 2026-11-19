"""Spec/rule/event loading with strict validation and fixed-point helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import SpecError

# int64 saturation bounds used for every fixed-point accumulator.
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1

PARTITION_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
VALID_AGGS = ("sum", "count")
MAX_SCALE = 6


def saturate_int64(value: int) -> int:
    """Clamp an arbitrary-precision integer into the signed 64-bit range."""
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return value


def encode_fixed(value: int, scale: int) -> int:
    """Encode an already-integer sample value at ``scale`` decimal places.

    Inputs are integers, so multiplication by 10**scale is exact; truncation
    direction is therefore *not applicable* and the per-sample absolute
    quantization error is 0. Saturation at int64 boundaries is applied when
    the result cannot be represented.
    """
    return saturate_int64(int(value) * (10 ** scale))


def decode_fixed(fixed: int, scale: int) -> str:
    """Render a fixed-point integer as an exact decimal string."""
    if scale == 0:
        return str(fixed)
    sign = "-" if fixed < 0 else ""
    digits = str(abs(fixed)).zfill(scale + 1)
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}"


def canonical_dumps(payload: Any) -> str:
    """Deterministic serialization: sorted keys, no whitespace, UTF-8.

    Two structurally equal payloads always produce byte-identical output,
    which is the basis for the byte-for-byte view comparison requirement.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _require_int(value: Any, field_name: str, *, line: int | None,
                 minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"field '{field_name}' must be an integer",
                        line=line, field=field_name, got=repr(value))
    if minimum is not None and value < minimum:
        raise SpecError(
            f"field '{field_name}' must be >= {minimum}",
            line=line, field=field_name, got=value)
    return value


def _require_str(value: Any, field_name: str, *, line: int | None) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"field '{field_name}' must be a non-empty string",
                        line=line, field=field_name, got=repr(value))
    return value


@dataclass(frozen=True)
class PartitionRule:
    key: str
    aggregations: tuple[str, ...]


@dataclass(frozen=True)
class SessionSpec:
    """Immutable session parameters declared in the local JSON spec file."""

    out_of_order_ms: int
    latency_budget_ms: int
    scale: int
    rules: tuple[PartitionRule, ...]
    raw: dict[str, Any] = field(repr=False)

    @property
    def partition_keys(self) -> tuple[str, ...]:
        return tuple(rule.key for rule in self.rules)

    def digest_input(self) -> dict[str, Any]:
        return {
            "out_of_order_ms": self.out_of_order_ms,
            "latency_budget_ms": self.latency_budget_ms,
            "scale": self.scale,
            "rules": [
                {"key": rule.key, "aggregations": list(rule.aggregations)}
                for rule in self.rules
            ],
        }


def load_spec(path: str) -> SessionSpec:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise SpecError(f"spec file not found: {path}", field="<file>")
    except json.JSONDecodeError as exc:
        raise SpecError("spec file is not valid JSON", line=exc.lineno,
                        field="<json>", detail=exc.msg)
    if not isinstance(raw, dict):
        raise SpecError("spec root must be a JSON object", field="<root>")

    allowed = {"out_of_order_ms", "latency_budget_ms", "scale", "partitions"}
    unknown = set(raw) - allowed
    if unknown:
        raise SpecError(f"unknown spec field(s): {sorted(unknown)}",
                        field=sorted(unknown)[0])

    out_of_order = _require_int(raw.get("out_of_order_ms"),
                                "out_of_order_ms", line=None, minimum=1)
    budget = _require_int(raw.get("latency_budget_ms"),
                          "latency_budget_ms", line=None, minimum=1)
    scale = _require_int(raw.get("scale"), "scale", line=None, minimum=0)
    if scale > MAX_SCALE:
        raise SpecError(f"field 'scale' must be <= {MAX_SCALE}",
                        field="scale", got=scale)

    partitions = raw.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise SpecError("field 'partitions' must be a non-empty array",
                        field="partitions")

    rules: list[PartitionRule] = []
    seen: set[str] = set()
    for index, entry in enumerate(partitions):
        pfield = f"partitions[{index}]"
        if not isinstance(entry, dict):
            raise SpecError(f"{pfield} must be an object", field=pfield)
        key = _require_str(entry.get("key"), f"{pfield}.key", line=None)
        if not PARTITION_KEY_RE.match(key):
            raise SpecError(
                f"partition key '{key}' must match {PARTITION_KEY_RE.pattern}",
                field=f"{pfield}.key", got=key)
        if key in seen:
            raise SpecError(f"duplicate partition key '{key}'",
                            field=f"{pfield}.key")
        seen.add(key)
        aggs = entry.get("aggregations")
        if not isinstance(aggs, list) or not aggs:
            raise SpecError(
                f"{pfield}.aggregations must be a non-empty array",
                field=f"{pfield}.aggregations")
        for agg in aggs:
            if agg not in VALID_AGGS:
                raise SpecError(
                    f"unsupported aggregation '{agg}' (allowed: "
                    f"{list(VALID_AGGS)})", field=f"{pfield}.aggregations")
        rules.append(PartitionRule(key=key,
                                   aggregations=tuple(dict.fromkeys(aggs))))

    return SessionSpec(out_of_order_ms=out_of_order,
                       latency_budget_ms=budget, scale=scale,
                       rules=tuple(rules), raw=raw)


@dataclass(frozen=True)
class Event:
    partition: str
    seq: int
    ts_ms: int
    value: int
    source_line: int


def load_events(path: str, spec: SessionSpec) -> list[Event]:
    """Load and strictly validate an NDJSON event stream.

    Validation is whole-file: the first offending line/field aborts the load
    with E_SPEC and nothing is committed to a session. Per-partition sequence
    numbers must start at 0 and be contiguous and strictly monotonic.
    """
    allowed_keys = {"partition", "seq", "ts_ms", "value"}
    events: list[Event] = []
    next_seq: dict[str, int] = {key: 0 for key in spec.partition_keys}
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        raise SpecError(f"event file not found: {path}", field="<file>")
    with handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SpecError("event line is not valid JSON",
                                line=line_no, field="<json>", detail=exc.msg)
            if not isinstance(obj, dict):
                raise SpecError("event must be a JSON object",
                                line=line_no, field="<root>")
            unknown = set(obj) - allowed_keys
            if unknown:
                raise SpecError(f"unknown event field(s): {sorted(unknown)}",
                                line=line_no, field=sorted(unknown)[0])
            partition = _require_str(obj.get("partition"), "partition",
                                     line=line_no)
            if not PARTITION_KEY_RE.match(partition):
                raise SpecError(
                    f"partition key '{partition}' has an illegal format",
                    line=line_no, field="partition", got=partition)
            if partition not in next_seq:
                raise SpecError(
                    f"partition '{partition}' is not declared in the spec",
                    line=line_no, field="partition")
            seq = _require_int(obj.get("seq"), "seq", line=line_no,
                               minimum=0)
            expected = next_seq[partition]
            if seq != expected:
                raise SpecError(
                    f"partition '{partition}' seq must be contiguous and "
                    f"monotonic; expected {expected} got {seq}",
                    line=line_no, field="seq", expected=expected, got=seq)
            next_seq[partition] = seq + 1
            ts_ms = _require_int(obj.get("ts_ms"), "ts_ms", line=line_no,
                                 minimum=0)
            value = _require_int(obj.get("value"), "value", line=line_no)
            events.append(Event(partition=partition, seq=seq, ts_ms=ts_ms,
                                value=value, source_line=line_no))
    if not events:
        raise SpecError("event stream contains no events", field="<file>")
    for key, expected_next in next_seq.items():
        if expected_next == 0:
            raise SpecError(
                f"declared partition '{key}' has no events", field="partitions")
    return events
