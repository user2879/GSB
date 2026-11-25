"""v1/v2/v3 observation record codecs and normalization (requirement 2).

On-disk line formats (pipe separated, one record per line):
  v1: v1|<obs_seq>|<value>
  v2: v2|<obs_seq>|<source_id>|<timestamp>|<value>
  v3: v3|<obs_seq>|<source_id>|<timestamp>|<metric_key>|<scale>|<value>

<value> is an integer or decimal literal, or the literal "null".

Normalization fills fields missing from older formats and records, per
record, the source format version, the set of filled fields and the fill
rationale. Decode validation:
  - obs_seq strictly increasing within a stream  -> E_FIELD (obs_seq)
  - timestamp non-negative                        -> E_FIELD (timestamp)
  - scale in the supported set                    -> E_VERSION (scale)
Raw 1-based line numbers are preserved on every error.

Fixed-point conversion: Decimal(value) * scale, rounded ROUND_HALF_EVEN to an
integer. This is the declared single-value quantization rule (half ulp bound).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

from .errors import FieldError, VersionError
from .spec import SUPPORTED_SCALES, Source

NULL_LITERAL = "null"

_FIELD_COUNTS = {1: 3, 2: 5, 3: 7}


@dataclass(frozen=True)
class NormalizedRecord:
    version: int
    seq: int
    source_id: str
    timestamp: int
    metric_key: str
    scale: int
    value_fixed: int | None  # None when null
    filled_fields: tuple
    fill_rationale: tuple  # tuple of (field, rationale) pairs, sorted by field

    def canonical(self):
        value = NULL_LITERAL if self.value_fixed is None else str(self.value_fixed)
        return "%d|%d|%s|%d|%s|%d|%s" % (
            self.version,
            self.seq,
            self.source_id,
            self.timestamp,
            self.metric_key,
            self.scale,
            value,
        )


def decimal_to_fixed(raw, scale):
    """Declared quantization: Decimal(raw) * scale, ROUND_HALF_EVEN."""
    return int((Decimal(raw) * scale).to_integral_value(rounding=ROUND_HALF_EVEN))


def _parse_int(token, line_no, field):
    try:
        return int(token)
    except (TypeError, ValueError):
        raise FieldError(line_no, field, "expected integer, got %r" % (token,))


def decode_line(line, line_no, source: Source, prev_seq):
    """Decode one raw line into a NormalizedRecord (null bitmap not applied)."""
    parts = line.rstrip("\n").split("|")
    tag = parts[0]
    if tag not in ("v1", "v2", "v3"):
        raise VersionError(line_no, "version", "unknown record format tag %r" % (tag,))
    version = int(tag[1])
    if version != source.record_format:
        raise VersionError(
            line_no, "version", "line format v%d does not match declared source format v%d" % (version, source.record_format)
        )
    if len(parts) != _FIELD_COUNTS[version]:
        raise FieldError(line_no, "format", "v%d expects %d fields, got %d" % (version, _FIELD_COUNTS[version], len(parts)))

    filled = []
    rationale = []

    seq = _parse_int(parts[1], line_no, "obs_seq")
    if prev_seq is not None and seq <= prev_seq:
        raise FieldError(line_no, "obs_seq", "obs_seq %d not greater than previous %d" % (seq, prev_seq))

    if version >= 2:
        source_id = parts[2]
        timestamp = _parse_int(parts[3], line_no, "timestamp")
        if timestamp < 0:
            raise FieldError(line_no, "timestamp", "timestamp %d is negative" % timestamp)
    else:
        source_id = source.id
        timestamp = 0
        filled.extend(["source_id", "timestamp"])
        rationale.append(("source_id", "missing-in-v1: filled from source declaration id"))
        rationale.append(("timestamp", "missing-in-v1: filled with 0 (no acquisition time in format)"))

    if version >= 3:
        metric_key = parts[4]
        scale = _parse_int(parts[5], line_no, "scale")
        value_token = parts[6]
        if scale not in SUPPORTED_SCALES:
            raise VersionError(line_no, "scale", "scale %d not in supported set %s" % (scale, SUPPORTED_SCALES))
    else:
        metric_key = source.metrics[0]
        scale = source.default_scale
        value_token = parts[2] if version == 1 else parts[4]
        filled.extend(["metric_key", "scale"])
        rationale.append(("metric_key", "missing-in-v%d: filled with first declared metric of source" % version))
        rationale.append(("scale", "missing-in-v%d: filled with source default_scale" % version))

    if value_token == NULL_LITERAL:
        value_fixed = None
    else:
        try:
            value_fixed = decimal_to_fixed(value_token, scale)
        except InvalidOperation:
            raise FieldError(line_no, "value", "unparseable numeric value %r" % (value_token,))

    return NormalizedRecord(
        version=version,
        seq=seq,
        source_id=source_id,
        timestamp=timestamp,
        metric_key=metric_key,
        scale=scale,
        value_fixed=value_fixed,
        filled_fields=tuple(sorted(filled)),
        fill_rationale=tuple(sorted(rationale)),
    )


def decode_stream(lines, source: Source):
    """Decode a whole stream, then apply the source null bitmap by position.

    The null bitmap is authoritative: position i with bitmap 1 forces a null
    value regardless of the literal on the line.
    """
    records = []
    prev_seq = None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        record = decode_line(line, index + 1, source, prev_seq)
        prev_seq = record.seq
        records.append(record)
    if len(records) != source.record_count:
        raise FieldError(
            len(lines), "record_count",
            "decoded %d records but source declares %d" % (len(records), source.record_count),
        )
    out = []
    for index, record in enumerate(records):
        if source.null_bitmap[index] == 1 and record.value_fixed is not None:
            record = NormalizedRecord(
                version=record.version,
                seq=record.seq,
                source_id=record.source_id,
                timestamp=record.timestamp,
                metric_key=record.metric_key,
                scale=record.scale,
                value_fixed=None,
                filled_fields=record.filled_fields,
                fill_rationale=record.fill_rationale,
            )
        out.append(record)
    return out


def encode_line(record: NormalizedRecord):
    """Encode a normalized record back to its on-disk format version."""
    value = NULL_LITERAL if record.value_fixed is None else _fixed_to_decimal(record.value_fixed, record.scale)
    if record.version == 1:
        return "v1|%d|%s" % (record.seq, value)
    if record.version == 2:
        return "v2|%d|%s|%d|%s" % (record.seq, record.source_id, record.timestamp, value)
    return "v3|%d|%s|%d|%s|%d|%s" % (
        record.seq, record.source_id, record.timestamp, record.metric_key, record.scale, value,
    )


def _fixed_to_decimal(value_fixed, scale):
    quantized = (Decimal(value_fixed) / scale).normalize()
    return format(quantized, "f")
