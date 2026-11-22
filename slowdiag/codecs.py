from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, round_div, true_int
from .config import SourceConfig
from .constants import (
    APP_LOG_SUMMARY,
    DB_EXEC_STATS,
    ELAPSED_FACTOR_BP,
    KNOWN_VERSIONS,
    SCAN_FACTOR_BP,
    SLOW_LOG_EXCERPT,
)
from .errors import format_error


@dataclass(frozen=True)
class CodecSchema:
    query_field: str
    elapsed_fields: dict[str, str]
    scan_fields: dict[str, str]
    integer_fields: tuple[str, ...]
    sample_period_field: str | None


CODEC_SCHEMAS: dict[str, dict[int, CodecSchema]] = {
    APP_LOG_SUMMARY: {
        1: CodecSchema(
            query_field="qid",
            elapsed_fields={"p95": "p95_ms", "mean": "mean_ms"},
            scan_fields={"estimated": "rows_estimate", "sampled": "rows_sampled"},
            integer_fields=(
                "p95_ms",
                "mean_ms",
                "rows_estimate",
                "rows_sampled",
                "sample_n",
            ),
            sample_period_field=None,
        ),
        2: CodecSchema(
            query_field="query_id",
            elapsed_fields={"p95": "latency_p95_ms", "mean": "latency_mean_ms"},
            scan_fields={
                "estimated": "scanned_estimate",
                "sampled": "scanned_sampled",
            },
            integer_fields=(
                "latency_p95_ms",
                "latency_mean_ms",
                "scanned_estimate",
                "scanned_sampled",
                "sample_n",
                "sample_period_ms",
            ),
            sample_period_field="sample_period_ms",
        ),
    },
    DB_EXEC_STATS: {
        1: CodecSchema(
            query_field="query_id",
            elapsed_fields={
                "p95": "p95_ms",
                "mean": "mean_ms",
                "p50": "p50_ms",
            },
            scan_fields={"exact": "rows_exact"},
            integer_fields=(
                "p95_ms",
                "mean_ms",
                "p50_ms",
                "rows_exact",
                "sample_n",
            ),
            sample_period_field=None,
        ),
        2: CodecSchema(
            query_field="qid",
            elapsed_fields={
                "p95": "latency_p95_ms",
                "mean": "latency_mean_ms",
                "p50": "latency_p50_ms",
            },
            scan_fields={"exact": "scans_exact"},
            integer_fields=(
                "latency_p95_ms",
                "latency_mean_ms",
                "latency_p50_ms",
                "scans_exact",
                "executions",
                "sample_period_ms",
            ),
            sample_period_field="sample_period_ms",
        ),
    },
    SLOW_LOG_EXCERPT: {
        1: CodecSchema(
            query_field="query_id",
            elapsed_fields={"max": "max_ms", "p95": "p95_ms", "mean": "mean_ms"},
            scan_fields={"exact": "rows_exact", "sampled": "rows_sampled"},
            integer_fields=(
                "max_ms",
                "p95_ms",
                "mean_ms",
                "rows_exact",
                "rows_sampled",
                "sample_n",
            ),
            sample_period_field=None,
        ),
        2: CodecSchema(
            query_field="qid",
            elapsed_fields={
                "max": "latency_max_ms",
                "p95": "latency_p95_ms",
                "mean": "latency_mean_ms",
            },
            scan_fields={"exact": "scans_exact", "sampled": "scans_sampled"},
            integer_fields=(
                "latency_max_ms",
                "latency_p95_ms",
                "latency_mean_ms",
                "scans_exact",
                "scans_sampled",
                "sample_n",
                "sample_period_ms",
            ),
            sample_period_field="sample_period_ms",
        ),
    },
}


@dataclass(frozen=True)
class Evidence:
    source_id: str
    source_type: str
    query_id: str
    row_no: int
    record_version: int
    source_format_version: int
    original_fields: tuple[str, ...]
    padded_fields: tuple[str, ...]
    elapsed_raw: int
    scanned_raw: int
    elapsed_value: int
    scanned_value: int
    elapsed_semantics: str
    scan_semantics: str
    sample_period_ms: int

    def metric_raw(self, metric: str) -> int:
        if metric == "elapsed_ms":
            return self.elapsed_raw
        if metric == "scanned_rows":
            return self.scanned_raw
        raise KeyError(metric)

    def metric_value(self, metric: str) -> int:
        if metric == "elapsed_ms":
            return self.elapsed_value
        if metric == "scanned_rows":
            return self.scanned_value
        raise KeyError(metric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "query_id": self.query_id,
            "row_no": self.row_no,
            "record_version": self.record_version,
            "source_format_version": self.source_format_version,
            "original_fields": list(self.original_fields),
            "padded_fields": list(self.padded_fields),
            "elapsed_raw": self.elapsed_raw,
            "scanned_raw": self.scanned_raw,
            "elapsed_value": self.elapsed_value,
            "scanned_value": self.scanned_value,
            "elapsed_semantics": self.elapsed_semantics,
            "scan_semantics": self.scan_semantics,
            "sample_period_ms": self.sample_period_ms,
        }


def expected_fields(source_type: str, version: int) -> tuple[str, ...]:
    schema = CODEC_SCHEMAS[source_type][version]
    values = {schema.query_field, *schema.integer_fields}
    return tuple(sorted(values))


def make_body(source_type: str, version: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "length": len(canonical_bytes(body)),
        "version": version,
        "body": body,
    }


def encode_record(source_type: str, version: int, body: dict[str, Any]) -> bytes:
    return canonical_bytes(make_body(source_type, version, body))


def _decode_error(
    source: SourceConfig,
    row_no: int,
    message: str,
    path: str,
) -> Exception:
    return format_error(
        message,
        source_id=source.id,
        row_no=row_no,
        path=path,
    )


def decode_record(source: SourceConfig, row_no: int, payload: bytes) -> Evidence:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _decode_error(source, row_no, "record is not valid UTF-8", "$.body") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _decode_error(
            source,
            row_no,
            f"record JSON is invalid at column {exc.colno}",
            "$.body",
        ) from exc
    if not isinstance(raw, dict):
        raise _decode_error(source, row_no, "record envelope must be an object", "$")

    wrapper_keys = sorted(raw.keys())
    if wrapper_keys != ["body", "length", "version"]:
        expected = ["body", "length", "version"]
        extra = [key for key in wrapper_keys if key not in expected]
        missing = [key for key in expected if key not in raw]
        if missing:
            raise _decode_error(
                source,
                row_no,
                f"record envelope missing field {missing[0]}",
                f"$.{missing[0]}",
            )
        raise _decode_error(
            source,
            row_no,
            f"record envelope has unknown field {extra[0]}",
            f"$.{extra[0]}",
        )

    version = raw["version"]
    if not true_int(version) or version <= 0:
        raise _decode_error(
            source, row_no, "version must be a positive integer", "$.version"
        )
    if version not in KNOWN_VERSIONS[source.type]:
        raise _decode_error(
            source,
            row_no,
            f"unknown record format version {version} for {source.type}",
            "$.version",
        )
    if version > source.format_version:
        raise _decode_error(
            source,
            row_no,
            "record format version exceeds the source's declared format_version",
            "$.version",
        )

    body = raw["body"]
    if not isinstance(body, dict):
        raise _decode_error(source, row_no, "body must be an object", "$.body")

    length = raw["length"]
    if not true_int(length) or length < 0:
        raise _decode_error(
            source, row_no, "length must be a non-negative integer", "$.length"
        )
    actual_length = len(canonical_bytes(body))
    if length != actual_length:
        raise _decode_error(
            source,
            row_no,
            f"length mismatch: declared {length}, actual {actual_length}",
            "$.length",
        )

    schema = CODEC_SCHEMAS[source.type][version]
    allowed = set(expected_fields(source.type, version))
    body_keys = set(body.keys())
    missing = sorted(allowed - body_keys)
    if missing:
        raise _decode_error(
            source,
            row_no,
            f"required field missing: {missing[0]}",
            f"$.body.{missing[0]}",
        )
    extra = sorted(body_keys - allowed)
    if extra:
        raise _decode_error(
            source,
            row_no,
            f"field conflict: {extra[0]} does not belong to version {version}",
            f"$.body.{extra[0]}",
        )

    query_id = body.get(schema.query_field)
    if not isinstance(query_id, str) or not query_id.strip():
        raise _decode_error(
            source,
            row_no,
            "query identifier must be a non-empty string",
            f"$.body.{schema.query_field}",
        )
    query_id = query_id.strip()

    for field in schema.integer_fields:
        value = body.get(field)
        if not true_int(value) or value < 0:
            raise _decode_error(
                source,
                row_no,
                "metric value must be a non-negative integer (booleans are not accepted)",
                f"$.body.{field}",
            )

    elapsed_field = schema.elapsed_fields.get(source.elapsed_semantics)
    if elapsed_field is None:
        raise _decode_error(
            source,
            row_no,
            "record version does not support the declared elapsed semantics",
            "$.body",
        )
    scan_field = schema.scan_fields.get(source.scan_semantics)
    if scan_field is None:
        raise _decode_error(
            source,
            row_no,
            "record version does not support the declared scan semantics",
            "$.body",
        )

    padded: list[str] = []
    if schema.sample_period_field is None:
        padded.append("sample_period_ms")
    else:
        encoded_period = body.get(schema.sample_period_field)
        if encoded_period != source.sample_period_ms:
            raise _decode_error(
                source,
                row_no,
                "encoded sample period conflicts with the source declaration",
                f"$.body.{schema.sample_period_field}",
            )

    elapsed_raw = body[elapsed_field]
    scanned_raw = body[scan_field]
    elapsed_value = round_div(elapsed_raw * 10000, ELAPSED_FACTOR_BP[source.elapsed_semantics])
    scanned_value = round_div(scanned_raw * 10000, SCAN_FACTOR_BP[source.scan_semantics])

    return Evidence(
        source_id=source.id,
        source_type=source.type,
        query_id=query_id,
        row_no=row_no,
        record_version=version,
        source_format_version=source.format_version,
        original_fields=tuple(sorted(body.keys())),
        padded_fields=tuple(padded),
        elapsed_raw=elapsed_raw,
        scanned_raw=scanned_raw,
        elapsed_value=elapsed_value,
        scanned_value=scanned_value,
        elapsed_semantics=source.elapsed_semantics,
        scan_semantics=source.scan_semantics,
        sample_period_ms=source.sample_period_ms,
    )
