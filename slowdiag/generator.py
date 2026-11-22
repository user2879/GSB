from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, sha256_bytes, sha256_canonical
from .codecs import CODEC_SCHEMAS, make_body
from .config import SourceConfig, WorkbenchConfig
from .constants import (
    APP_LOG_SUMMARY,
    DB_EXEC_STATS,
    ELAPSED_FACTOR_BP,
    SCAN_FACTOR_BP,
    SLOW_LOG_EXCERPT,
)


GENERATOR_VERSION = 1

BASE_QUERY_PROFILES: tuple[tuple[int, int], ...] = (
    (120, 900),
    (700, 42000),
    (265, 11800),
    (95, 300),
    (1450, 85000),
    (420, 5000),
    (235, 9000),
    (90, 700),
)

TYPE_QUERY_OVERRIDES: dict[str, dict[str, tuple[int, int]]] = {
    APP_LOG_SUMMARY: {
        "q002": (620, 38000),
        "q003": (225, 11000),
        "q007": (215, 8000),
    },
    DB_EXEC_STATS: {
        "q002": (700, 42000),
        "q003": (268, 12100),
        "q007": (240, 11000),
    },
    SLOW_LOG_EXCERPT: {
        "q002": (790, 47000),
        "q003": (295, 13200),
        "q007": (315, 14500),
    },
}


def query_id_for_index(index: int) -> str:
    return f"q{index + 1:03d}"


def query_count_for_config(config: WorkbenchConfig) -> int:
    return config.dataset.query_count


def _base_profile(query_index: int) -> tuple[int, int]:
    elapsed, scans = BASE_QUERY_PROFILES[query_index % len(BASE_QUERY_PROFILES)]
    cycle = query_index // len(BASE_QUERY_PROFILES)
    if cycle:
        elapsed = elapsed + cycle * 17
        scans = scans + cycle * 503
    return elapsed, scans


def _scaled(value: int, factor_bp: int) -> int:
    return (value * factor_bp + 5000) // 10000


def _version_for(source: SourceConfig, row_no: int) -> int:
    if source.format_version == 1:
        return 1
    return 1 if (row_no + _source_ordinal(source)) % 5 == 0 else 2


def _noise(source: SourceConfig, query_index: int, row_no: int, version: int) -> int:
    period_weight = 1 + ((source.sample_period_ms - 1) // 250)
    raw = (
        _source_ordinal(source)
        + query_index * 7
        + row_no * 11
        + version * 3
    ) % 9
    return (raw - 4) * period_weight


def _make_query_values(
    source: SourceConfig,
    query_index: int,
    row_no: int,
    version: int,
) -> tuple[int, int]:
    query_id = query_id_for_index(query_index)
    elapsed, scans = _base_profile(query_index)
    override = TYPE_QUERY_OVERRIDES.get(source.type, {}).get(query_id)
    if override is not None:
        elapsed, scans = override
    noise = _noise(source, query_index, row_no, version)
    elapsed = max(1, elapsed + noise)
    scans = max(1, scans + noise * 3)
    return elapsed, scans


def _source_ordinal(source: SourceConfig) -> int:
    chars = source.id.encode("utf-8")
    return sum(chars) + len(chars) * 13


def _build_body(
    source: SourceConfig,
    version: int,
    query_id: str,
    elapsed: int,
    scans: int,
    row_no: int,
) -> dict[str, Any]:
    schema = CODEC_SCHEMAS[source.type][version]
    common: dict[str, Any] = {schema.query_field: query_id}
    sample_n = 1 + ((row_no + len(query_id)) % 3)
    if source.type == APP_LOG_SUMMARY:
        common.update(
            {
                "p95_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p95"]),
                "mean_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["mean"]),
                "rows_estimate": _scaled(scans, SCAN_FACTOR_BP["estimated"]),
                "rows_sampled": _scaled(scans, SCAN_FACTOR_BP["sampled"]),
                "sample_n": sample_n,
            }
        )
        if version == 2:
            common.update(
                {
                    "latency_p95_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p95"]),
                    "latency_mean_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["mean"]),
                    "scanned_estimate": _scaled(
                        scans, SCAN_FACTOR_BP["estimated"]
                    ),
                    "scanned_sampled": _scaled(scans, SCAN_FACTOR_BP["sampled"]),
                    "sample_period_ms": source.sample_period_ms,
                }
            )
            for key in (
                "p95_ms",
                "mean_ms",
                "rows_estimate",
                "rows_sampled",
            ):
                common.pop(key, None)
    elif source.type == DB_EXEC_STATS:
        common.update(
            {
                "p95_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p95"]),
                "mean_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["mean"]),
                "p50_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p50"]),
                "rows_exact": _scaled(scans, SCAN_FACTOR_BP["exact"]),
                "sample_n": sample_n,
            }
        )
        if version == 2:
            common.update(
                {
                    "latency_p95_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p95"]),
                    "latency_mean_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["mean"]),
                    "latency_p50_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p50"]),
                    "scans_exact": _scaled(scans, SCAN_FACTOR_BP["exact"]),
                    "executions": 1 + ((row_no + len(query_id)) % 5),
                    "sample_period_ms": source.sample_period_ms,
                }
            )
            for key in ("p95_ms", "mean_ms", "p50_ms", "rows_exact", "sample_n"):
                common.pop(key, None)
    elif source.type == SLOW_LOG_EXCERPT:
        common.update(
            {
                "max_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["max"]),
                "p95_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p95"]),
                "mean_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["mean"]),
                "rows_exact": _scaled(scans, SCAN_FACTOR_BP["exact"]),
                "rows_sampled": _scaled(scans, SCAN_FACTOR_BP["sampled"]),
                "sample_n": sample_n,
            }
        )
        if version == 2:
            common.update(
                {
                    "latency_max_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["max"]),
                    "latency_p95_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["p95"]),
                    "latency_mean_ms": _scaled(elapsed, ELAPSED_FACTOR_BP["mean"]),
                    "scans_exact": _scaled(scans, SCAN_FACTOR_BP["exact"]),
                    "scans_sampled": _scaled(scans, SCAN_FACTOR_BP["sampled"]),
                    "sample_n": sample_n,
                    "sample_period_ms": source.sample_period_ms,
                }
            )
            for key in (
                "max_ms",
                "p95_ms",
                "mean_ms",
                "rows_exact",
                "rows_sampled",
            ):
                common.pop(key, None)
    else:
        raise AssertionError(f"unsupported source type: {source.type}")

    expected = set(
        CODEC_SCHEMAS[source.type][version].integer_fields
        + (CODEC_SCHEMAS[source.type][version].query_field,)
    )
    if set(common) != expected:
        raise AssertionError(
            f"generator field mismatch for {source.type} v{version}: "
            f"missing={sorted(expected - set(common))}, extra={sorted(set(common) - expected)}"
        )
    return common


def build_source_records(
    config: WorkbenchConfig, source: SourceConfig
) -> list[tuple[int, int, bytes]]:
    records: list[tuple[int, int, bytes]] = []
    query_count = query_count_for_config(config)
    for row_no in range(1, source.sample_rows + 1):
        query_index = (row_no - 1) % query_count
        query_id = query_id_for_index(query_index)
        version = _version_for(source, row_no)
        elapsed, scans = _make_query_values(source, query_index, row_no, version)
        body = _build_body(source, version, query_id, elapsed, scans, row_no)
        envelope = make_body(source.type, version, body)
        records.append((row_no, version, canonical_bytes(envelope)))
    return records


def _metadata(config: WorkbenchConfig, source: SourceConfig) -> dict[str, str]:
    return {
        "source_id": source.id,
        "source_type": source.type,
        "format_version": str(source.format_version),
        "sample_rows": str(source.sample_rows),
        "sample_period_ms": str(source.sample_period_ms),
        "elapsed_semantics": source.elapsed_semantics,
        "scan_semantics": source.scan_semantics,
        "generator_version": str(GENERATOR_VERSION),
        "seed": str(config.dataset.seed),
        "query_count": str(config.dataset.query_count),
    }


def logical_digest(
    metadata: dict[str, str], records: list[tuple[int, int, bytes]]
) -> str:
    digest_metadata = {
        key: value for key, value in metadata.items() if key != "content_sha256"
    }
    digest_records = [
        [row_no, version, sha256_bytes(payload)]
        for row_no, version, payload in sorted(records)
    ]
    return sha256_canonical(
        {"metadata": digest_metadata, "records": digest_records}
    )


def write_source_database(
    destination: Path,
    metadata: dict[str, str],
    records: list[tuple[int, int, bytes]],
) -> str:
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".generating")
    if temp_path.exists():
        temp_path.unlink()
    content_sha = logical_digest(metadata, records)
    metadata = dict(metadata)
    metadata["content_sha256"] = content_sha
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE records ("
                "row_no INTEGER PRIMARY KEY, "
                "format_version INTEGER NOT NULL, "
                "payload BLOB NOT NULL, "
                "payload_sha256 TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            connection.executemany(
                "INSERT INTO records(row_no, format_version, payload, payload_sha256) "
                "VALUES (?, ?, ?, ?)",
                [
                    (row_no, version, payload, sha256_bytes(payload))
                    for row_no, version, payload in sorted(records)
                ],
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temp_path, destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return content_sha


def generate_source(
    config: WorkbenchConfig,
    source: SourceConfig,
    output_path: Path | None = None,
) -> dict[str, Any]:
    records = build_source_records(config, source)
    metadata = _metadata(config, source)
    destination = output_path if output_path is not None else source.resolved_data_path
    content_sha = write_source_database(destination, metadata, records)
    return {
        "source_id": source.id,
        "type": source.type,
        "path": str(destination.resolve(strict=False)),
        "sample_rows": source.sample_rows,
        "sample_period_ms": source.sample_period_ms,
        "elapsed_semantics": source.elapsed_semantics,
        "scan_semantics": source.scan_semantics,
        "content_sha256": content_sha,
        "format_versions": sorted({version for _, version, _ in records}),
    }


def generate_all(
    config: WorkbenchConfig,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for source in config.sorted_sources():
        destination = None
        if output_root is not None:
            destination = output_root / source.id / "records.sqlite3"
        summaries.append(generate_source(config, source, destination))
    return summaries


def check_repeatable_generation(config: WorkbenchConfig) -> dict[str, Any]:
    started = time.perf_counter_ns()
    digests: list[list[dict[str, Any]]] = []
    with tempfile.TemporaryDirectory(prefix="slowdiag-repeat-") as temp:
        root = Path(temp)
        digests.append(generate_all(config, root / "run-a"))
        digests.append(generate_all(config, root / "run-b"))
    first = {
        item["source_id"]: item["content_sha256"] for item in digests[0]
    }
    second = {
        item["source_id"]: item["content_sha256"] for item in digests[1]
    }
    equal = first == second
    elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    return {
        "repeatable": equal,
        "first": first,
        "second": second,
        "clock": "perf_counter_ns",
        "elapsed_monotonic_ms": elapsed_ms,
    }
