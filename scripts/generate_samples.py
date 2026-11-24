"""Generate deterministic local sample data (no network, repeatable).

Run::

    python scripts/generate_samples.py

Produces under ``examples/``:
    * four binary multi-version frame files (one per observation source);
    * ``declaration.json`` describing sources and rewrite candidates;
    * ``replay_catalog.json`` consumed by the loopback replay service.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slowquery_diag.codec import FrameBuilder  # noqa: E402
from slowquery_diag.models import (  # noqa: E402
    SOURCE_APP,
    SOURCE_DB,
    SOURCE_SLOW,
)

SEED = 20261125
OUT_DIR = os.path.join(ROOT, "examples")
QUERY_IDS = (101, 102, 103)


def _write_frame(filename: str, builder: FrameBuilder) -> str:
    path = os.path.join(OUT_DIR, filename)
    with open(path, "wb") as handle:
        handle.write(builder.build())
    return path


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # -- DB execution stats, two sources with different format versions ----
    db_v2 = FrameBuilder(SOURCE_DB, 2)
    db_v1 = FrameBuilder(SOURCE_DB, 1)
    for query_id in QUERY_IDS:
        for _ in range(8):
            latency = int(rng.integers(800, 90_000))
            rows = int(rng.integers(50, 5_000))
            calls = int(rng.integers(1, 12))
            db_v2.add(2, (query_id, latency, rows, calls))
    for query_id in QUERY_IDS:
        for _ in range(6):
            latency = int(rng.integers(1_000, 80_000))
            rows = int(rng.integers(40, 4_000))
            db_v1.add(1, (query_id, latency, rows))
    _write_frame("db_stats_v2.frame", db_v2)
    _write_frame("db_stats_v1.frame", db_v1)

    # -- Application log summaries, v3 and v1 (different field sets) --------
    app_v3 = FrameBuilder(SOURCE_APP, 3)
    app_v1 = FrameBuilder(SOURCE_APP, 1)
    for query_id in QUERY_IDS:
        for _ in range(10):
            latency_ms = int(rng.integers(2, 120))
            calls = int(rng.integers(1, 20))
            rows = int(rng.integers(10, 2_000))
            errors = int(rng.integers(0, 3))
            app_v3.add(3, (query_id, latency_ms, calls, rows, errors))
    for query_id in QUERY_IDS:
        for _ in range(7):
            latency_ms = int(rng.integers(3, 100))
            calls = int(rng.integers(1, 15))
            app_v1.add(1, (query_id, latency_ms, calls))
    _write_frame("app_log_v3.frame", app_v3)
    _write_frame("app_log_v1.frame", app_v1)

    # -- Slow-log excerpts, v2 and v1 (interval algebra) --------------------
    slow_v2 = FrameBuilder(SOURCE_SLOW, 2)
    slow_v1 = FrameBuilder(SOURCE_SLOW, 1)
    for query_id in QUERY_IDS:
        for _ in range(5):
            low = int(rng.integers(10, 60))
            high = low + int(rng.integers(5, 80))
            count = int(rng.integers(1, 8))
            rows = int(rng.integers(100, 6_000))
            slow_v2.add(2, (query_id, low, high, count, rows))
    for query_id in QUERY_IDS:
        for _ in range(4):
            low = int(rng.integers(12, 55))
            high = low + int(rng.integers(4, 70))
            count = int(rng.integers(1, 6))
            slow_v1.add(1, (query_id, low, high, count))
    _write_frame("slow_log_v2.frame", slow_v2)
    _write_frame("slow_log_v1.frame", slow_v1)

    declaration = {
        "schema_version": 1,
        "sources": [
            {
                "source_id": "src_db_primary",
                "source_type": SOURCE_DB,
                "format_version": 2,
                "sample_rows": 24,
                "latency_caliber": "p95",
                "sample_period_ms": 1000,
                "latency_algebra": {"percentile_rank": 95, "min_samples": 8},
                "frame_file": "db_stats_v2.frame",
            },
            {
                "source_id": "src_db_legacy",
                "source_type": SOURCE_DB,
                "format_version": 1,
                "sample_rows": 18,
                "latency_caliber": "avg",
                "sample_period_ms": 5000,
                "latency_algebra": {"weight": 1, "min_samples": 4},
                "frame_file": "db_stats_v1.frame",
            },
            {
                "source_id": "src_app_current",
                "source_type": SOURCE_APP,
                "format_version": 3,
                "sample_rows": 30,
                "latency_caliber": "p99",
                "sample_period_ms": 2000,
                "latency_algebra": {"percentile_rank": 99, "min_samples": 10},
                "frame_file": "app_log_v3.frame",
            },
            {
                "source_id": "src_app_legacy",
                "source_type": SOURCE_APP,
                "format_version": 1,
                "sample_rows": 21,
                "latency_caliber": "avg",
                "sample_period_ms": 10000,
                "latency_algebra": {"weight": 1},
                "frame_file": "app_log_v1.frame",
            },
            {
                "source_id": "src_slow_excerpt",
                "source_type": SOURCE_SLOW,
                "format_version": 2,
                "sample_rows": 15,
                "latency_caliber": "max",
                "sample_period_ms": 6000,
                "latency_algebra": {"window_ms": 6000},
                "frame_file": "slow_log_v2.frame",
            },
            {
                "source_id": "src_slow_legacy",
                "source_type": SOURCE_SLOW,
                "format_version": 1,
                "sample_rows": 12,
                "latency_caliber": "p50",
                "sample_period_ms": 15000,
                "latency_algebra": {"percentile_rank": 50},
                "frame_file": "slow_log_v1.frame",
            },
        ],
        "candidates": [
            {
                "candidate_id": "cand_101_index_only",
                "original_query": "query 101 summary",
                "rewritten_sql": "SELECT id FROM t101 WHERE k=? INDEXED BY idx_k",
                "target_sources": ["src_db_primary", "src_app_current"],
                "expected_gain_bps": 3500,
            },
            {
                "candidate_id": "cand_101_cover_index",
                "original_query": "query 101 summary",
                "rewritten_sql": "SELECT k,v FROM t101 WHERE k=? COVERING INDEX",
                "target_sources": ["src_db_primary", "src_slow_excerpt"],
                "expected_gain_bps": 3600,
            },
            {
                "candidate_id": "cand_102_rewrite_join",
                "original_query": "query 102 summary",
                "rewritten_sql": "SELECT /*+ HASH_JOIN */ * FROM t102 JOIN u ON ...",
                "target_sources": ["src_db_primary", "src_db_legacy"],
                "expected_gain_bps": 5200,
            },
            {
                "candidate_id": "cand_103_partition_prune",
                "original_query": "query 103 summary",
                "rewritten_sql": "SELECT * FROM t103 WHERE dt BETWEEN ? AND ?",
                "target_sources": ["src_slow_excerpt", "src_app_current"],
                "expected_gain_bps": 2100,
            },
        ],
    }
    with open(os.path.join(OUT_DIR, "declaration.json"), "w",
              encoding="utf-8") as handle:
        json.dump(declaration, handle, indent=2, ensure_ascii=False)

    # Deterministic replay catalog estimates.
    catalog_entries = []
    estimate_latency = {
        "cand_101_index_only": 4200,
        "cand_101_cover_index": 4150,  # intentionally close -> tie-break path
        "cand_102_rewrite_join": 2600,
        "cand_103_partition_prune": 6100,
    }
    for candidate in declaration["candidates"]:
        for source_id in candidate["target_sources"]:
            catalog_entries.append({
                "candidate_id": candidate["candidate_id"],
                "source_id": source_id,
                "est_latency_us": estimate_latency[candidate["candidate_id"]],
                "est_rows_scanned": 320,
                "stall_ms": 0,
            })
    with open(os.path.join(OUT_DIR, "replay_catalog.json"), "w",
              encoding="utf-8") as handle:
        json.dump(
            {"catalog_format": 1, "replay_catalog": catalog_entries},
            handle, indent=2,
        )

    print(f"sample data generated in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
