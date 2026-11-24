"""Internal data models shared by declaration loading, decoding and arbitration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Supported observation source types, versions and reporting calibers.
# ---------------------------------------------------------------------------

SOURCE_DB = "db_exec_stats"        # library-side execution statistics
SOURCE_APP = "app_log_summary"     # application-side log summaries
SOURCE_SLOW = "slow_log_excerpt"   # slow-log excerpts

SUPPORTED_SOURCE_TYPES: tuple[str, ...] = (SOURCE_DB, SOURCE_APP, SOURCE_SLOW)

# Each source type carries its own integer format versions. Historical
# versions intentionally expose different field sets.
SUPPORTED_VERSIONS: dict[str, tuple[int, ...]] = {
    SOURCE_DB: (1, 2),
    SOURCE_APP: (1, 2, 3),
    SOURCE_SLOW: (1, 2),
}

# Reporting "caliber" (口径) of the latency algebra.
CALIBER_P50 = "p50"
CALIBER_P95 = "p95"
CALIBER_P99 = "p99"
CALIBER_AVG = "avg"
CALIBER_MAX = "max"
SUPPORTED_CALIBERS: tuple[str, ...] = (
    CALIBER_P50, CALIBER_P95, CALIBER_P99, CALIBER_AVG, CALIBER_MAX,
)

# Deterministic source-type bias: smaller index wins evidence arbitration.
SOURCE_TYPE_BIAS: dict[str, int] = {
    SOURCE_DB: 0,
    SOURCE_SLOW: 1,
    SOURCE_APP: 2,
}
BIAS_RANK: tuple[str, ...] = (SOURCE_DB, SOURCE_SLOW, SOURCE_APP)


@dataclass(frozen=True)
class SourceDeclaration:
    source_id: str
    source_type: str
    format_version: int
    sample_rows: int            # declared integer sample row count
    latency_caliber: str        # reporting latency algebra
    sample_period_ms: int       # integer sampling period, milliseconds
    latency_algebra: dict[str, int]  # extra integer algebra parameters
    frame_file: str


@dataclass(frozen=True)
class CandidateRule:
    candidate_id: str
    original_query: str         # original query summary
    rewritten_sql: str          # rewritten form
    target_sources: tuple[str, ...]
    expected_gain_bps: int      # expected gain in integer basis points


@dataclass(frozen=True)
class DeclarationBundle:
    declaration_path: str
    schema_version: int
    sources: tuple[SourceDeclaration, ...]
    candidates: tuple[CandidateRule, ...]


@dataclass
class NormalizedRecord:
    """Canonical record after multi-version decode + backfill."""

    source_id: str
    source_type: str
    format_version: int
    row_number: int                 # 1-based row inside the frame file
    query_id: int
    latency_us: int                 # canonical latency in integer microseconds
    rows_scanned: int               # canonical scanned row count
    call_count: int                 # 0 when not applicable
    filled_fields: tuple[str, ...] = ()
    fill_reason: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceBatch:
    """All normalized records emitted by one observation source."""

    source: SourceDeclaration
    records: tuple[NormalizedRecord, ...]


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sort keys, fixed separators, no whitespace.

    Two structurally equal objects always produce byte-identical output,
    independent of dictionary insertion or hash-iteration order.
    """

    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"object not JSON serializable: {type(value)!r}")
