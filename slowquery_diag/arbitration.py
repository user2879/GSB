"""Stable multi-source evidence arbitration.

For every query and every metric (latency representative value, scanned
rows) the adopted evidence is chosen with the total order::

    1. source-type bias (db_exec_stats < slow_log_excerpt < app_log_summary)
    2. format version descending (higher wins)
    3. source identifier lexicographic ascending

The ordering is explicit (never dict/hash iteration), so the output is
byte-identical for any source reporting order. Rejected evidence is kept
as a control with the reason it lost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .aggregation import Evidence
from .models import SOURCE_TYPE_BIAS


@dataclass(frozen=True)
class MetricEvidenceView:
    source_id: str
    source_type: str
    format_version: int
    value: int
    adopted: bool
    reject_reason: str | None


@dataclass(frozen=True)
class MetricVerdict:
    metric: str                    # "latency_us" | "rows_scanned"
    winner_source: str
    adopted_value: int
    views: tuple[MetricEvidenceView, ...]


@dataclass(frozen=True)
class QueryVerdict:
    query_id: int
    latency: MetricVerdict
    rows: MetricVerdict
    participating_sources: tuple[str, ...]


def _arbitration_key(evidence: Evidence) -> tuple[int, int, str]:
    # Smaller tuple wins: bias rank asc, then version DESC (negated),
    # then source id asc.
    return (
        SOURCE_TYPE_BIAS[evidence.source_type],
        -evidence.format_version,
        evidence.source_id,
    )


def _rank_evidence(evidences: list[Evidence]) -> list[Evidence]:
    return sorted(evidences, key=_arbitration_key)


def _build_metric(metric: str, evidences: list[Evidence],
                  value_getter) -> MetricVerdict:
    ranked = _rank_evidence(evidences)
    winner = ranked[0]
    winner_key = _arbitration_key(winner)
    views: list[MetricEvidenceView] = []
    for evidence in ranked:
        reason: str | None = None
        adopted = evidence.source_id == winner.source_id
        if not adopted:
            reason = _reject_reason(evidence, winner, winner_key)
        views.append(
            MetricEvidenceView(
                source_id=evidence.source_id,
                source_type=evidence.source_type,
                format_version=evidence.format_version,
                value=int(value_getter(evidence)),
                adopted=adopted,
                reject_reason=reason,
            )
        )
    return MetricVerdict(
        metric=metric,
        winner_source=winner.source_id,
        adopted_value=int(value_getter(winner)),
        views=tuple(views),
    )


def _reject_reason(loser: Evidence, winner: Evidence,
                   winner_key: tuple[int, int, str]) -> str:
    loser_key = _arbitration_key(loser)
    loser_bias, loser_neg_ver, _ = loser_key
    winner_bias, winner_neg_ver, winner_id = winner_key
    if loser_bias != winner_bias:
        return (
            f"source-type bias {loser.source_type} ranks after "
            f"{winner.source_type}"
        )
    if loser_neg_ver != winner_neg_ver:
        return (
            f"format version v{loser.format_version} older than "
            f"winner v{winner.format_version}"
        )
    return f"source id {loser.source_id!r} orders after {winner_id!r}"


def merge_evidence(
    evidence_by_source: dict[str, dict[int, Evidence]]
) -> dict[int, QueryVerdict]:
    query_ids = sorted({
        query_id
        for per_source in evidence_by_source.values()
        for query_id in per_source
    })
    verdicts: dict[int, QueryVerdict] = {}
    for query_id in query_ids:
        evidences = [
            per_source[query_id]
            for per_source in evidence_by_source.values()
            if query_id in per_source
        ]
        participating = tuple(sorted(e.source_id for e in evidences))
        latency = _build_metric("latency_us", evidences, lambda e: e.latency_us)
        rows = _build_metric("rows_scanned", evidences, lambda e: e.rows_scanned)
        verdicts[query_id] = QueryVerdict(
            query_id=query_id,
            latency=latency,
            rows=rows,
            participating_sources=participating,
        )
    return verdicts
