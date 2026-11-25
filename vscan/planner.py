from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import digest_value
from .codec import DECODED_CELL_BYTES
from .spec import QuerySpec, ScanSpec, SegmentSpec, predicate_leaf_count


@dataclass(frozen=True)
class BatchPlan:
    query_id: str
    segment_id: str
    batch_no: int
    start_row: int
    end_row: int
    decode_columns: tuple[str, ...]
    predicate_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    pruned_columns: tuple[str, ...]
    predicate: dict[str, Any] | None
    predicate_leaf_count: int
    all_columns: tuple[str, ...]
    plan_digest: str

    @property
    def rows(self) -> int:
        return self.end_row - self.start_row

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.query_id, self.segment_id, self.batch_no)

    @property
    def uncropped_decoded_bytes(self) -> int:
        return self.rows * len(self.all_columns) * DECODED_CELL_BYTES

    @property
    def pruned_full_scan_decoded_bytes(self) -> int:
        return self.rows * len(self.decode_columns) * DECODED_CELL_BYTES

    @property
    def admission_decoded_bytes(self) -> int:
        return self.pruned_full_scan_decoded_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "segment_id": self.segment_id,
            "batch_no": self.batch_no,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "row_count": self.rows,
            "decode_columns": list(self.decode_columns),
            "predicate_columns": list(self.predicate_columns),
            "output_columns": list(self.output_columns),
            "pruned_columns": list(self.pruned_columns),
            "predicate": self.predicate,
            "predicate_leaf_count": self.predicate_leaf_count,
            "uncropped_decoded_bytes": self.uncropped_decoded_bytes,
            "pruned_full_scan_decoded_bytes": self.pruned_full_scan_decoded_bytes,
            "admission_decoded_bytes": self.admission_decoded_bytes,
            "batch_digest": self.plan_digest,
        }


@dataclass(frozen=True)
class QueryPlan:
    query: QuerySpec
    per_segment: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    plan_digest: str
    batches: tuple[BatchPlan, ...]

    @property
    def batch_boundaries_digest(self) -> str:
        return digest_value(
            [
                {
                    "segment_id": batch.segment_id,
                    "batch_no": batch.batch_no,
                    "start_row": batch.start_row,
                    "end_row": batch.end_row,
                }
                for batch in self.batches
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "plan_digest": self.plan_digest,
            "batch_boundaries_digest": self.batch_boundaries_digest,
            "per_segment": list(self.per_segment),
            "batches": [batch.to_dict() for batch in self.batches],
        }


@dataclass(frozen=True)
class ExecutionPlan:
    spec_digest: str
    batch_rows: int
    queries: tuple[QueryPlan, ...]

    def query_map(self) -> dict[str, QueryPlan]:
        return {query_plan.query.id: query_plan for query_plan in self.queries}

    def batches(self) -> tuple[BatchPlan, ...]:
        return tuple(
            sorted(
                (batch for query_plan in self.queries for batch in query_plan.batches),
                key=lambda batch: batch.key,
            )
        )

    def to_dict(self, *, include_batches: bool = True) -> dict[str, Any]:
        return {
            "format_version": 1,
            "spec_digest": self.spec_digest,
            "batch_rows": self.batch_rows,
            "plan_summary_digest": digest_value(
                [query_plan.summary for query_plan in self.queries]
            ),
            "queries": [
                (
                    query_plan.to_dict()
                    if include_batches
                    else {
                        "summary": query_plan.summary,
                        "plan_digest": query_plan.plan_digest,
                        "batch_boundaries_digest": query_plan.batch_boundaries_digest,
                        "per_segment": list(query_plan.per_segment),
                    }
                )
                for query_plan in self.queries
            ],
        }


def _batch_ranges(rows: int, batch_rows: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < rows:
        end = min(rows, start + batch_rows)
        yield start, end
        start = end


def _query_summary(
    spec: ScanSpec,
    query: QuerySpec,
    segment_map: dict[str, SegmentSpec],
) -> dict[str, Any]:
    per_segment = []
    for segment_id in query.segments:
        segment = segment_map[segment_id]
        all_columns = tuple(sorted(column.name for column in segment.columns))
        projection = query.effective_projection()
        decode_columns = tuple(sorted(set(projection) | set(query.predicate_columns)))
        pruned = tuple(sorted(set(all_columns) - set(decode_columns)))
        per_segment.append(
            {
                "segment_id": segment.id,
                "all_columns": list(all_columns),
                "declared_projection_columns": list(query.project),
                "effective_projection_columns": list(projection),
                "predicate_columns": list(query.predicate_columns),
                "decode_columns": list(decode_columns),
                "pruned_columns": list(pruned),
            }
        )
    return {
        "query_id": query.id,
        "segments": list(query.segments),
        "projection_columns": list(query.project),
        "effective_projection_columns": list(query.effective_projection()),
        "predicate_columns": list(query.predicate_columns),
        "group_by": list(query.group_by),
        "aggregates": [aggregate.to_dict() for aggregate in query.aggregates],
        "predicate": query.predicate,
        "predicate_leaf_count": predicate_leaf_count(query.predicate),
        "per_segment": per_segment,
        "normalization": {
            "predicate": "flatten-and-sort; duplicate leaves removed",
            "projection": "sorted unique",
            "group_by": "sorted unique",
            "aggregates": "sorted by name",
            "batch_rows": "excluded from structural summary",
        },
    }


def compile_query_plan(
    spec: ScanSpec,
    query: QuerySpec,
    *,
    batch_rows: int | None = None,
) -> QueryPlan:
    settings = spec.execution
    effective_batch_rows = settings.batch_rows if batch_rows is None else batch_rows
    if effective_batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    segment_map = spec.segment_map()
    summary = _query_summary(spec, query, segment_map)
    plan_digest = digest_value(summary)
    batches: list[BatchPlan] = []
    for segment_id in query.segments:
        segment = segment_map[segment_id]
        all_columns = tuple(sorted(column.name for column in segment.columns))
        projection = query.effective_projection()
        decode_columns = tuple(sorted(set(projection) | set(query.predicate_columns)))
        pruned = tuple(sorted(set(all_columns) - set(decode_columns)))
        for batch_no, (start, end) in enumerate(_batch_ranges(segment.rows, effective_batch_rows)):
            batch_body = {
                "query_id": query.id,
                "segment_id": segment_id,
                "batch_no": batch_no,
                "start_row": start,
                "end_row": end,
                "decode_columns": list(decode_columns),
                "predicate": query.predicate,
                "predicate_leaf_count": predicate_leaf_count(query.predicate),
            }
            batches.append(
                BatchPlan(
                    query_id=query.id,
                    segment_id=segment_id,
                    batch_no=batch_no,
                    start_row=start,
                    end_row=end,
                    decode_columns=decode_columns,
                    predicate_columns=query.predicate_columns,
                    output_columns=projection,
                    pruned_columns=pruned,
                    predicate=query.predicate,
                    predicate_leaf_count=predicate_leaf_count(query.predicate),
                    all_columns=all_columns,
                    plan_digest=digest_value(batch_body),
                )
            )
    return QueryPlan(
        query=query,
        per_segment=tuple(summary["per_segment"]),
        summary=summary,
        plan_digest=plan_digest,
        batches=tuple(sorted(batches, key=lambda batch: (batch.segment_id, batch.batch_no))),
    )


def compile_plan(
    spec: ScanSpec,
    *,
    batch_rows: int | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        spec_digest=spec.digest,
        batch_rows=spec.execution.batch_rows if batch_rows is None else batch_rows,
        queries=tuple(
            compile_query_plan(spec, query, batch_rows=batch_rows)
            for query in sorted(spec.queries, key=lambda item: item.id)
        ),
    )


def plan_justification(plan: QueryPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in plan.per_segment:
        rows.append(
            {
                "segment_id": segment["segment_id"],
                "decode_columns": segment["decode_columns"],
                "pruned_columns": segment["pruned_columns"],
                "rules": {
                    "decode_columns": "effective projection union predicate reference columns",
                    "pruned_columns": "all segment columns minus decode columns",
                    "effective_projection": "declared project union aggregate source columns",
                },
                "sources": {
                    "declared_projection": segment["declared_projection_columns"],
                    "aggregate_columns": list(plan.query.aggregate_columns()),
                    "predicate_columns": segment["predicate_columns"],
                },
            }
        )
    return rows
