from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ENCODINGS = ("raw", "dictionary", "rle")
SEGMENT_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ColumnDef:
    name: str
    kind: str = "int"


@dataclass(frozen=True)
class RowGroup:
    id: str
    start_row: int
    row_count: int

    @property
    def end_row(self) -> int:
        return self.start_row + self.row_count


@dataclass(frozen=True)
class BlockSpec:
    id: str
    column: str
    row_group: str
    encoding: str
    compressed: bool
    values: tuple[int, ...]
    declared_pre_bytes: int | None = None
    declared_post_bytes: int | None = None

    def with_update(
        self,
        *,
        values: tuple[int, ...] | None = None,
        encoding: str | None = None,
        compressed: bool | None = None,
    ) -> "BlockSpec":
        return BlockSpec(
            id=self.id,
            column=self.column,
            row_group=self.row_group,
            encoding=encoding if encoding is not None else self.encoding,
            compressed=compressed if compressed is not None else self.compressed,
            values=values if values is not None else self.values,
            declared_pre_bytes=None,
            declared_post_bytes=None,
        )


@dataclass(frozen=True)
class QuerySpec:
    id: str
    select: tuple[str, ...]
    where: str | dict[str, Any] | None = None


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    columns: tuple[ColumnDef, ...]
    row_groups: tuple[RowGroup, ...]
    blocks: tuple[BlockSpec, ...]
    queries: tuple[QuerySpec, ...] = ()
    format_version: int = SEGMENT_FORMAT_VERSION

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def row_group_map(self) -> dict[str, RowGroup]:
        return {group.id: group for group in self.row_groups}

    @property
    def block_map(self) -> dict[str, BlockSpec]:
        return {block.id: block for block in self.blocks}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "segment_id": self.segment_id,
            "columns": [
                {"name": column.name, "type": column.kind} for column in self.columns
            ],
            "row_groups": [
                {
                    "id": group.id,
                    "start_row": group.start_row,
                    "row_count": group.row_count,
                }
                for group in self.row_groups
            ],
            "blocks": [
                {
                    "id": block.id,
                    "column": block.column,
                    "row_group": block.row_group,
                    "encoding": block.encoding,
                    "compressed": block.compressed,
                    "values": list(block.values),
                    "uncompressed_bytes": block.declared_pre_bytes,
                    "compressed_bytes": block.declared_post_bytes,
                }
                for block in self.blocks
            ],
            "queries": [
                {
                    "id": query.id,
                    "select": list(query.select),
                    "where": query.where,
                }
                for query in self.queries
            ],
        }


@dataclass(frozen=True)
class EncodedBlock:
    block_id: str
    column: str
    row_group: str
    encoding: str
    compressed: bool
    row_count: int
    pre_bytes: int
    post_bytes: int
    payload: bytes
    header: dict[str, Any]
    header_bytes: bytes
    block_digest: str
    record_bytes: bytes


@dataclass(frozen=True)
class BudgetConfig:
    global_parallelism: int = 3
    column_memory_budget_bytes: int = 4096
    segment_concurrency: int = 2

    def to_dict(self) -> dict[str, int]:
        return {
            "global_parallelism": self.global_parallelism,
            "column_memory_budget_bytes": self.column_memory_budget_bytes,
            "segment_concurrency": self.segment_concurrency,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BudgetConfig":
        return cls(
            global_parallelism=int(payload.get("global_parallelism", 3)),
            column_memory_budget_bytes=int(
                payload.get("column_memory_budget_bytes", 4096)
            ),
            segment_concurrency=int(payload.get("segment_concurrency", 2)),
        )


@dataclass(frozen=True)
class ScanTask:
    task_id: str
    query_id: str
    segment_id: str
    row_group_id: str
    block_id: str
    column: str
    block_ordinal: int
    decode_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query_id": self.query_id,
            "segment_id": self.segment_id,
            "row_group_id": self.row_group_id,
            "block_id": self.block_id,
            "column": self.column,
            "block_ordinal": self.block_ordinal,
            "decode_memory_bytes": self.decode_memory_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScanTask":
        return cls(
            task_id=str(payload["task_id"]),
            query_id=str(payload["query_id"]),
            segment_id=str(payload["segment_id"]),
            row_group_id=str(payload["row_group_id"]),
            block_id=str(payload["block_id"]),
            column=str(payload["column"]),
            block_ordinal=int(payload["block_ordinal"]),
            decode_memory_bytes=int(payload["decode_memory_bytes"]),
        )


@dataclass(frozen=True)
class QueryPlan:
    query_id: str
    select: tuple[str, ...]
    predicate_columns: tuple[str, ...]
    candidate_blocks: tuple[str, ...]
    pruned_columns: tuple[str, ...]
    row_groups: tuple[dict[str, Any], ...]
    basis: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "select": list(self.select),
            "predicate_columns": list(self.predicate_columns),
            "candidate_blocks": list(self.candidate_blocks),
            "pruned_columns": list(self.pruned_columns),
            "row_groups": list(self.row_groups),
            "basis": list(self.basis),
            "plan_digest": self.digest,
        }


@dataclass(frozen=True)
class TaskResult:
    task: ScanTask
    completed_rows: int
    decode_memory_bytes: int
    queue_started_ms: float
    wait_ended_ms: float
    run_started_ms: float
    run_ended_ms: float
    duration_ms: float
    block_digest: str
    result_digest: str
    result_json: dict[str, Any]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.task.to_dict(),
            "completed_rows": self.completed_rows,
            "decode_memory_bytes": self.decode_memory_bytes,
            "queue_started_ms": self.queue_started_ms,
            "wait_ended_ms": self.wait_ended_ms,
            "run_started_ms": self.run_started_ms,
            "run_ended_ms": self.run_ended_ms,
            "duration_ms": self.duration_ms,
            "block_digest": self.block_digest,
            "result_digest": self.result_digest,
            "result_json": self.result_json,
            "stats": self.stats,
        }
