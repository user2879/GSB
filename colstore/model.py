"""Domain dataclasses: schema, segment definitions, tasks, results, stats."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Encoding(str, Enum):
    RAW = "raw"
    DICTIONARY = "dictionary"
    RLE = "rle"


class ColumnType(str, Enum):
    INT32 = "int32"
    INT64 = "int64"

    @property
    def width(self) -> int:
        return 4 if self is ColumnType.INT32 else 8

    @property
    def numpy_dtype(self) -> str:
        return "<i4" if self is ColumnType.INT32 else "<i8"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: ColumnType
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype.value, "compressed": self.compressed}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ColumnSpec":
        return cls(
            name=raw["name"],
            dtype=ColumnType(raw["dtype"]),
            compressed=bool(raw.get("compressed", False)),
        )


@dataclass(frozen=True)
class BlockSpec:
    """Per (row group, column) encoding instruction inside a segment definition."""

    row_group: int
    column: str
    encoding: Optional[Encoding]  # None => deterministic automatic selection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_group": self.row_group,
            "column": self.column,
            "encoding": None if self.encoding is None else self.encoding.value,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BlockSpec":
        enc = raw.get("encoding")
        return cls(
            row_group=int(raw["row_group"]),
            column=raw["column"],
            encoding=None if enc is None else Encoding(enc),
        )


@dataclass
class SegmentDefinition:
    """In-memory representation of a segment definition JSON file."""

    name: str
    columns: List[ColumnSpec]
    row_group_rows: List[int]
    values: Dict[str, List[int]]
    block_specs: List[BlockSpec] = field(default_factory=list)
    format_version: int = 1

    @property
    def total_rows(self) -> int:
        return sum(self.row_group_rows)

    @property
    def num_row_groups(self) -> int:
        return len(self.row_group_rows)

    def column(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def row_group_bounds(self) -> List[Tuple[int, int]]:
        bounds: List[Tuple[int, int]] = []
        start = 0
        for size in self.row_group_rows:
            bounds.append((start, start + size))
            start += size
        return bounds

    def block_encoding(self, row_group: int, column: str) -> Optional[Encoding]:
        for spec in self.block_specs:
            if spec.row_group == row_group and spec.column == column:
                return spec.encoding
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "format_version": self.format_version,
            "columns": [c.to_dict() for c in self.columns],
            "row_group_rows": list(self.row_group_rows),
            "values": {k: list(v) for k, v in self.values.items()},
            "block_specs": [b.to_dict() for b in self.block_specs],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SegmentDefinition":
        definition = cls(
            name=raw["name"],
            columns=[ColumnSpec.from_dict(c) for c in raw["columns"]],
            row_group_rows=[int(x) for x in raw["row_group_rows"]],
            values={k: [int(v) for v in vals] for k, vals in raw["values"].items()},
            block_specs=[BlockSpec.from_dict(b) for b in raw.get("block_specs", [])],
            format_version=int(raw.get("format_version", 1)),
        )
        definition.validate()
        return definition

    def validate(self) -> None:
        """Local structural validation; semantic corruption is caught at load."""
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("duplicate column names in segment definition")
        for col in self.columns:
            if col.name not in self.values:
                raise ValueError(f"missing sample values for column {col.name}")
            if len(self.values[col.name]) != self.total_rows:
                raise ValueError(
                    f"column {col.name} has {len(self.values[col.name])} values, "
                    f"expected {self.total_rows}"
                )
        if not self.row_group_rows or any(n <= 0 for n in self.row_group_rows):
            raise ValueError("row_group_rows must be non-empty positive integers")
        for spec in self.block_specs:
            if not (0 <= spec.row_group < self.num_row_groups):
                raise ValueError(f"block spec row group out of range: {spec.row_group}")
            if spec.column not in names:
                raise ValueError(f"block spec references unknown column: {spec.column}")
            if spec.encoding is not None and not isinstance(spec.encoding, Encoding):
                raise ValueError("block spec encoding must be raw|dictionary|rle|null")


@dataclass(frozen=True)
class BlockId:
    """Stable coordinate of one encoded block inside a segment."""

    segment: str
    row_group: int
    column: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BlockId":
        return cls(segment=raw["segment"], row_group=int(raw["row_group"]), column=raw["column"])


@dataclass
class BudgetParams:
    """Triple scheduling constraint. Versioned monotonically by the engine."""

    max_parallelism: int = 4
    memory_budget_bytes: int = 1 << 20
    per_segment_concurrency: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BudgetParams":
        return cls(
            max_parallelism=int(raw["max_parallelism"]),
            memory_budget_bytes=int(raw["memory_budget_bytes"]),
            per_segment_concurrency=int(raw["per_segment_concurrency"]),
        )


@dataclass
class TaskRecord:
    """Full lifecycle record of one block-level scan task."""

    task_id: int
    block: BlockId
    encoding: Encoding
    memory_bytes: int
    rows: int
    matched_rows: int = 0
    enqueue_time_ms: float = 0.0
    wait_start_ms: float = 0.0
    start_ms: float = 0.0
    end_ms: float = 0.0
    status: str = "queued"  # queued|running|done|abandoned
    budget_version: int = 0
    thread_name: str = ""

    @property
    def wait_ms(self) -> float:
        if self.status == "queued":
            return 0.0
        return max(0.0, self.start_ms - self.wait_start_ms)

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["block"] = self.block.to_dict()
        data["encoding"] = self.encoding.value
        data["wait_ms"] = self.wait_ms
        return data

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskRecord":
        return cls(
            task_id=int(raw["task_id"]),
            block=BlockId.from_dict(raw["block"]),
            encoding=Encoding(raw["encoding"]),
            memory_bytes=int(raw["memory_bytes"]),
            rows=int(raw["rows"]),
            matched_rows=int(raw.get("matched_rows", 0)),
            enqueue_time_ms=float(raw.get("enqueue_time_ms", 0.0)),
            wait_start_ms=float(raw.get("wait_start_ms", 0.0)),
            start_ms=float(raw.get("start_ms", 0.0)),
            end_ms=float(raw.get("end_ms", 0.0)),
            status=raw.get("status", "done"),
            budget_version=int(raw.get("budget_version", 0)),
            thread_name=raw.get("thread_name", ""),
        )


@dataclass
class ScanStats:
    """Aggregate statistics of one scan generation."""

    budget_version: int
    encoding_selection_version: int
    segment_format_versions: Dict[str, int]
    tasks_total: int = 0
    tasks_done: int = 0
    tasks_abandoned: int = 0
    blocks_scanned: int = 0
    blocks_skipped_by_pushdown: int = 0
    columns_pruned: List[str] = field(default_factory=list)
    rows_completed: int = 0
    rows_matched: int = 0
    peak_memory_bytes: int = 0
    total_memory_reserved_ms: float = 0.0
    elapsed_ms: float = 0.0
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScanStats":
        return cls(**raw)


@dataclass
class ScanResult:
    """Materialized, column-ordered result set plus planning/tracing metadata."""

    result_id: str
    query_id: str
    budget_version: int
    encoding_selection_version: int
    selected_columns: List[str]
    rows: List[Tuple[Any, ...]]
    stats: ScanStats
    plan_summary: Dict[str, Any]
    tasks: List[TaskRecord]
    segment_versions: Dict[str, int]
    result_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "query_id": self.query_id,
            "budget_version": self.budget_version,
            "encoding_selection_version": self.encoding_selection_version,
            "selected_columns": list(self.selected_columns),
            "rows": [list(row) for row in self.rows],
            "stats": self.stats.to_dict(),
            "plan_summary": self.plan_summary,
            "tasks": [t.to_dict() for t in self.tasks],
            "segment_versions": dict(self.segment_versions),
            "result_hash": self.result_hash,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScanResult":
        return cls(
            result_id=raw["result_id"],
            query_id=raw["query_id"],
            budget_version=int(raw["budget_version"]),
            encoding_selection_version=int(raw["encoding_selection_version"]),
            selected_columns=list(raw["selected_columns"]),
            rows=[tuple(r) for r in raw["rows"]],
            stats=ScanStats.from_dict(raw["stats"]),
            plan_summary=dict(raw["plan_summary"]),
            tasks=[TaskRecord.from_dict(t) for t in raw["tasks"]],
            segment_versions=dict(raw["segment_versions"]),
            result_hash=raw.get("result_hash", ""),
        )
