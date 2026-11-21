"""Immutable-ish domain models shared by loader, engine, ledger, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class ShardingSpec:
    target_rows: int
    remainder_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_rows": self.target_rows,
            "remainder_policy": self.remainder_policy,
        }


@dataclass(frozen=True)
class StageSpec:
    id: str
    type: str
    version: int
    upstream: str | None
    params: dict[str, int]
    ordinal: int = 0
    config_digest: str = ""
    input_digest: str = ""
    output_digest: str = ""
    row_count: int = 0

    def with_ordinal(self, ordinal: int, config_digest: str) -> "StageSpec":
        return StageSpec(
            id=self.id,
            type=self.type,
            version=self.version,
            upstream=self.upstream,
            params=dict(self.params),
            ordinal=ordinal,
            config_digest=config_digest,
        )

    def config_object(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "version": self.version,
            "upstream": self.upstream,
            "params": {key: int(self.params[key]) for key in sorted(self.params)},
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.config_object()
        payload["ordinal"] = self.ordinal
        payload["config_digest"] = self.config_digest
        return payload


@dataclass(frozen=True)
class PipelineSpec:
    fixed_scale: int
    sharding: ShardingSpec
    stages: tuple[StageSpec, ...]
    canonical: dict[str, Any]
    digest: str

    @property
    def stage_by_id(self) -> dict[str, StageSpec]:
        return {stage.id: stage for stage in self.stages}

    @property
    def shard_stage(self) -> StageSpec:
        return self.stages[-1]


@dataclass
class ColumnData:
    source_id: str
    name: str
    kind: str
    values: list[int | Decimal | None]
    nulls: list[bool]

    @property
    def key(self) -> str:
        return f"{self.source_id}.{self.name}"


@dataclass
class SourceData:
    id: str
    rows: int
    columns: list[ColumnData]
    start_row: int


@dataclass
class DatasetSpec:
    sources: tuple[SourceData, ...]
    columns: tuple[ColumnData, ...]
    row_count: int
    canonical: dict[str, Any]
    digest: str
    seed: int | None = None


@dataclass
class PreparedSpec:
    dataset: DatasetSpec
    pipeline: PipelineSpec
    fingerprint: str

    @property
    def dataset_digest(self) -> str:
        return self.dataset.digest

    @property
    def pipeline_digest(self) -> str:
        return self.pipeline.digest


@dataclass
class TableColumn:
    key: str
    type: str
    values: list[int | Decimal | None]
    nulls: list[bool]


@dataclass
class Table:
    row_count: int
    columns: list[TableColumn]

    def clone(self) -> "Table":
        return Table(
            row_count=self.row_count,
            columns=[
                TableColumn(col.key, col.type, list(col.values), list(col.nulls))
                for col in self.columns
            ],
        )


@dataclass(frozen=True)
class BatchObservation:
    batch_index: int
    start_row: int
    end_row: int
    row_count: int
    input_digest: str
    output_digest: str


@dataclass
class StageComputation:
    stage: StageSpec
    input_digest: str
    output_digest: str
    row_count: int
    batches: list[BatchObservation]
    shards: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    status: str
    spec_version: int
    fingerprint: str
    dataset_digest: str
    stage_statuses: list[dict[str, Any]]
    shards: list[dict[str, Any]]
    final_data_digest: str
    elapsed_ms: int
    evolution: dict[str, Any] | None = None
