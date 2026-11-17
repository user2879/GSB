"""Normalized domain model shared by every storage format."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .util import sha256_json

CURRENT_FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
AGGREGATION_VERSION = "agg-v1"
AGGREGATION_NAMES = ("last", "mean_last_5", "mean")


@dataclass(frozen=True)
class Aggregation:
    name: str
    version: str = AGGREGATION_VERSION

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


LAST_AGGREGATION = Aggregation("last")


@dataclass(frozen=True)
class Parameters:
    seed: int
    model_dim: int
    learning_rate: float
    hyperparameters: dict[str, Any]

    @property
    def steps(self) -> int:
        return int(self.hyperparameters["steps"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model_dim": self.model_dim,
            "learning_rate": self.learning_rate,
            "hyperparameters": dict(self.hyperparameters),
        }


@dataclass(frozen=True)
class MetricPoint:
    step: int
    loss: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "loss": self.loss, "accuracy": self.accuracy}


@dataclass(frozen=True)
class Provenance:
    source_format_version: int
    upgraded: bool = False
    original_digest: str | None = None
    aggregation_defaulted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format_version": self.source_format_version,
            "upgraded": self.upgraded,
            "original_digest": self.original_digest,
            "aggregation_defaulted": self.aggregation_defaulted,
        }


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_version: int
    parameters: Parameters
    step_metrics: tuple[MetricPoint, ...]
    timestamp: int
    aggregation: Aggregation
    parent_run_id: str | None
    final_metrics: dict[str, float]
    format_version: int = CURRENT_FORMAT_VERSION
    provenance: Provenance = field(
        default_factory=lambda: Provenance(CURRENT_FORMAT_VERSION)
    )

    @property
    def key(self) -> str:
        return f"{self.run_id}@v{self.run_version}"

    def common_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_version": self.run_version,
            "parameters": self.parameters.to_dict(),
            "final_metrics": dict(self.final_metrics),
        }

    def to_storage_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "format_version": self.format_version,
            **self.common_dict(),
        }
        if self.format_version >= 2:
            record["step_metrics"] = [point.to_dict() for point in self.step_metrics]
            record["timestamp"] = self.timestamp
        if self.format_version >= 3:
            record["aggregation"] = self.aggregation.to_dict()
            record["parent_run_id"] = self.parent_run_id
            record["provenance"] = self.provenance.to_dict()
        return record

    def to_normalized_dict(self) -> dict[str, Any]:
        return {
            "format_version": CURRENT_FORMAT_VERSION,
            "run_id": self.run_id,
            "run_version": self.run_version,
            "parameters": self.parameters.to_dict(),
            "step_metrics": [point.to_dict() for point in self.step_metrics],
            "timestamp": self.timestamp,
            "aggregation": self.aggregation.to_dict(),
            "parent_run_id": self.parent_run_id,
            "final_metrics": dict(self.final_metrics),
            "provenance": self.provenance.to_dict(),
        }

    def content_digest(self) -> str:
        return sha256_json(self.to_normalized_dict())

    def upgraded(self, *, original_digest: str | None = None) -> "RunRecord":
        if self.format_version >= CURRENT_FORMAT_VERSION:
            return self
        provenance = Provenance(
            source_format_version=self.provenance.source_format_version,
            upgraded=True,
            original_digest=original_digest or self.provenance.original_digest,
            aggregation_defaulted=self.provenance.aggregation_defaulted,
        )
        return replace(
            self,
            format_version=CURRENT_FORMAT_VERSION,
            provenance=provenance,
        )


@dataclass
class Library:
    library_id: str
    runs: list[RunRecord]
    history: dict[str, list[RunRecord]] = field(default_factory=dict)

    def sorted_runs(self) -> list[RunRecord]:
        return sorted(self.runs, key=lambda record: record.run_id)

    def current_map(self) -> dict[str, RunRecord]:
        return {record.run_id: record for record in self.runs}

    def all_records(self) -> Iterable[RunRecord]:
        for record in self.sorted_runs():
            yield record
            for historical in sorted(
                self.history.get(record.run_id, []),
                key=lambda item: item.run_version,
            ):
                yield historical

    def history_for(self, run_id: str) -> list[RunRecord]:
        return sorted(
            self.history.get(run_id, []), key=lambda record: record.run_version
        )

    def to_envelope(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        for run_id in sorted(self.history):
            for record in sorted(
                self.history[run_id], key=lambda item: item.run_version
            ):
                history.append(record.to_storage_dict())
        return {
            "library_schema": 1,
            "library_id": self.library_id,
            "runs": [record.to_storage_dict() for record in self.sorted_runs()],
            "history": history,
        }

    def content_digests(self) -> dict[str, str]:
        return {
            record.run_id: record.content_digest()
            for record in self.sorted_runs()
        }

    def format_versions(self) -> list[int]:
        return sorted({record.format_version for record in self.all_records()})

    def source_format_versions(self) -> list[int]:
        return sorted(
            {record.provenance.source_format_version for record in self.all_records()}
        )

    def aggregation_versions(self) -> list[str]:
        return sorted({record.aggregation.version for record in self.all_records()})

