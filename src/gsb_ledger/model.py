"""Canonical models shared by all stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class _Missing:
    __slots__ = ("field_name",)

    def __init__(self, field_name: str = "") -> None:
        self.field_name = field_name

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING" if not self.field_name else f"MISSING({self.field_name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Missing) and self.field_name == other.field_name

    def __hash__(self) -> int:
        return hash(("__missing__", self.field_name))

    def to_jsonable(self) -> dict[str, str]:
        return {"__missing__": self.field_name or "unknown"}


MISSING = _Missing()


def missing(field_name: str) -> _Missing:
    return _Missing(field_name)


def is_missing(value: object) -> bool:
    return isinstance(value, _Missing)


def jsonable(value: Any) -> Any:
    """Convert the internal missing marker to an explicit JSON object."""

    if isinstance(value, _Missing):
        return value.to_jsonable()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    param_version: int
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    quant_scale: int
    known_param_versions: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "port": self.port,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "quant_scale": self.quant_scale,
            "known_param_versions": list(self.known_param_versions or (self.param_version,)),
        }


@dataclass(frozen=True)
class MergeEntry:
    worker_id: str
    param_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "param_version": self.param_version,
        }


@dataclass(frozen=True)
class ReceiptSource:
    path: str
    default_worker_id: str | None
    default_param_version: int | None
    format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "default_worker_id": self.default_worker_id,
            "default_param_version": self.default_param_version,
            "format": self.format,
        }


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    workers: tuple[WorkerSpec, ...]
    port_min: int
    port_max: int
    batch_size: int
    fixed_point_bits: int
    throughput_ops_per_ms: int
    merge_entries: tuple[MergeEntry, ...]
    tie_break: tuple[str, ...]
    receipt_sources: tuple[ReceiptSource, ...]
    interaction_budget_ms: int
    fault_config: str | None
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def worker_by_id(self) -> dict[str, WorkerSpec]:
        return {worker.worker_id: worker for worker in self.workers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workers": [worker.to_dict() for worker in self.workers],
            "port_range": {"min": self.port_min, "max": self.port_max},
            "batch_size": self.batch_size,
            "fixed_point_bits": self.fixed_point_bits,
            "throughput_ops_per_ms": self.throughput_ops_per_ms,
            "merge_rule": {
                "entries": [entry.to_dict() for entry in self.merge_entries],
                "tie_break": list(self.tie_break),
            },
            "receipt_sources": [source.to_dict() for source in self.receipt_sources],
            "interaction_budget_ms": self.interaction_budget_ms,
            "fault_config": self.fault_config,
        }


@dataclass(frozen=True)
class RequestSpec:
    seq: int
    request_id: str
    arrival_ms: int
    prompt: str = ""
    payload: tuple[int, ...] = ()
    source_path: str = ""
    source_line: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seq": self.seq,
            "request_id": self.request_id,
            "arrival_ms": self.arrival_ms,
            "prompt": self.prompt,
        }
        if self.payload:
            result["payload"] = list(self.payload)
        return result


@dataclass(frozen=True)
class Receipt:
    version: str
    batch_id: str
    request_id: str
    output_digest: str
    param_version: int | _Missing = MISSING
    duration_ms: int | _Missing = MISSING
    worker_id: str | _Missing = MISSING
    rank: int | _Missing = MISSING
    covered_refs: tuple[str, ...] | _Missing = MISSING
    param_digest: str | _Missing = MISSING
    missing_fields: tuple[str, ...] = ()
    completeness: str = "complete"
    source_path: str = ""
    source_line: int = 0
    default_worker_id: str | None = None
    default_param_version: int | None = None
    injection_ids: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def key(self) -> str:
        return f"{self.batch_id}/{self.request_id}"

    @property
    def resolved_worker_id(self) -> str | None:
        if not is_missing(self.worker_id):
            return str(self.worker_id)
        return self.default_worker_id

    @property
    def resolved_param_version(self) -> int | None:
        if not is_missing(self.param_version):
            return int(self.param_version)
        return self.default_param_version

    def with_fields(self, **updates: Any) -> "Receipt":
        values = {
            "version": self.version,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "param_version": self.param_version,
            "duration_ms": self.duration_ms,
            "worker_id": self.worker_id,
            "rank": self.rank,
            "covered_refs": self.covered_refs,
            "param_digest": self.param_digest,
            "missing_fields": self.missing_fields,
            "completeness": self.completeness,
            "source_path": self.source_path,
            "source_line": self.source_line,
            "default_worker_id": self.default_worker_id,
            "default_param_version": self.default_param_version,
            "injection_ids": self.injection_ids,
            "raw": self.raw,
        }
        values.update(updates)
        return Receipt(**values)

    def to_dict(self, *, include_provenance: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": self.version,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "param_version": jsonable(self.param_version),
            "duration_ms": jsonable(self.duration_ms),
            "worker_id": jsonable(self.worker_id),
            "rank": jsonable(self.rank),
            "covered_refs": jsonable(self.covered_refs),
            "param_digest": jsonable(self.param_digest),
            "missing_fields": list(self.missing_fields),
            "completeness": self.completeness,
            "injection_ids": list(self.injection_ids),
        }
        if include_provenance:
            result["provenance"] = {
                "path": self.source_path,
                "line": self.source_line,
                "default_worker_id": self.default_worker_id,
                "default_param_version": self.default_param_version,
            }
        return result


def stable_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)
