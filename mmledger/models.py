"""Domain model shared by parsing, normalization, scheduling, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .jsonutil import MISSING, MissingField, hash_json, is_missing, json_ready


R1 = "r1"
R2 = "r2"
R3 = "r3"
VERSIONS = (R1, R2, R3)

BASE_FIELDS = ("batch_id", "request_id", "output_digest")
VERSION_FIELDS: dict[str, tuple[str, ...]] = {
    R1: BASE_FIELDS,
    R2: BASE_FIELDS + ("parameter_version", "duration_ms", "worker_id"),
    R3: BASE_FIELDS
    + (
        "parameter_version",
        "duration_ms",
        "worker_id",
        "stable_rank",
        "overwritten_ref",
        "parameter_digest",
    ),
}
ALL_RECEIPT_FIELDS = VERSION_FIELDS[R3]

IMAGE_TEXT = "image_text"
IMAGE_ONLY = "image_only"
REQUEST_KINDS = (IMAGE_TEXT, IMAGE_ONLY)


@dataclass(frozen=True)
class Receipt:
    batch_id: str
    request_id: str
    output_digest: str
    parameter_version: int | MissingField = MISSING
    duration_ms: int | MissingField = MISSING
    worker_id: str | MissingField = MISSING
    stable_rank: int | MissingField = MISSING
    overwritten_ref: str | None | MissingField = MISSING
    parameter_digest: str | MissingField = MISSING
    format_version: str = R3
    source_version: str = R3
    uid: str = ""
    source_file: str = ""
    source_index: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def missing_fields(self) -> tuple[str, ...]:
        names = []
        for name in ALL_RECEIPT_FIELDS:
            if is_missing(getattr(self, name)):
                names.append(name)
        return tuple(names)

    @property
    def completeness(self) -> str:
        return "complete" if not self.missing_fields else "partial"

    @property
    def delivery_key(self) -> str:
        worker = "-" if is_missing(self.worker_id) else str(self.worker_id)
        rank = "-" if is_missing(self.stable_rank) else str(self.stable_rank)
        return f"{self.batch_id}:{self.request_id}:{worker}:{rank}"

    @property
    def log_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.batch_id,
            self.request_id,
            "-" if is_missing(self.worker_id) else str(self.worker_id),
            "-" if is_missing(self.stable_rank) else str(self.stable_rank),
            self.uid,
        )

    def as_fields(self, include_all: bool = True) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        names = ALL_RECEIPT_FIELDS if include_all else VERSION_FIELDS[self.format_version]
        for name in names:
            fields[name] = json_ready(getattr(self, name))
        return fields

    def as_ledger_dict(self) -> dict[str, Any]:
        payload = self.as_fields(include_all=True)
        payload.update(
            {
                "format_version": self.format_version,
                "source_version": self.source_version,
                "missing_fields": list(self.missing_fields),
                "completeness": self.completeness,
                "receipt_uid": self.uid,
                "source_file": self.source_file,
                "source_index": self.source_index,
            }
        )
        if self.provenance:
            payload["provenance"] = json_ready(dict(self.provenance))
        return payload

    def version_payload(self) -> dict[str, Any]:
        payload = {"version": self.format_version}
        payload.update(self.as_fields(include_all=False))
        return payload

    def with_uid(self, uid: str) -> "Receipt":
        return replace(self, uid=uid)

    def field(self, name: str) -> Any:
        if name not in ALL_RECEIPT_FIELDS:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    parameter_version: int
    parameter_digest: str
    quota_points: int | None = None


@dataclass(frozen=True)
class ParameterSpec:
    parameter_version: int
    digest: str


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int
    hidden_dim: int
    quantization_bits: int

    @property
    def base_parameters(self) -> int:
        # Two deterministic affine projections used by the local worker.
        return self.input_dim * self.hidden_dim + self.hidden_dim * 2

    @property
    def macs_per_request(self) -> int:
        return 2 * self.base_parameters

    @property
    def quantized_parameter_units(self) -> int:
        return (self.base_parameters * self.quantization_bits + 7) // 8


@dataclass(frozen=True)
class LimitsSpec:
    batch_size: int
    normalization_budget: int
    worker_quota_points: int


@dataclass(frozen=True)
class MergeRules:
    worker_sets: Mapping[str, tuple[str, ...]]
    parameter_version_sets: Mapping[str, tuple[int, ...]]
    active_worker_set: str
    active_parameter_version_set: str
    worker_priority: tuple[str, ...]

    def worker_priority_index(self, worker_id: str) -> int:
        try:
            return self.worker_priority.index(worker_id)
        except ValueError:
            return len(self.worker_priority) + 10_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_sets": {k: list(v) for k, v in self.worker_sets.items()},
            "parameter_version_sets": {
                k: list(v) for k, v in self.parameter_version_sets.items()
            },
            "active_worker_set": self.active_worker_set,
            "active_parameter_version_set": self.active_parameter_version_set,
            "worker_priority": list(self.worker_priority),
        }


@dataclass(frozen=True)
class RequestSpec:
    sequence: int
    request_id: str
    kind: str
    arrival_ms: int
    batch_id: str
    worker_id: str
    input_seed: int
    overwritten_ref: str | None = None

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.sequence, self.batch_id, self.request_id)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "sequence": self.sequence,
            "request_id": self.request_id,
            "kind": self.kind,
            "arrival_ms": self.arrival_ms,
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "input_seed": self.input_seed,
        }
        if self.overwritten_ref is not None:
            out["overwritten_ref"] = self.overwritten_ref
        return out


@dataclass(frozen=True)
class ConfigBundle:
    config_path: str
    root: str
    raw_config: Mapping[str, Any]
    raw_requests: Mapping[str, Any]
    raw_merge_rules: Mapping[str, Any]
    seed: int
    model: ModelSpec
    limits: LimitsSpec
    workers: tuple[WorkerSpec, ...]
    parameters: tuple[ParameterSpec, ...]
    requests: tuple[RequestSpec, ...]
    merge_rules: MergeRules
    receipt_glob: str
    fault_injections_path: str
    fault_receipt_glob: str
    config_hash: str
    requests_hash: str
    merge_rules_hash: str

    @property
    def workers_by_id(self) -> dict[str, WorkerSpec]:
        return {worker.worker_id: worker for worker in self.workers}

    @property
    def parameters_by_version(self) -> dict[int, ParameterSpec]:
        return {parameter.parameter_version: parameter for parameter in self.parameters}

    @property
    def requests_by_id(self) -> dict[str, tuple[RequestSpec, ...]]:
        grouped: dict[str, list[RequestSpec]] = {}
        for request in self.requests:
            grouped.setdefault(request.request_id, []).append(request)
        return {key: tuple(value) for key, value in grouped.items()}

    @property
    def batches(self) -> dict[str, tuple[RequestSpec, ...]]:
        grouped: dict[str, list[RequestSpec]] = {}
        for request in self.requests:
            grouped.setdefault(request.batch_id, []).append(request)
        return {
            key: tuple(sorted(value, key=lambda req: (req.request_id, req.sequence)))
            for key, value in grouped.items()
        }

    @property
    def batch_worker(self) -> dict[str, str]:
        return {batch_id: requests[0].worker_id for batch_id, requests in sorted(self.batches.items())}

    @property
    def state_hash(self) -> str:
        return hash_json(
            {
                "config_hash": self.config_hash,
                "requests_hash": self.requests_hash,
                "merge_rules_hash": self.merge_rules_hash,
            }
        )


@dataclass(frozen=True)
class CandidateDecision:
    receipt_uid: str
    batch_id: str
    worker_id: str | MissingField
    parameter_version: int | MissingField
    duration_ms: int | MissingField
    stable_rank: int | MissingField
    output_digest: str

    @classmethod
    def from_receipt(cls, receipt: Receipt) -> "CandidateDecision":
        return cls(
            receipt_uid=receipt.uid,
            batch_id=receipt.batch_id,
            worker_id=receipt.worker_id,
            parameter_version=receipt.parameter_version,
            duration_ms=receipt.duration_ms,
            stable_rank=receipt.stable_rank,
            output_digest=receipt.output_digest,
        )

    def as_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "receipt_uid": self.receipt_uid,
                "batch_id": self.batch_id,
                "worker_id": self.worker_id,
                "parameter_version": self.parameter_version,
                "duration_ms": self.duration_ms,
                "stable_rank": self.stable_rank,
                "output_digest": self.output_digest,
            }
        )


@dataclass(frozen=True)
class RankConflict:
    batch_id: str
    request_id: str
    worker_a: str
    worker_b: str
    rank_a: int
    rank_b: int
    winner_worker: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "worker_a": self.worker_a,
            "worker_b": self.worker_b,
            "rank_a": self.rank_a,
            "rank_b": self.rank_b,
            "winner_worker": self.winner_worker,
            "rule": "worker_id_ascending",
        }


@dataclass(frozen=True)
class BatchNormalized:
    batch_id: str
    worker_id: str
    param_version: int | MissingField
    receipts: tuple[Receipt, ...]
    candidates: tuple[Receipt, ...]
    conflicts: tuple[RankConflict, ...]
    input_hash: str
    cache_hit: bool = False
    task_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": json_ready(self.param_version),
            "receipt_uids": [receipt.uid for receipt in self.receipts],
            "candidate_uids": [receipt.uid for receipt in self.candidates],
            "conflicts": [conflict.as_dict() for conflict in self.conflicts],
            "input_hash": self.input_hash,
            "cache_hit": self.cache_hit,
            "task_key": self.task_key,
        }


@dataclass(frozen=True)
class OwnershipDecision:
    request_id: str
    request_sequence: int
    kind: str
    batch_id: str
    worker_id: str
    authoritative_uid: str
    candidates: tuple[CandidateDecision, ...]
    basis: Mapping[str, Any]
    rank_conflicts: tuple[RankConflict, ...] = ()
    duration_ms: int | MissingField = MISSING
    parameter_version: int | MissingField = MISSING
    output_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "request_id": self.request_id,
                "request_sequence": self.request_sequence,
                "kind": self.kind,
                "batch_id": self.batch_id,
                "worker_id": self.worker_id,
                "authoritative_uid": self.authoritative_uid,
                "duration_ms": self.duration_ms,
                "parameter_version": self.parameter_version,
                "output_digest": self.output_digest,
                "candidates": [candidate.as_dict() for candidate in self.candidates],
                "basis": dict(self.basis),
                "rank_conflicts": [conflict.as_dict() for conflict in self.rank_conflicts],
            }
        )

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.batch_id, self.request_id, self.worker_id, self.authoritative_uid)


@dataclass(frozen=True)
class BatchResource:
    batch_id: str
    worker_id: str
    parameter_version: int | MissingField
    request_count: int
    duration_ms: int | MissingField
    duration_observed_count: int
    duration_missing_count: int
    macs: int
    quantized_macs: int
    quantized_parameter_units: int
    utilization_ppm: int | MissingField
    cache_hit: bool = False
    task_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "batch_id": self.batch_id,
                "worker_id": self.worker_id,
                "parameter_version": self.parameter_version,
                "request_count": self.request_count,
                "duration_ms": self.duration_ms,
                "duration_observed_count": self.duration_observed_count,
                "duration_missing_count": self.duration_missing_count,
                "macs": self.macs,
                "quantized_macs": self.quantized_macs,
                "quantized_parameter_units": self.quantized_parameter_units,
                "utilization_ppm": self.utilization_ppm,
                "cache_hit": self.cache_hit,
                "task_key": self.task_key,
            }
        )


@dataclass(frozen=True)
class ScheduleTicket:
    sequence: int
    batch_id: str
    worker_id: str
    points: int
    priority_pending: bool
    budget_before: int
    budget_after: int
    worker_used_before: int
    worker_used_after: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "points": self.points,
            "priority_pending": self.priority_pending,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "worker_used_before": self.worker_used_before,
            "worker_used_after": self.worker_used_after,
        }


@dataclass(frozen=True)
class ScheduleResult:
    tickets: tuple[ScheduleTicket, ...]
    pending_batches: tuple[str, ...]
    quota_blocked_batches: tuple[str, ...]
    consumed_budget: int
    trace: tuple[Mapping[str, Any], ...]
    failures: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tickets": [ticket.as_dict() for ticket in self.tickets],
            "pending_batches": list(self.pending_batches),
            "quota_blocked_batches": list(self.quota_blocked_batches),
            "consumed_budget": self.consumed_budget,
            "trace": [dict(item) for item in self.trace],
            "failures": [failure.to_dict() for failure in self.failures],
        }
