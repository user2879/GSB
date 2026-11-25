from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    param_version: int
    quota: int
    batch_size: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    worker_id: str
    arrival_seq: int
    vector: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["vector"] = list(self.vector)
        return data


@dataclass(frozen=True)
class MergeRules:
    default_batch_size: int = 2
    global_concurrency: int = 2
    lease_ttl_ms: int = 5_000
    lease_renew_ms: int = 1_000
    digest_algorithm: str = "sha256-v1"


@dataclass(frozen=True)
class SpecBundle:
    workers: tuple[WorkerSpec, ...]
    requests: tuple[RequestSpec, ...]
    rules: MergeRules
    source_dir: str = "."

    def worker_map(self) -> dict[str, WorkerSpec]:
        return {worker.worker_id: worker for worker in self.workers}

    def request_map(self) -> dict[str, RequestSpec]:
        return {request.request_id: request for request in self.requests}


@dataclass(frozen=True)
class Batch:
    batch_id: str
    worker_id: str
    batch_index: int
    arrival_seqs: tuple[int, ...]
    request_ids: tuple[str, ...]
    stable_ranks: dict[str, int] = field(default_factory=dict)

    def material(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "batch_index": self.batch_index,
            "arrival_seqs": list(self.arrival_seqs),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_index": self.batch_index,
            "arrival_seqs": list(self.arrival_seqs),
            "request_ids": list(self.request_ids),
            "stable_ranks": dict(sorted(self.stable_ranks.items())),
        }


@dataclass(frozen=True)
class Plan:
    batches: tuple[Batch, ...]

    def batch_map(self) -> dict[str, Batch]:
        return {batch.batch_id: batch for batch in self.batches}

    def canonical(self) -> bytes:
        from .canonical import canonical_bytes

        return canonical_bytes([batch.to_json() for batch in self.batches])
