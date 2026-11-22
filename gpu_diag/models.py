"""Typed domain models for pools, tasks, shards and the session parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ShardState(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    AFFECTED = "AFFECTED"


@dataclass(frozen=True)
class Pool:
    pool_id: str
    gpu_count: int
    memory_per_gpu_mib: int
    compute_units: int
    quota_share: int

    @property
    def total_memory_mib(self) -> int:
        return self.gpu_count * self.memory_per_gpu_mib


@dataclass(frozen=True)
class Task:
    task_id: str
    gpu_need: int
    memory_per_gpu_mib: int
    priority: int
    duration_ms: int
    shard_count: int
    submit_time_ms: int
    memory_budget_mib: int
    deps: tuple[str, ...] = field(default=())


@dataclass
class Shard:
    shard_id: str
    task_id: str
    index: int
    deps: tuple[str, ...] = field(default=())
    state: ShardState = ShardState.PENDING
    result_digest: str | None = None
    pool_id: str | None = None
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    start_ms: int | None = None
    finish_ms: int | None = None
    attempts: int = 0

    @property
    def global_id(self) -> str:
        return f"{self.task_id}#{self.index}"


@dataclass(frozen=True)
class SessionParams:
    global_gpu_quota: int
    pool_concurrency_limit: int
    scale_factor: int
    truncation: str  # "floor" — documented quantization direction
    ledger_format_version: int
    pool_version: str
    task_version: str
    scale_version: str


@dataclass
class Declaration:
    pools: dict[str, Pool]
    tasks: dict[str, Task]
    shards: dict[str, Shard]
    params: SessionParams
