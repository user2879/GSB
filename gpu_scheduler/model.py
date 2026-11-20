"""In-memory domain model: resource pools, tasks, shards and session config.

The model is intentionally dependency free (plain dataclasses) so it can be
serialized to JSON snapshots, compared during replay and reused by the Tkinter
diagnostic view without importing the scheduler.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Shard lifecycle states.
READY = "READY"            # dependencies satisfied, eligible for scheduling
WAITING = "WAITING"        # eligible but blocked by quota/memory
RUNNING = "RUNNING"        # dispatched in an earlier committed round
SUCCESS = "SUCCESS"        # finished successfully, output digest retained
FAILED = "FAILED"          # terminal failure injected by the sample
PREEMPTED = "PREEMPTED"    # safely suspended at a shard boundary, recoverable
INVALIDATED = "INVALIDATED"  # downstream of a failure/preemption change
UNKNOWN = "UNKNOWN"        # running but never committed after a hard kill

TERMINAL_STATES = frozenset({SUCCESS, FAILED})
RECOVERABLE_STATES = frozenset({READY, WAITING, PREEMPTED, INVALIDATED, UNKNOWN})


def canonical_json(payload: object) -> str:
    """Deterministic JSON text used for version and digest computation."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def content_digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Pool:
    pool_id: str
    gpu_count: int
    memory_per_gpu_mib: int
    compute_units_per_gpu: int
    quota_shares: int
    max_concurrent_tasks: int

    @property
    def total_memory_mib(self) -> int:
        return self.gpu_count * self.memory_per_gpu_mib

    @property
    def shares_per_gpu(self) -> int:
        # Validated to be exact; helper kept here so scheduling stays integer.
        return self.quota_shares // self.gpu_count

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shard:
    task_id: str
    shard_index: int
    gpu_need: int
    memory_per_gpu_mib: int
    duration_ms: int
    # Intra-task dependency: list of shard indices of the same task.
    depends_on: List[int] = field(default_factory=list)
    # Cross-task dependency: list of task ids whose shards must all succeed.
    depends_on_tasks: List[str] = field(default_factory=list)
    # Mutable runtime state.
    state: str = READY
    version: int = 0
    output_digest: Optional[str] = None
    wait_started_ms: Optional[int] = None
    # Last dispatch bookkeeping (cleared/reused across retries).
    pool_id: Optional[str] = None
    started_ms: Optional[int] = None
    finish_ms: Optional[int] = None
    ended_ms: Optional[int] = None
    wait_duration_ms: int = 0
    attempts: int = 0

    @property
    def shard_id(self) -> str:
        return f"{self.task_id}#S{self.shard_index:02d}"

    @property
    def total_memory_mib(self) -> int:
        return self.gpu_need * self.memory_per_gpu_mib

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Task:
    task_id: str
    kind: str                 # "training" or "inference"
    gpu_need: int
    memory_per_gpu_mib: int
    priority: int
    estimated_ms: int
    shard_count: int
    memory_budget_mib: int
    arrival_ms: int = 0
    depends_on_tasks: List[str] = field(default_factory=list)
    # Shard indices that fail once on their first dispatch (deterministic demo).
    fail_once_shards: List[int] = field(default_factory=list)
    shards: List[Shard] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class SessionConfig:
    total_quota_shares: int
    global_gpu_cap: int
    default_memory_budget_mib: int
    fixed_point_scale: int
    seed: int
    # Clock always starts at zero; kept for explicit, reproducible sessions.
    clock_origin_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Declaration:
    """Fully validated immutable input plus precomputed version hashes."""

    config: SessionConfig
    pools: List[Pool]
    tasks: List[Task]
    pool_version: str
    task_version: str
    config_version: str

    def pool_map(self) -> Dict[str, Pool]:
        return {pool.pool_id: pool for pool in self.pools}

    def task_map(self) -> Dict[str, Task]:
        return {task.task_id: task for task in self.tasks}

    def all_shards(self) -> List[Shard]:
        shards: List[Shard] = []
        for task in self.tasks:
            shards.extend(task.shards)
        return shards

    def shard_index(self) -> Dict[Tuple[str, int], Shard]:
        return {(shard.task_id, shard.shard_index): shard
                for shard in self.all_shards()}
