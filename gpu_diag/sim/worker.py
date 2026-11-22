"""Real local thread-pool projection of shard execution.

Shards genuinely run on a concurrent.futures worker thread.  The *work* is a
deterministic NumPy computation; wall-clock sleep is never used for scheduling
decisions.  Timing in the ledger comes from the virtual monotonic clock driven
by declared ``duration_ms`` estimates, keeping results reproducible while the
execution itself still crosses thread boundaries.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShardResult:
    shard_id: str
    digest: str
    worker: str
    checksum: int


def _deterministic_work(shard_id: str, task_id: str, index: int,
                        compute_units: int) -> int:
    seed = int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8],
                          "little")
    rng = np.random.default_rng(seed)
    size = 64 + (compute_units % 64)
    vector = rng.integers(0, 1_000_000, size=size, dtype=np.int64)
    return int(np.sum(vector) + np.int64(index) * 7 + len(task_id))


def execute_shard(shard_id: str, task_id: str, index: int,
                  compute_units: int) -> ShardResult:
    checksum = _deterministic_work(shard_id, task_id, index, compute_units)
    digest = hashlib.sha256(
        f"{shard_id}|{checksum}".encode("utf-8")).hexdigest()
    return ShardResult(shard_id=shard_id, digest=digest,
                       worker=threading.current_thread().name, checksum=checksum)


class ShardThreadPool:
    """Thin wrapper guaranteeing work runs off the calling thread."""

    def __init__(self, max_workers: int):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="gpu-shard")

    def submit(self, shard_id: str, task_id: str, index: int,
               compute_units: int):
        return self._executor.submit(
            execute_shard, shard_id, task_id, index, compute_units)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
