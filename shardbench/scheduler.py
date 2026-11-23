"""Parallel probe scheduling under dual quotas.

A global concurrency cap plus a per-physical-db quota gate every probe task.
Tasks that cannot be admitted immediately block on real semaphores (no sleep
substitutes); wait start/end are recorded in monotonic-clock milliseconds.
Each probe runs while holding a lease on its target db.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .clock import monotonic_ms


@dataclass
class ProbeTask:
    task_id: str
    unit_id: str
    db: int
    table: int
    work: callable  # () -> result payload (JSON-safe)


@dataclass
class ProbeRecord:
    task_id: str
    unit_id: str
    db: int
    table: int
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    exec_start_ms: int = 0
    exec_end_ms: int = 0
    lease: dict = field(default_factory=dict)
    status: str = "ok"  # ok | error
    error: str = ""
    result: object = None

    @property
    def wait_ms(self):
        return self.wait_end_ms - self.wait_start_ms

    @property
    def exec_ms(self):
        return self.exec_end_ms - self.exec_start_ms

    def as_dict(self):
        return {
            "task_id": self.task_id,
            "unit_id": self.unit_id,
            "db": self.db,
            "table": self.table,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "exec_start_ms": self.exec_start_ms,
            "exec_end_ms": self.exec_end_ms,
            "wait_ms": self.wait_ms,
            "exec_ms": self.exec_ms,
            "lease": self.lease,
            "status": self.status,
            "error": self.error,
            "result": self.result,
        }


class QuotaExceeded(RuntimeError):
    pass


class ProbeScheduler:
    def __init__(self, global_limit: int, per_db_limit: int, lease_manager,
                 clock=monotonic_ms):
        if global_limit <= 0 or per_db_limit <= 0:
            raise ValueError("limits must be positive")
        self.global_limit = global_limit
        self.per_db_limit = per_db_limit
        self.lease_manager = lease_manager
        self.clock = clock
        self._global_sem = threading.BoundedSemaphore(global_limit)
        self._db_sems = {}
        self._db_sems_lock = threading.Lock()
        self._running_global = 0
        self._running_per_db = {}
        self._max_observed_global = 0
        self._max_observed_per_db = {}
        self._count_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=global_limit, thread_name_prefix="probe"
        )

    def _db_sem(self, db):
        with self._db_sems_lock:
            if db not in self._db_sems:
                self._db_sems[db] = threading.BoundedSemaphore(self.per_db_limit)
            return self._db_sems[db]

    def _enter(self, db):
        with self._count_lock:
            self._running_global += 1
            self._running_per_db[db] = self._running_per_db.get(db, 0) + 1
            self._max_observed_global = max(
                self._max_observed_global, self._running_global
            )
            self._max_observed_per_db[db] = max(
                self._max_observed_per_db.get(db, 0), self._running_per_db[db]
            )
            if self._running_global > self.global_limit:
                raise QuotaExceeded("global quota violated")
            if self._running_per_db[db] > self.per_db_limit:
                raise QuotaExceeded(f"per-db quota violated for db {db}")

    def _exit(self, db):
        with self._count_lock:
            self._running_global -= 1
            self._running_per_db[db] -= 1

    def submit(self, task: ProbeTask):
        return self._pool.submit(self._run, task)

    def run_all(self, tasks):
        futures = [self.submit(t) for t in tasks]
        return [f.result() for f in futures]

    def _run(self, task: ProbeTask) -> ProbeRecord:
        rec = ProbeRecord(
            task_id=task.task_id, unit_id=task.unit_id, db=task.db, table=task.table
        )
        rec.wait_start_ms = self.clock()
        self._global_sem.acquire()
        db_sem = self._db_sem(task.db)
        db_sem.acquire()
        rec.wait_end_ms = self.clock()
        lease = None
        try:
            self._enter(task.db)
            lease = self.lease_manager.acquire(task.db)
            rec.lease = lease.as_dict()
            rec.exec_start_ms = self.clock()
            rec.result = task.work()
            rec.status = "ok"
        except Exception as exc:  # noqa: BLE001 - recorded into the ledger
            rec.status = "error"
            rec.error = f"{type(exc).__name__}: {exc}"
        finally:
            rec.exec_end_ms = self.clock()
            if lease is not None:
                try:
                    self.lease_manager.release(lease.lease_id)
                except Exception:
                    pass
                rec.lease = lease.as_dict()
            self._exit(task.db)
            db_sem.release()
            self._global_sem.release()
        return rec

    def stats(self):
        with self._count_lock:
            return {
                "max_observed_global": self._max_observed_global,
                "max_observed_per_db": dict(self._max_observed_per_db),
            }

    def shutdown(self):
        self._pool.shutdown(wait=True)
