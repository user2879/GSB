"""Bounded real thread-pool arbitration scheduler (requirement 6).

Two constraints are enforced simultaneously:

* at most ``global_limit`` round jobs are running in worker threads;
* at most ``per_session_limit`` jobs of one session are running.

A single dispatcher moves FIFO-queued jobs to eligible worker slots. Waiting
is driven only by contention on real threading primitives; there are no sleep
calls or artificial delays. Wait intervals use ``time.monotonic_ns`` and are
reported in whole milliseconds.
"""

import threading
import time


def monotonic_ms():
    return time.monotonic_ns() // 1_000_000


class RoundJob:
    def __init__(self, session_id, round_index, func):
        self.session_id = session_id
        self.round_index = round_index
        self.func = func
        self.enqueued_at = monotonic_ms()
        self.wait_started_at = None
        self.wait_ended_at = None
        self.started_at = None
        self.finished_at = None
        self.result = None
        self.error = None


class Scheduler:
    def __init__(self, global_limit, per_session_limit):
        if global_limit < 1 or per_session_limit < 1:
            raise ValueError("concurrency limits must be positive")
        if per_session_limit > global_limit:
            raise ValueError("per-session limit cannot exceed global limit")
        self.global_limit = global_limit
        self.per_session_limit = per_session_limit
        self._lock = threading.Lock()
        self._dispatchable = threading.Condition(self._lock)
        self._work_available = threading.Condition(self._lock)
        self._all_done = threading.Condition(self._lock)
        self._pending = []
        self._ready = []
        self._running = 0
        self._session_running = {}
        self._peak_running = 0
        self._peak_session = {}
        self._stopping = False
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"rdiq-worker-{index}",
                daemon=True,
            )
            for index in range(global_limit)
        ]
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="rdiq-dispatcher", daemon=True
        )
        for worker in self._workers:
            worker.start()
        self._dispatcher.start()

    def submit(self, session_id, round_index, func):
        job = RoundJob(session_id, round_index, func)
        with self._lock:
            self._pending.append(job)
            self._session_running.setdefault(session_id, 0)
            self._peak_session.setdefault(session_id, 0)
            self._dispatchable.notify()
        return job

    def _dispatch_loop(self):
        while True:
            with self._lock:
                while not self._stopping and not self._has_dispatch_work():
                    self._dispatchable.wait()
                if self._stopping and not self._pending and not self._ready:
                    return
                progressed = False
                blocked_by_session = set()
                still_pending = []
                for job in self._pending:
                    if self._running >= self.global_limit:
                        still_pending.append(job)
                        continue
                    session_count = self._session_running[job.session_id]
                    if session_count >= self.per_session_limit:
                        blocked_by_session.add(job.session_id)
                        still_pending.append(job)
                        continue
                    if job.wait_started_at is None:
                        job.wait_started_at = monotonic_ms()
                    job.wait_ended_at = monotonic_ms()
                    self._running += 1
                    self._session_running[job.session_id] += 1
                    self._peak_running = max(self._peak_running, self._running)
                    self._peak_session[job.session_id] = max(
                        self._peak_session[job.session_id],
                        self._session_running[job.session_id],
                    )
                    self._ready.append(job)
                    progressed = True
                self._pending = still_pending
                if progressed:
                    self._work_available.notify_all()

    def _has_dispatch_work(self):
        if not self._pending:
            return False
        if self._running >= self.global_limit:
            return False
        return any(
            self._session_running[job.session_id] < self.per_session_limit
            for job in self._pending
        )

    def _worker_loop(self):
        while True:
            with self._lock:
                while not self._ready and not self._stopping:
                    self._work_available.wait()
                if not self._ready and self._stopping:
                    return
                job = self._ready.pop(0)
                job.started_at = monotonic_ms()
            try:
                job.result = job.func()
            except BaseException as exc:  # captured and re-raised on join
                job.error = exc
            finally:
                job.finished_at = monotonic_ms()
                with self._lock:
                    self._running -= 1
                    self._session_running[job.session_id] -= 1
                    self._all_done.notify_all()
                    self._dispatchable.notify()

    def shutdown(self, jobs):
        with self._lock:
            while self._pending or self._ready or self._running:
                self._all_done.wait()
            self._stopping = True
            self._work_available.notify_all()
            self._dispatchable.notify_all()
        self._dispatcher.join()
        for worker in self._workers:
            worker.join()
        return [self._job_report(job) for job in jobs]

    def _job_report(self, job):
        if job.error is not None:
            raise job.error
        return {
            "session_id": job.session_id,
            "round": job.round_index,
            "enqueued_at_ms": job.enqueued_at,
            "wait_started_at_ms": job.wait_started_at,
            "wait_ended_at_ms": job.wait_ended_at,
            "wait_elapsed_ms": (job.wait_ended_at or 0)
            - (job.wait_started_at or 0),
            "started_at_ms": job.started_at,
            "finished_at_ms": job.finished_at,
            "running_elapsed_ms": job.finished_at - job.started_at,
        }

    def observed_limits(self):
        with self._lock:
            return {
                "global_limit": self.global_limit,
                "per_session_limit": self.per_session_limit,
                "peak_running": self._peak_running,
                "peak_per_session": dict(self._peak_session),
            }
