"""Parallel adjudication with global concurrency limit, per-node quotas,
stable-order per-sector serialization and real per-attempt timeouts.

Retrying a timed-out attempt never changes the granted scope: the scope is
a pure function of the request, and each attempt is recorded separately in
the adjudication trace.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .errors import E_OWNER, E_SCOPE, E_TIMEOUT, PlatformError


@dataclass
class Request:
    req_id: str
    node_id: str
    op: str               # "read" | "write"
    agg: str              # sum | mean | min | max
    sectors: tuple
    delta: float = 0.0
    count: int = 1


@dataclass
class Grant:
    lease_id: str
    node_id: str
    sectors: tuple
    seq_lo: int
    seq_hi: int
    ttl_ms: int


class _AttemptTimeout(Exception):
    pass


class TicketLock:
    """FIFO ticket lock serializing conflicting requests by stable order."""

    def __init__(self):
        self._cond = threading.Condition()
        self._next = 0
        self._serving = 0

    def acquire(self, timeout):
        with self._cond:
            ticket = self._next
            self._next += 1
            if not self._cond.wait_for(lambda: self._serving == ticket, timeout):
                if self._serving == ticket:
                    self._serving += 1
                    self._cond.notify_all()
                return False
            return True

    def release(self):
        with self._cond:
            self._serving += 1
            self._cond.notify_all()


class Adjudicator:
    def __init__(self, platform, global_limit=4, node_quota=2,
                 timeout_s=2.0, max_retries=1, clock=time.monotonic):
        self.platform = platform
        self.clock = clock
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.pool = ThreadPoolExecutor(max_workers=global_limit,
                                       thread_name_prefix="adj")
        self.node_quota = node_quota
        self.node_sems = defaultdict(lambda: threading.Semaphore(node_quota))
        self.tickets = defaultdict(TicketLock)
        self._metrics_lock = threading.Lock()
        self.node_inflight = defaultdict(int)
        self.node_max_inflight = defaultdict(int)
        self.inflight = 0
        self.max_inflight = 0

    def shutdown(self):
        self.pool.shutdown(wait=True)

    def submit(self, request):
        return self.pool.submit(self._adjudicate, request)

    def _record(self, req, attempt, result, detail):
        self.platform.store.insert_adjudication(
            req.req_id, attempt, req.node_id, req.sectors, result, detail,
            self.clock())

    def _adjudicate(self, req):
        with self._metrics_lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            sem = self.node_sems[req.node_id]
            acquired = False
            grant = None
            for attempt in range(self.max_retries + 1):
                deadline = self.clock() + self.timeout_s
                remaining = deadline - self.clock()
                if not sem.acquire(timeout=max(remaining, 0.0)):
                    self._record(req, attempt, "timeout",
                                 {"stage": "node_quota", "code": E_TIMEOUT})
                    continue
                acquired = True
                with self._metrics_lock:
                    self.node_inflight[req.node_id] += 1
                    cur = self.node_inflight[req.node_id]
                    self.node_max_inflight[req.node_id] = max(
                        self.node_max_inflight[req.node_id], cur)
                try:
                    grant = self._attempt(req, attempt, deadline)
                    break
                except _AttemptTimeout as exc:
                    self._record(req, attempt, "timeout",
                                 {"stage": exc.args[0], "code": E_TIMEOUT})
                    grant = None
                finally:
                    if grant is None:
                        with self._metrics_lock:
                            self.node_inflight[req.node_id] -= 1
                        sem.release()
                        acquired = False
            if grant is None:
                raise PlatformError(E_TIMEOUT, req_id=req.req_id,
                                    attempts=self.max_retries + 1)
            try:
                result = self.platform.execute(grant, req)
            finally:
                with self._metrics_lock:
                    self.node_inflight[req.node_id] -= 1
                sem.release()
            return result
        finally:
            with self._metrics_lock:
                self.inflight -= 1

    def _attempt(self, req, attempt, deadline):
        platform = self.platform
        for sid in req.sectors:
            if sid not in platform.owners:
                err = PlatformError(E_SCOPE, sector=sid,
                                    reason="unknown sector")
                self._record(req, attempt, "rejected", err.to_dict())
                raise err
        if req.op == "write":
            for sid in req.sectors:
                if platform.owners[sid] != req.node_id:
                    err = PlatformError(E_OWNER, sector=sid, node=req.node_id,
                                        owner=platform.owners[sid],
                                        reason="write to non-owned sector")
                    self._record(req, attempt, "rejected", err.to_dict())
                    raise err
        acquired = []
        try:
            for sid in sorted(req.sectors):
                remaining = deadline - self.clock()
                if remaining <= 0 or not self.tickets[sid].acquire(remaining):
                    raise _AttemptTimeout("sector_ticket")
                acquired.append(self.tickets[sid])
            while True:
                platform.leases.reap()
                if all(platform.leases.sector_free(s) for s in req.sectors):
                    break
                if self.clock() >= deadline:
                    raise _AttemptTimeout("sector_busy")
                time.sleep(0.005)
            grant = platform.grant_lease(req)
        finally:
            for ticket in acquired:
                ticket.release()
        self._record(req, attempt, "granted",
                     {"lease_id": grant.lease_id, "sectors": list(grant.sectors),
                      "seq_lo": grant.seq_lo, "seq_hi": grant.seq_hi})
        return grant
