"""Gateway engine: deterministic decisions + scheduled forwarding + ledger.

Token and auth decisions for each service are evaluated in request-seq
order under a per-service ordered barrier, driven by the declared
arrival_ms clock. Forwarding runs concurrently on the worker pool under
the dual quota constraints. The incremental cache is updated once per
committed batch; the GUI quick view reads it in a single snapshot and
never triggers a replay.
"""

import threading
from dataclasses import dataclass

from . import E_AUTH, E_RATE, OK
from .authn import verify_signature
from .clock import monotonic_ms
from .forward import forward_request
from .scheduler import Scheduler
from .tokens import TokenBucket


@dataclass
class DecisionRecord:
    seq: int
    token_before: int
    token_refilled: int
    allowed: bool
    token_after: int
    auth_ok: bool
    auth_reason: str
    code: str

    @property
    def passed(self):
        return self.code == OK


class OrderedBarrier:
    """Serializes the decision phase of one service in request order."""

    def __init__(self):
        self._cond = threading.Condition()
        self._next = 0

    def wait_turn(self, index):
        with self._cond:
            while index != self._next:
                self._cond.wait()

    def advance(self):
        with self._cond:
            self._next += 1
            self._cond.notify_all()


class IncrementalCache:
    """Per-request conclusions updated batch-by-batch (no replay)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conclusions = {}
        self._code_counts = {}

    def update(self, rows):
        with self._lock:
            for row in rows:
                self._conclusions[row["seq"]] = {
                    "seq": row["seq"],
                    "service_id": row["service_id"],
                    "code": row["decision_code"],
                    "auth_reason": row["auth_reason"],
                    "token_before": row["token_before"],
                    "token_refilled": row["token_refilled"],
                    "token_after": row["token_after"],
                    "receipt_summary": row["receipt_summary"],
                    "wait_ms": row["wait_end_ms"] - row["wait_start_ms"],
                }
                code = row["decision_code"]
                self._code_counts[code] = self._code_counts.get(code, 0) + 1

    def snapshot(self):
        """One read of the whole cache; never triggers recompute."""
        with self._lock:
            return {
                "conclusions": [self._conclusions[k]
                                for k in sorted(self._conclusions)],
                "summary": dict(sorted(self._code_counts.items())),
                "total": len(self._conclusions),
            }


def decide_from_spec(spec, buckets, req):
    """Pure deterministic decision shared by live engine and replay."""
    svc = spec.services[req.service_id]
    bucket = buckets[(req.service_id, req.bucket_key)]
    token = bucket.evaluate(req.arrival_ms)
    auth_ok, auth_reason = verify_signature(
        spec.keys[req.service_id], svc.signature_window_ms,
        req.cred_version, req.signature, req.client_key,
        req.service_id, req.seq, req.arrival_ms)
    if not token.allowed:
        code = E_RATE
    elif not auth_ok:
        code = E_AUTH
    else:
        code = OK
    return DecisionRecord(
        seq=req.seq, token_before=token.before,
        token_refilled=token.refilled, allowed=token.allowed,
        token_after=token.after, auth_ok=auth_ok,
        auth_reason=auth_reason, code=code)


def make_buckets(spec, scale):
    buckets = {}
    for sid, svc in spec.services.items():
        for bk in svc.bucket_keys:
            buckets[(sid, bk)] = TokenBucket(
                svc.capacity_tokens, svc.refill_interval_ms,
                svc.cost_tokens, scale)
    return buckets


class Engine:
    def __init__(self, spec, ledger, global_limit=4, per_service_quota=2,
                 upstream_timeout_s=2.0):
        self.spec = spec
        self.ledger = ledger
        self.scale = ledger.scale
        self.buckets = make_buckets(spec, self.scale)
        self.cache = IncrementalCache()
        quotas = {sid: per_service_quota for sid in spec.services}
        self.scheduler = Scheduler(global_limit, quotas)
        self.upstream_timeout_s = upstream_timeout_s
        self._barriers = {sid: OrderedBarrier() for sid in spec.services}
        self._index = {}
        per_service = {sid: [] for sid in spec.services}
        for req in spec.requests:
            per_service[req.service_id].append(req.seq)
        for sid, seqs in per_service.items():
            for idx, seq in enumerate(seqs):
                self._index[seq] = idx

    def _gate(self, req):
        """Wait for this request's ordered decision turn (pre-quota)."""
        self._barriers[req.service_id].wait_turn(self._index[req.seq])

    def _decide(self, req):
        try:
            return decide_from_spec(self.spec, self.buckets, req)
        finally:
            self._barriers[req.service_id].advance()

    def _forward(self, req):
        svc = self.spec.services[req.service_id]
        return forward_request(svc.host, svc.port, req.seq,
                               req.service_id, req.client_key,
                               timeout_s=self.upstream_timeout_s)

    def run_batch(self, requests):
        batch_id = self.ledger.begin_batch(requests[0].seq,
                                           requests[-1].seq)
        traces = self.scheduler.run(requests, self._decide, self._forward,
                                    gate_fn=self._gate)
        rows = []
        for trace in traces:
            req = requests[trace.seq - requests[0].seq]
            dec = trace.decision
            receipt = trace.receipt
            code = dec.code
            receipt_summary = "none"
            if dec.passed and receipt is not None:
                code = receipt.code
                receipt_summary = receipt.summary
            rows.append({
                "seq": trace.seq,
                "client_key": req.client_key,
                "service_id": req.service_id,
                "bucket_key": req.bucket_key,
                "arrival_ms": req.arrival_ms,
                "token_before": dec.token_before,
                "token_refilled": dec.token_refilled,
                "allowed": 1 if dec.allowed else 0,
                "token_after": dec.token_after,
                "auth_ok": 1 if dec.auth_ok else 0,
                "auth_reason": dec.auth_reason,
                "decision_code": code,
                "receipt_summary": receipt_summary,
                "wait_start_ms": trace.wait_start_ms,
                "wait_end_ms": trace.wait_end_ms,
                "decide_start_ms": trace.decide_start_ms,
                "decide_end_ms": trace.decide_end_ms,
                "fwd_start_ms": trace.fwd_start_ms,
                "fwd_end_ms": trace.fwd_end_ms,
            })
        self.ledger.commit_batch(batch_id, rows)
        self.cache.update(rows)
        return rows

    def run_all(self, batch_size=8):
        all_rows = []
        reqs = list(self.spec.requests)
        for start in range(0, len(reqs), batch_size):
            all_rows.extend(self.run_batch(reqs[start:start + batch_size]))
        return all_rows

    def quick_view(self):
        """Single cache read; records the observed interaction latency."""
        t0 = monotonic_ms()
        view = self.cache.snapshot()
        elapsed = monotonic_ms() - t0
        self.ledger.record_observation("gui_query", "overall", elapsed)
        return view
