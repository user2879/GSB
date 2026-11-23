"""Authoritative recompute: replay the ledger request-by-request.

Replays rate-limit and auth decisions deterministically from the spec and
the ledger-recorded arrival times, re-forwards admitted requests to the
live upstreams, and compares every field against the ledger rows. The
first divergence is reported with its request seq, field and business
code. Also compares the GUI incremental-cache view against the
authoritative result (dual-path consistency).
"""

from dataclasses import dataclass, field

from . import E_REPLAY
from .clock import monotonic_ms
from .engine import decide_from_spec, make_buckets
from .forward import forward_request
from .stats import aggregate_by_scope

COMPARE_FIELDS = (
    "token_before", "token_refilled", "allowed", "token_after",
    "auth_ok", "auth_reason", "decision_code", "receipt_summary")


@dataclass
class Divergence:
    seq: int
    field: str
    expected: object
    actual: object
    code: str

    def __str__(self):
        return (f"first divergence at seq={self.seq} "
                f"field={self.field}: ledger={self.expected!r} "
                f"recomputed={self.actual!r} code={self.code}")


@dataclass
class RecomputeResult:
    conclusions: list = field(default_factory=list)
    divergence: object = None
    stats: dict = field(default_factory=dict)
    elapsed_ms: int = 0

    @property
    def consistent(self):
        return self.divergence is None


def authoritative_recompute(spec, ledger, forward_fn=None,
                            record_observation=True):
    """Replay every done-row in the ledger; compare field by field."""
    t0 = monotonic_ms()
    rows = ledger.done_requests()
    reqs_by_seq = {r.seq: r for r in spec.requests}
    buckets = make_buckets(spec, ledger.scale)
    conclusions = []
    divergence = None
    per_service_ms = {}
    for row in rows:
        s0 = monotonic_ms()
        req = reqs_by_seq.get(row["seq"])
        if req is None:
            divergence = Divergence(row["seq"], "seq", "absent in spec",
                                    "present in ledger", E_REPLAY)
            break
        dec = decide_from_spec(spec, buckets, req)
        code = dec.code
        receipt_summary = "none"
        if dec.passed:
            if forward_fn is not None:
                receipt = forward_fn(req)
            else:
                svc = spec.services[req.service_id]
                receipt = forward_request(svc.host, svc.port, req.seq,
                                          req.service_id, req.client_key)
            code = receipt.code
            receipt_summary = receipt.summary
        recomputed = {
            "token_before": dec.token_before,
            "token_refilled": dec.token_refilled,
            "allowed": 1 if dec.allowed else 0,
            "token_after": dec.token_after,
            "auth_ok": 1 if dec.auth_ok else 0,
            "auth_reason": dec.auth_reason,
            "decision_code": code,
            "receipt_summary": receipt_summary,
        }
        conclusions.append({"seq": row["seq"],
                            "service_id": row["service_id"], **recomputed})
        for fname in COMPARE_FIELDS:
            if row[fname] != recomputed[fname]:
                divergence = Divergence(row["seq"], fname, row[fname],
                                        recomputed[fname], E_REPLAY)
                break
        per_service_ms.setdefault(row["service_id"], 0)
        per_service_ms[row["service_id"]] += monotonic_ms() - s0
        if divergence is not None:
            break
    elapsed = monotonic_ms() - t0
    if record_observation:
        ledger.record_observation("recompute", "overall", elapsed)
        for sid in sorted(per_service_ms):
            ledger.record_observation("recompute", sid,
                                      per_service_ms[sid])
    stats = {"recompute": aggregate_by_scope(
        ledger.observations("recompute"))}
    return RecomputeResult(conclusions=conclusions, divergence=divergence,
                           stats=stats, elapsed_ms=elapsed)


def compare_cache_with_authoritative(cache_view, recompute_result):
    """Dual-path check: GUI incremental cache vs authoritative recompute.
    Returns a Divergence for the first mismatching request, else None."""
    auth_by_seq = {c["seq"]: c for c in recompute_result.conclusions}
    for item in cache_view["conclusions"]:
        auth = auth_by_seq.get(item["seq"])
        if auth is None:
            return Divergence(item["seq"], "seq", "missing in recompute",
                              "present in cache", E_REPLAY)
        for fname in ("decision_code", "auth_reason", "token_before",
                      "token_refilled", "token_after", "receipt_summary"):
            cache_key = "code" if fname == "decision_code" else fname
            if item[cache_key] != auth[fname]:
                return Divergence(item["seq"], fname, item[cache_key],
                                  auth[fname], auth["decision_code"])
    if cache_view["total"] != len(recompute_result.conclusions):
        return Divergence(-1, "total", cache_view["total"],
                          len(recompute_result.conclusions), E_REPLAY)
    return None
