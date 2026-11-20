"""三条路径结局序列与配额终态的确定性校验（NumPy）。"""
from __future__ import annotations

import hashlib

import numpy as np

from .models import RunResult

# 结局规范列：全部为可比较的整数 / 定长字符串，避免对象比较的歧义
OUTCOME_DTYPE = "U64"


def outcome_matrix(result: RunResult) -> np.ndarray:
    rows = ["|".join(str(x) for x in o.canonical_tuple()) for o in result.outcomes]
    return np.array(rows, dtype=OUTCOME_DTYPE)


def final_state_matrix(result: RunResult) -> np.ndarray:
    tids = sorted(result.final_tenant_quota)
    pids = sorted(result.final_pool_capacity)
    rows = []
    for tid in tids:
        rows.append(
            "T|{}|quota={}|burst={}|inflight={}".format(
                tid,
                result.final_tenant_quota[tid],
                result.final_tenant_burst[tid],
                result.final_inflight[tid],
            )
        )
    for pid in pids:
        rows.append("P|{}|capacity={}".format(pid, result.final_pool_capacity[pid]))
    return np.array(rows, dtype=OUTCOME_DTYPE)


def queue_trace_matrix(result: RunResult) -> np.ndarray:
    rows = [
        "|".join((e.rid, e.tid, e.pid, e.kind, str(e.at_ms), str(e.wait_ms), e.basis, e.detail))
        for e in result.queue_events
    ]
    return np.array(rows, dtype=OUTCOME_DTYPE)


def matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(matrix.tolist())).encode("utf-8")).hexdigest()


def compare_results(results: dict[str, RunResult]):
    """返回逐字节一致性报告。键为路径名。"""
    names = list(results)
    base = results[names[0]]
    base_outcomes = outcome_matrix(base)
    base_state = final_state_matrix(base)
    base_trace = queue_trace_matrix(base)

    report = {
        "reference_path": names[0],
        "outcome_bytes": matrix_sha256(base_outcomes),
        "state_bytes": matrix_sha256(base_state),
        "trace_bytes": matrix_sha256(base_trace),
        "identical": True,
        "per_path": {},
        "first_mismatch": None,
    }

    for name in names[1:]:
        other = results[name]
        oc = outcome_matrix(other)
        st = final_state_matrix(other)
        tr = queue_trace_matrix(other)
        same_shape = oc.shape == base_outcomes.shape and st.shape == base_state.shape
        same_outcomes = same_shape and bool(np.array_equal(oc, base_outcomes))
        same_state = same_shape and bool(np.array_equal(st, base_state))
        same_trace = bool(np.array_equal(tr, base_trace))
        report["per_path"][name] = {
            "outcomes_identical": same_outcomes,
            "final_state_identical": same_state,
            "queue_trace_identical": same_trace,
            "outcome_bytes": matrix_sha256(oc),
            "state_bytes": matrix_sha256(st),
            "trace_bytes": matrix_sha256(tr),
        }
        if not (same_outcomes and same_state and same_trace):
            report["identical"] = False
            if report["first_mismatch"] is None:
                report["first_mismatch"] = _first_mismatch(base, other)
    return report


def _first_mismatch(a: RunResult, b: RunResult) -> dict:
    for idx, (oa, ob) in enumerate(zip(a.outcomes, b.outcomes)):
        ta = oa.canonical_tuple()
        tb = ob.canonical_tuple()
        if ta != tb:
            fields = (
                "rid", "tid", "pid", "arrival", "verdict", "reason", "basis",
                "decided_at", "granted_at", "completed_at", "wait_ms",
                "tenant_quota_remaining", "tenant_burst_remaining", "inflight",
                "pool_capacity_remaining", "demand", "enqueued",
            )
            for field_name, x, y in zip(fields, ta, tb):
                if x != y:
                    return {"rid": oa.rid, "field": field_name, "expected": x, "actual": y}
    return {"rid": None, "field": "final_state", "expected": "reference", "actual": "other"}
