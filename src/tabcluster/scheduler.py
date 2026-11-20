"""Budget-aware parallel scheduler.

Candidates carry a stable sort key derived from method name + parameter
fingerprint. Compute runs on a thread pool (max workers from config), but
the budget ledger is applied in a deterministic serial pass over the stable
key order, so the deduction sequence, rankings, label assignments and
failure order are identical for any concurrency level and any thread
interleaving.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import (FailureLog, E_CAND_BUDGET_EXHAUSTED,
                     E_ITER_BUDGET_PENDING, E_CANDIDATE_FAULT)
from .fingerprint import fingerprint, canonical_json


@dataclass
class Candidate:
    method: str
    params: dict
    stable_key: str = ""
    short_key: str = ""

    def __post_init__(self) -> None:
        params_fp = fingerprint("params", self.method, self.params)
        self.stable_key = f"{self.method}|{canonical_json(self.params)}"
        self.short_key = f"{self.method}:{params_fp[:12]}"


def build_candidates(methods: dict[str, list[dict]]) -> list[Candidate]:
    cands = [Candidate(method=m, params=p) for m, grids in methods.items()
             for p in grids]
    cands.sort(key=lambda c: c.stable_key)
    return cands


@dataclass
class CandidateResult:
    candidate: Candidate
    status: str = "ok"           # ok | quota_exhausted | fault | pending
    output: dict | None = None
    iterations: int = 0
    cache_hit: bool = False
    error_code: str | None = None
    error_stage: str | None = None


@dataclass
class ScheduleReport:
    results: list = field(default_factory=list)   # CandidateResult, stable order
    ledger: list = field(default_factory=list)    # budget deduction sequence
    todo: list = field(default_factory=list)      # pending candidate keys
    consumed: int = 0
    remaining: int = 0


def run_schedule(candidates: list[Candidate],
                 compute_fn: Callable[[Candidate], tuple[dict, int]],
                 global_budget: int, candidate_quota: int,
                 concurrency: int, failures: FailureLog,
                 on_committed: Callable[[int, CandidateResult, int], None] | None = None
                 ) -> ScheduleReport:
    """compute_fn(candidate) -> (output_dict, iterations_used).

    It must be deterministic and must itself enforce the per-candidate
    quota (outputs carry 'quota_exhausted').
    """
    # --- concurrent compute (order-free) ----------------------------------
    outputs: dict[str, Any] = {}
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_map = {pool.submit(compute_fn, c): c for c in candidates}
            for fut, cand in future_map.items():
                try:
                    outputs[cand.stable_key] = fut.result()
                except BaseException as exc:  # isolated per candidate
                    outputs[cand.stable_key] = exc
    else:
        for cand in candidates:
            try:
                outputs[cand.stable_key] = compute_fn(cand)
            except BaseException as exc:
                outputs[cand.stable_key] = exc

    # --- deterministic serial ledger in stable-key order ------------------
    report = ScheduleReport()
    remaining = int(global_budget)
    for idx, cand in enumerate(candidates):  # already sorted by stable_key
        res = CandidateResult(candidate=cand)
        got = outputs[cand.stable_key]
        if isinstance(got, BaseException):
            res.status = "fault"
            res.error_code = getattr(got, "code", E_CANDIDATE_FAULT)
            res.error_stage = "iterate"
            failures.add(res.error_code, f"{type(got).__name__}: {got}",
                         stage="iterate", candidate_key=cand.short_key)
            report.results.append(res)
            if on_committed is not None:
                on_committed(idx, res, report.consumed)
            continue
        output, iterations = got
        res.output = output
        res.iterations = int(iterations)
        if output.get("quota_exhausted"):
            res.status = "quota_exhausted"
            res.error_code = E_CAND_BUDGET_EXHAUSTED
            res.error_stage = "iterate"
            failures.add(E_CAND_BUDGET_EXHAUSTED,
                         "candidate exceeded per-candidate quota",
                         stage="iterate", candidate_key=cand.short_key,
                         quota=candidate_quota)
        # budget deduction (auditable sequence)
        need = res.iterations
        if res.status == "ok" and need > remaining:
            res.status = "pending"
            res.error_code = E_ITER_BUDGET_PENDING
            res.error_stage = "schedule"
            failures.add(E_ITER_BUDGET_PENDING,
                         "global iteration budget exhausted; candidate kept "
                         "in todo queue", stage="schedule",
                         candidate_key=cand.short_key, needed=need,
                         remaining=remaining)
            report.todo.append(cand.short_key)
            report.ledger.append({"seq": len(report.ledger),
                                  "candidate": cand.short_key,
                                  "requested": need, "deducted": 0,
                                  "remaining": remaining,
                                  "event": "pending"})
            report.results.append(res)
            if on_committed is not None:
                on_committed(idx, res, report.consumed)
            continue
        if res.status == "quota_exhausted":
            deducted = min(need, remaining)  # quota-aborted run still consumed
        else:
            deducted = need
        remaining -= deducted
        report.consumed += deducted
        report.ledger.append({"seq": len(report.ledger),
                              "candidate": cand.short_key,
                              "requested": need, "deducted": deducted,
                              "remaining": remaining,
                              "event": "deducted"})
        report.results.append(res)
        if on_committed is not None:
            on_committed(idx, res, report.consumed)
    report.remaining = remaining
    return report
