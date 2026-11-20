from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .adjudicate import adjudicate
from .ledger import summarize_ledger
from .model import E_BUDGET_DENIED, Config, canonical_bytes
from .parser import parse_lines
from .storage import CrashAbort, MatchStore, atomic_write


@dataclass
class MatchTask:
    match_id: str
    sched_priority: int
    est_mem_mb: int
    sample_path: Path


def run_match(task: MatchTask, cfg: Config, runs_root, crash_after=None):
    """Adjudicate one match strictly serially; recover if partially committed."""
    store = MatchStore(runs_root, task.match_id)
    if store.is_closed():
        return store.load_result()
    lines = task.sample_path.read_text(encoding="utf-8").splitlines()
    orders, errors, parse_anomalies = parse_lines(lines, cfg)
    res = adjudicate(task.match_id, orders, parse_anomalies, cfg,
                     store=store, crash_after=crash_after)
    ledger_summary = summarize_ledger(res["ledger_entries"], cfg)
    atomic_write(store.dir / "ledger_summary.json", canonical_bytes(ledger_summary))
    result = {
        "match_id": task.match_id,
        "status": "ERRORS" if errors else "OK",
        "steps": len(res["steps"]),
        "final_hash": res["final_hash"],
        "errors": errors,
        "anomalies": res["anomalies"],
        "overflow_count": res["overflow_count"],
        "first_overflow": res["first_overflow"],
        "ledger_summary": ledger_summary,
    }
    store.close(result)
    return result


def schedule_matches(tasks, cfg: Config, runs_root, max_threads: int,
                     mem_budget_mb: int, crash_match=None, crash_after=None):
    """Budget-constrained parallel scheduling.

    Matches are admitted by fixed (sched_priority, match_id) order against the
    memory budget; rejected matches get E_BUDGET_DENIED. Parallelism is
    downgraded deterministically when the budget cannot sustain the requested
    thread count. Results are merged by match_id, never by completion order.
    """
    ordered = sorted(tasks, key=lambda t: (t.sched_priority, t.match_id))
    admitted = []
    denied = []
    used_mb = 0
    for task in ordered:
        if used_mb + task.est_mem_mb <= mem_budget_mb:
            admitted.append(task)
            used_mb += task.est_mem_mb
        else:
            denied.append({
                "match_id": task.match_id,
                "code": E_BUDGET_DENIED,
                "reason": "memory_budget_exceeded",
                "est_mem_mb": task.est_mem_mb,
                "used_mb": used_mb,
                "budget_mb": mem_budget_mb,
            })

    downgrades = []
    workers = 0
    if admitted:
        per_match = max(t.est_mem_mb for t in admitted)
        mem_workers = max(1, mem_budget_mb // per_match)
        workers = min(max_threads, mem_workers, len(admitted))
        if workers < max_threads:
            reason = "memory_budget" if mem_workers < max_threads else "admitted_match_count"
            downgrades.append({"reason": reason, "before": max_threads, "after": workers})

    results = []
    crashed = False
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_match, task, cfg, runs_root,
                    crash_after if task.match_id == crash_match else None,
                ): task
                for task in admitted
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except CrashAbort:
                    crashed = True
    results.sort(key=lambda r: r["match_id"])

    schedule_record = {
        "max_threads": max_threads,
        "mem_budget_mb": mem_budget_mb,
        "workers": workers,
        "admitted": [t.match_id for t in admitted],
        "denied": denied,
        "downgrades": downgrades,
    }
    return {
        "results": results,
        "denied": denied,
        "schedule": schedule_record,
        "crashed": crashed,
    }
