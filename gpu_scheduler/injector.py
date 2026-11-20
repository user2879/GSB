"""Deterministic fault injection for the four required business codes.

Injection only rewrites a *copy* of the declaration files inside the workspace
or adjusts a parameter; the original samples are never modified.  Each injector
returns the paths to load plus an expected business code.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from .errors import (
    E_CYCLE,
    E_LEDGER,
    E_MEMORY,
    E_QUOTA,
    E_SPEC,
    SchedulerError,
)
from .loader import load_declaration

INJECT_MEMORY = "memory"
INJECT_QUOTA = "quota"
INJECT_CYCLE = "cycle"
INJECT_LEDGER = "ledger"
INJECTION_KINDS = (INJECT_MEMORY, INJECT_QUOTA, INJECT_CYCLE, INJECT_LEDGER)


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def inject_declaration(kind: str, session_path: str, pools_path: str,
                       tasks_path: str, out_dir: str) -> Dict[str, str]:
    """Produce an injected declaration copy and return its file paths."""
    if kind not in INJECTION_KINDS:
        raise ValueError(f"unknown injection kind: {kind}")
    session = json.load(open(session_path, encoding="utf-8"))
    pools = json.load(open(pools_path, encoding="utf-8"))
    tasks = json.load(open(tasks_path, encoding="utf-8"))

    if kind == INJECT_MEMORY:
        # Make the first task's budget smaller than one shard's footprint.
        target = tasks["tasks"][0]
        target["memory_budget_mib"] = max(
            1, target["gpu_need"] * target["memory_per_gpu_mib"] - 1)
    elif kind == INJECT_QUOTA:
        # Push the pool share total above the declared session constant.
        pools["pools"][0]["quota_shares"] += 1
    elif kind == INJECT_CYCLE:
        # Introduce an intra-task shard cycle on the first multi-shard task.
        target = next(task for task in tasks["tasks"]
                      if task["shard_count"] >= 2)
        target["shards"][0]["depends_on"] = [1]
        target["shards"][1]["depends_on"] = [0]
    elif kind == INJECT_LEDGER:
        # The declaration stays valid; ledger tampering happens post-load.
        pass

    paths = {
        "session": os.path.join(out_dir, f"session_{kind}.json"),
        "pools": os.path.join(out_dir, f"pools_{kind}.json"),
        "tasks": os.path.join(out_dir, f"tasks_{kind}.json"),
    }
    _write(paths["session"], session)
    _write(paths["pools"], pools)
    _write(paths["tasks"], tasks)
    return paths


def expected_code(kind: str) -> str:
    return {
        INJECT_MEMORY: E_MEMORY,
        INJECT_QUOTA: E_QUOTA,
        INJECT_CYCLE: E_CYCLE,
        INJECT_LEDGER: E_LEDGER,
    }[kind]


def run_injected_validation(kind: str, session_path: str, pools_path: str,
                            tasks_path: str, out_dir: str) -> Tuple[str, dict]:
    """Load an injected copy and reclassify the granular problem to its code.

    Normal loads raise the umbrella ``E_SPEC``; the injection harness maps the
    precise sub-code stored on each problem to the required business code.
    """
    paths = inject_declaration(
        kind, session_path, pools_path, tasks_path, out_dir)
    try:
        load_declaration(paths["session"], paths["pools"], paths["tasks"])
    except SchedulerError as exc:
        if kind == INJECT_LEDGER:
            raise
        sub_codes = {problem.code for problem in exc.problems}
        precise = expected_code(kind)
        if precise in sub_codes:
            exc.code = precise
            exc.message = f"injected {kind} fault: {exc.message}"
        return exc.code, exc.to_dict()
    # A clean load is only acceptable for the ledger injection, which tampers
    # with the database rather than the declaration.
    if kind == INJECT_LEDGER:
        return E_SPEC, {"code": E_SPEC,
                        "message": "ledger injection applied at runtime",
                        "problems": []}
    raise AssertionError(f"injection {kind} failed to trigger a validation error")


def tamper_ledger(ledger_path: str) -> None:
    """Corrupt a committed ledger byte to force E_LEDGER on verification."""
    import sqlite3
    conn = sqlite3.connect(ledger_path)
    row = conn.execute(
        "SELECT round_no, round_digest FROM rounds ORDER BY round_no LIMIT 1"
    ).fetchone()
    if row is None:
        raise AssertionError("cannot tamper an empty ledger")
    round_no, digest = row
    corrupted = ("0" if digest[0] != "0" else "1") + digest[1:]
    conn.execute("UPDATE rounds SET round_digest=? WHERE round_no=?",
                 (corrupted, round_no))
    conn.commit()
    conn.close()
