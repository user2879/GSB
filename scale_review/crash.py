from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .propagation import VersionedPlan
from .spec import load_spec_file
from .workbench import RoundRunner


def force_kill_current_process() -> None:
    """Terminate immediately without cleanup or an exit report."""

    if sys.platform.startswith("win"):
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        kernel32.TerminateProcess(handle, 0xF3)
        os._exit(243)
    os.kill(os.getpid(), signal.SIGKILL)
    os._exit(137)


def run_crash_child(
    spec_path: str | Path,
    db_path: str | Path,
    *,
    minute: int = 4,
    attempt_id: str = "forced-kill-attempt",
) -> dict[str, Any]:
    spec = load_spec_file(spec_path)
    ledger = Ledger(db_path)
    plan = VersionedPlan(spec)
    runner = RoundRunner(spec, ledger, plan)
    committed = runner.run_round(
        minute,
        round_id="committed-before-forced-kill",
        batch_id="batch-before-kill",
    )
    ledger.begin_attempt(
        attempt_id,
        "unknown-round-after-forced-kill",
        minute + 1,
        plan.version,
        {
            "lease_id": "tentative-lease-not-committed",
            "service_id": spec.service_ids[0],
            "version": 1,
            "global_slots": [0],
            "basis": "running_attempt_marker_committed_before_sigkill",
        },
    )
    cursor = ledger.get_cursor()
    ledger.close()
    result = {
        "committed_round_seq": committed["commit"]["round_seq"],
        "cursor": cursor,
        "attempt_id": attempt_id,
        "pid": os.getpid(),
    }
    print(result, flush=True)
    force_kill_current_process()
    return result  # pragma: no cover
