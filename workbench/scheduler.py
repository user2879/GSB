"""Deterministic parallel scheduler with global budget and per-sequence quotas.

Determinism contract: the frame schedule order and the budget ledger deduction
order are fixed before any thread runs (round-robin over sequence names in
lexicographic order, frame index ascending). Worker threads only compute
evidence; the ledger, cursors and all outputs are updated by the main thread
in schedule order, so results are identical at any concurrency level.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from .errors import E_BUDGET_EXHAUSTED


class BudgetLedger:
    def __init__(self, global_budget, per_sequence_quota, sequence_names):
        self.global_remaining = int(global_budget)
        self.quotas = {name: int(per_sequence_quota) for name in sequence_names}
        self.deduction_order = []  # [{"sequence":..., "frame":...}, ...]

    def can_afford(self, seq_name):
        return self.global_remaining > 0 and self.quotas[seq_name] > 0

    def deduct(self, seq_name, frame_index):
        self.global_remaining -= 1
        self.quotas[seq_name] -= 1
        self.deduction_order.append({"sequence": seq_name, "frame": frame_index})


class ThreadPeakTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def __enter__(self):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self._active -= 1
        return False


def build_schedule(sequence_names, cursors, frame_counts):
    """Round-robin (sequence-name order, frame ascending) deterministic schedule."""
    schedule = []
    max_frames = max(frame_counts.values(), default=0)
    for frame_index in range(max_frames):
        for name in sorted(sequence_names):
            if cursors[name] <= frame_index < frame_counts[name]:
                schedule.append((name, frame_index))
    return schedule


def run_frame_stage(schedule, ledger, work_fn, max_workers, should_stop=None,
                    on_frame_done=None):
    """Execute per-frame work concurrently; deduct the ledger in schedule order.

    work_fn(seq_name, frame_index) -> result or raises.
    Returns (results, failures, truncated) where truncated maps
    seq_name -> first unprocessed frame index when the budget ran out.
    """
    results = {}
    failures = []
    truncated = {}
    tracker = ThreadPeakTracker()
    exhausted = False

    def guarded(item):
        seq_name, frame_index = item
        with tracker:
            return item, work_fn(seq_name, frame_index)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = []
        for item in schedule:
            seq_name, frame_index = item
            if exhausted or not ledger.can_afford(seq_name):
                truncated.setdefault(seq_name, frame_index)
                continue
            ledger.deduct(seq_name, frame_index)
            futures.append((item, pool.submit(guarded, item)))
            if ledger.global_remaining <= 0:
                exhausted = True
        for (seq_name, frame_index), future in futures:
            if should_stop is not None and should_stop():
                truncated.setdefault(seq_name, frame_index)
                continue
            try:
                _, result = future.result()
            except Exception as exc:  # isolated per frame; recorded by caller
                failures.append({
                    "sequence": seq_name,
                    "frame": frame_index,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                })
                continue
            results[(seq_name, frame_index)] = result
            if on_frame_done is not None:
                on_frame_done(seq_name, frame_index, result)
    return results, failures, truncated, tracker.peak
