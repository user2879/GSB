"""Bounded thread-pool scheduler with stable ordering and point budget.

Every task carries a stable sort key derived from device id, window/spectrum
position and stage name.  Completed tasks are re-sorted by key before output,
so serial and parallel executions are field-identical.  In-flight tasks never
exceed ``scheduler.max_workers``.
"""

import threading

from concurrent.futures import ThreadPoolExecutor

from .errors import E_POINT_BUDGET_PENDING, failure_entry


class ShutdownRequested(Exception):
    pass


class Scheduler:
    def __init__(self, cfg, budget_remaining, shutdown_event=None):
        self.max_workers = max(1, int(cfg["scheduler"]["max_workers"]))
        self.serial = bool(cfg["scheduler"].get("serial", False))
        self.budget_total = int(cfg["scheduler"]["point_budget"])
        self.budget_remaining = int(budget_remaining)
        self._lock = threading.Lock()
        self.shutdown_event = shutdown_event

    def check_shutdown(self):
        if self.shutdown_event is not None and self.shutdown_event.is_set():
            raise ShutdownRequested("shutdown signal received")

    def consume(self, points):
        with self._lock:
            if points > self.budget_remaining:
                return False
            self.budget_remaining -= points
            return True

    def map_ordered(self, tasks):
        """Run (key, callable) tasks; return ordered (key, result) list.

        Tasks are submitted in key order, at most max_workers concurrently,
        and outputs are re-sorted before return.
        """
        ordered = sorted(tasks, key=lambda item: item[0])
        if self.serial or self.max_workers == 1:
            output = []
            for key, func in ordered:
                self.check_shutdown()
                output.append((key, func()))
            return output
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            in_flight = []
            for key, func in ordered:
                while len(in_flight) >= self.max_workers:
                    done_index = self._wait_one(in_flight)
                    done_key, done_future = in_flight.pop(done_index)
                    output.append((done_key, done_future.result()))
                in_flight.append((key, pool.submit(func)))
            for key, future in in_flight:
                output.append((key, future.result()))
        output.sort(key=lambda item: item[0])
        return output

    @staticmethod
    def _wait_one(in_flight):
        from concurrent.futures import wait, FIRST_COMPLETED
        futures = [item[1] for item in in_flight]
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for index, (_, future) in enumerate(in_flight):
            if future in done:
                return index
        return 0

    def budget_pending(self, device_id, position, points):
        return failure_entry(
            E_POINT_BUDGET_PENDING,
            f"spectrum position {position} of {device_id} needs {points} "
            f"points, {self.budget_remaining} remain",
            device=device_id, stage="scoring", position=position)
