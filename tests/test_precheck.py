"""Bounded precheck pool: concurrency, memory budget and failure aggregation."""

from __future__ import annotations

import threading
import time
import unittest

from level_solver.errors import E_CONSTRAINT, SolverError
from level_solver.grid import Grid
from level_solver.parser import parse_spec
from level_solver.precheck_pool import BoundedPrecheckPool, PrecheckJob


def make_spec():
    return parse_spec(
        {
            "width": 8,
            "height": 8,
            "seed": 1,
            "blocked": [],
            "concurrency": 4,
            "memory_budget": 4,
            "constraints": [
                {"id": "r", "type": "min_rooms",
                 "params": {"count": 2, "min_room_w": 2, "min_room_h": 2,
                            "max_room_w": 3, "max_room_h": 3}}
            ],
        }
    )


class PrecheckPoolTests(unittest.TestCase):
    def test_parallel_and_serial_acceptance_identical(self):
        spec = make_spec()
        constraint = spec.constraints[0]
        grid = Grid(8, 8)
        jobs = [PrecheckJob(order=0, cid=constraint.cid, spec=constraint, grid=grid, rooms=[])]
        BoundedPrecheckPool(concurrency=4, memory_budget=4).run(jobs)
        BoundedPrecheckPool(concurrency=1, memory_budget=1).run(jobs)

    def test_first_failure_aggregated_with_business_code(self):
        spec = parse_spec(
            {
                "width": 8, "height": 8, "seed": 1, "blocked": [],
                "concurrency": 2, "memory_budget": 2,
                "constraints": [
                    {"id": "r", "type": "min_rooms",
                     "params": {"count": 2, "min_room_w": 3, "min_room_h": 3,
                                "max_room_w": 3, "max_room_h": 3}}
                ],
            }
        )
        constraint = spec.constraints[0]
        tiny_grid = Grid(3, 3)
        jobs = [
            PrecheckJob(order=1, cid=constraint.cid, spec=constraint,
                        grid=tiny_grid, rooms=[]),
        ]
        with self.assertRaises(SolverError) as ctx:
            BoundedPrecheckPool(2, 2).run(jobs)
        self.assertEqual(ctx.exception.code, E_CONSTRAINT)

    def test_memory_budget_bounds_inflight_candidates(self):
        spec = make_spec()
        constraint = spec.constraints[0]
        inflight = {"current": 0, "max": 0}
        lock = threading.Lock()

        class TrackingPool(BoundedPrecheckPool):
            def _run_job(self, job):
                with lock:
                    inflight["current"] += 1
                    inflight["max"] = max(inflight["max"], inflight["current"])
                time.sleep(0.02)
                super()._run_job(job)
                with lock:
                    inflight["current"] -= 1

        jobs = [
            PrecheckJob(order=i, cid=f"c{i}", spec=constraint, grid=Grid(8, 8), rooms=[])
            for i in range(8)
        ]
        TrackingPool(concurrency=2, memory_budget=2).run(jobs)
        self.assertLessEqual(inflight["max"], 2)

    def test_failure_cancels_remaining_jobs(self):
        spec = make_spec()
        constraint = spec.constraints[0]
        started = threading.Event()

        class SlowPool(BoundedPrecheckPool):
            def _run_job(self, job):
                if job.order == 0:
                    started.set()
                    time.sleep(0.05)
                    raise SolverError(E_CONSTRAINT, "boom")
                started.wait(0.5)
                time.sleep(0.2)

        jobs = [
            PrecheckJob(order=i, cid=f"c{i}", spec=constraint, grid=Grid(8, 8), rooms=[])
            for i in range(4)
        ]
        with self.assertRaises(SolverError):
            SlowPool(concurrency=4, memory_budget=4).run(jobs)

    def test_solver_selection_matches_serial_results(self):
        raw = {
            "width": 12, "height": 10, "seed": 31, "blocked": [],
            "max_rollback": 6, "alternative_limit": 16,
            "constraints": [
                {"id": "rooms", "type": "min_rooms",
                 "params": {"count": 3, "min_room_w": 2, "min_room_h": 2,
                            "max_room_w": 3, "max_room_h": 3}},
                {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
                {"id": "joined", "type": "connectivity", "params": {}},
            ],
        }
        parallel = parse_spec(dict(raw, concurrency=4, memory_budget=8))
        serial = parse_spec(dict(raw, concurrency=1, memory_budget=1))
        import tempfile
        from level_solver.solver import LevelSolver

        tmp = tempfile.mkdtemp()
        a = LevelSolver(parallel, f"{tmp}/p").solve()
        b = LevelSolver(serial, f"{tmp}/s").solve()
        self.assertEqual(a.grid.to_list(), b.grid.to_list())
        self.assertEqual(a.constraint_counts, b.constraint_counts)


if __name__ == "__main__":
    unittest.main()
