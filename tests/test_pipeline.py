import tempfile
import unittest

import numpy as np

from animpipe.budget import Budget
from animpipe.errors import E_BUDGET_DENIED, E_TIME_NONMONOTONIC
from animpipe.pipeline import run_pipeline
from animpipe.samples import generate_all


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sample = generate_all(cls.tmp)
        cls.budget = Budget(
            max_workers=8, max_buffer_bytes=1 << 24, max_frame_cost=1 << 20
        )

    def _run(self, workers, block_size):
        return run_pipeline(
            self.sample.source_anim,
            self.sample.source_skel,
            self.sample.target_skel,
            self.sample.target_fps,
            budget=self.budget,
            workers=workers,
            block_size=block_size,
        )

    def test_block_partition_byte_invariant(self):
        hashes = set()
        tags = set()
        for workers, block in ((1, 1), (2, 3), (3, 5), (8, 24), (4, 7)):
            result = self._run(workers, block)
            hashes.add(result.output.content_hash())
            tags.add(result.frame_tags)
        self.assertEqual(len(hashes), 1)
        self.assertEqual(len(tags), 1)

    def test_budget_degradation_preserves_output(self):
        tight = Budget(max_workers=1, max_buffer_bytes=4096, max_frame_cost=1 << 20)
        degraded = run_pipeline(
            self.sample.source_anim,
            self.sample.source_skel,
            self.sample.target_skel,
            self.sample.target_fps,
            budget=tight,
            workers=8,
            block_size=10,
        )
        free = self._run(8, 10)
        self.assertEqual(
            degraded.output.content_hash(), free.output.content_hash()
        )
        self.assertEqual(
            degraded.output.frame_count, free.output.frame_count
        )
        decisions = degraded.budget_decisions
        self.assertTrue(any(d["degradation"] for d in decisions))
        for decision in decisions:
            self.assertEqual(decision["workers"], 1)
            self.assertLessEqual(decision["block_size"], 10)

    def test_budget_denial(self):
        planner_budget = Budget(
            max_workers=2, max_buffer_bytes=256, max_frame_cost=1 << 20
        )
        from animpipe.budget import BudgetPlanner

        planner = BudgetPlanner(planner_budget)
        with self.assertRaises(E_BUDGET_DENIED):
            planner.submit_task(
                stage="x",
                frame_count=8,
                joint_count=self.sample.source_anim.joint_count,
            )

    def test_non_monotonic_time_rejected(self):
        from animpipe.animation import Animation

        src = self.sample.source_anim
        bad_times = np.concatenate([src.times[:4], src.times[3:4], src.times[4:]])
        bad_rot = np.concatenate(
            [src.rotations[:4], src.rotations[3:4], src.rotations[4:]], axis=0
        )
        bad_root = np.concatenate(
            [src.root_translations[:4], src.root_translations[3:4], src.root_translations[4:]],
            axis=0,
        )
        bad = Animation(
            joint_names=src.joint_names,
            times=bad_times,
            rotations=bad_rot,
            root_translations=bad_root,
            interp=src.interp,
        )
        with self.assertRaises(E_TIME_NONMONOTONIC):
            run_pipeline(
                bad,
                self.sample.source_skel,
                self.sample.target_skel,
                self.sample.target_fps,
                budget=self.budget,
                workers=2,
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
