"""任意命令序列后增量运行 == 从空白全量重建（逐字段）。"""
import tempfile
import unittest

from reward_workbench.cli import _load_all
from reward_workbench.commands import load_commands
from reward_workbench.errors import FailureLog
from reward_workbench.pipeline import Pipeline
from reward_workbench.samplegen import write_workspace


class EquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rwb-eq-")
        self.paths = write_workspace(self.tmp)
        fl = FailureLog()
        self.cfg, self.pairs = _load_all(self.paths["pairs"], self.paths["config"], fl)
        self.commands = load_commands(self.paths["commands"], FailureLog())

    def test_replay_matches_full_rebuild(self):
        inc = Pipeline(self.cfg, dict(self.pairs), parallel=True)
        inc.run_all_batches()
        for cmd in self.commands:
            inc.apply_and_run(cmd)

        rebuild = Pipeline(inc.cfg, inc.pairs, parallel=True, full_compute=True)
        rebuild.run_all_batches()

        self.assertEqual(
            inc.report.data_artifacts(inc.cache),
            rebuild.report.data_artifacts(rebuild.cache),
        )

    def test_serial_parallel_identical(self):
        p_par = Pipeline(self.cfg, dict(self.pairs), parallel=True)
        p_par.run_all_batches()
        p_ser = Pipeline(self.cfg, dict(self.pairs), parallel=False)
        p_ser.run_all_batches()
        self.assertEqual(
            p_par.report.canonical_results(p_par.cache),
            p_ser.report.canonical_results(p_ser.cache),
        )

    def test_ordering_stable_under_worker_count(self):
        results = []
        for workers in (1, 2, 8):
            cfg = self.__class__ and _with_workers(self.cfg, workers)
            pipe = Pipeline(cfg, dict(self.pairs), parallel=True)
            pipe.run_all_batches()
            results.append(pipe.report.canonical_results(pipe.cache))
        for r in results[1:]:
            self.assertEqual(r, results[0])


def _with_workers(cfg, n):
    from dataclasses import replace
    return replace(cfg, max_workers=n)


if __name__ == "__main__":
    unittest.main()
