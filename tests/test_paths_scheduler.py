from __future__ import annotations

import unittest

from txdiag.canonical import canonical_text
from txdiag.engine import compare_canonical, execute
from txdiag.model import RuntimeConfig, choose_lock_path
from txdiag.spec import load_spec

from .helpers import sample


class PathAndSchedulerTests(unittest.TestCase):
    def _compare(self, name: str, inject: tuple[str, ...] = ()) -> None:
        spec = load_spec(sample(name))
        cfg = RuntimeConfig(max_running=4, wait_timeout_ms=1000,
                            wait_budget_ms=5000, inject=inject)
        row = execute(spec, cfg, "row")
        interval = execute(spec, cfg, "range")
        comparison = compare_canonical(row, interval)
        self.assertTrue(comparison["equal"], comparison)
        self.assertEqual(canonical_text(row.rounds), canonical_text(interval.rounds))

    def test_paths_match_on_all_samples(self) -> None:
        for name in ("all_isolation", "deadlock", "wait", "large_range"):
            with self.subTest(name=name):
                self._compare(name)

    def test_paths_match_with_real_timeout(self) -> None:
        self._compare("timeout", inject=("hold-lock=SLOW:1",))

    def test_auto_threshold_and_range_merge(self) -> None:
        spec = load_spec(sample("large_range"))
        cfg = RuntimeConfig(max_running=4)
        selected, selection = choose_lock_path(spec, cfg)
        self.assertEqual(selected, "range")
        self.assertIn("transactions>=8", selection["signals"])
        result = execute(spec, cfg, selected)
        self.assertEqual(result.path, "range")
        self.assertTrue(any(len(interval["rows"]) >= 2
                            for item in result.range_plan_rounds
                            for interval in item["intervals"]))

    def test_dual_quota_and_worker_pool(self) -> None:
        spec = load_spec(sample("large_range"))
        cfg = RuntimeConfig(
            max_running=3,
            quotas={"read_committed": 2, "serializable": 1},
            wait_timeout_ms=1000,
            wait_budget_ms=5000,
        )
        result = execute(spec, cfg, "range")
        for payload in result.rounds:
            self.assertLessEqual(payload["quota"]["global"]["used"], 3)
            self.assertLessEqual(payload["quota"]["isolation"]["read_committed"]["used"], 2)
            self.assertLessEqual(payload["quota"]["isolation"]["serializable"]["used"], 1)
            self.assertLessEqual(len(payload["running"]), 3)
        self.assertGreaterEqual(len(set(result.worker_threads)), 2)
        self.assertTrue(any(item["admission_queue"] for item in result.rounds))

    def test_first_divergence_location(self) -> None:
        spec = load_spec(sample("wait"))
        cfg = RuntimeConfig(max_running=4, wait_timeout_ms=1000, wait_budget_ms=5000,
                            inject=("diverge",))
        row = execute(spec, cfg, "row")
        interval = execute(spec, cfg, "range")
        comparison = compare_canonical(row, interval)
        self.assertFalse(comparison["equal"])
        self.assertEqual((comparison["first_txid"], comparison["first_row"],
                          comparison["first_field"]), ("HOLDER", "lock:A", "event"))


if __name__ == "__main__":
    unittest.main()
