"""异常注入隔离与业务码映射。"""
import tempfile
import unittest

from reward_workbench import injection as inj
from reward_workbench.samplegen import write_workspace
from reward_workbench.cli import _load_all
from reward_workbench.errors import FailureLog


class InjectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rwb-inj-")
        self.paths = write_workspace(self.tmp)

    def test_all_injections_hit_expected_codes(self):
        results = inj.run_config_injections(self.paths["config"])
        ids = {r["injection_id"] for r in results}
        self.assertEqual(ids, {"INJ-01", "INJ-02", "INJ-03"})
        for r in results:
            self.assertTrue(r["hit"], r)
            self.assertIn(r["expected_code"], r["observed_codes"])

        fl = FailureLog()
        cfg, pairs = _load_all(self.paths["pairs"], self.paths["config"], fl)
        cmd_result = inj.run_command_injection(pairs, cfg)
        self.assertTrue(cmd_result["hit"])
        self.assertEqual(cmd_result["expected_code"], "E_PAIR_NOT_FOUND")
        self.assertTrue(cmd_result["state_unchanged"])


if __name__ == "__main__":
    unittest.main()
