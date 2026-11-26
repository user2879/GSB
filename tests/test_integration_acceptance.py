import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shardmesh.chaos import free_port, run_kill_recovery
from shardmesh.generator import demo_spec_dict, generate_samples
from shardmesh.spec import parse_spec
from shardmesh.verification import verify_spec


class AcceptanceIntegrationTests(unittest.TestCase):
    def test_invalidation_rebuild_and_byte_preservation(self):
        ports = [free_port(), free_port(), free_port()]
        raw = demo_spec_dict(ports=ports)
        spec = parse_spec(raw)
        samples = generate_samples(raw)
        with TemporaryDirectory() as td:
            report = verify_spec(spec, samples, Path(td) / "verify")
        self.assertTrue(report["consistent"])
        self.assertEqual(report["code"], "OK")
        self.assertEqual(report["byte_check"]["violations"], [])
        self.assertIn("s01", report["byte_check"]["unchanged"])
        self.assertGreaterEqual(len(report["invalidation_plans"]), 2)

    def test_sigkill_ledger_replay(self):
        with TemporaryDirectory() as td:
            report = run_kill_recovery(Path(td) / "chaos")
        self.assertTrue(report["consistent"])
        self.assertEqual(report["sigkill_returncode"], -9)
        self.assertIn("s01", report["recovery"]["unknown_shards"])
        self.assertTrue(report["recovery"]["quota_reclaimed_first"])
        self.assertGreaterEqual(report["post_rerun_replay"]["checked_count"], 8)


if __name__ == "__main__":
    unittest.main()
