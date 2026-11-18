import os
import shutil
import tempfile
import unittest

from layerbench.app import Workbench
from layerbench.errors import E_COMMAND, E_TIMEOUT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "samples", "workspace")
FAILURE = os.path.join(ROOT, "samples", "manifests", "failure_manifest.json")
TIMEOUT = os.path.join(ROOT, "samples", "manifests", "timeout_manifest.json")


class FailureAndTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lb-ft-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failure_propagates_skip_with_trigger(self):
        bench = Workbench(os.path.join(self.tmp, "fail"))
        result = bench.build(FAILURE, source_root=SRC)
        nodes = {n["node_uid"]: n for n in result["report"]["nodes"]}
        bad = nodes["base#2:f-run-bad"]
        self.assertEqual(bad["result_code"], E_COMMAND)
        self.assertEqual(bad["exit_code"], 3)
        skipped = nodes["base#3:f-after-bad"]
        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["skipped_trigger"], "base#2:f-run-bad")
        self.assertIsNone(skipped["layer_digest"])
        # 独立无依赖指令仍正常提交且带层摘要
        independent = nodes["base#4:f-label"]
        self.assertEqual(independent["status"], "committed")
        self.assertIsNotNone(independent["layer_digest"])
        self.assertTrue(result["report"]["audit"]["ok"])

    def test_timeout_records_mechanism_and_residual(self):
        bench = Workbench(os.path.join(self.tmp, "timeout"))
        result = bench.build(TIMEOUT, source_root=SRC)
        node = result["report"]["nodes"][0]
        self.assertEqual(node["result_code"], E_TIMEOUT)
        self.assertTrue(node["timeout_mechanism"])
        self.assertIn("回收", node["timeout_residual"])
        # 超时不是“失控命令”：说明里不允许出现失控措辞
        self.assertNotIn("失控", node["timeout_mechanism"])
        self.assertTrue(result["report"]["audit"]["ok"])


if __name__ == "__main__":
    unittest.main()
