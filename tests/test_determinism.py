import json
import os
import shutil
import tempfile
import unittest

from layerbench.app import Workbench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "samples", "workspace")
MANIFEST = os.path.join(ROOT, "samples", "manifests", "sample_manifest.json")


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lb-det-")
        self.workdir = os.path.join(self.tmp, "wd")
        self.bench = Workbench(self.workdir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_repeated_builds_byte_identical_canonical(self):
        first = self.bench.build(MANIFEST, source_root=SRC)
        second = self.bench.build(MANIFEST, source_root=SRC)
        self.assertEqual(first["canonical_digest"], second["canonical_digest"])
        with open(first["canonical_path"], "rb") as handle:
            a = handle.read()
        with open(second["canonical_path"], "rb") as handle:
            b = handle.read()
        self.assertEqual(a, b)

    def test_layer_digests_stable_and_cache_hits(self):
        first = self.bench.build(MANIFEST, source_root=SRC)
        first_layers = {n["node_uid"]: n["layer_digest"]
                        for n in first["report"]["nodes"]}
        second = self.bench.build(MANIFEST, source_root=SRC)
        second_layers = {n["node_uid"]: n["layer_digest"]
                         for n in second["report"]["nodes"]}
        self.assertEqual(first_layers, second_layers)
        # 第二次全部命中缓存，且没有任何 run 被真实执行
        self.assertTrue(all(n["cache_hit"] for n in second["report"]["nodes"]))
        run_nodes = [n for n in second["report"]["nodes"] if n["type"] == "run"]
        self.assertTrue(all(n["elapsed_ms"] == 0 for n in run_nodes))

    def test_budget_never_exceeds_limits(self):
        result = self.bench.build(MANIFEST, source_root=SRC)
        report = result["report"]
        self.assertLessEqual(
            report["budget"]["peak_global_running"],
            report["global_slots"],
        )
        for stage_name, usage in report["budget"]["stage_usage"].items():
            self.assertLessEqual(usage["peak_running"], usage["quota"])
        self.assertTrue(report["audit"]["ok"])


if __name__ == "__main__":
    unittest.main()
